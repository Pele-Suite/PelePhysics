#include <iostream>
#include <vector>
#include <cmath>

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include "mechanism.H"
#include <GPU_misc.H>
#include <PelePhysics.H>

namespace {

using TransParmType = pele::physics::transport::TransParm<
  pele::physics::PhysicsType::eos_type,
  pele::physics::PhysicsType::transport_type>;

constexpr int NS = NUM_SPECIES;

// Record the worst violation of a check and its location
struct CheckMax
{
  amrex::Real value{0.0};
  amrex::IntVect cell{};
  void update(amrex::Real v, const amrex::IntVect& iv)
  {
    if (v > value) {
      value = v;
      cell = iv;
    }
  }
};

void
report(
  const std::string& name, const CheckMax& cm, amrex::Real tol, bool& failed)
{
  const bool ok = (cm.value <= tol);
  amrex::Print() << "  " << name << ": max = " << cm.value << " (tol = " << tol
                 << ") at cell " << cm.cell << (ok ? "  PASS" : "  FAIL")
                 << "\n";
  failed = failed || !ok;
}

#ifdef USE_SIMPLE_TRANSPORT
// Trace-adjusted mass/mole fractions exactly as SimpleTransport computes them
void
trace_fractions(
  const amrex::Real* Y,
  const TransParmType& tp,
  amrex::Real* Ytr,
  amrex::Real* Xtr,
  amrex::Real& wbar)
{
  amrex::Real sum = 0.0;
  for (int i = 0; i < NS; ++i) {
    sum += Y[i];
  }
  for (int i = 0; i < NS; ++i) {
    Ytr[i] = Y[i] + tp.trace_epsilon * (sum / NS - Y[i]);
  }
  wbar = 0.0;
  for (int i = 0; i < NS; ++i) {
    wbar += Ytr[i] * tp.iwt[i];
  }
  wbar = 1.0 / wbar;
  for (int i = 0; i < NS; ++i) {
    Xtr[i] = Ytr[i] * wbar * tp.iwt[i];
  }
}

// Invert a dense N x N matrix in place with Gauss-Jordan elimination and
// partial pivoting
void
invert_dense(std::vector<amrex::Real>& A)
{
  std::vector<amrex::Real> inv(NS * NS, 0.0);
  for (int i = 0; i < NS; ++i) {
    inv[i * NS + i] = 1.0;
  }
  for (int col = 0; col < NS; ++col) {
    int piv = col;
    for (int r = col + 1; r < NS; ++r) {
      if (std::abs(A[r * NS + col]) > std::abs(A[piv * NS + col])) {
        piv = r;
      }
    }
    if (piv != col) {
      for (int c = 0; c < NS; ++c) {
        std::swap(A[col * NS + c], A[piv * NS + c]);
        std::swap(inv[col * NS + c], inv[piv * NS + c]);
      }
    }
    const amrex::Real d = 1.0 / A[col * NS + col];
    for (int c = 0; c < NS; ++c) {
      A[col * NS + c] *= d;
      inv[col * NS + c] *= d;
    }
    for (int r = 0; r < NS; ++r) {
      if (r != col) {
        const amrex::Real f = A[r * NS + col];
        if (f != 0.0) {
          for (int c = 0; c < NS; ++c) {
            A[r * NS + c] -= f * A[col * NS + c];
            inv[r * NS + c] -= f * inv[col * NS + c];
          }
        }
      }
    }
  }
  A = inv;
}

// Direct (EGlib EGSDR2-style) evaluation of the flux diffusion matrix:
// rho * diag(Y) * [(L00 + Yhat Yhat^T)^-1 - u u^T], then projected.
void
reference_matrix(
  const amrex::Real T,
  const amrex::Real rho,
  const amrex::Real* Y,
  const TransParmType& tp,
  std::vector<amrex::Real>& Dref)
{
  amrex::Real Ytr[NS], Xtr[NS], wbar;
  trace_fractions(Y, tp, Ytr, Xtr, wbar);
  amrex::Real logT[3];
  logT[0] = std::log(T);
  logT[1] = logT[0] * logT[0];
  logT[2] = logT[0] * logT[1];

  std::vector<amrex::Real> binv(
    pele::physics::transport::NUM_BINARY_PAIRS, 0.0);
  const amrex::Real scale = pele::physics::transport::BinaryDiffInv<
    pele::physics::PhysicsType::eos_type>()(
    Ytr, logT, rho, T, &tp, [&](int i, int j, amrex::Real b) {
      binv[pele::physics::transport::binary_pair_index(i, j)] = b;
    });

  amrex::Real s = 0.0;
  for (int i = 0; i < NS; ++i) {
    s += Ytr[i];
  }
  std::vector<amrex::Real> A(NS * NS, 0.0);
  for (int i = 0; i < NS; ++i) {
    for (int j = 0; j < NS; ++j) {
      if (i != j) {
        const amrex::Real g =
          Xtr[i] * Xtr[j] *
          binv[pele::physics::transport::binary_pair_index(i, j)];
        A[i * NS + j] -= g;
        A[i * NS + i] += g;
      }
    }
  }
  for (int i = 0; i < NS; ++i) {
    for (int j = 0; j < NS; ++j) {
      A[i * NS + j] += (Ytr[i] / s) * (Ytr[j] / s);
    }
  }
  invert_dense(A);
  Dref.assign(NS * NS, 0.0);
  for (int i = 0; i < NS; ++i) {
    const amrex::Real fac = scale * Xtr[i] * tp.wt[i];
    for (int j = 0; j < NS; ++j) {
      Dref[i * NS + j] = fac * (A[i * NS + j] - 1.0);
    }
  }
  pele::physics::transport::project_diffusion_matrix(Ytr, Dref.data());
}
#endif

// Verify the multicomponent diffusion matrix on host copies of the data.
// Aborts if any check exceeds its tolerance.
void
check_matrix_diffusion(
  const amrex::MultiFab& Y_d,
  const amrex::MultiFab& T_d,
  const amrex::MultiFab& rho_d,
  const amrex::MultiFab& D_d,
  const amrex::MultiFab& Dmat_d,
  const TransParmType& tp)
{
  const amrex::BoxArray& ba = Dmat_d.boxArray();
  const amrex::DistributionMapping& dm = Dmat_d.DistributionMap();
  amrex::MFInfo pinned;
  pinned.SetArena(amrex::The_Pinned_Arena());
  amrex::MultiFab Y_h(ba, dm, NS, 0, pinned);
  amrex::MultiFab T_h(ba, dm, 1, 0, pinned);
  amrex::MultiFab rho_h(ba, dm, 1, 0, pinned);
  amrex::MultiFab D_h(ba, dm, NS, 0, pinned);
  amrex::MultiFab Dmat_h(ba, dm, NS * NS, 0, pinned);
  amrex::dtoh_memcpy(Y_h, Y_d);
  amrex::dtoh_memcpy(T_h, T_d);
  amrex::dtoh_memcpy(rho_h, rho_d);
  amrex::dtoh_memcpy(D_h, D_d);
  amrex::dtoh_memcpy(Dmat_h, Dmat_d);
  amrex::Gpu::streamSynchronize();

  amrex::ParmParse pp("tran_eval");
  amrex::Real tol_colsum = 1.e-12;
  amrex::Real tol_sym = 1.e-10;
  amrex::Real tol_binary = 1.e-11;
  amrex::Real tol_hc = 1.e-11;
  amrex::Real tol_ref = 0.5;
  int ref_samples = 10;
  pp.query("matrix_colsum_tol", tol_colsum);
  pp.query("matrix_symmetry_tol", tol_sym);
  pp.query("matrix_binary_tol", tol_binary);
  pp.query("matrix_hc_tol", tol_hc);
  pp.query("matrix_ref_tol", tol_ref);
  pp.query("matrix_ref_samples", ref_samples);

  // Global scale of the matrix entries for relative measures
  amrex::Real dmax = 0.0;
  for (amrex::MFIter mfi(Dmat_h); mfi.isValid(); ++mfi) {
    auto const& Ma = Dmat_h.const_array(mfi);
    amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
      for (int n = 0; n < NS * NS; ++n) {
        dmax = std::max(dmax, std::abs(Ma(i, j, k, n)));
      }
    });
  }
  amrex::ParallelDescriptor::ReduceRealMax(dmax);
  amrex::Print() << "\nMulticomponent diffusion matrix checks (max|Dmat| = "
                 << dmax << ")\n";

  CheckMax colsum, sym, binary, hc, ref;
  int nsampled = 0;
  const amrex::Long ncells_total = ba.numPts();
  const amrex::Long sample_stride = std::max<amrex::Long>(
    1, ncells_total / static_cast<amrex::Long>(std::max(1, ref_samples)));
  amrex::Long cell_counter = 0;

  for (amrex::MFIter mfi(Dmat_h); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.validbox();
    auto const& Ya = Y_h.const_array(mfi);
    auto const& Ta = T_h.const_array(mfi);
    auto const& ra = rho_h.const_array(mfi);
    auto const& Da = D_h.const_array(mfi);
    auto const& Ma = Dmat_h.const_array(mfi);
    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
      const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
      amrex::Real Y[NS], M[NS * NS];
      for (int n = 0; n < NS; ++n) {
        Y[n] = Ya(i, j, k, n);
      }
      for (int n = 0; n < NS * NS; ++n) {
        M[n] = Ma(i, j, k, n);
      }

      // (a) mass conservation: every column sums to zero
      for (int c = 0; c < NS; ++c) {
        amrex::Real acc = 0.0;
        for (int r = 0; r < NS; ++r) {
          acc += M[r * NS + c];
        }
        colsum.update(std::abs(acc) / dmax, iv);
      }

#ifdef USE_SIMPLE_TRANSPORT
      amrex::Real Ytr[NS], Xtr[NS], wbar;
      trace_fractions(Y, tp, Ytr, Xtr, wbar);

      // (b) Dmat_ij / Ytr_i is symmetric
      amrex::Real smax = 0.0;
      for (int r = 0; r < NS; ++r) {
        for (int c = 0; c < NS; ++c) {
          smax = std::max(smax, std::abs(M[r * NS + c] / Ytr[r]));
        }
      }
      for (int r = 0; r < NS; ++r) {
        for (int c = r + 1; c < NS; ++c) {
          sym.update(
            std::abs(M[r * NS + c] / Ytr[r] - M[c * NS + r] / Ytr[c]) / smax,
            iv);
        }
      }

      // (d) binary mixture: exact result rho_atm * D_12 * W1 W2 / Wbar^2
      if (NS == 2) {
        amrex::Real logT[3];
        logT[0] = std::log(Ta(i, j, k));
        logT[1] = logT[0] * logT[0];
        logT[2] = logT[0] * logT[1];
        amrex::Real b12 = 0.0;
        const amrex::Real scale = pele::physics::transport::BinaryDiffInv<
          pele::physics::PhysicsType::eos_type>()(
          Ytr, logT, ra(i, j, k), Ta(i, j, k), &tp,
          [&](int, int, amrex::Real b) { b12 = b; });
        const amrex::Real exact =
          scale * wbar / b12 * tp.wt[0] * tp.wt[1] / (wbar * wbar);
        binary.update(std::abs((M[0] - M[1]) - exact) / std::abs(exact), iv);
      }

      // (e) one iteration reproduces the projected Hirschfelder-Curtiss
      // diagonal
      if (tp.matrix_diffusion_niter == 1) {
        amrex::Real P[NS * NS] = {0.0};
        for (int n = 0; n < NS; ++n) {
          P[n * NS + n] = Da(i, j, k, n);
        }
        pele::physics::transport::project_diffusion_matrix(Ytr, P);
        for (int n = 0; n < NS * NS; ++n) {
          hc.update(std::abs(P[n] - M[n]) / dmax, iv);
        }
      }

      // (c) sampled cells: compare with the direct dense solution
      if (cell_counter % sample_stride == 0) {
        std::vector<amrex::Real> Dref;
        reference_matrix(Ta(i, j, k), ra(i, j, k), Y, tp, Dref);
        amrex::Real num = 0.0, den = 0.0;
        for (int n = 0; n < NS * NS; ++n) {
          num += (M[n] - Dref[n]) * (M[n] - Dref[n]);
          den += Dref[n] * Dref[n];
        }
        ref.update(std::sqrt(num / den), iv);
        ++nsampled;
      }
#else
      amrex::ignore_unused(Ta, ra, Da, tp, sample_stride, nsampled);
#endif
      ++cell_counter;
    });
  }

  bool failed = false;
  report("(a) relative column-sum residual", colsum, tol_colsum, failed);
#ifdef USE_SIMPLE_TRANSPORT
  report("(b) relative asymmetry of Dmat_ij/Y_i", sym, tol_sym, failed);
  if (NS == 2) {
    report("(d) binary mixture exact identity", binary, tol_binary, failed);
  }
  if (tp.matrix_diffusion_niter == 1) {
    report("(e) niter=1 vs projected Ddiag", hc, tol_hc, failed);
  }
  amrex::ParallelDescriptor::ReduceIntSum(nsampled);
  amrex::Print() << "  (c) " << nsampled
                 << " cells compared with the direct dense solution, niter = "
                 << tp.matrix_diffusion_niter << "\n";
  report("(c) relative Frobenius error vs direct", ref, tol_ref, failed);
#else
  amrex::ignore_unused(
    sym, binary, hc, ref, tol_sym, tol_binary, tol_hc, tol_ref, nsampled);
#endif
  if (failed) {
    amrex::Abort("Multicomponent diffusion matrix checks FAILED");
  }
  amrex::Print() << "Multicomponent diffusion matrix checks passed\n\n";
}

} // namespace

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  {

    amrex::ParmParse pp;

    pele::physics::PeleParams<TransParmType> trans_parms;
    pele::physics::PeleParams<
      pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>>
      eos_parms;
    eos_parms.initialize();
#ifdef USE_MANIFOLD_EOS
    trans_parms.host_only_parm().manfunc_par =
      eos_parms.host_only_parm().manfunc_par;
#endif
    trans_parms.initialize();

    amrex::ParmParse ppte("tran_eval");
    int matrix_diffusion = 0;
    int write_matrix = 1;
    int npts_in = 128;
    ppte.query("matrix_diffusion", matrix_diffusion);
    ppte.query("write_matrix", write_matrix);
    ppte.query("npts", npts_in);

    // Define geometry
    amrex::Array<int, AMREX_SPACEDIM> npts{AMREX_D_DECL(1, 1, 1)};
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
      npts[i] = npts_in;
    }

    amrex::Box domain(
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(npts[0] - 1, npts[1] - 1, npts[2] - 1)));

    amrex::RealBox real_box(
      {AMREX_D_DECL(-1.0, -1.0, -1.0)}, {AMREX_D_DECL(1.0, 1.0, 1.0)});

    int coord = 0;

    amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};

    amrex::Geometry geom(domain, real_box, coord, is_periodic);

    // Define BoxArray
    int max_size = 32;
    pp.query("max_size", max_size);
    amrex::BoxArray ba(domain);
    ba.maxSize(max_size);

    amrex::ParmParse ppa("amr");
    std::string pltfile("plt");
    ppa.query("plot_file", pltfile);

    amrex::DistributionMapping dm{ba};
    int num_grow = 0;

    // Data MFs
    amrex::MultiFab mass_frac(ba, dm, NUM_SPECIES, num_grow);
    amrex::MultiFab temperature(ba, dm, 1, num_grow);
    amrex::MultiFab density(ba, dm, 1, num_grow);

    const auto geomdata = geom.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mass_frac, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {

      const amrex::Box& bx = mfi.tilebox();

      auto const& Y_a = mass_frac.array(mfi);
      auto const& T_a = temperature.array(mfi);
      auto const& rho_a = density.array(mfi);

      amrex::ParallelFor(
        bx, [Y_a, T_a, rho_a,
             geomdata] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          initialize_data(i, j, k, Y_a, T_a, rho_a, geomdata);
        });
    }

    amrex::MultiFab D(ba, dm, NUM_SPECIES, num_grow);
    amrex::MultiFab mu(ba, dm, 1, num_grow);
    amrex::MultiFab xi(ba, dm, 1, num_grow);
    amrex::MultiFab lam(ba, dm, 1, num_grow);
    amrex::MultiFab chi(ba, dm, NUM_SPECIES, num_grow);
    // Full multicomponent flux diffusion matrix, component i*NS + j
    amrex::MultiFab Dmat;
    if (matrix_diffusion != 0) {
      Dmat.define(ba, dm, NUM_SPECIES * NUM_SPECIES, num_grow);
    }

    // Get the transport data pointer
    auto const* ltransparm = trans_parms.device_parm();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mass_frac, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {

      const amrex::Box& gbox = mfi.tilebox();

      amrex::Array4<amrex::Real> const& Y_a = mass_frac.array(mfi);
      amrex::Array4<amrex::Real> const& T_a = temperature.array(mfi);
      amrex::Array4<amrex::Real> const& rho_a = density.array(mfi);
      amrex::Array4<amrex::Real> const& D_a = D.array(mfi);
      amrex::Array4<amrex::Real> const& mu_a = mu.array(mfi);
      amrex::Array4<amrex::Real> const& xi_a = xi.array(mfi);
      amrex::Array4<amrex::Real> const& lam_a = lam.array(mfi);
      amrex::Array4<amrex::Real> const& chi_a = chi.array(mfi);

      if (matrix_diffusion != 0) {
        amrex::Array4<amrex::Real> const& Dmat_a = Dmat.array(mfi);
        amrex::launch(gbox, [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) {
          auto trans = pele::physics::PhysicsType::transport();
          trans.get_transport_coeffs(
            tbx, Y_a, T_a, rho_a, D_a, chi_a, Dmat_a, mu_a, xi_a, lam_a,
            ltransparm);
        });
      } else {
        amrex::launch(gbox, [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) {
          auto trans = pele::physics::PhysicsType::transport();
          trans.get_transport_coeffs(
            tbx, Y_a, T_a, rho_a, D_a, chi_a, mu_a, xi_a, lam_a, ltransparm);
        });
      }
    }
    amrex::Gpu::streamSynchronize();

    if (matrix_diffusion != 0) {
      check_matrix_diffusion(
        mass_frac, temperature, density, D, Dmat, trans_parms.host_parm());
    }

    trans_parms.deallocate();
    eos_parms.deallocate();

    const int nmat = (matrix_diffusion != 0 && write_matrix != 0)
                       ? NUM_SPECIES * NUM_SPECIES
                       : 0;
    amrex::MultiFab VarPlt(ba, dm, NUM_SPECIES + 3 + nmat, num_grow);
    amrex::MultiFab::Copy(VarPlt, D, 0, 0, NUM_SPECIES, num_grow);
    amrex::MultiFab::Copy(VarPlt, mu, 0, NUM_SPECIES, 1, num_grow);
    amrex::MultiFab::Copy(VarPlt, xi, 0, NUM_SPECIES + 1, 1, num_grow);
    amrex::MultiFab::Copy(VarPlt, lam, 0, NUM_SPECIES + 2, 1, num_grow);
    if (nmat > 0) {
      amrex::MultiFab::Copy(VarPlt, Dmat, 0, NUM_SPECIES + 3, nmat, num_grow);
    }

    std::string outfile = amrex::Concatenate(pltfile, 1);
    amrex::Vector<std::string> plt_VarsName;
    for (int k = 0; k < NUM_SPECIES; ++k) {
      plt_VarsName.push_back("D" + std::to_string(k));
    }
    plt_VarsName.push_back("mu");
    plt_VarsName.push_back("xi");
    plt_VarsName.push_back("lam");
    if (nmat > 0) {
      for (int i = 0; i < NUM_SPECIES; ++i) {
        for (int j = 0; j < NUM_SPECIES; ++j) {
          plt_VarsName.push_back(
            "Dmat_" + std::to_string(i) + "_" + std::to_string(j));
        }
      }
    }
    amrex::WriteSingleLevelPlotfile(
      outfile, VarPlt, plt_VarsName, geom, 0.0, 0);
  }

  amrex::Finalize();

  return 0;
}
