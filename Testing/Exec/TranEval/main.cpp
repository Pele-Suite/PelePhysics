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
using TransportType = pele::physics::PhysicsType::transport_type;
using pele::physics::transport::ConductivityModel;
using pele::physics::transport::TransportOutputs;
using pele::physics::transport::TransportRequest;

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

// Copy a device MultiFab into a pinned host MultiFab
amrex::MultiFab
to_host(const amrex::MultiFab& mf)
{
  amrex::MFInfo pinned;
  pinned.SetArena(amrex::The_Pinned_Arena());
  amrex::MultiFab h(mf.boxArray(), mf.DistributionMap(), mf.nComp(), 0, pinned);
  amrex::dtoh_memcpy(h, mf);
  amrex::Gpu::streamSynchronize();
  return h;
}

// Invert a dense n x n row-major matrix in place with Gauss-Jordan
// elimination and partial pivoting
void
invert_dense(const int n, std::vector<amrex::Real>& A)
{
  std::vector<amrex::Real> inv(n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    inv[i * n + i] = 1.0;
  }
  for (int col = 0; col < n; ++col) {
    int piv = col;
    for (int r = col + 1; r < n; ++r) {
      if (std::abs(A[r * n + col]) > std::abs(A[piv * n + col])) {
        piv = r;
      }
    }
    if (piv != col) {
      for (int c = 0; c < n; ++c) {
        std::swap(A[col * n + c], A[piv * n + c]);
        std::swap(inv[col * n + c], inv[piv * n + c]);
      }
    }
    const amrex::Real d = 1.0 / A[col * n + col];
    for (int c = 0; c < n; ++c) {
      A[col * n + c] *= d;
      inv[col * n + c] *= d;
    }
    for (int r = 0; r < n; ++r) {
      if (r != col) {
        const amrex::Real f = A[r * n + col];
        if (f != 0.0) {
          for (int c = 0; c < n; ++c) {
            A[r * n + c] -= f * A[col * n + c];
            inv[r * n + c] -= f * inv[col * n + c];
          }
        }
      }
    }
  }
  A = inv;
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

// Packed reciprocal binary diffusion coefficients (dilute gas at 1 atm and
// effective) at T, and the Ddiag scale factor
amrex::Real
binary_coefficients(
  const amrex::Real T,
  const amrex::Real rho,
  const amrex::Real* Ytr,
  const TransParmType& tp,
  std::vector<amrex::Real>& batm,
  std::vector<amrex::Real>& beff)
{
  amrex::Real logT[3];
  logT[0] = std::log(T);
  logT[1] = logT[0] * logT[0];
  logT[2] = logT[0] * logT[1];
  batm.assign(pele::physics::transport::NUM_BINARY_PAIRS, 0.0);
  beff.assign(pele::physics::transport::NUM_BINARY_PAIRS, 0.0);
  return pele::physics::transport::BinaryDiffInv<
    pele::physics::PhysicsType::eos_type>()(
    Ytr, logT, rho, T, true, &tp,
    [&](int i, int j, amrex::Real ba, amrex::Real be) {
      const int p = pele::physics::transport::binary_pair_index(i, j);
      batm[p] = ba;
      beff[p] = be;
    });
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
  std::vector<amrex::Real> batm, binv;
  const amrex::Real scale = binary_coefficients(T, rho, Ytr, tp, batm, binv);

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
  invert_dense(NS, A);
  Dref.assign(NS * NS, 0.0);
  for (int i = 0; i < NS; ++i) {
    const amrex::Real fac = scale * Xtr[i] * tp.wt[i];
    for (int j = 0; j < NS; ++j) {
      Dref[i * NS + j] = fac * (A[i * NS + j] - 1.0);
    }
  }
  pele::physics::transport::project_diffusion_matrix(Ytr, Dref.data());
}

#ifndef PELE_TRANSPORT_NO_LCT
// Host evaluation of lambda (and chit) through the public extended API with
// a given parameter set
amrex::Real
host_lambda(
  const TransParmType& tp,
  const amrex::Real T,
  const amrex::Real rho,
  const amrex::Real* Y,
  amrex::Real* chit)
{
  TransportRequest req;
  req.lam = true;
  req.chit = (chit != nullptr);
  amrex::Real lam = 0.0;
  TransportOutputs out;
  out.lam = &lam;
  out.chit = chit;
  TransportType::transport(req, T, rho, Y, out, &tp);
  return lam;
}

// Per-species LCT inputs (cint, cxi, fac, internal self diffusion) and the
// packed dilute-gas binary coefficients at a state
void
lct_inputs(
  const amrex::Real T,
  const amrex::Real rho,
  const amrex::Real* Y,
  const TransParmType& tp,
  amrex::Real* Xtr,
  std::vector<amrex::Real>& binv,
  amrex::Real* eta,
  amrex::Real* cint,
  amrex::Real* cxi,
  amrex::Real* fac)
{
  amrex::Real Ytr[NS], wbar;
  trace_fractions(Y, tp, Ytr, Xtr, wbar);
  std::vector<amrex::Real> beff;
  binary_coefficients(T, rho, Ytr, tp, binv, beff);
  const amrex::Real lnT = std::log(T);
  amrex::Real cvk[NS];
  CKCVMS(T, cvk);
  for (int i = 0; i < NS; ++i) {
    eta[i] = std::exp(
      tp.fitmu[4 * i] + tp.fitmu[1 + 4 * i] * lnT +
      tp.fitmu[2 + 4 * i] * lnT * lnT + tp.fitmu[3 + 4 * i] * lnT * lnT * lnT);
  }
  pele::physics::transport::lct_species_setup(
    T, lnT, eta, cvk, &tp, binv.data(), cint, cxi, fac);
}

// Dense assembly of the LCT1 (Lambda_[e], n = N) or LCT3 (Lambda, n = 2N)
// system from the EGlib formulas (EGSEMAE / EGSEMA), independent of the
// matrix-free operator in SimpleConductivity.H.
int
lct_assemble(
  const int model,
  const amrex::Real T,
  const amrex::Real* X,
  const std::vector<amrex::Real>& binv,
  const amrex::Real* cint,
  const amrex::Real* cxi,
  const amrex::Real* fac,
  const TransParmType& tp,
  std::vector<amrex::Real>& G,
  std::vector<amrex::Real>& beta)
{
  const amrex::Real lnT = std::log(T);
  const bool full = (model == ConductivityModel::LCT3);
  const int n = full ? 2 * NS : NS;
  G.assign(n * n, 0.0);
  beta.assign(n, 0.0);
  auto g = [&](int r, int c) -> amrex::Real& { return G[r * n + c]; };
  for (int i = 0; i < NS; ++i) {
    const int ii = pele::physics::transport::binary_pair_index(i, i);
    const amrex::Real fx2 = fac[i] * X[i] * X[i];
    if (full) {
      g(i, i) = fx2 * (1.0 + 10.0 * cxi[i] / 3.0);
      g(i, NS + i) = -fx2 * 2.0 * cxi[i];
      g(NS + i, i) = g(i, NS + i);
      g(NS + i, NS + i) = fx2 * 1.2 * cxi[i] + X[i] * X[i] * cint[i] * binv[ii];
      beta[i] = 2.5 * X[i];
      beta[NS + i] = cint[i] * X[i];
    } else {
      g(i, i) =
        fx2 * (1.0 + 8.0 * cxi[i] / 15.0) + X[i] * X[i] * cint[i] * binv[ii];
      beta[i] = (2.5 + cint[i]) * X[i];
    }
  }
  for (int i = 0; i < NS; ++i) {
    for (int j = i + 1; j < NS; ++j) {
      const int p = pele::physics::transport::binary_pair_index(i, j);
      amrex::Real A, B, C;
      pele::physics::transport::collision_ratios(
        lnT - tp.log_epsij[p], A, B, C);
      const amrex::Real aaa = X[i] * X[j] * binv[p];
      const amrex::Real wi = tp.wt[i] / (tp.wt[i] + tp.wt[j]);
      const amrex::Real wj = tp.wt[j] / (tp.wt[i] + tp.wt[j]);
      if (full) {
        const amrex::Real bb1 = (20.0 / 3.0) * A * (cxi[i] + cxi[j]);
        const amrex::Real bbb = wi * wj * (13.75 - 3.0 * B - 4.0 * A - bb1);
        const amrex::Real bbj = 7.5 * wi * wi + (6.25 - 3.0 * B) * wj * wj +
                                4.0 * A * wi * wj + wi * wj * bb1;
        const amrex::Real bbi = 7.5 * wj * wj + (6.25 - 3.0 * B) * wi * wi +
                                4.0 * A * wi * wj + wi * wj * bb1;
        g(i, j) = -bbb * aaa;
        g(j, i) = g(i, j);
        g(i, i) += bbj * aaa;
        g(j, j) += bbi * aaa;
        const amrex::Real cci = wi * 4.0 * A * cxi[i];
        const amrex::Real ccj = wj * 4.0 * A * cxi[j];
        g(i, NS + j) = -ccj * aaa;
        g(NS + j, i) = g(i, NS + j);
        g(j, NS + i) = -cci * aaa;
        g(NS + i, j) = g(j, NS + i);
        g(i, NS + i) -= cci * aaa;
        g(NS + i, i) = g(i, NS + i);
        g(j, NS + j) -= ccj * aaa;
        g(NS + j, j) = g(j, NS + j);
        g(NS + i, NS + i) +=
          aaa * cint[i] + aaa * 2.4 * A * (tp.wt[i] / tp.wt[j]) * cxi[i];
        g(NS + j, NS + j) +=
          aaa * cint[j] + aaa * 2.4 * A * (tp.wt[j] / tp.wt[i]) * cxi[j];
      } else {
        const amrex::Real wioj = 3.0 * tp.wt[i] / tp.wt[j] - 2.0;
        const amrex::Real wjoi = 3.0 * tp.wt[j] / tp.wt[i] - 2.0;
        const amrex::Real tti = (4.0 / 3.0) * A * wioj * cxi[i];
        const amrex::Real ttj = (4.0 / 3.0) * A * wjoi * cxi[j];
        const amrex::Real bbb =
          wi * wj * (13.75 - 3.0 * B - 4.0 * A + tti + ttj);
        const amrex::Real bbj =
          7.5 * wi * wi + (6.25 - 3.0 * B) * wj * wj + 4.0 * A * wi * wj +
          (4.0 / 15.0) * A * cxi[i] * wi * wj * wioj * wioj +
          (20.0 / 3.0) * A * cxi[j] * wi * wj;
        const amrex::Real bbi =
          7.5 * wj * wj + (6.25 - 3.0 * B) * wi * wi + 4.0 * A * wi * wj +
          (4.0 / 15.0) * A * cxi[j] * wi * wj * wjoi * wjoi +
          (20.0 / 3.0) * A * cxi[i] * wi * wj;
        g(i, j) = -bbb * aaa;
        g(j, i) = g(i, j);
        g(i, i) += bbj * aaa + aaa * cint[i];
        g(j, j) += bbi * aaa + aaa * cint[j];
      }
    }
  }
  if (full) {
    for (int i = 0; i < NS; ++i) {
      if (tp.nlin[i] == 0) {
        g(NS + i, NS + i) = 1.0;
      }
    }
  }
  return n;
}

// Direct dense solution of the LCT system: lambda and chit
amrex::Real
lct_direct(
  const amrex::Real T,
  const amrex::Real rho,
  const amrex::Real* Y,
  const TransParmType& tp,
  amrex::Real* chit_ref)
{
  amrex::Real Xtr[NS], eta[NS], cint[NS], cxi[NS], fac[NS];
  std::vector<amrex::Real> binv;
  lct_inputs(T, rho, Y, tp, Xtr, binv, eta, cint, cxi, fac);
  std::vector<amrex::Real> G, beta;
  const int n = lct_assemble(
    tp.conductivity_model, T, Xtr, binv, cint, cxi, fac, tp, G, beta);
  invert_dense(n, G);
  std::vector<amrex::Real> a(n, 0.0);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      a[r] += G[r * n + c] * beta[c];
    }
  }
  amrex::Real lam = 0.0;
  for (int r = 0; r < n; ++r) {
    lam += a[r] * beta[r];
  }
  lam *= pele::physics::Constants::PATM / T;
  // The code applies the real-gas (Chung) correction to lambda; do the same
  // here (no-op for an ideal gas) so that both values are comparable
  {
    amrex::Real Ytr[NS], wbar, mu_dummy = 0.0;
    trace_fractions(Y, tp, Ytr, Xtr, wbar);
    pele::physics::transport::NonIdealChungCorrections<
      pele::physics::PhysicsType::eos_type>()(
      false, true, T, Xtr, rho, wbar, mu_dummy, lam, &tp);
  }
  if (chit_ref != nullptr) {
    pele::physics::transport::lct_thermal_diffusion_ratios(
      std::log(T), Xtr, binv.data(), a.data(), &tp, chit_ref);
  }
  return lam;
}

// Pure-species closed form of the LCT conductivity for species k at density
// rho (which only enters the real-gas correction)
amrex::Real
lct_pure_closed_form(
  const int k,
  const amrex::Real T,
  const amrex::Real rho,
  const TransParmType& tp)
{
  amrex::Real Y[NS] = {0.0};
  Y[k] = 1.0;
  amrex::Real Xtr[NS], eta[NS], cint[NS], cxi[NS], fac[NS];
  std::vector<amrex::Real> binv;
  lct_inputs(T, rho, Y, tp, Xtr, binv, eta, cint, cxi, fac);
  amrex::Real Ytr[NS], wbar;
  trace_fractions(Y, tp, Ytr, Xtr, wbar);
  auto chung = [&](amrex::Real lam) {
    amrex::Real mu_dummy = 0.0;
    pele::physics::transport::NonIdealChungCorrections<
      pele::physics::PhysicsType::eos_type>()(
      false, true, T, Xtr, rho, wbar, mu_dummy, lam, &tp);
    return lam;
  };
  const amrex::Real bkk =
    binv[pele::physics::transport::binary_pair_index(k, k)];
  const amrex::Real x2 = 1.0; // the pure limit X_k -> 1
  if (tp.conductivity_model == ConductivityModel::LCT3) {
    amrex::Real g11 = fac[k] * x2 * (1.0 + 10.0 * cxi[k] / 3.0);
    amrex::Real g12 = -fac[k] * x2 * 2.0 * cxi[k];
    amrex::Real g22 = fac[k] * x2 * 1.2 * cxi[k] + x2 * cint[k] * bkk;
    amrex::Real b1 = 2.5, b2 = cint[k];
    if (tp.nlin[k] == 0) {
      g22 = 1.0;
      b2 = 0.0;
    }
    const amrex::Real det = g11 * g22 - g12 * g12;
    const amrex::Real a1 = (g22 * b1 - g12 * b2) / det;
    const amrex::Real a2 = (-g12 * b1 + g11 * b2) / det;
    return chung(pele::physics::Constants::PATM / T * (a1 * b1 + a2 * b2));
  }
  const amrex::Real g = fac[k] * (1.0 + 8.0 * cxi[k] / 15.0) + cint[k] * bkk;
  const amrex::Real b = 2.5 + cint[k];
  return chung(pele::physics::Constants::PATM / T * b * b / g);
}

// Verify the LCT thermal conductivity and rescaled thermal diffusion ratios
void
check_conductivity(
  const amrex::MultiFab& Y_d,
  const amrex::MultiFab& T_d,
  const amrex::MultiFab& rho_d,
  const amrex::MultiFab& lam_d,
  const amrex::MultiFab& chi_d,
  const amrex::MultiFab& chit_d,
  const TransParmType& tp)
{
  const amrex::MultiFab Y_h = to_host(Y_d);
  const amrex::MultiFab T_h = to_host(T_d);
  const amrex::MultiFab rho_h = to_host(rho_d);
  const amrex::MultiFab lam_h = to_host(lam_d);
  const amrex::MultiFab chi_h = to_host(chi_d);
  const amrex::MultiFab chit_h = to_host(chit_d);

  amrex::ParmParse pp("tran_eval");
  amrex::Real tol_antisym = 1.e-13;
  amrex::Real tol_ref = 0.5;
  amrex::Real tol_mono = 1.e-12;
  amrex::Real ratio_lo = 0.5, ratio_hi = 2.0;
  amrex::Real tol_pure = 1.e-10;
  amrex::Real tol_pure_fit = 0.25; // polar species differ by up to ~20 %
  int ref_samples = 10;
  pp.query("lct_antisym_tol", tol_antisym);
  pp.query("lct_ref_tol", tol_ref);
  pp.query("lct_mono_tol", tol_mono);
  pp.query("lct_ratio_lo", ratio_lo);
  pp.query("lct_ratio_hi", ratio_hi);
  pp.query("lct_pure_tol", tol_pure);
  pp.query("lct_pure_fit_tol", tol_pure_fit);
  pp.query("lct_ref_samples", ref_samples);

  TransParmType tp_mix = tp;
  tp_mix.conductivity_model = ConductivityModel::Mixture;

  amrex::Print() << "\nMulticomponent conductivity checks (model = "
                 << (tp.conductivity_model == ConductivityModel::LCT3 ? "lct3"
                                                                      : "lct1")
                 << ", niter = " << tp.conductivity_niter << ")\n";

  CheckMax antisym, ref_lam, ref_chit, mono, ratio_dev;
  amrex::Real ratio_min = 1.e30, ratio_max = 0.0;
  int nsampled = 0;
  const amrex::Long ncells_total = Y_h.boxArray().numPts();
  const amrex::Long sample_stride = std::max<amrex::Long>(
    1, ncells_total / static_cast<amrex::Long>(std::max(1, ref_samples)));
  amrex::Long cell_counter = 0;
  std::vector<amrex::Real> soret_ratio_sum(NS, 0.0);
  std::vector<int> soret_ratio_cnt(NS, 0);

  for (amrex::MFIter mfi(Y_h); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.validbox();
    auto const& Ya = Y_h.const_array(mfi);
    auto const& Ta = T_h.const_array(mfi);
    auto const& ra = rho_h.const_array(mfi);
    auto const& La = lam_h.const_array(mfi);
    auto const& Ca = chi_h.const_array(mfi);
    auto const& Xa = chit_h.const_array(mfi);
    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
      const amrex::IntVect iv(AMREX_D_DECL(i, j, k));
      amrex::Real Y[NS], chit[NS], Ytr[NS], Xtr[NS], wbar;
      for (int n = 0; n < NS; ++n) {
        Y[n] = Ya(i, j, k, n);
        chit[n] = Xa(i, j, k, n);
      }
      trace_fractions(Y, tp, Ytr, Xtr, wbar);
      const amrex::Real T = Ta(i, j, k);
      const amrex::Real rho = ra(i, j, k);
      const amrex::Real lam = La(i, j, k);

      // (f) sum_i X_i chit_i = 0
      amrex::Real num = 0.0, den = 0.0;
      for (int n = 0; n < NS; ++n) {
        num += Xtr[n] * chit[n];
        den += Xtr[n] * std::abs(chit[n]);
      }
      antisym.update(std::abs(num) / den, iv);

      // (h) ratio to the mixture rule
      const amrex::Real lam_mix = host_lambda(tp_mix, T, rho, Y, nullptr);
      const amrex::Real ratio = lam / lam_mix;
      ratio_min = std::min(ratio_min, ratio);
      ratio_max = std::max(ratio_max, ratio);
      ratio_dev.update(
        std::max(ratio_lo - ratio, ratio - ratio_hi) /* <= 0 when inside */,
        iv);

      // (j) informational: X_i chit_i vs the mixture-averaged Soret fit
      if (tp.use_soret) {
        for (int n = 0; n < tp.numLite; ++n) {
          const int s = tp.liteSpec[n];
          const amrex::Real chi_mix = Ca(i, j, k, s);
          if (std::abs(chi_mix) > 1.e-8) {
            soret_ratio_sum[s] += Xtr[s] * chit[s] / chi_mix;
            soret_ratio_cnt[s] += 1;
          }
        }
      }

      // (g), (g') sampled cells: direct dense solution and monotonicity
      if (cell_counter % sample_stride == 0) {
        amrex::Real chit_ref[NS];
        const amrex::Real lam_ref = lct_direct(T, rho, Y, tp, chit_ref);
        ref_lam.update(std::abs(lam - lam_ref) / lam_ref, iv);
        amrex::Real e = 0.0, m = 0.0;
        for (int n = 0; n < NS; ++n) {
          e = std::max(e, std::abs(Xtr[n] * (chit[n] - chit_ref[n])));
          m = std::max(m, std::abs(Xtr[n] * chit_ref[n]));
        }
        ref_chit.update(e / m, iv);

        amrex::Real prev = 0.0;
        for (int it = 1; it <= 3; ++it) {
          TransParmType tp_it = tp;
          tp_it.conductivity_niter = it;
          const amrex::Real l = host_lambda(tp_it, T, rho, Y, nullptr);
          mono.update((prev - l) / lam_ref, iv);
          prev = l;
        }
        mono.update((prev - lam_ref) / lam_ref, iv);
        ++nsampled;
      }
      ++cell_counter;
    });
  }

  bool failed = false;
  report(
    "(f) |sum X_i chit_i| / sum X_i |chit_i|", antisym, tol_antisym, failed);
  amrex::Print() << "  (h) lambda_LCT / lambda_mixture in [" << ratio_min
                 << ", " << ratio_max << "]\n";
  report(
    "(h) excess of lambda_LCT/lambda_mixture outside [lo,hi]", ratio_dev, 0.0,
    failed);
  amrex::ParallelDescriptor::ReduceIntSum(nsampled);
  amrex::Print() << "  (g) " << nsampled
                 << " cells compared with the direct dense solution\n";
  report("(g) relative error of lambda vs direct", ref_lam, tol_ref, failed);
  report(
    "(g) relative error of X_i chit_i vs direct", ref_chit, tol_ref, failed);
  report(
    "(g') monotonicity defect lambda(1)<=lambda(2)<=lambda(3)<=direct", mono,
    tol_mono, failed);

  // (i) pure species: closed form and the pure-species conductivity fit
  CheckMax pure, pure_fit;
  amrex::Real fit_ratio_min = 1.e30, fit_ratio_max = 0.0;
  const amrex::Real temps[3] = {300.0, 1000.0, 2500.0};
  for (int k = 0; k < NS; ++k) {
    for (amrex::Real T : temps) {
      amrex::Real Y[NS] = {0.0};
      Y[k] = 1.0;
      // dilute gas at 1 atm (matters only for the SRK real-gas correction)
      const amrex::Real rho_atm = pele::physics::Constants::PATM * tp.wt[k] /
                                  (pele::physics::Constants::RU * T);
      const amrex::Real lam = host_lambda(tp, T, rho_atm, Y, nullptr);
      const amrex::Real exact = lct_pure_closed_form(k, T, rho_atm, tp);
      pure.update(std::abs(lam - exact) / exact, amrex::IntVect(k));
      const amrex::Real lnT = std::log(T);
      const amrex::Real lam_fit = std::exp(
        tp.fitlam[4 * k] + tp.fitlam[1 + 4 * k] * lnT +
        tp.fitlam[2 + 4 * k] * lnT * lnT +
        tp.fitlam[3 + 4 * k] * lnT * lnT * lnT);
      const amrex::Real r = lam / lam_fit;
      fit_ratio_min = std::min(fit_ratio_min, r);
      fit_ratio_max = std::max(fit_ratio_max, r);
      pure_fit.update(std::abs(r - 1.0), amrex::IntVect(k));
    }
  }
  amrex::Print() << "  (i) pure species lambda_LCT / lambda_fit(COFLAM) in ["
                 << fit_ratio_min << ", " << fit_ratio_max
                 << "] over T = 300, 1000, 2500 K (cell index = species)\n";
  report("(i) pure species vs closed form", pure, tol_pure, failed);
  report(
    "(i) pure species vs COFLAM fit |ratio - 1|", pure_fit, tol_pure_fit,
    failed);

  if (tp.use_soret) {
    for (int n = 0; n < tp.numLite; ++n) {
      const int s = tp.liteSpec[n];
      if (soret_ratio_cnt[s] > 0) {
        amrex::Print() << "  (j) species " << s
                       << ": mean X chit / chi_mix (ceptr Soret fit) = "
                       << soret_ratio_sum[s] / soret_ratio_cnt[s]
                       << " (EGlib and CHEMKIN sign conventions differ; "
                          "informational)\n";
      }
    }
  }

  if (failed) {
    amrex::Abort("Multicomponent conductivity checks FAILED");
  }
  amrex::Print() << "Multicomponent conductivity checks passed\n\n";
}
#endif // PELE_TRANSPORT_NO_LCT
#endif // USE_SIMPLE_TRANSPORT

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
  const amrex::MultiFab Y_h = to_host(Y_d);
  const amrex::MultiFab T_h = to_host(T_d);
  const amrex::MultiFab rho_h = to_host(rho_d);
  const amrex::MultiFab D_h = to_host(D_d);
  const amrex::MultiFab Dmat_h = to_host(Dmat_d);
  const amrex::BoxArray& ba = Dmat_d.boxArray();

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
        std::vector<amrex::Real> batm, beff;
        const amrex::Real scale =
          binary_coefficients(Ta(i, j, k), ra(i, j, k), Ytr, tp, batm, beff);
        const amrex::Real b12 =
          beff[pele::physics::transport::binary_pair_index(0, 1)];
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
    int conductivity_check = 0;
    int npts_in = 128;
    ppte.query("matrix_diffusion", matrix_diffusion);
    ppte.query("write_matrix", write_matrix);
    ppte.query("conductivity_check", conductivity_check);
    ppte.query("npts", npts_in);
#if defined(USE_SIMPLE_TRANSPORT) && !defined(PELE_TRANSPORT_NO_LCT)
    const bool lct_model =
      trans_parms.host_parm().conductivity_model != ConductivityModel::Mixture;
#else
    const bool lct_model = false;
#endif
    const bool do_chit = (conductivity_check != 0) && lct_model;
    const bool extended = (matrix_diffusion != 0) || do_chit;

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
    chi.setVal(0.0);
    // Full multicomponent flux diffusion matrix, component i*NS + j, and
    // rescaled thermal diffusion ratios
    amrex::MultiFab Dmat;
    amrex::MultiFab chit;
    if (matrix_diffusion != 0) {
      Dmat.define(ba, dm, NUM_SPECIES * NUM_SPECIES, num_grow);
    }
    if (do_chit) {
      chit.define(ba, dm, NUM_SPECIES, num_grow);
    }

    // Get the transport data pointer
    auto const* ltransparm = trans_parms.device_parm();

    TransportRequest req;
    req.xi = true;
    req.mu = true;
    req.lam = true;
    req.Ddiag = true;
#ifdef USE_MANIFOLD_TRANSPORT
    req.chi = false;
#else
    req.chi = trans_parms.host_parm().use_soret;
#endif
    req.Dmat = (matrix_diffusion != 0);
    req.chit = do_chit;

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

      if (extended) {
        amrex::Array4<amrex::Real> const& Dmat_a =
          (matrix_diffusion != 0) ? Dmat.array(mfi) : D.array(mfi);
        amrex::Array4<amrex::Real> const& chit_a =
          do_chit ? chit.array(mfi) : D.array(mfi);
        amrex::launch(gbox, [=] AMREX_GPU_DEVICE(amrex::Box const& tbx) {
          auto trans = pele::physics::PhysicsType::transport();
          trans.get_transport_coeffs(
            tbx, req, Y_a, T_a, rho_a, D_a, chi_a, Dmat_a, chit_a, mu_a, xi_a,
            lam_a, ltransparm);
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
#if defined(USE_SIMPLE_TRANSPORT) && !defined(PELE_TRANSPORT_NO_LCT)
    if (do_chit) {
      check_conductivity(
        mass_frac, temperature, density, lam, chi, chit,
        trans_parms.host_parm());
    }
#endif

    trans_parms.deallocate();
    eos_parms.deallocate();

    const int nmat = (matrix_diffusion != 0 && write_matrix != 0)
                       ? NUM_SPECIES * NUM_SPECIES
                       : 0;
    const int nchit = do_chit ? NUM_SPECIES : 0;
    amrex::MultiFab VarPlt(ba, dm, NUM_SPECIES + 3 + nmat + nchit, num_grow);
    amrex::MultiFab::Copy(VarPlt, D, 0, 0, NUM_SPECIES, num_grow);
    amrex::MultiFab::Copy(VarPlt, mu, 0, NUM_SPECIES, 1, num_grow);
    amrex::MultiFab::Copy(VarPlt, xi, 0, NUM_SPECIES + 1, 1, num_grow);
    amrex::MultiFab::Copy(VarPlt, lam, 0, NUM_SPECIES + 2, 1, num_grow);
    if (nmat > 0) {
      amrex::MultiFab::Copy(VarPlt, Dmat, 0, NUM_SPECIES + 3, nmat, num_grow);
    }
    if (nchit > 0) {
      amrex::MultiFab::Copy(
        VarPlt, chit, 0, NUM_SPECIES + 3 + nmat, nchit, num_grow);
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
    for (int k = 0; k < nchit; ++k) {
      plt_VarsName.push_back("chit_" + std::to_string(k));
    }
    amrex::WriteSingleLevelPlotfile(
      outfile, VarPlt, plt_VarsName, geom, 0.0, 0);
  }

  amrex::Finalize();

  return 0;
}
