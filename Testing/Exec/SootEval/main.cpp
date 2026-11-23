// Checks of SootModel::computeSootSourceTerm (PeleC conventions: CGS units,
// primitive state with mass fractions):
// 1. Mass balance: with soot.conserve_mass, the gas density source plus the
//    soot mass source (the volume moment is the soot volume fraction) is zero
//    in every cell, and the species sources sum to the density source.
// 2. The routine adds into its destination: calling it on a destination that
//    already holds its source doubles the source exactly.
// 3. Closed homogeneous reactor: advancing a box of hot, sooting gas states
//    with the source for nsteps steps conserves gas plus soot mass while soot
//    forms.
// 4. Energy: without the pressure term the energy source is the sum of the
//    species sources times their enthalpies.
// 5. The moments stay realizable (positive, second mode no smaller than a
//    nucleus) through the reactor run.
// 6. Clipping the incoming moments is not a source, cells below
//    soot.temp_cutoff get no source, the time step estimate is finite for
//    zero or negative moments, the dimer concentration does not cancel where
//    condensation dominates, and the large-particle derive.

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

#include "mechanism.H"
#include <PelePhysics.H>
#include "SootModel.H"
#include "SootModel_derive.H"

namespace {

// Component layout of the primitive state and of the source
constexpr int qRho = 0;
constexpr int qTemp = 1;
constexpr int qSpec = 2;
constexpr int qSoot = qSpec + NUM_SPECIES;
constexpr int nQ = qSoot + NUM_SOOT_MOMENTS + 1;
constexpr int sRho = 0;
constexpr int sEng = 1;
constexpr int sSpec = 2;
constexpr int sSoot = sSpec + NUM_SPECIES;
constexpr int nS = sSoot + NUM_SOOT_MOMENTS + 1;

struct MassBalance
{
  amrex::Real err_soot = 0.; // max |rho source + soot mass source|
  amrex::Real err_spec = 0.; // max |sum of species sources - rho source|
  amrex::Real soot_src = 0.; // max |soot mass source|
  amrex::Real rho_src = 0.;  // max |rho source|
};

MassBalance
checkMassBalance(
  const amrex::Box& bx, amrex::Array4<const amrex::Real> const& s)
{
  const SootConst sc;
  MassBalance mb;
  amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
    const amrex::Real srho = s(i, j, k, sRho);
    const amrex::Real ssoot = s(i, j, k, sSoot + 1) * sc.SootDensity;
    amrex::Real sspec = 0.;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      sspec += s(i, j, k, sSpec + n);
    }
    mb.err_soot = amrex::max(mb.err_soot, std::abs(srho + ssoot));
    mb.err_spec = amrex::max(mb.err_spec, std::abs(sspec - srho));
    mb.soot_src = amrex::max(mb.soot_src, std::abs(ssoot));
    mb.rho_src = amrex::max(mb.rho_src, std::abs(srho));
  });
  return mb;
}

// Total gas and soot mass of the box
std::pair<amrex::Real, amrex::Real>
totalMass(const amrex::FArrayBox& qfab)
{
  const SootConst sc;
  return {
    qfab.sum<amrex::RunOn::Host>(qRho),
    qfab.sum<amrex::RunOn::Host>(qSoot + 1) * sc.SootDensity};
}

} // namespace

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  int nfail = 0;
  {
    amrex::ParmParse pp;
    int nsteps = 50;
    amrex::Real dt = 1.e-4;
    pp.query("nsteps", nsteps);
    pp.query("dt", dt);

    SootModel soot;
    soot.readSootParams();
    SootComps idx;
    idx.qRhoIndx = qRho;
    idx.qTempIndx = qTemp;
    idx.qSpecIndx = qSpec;
    idx.qSootIndx = qSoot;
    idx.rhoIndx = sRho;
    idx.engIndx = sEng;
    idx.specIndx = sSpec;
    idx.sootIndx = sSoot;
    soot.setIndices(idx);
    soot.define();

    // Post-flame rich mixture with soot precursors (mole fractions)
    amrex::Real X[NUM_SPECIES] = {0.};
    const std::vector<std::pair<int, amrex::Real>> mix = {
      {N2_ID, 0.60},  {H2O_ID, 0.12},  {CO_ID, 0.10}, {H2_ID, 0.08},
      {CO2_ID, 0.03}, {C2H2_ID, 0.05}, {H_ID, 5.e-3}, {OH_ID, 1.e-3},
      {O2_ID, 1.e-3}, {A1_ID, 1.e-4},  {A2_ID, 1.e-4}};
    amrex::Real xsum = 0.;
    for (const auto& [id, x] : mix) {
      X[id] = x;
      xsum += x;
    }
    for (amrex::Real& x : X) {
      x /= xsum;
    }
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real Y[NUM_SPECIES];
    eos.X2Y(X, Y);
    const amrex::Real P = 1013250.; // dyn/cm^2
    amrex::Real mom_init[NUM_SOOT_MOMENTS + 1] = {0.};
    soot.getSootData()->initialSmallMomVals(mom_init);

    // Box of states, temperature varying from 1400 K to 2000 K
    const int n = 8;
    const amrex::Box bx(
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(n - 1, n - 1, 0)));
    amrex::FArrayBox qfab(bx, nQ, amrex::The_Managed_Arena());
    amrex::FArrayBox mufab(bx, 1, amrex::The_Managed_Arena());
    amrex::FArrayBox sfab(bx, nS, amrex::The_Managed_Arena());
    amrex::FArrayBox s1fab(bx, nS, amrex::The_Managed_Arena());
    auto const& q = qfab.array();
    auto const& mu = mufab.array();
    auto const& s = sfab.array();
    auto const& s1 = s1fab.array();
    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
      const amrex::Real T =
        1400. + 600. * static_cast<amrex::Real>(i + n * j) / (n * n - 1);
      amrex::Real rho;
      eos.PYT2R(P, Y, T, rho);
      q(i, j, k, qRho) = rho;
      q(i, j, k, qTemp) = T;
      for (int sp = 0; sp < NUM_SPECIES; ++sp) {
        q(i, j, k, qSpec + sp) = Y[sp];
      }
      for (int m = 0; m < NUM_SOOT_MOMENTS + 1; ++m) {
        q(i, j, k, qSoot + m) = mom_init[m];
      }
      mu(i, j, k) = 5.e-4; // g/(cm s)
    });

    auto source = [&](amrex::FArrayBox& dst) {
      soot.computeSootSourceTerm(
        bx, qfab.const_array(), mufab.const_array(), dst.array(), 0., dt);
      amrex::Gpu::streamSynchronize();
    };
    auto check = [&](bool ok, const std::string& what) {
      amrex::Print() << (ok ? "  PASS: " : "  FAIL: ") << what << "\n";
      if (!ok) {
        ++nfail;
      }
    };
    const amrex::Real tol = 1.e-10;

    // 2. Adds into its destination
    sfab.setVal<amrex::RunOn::Host>(0.);
    source(sfab);
    s1fab.copy<amrex::RunOn::Host>(sfab);
    source(sfab);
    amrex::Real err_add = 0.;
    amrex::Real ref_add = 0.;
    amrex::LoopOnCpu(bx, nS, [&](int i, int j, int k, int c) {
      err_add =
        amrex::max(err_add, std::abs(s(i, j, k, c) - 2. * s1(i, j, k, c)));
      ref_add = amrex::max(ref_add, std::abs(s1(i, j, k, c)));
    });
    amrex::Print() << "Source added to a filled destination: max |S2 - 2 S1| = "
                   << err_add << " (max |S1| = " << ref_add << ")\n";
    check(err_add <= tol * ref_add, "the source adds into its destination");

    // 1 and 3. Mass balance of each call, and over the reactor run
    const auto [gas0, soot0] = totalMass(qfab);
    const amrex::Real m0 = gas0 + soot0;
    amrex::Real worst_soot = 0.;
    amrex::Real worst_spec = 0.;
    bool all_finite = true;
    // err / src, but the largest Real rather than NaN or inf when src is zero
    // and err is not, so the max keeps it and the check fails
    const auto ratio = [](const amrex::Real err, const amrex::Real src) {
      return (src > 0.)
               ? err / src
               : ((err > 0.) ? std::numeric_limits<amrex::Real>::max() : 0.);
    };
    for (int step = 0; step < nsteps; ++step) {
      sfab.setVal<amrex::RunOn::Host>(0.);
      source(sfab);
      const MassBalance mb = checkMassBalance(bx, sfab.const_array());
      worst_soot = amrex::max(worst_soot, ratio(mb.err_soot, mb.soot_src));
      worst_spec = amrex::max(worst_spec, ratio(mb.err_spec, mb.rho_src));
      // max drops NaN, so check every component directly
      all_finite = all_finite && !sfab.contains_nan<amrex::RunOn::Host>() &&
                   !sfab.contains_inf<amrex::RunOn::Host>();
      // Explicit update of the closed reactor (temperature held)
      amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
        const amrex::Real rho = q(i, j, k, qRho);
        const amrex::Real rho_new = rho + dt * s(i, j, k, sRho);
        for (int sp = 0; sp < NUM_SPECIES; ++sp) {
          q(i, j, k, qSpec + sp) =
            (rho * q(i, j, k, qSpec + sp) + dt * s(i, j, k, sSpec + sp)) /
            rho_new;
        }
        q(i, j, k, qRho) = rho_new;
        for (int m = 0; m < NUM_SOOT_MOMENTS + 1; ++m) {
          q(i, j, k, qSoot + m) += dt * s(i, j, k, sSoot + m);
        }
      });
    }
    const auto [gas1, soot1] = totalMass(qfab);
    const amrex::Real m1 = gas1 + soot1;
    amrex::Print() << "Per call: max |rho src + soot mass src| / max |soot "
                      "mass src| = "
                   << worst_soot << "\n"
                   << "Per call: max |sum species src - rho src| / max |rho "
                      "src| = "
                   << worst_spec << "\n"
                   << "Reactor: soot mass formed / total mass = "
                   << (soot1 - soot0) / m0
                   << ", relative change of gas + soot mass = "
                   << (m1 - m0) / m0 << "\n";
    check(all_finite, "the sources are finite");
    check(
      worst_soot <= tol, "gas density source balances the soot mass source");
    check(worst_spec <= tol, "species sources sum to the density source");
    check((soot1 - soot0) / m0 > 1.e-6, "soot forms in the reactor");
    check(
      std::abs(m1 - m0) <= tol * m0, "the reactor conserves gas + soot mass");

    // 4. Energy source from the species sources and their enthalpies
    sfab.setVal<amrex::RunOn::Host>(0.);
    soot.computeSootSourceTerm(
      bx, qfab.const_array(), mufab.const_array(), sfab.array(), 0., dt, false);
    amrex::Gpu::streamSynchronize();
    amrex::Real err_eng = 0.;
    amrex::Real ref_eng = 0.;
    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
      amrex::Real h[NUM_SPECIES];
      eos.T2Hi(q(i, j, k, qTemp), h);
      amrex::Real sum = 0.;
      amrex::Real sum_abs = 0.;
      for (int sp = 0; sp < NUM_SPECIES; ++sp) {
        sum += s(i, j, k, sSpec + sp) * h[sp];
        sum_abs += std::abs(s(i, j, k, sSpec + sp) * h[sp]);
      }
      err_eng = amrex::max(err_eng, std::abs(s(i, j, k, sEng) - sum));
      ref_eng = amrex::max(ref_eng, sum_abs);
    });
    amrex::Print() << "Energy source: max |S_E - sum h_k S_k| = " << err_eng
                   << " (max sum |h_k S_k| = " << ref_eng << ")\n";
    check(
      std::isfinite(err_eng) && err_eng <= tol * ref_eng,
      "the energy source is the species sources times their enthalpies");

    // 5. Realizable moments at the end of the reactor run (mol of C)
    const SootData* sd = soot.getSootData();
    bool realizable = true;
    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
      amrex::Real m[NUM_SOOT_MOMENTS + 1];
      for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
        m[mom] = q(i, j, k, qSoot + mom) / sd->unitConv[mom];
        realizable = realizable && std::isfinite(m[mom]) && m[mom] > 0.;
      }
      const amrex::Real slack = 1. - 1.e-12;
      realizable = realizable && m[1] >= slack * sd->nuclVol * m[0] &&
                   m[2] >= slack * sd->nuclSurf * m[0];
    });
    check(realizable, "the moments stay realizable in the reactor");

    // 6. On a two-cell box at 1800 K: cell 0 holds zero
    //    moments (below the clipping floor), cell 1 the same state with the
    //    moments already clipped
    const SootData* sd_d = soot.getSootData_d();
    const SootConst sc;
    const amrex::Box bx2(
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(1, 0, 0)));
    amrex::FArrayBox q2fab(bx2, nQ, amrex::The_Managed_Arena());
    amrex::FArrayBox mu2fab(bx2, 1, amrex::The_Managed_Arena());
    amrex::FArrayBox s2fab(bx2, nS, amrex::The_Managed_Arena());
    auto const& q2 = q2fab.array();
    auto const& s2 = s2fab.array();
    q2fab.setVal<amrex::RunOn::Host>(0.);
    mu2fab.setVal<amrex::RunOn::Host>(5.e-4);
    const amrex::Real T2 = 1800.;
    amrex::Real rho2;
    eos.PYT2R(P, Y, T2, rho2);
    amrex::LoopOnCpu(bx2, [&](int i, int j, int k) {
      q2(i, j, k, qRho) = rho2;
      q2(i, j, k, qTemp) = T2;
      for (int sp = 0; sp < NUM_SPECIES; ++sp) {
        q2(i, j, k, qSpec + sp) = Y[sp];
      }
    });
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) noexcept {
      amrex::Real m[NUM_SOOT_MOMENTS + 1] = {0.};
      sd_d->clipMoments(m);
      sd_d->convertFromMol(m);
      for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
        q2(1, 0, 0, qSoot + mom) = m[mom];
      }
    });
    amrex::Gpu::streamSynchronize();
    s2fab.setVal<amrex::RunOn::Host>(0.);
    soot.computeSootSourceTerm(
      bx2, q2fab.const_array(), mu2fab.const_array(), s2fab.array(), 0., dt);
    amrex::Gpu::streamSynchronize();
    amrex::Real err_clip = 0.;
    amrex::Real ref_clip = 0.;
    for (int c = 0; c < nS; ++c) {
      err_clip =
        amrex::max(err_clip, std::abs(s2(0, 0, 0, c) - s2(1, 0, 0, c)));
      ref_clip = amrex::max(ref_clip, std::abs(s2(1, 0, 0, c)));
    }
    amrex::Print() << "Zero vs clipped incoming moments: max |S0 - S1| = "
                   << err_clip << " (max |S1| = " << ref_clip << ")\n";
    check(
      err_clip <= tol * ref_clip,
      "clipping the incoming moments is not a soot source");

    // Below soot.temp_cutoff (290 K in the inputs) there is no source
    q2(0, 0, 0, qTemp) = 280.;
    s2fab.setVal<amrex::RunOn::Host>(0.);
    soot.computeSootSourceTerm(
      bx2, q2fab.const_array(), mu2fab.const_array(), s2fab.array(), 0., dt);
    amrex::Gpu::streamSynchronize();
    bool cold_zero = true;
    for (int c = 0; c < nS; ++c) {
      cold_zero = cold_zero && s2(0, 0, 0, c) == 0.;
    }
    check(cold_zero, "no source below soot.temp_cutoff");
    q2(0, 0, 0, qTemp) = T2;

    for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
      q2(1, 0, 0, qSoot + mom) = -q2(1, 0, 0, qSoot + mom);
    }
    const amrex::Real dt_est = soot.estSootDt(bx2, q2fab.const_array());
    amrex::Print() << "Time step estimate with zero and negative moments = "
                   << dt_est << "\n";
    check(
      std::isfinite(dt_est) && dt_est > 0.,
      "the time step estimate with zero or negative moments is finite");

    // Dimer concentration where condensation dominates (b^2 = 1e12 * 4ac):
    // the residual of betaNucl*x^2 + betaCond*x - dimerRate
    amrex::Gpu::ManagedVector<amrex::Real> dimer(3);
    amrex::Real* dimer_p = dimer.data();
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) noexcept {
      amrex::Real m[NUM_SOOT_MOMENTS + 1];
      amrex::Real fv[NUM_SOOT_MOMENTS + 2];
      sd_d->initialSmallMomVals(m);
      sd_d->convertToMol(m);
      sd_d->clipMoments(m);
      sd_d->computeFracMomVect(m, fv);
      const amrex::Real convT = std::sqrt(sc.colFact * T2);
      const amrex::Real b = sd_d->getBetaCond(convT, fv);
      const amrex::Real c = 0.25e-12 * b * b;
      dimer_p[0] = sd_d->dimerization(convT, 1., c, fv);
      dimer_p[1] = b;
      dimer_p[2] = c;
    });
    amrex::Gpu::streamSynchronize();
    const amrex::Real xd = dimer[0];
    const amrex::Real res = xd * xd + dimer[1] * xd - dimer[2];
    amrex::Print() << "Dimer root residual / dimerRate = " << res / dimer[2]
                   << "\n";
    check(
      xd > 0. && std::abs(res) <= tol * dimer[2],
      "the dimer concentration where condensation dominates");

    // Large-particle derive: N0 nuclei plus N_L particles of 50 nuclei
    const amrex::Real V_nucl = sd->nuclVol * sc.V0 / std::pow(sc.len_conv, 3);
    const amrex::Real S_nucl = sd->nuclSurf * sc.S0 / std::pow(sc.len_conv, 2);
    const amrex::Real N0 = 1.e8;
    const amrex::Real N_L = 1.e6;
    const amrex::Real V_L = 50. * V_nucl;
    const amrex::Real S_L = std::pow(50., 2. / 3.) * S_nucl;
    amrex::FArrayBox dfab(
      bx2, NUM_SOOT_MOMENTS + 1, amrex::The_Managed_Arena());
    amrex::FArrayBox lfab(bx2, 3, amrex::The_Managed_Arena());
    dfab.setVal<amrex::RunOn::Host>(0.);
    dfab.setVal<amrex::RunOn::Host>(N0 + N_L, bx2, 0, 1);
    dfab.setVal<amrex::RunOn::Host>(N0 * V_nucl + N_L * V_L, bx2, 1, 1);
    dfab.setVal<amrex::RunOn::Host>(N0 * S_nucl + N_L * S_L, bx2, 2, 1);
    dfab.setVal<amrex::RunOn::Host>(N0, bx2, NUM_SOOT_MOMENTS, 1);
    soot_largeparticledata(
      bx2, lfab, 0, 3, dfab, amrex::Geometry(), 0., nullptr, 0);
    amrex::Gpu::streamSynchronize();
    auto const& l = lfab.const_array();
    check(
      std::abs(l(0, 0, 0, 0) - N_L) <= tol * N_L &&
        std::abs(l(0, 0, 0, 1) - V_L) <= tol * V_L &&
        std::abs(l(0, 0, 0, 2) - S_L) <= tol * S_L,
      "the large-particle derive subtracts the nucleus size");

    soot.cleanup();
  }
  amrex::Finalize();
  return nfail;
}
