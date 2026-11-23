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

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

#include "mechanism.H"
#include <PelePhysics.H>
#include "SootModel.H"

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
  bool finite = true;        // no NaN or inf, which max would drop
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
    mb.finite = mb.finite && std::isfinite(srho) && std::isfinite(ssoot) &&
                std::isfinite(sspec);
  });
  return mb;
}

// Total gas and soot mass of the box
std::pair<amrex::Real, amrex::Real>
totalMass(const amrex::Box& bx, amrex::Array4<const amrex::Real> const& q)
{
  const SootConst sc;
  amrex::Real gas = 0.;
  amrex::Real soot = 0.;
  amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
    gas += q(i, j, k, qRho);
    soot += q(i, j, k, qSoot + 1) * sc.SootDensity;
  });
  return {gas, soot};
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
    const auto [gas0, soot0] = totalMass(bx, qfab.const_array());
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
      all_finite = all_finite && mb.finite;
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
    const auto [gas1, soot1] = totalMass(bx, qfab.const_array());
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

    soot.cleanup();
  }
  amrex::Finalize();
  return nfail;
}
