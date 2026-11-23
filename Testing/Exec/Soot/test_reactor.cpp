// Closed homogeneous reactor: a box of hot sooting gas (1400 K to 2000 K,
// temperature held) advanced explicitly with the soot source for nsteps
// steps of dt seconds. Every call balances mass, the reactor conserves gas +
// soot mass while soot forms, and the moments stay realizable.

#include <AMReX_Print.H>

#include <cmath>

#include "SootTest.H"

namespace soottest {

void
testReactor(const SootModel& soot, Checker& check)
{
  // With 6 moments and dt = 1e-4, the nucleus weight N0 of the hottest
  // cells oscillates from step to step
  constexpr int nsteps = 250;
  constexpr amrex::Real dt = 2.e-5;
  const SootConst sc;
  const SootData* sd = soot.getSootData();

  const int n = 8;
  const amrex::Box bx(
    amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
    amrex::IntVect(AMREX_D_DECL(n - 1, n - 1, 0)));
  Fabs f(bx);
  amrex::FArrayBox& qfab = f.q;
  auto const& q = qfab.array();
  auto const& s = f.s.const_array();
  const auto Y = sootingMixture();
  const Moments fresh = freshMoments(*sd);
  amrex::LoopOnCpu(bx, [&](int i, int j, int /*k*/) {
    const amrex::Real T =
      1400. + 600. * static_cast<amrex::Real>(i + n * j) / (n * n - 1);
    setCell(q, amrex::IntVect(AMREX_D_DECL(i, j, 0)), T, Y.data(), fresh);
  });
  auto totalMass = [&]() {
    return qfab.sum<amrex::RunOn::Host>(qRho) +
           qfab.sum<amrex::RunOn::Host>(qSoot + 1) * sc.SootDensity;
  };

  const amrex::Real m0 = totalMass();
  const amrex::Real soot0 = qfab.sum<amrex::RunOn::Host>(qSoot + 1);
  bool balanced = true;
  bool finite = true;
  for (int step = 0; step < nsteps; ++step) {
    sourceOn(soot, qfab, f.mu, f.s, dt);
    finite = finite && !f.s.contains_nan<amrex::RunOn::Host>() &&
             !f.s.contains_inf<amrex::RunOn::Host>();
    const auto [err, ref] = massBalance(bx, s);
    balanced = balanced && err <= 1.e-10 * ref;
    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
      // Explicit update
      const amrex::Real rho = q(i, j, k, qRho);
      const amrex::Real rho_new = rho + dt * s(i, j, k, sRho);
      for (int sp = 0; sp < NUM_SPECIES; ++sp) {
        q(i, j, k, qSpec + sp) =
          (rho * q(i, j, k, qSpec + sp) + dt * s(i, j, k, sSpec + sp)) /
          rho_new;
      }
      q(i, j, k, qRho) = rho_new;
      for (int m = 0; m < NMOM; ++m) {
        q(i, j, k, qSoot + m) += dt * s(i, j, k, sSoot + m);
      }
    });
  }
  const amrex::Real m1 = totalMass();
  const amrex::Real soot1 = qfab.sum<amrex::RunOn::Host>(qSoot + 1);
  amrex::Print() << "Reactor: soot mass formed / total mass = "
                 << (soot1 - soot0) * sc.SootDensity / m0
                 << ", relative change of gas + soot mass = " << (m1 - m0) / m0
                 << "\n";
  check(finite, "reactor: the sources are finite");
  check(balanced, "reactor: every call balances gas and soot mass", "#732");
  check((soot1 - soot0) * sc.SootDensity / m0 > 1.e-6, "reactor: soot forms");
  check(
    std::abs(m1 - m0) <= 1.e-10 * m0, "reactor: gas + soot mass is conserved",
    "#732");

  // Realizable moments (mol of C)
  bool realizable = true;
  amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
    amrex::Real m[NMOM];
    for (int mom = 0; mom < NMOM; ++mom) {
      m[mom] = q(i, j, k, qSoot + mom) / sd->unitConv[mom];
      realizable = realizable && std::isfinite(m[mom]) && m[mom] > 0.;
    }
    const amrex::Real slack = 1. - 1.e-12;
    realizable = realizable && m[NUM_SOOT_MOMENTS] <= m[0] &&
                 m[1] >= slack * sd->nuclVol * m[0] &&
                 m[2] >= slack * sd->nuclSurf * m[0];
#if NUM_SOOT_MOMENTS == 6
    realizable = realizable && m[3] >= slack * m[1] * m[1] / m[0] &&
                 m[5] >= slack * m[2] * m[2] / m[0];
#endif
  });
  check(realizable, "reactor: the moments stay realizable");
}

} // namespace soottest
