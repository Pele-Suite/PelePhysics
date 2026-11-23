// SootModel::estSootDt: the time for the soot sources to deplete a gas
// species, -rho Y_k / S_k, minimized over the species above soot.X_cutoff
// and over the cells. Without soot only the dimerization acts, so the
// estimate is the PAH depletion time 1 / (2 k_dimer [PAH]).

#include <AMReX_Print.H>

#include <cmath>

#include <PelePhysics.H>
#include "SootTest.H"

namespace soottest {

void
testTimeStep(const SootModel& soot, Checker& check)
{
  const SootData& sd = *soot.getSootData();
  const auto Y = sootingMixture();
  // Twice the PAH, and PAH below X_cutoff (1e-12 mol/cm^3)
  auto Y2 = Y;
  Y2[A2_ID] *= 2.;
  Y2[N2_ID] -= Y[A2_ID];
  auto Ytiny = Y;
  Ytiny[N2_ID] += Y[A2_ID] - 1.e-12;
  Ytiny[A2_ID] = 1.e-12;
  const Moments fresh = freshMoments(sd);
  Moments bimodal = bimodalMoments(sd);
  sd.convertFromMol(bimodal.data());
  const amrex::Real T = 1800.;

  // Cells: fresh, fresh with twice the PAH, bimodal, cold, PAH-free
  const amrex::Box bx = rowBox(0, 4);
  amrex::FArrayBox qfab(bx, nQ, amrex::The_Managed_Arena());
  auto const& q = qfab.array();
  setCell(q, amrex::IntVect(AMREX_D_DECL(0, 0, 0)), T, Y.data(), fresh);
  setCell(q, amrex::IntVect(AMREX_D_DECL(1, 0, 0)), T, Y2.data(), fresh);
  setCell(q, amrex::IntVect(AMREX_D_DECL(2, 0, 0)), T, Y.data(), bimodal);
  setCell(q, amrex::IntVect(AMREX_D_DECL(3, 0, 0)), 280., Y.data(), bimodal);
  setCell(q, amrex::IntVect(AMREX_D_DECL(4, 0, 0)), T, Ytiny.data(), fresh);
  amrex::Real est[5];
  for (int i = 0; i < 5; ++i) {
    est[i] = soot.estSootDt(cellBox(i), qfab.const_array());
  }
  const amrex::Real est_all = soot.estSootDt(bx, qfab.const_array());

  // PAH depletion time
  const SootReaction* sr = soot.d_sootReact;
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real mw[NUM_SPECIES];
  eos.molecular_weight(mw);
  const amrex::Real xiPAH = q(0, 0, 0, qRho) * Y[A2_ID] / mw[A2_ID];
  amrex::Gpu::ManagedVector<amrex::Real> kd(1);
  amrex::Real* kp = kd.data();
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) noexcept {
    kp[0] = sr->dimerRate(T, 1.);
  });
  amrex::Gpu::streamSynchronize();
  const amrex::Real t_pah = 1. / (2. * kd[0] * xiPAH);

  amrex::Print() << "estSootDt: fresh " << est[0] << " (PAH depletion " << t_pah
                 << "), twice the PAH " << est[1] << ", bimodal " << est[2]
                 << ", cold " << est[3] << ", no PAH " << est[4] << ", box "
                 << est_all << "\n";
  check(
    relErr(est[0], t_pah) <= 1.e-6,
    "estSootDt without soot is the PAH depletion time");
  // Twice the PAH mass fraction (the density changes slightly)
  const amrex::Real xiPAH2 = q(1, 0, 0, qRho) * Y2[A2_ID] / mw[A2_ID];
  check(
    relErr(est[1], 1. / (2. * kd[0] * xiPAH2)) <= 1.e-6 &&
      est[1] < 0.51 * est[0],
    "estSootDt halves with twice the PAH");
  check(
    est[2] > 0. && est[2] <= est[0],
    "estSootDt: the surface reactions can only shorten it");
  check(est[3] >= 1.e20, "estSootDt: no limit below soot.temp_cutoff");
  check(
    est[4] >= 1.e20 && std::isfinite(est[4]),
    "estSootDt: no limit from species below soot.X_cutoff");
  check(
    est_all == amrex::min(est[0], est[1], est[2], est[3], est[4]),
    "estSootDt of a box is the minimum over its cells");

  // Zero and negative moments
  for (int mom = 0; mom < NMOM; ++mom) {
    q(0, 0, 0, qSoot + mom) = 0.;
    q(1, 0, 0, qSoot + mom) = -bimodal[mom];
  }
  const amrex::Real est_bad = soot.estSootDt(rowBox(0, 1), qfab.const_array());
  check(
    std::isfinite(est_bad) && est_bad > 0.,
    "estSootDt is finite for zero or negative moments");
}

} // namespace soottest
