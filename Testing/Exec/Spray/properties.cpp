// Liquid properties (SprayPropertiesPeleMP.H) and the species mapping of
// SprayParticleContainer::spraySetup.

#include "SprayTest.H"

namespace spraytest {

void
testProperties(Checker& check, Env& env)
{
  amrex::Print() << "\nProperties. Liquid properties and spray setup\n";
  const SprayData& fdh = env.base;
  const Real tol = env.tol;
  Real mw[NUM_SPECIES];
  pele::physics::PhysicsType::eos().molecular_weight(mw);
  check(
    fdh.dep_indx[0] == fuel && fdh.N_pc == 1 && fdh.pc_indx[0] == fuel &&
      fdh.liqprops.mw_i(0) == mw[fuel],
    "spraySetup maps the liquid to the gas fuel species");

  const auto& lp = fdh.liqprops;
  const Real Tb = lp.boilT_i(0);
  bool ok = true;
  bool psat_up = true;
  bool latent_down = true;
  Real psat_prev = 0.;
  Real latent_prev = 1.e300;
  for (int i = 0; i <= 20; ++i) {
    const Real T = 250. + (0.99 * Tb - 250.) * i / 20.;
    Real h[NUM_SPECIES];
    pele::physics::PhysicsType::eos().T2Hi(T, h);
    const Real latent = lp.latent_i(T, h[fuel] * SprayUnits::eng_conv, 0);
    const Real psat = lp.psat_i(T, 0, latent);
    const Real vals[6] = {lp.rho_i(T, 0), lp.cp_i(T, 0), latent,
                          psat,           lp.mu_i(T, 0), lp.lambda_i(T, 0)};
    for (const Real val : vals) {
      ok = ok && std::isfinite(val) && val > 0.;
    }
    psat_up = psat_up && psat > psat_prev;
    latent_down = latent_down && latent < latent_prev;
    psat_prev = psat;
    latent_prev = latent;
  }
  check(
    ok, "density, cp, latent heat, psat, viscosity and conductivity positive "
        "below boiling");
  check(psat_up && latent_down, "psat rises, latent heat falls with T");

  GasPhaseVals gpv;
  const Real PATM = pele::physics::Constants::PATM * SprayUnits::pres_conv;
  Real cBoilT[3];
  const Real pres[3] = {1., 10., 1.e3};
  for (int i = 0; i < 3; ++i) {
    gpv.p_fluid = pres[i] * PATM;
    fdh.calcBoilT(gpv, cBoilT + i);
  }
  const Real psat_b = lp.psat_i(Tb, 0, 0.);
  amrex::Print() << "psat(boil_temp) / 1 atm = " << psat_b / PATM
                 << ", boiling T at 1, 10, 1000 atm: " << cBoilT[0] << ", "
                 << cBoilT[1] << ", " << cBoilT[2] << "\n";
  check(
    std::abs(psat_b / PATM - 1.) < 0.02 && relErr(cBoilT[0], Tb) <= tol &&
      cBoilT[1] > Tb && cBoilT[1] < lp.critT_i(0) && cBoilT[2] == lp.critT_i(0),
    "boiling temperature: boil_temp at 1 atm, rising with pressure up to "
    "crit_temp");
}

} // namespace spraytest
