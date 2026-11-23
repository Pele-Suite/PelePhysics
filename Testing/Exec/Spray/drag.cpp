// Single droplet source terms (Drag.H, calculateSpraySource), through the
// calls of SprayParticleContainer::updateParticles: trilinear_interp,
// InterpolateGasPhase, SprayData::calcBoilT, calculateSpraySource. The gas
// sources are those of the parcel (the gas gets their negative), over the
// parcel's time step dtmod * dt.

#include "SprayTest.H"
#include "SprayInterpolation.H"
#include "Drag.H"
#include "WallFilm.H"

namespace spraytest {

Result
advance(
  Env& env,
  const Gas& gas,
  const SprayData& fd,
  const PType& p0,
  const Real dt,
  const bool film)
{
  const amrex::FArrayBox s = gasFab(gas);
  amrex::Gpu::ManagedVector<Result> res(1);
  Result* r = res.data();
  r->p = p0;
  r->dtp = fd.dtmod * dt;
  *env.scratch = fd;
  const SprayData* fdp = env.scratch;
  const TransParmType* tp = env.tp;
  const amrex::Box sbox = s.box();
  auto const& rho = s.const_array(cRho);
  auto const& mom = s.const_array(cMom);
  auto const& eng = s.const_array(cEng);
  auto const& T = s.const_array(cTemp);
  auto const& rhoY = s.const_array(cRhoY);
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) noexcept {
    PType& p = r->p;
    const RealVect lx = p.pos() / fabDx + 0.5;
    const amrex::IntVect bflags(amrex::IntVect::TheZeroVector());
    amrex::GpuArray<amrex::IntVect, AMREX_D_PICK(2, 4, 8)> indx;
    amrex::GpuArray<Real, AMREX_D_PICK(2, 4, 8)> w;
    trilinear_interp(lx.floor(), lx, indx.data(), w.data(), bflags);
    GasPhaseVals& gpv = r->gpv;
    pele::physics::PhysicsType::eos().molecular_weight(gpv.mw.data());
    gpv.reset();
    InterpolateGasPhase(
      gpv, sbox, rho, rhoY, T, mom, eng, indx.data(), w.data(), fdp->eosparm,
      fdp);
    Real cBoilT[SPRAY_FUEL_NUM];
    fdp->calcBoilT(gpv, cBoilT);
    r->cBoilT = cBoilT[0];
    if (film) {
      calculateFilmSource(dt, gpv, *fdp, p, cBoilT, tp);
    } else {
      r->Reyn = calculateSpraySource(dt, gpv, *fdp, p, cBoilT, tp);
    }
  });
  amrex::Gpu::streamSynchronize();
  return *r;
}

Real
massErr(const Result& r, const Real m0, const Real m1, const Real nppp)
{
  return relErr(r.gpv.fluid_mass_src * r.dtp, nppp * (m1 - m0));
}

bool
finite(const GasPhaseVals& gpv)
{
  bool ok =
    std::isfinite(gpv.fluid_mass_src) && std::isfinite(gpv.fluid_eng_src);
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    ok = ok && std::isfinite(gpv.fluid_mom_src[dir]);
  }
  for (int n = 0; n < NUM_SPECIES; ++n) {
    ok = ok && std::isfinite(gpv.fluid_Y_dot[n]);
  }
  return ok;
}

// Only the fuel gets a species source, equal to the mass source
bool
speciesOk(const GasPhaseVals& gpv, const Real tol)
{
  bool ok = relErr(gpv.fluid_Y_dot[fuel], gpv.fluid_mass_src) <= tol ||
            (gpv.fluid_Y_dot[fuel] == 0. && gpv.fluid_mass_src == 0.);
  for (int n = 0; n < NUM_SPECIES; ++n) {
    ok = ok && (n == fuel || gpv.fluid_Y_dot[n] == 0.);
  }
  return ok;
}

void
testDrag(Checker& check, Env& env)
{
  const SprayData& fdh = env.base;
  const Real tol = env.tol;
  // Variants of the spray data
  SprayData no_mass = fdh;
  no_mass.mass_trans = false;
  SprayData gravity = no_mass;
  gravity.body_force[1] = -981.;

  // The droplet in the middle of a cell
  const RealVect x0(0.55);
  const RealVect zero = RealVect::TheZeroVector();

  amrex::Print() << "\nDrag 1. Evaporating droplet in hot, moving gas\n";
  const Gas hot{.T = 1000., .vel = RealVect(AMREX_D_DECL(1000., 0., 0.))};
  const RealVect v0(AMREX_D_DECL(0., 500., 0.));
  const Real nppp = 10.;
  const PType d1 = makeDrop(x0, v0, 300., 5.e-3, nppp);
  Real dt = 1.e-5;
  Real energy_err[2];
  for (int h = 0; h < 2; ++h) {
    const Real dth = dt / (1 << h);
    const Result r = advance(env, hot, fdh, d1, dth);
    const Real m0 = dropMass(fdh, d1, r.cBoilT);
    const Real m1 = dropMass(fdh, r.p, r.cBoilT);
    const Real E0 = dropEnergy(fdh, d1, r.cBoilT);
    const Real E1 = dropEnergy(fdh, r.p, r.cBoilT);
    const Real deng = r.gpv.fluid_eng_src * r.dtp;
    energy_err[h] = std::abs(deng - nppp * (E1 - E0)) / std::abs(deng);
    if (h > 0) {
      continue;
    }
    const RealVect dmom = nppp * (m1 * dropVel(r.p) - m0 * v0);
    const Real mom_err =
      (r.gpv.fluid_mom_src * r.dtp - dmom).vectorLength() / dmom.vectorLength();
    amrex::Print() << "Re = " << r.Reyn << ", T: 300 -> "
                   << r.p.rdata(SC::pstateT) << " K, mass change "
                   << (m1 - m0) / m0 << ", mass source rel. err "
                   << massErr(r, m0, m1, nppp) << ", momentum rel. err "
                   << mom_err << "\n";
    check(finite(r.gpv), "the sources are finite");
    check(
      m1 < m0 && r.p.rdata(SC::pstateT) > 300. && r.Reyn > 0.,
      "the droplet evaporates and heats up");
    check(
      massErr(r, m0, m1, nppp) <= tol,
      "the mass source is the droplet's rate of change of mass");
    check(
      speciesOk(r.gpv, tol),
      "only the fuel gets a species source, equal to the mass source");
    check(
      mom_err <= tol,
      "the momentum source is the droplet's rate of change of momentum");
  }
  amrex::Print() << "Energy source vs droplet energy change, relative to the "
                    "source: "
                 << energy_err[0] << " (dt), " << energy_err[1] << " (dt/2)\n";
  check(
    energy_err[0] < 1.e-2 && energy_err[1] < 0.7 * energy_err[0],
    "the energy source is the droplet's rate of change of energy, to first "
    "order in dt");

  amrex::Print() << "\nDrag 2. No mass transfer\n";
  {
    const PType d = makeDrop(x0, zero, 300., 5.e-3, nppp);
    const Result r = advance(env, hot, no_mass, d, dt);
    check(
      r.gpv.fluid_mass_src == 0. && r.gpv.fluid_Y_dot[fuel] == 0.,
      "no mass or species source", "#733");
    check(
      r.p.rdata(SC::pstateDia) == d.rdata(SC::pstateDia) &&
        r.p.rdata(SC::pstateT) == d.rdata(SC::pstateT),
      "the droplet keeps its diameter and temperature");
    check(r.gpv.fluid_mom_src[0] > 0., "momentum is still exchanged");
  }

  amrex::Print() << "\nDrag 3. Drag\n";
  {
    const Gas air{.vel = RealVect(AMREX_D_DECL(1000., 0., 0.))};
    const PType d = makeDrop(x0, zero, 300., 2.e-3, 1.);
    dt = 5.e-3; // several relaxation times, so several substeps
    const Result r = advance(env, air, no_mass, d, dt);
    const RealVect v1 = dropVel(r.p);
    const Real m = dropMass(fdh, d, r.cBoilT);
    const RealVect dmom = m * v1;
    const Real mom_err =
      (r.gpv.fluid_mom_src * r.dtp - dmom).vectorLength() / dmom.vectorLength();
    amrex::Print() << "Droplet velocity after " << r.dtp << " s: " << v1
                   << ", momentum rel. err " << mom_err << "\n";
    check(
      v1[0] > 500. && v1[0] < 1000. && v1[1] == 0.,
      "the droplet accelerates toward the gas velocity without overshoot");
    check(
      mom_err <= tol,
      "the momentum source over substeps is the droplet's momentum change");

    const Gas still{};
    const Result r0 =
      advance(env, still, fdh, makeDrop(x0, zero, 300., 5.e-3, 1.), 1.e-5);
    check(
      r0.gpv.fluid_mom_src.vectorLength() == 0. &&
        dropVel(r0.p).vectorLength() == 0.,
      "no momentum source with gas and droplet at rest");

    // Falling already, so that the drag coefficient is not zero
    const RealVect vfall(AMREX_D_DECL(0., -1., 0.));
    const PType dg = makeDrop(x0, vfall, 300., 5.e-3, 1.);
    const Result rg = advance(env, still, gravity, dg, dt);
    const RealVect vg = dropVel(rg.p);
    const Real mg = dropMass(fdh, dg, rg.cBoilT);
    // The droplet's momentum change less that from gravity: the drag
    const Real drag = mg * (vg[1] - vfall[1] - gravity.body_force[1] * rg.dtp);
    amrex::Print() << "Velocity under gravity after " << rg.dtp << " s: " << vg
                   << "\n";
    check(
      vg[1] < 0. && vg[1] > vfall[1] + gravity.body_force[1] * rg.dtp &&
        relErr(rg.gpv.fluid_mom_src[1] * rg.dtp, drag) <= tol,
      "under gravity the droplet falls, slowed by drag, and the gas gets the "
      "drag only");
  }

  amrex::Print() << "\nDrag 4. Large time step\n";
  {
    // A 2 um droplet relaxes in about 2e-5 s; one step of 1e-3 s. The gas
    // is saturated with fuel, so the droplet only exchanges heat.
    const Gas warm{
      .T = 350., .X_fuel = 0.4, .vel = RealVect(AMREX_D_DECL(1000., 0., 0.))};
    const PType d = makeDrop(x0, zero, 300., 2.e-4, nppp);
    dt = 1.e-3;
    const Result r = advance(env, warm, fdh, d, dt);
    const RealVect v1 = dropVel(r.p);
    const Real T1 = r.p.rdata(SC::pstateT);
    amrex::Print() << "After " << r.dtp << " s: velocity " << v1 << ", T " << T1
                   << " K\n";
    check(
      finite(r.gpv) && finiteParticle(r.p) && r.p.id() > 0,
      "particle and sources are finite");
    check(
      v1[0] > 900. && v1[0] <= 1000. * (1. + 1.e-12) && v1[1] == 0.,
      "the velocity relaxes to the gas velocity without overshoot");
    check(
      T1 > 340. && T1 <= warm.T * (1. + 1.e-12),
      "the temperature relaxes to the gas temperature without overshoot");
  }

  amrex::Print() << "\nDrag 5. Complete evaporation within a step\n";
  {
    const Gas gas{.T = 350.};
    const Real m0 = 1.1 * SprayUnits::min_mass;
    const Real dia = std::cbrt(6. * m0 / (M_PI * liqRho(fdh, 300., 1.e3)));
    const PType d = makeDrop(x0, zero, 300., dia, nppp);
    dt = 1.e-3;
    const Result r = advance(env, gas, fdh, d, dt);
    const Real m0c = dropMass(fdh, d, r.cBoilT);
    const Real E0 = dropEnergy(fdh, d, r.cBoilT);
    const Real mL = nppp * m0c * fdh.liqprops.latent[0];
    const Real eng_err = std::abs(r.gpv.fluid_eng_src * r.dtp + nppp * E0) / mL;
    amrex::Print() << "Mass source rel. err " << massErr(r, m0c, 0., nppp)
                   << ", energy source + droplet energy, relative to m L: "
                   << eng_err << "\n";
    check(r.p.id() < 0, "the droplet is removed");
    check(finite(r.gpv), "the sources are finite");
    check(
      massErr(r, m0c, 0., nppp) <= tol && speciesOk(r.gpv, tol),
      "the gas gets the droplet's mass, all as fuel");
    check(eng_err <= tol, "the gas gets the droplet's energy", "#735");
  }

  amrex::Print() << "\nDrag 6. Saturated and cold gas\n";
  {
    const Gas sat{.T = 400., .X_fuel = 0.2};
    const PType d = makeDrop(x0, zero, 300., 5.e-3, nppp);
    dt = 1.e-4;
    const Result r = advance(env, sat, fdh, d, dt);
    amrex::Print() << "Saturated: T 300 -> " << r.p.rdata(SC::pstateT)
                   << " K\n";
    check(
      r.gpv.fluid_mass_src == 0. && r.gpv.fluid_Y_dot[fuel] == 0. &&
        relErr(dropMass(fdh, r.p, r.cBoilT), dropMass(fdh, d, r.cBoilT)) <= tol,
      "no mass transfer in fuel-saturated gas");
    check(
      r.p.rdata(SC::pstateT) > 300. && r.gpv.fluid_eng_src > 0.,
      "the droplet still heats up in saturated gas");

    const Gas cold{.T = 250.};
    const Result rc = advance(env, cold, fdh, d, dt);
    const Real m0 = dropMass(fdh, d, rc.cBoilT);
    const Real m1 = dropMass(fdh, rc.p, rc.cBoilT);
    amrex::Print() << "Cold: T 300 -> " << rc.p.rdata(SC::pstateT)
                   << " K, mass change " << (m1 - m0) / m0 << "\n";
    check(
      m1 < m0 && massErr(rc, m0, m1, nppp) <= tol &&
        rc.p.rdata(SC::pstateT) < 300.,
      "in colder gas the droplet evaporates, conserving mass, and cools");
  }
}

} // namespace spraytest
