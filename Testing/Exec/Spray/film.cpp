// Wall film source terms (WallFilm.H, calculateFilmSource) for one film
// parcel, as updateParticles calls it.

#include "SprayTest.H"

namespace spraytest {

void
testFilm(Checker& check, Env& env)
{
  const SprayData& fdh = env.base;
  const Real tol = env.tol;
  SprayData fd = fdh;
  fd.wall_T = 350.;
  auto run = [&](const Gas& g, const PType& p0, const Real dt) {
    return advance(env, g, fd, p0, dt, true);
  };

  PType f = makeDrop(RealVect(0.55), RealVect(0.), 300., 0.05, 1.);
  f.rdata(SC::pstateFilmHght) = 1.e-3;

  amrex::Print() << "\nFilm 1. Evaporating film in warm gas\n";
  {
    const Gas warm{.T = 800.};
    const Real dt = 1.e-5;
    const Result r = run(warm, f, dt);
    const Real m0 = filmMass(fdh, f, r.cBoilT);
    const Real m1 = filmMass(fdh, r.p, r.cBoilT);
    amrex::Print() << "T 300 -> " << r.p.rdata(SC::pstateT)
                   << " K, film mass change " << (m1 - m0) / m0
                   << ", mass source rel. err " << massErr(r, m0, m1) << "\n";
    check(finite(r.gpv) && r.p.id() > 0, "the film sources are finite");
    check(
      m1 < m0 && massErr(r, m0, m1) <= tol && speciesOk(r.gpv, tol),
      "the mass source is the film's rate of change of mass, at its new "
      "density",
      "#735");
    check(r.p.rdata(SC::pstateT) > 300., "the film heats up");
    check(
      r.p.rdata(SC::pstateDia) == f.rdata(SC::pstateDia),
      "the film keeps its diameter");
  }

  amrex::Print() << "\nFilm 2. Saturated gas\n";
  {
    const Gas sat{.T = 400., .X_fuel = 0.2};
    const Real dt = 1.e-5;
    const Result rs = run(sat, f, dt);
    amrex::Print() << "T 300 -> " << rs.p.rdata(SC::pstateT) << " K\n";
    check(
      rs.gpv.fluid_mass_src == 0. &&
        relErr(filmMass(fdh, rs.p, rs.cBoilT), filmMass(fdh, f, rs.cBoilT)) <=
          tol,
      "no film mass transfer in saturated gas; the film mass is kept");
    check(
      rs.p.rdata(SC::pstateT) > 300. && rs.gpv.fluid_eng_src > 0.,
      "the film still heats up in saturated gas", "#733");
  }

  amrex::Print() << "\nFilm 3. Large time step\n";
  {
    // A thin film relaxes to the wall temperature in far less than the step
    PType thin = f;
    thin.rdata(SC::pstateFilmHght) = 1.e-4;
    const Gas sat{.T = 400., .X_fuel = 0.2};
    const Real dt = 1.e-2;
    const Result r = run(sat, thin, dt);
    const Real T1 = r.p.rdata(SC::pstateT);
    amrex::Print() << "Wall " << fd.wall_T << " K, gas " << sat.T
                   << " K: film T 300 -> " << T1 << " K\n";
    check(
      finite(r.gpv) && finiteParticle(r.p) && T1 >= fd.wall_T && T1 <= sat.T,
      "the film temperature stays between the wall and gas temperatures",
      "#735");
  }

  amrex::Print() << "\nFilm 4. A film that evaporates completely\n";
  {
    PType thin = f;
    thin.rdata(SC::pstateFilmHght) = 2.e-5;
    const Gas hot{.T = 800.};
    const Real dt = 1.e-2;
    const Result r = run(hot, thin, dt);
    const Real m0 = filmMass(fdh, thin, r.cBoilT);
    check(r.p.id() < 0, "the film is removed");
    check(
      finite(r.gpv) && massErr(r, m0, 0.) <= tol && speciesOk(r.gpv, tol),
      "the gas gets the film's mass, all as fuel");
  }
}

} // namespace spraytest
