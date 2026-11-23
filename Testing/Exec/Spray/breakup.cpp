// New droplets in the container: TAB and KH-RT breakup and the splash model
// at a wall (CreateSBDroplets), against the mass and momentum of the parcels
// and the gas sources.

#include "SprayTest.H"

namespace spraytest {

namespace {

struct BreakupResult
{
  int steps = 0;
  int parcels = 0;
  Real mass_err = 0.;
  Real momx_err = 0.;
  Real max_vy = 0.; // largest child velocity across the parent's path
  bool finite = true;
};

// A fast droplet in still or moving gas, stepped until it breaks up
BreakupResult
breakup(const Env& env, const Real gas_u, const Real rel_u)
{
  MeshSpec spec;
  spec.bc_lo = amrex::IntVect(0);
  spec.bc_hi = amrex::IntVect(0);
  Gas gas;
  gas.vel[0] = gas_u;
  Mesh m = makeMesh(spec, gas);
  const Real dx = spec.length / spec.ncell;
  const RealVect v0(AMREX_D_DECL(gas_u + rel_u, 0., 0.));
  // The same step in both frames; the parcel moves at most 0.6 dx
  const Real dt = 0.4 * dx / std::abs(rel_u);
  addParticles(m, {makeDrop(spec.plo + RealVect(0.5), v0, 300., 1.e-2, 5.)});
  const Totals t0 = totals(collect(m), env);
  BreakupResult r;
  std::map<int, PType> parts;
  while (r.steps < 400 && r.parcels < 2) {
    sprayStep(m, env, dt, true);
    m.pc->Redistribute();
    parts = collect(m);
    r.parcels = static_cast<int>(parts.size());
    ++r.steps;
  }
  const Totals t1 = totals(parts, env);
  const auto s = sumSource(m, m.act);
  const Real fac = dt * env.base.dtmod;
  r.mass_err = std::abs(t1.mass + s[cRho] * fac - t0.mass) / t0.mass;
  r.momx_err =
    std::abs(t1.mom[0] + s[cMom] * fac - t0.mom[0]) / std::abs(t0.mom[0]);
  for (const auto& kv : parts) {
    r.finite = r.finite && finiteParticle(kv.second);
    r.max_vy = std::max(r.max_vy, std::abs(kv.second.rdata(SC::pstateVel + 1)));
  }
  return r;
}

struct SplashResult
{
  Totals t1; // the parcels after the last droplet hits the wall
  Real mass_err = 0.;
  Real derived_film_mass = 0.;
  RealVect vel; // of the parent if still a droplet
  bool inside = true;
  bool finite = true;
};

// Droplets hitting the slip wall at y low with the splash model on; the
// first in the list arrives a step before the others
SplashResult
splash(const Env& env, const std::vector<RealVect>& vels, const Real wall_T)
{
  SprayData sd = env.base;
  sd.do_splash = true;
  sd.do_breakup = 0;
  sd.wall_T = wall_T;
  const ScopedSprayData scoped(env, sd);
  MeshSpec spec;
  Mesh m = makeMesh(spec, Gas());
  const Real dx = spec.length / spec.ncell;
  const RealVect x(AMREX_D_DECL(spec.plo[0] + 0.5, spec.plo[1] + 0.2 * dx, 0.));
  SplashResult r;
  Totals t0;
  Real fac = 0.;
  for (size_t n = 0; n < vels.size(); ++n) {
    const Real dt = 0.4 * dx / std::abs(vels[n][1]);
    addParticles(m, {makeDrop(x, vels[n], 300., 5.e-3, 10.)});
    if (n == vels.size() - 1) {
      t0 = totals(collect(m), env);
      for (int lev = 0; lev <= m.amr->finestLevel(); ++lev) {
        m.act[lev].setVal(0.);
      }
      fac = dt * env.base.dtmod;
    }
    sprayStep(m, env, dt, true);
    m.pc->Redistribute();
  }
  const auto parts = collect(m);
  r.t1 = totals(parts, env);
  const auto s = sumSource(m, m.act);
  r.mass_err =
    std::abs(
      r.t1.mass + r.t1.film_mass + s[cRho] * fac - t0.mass - t0.film_mass) /
    (t0.mass + t0.film_mass);
  for (const auto& kv : parts) {
    const PType& p = kv.second;
    r.finite = r.finite && finiteParticle(p);
    r.inside = r.inside && p.pos(1) >= spec.plo[1];
    if (p.rdata(SC::pstateFilmHght) == 0.) {
      r.vel = dropVel(p);
    }
  }
  // Film mass from computeDerivedVars
  amrex::MultiFab dv(
    m.amr->boxArray(0), m.amr->DistributionMap(0),
    SprayParticleContainer::NumDeriveVars(), 0);
  dv.setVal(0.);
  m.pc->computeDerivedVars(dv, 0, 0);
  r.derived_film_mass = dv.sum(deriveIndex("wall_film_mass"));
  amrex::Print() << r.t1.count << " droplets, " << r.t1.films
                 << " film(s), mass rel. err " << r.mass_err << ", film mass "
                 << r.t1.film_mass << " (derived " << r.derived_film_mass
                 << "), droplet velocity " << r.vel << "\n";
  return r;
}

} // namespace

void
testBreakupSplash(Checker& check, Env& env)
{
  const Real tol = env.tol;
  // Breakup without evaporation, to see the new parcels alone
  SprayData sd = env.base;
  sd.mass_trans = false;
  sd.do_splash = false;
  const Real rel_u = 1.e4;
  const Real gas_u = 5.e3;
  for (const int model : {1, 2}) {
    const std::string name = (model == 1) ? "TAB" : "KH-RT";
    amrex::Print() << "\nBreakup " << model << ". " << name << "\n";
    sd.do_breakup = model;
    BreakupResult r, rg;
    {
      const ScopedSprayData scoped(env, sd);
      r = breakup(env, 0., rel_u);
      rg = breakup(env, gas_u, rel_u);
    }
    amrex::Print() << r.parcels << " parcels after " << r.steps
                   << " steps; mass rel. err " << r.mass_err
                   << ", x momentum rel. err " << r.momx_err
                   << "; child velocity across the path " << r.max_vy
                   << " (still gas, " << r.steps << " steps), " << rg.max_vy
                   << " (gas at " << gas_u << " cm/s, " << rg.steps
                   << " steps)\n";
    check(r.parcels > 1 && r.finite, name + " breakup makes finite parcels");
    check(r.mass_err <= tol, name + " breakup conserves the liquid mass");
    check(
      r.momx_err <= 1.e-8,
      name + " breakup conserves momentum along the parent's path, with the "
             "drag source");
    check(
      rg.steps == r.steps && relErr(rg.max_vy, r.max_vy) <= 1.e-6,
      name + " breakup does not depend on the frame of reference",
      (model == 2) ? "#735" : "");
  }

  const RealVect slow(AMREX_D_DECL(30., -100., 0.));
  const RealVect fast(AMREX_D_DECL(900., -3000., 0.));
  amrex::Print() << "\nSplash 1. Slow droplet on a cool wall: deposits\n";
  {
    const SplashResult r = splash(env, {slow}, 300.);
    check(
      r.t1.count == 0 && r.t1.films == 1 && r.finite,
      "the droplet becomes a wall film");
    check(r.mass_err <= tol, "the film has the droplet's mass");
    check(
      relErr(r.derived_film_mass, r.t1.film_mass) <= tol,
      "the derived wall_film_mass is the film's mass", "#735");
  }
  amrex::Print() << "\nSplash 2. Fast droplet on a cool wall: splashes\n";
  {
    const SplashResult r = splash(env, {fast}, 300.);
    check(
      r.t1.count > 0 && r.t1.films == 1 && r.finite && r.inside,
      "splashed droplets inside the domain and a film");
    check(r.mass_err <= tol, "splashing conserves the liquid mass");
  }
  amrex::Print() << "\nSplash 3. Fast droplet on a hot wall: thermal "
                    "breakup\n";
  {
    const SplashResult r = splash(env, {fast}, 500.);
    check(
      r.t1.count > 1 && r.t1.films == 0 && r.finite && r.inside,
      "the droplet breaks up into droplets inside the domain, no film");
    check(r.mass_err <= tol, "thermal breakup conserves the liquid mass");
  }
  amrex::Print() << "\nSplash 4. Slow droplet on a hot wall: rebounds\n";
  {
    const SplashResult r = splash(env, {slow}, 500.);
    check(
      r.t1.count == 1 && r.t1.films == 0 && r.inside && r.vel[1] > 0. &&
        r.vel[1] < -slow[1],
      "the droplet rebounds off the wall, slower");
    check(r.mass_err <= tol, "rebound conserves the liquid mass");
  }
  amrex::Print() << "\nSplash 5. Fast droplet onto an existing film\n";
  {
    const SplashResult r = splash(env, {slow, fast}, 300.);
    check(r.finite && r.inside, "the parcels are finite and inside");
    check(
      r.mass_err <= tol, "a droplet hitting a film conserves the liquid mass",
      "#735");
  }
}

} // namespace spraytest
