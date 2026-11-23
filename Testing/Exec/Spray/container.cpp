// The SprayParticleContainer on a mesh: source deposition through
// updateParticles and transferSource, against the change of the particles'
// mass, momentum and energy; independence of the box layout; estTimestep;
// computeDerivedVars; two AMR levels.

#include "SprayTest.H"

namespace spraytest {

namespace {

// Deterministic pseudo-random numbers in [0, 1)
class Lcg
{
public:
  Real operator()()
  {
    m_s = m_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<Real>(m_s >> 11) / static_cast<Real>(1ULL << 53);
  }

private:
  unsigned long long m_s = 12345;
};

// Host copy of a MultiFab, ghost cells included
amrex::MultiFab
toHost(const amrex::MultiFab& mf)
{
  amrex::MultiFab h(
    mf.boxArray(), mf.DistributionMap(), mf.nComp(), mf.nGrowVect(),
    amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
  amrex::MultiFab::Copy(h, mf, 0, 0, mf.nComp(), mf.nGrowVect());
  amrex::Gpu::streamSynchronize();
  return h;
}

// Droplets spread over the domain, with some on box and cell edges, next to
// the wall and the outflow, and next to the periodic sides. Velocities up to
// umax per component, away from the non-periodic sides.
std::vector<PType>
makeSpray(const MeshSpec& spec, const Real umax, const Env& env)
{
  Lcg rnd;
  const RealVect plo = spec.plo;
  const Real L = spec.length;
  const Real dx = L / spec.ncell;
  const Real mgs = spec.max_grid_size * dx;
  std::vector<RealVect> pos = {
    {AMREX_D_DECL(plo[0] + mgs, plo[1] + 0.37 * L, 0.)},
    {AMREX_D_DECL(plo[0] + mgs * (1. - 1.e-14), plo[1] + 0.41 * L, 0.)},
    {AMREX_D_DECL(plo[0] + 2. * mgs, plo[1] + 2. * mgs, 0.)},
    {AMREX_D_DECL(plo[0] + 3. * dx, plo[1] + 0.3 * dx, 0.)},
    {AMREX_D_DECL(plo[0] + mgs, plo[1] + 0.05 * dx, 0.)},
    {AMREX_D_DECL(plo[0] + 0.5 * L, plo[1] + L - 0.3 * dx, 0.)},
    {AMREX_D_DECL(plo[0] + L - 0.1 * dx, plo[1] + 0.5 * L, 0.)},
    {AMREX_D_DECL(plo[0] + 0.1 * dx, plo[1] + 0.6 * L, 0.)}};
  for (int n = 0; n < 32; ++n) {
    pos.push_back({AMREX_D_DECL(
      plo[0] + L * rnd(), plo[1] + 2. * dx + (L - 4. * dx) * rnd(), 0.)});
  }
  std::vector<PType> parts;
  for (const auto& x : pos) {
    RealVect v(
      AMREX_D_DECL(umax * (2. * rnd() - 1.), umax * (2. * rnd() - 1.), 0.));
    if (x[1] < plo[1] + 2. * dx) {
      v[1] = std::abs(v[1]);
    } else if (x[1] > plo[1] + L - 2. * dx) {
      v[1] = -std::abs(v[1]);
    }
    const Real dia = 1.e-3 + 7.e-3 * rnd();
    const Real T = 300. + 50. * rnd();
    const Real nppp = 1. + 49. * rnd();
    parts.push_back(makeDrop(x, v, T, dia, nppp));
  }
  // A droplet small enough to evaporate completely
  const Real m0 = 1.1 * SprayUnits::min_mass;
  const Real dia = std::cbrt(6. * m0 / (M_PI * liqRho(env.base, 300., 1.e3)));
  parts.push_back(makeDrop(
    {AMREX_D_DECL(plo[0] + 0.52 * L, plo[1] + 0.48 * L, 0.)}, RealVect(0.),
    300., dia, 5.));
  return parts;
}

// Gas mass, momentum and energy given to the gas over a step, from the
// summed sources
struct Budget
{
  Real mass;
  RealVect mom;
  Real energy;
};
Budget
gasGain(const Mesh& m, const Real dt, const Real dtmod)
{
  const auto s = sumSource(m, m.act);
  Budget b;
  b.mass = s[cRho] * dt * dtmod;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    b.mom[dir] = s[cMom + dir] * dt * dtmod;
  }
  b.energy = s[cEng] * dt * dtmod;
  return b;
}

bool
allFinite(const std::map<int, PType>& parts)
{
  bool ok = true;
  for (const auto& kv : parts) {
    ok = ok && finiteParticle(kv.second);
  }
  return ok;
}

void
testDeposition(Checker& check, Env& env)
{
  const Real tol = env.tol;
  const Real dtmod = env.base.dtmod;
  amrex::Print() << "\nContainer 1. Source deposition over a step\n";
  MeshSpec spec;
  Gas gas;
  gas.T = 1000.;
  gas.vel = RealVect(AMREX_D_DECL(2000., 500., 0.));
  const Real dx = spec.length / spec.ncell;
  const Real umax = 3000.;
  const Real dt = 0.4 * dx / umax;
  const std::vector<PType> spray = makeSpray(spec, umax, env);
  Mesh m = makeMesh(spec, gas);
  addParticles(m, spray);
  const auto p0 = collect(m);
  const Totals t0 = totals(p0, env);
  amrex::Print() << p0.size() << " parcels on " << m.amr->boxArray(0).size()
                 << " boxes\n";
  check(p0.size() == spray.size(), "Redistribute keeps every parcel");

  // moveKickDrift, as PeleC calls it: no subcycles
  sprayStep(m, env, dt, true);
  m.pc->Redistribute();
  const auto p1 = collect(m);
  const Totals t1 = totals(p1, env);
  const Budget g1 = gasGain(m, dt, dtmod);
  const Real dm = t1.mass - t0.mass;
  const RealVect dp = t1.mom - t0.mom;
  const Real mass_err = std::abs(g1.mass + dm) / std::abs(dm);
  const Real mom_err = (g1.mom + dp).vectorLength() / dp.vectorLength();
  const Real eng_err =
    std::abs(g1.energy + t1.energy - t0.energy) / std::abs(g1.energy);
  amrex::Print() << "Particle mass change " << dm / t0.mass << ", "
                 << p0.size() - p1.size() << " parcel(s) evaporated\n"
                 << "Mesh sources + particle change, relative: mass "
                 << mass_err << ", momentum " << mom_err << ", energy "
                 << eng_err << "\n";
  check(
    !m.act[0].contains_nan() && allFinite(p1),
    "the sources and particles are finite");
  check(
    dm < 0. && mass_err <= tol,
    "the mesh mass source is the particles' loss of mass");
  check(
    mom_err <= 1.e-6,
    "the mesh momentum source is the particles' loss of momentum", "#733");
  check(
    eng_err <= 1.e-2,
    "the mesh energy source is the particles' loss of energy, to first order");

  // Per cell: each parcel deposits in the cell it starts the step in
  {
    const auto& geom = m.amr->Geom(0);
    std::map<std::pair<int, int>, Real> expect;
    Real dm_max = 0.;
    for (const auto& kv : p0) {
      const PType& p = kv.second;
      const amrex::IntVect c(AMREX_D_DECL(
        static_cast<int>(std::floor((p.pos(0) - geom.ProbLo(0)) / dx)),
        static_cast<int>(std::floor((p.pos(1) - geom.ProbLo(1)) / dx)), 0));
      const Real mp0 =
        p.rdata(SC::pstateNumDens) * dropMass(env.base, p, env.cBoilT);
      Real mp1 = 0.;
      if (p1.count(kv.first) > 0) {
        const PType& q = p1.at(kv.first);
        mp1 = q.rdata(SC::pstateNumDens) * dropMass(env.base, q, env.cBoilT);
      }
      expect[{c[0], c[1]}] += mp0 - mp1;
      dm_max = std::max(dm_max, std::abs(mp0 - mp1));
    }
    const amrex::MultiFab act = toHost(m.act[0]);
    const Real fac = m.vol(0) * dt * dtmod;
    Real cell_err = 0.;
    bool species_ok = true;
    bool others_zero = true;
    int ncells = 0;
    for (amrex::MFIter mfi(act); mfi.isValid(); ++mfi) {
      auto const& a = act.const_array(mfi);
      amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
        const auto it = expect.find({i, j});
        const Real e = (it == expect.end()) ? 0. : it->second;
        cell_err = std::max(cell_err, std::abs(a(i, j, k, cRho) * fac - e));
        if (a(i, j, k, cRho) != 0.) {
          ++ncells;
        }
        species_ok =
          species_ok && std::abs(a(i, j, k, cRhoY + fuel) - a(i, j, k, cRho)) <=
                          tol * std::abs(a(i, j, k, cRho));
        others_zero = others_zero && a(i, j, k, cTemp) == 0.;
        for (int n = 0; n < NUM_SPECIES; ++n) {
          others_zero =
            others_zero && (n == fuel || a(i, j, k, cRhoY + n) == 0.);
        }
      });
    }
    amrex::Print() << ncells << " cells with a source; largest cell error "
                   << cell_err / dm_max << " of the largest parcel change\n";
    check(
      cell_err <= tol * dm_max,
      "each cell's mass source is the mass lost by the parcels that start "
      "the step in it");
    check(
      species_ok && others_zero,
      "transferSource puts the spray source in the state layout: fuel "
      "species equal to the mass, nothing in T or other species");
  }

  // Drift: x^{n+1} = x^n + dt v^{n+1/2}, wrapped across the periodic side
  {
    const auto& geom = m.amr->Geom(0);
    Real err = 0.;
    for (const auto& kv : p1) {
      const PType& p = p0.at(kv.first);
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        Real x = p.pos(dir) + dt * kv.second.rdata(SC::pstateVel + dir);
        if (geom.isPeriodic(dir)) {
          if (x >= geom.ProbHi(dir)) {
            x -= geom.ProbLength(dir);
          } else if (x < geom.ProbLo(dir)) {
            x += geom.ProbLength(dir);
          }
        }
        err = std::max(err, std::abs(kv.second.pos(dir) - x));
      }
    }
    check(
      err <= 1.e-12, "parcels move by dt times their updated velocity, "
                     "wrapping across the periodic side");
  }

  // moveKick: the second half of the step, without moving
  sprayStep(m, env, dt, false);
  const auto p2 = collect(m);
  const Totals t2 = totals(p2, env);
  const Budget g2 = gasGain(m, dt, dtmod);
  bool still = p2.size() == p1.size();
  for (const auto& kv : p2) {
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      still = still && kv.second.pos(dir) == p1.at(kv.first).pos(dir);
    }
  }
  const Real dm2 = t2.mass - t0.mass;
  amrex::Print() << "Over moveKickDrift and moveKick: particle mass change "
                 << dm2 / t0.mass << ", mass balance rel. err "
                 << std::abs(g2.mass + dm2) / std::abs(dm2) << "\n";
  check(still, "moveKick does not move the parcels");
  check(
    std::abs(g2.mass + dm2) <= tol * std::abs(dm2),
    "the sources of both half steps balance the particle mass over the "
    "step");

  amrex::Print() << "\nContainer 2. Derived spray variables\n";
  {
    const int nvar = SprayParticleContainer::NumDeriveVars();
    amrex::MultiFab dv(m.amr->boxArray(0), m.amr->DistributionMap(0), nvar, 0);
    dv.setVal(0.);
    m.pc->computeDerivedVars(dv, 0, 0);
    // Expected sums over the parcels (liquid density without the boiling
    // temperature limit, as computeDerivedVars)
    Real mass = 0., num = 0., vol = 0., surf = 0., mT = 0., nd = 0.;
    RealVect mv = RealVect::TheZeroVector();
    for (const auto& kv : p2) {
      const PType& p = kv.second;
      const Real n = p.rdata(SC::pstateNumDens);
      const Real d = p.rdata(SC::pstateDia);
      const Real mp = n * dropMass(env.base, p, 1.e4);
      mass += mp;
      num += n;
      vol += n * M_PI / 6. * d * d * d;
      surf += n * M_PI * d * d;
      mT += mp * p.rdata(SC::pstateT);
      mv += mp * dropVel(p);
      nd += n * d;
    }
    const int im = deriveIndex("spray_mass");
    const int inum = deriveIndex("spray_num");
    const int isurf = deriveIndex("spray_surf_area");
    const int iu = deriveIndex("spray_x_vel");
    // Sums over the cells, and sums weighted by mass, number and surface
    auto sum = [&](const std::string& name) {
      return dv.sum(deriveIndex(name));
    };
    auto dot = [&](const int iw, const int iv) {
      return amrex::MultiFab::Dot(dv, iw, dv, iv, 1, 0);
    };
    const Real s_m = dv.sum(im);
    const Real s_rho = sum("spray_density");
    const Real s_num = dv.sum(inum);
    const Real s_vol = sum("spray_vol");
    const Real s_surf = dv.sum(isurf);
    const Real s_vf = sum("spray_vol_frac");
    const Real s_np = sum("num_parcels");
    const Real s_mT = dot(im, deriveIndex("spray_temp"));
    const Real s_d10 = dot(inum, deriveIndex("d10"));
    const Real s_d32 = dot(isurf, deriveIndex("d32"));
    RealVect s_mv;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      s_mv[dir] = dot(im, iu + dir);
    }
    const Real cv = m.vol(0);
    amrex::Print() << "Spray mass on the mesh " << s_m << ", of the parcels "
                   << mass << "\n";
    check(
      relErr(s_m, mass) <= tol && relErr(s_rho * cv, mass) <= tol,
      "spray_mass and spray_density sum to the parcels' mass");
    check(
      relErr(s_num, num) <= tol && s_np == static_cast<Real>(p2.size()) &&
        relErr(s_vol, vol) <= tol && relErr(s_vf * cv, vol) <= tol &&
        relErr(s_surf, surf) <= tol,
      "droplet and parcel counts, volume, volume fraction and surface area");
    check(
      relErr(s_mT, mT) <= tol &&
        (s_mv - mv).vectorLength() <= tol * mv.vectorLength(),
      "spray_temp and spray velocity are mass-weighted averages");
    check(
      relErr(s_d10, nd) <= tol && relErr(s_d32, 6. * vol) <= tol,
      "d10 is the mean and d32 the Sauter mean diameter");
  }
}

void
testLayout(Checker& check, Env& env)
{
  amrex::Print() << "\nContainer 3. Subcycled step across boxes and the "
                    "periodic side\n";
  const Real tol = env.tol;
  const Real dtmod = env.base.dtmod;
  MeshSpec spec;
  const Real cfl = 1.5;
  spec.cfl = cfl;
  Gas gas;
  gas.T = 1000.;
  gas.vel = RealVect(AMREX_D_DECL(2000., 500., 0.));
  const Real dx = spec.length / spec.ncell;
  const Real umax = 3000.;
  // Up to 1.5 cells per step; the parcels near y walls are kept clear
  const Real dt = 0.9 * dx / umax;
  std::vector<PType> spray = makeSpray(spec, umax, env);
  amrex::Vector<amrex::MultiFab> act;
  amrex::Long nboxes = 0;
  for (const int mgs : {8, 32}) {
    spec.max_grid_size = mgs;
    Mesh m = makeMesh(spec, gas);
    addParticles(m, spray);
    const Totals t0 = totals(collect(m), env);
    m.pc->moveKickDrift(
      m.state[0], m.tmp[0], 0, dt, 0., false, false, m.state_ghosts,
      m.source_ghosts, true, env.tp, cfl);
    // Source on the ghost cells, before transferSource sums it up
    const amrex::MultiFab tmp = toHost(m.tmp[0]);
    Real all = 0., ghost = 0.;
    for (amrex::MFIter mfi(tmp); mfi.isValid(); ++mfi) {
      auto const& a = tmp.const_array(mfi, sRho);
      const amrex::Box vb = mfi.validbox();
      amrex::LoopOnCpu(mfi.fabbox(), [&](int i, int j, int k) {
        all += a(i, j, k);
        if (!vb.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
          ghost += a(i, j, k);
        }
      });
    }
    m.pc->transferSource(m.source_ghosts, 0, m.tmp[0], m.act[0]);
    m.pc->Redistribute();
    const Totals t1 = totals(collect(m), env);
    const Real valid = m.act[0].sum(cRho);
    const Real dm = t1.mass - t0.mass;
    const Real gain = valid * m.vol(0) * dt * dtmod;
    amrex::Print() << m.amr->boxArray(0).size() << " box(es): ghost cell "
                   << "share of the mass source " << ghost / all
                   << ", mass balance rel. err "
                   << std::abs(gain + dm) / std::abs(dm) << "\n";
    if (mgs == 8) {
      nboxes = m.amr->boxArray(0).size();
      check(ghost != 0., "parcels moving across boxes deposit in ghost cells");
    }
    check(
      std::abs(valid - all) <= tol * std::abs(all),
      "transferSource moves all of the ghost cell source onto valid cells");
    check(
      std::abs(gain + dm) <= tol * std::abs(dm),
      "with subcycles the mesh mass source is the particles' loss of mass");
    act.emplace_back(
      amrex::BoxArray(m.amr->Geom(0).Domain()),
      amrex::DistributionMapping(amrex::BoxArray(m.amr->Geom(0).Domain())),
      nComp, 0);
    act.back().ParallelCopy(m.act[0]);
  }
  amrex::MultiFab::Subtract(act[0], act[1], 0, 0, nComp, 0);
  Real diff = 0., ref = 0.;
  for (int n = 0; n < nComp; ++n) {
    diff = std::max(diff, act[0].norm0(n) / std::max(act[1].norm0(n), 1.e-300));
    ref = std::max(ref, act[1].norm0(n));
  }
  amrex::Print() << "Largest relative difference between " << nboxes
                 << " boxes and one: " << diff << "\n";
  check(
    diff <= 1.e-12 && ref > 0., "the sources do not depend on the box layout");
}

void
testTimestep(Checker& check, Env& env)
{
  amrex::Print() << "\nContainer 4. Particle time step\n";
  MeshSpec spec;
  Mesh m = makeMesh(spec, Gas());
  const Real dx = spec.length / spec.ncell;
  const Real cfl = SprayParticleContainer::spray_cfl;
  const Real est0 = m.pc->estTimestep(0);
  check(est0 > 1.e40, "no particles: no limit");
  const std::vector<RealVect> vels = {
    {AMREX_D_DECL(100., -300., 0.)},
    {AMREX_D_DECL(-700., 20., 0.)},
    {AMREX_D_DECL(0., 0., 0.)}};
  std::vector<PType> parts;
  Real vmax = 0.;
  for (int n = 0; n < 3; ++n) {
    const RealVect x(
      AMREX_D_DECL(spec.plo[0] + 0.2 + 0.3 * n, spec.plo[1] + 0.5, 0.));
    parts.push_back(makeDrop(x, vels[n], 300., 1.e-3, 1.));
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      vmax = std::max(vmax, std::abs(vels[n][dir]));
    }
  }
  addParticles(m, parts);
  const Real est = m.pc->estTimestep(0);
  const Real expect = cfl * dx / vmax;
  amrex::Print() << "estTimestep " << est << ", cfl dx / max|v_i| " << expect
                 << "\n";
  check(
    std::isfinite(est) && est > 0. && relErr(est, expect) <= 1.e-14,
    "estTimestep is the particle CFL times dx over the largest velocity "
    "component");
  // A removed particle, not yet redistributed, does not count
  PType dead = makeDrop(
    RealVect(AMREX_D_DECL(spec.plo[0] + 0.5, spec.plo[1] + 0.5, 0.)),
    RealVect(1.e6), 300., 1.e-3, 1.);
  dead.id() = -1;
  insertParticle(m, dead);
  check(
    m.pc->estTimestep(0) == est, "a removed particle (id < 0) is ignored",
    "#733");
  SprayData sd = env.base;
  sd.fixed_parts = true;
  setSprayData(sd);
  check(m.pc->estTimestep(0) < 0., "fixed particles set no limit");
  setSprayData(env.base);
}

void
testTwoLevels(Checker& check, Env& env)
{
  amrex::Print() << "\nContainer 5. Two AMR levels\n";
  const Real tol = env.tol;
  MeshSpec spec;
  spec.max_level = 1;
  spec.fine_region = amrex::Box(amrex::IntVect(8), amrex::IntVect(23));
  Gas gas;
  gas.T = 1000.;
  gas.vel = RealVect(AMREX_D_DECL(2000., 500., 0.));
  const Real dx = spec.length / spec.ncell;
  const Real umax = 3000.;
  const Real dt = 0.2 * dx / umax;
  Mesh m = makeMesh(spec, gas);
  check(m.amr->finestLevel() == 1, "the AmrCore makes the fine level");
  addParticles(m, makeSpray(spec, umax, env));
  std::map<int, int> lev;
  const auto p0 = collect(m, &lev);
  const amrex::BoxArray& fba = m.amr->boxArray(1);
  bool levels_ok = true;
  int nfine = 0;
  for (const auto& kv : p0) {
    amrex::IntVect c;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      c[dir] = static_cast<int>(
        std::floor((kv.second.pos(dir) - spec.plo[dir]) / (0.5 * dx)));
    }
    const int expect = fba.contains(c) ? 1 : 0;
    levels_ok = levels_ok && lev[kv.first] == expect;
    nfine += expect;
  }
  amrex::Print() << nfine << " of " << p0.size()
                 << " parcels on the fine level\n";
  check(
    levels_ok && nfine > 0,
    "Redistribute puts each parcel on the finest level covering it");
  const Totals t0 = totals(p0, env);
  sprayStep(m, env, dt, true);
  m.pc->Redistribute();
  const Totals t1 = totals(collect(m), env);
  const Budget g = gasGain(m, dt, env.base.dtmod);
  const Real dm = t1.mass - t0.mass;
  amrex::Print() << "Mass balance rel. err "
                 << std::abs(g.mass + dm) / std::abs(dm) << "\n";
  check(
    std::abs(g.mass + dm) <= tol * std::abs(dm),
    "each level's sources balance the mass of its parcels");
}

} // namespace

void
testContainer(Checker& check, Env& env)
{
  testDeposition(check, env);
  testLayout(check, env);
  testTimestep(check, env);
  testTwoLevels(check, env);
}

} // namespace spraytest
