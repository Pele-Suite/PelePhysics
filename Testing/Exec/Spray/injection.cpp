// Spray jets and initial sprays: SprayJet coordinate transforms,
// SprayParticleContainer::sprayInjection with a droplet size distribution,
// estTimestep with a jet, and uniformSprayInit.

#include "SprayTest.H"
#include "SprayJet.H"

namespace spraytest {

namespace {

// Exposes the coordinate transform of a jet
class TestJet : public SprayJet
{
public:
  TestJet(
    const RealVect& norm,
    const RealVect& cent,
    const amrex::GpuArray<Real, AMREX_SPACEDIM * AMREX_SPACEDIM>& rot,
    const amrex::GpuArray<Real, AMREX_SPACEDIM>& shift)
  {
    set_jet_norm(norm);
    m_cent = cent;
    m_trans_matrix = rot;
    m_translation = shift;
  }
};

void
testTransforms(Checker& check, const Real tol)
{
  amrex::Print() << "\nInjection 1. Spray jet transforms\n";
  const Real a = M_PI / 6.;
  amrex::GpuArray<Real, AMREX_SPACEDIM * AMREX_SPACEDIM> rot{};
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
    rot[i * AMREX_SPACEDIM + i] = 1.;
  }
  rot[0] = std::cos(a);
  rot[1] = -std::sin(a);
  rot[AMREX_SPACEDIM] = std::sin(a);
  rot[AMREX_SPACEDIM + 1] = std::cos(a);
  const amrex::GpuArray<Real, AMREX_SPACEDIM> shift = {
    AMREX_D_DECL(0.1, -0.2, 0.3)};
  const RealVect norm(AMREX_D_DECL(1., 1., 0.));
  const RealVect cent(AMREX_D_DECL(0.5, 0.25, 0.5));
  TestJet jet(norm, cent, rot, shift);
  Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(3., 4., 12.)};
  const RealVect xin(x);
  jet.transform_to_new_cs(x);
  RealVect xexp = xin;
  xexp[0] = std::cos(a) * xin[0] - std::sin(a) * xin[1];
  xexp[1] = std::sin(a) * xin[0] + std::cos(a) * xin[1];
  bool ok_pos = true;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    ok_pos = ok_pos && std::abs(x[dir] - xexp[dir] - shift[dir]) <= tol;
  }
  check(ok_pos, "a position is rotated and translated");

  // Injection velocity and location in the jet coordinates
  const RealVect n = norm / norm.vectorLength();
  const Real umag = 2000.;
  const Real radius = 0.01;
  RealVect vel, loc;
  jet.transform_loc_vel(0., 0.4, radius, umag, 0., vel, loc);
  const bool along = (vel / umag - n).vectorLength() <= tol;
  const Real spread = 0.3;
  jet.transform_loc_vel(spread, 0.4, radius, umag, 0., vel, loc);
  const Real angle = std::acos(vel.dotProduct(n) / vel.vectorLength());
  check(
    along && relErr(vel.vectorLength(), umag) <= tol &&
      std::abs(angle - spread) <= 1.e-8,
    "the injection velocity has the jet speed and spread angle");
  check(
    std::abs((loc - cent).dotProduct(n)) <= tol &&
      relErr((loc - cent).vectorLength(), radius) <= tol,
    "the injection location is on the jet's cross section");
}

void
testJetInjection(Checker& check, Env& env)
{
  amrex::Print() << "\nInjection 2. Spray jet with a size distribution\n";
  const Real tol = env.tol;
  MeshSpec spec;
  Mesh m = makeMesh(spec, Gas());
  // The container's jets (InitSprayParticles in SprayParticlesInitInsert.cpp)
  // set its time step; the test injects through its own copy of the first
  std::vector<std::string> names;
  amrex::ParmParse("spraytest").getarr("jet_names", names);
  m.pc->InitSprayParticles(true);
  SprayJet test_jet(names[0], m.amr->Geom(0));
  SprayJet* jet = &test_jet;
  const RealVect n = jet->jet_norm();
  const RealVect t(AMREX_D_DECL(-n[1], n[0], 0.));
  const RealVect c = jet->jet_cent();
  const Real umag = jet->jet_vel(0.);
  const Real half_spread = 0.5 * jet->spread_angle();
  const Real dx = spec.length / spec.ncell;
  const Real cfl = SprayParticleContainer::spray_cfl;

  // With no particles, the jet velocity sets the time step
  const Real est = m.pc->estTimestep(0);
  check(
    relErr(est, cfl * dx / jet->max_jet_vel()) <= 1.e-14,
    "estTimestep accounts for the jet velocity");

  const Real dt = 2.e-6;
  const int nstep = 300;
  Real time = 0.;
  bool vel_ok = true;
  bool pos_ok = true;
  std::map<int, PType> seen;
  for (int s = 0; s < nstep; ++s) {
    const Real inj_time0 = jet->m_totalInjTime;
    time += dt;
    m.pc->sprayInjection(time, jet, dt, 0);
    m.pc->Redistribute();
    const Real dt_inj = jet->m_totalInjTime - inj_time0;
    for (const auto& kv : collect(m)) {
      if (seen.count(kv.first) > 0) {
        continue;
      }
      const PType& p = kv.second;
      seen[kv.first] = p;
      const RealVect v = dropVel(p);
      const RealVect dxp = p.pos() - c;
      vel_ok = vel_ok && relErr(v.vectorLength(), umag) <= tol &&
               std::acos(std::min<Real>(1., v.dotProduct(n) / umag)) <=
                 half_spread + 1.e-12;
      // On the jet's cross section, then moved by a random part of the
      // injection interval
      const Real along = dxp.dotProduct(n);
      const Real across = dxp.dotProduct(t);
      pos_ok = pos_ok && along >= -tol &&
               along <= dt_inj * v.dotProduct(n) + tol &&
               std::abs(across) <= 0.5 * jet->jet_dia() +
                                     dt_inj * std::abs(v.dotProduct(t)) + tol;
    }
  }
  const int nparts = static_cast<int>(seen.size());
  Real mass = 0.;
  Real sum_d = 0.;
  for (const auto& kv : seen) {
    const PType& p = kv.second;
    mass += p.rdata(SC::pstateNumDens) * dropMass(env.base, p, 1.e4);
    sum_d += p.rdata(SC::pstateDia);
  }
  const Real mdot = jet->mass_flow_rate(0.);
  const Real mean_d = sum_d / nparts;
  amrex::ParmParse pp("spray." + jet->jet_name());
  Real d_mean = 0., d_std = 0.;
  pp.get("mean_dia", d_mean);
  pp.get("std_dev", d_std);
  amrex::Print() << nparts << " parcels injected over " << nstep
                 << " steps; injected mass / (mdot t) "
                 << jet->m_totalInjMass / (mdot * time) << ", mean diameter "
                 << mean_d << " (" << d_mean << " expected)\n";
  check(nparts > 50, "the jet injects parcels");
  check(
    relErr(mass, jet->m_totalInjMass) <= tol,
    "the parcels have the mass the jet counts as injected");
  check(
    std::abs(jet->m_totalInjMass / (mdot * time) - 1.) <= 0.1 &&
      jet->m_totalInjTime <= time * (1. + tol),
    "the injected mass follows the mass flow rate (within 10%)", "#735");
  check(
    std::abs(mean_d - d_mean) <= 5. * d_std / std::sqrt(Real(nparts)),
    "the droplet diameters follow the distribution");
  check(vel_ok, "the parcels have the jet speed, within the spread angle");
  check(pos_ok, "the parcels start on the jet's cross section");

  // An inactive jet injects nothing
  const Real end = jet->end_time();
  jet->set_end_time(0.5 * time);
  const auto before = collect(m).size();
  m.pc->sprayInjection(time + dt, jet, dt, 0);
  m.pc->Redistribute();
  jet->set_end_time(end);
  check(collect(m).size() == before, "a jet past its end time injects nothing");
}

void
testUniformInit(Checker& check)
{
  amrex::Print() << "\nInjection 3. Uniform initial spray\n";
  // A domain offset by less than half a lattice spacing: a lattice placed
  // from the origin instead of prob_lo is still inside it
  MeshSpec spec;
  spec.plo = RealVect(AMREX_D_DECL(0.1, 0.1, 0.));
  Mesh m = makeMesh(spec, Gas());
  const amrex::IntVect np(AMREX_D_DECL(2, 2, 1));
  const RealVect vel(AMREX_D_DECL(10., -20., 0.));
  const Real Y[SPRAY_FUEL_NUM] = {1.};
  m.pc->uniformSprayInit(np, vel, 3.e-3, 310., Y, 0, 1, 7.);
  const auto parts = collect(m);
  const int expect = AMREX_D_TERM(np[0], *np[1], *np[2]);
  // Every lattice point plo + (i + 1/2) L / n, once
  std::map<std::pair<long, long>, int> lattice;
  bool vals_ok = true;
  for (const auto& kv : parts) {
    const PType& p = kv.second;
    long ij[2];
    for (int dir = 0; dir < 2; ++dir) {
      const Real s = (p.pos(dir) - spec.plo[dir]) / spec.length * np[dir] - 0.5;
      ij[dir] = std::lround(s);
      vals_ok = vals_ok && std::abs(s - ij[dir]) <= 1.e-12 && ij[dir] >= 0 &&
                ij[dir] < np[dir];
    }
    ++lattice[{ij[0], ij[1]}];
    vals_ok = vals_ok && dropVel(p) == vel && p.rdata(SC::pstateDia) == 3.e-3 &&
              p.rdata(SC::pstateT) == 310. && p.rdata(SC::pstateY) == 1. &&
              p.rdata(SC::pstateNumDens) == 7. &&
              p.rdata(SC::pstateFilmHght) == 0.;
  }
  check(
    static_cast<int>(parts.size()) == expect &&
      static_cast<int>(lattice.size()) == expect,
    "uniformSprayInit makes one parcel per lattice point");
  check(
    vals_ok, "the parcels are at plo + (i + 1/2) L / n with the given state",
    "#733");
}

} // namespace

void
testInjection(Checker& check, Env& env)
{
  testTransforms(check, env.tol);
  testJetInjection(check, env);
  testUniformInit(check);
}

} // namespace spraytest
