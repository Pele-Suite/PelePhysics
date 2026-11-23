// Domain boundaries: check_bounds and the gas interpolation next to a wall,
// then the full particle move of the container through reflective walls,
// outflows and periodic boundaries.

#include "SprayTest.H"
#include "SprayInterpolation.H"

namespace spraytest {

namespace {

// A check_bounds case: the position, the expected and the returned outflow
// test and flags. At namespace scope, for the device lambda.
struct BoundsCase
{
  RealVect pos;
  bool periodic;
  bool out;
  amrex::IntVect flags;
  bool out_got = false;
  amrex::IntVect flags_got;
};

void
checkBoundsFlags(Checker& check)
{
  using Case = BoundsCase;
  constexpr Real dx = fabDx;
  // Reflective in x, outflow in y; then periodic
  const amrex::IntVect z(0);
  amrex::Gpu::ManagedVector<Case> cases = {
    {{AMREX_D_DECL(0.5, 0.5, 0.5)}, false, false, z},
    {{AMREX_D_DECL(0.03, 0.5, 0.5)}, false, false, {AMREX_D_DECL(-2, 0, 0)}},
    {{AMREX_D_DECL(0.98, 0.5, 0.5)}, false, false, {AMREX_D_DECL(2, 0, 0)}},
    {{AMREX_D_DECL(-0.01, 0.5, 0.5)}, false, false, {AMREX_D_DECL(-1, 0, 0)}},
    {{AMREX_D_DECL(0.5, 0.5, 0.5)}, false, false, z},
    {{AMREX_D_DECL(1.01, 0.5, 0.5)}, false, false, {AMREX_D_DECL(1, 0, 0)}},
    {{AMREX_D_DECL(0.5, 0.03, 0.5)}, false, false, {AMREX_D_DECL(0, -2, 0)}},
    {{AMREX_D_DECL(0.03, 0.98, 0.5)}, false, false, {AMREX_D_DECL(-2, 2, 0)}},
    {{AMREX_D_DECL(0.5, -0.01, 0.5)}, false, true, z},
    {{AMREX_D_DECL(0.5, 1.01, 0.5)}, false, true, z},
    {{AMREX_D_DECL(1.01, -0.01, 0.5)}, true, false, z}};
  const int nc = static_cast<int>(cases.size());
  Case* cp = cases.data();
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) noexcept {
    const RealVect plo(0.);
    const RealVect phi(1.);
    const RealVect dxv(dx);
    // Flags carried from one call to the next, as in updateParticles
    amrex::IntVect bflags(7);
    for (int c = 0; c < nc; ++c) {
      amrex::IntVect lo(AMREX_D_DECL(1, -1, 0));
      amrex::IntVect hi(AMREX_D_DECL(1, -1, 0));
      if (cp[c].periodic) {
        lo = hi = amrex::IntVect(0);
      }
      cp[c].out_got = check_bounds(cp[c].pos, plo, phi, dxv, lo, hi, bflags);
      cp[c].flags_got = bflags;
    }
  });
  amrex::Gpu::streamSynchronize();
  bool flags_ok = true;
  bool out_ok = true;
  for (const Case& c : cases) {
    out_ok = out_ok && c.out_got == c.out;
    if (!c.out && c.flags_got != c.flags) {
      flags_ok = false;
      amrex::Print() << "Position " << c.pos << ": flags " << c.flags_got
                     << ", expected " << c.flags << "\n";
    }
  }
  check(
    flags_ok,
    "check_bounds flags inside, next to and outside a wall, reset on each "
    "call",
    "#733");
  check(out_ok, "check_bounds returns true only outside an outflow");

  // Temperature linear in x: interpolated exactly inside, from the interior
  // cell next to a wall
  const amrex::FArrayBox sfab = gasFab(Gas{.T = 600., .dTdx = 1000.});
  const amrex::Box sbox = sfab.box();
  amrex::Gpu::ManagedVector<Real> Tint(2);
  Real* Tint_p = Tint.data();
  auto const& rho = sfab.const_array(cRho);
  auto const& mom = sfab.const_array(cMom);
  auto const& eng = sfab.const_array(cEng);
  auto const& T = sfab.const_array(cTemp);
  auto const& rhoY = sfab.const_array(cRhoY);
  amrex::ParallelFor(2, [=] AMREX_GPU_DEVICE(int n) noexcept {
    RealVect x(0.52);
    x[0] = (n == 0) ? 0.37 : 0.03;
    amrex::IntVect bflags;
    check_bounds(
      x, RealVect(0.), RealVect(1.), RealVect(dx),
      amrex::IntVect(AMREX_D_DECL(1, -1, -1)),
      amrex::IntVect(AMREX_D_DECL(1, -1, -1)), bflags);
    const RealVect lx = x / dx + 0.5;
    amrex::GpuArray<amrex::IntVect, AMREX_D_PICK(2, 4, 8)> indx;
    amrex::GpuArray<Real, AMREX_D_PICK(2, 4, 8)> w;
    trilinear_interp(lx.floor(), lx, indx.data(), w.data(), bflags);
    GasPhaseVals gpv;
    gpv.reset();
    InterpolateGasPhase(
      gpv, sbox, rho, rhoY, T, mom, eng, indx.data(), w.data());
    Tint_p[n] = gpv.T_fluid;
  });
  amrex::Gpu::streamSynchronize();
  amrex::Print() << "Interpolated T: " << Tint[0] << " (970 expected), "
                 << Tint[1] << " next to the wall (650 expected)\n";
  check(
    relErr(Tint[0], 970.) <= 1.e-8 && relErr(Tint[1], 650.) <= 1.e-8,
    "gas temperature interpolation, inside and next to a wall");
}

// Move particles one step in gas moving with velocity vel (no drag) and
// return them after Redistribute
std::map<int, PType>
moveParticles(
  const Env& env,
  const MeshSpec& spec,
  const RealVect& vel,
  const std::vector<PType>& parts,
  const Real dt,
  const int nstep = 1,
  const Real cfl = -1.)
{
  Gas gas;
  gas.vel = vel;
  Mesh m = makeMesh(spec, gas);
  addParticles(m, parts);
  for (int n = 0; n < nstep; ++n) {
    sprayStep(m, env, dt, true, cfl);
    m.pc->Redistribute();
  }
  return collect(m);
}

} // namespace

void
testBoundaries(Checker& check, Env& env)
{
  amrex::Print() << "\nBoundaries 1. Boundary flags and interpolation\n";
  checkBoundsFlags(check);

  amrex::Print() << "\nBoundaries 2. Particle move through the domain "
                    "boundaries\n";
  // No mass transfer; the particles move with the gas, so without drag
  SprayData sd = env.base;
  sd.mass_trans = false;
  const ScopedSprayData scoped(env, sd);
  MeshSpec spec; // periodic in x, wall at y low, outflow at y high
  const RealVect plo = spec.plo;
  const RealVect phi = plo + RealVect(spec.length);
  const Real dx = spec.length / spec.ncell;
  const Real u = 2000.;
  const Real dt = 0.4 * dx / u;
  const Real ptol = 1.e-12;
  auto drop = [&](const Real x, const Real y, const RealVect& v) {
    return makeDrop(RealVect(AMREX_D_DECL(x, y, 0.)), v, 300., 5.e-3, 2.);
  };
  auto close = [&](const PType& p, const Real x, const Real y) {
    return std::abs(p.pos(0) - x) <= ptol && std::abs(p.pos(1) - y) <= ptol;
  };
  {
    const RealVect v(AMREX_D_DECL(u, -u, 0.));
    const Real xm = plo[0] + 0.3;
    const std::vector<PType> parts = {
      drop(xm, plo[1] + 0.2 * dx, v),           // hits the wall
      drop(phi[0] - 0.1 * dx, plo[1] + 0.5, v), // crosses the periodic side
      drop(phi[0] - 0.1 * dx, plo[1] + 0.2 * dx, v)}; // both
    const int id0 = static_cast<int>(parts[0].id());
    for (const int wall : {3, 4, 5}) {
      MeshSpec s = spec;
      s.bc_lo[1] = wall;
      auto out = moveParticles(env, s, v, parts, dt);
      const bool ok =
        out.size() == 3 && close(out[id0], xm + 0.4 * dx, plo[1] + 0.2 * dx) &&
        close(out[id0 + 1], plo[0] + 0.3 * dx, plo[1] + 0.5 - 0.4 * dx) &&
        close(out[id0 + 2], plo[0] + 0.3 * dx, plo[1] + 0.2 * dx);
      bool vel_ok = true;
      for (int n = 0; n < 3; ++n) {
        const RealVect v1 = dropVel(out[id0 + n]);
        const Real vy = (n == 1) ? -u : u;
        vel_ok = vel_ok && std::abs(v1[0] - u) <= 1.e-9 * u &&
                 std::abs(v1[1] - vy) <= 1.e-9 * u;
      }
      if (!ok) {
        for (const auto& kv : out) {
          amrex::Print() << "  particle " << kv.first << " at "
                         << kv.second.pos() << "\n";
        }
      }
      check(
        ok && vel_ok, "a particle crossing a " +
                        std::string(
                          wall == 3   ? "symmetry plane"
                          : wall == 4 ? "slip wall"
                                      : "no-slip wall") +
                        " is reflected, its normal velocity reversed; one "
                        "crossing a periodic side wraps around");
    }
  }
  {
    const RealVect v(AMREX_D_DECL(0., u, 0.));
    const std::vector<PType> parts = {
      drop(plo[0] + 0.5, phi[1] - 0.1 * dx, v),
      drop(plo[0] + 0.5, phi[1] - 0.5 * dx, v)};
    const int id0 = static_cast<int>(parts[0].id());
    auto out = moveParticles(env, spec, v, parts, dt);
    check(
      out.size() == 1 && out.count(id0 + 1) == 1 &&
        close(out[id0 + 1], plo[0] + 0.5, phi[1] - 0.1 * dx),
      "a particle leaving through an outflow is removed, one staying inside "
      "is kept");

    MeshSpec s = spec;
    s.bc_lo[0] = 2;
    s.bc_hi[0] = 2;
    const RealVect vx(AMREX_D_DECL(-u, 0., 0.));
    const PType p = drop(plo[0] + 0.1 * dx, plo[1] + 0.5, vx);
    out = moveParticles(env, s, vx, {p}, dt);
    check(out.empty(), "a particle leaving through an x outflow is removed");
  }
  {
    // Three subcycles per step: the particle bounces off the wall within
    // a step and stays inside over many steps
    const RealVect v(AMREX_D_DECL(0., -u, 0.));
    const Real cfl = 1.5;
    MeshSpec s = spec;
    s.cfl = cfl;
    const PType p = drop(plo[0] + 0.5, plo[1] + 0.7 * dx, v);
    auto out = moveParticles(env, s, v, {p}, cfl * dx / u, 1, cfl);
    const PType q = out.empty() ? p : out.begin()->second;
    const RealVect vq = dropVel(q);
    amrex::Print() << "After one subcycled step: y - y_lo = "
                   << (q.pos(1) - plo[1]) / dx << " dx, v = " << dropVel(q)
                   << "\n";
    check(
      out.size() == 1 && q.pos(1) > plo[1] && q.pos(1) < plo[1] + dx &&
        vq[1] > 0. && vq[1] <= u,
      "with subcycles, a particle bounces off the wall within a step", "#733");
    // A small droplet, which the gas carries back to the wall after each
    // bounce
    PType small = p;
    small.rdata(SC::pstateDia) = 5.e-4;
    out = moveParticles(env, s, v, {small}, cfl * dx / u, 40, cfl);
    bool inside = out.size() == 1;
    for (const auto& kv : out) {
      inside = inside && kv.second.pos(1) > plo[1] &&
               kv.second.pos(1) < phi[1] && finiteParticle(kv.second);
    }
    check(inside, "over 40 subcycled steps the particle stays inside");
  }
}

} // namespace spraytest
