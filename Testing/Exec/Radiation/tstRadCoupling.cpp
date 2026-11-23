// Checks of the PeleLMeX coupling layer, PeleLMRad.H: PeleRad::Radiation
// computes kappa and the P1 coefficients from the mole fractions and
// temperature (updateSpecProp), solves for the incident radiation G
// (evaluateRad), and adds the radiative source kappa (G - 4 sigma T^4) to the
// energy equation (calcRadSource). SI units: P in Pa, G in W/m^2, the source
// in W/m^3. The inputs give every MLMGParam key.
// 1. Coefficients: kappa = 100 * getRadPropGas (m^-1) and the emission
//    4 kappa sigma T^4.
// 2. Equilibrium: uniform gas in a periodic domain, G = 4 sigma T^4 and no
//    source.
// 3. Gas varying across the boundaries of a periodic domain: the source
//    integrates to zero (no energy leaves; needs 1/kappa filled across the
//    periodic boundary, #731), and 4 sigma T^4 bounds G between its extremes.
// 4. Uniform gas between cold black walls (Robin BCs, the Marshak condition):
//    the 1D P1 solution G = Eb (1 - C cosh(sqrt(3) kappa x)), with
//    1/C = cosh(a) + 2/sqrt(3) sinh(a), a = sqrt(3) kappa L, for optically
//    thin, intermediate and thick slabs. The gas cools (source < 0), the
//    integrated source equals the analytic wall losses, the source tends to
//    -4 kappa sigma T^4 when thin and G to 4 sigma T^4 inside a thick slab.
//    kappa >= 1 1/m in all cases: the Robin condition uses max(kappa, 1 1/m).
// 5. Second-order convergence of the slab solution with the resolution.
// 6. Two AMR levels, the fine one at a wall: the composite solve matches the
//    slab solution on both levels.
// 7. Tiling: updateSpecProp and calcRadSource over small tiles give the
//    untiled G and source, for gas varying across boxes and walls (#731).
// Checks tagged #731 are expected failures until that PR's fixes land.

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

#include <PeleLMRad.H>

#include "POneCheck.H"

using amrex::Real;

namespace {

using pele::testing::sigma;

pele::testing::Checker check;

std::string
num(Real v)
{
  std::ostringstream s;
  s << std::setprecision(3) << v;
  return s.str();
}

// The slabs: uniform gas between cold walls at x = -L and L. The
// intermediate one is the case of the convergence and AMR checks.
constexpr Real slabT = 1500.0, slabXc = 0.1, slabXh = 0.2;
constexpr Real midL = 0.5, midP = 1.0e5;

Real
slabEb()
{
  return 4.0 * sigma * std::pow(slabT, 4);
}

// The constant of the P1 slab solution G = Eb (1 - C cosh(sqrt(3) kappa x))
// with the Marshak condition at the walls
Real
marshakC(Real kap, Real L)
{
  Real const a = std::sqrt(3.0) * kap * L;
  return 1.0 / (std::cosh(a) + 2.0 / std::sqrt(3.0) * std::sinh(a));
}

// The kpDB-based kappa in m^-1 of a uniform gas, without PeleLMRad.H
Real
kappa_uniform(
  std::string const& kppath, Real xco2, Real xh2o, Real xco, Real T, Real P)
{
  PeleRad::PlanckMean pm(kppath);
  int iL = 0;
  Real w = 0.0;
  PeleRad::RadProp::interpT(T, iL, w);
  Real const kp = xco2 * PeleRad::RadProp::interpk(iL, w, pm.kpco2()) +
                  xh2o * PeleRad::RadProp::interpk(iL, w, pm.kph2o()) +
                  xco * PeleRad::RadProp::interpk(iL, w, pm.kpco());
  return 100.0 * kp * P / 1.0e5;
}

} // namespace

// Mesh hierarchy: x in [-L, L], ny cells in y (periodic), square cells
struct Mesh
{
  amrex::Vector<amrex::Geometry> geom;
  amrex::Vector<amrex::BoxArray> grids;
  amrex::Vector<amrex::DistributionMapping> dmap;

  Mesh(int nx, int ny, Real L, bool periodic_x, int nlev = 1)
  {
    Real const dx = 2.0 * L / nx;
    amrex::RealBox rb(
      {AMREX_D_DECL(-L, 0.0, 0.0)}, {AMREX_D_DECL(L, ny * dx, dx)});
    amrex::Array<int, AMREX_SPACEDIM> is_per{
      AMREX_D_DECL(periodic_x ? 1 : 0, 1, 1)};
    amrex::Box dom(
      amrex::IntVect(0), amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, 0)));
    geom.resize(nlev);
    grids.resize(nlev);
    dmap.resize(nlev);
    for (int lev = 0; lev < nlev; ++lev) {
      geom[lev].define(dom, rb, amrex::CoordSys::cartesian, is_per);
      if (lev == 0) {
        grids[lev].define(dom);
      } else {
        // The lower half in x, at the lo wall
        amrex::Box fine = dom;
        fine.setBig(0, dom.length(0) / 2 - 1);
        grids[lev].define(fine);
      }
      grids[lev].maxSize(32);
      dmap[lev].define(grids[lev]);
      dom.refine(2);
    }
  }
};

// Gas state: mole fractions, T [K] and P [Pa] on each level
struct Gas
{
  amrex::Vector<amrex::MultiFab> xco2, xh2o, xco, T, P;

  explicit Gas(Mesh const& m)
  {
    int const nlev = static_cast<int>(m.geom.size());
    for (auto* v : {&xco2, &xh2o, &xco, &T, &P}) {
      v->resize(nlev);
      for (int lev = 0; lev < nlev; ++lev) {
        (*v)[lev].define(m.grids[lev], m.dmap[lev], 1, 0);
      }
    }
  }

  void uniform(Real c, Real h, Real o, Real t, Real p)
  {
    for (int lev = 0; lev < static_cast<int>(T.size()); ++lev) {
      xco2[lev].setVal(c);
      xh2o[lev].setVal(h);
      xco[lev].setVal(o);
      T[lev].setVal(t);
      P[lev].setVal(p);
    }
  }
};

// The PeleLMRad.H results on each level
struct Rad
{
  amrex::Vector<amrex::MultiFab> G, kappa, emis, src;
};

// The PeleLMeX sequence: updateSpecProp, evaluateRad, calcRadSource into a
// source initialised to src0, the MFIter loops tiled if tile > 0
Rad
solve(
  std::string const& prefix,
  Mesh& m,
  Gas const& gas,
  Real src0 = 0.0,
  int tile = 0)
{
  amrex::MFItInfo info;
  if (tile > 0) {
    info.EnableTiling(amrex::IntVect(tile));
  }
  amrex::ParmParse pp(prefix);
  PeleRad::RadComps rc;
  rc.co2Indx = 0;
  rc.h2oIndx = 1;
  rc.coIndx = 2;
  PeleRad::Radiation rad(m.geom, m.grids, m.dmap, rc, pp);
  int const nlev = static_cast<int>(m.geom.size());
  for (int lev = 0; lev < nlev; ++lev) {
    for (amrex::MFIter mfi(gas.T[lev], info); mfi.isValid(); ++mfi) {
      rad.updateSpecProp(
        mfi, gas.xco2[lev].const_array(mfi), gas.xh2o[lev].const_array(mfi),
        gas.xco[lev].const_array(mfi), gas.T[lev].const_array(mfi),
        gas.P[lev].const_array(mfi), lev);
    }
  }
  rad.evaluateRad();
  Rad r;
  r.G.resize(nlev);
  r.kappa.resize(nlev);
  r.emis.resize(nlev);
  r.src.resize(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    auto const& ba = m.grids[lev];
    auto const& dm = m.dmap[lev];
    for (auto* v : {&r.G, &r.kappa, &r.emis, &r.src}) {
      (*v)[lev].define(ba, dm, 1, 0);
    }
    amrex::MultiFab::Copy(r.G[lev], rad.G()[lev], 0, 0, 1, 0);
    amrex::MultiFab::Copy(r.kappa[lev], rad.kappa()[lev], 0, 0, 1, 0);
    amrex::MultiFab::Copy(r.emis[lev], rad.emis()[lev], 0, 0, 1, 0);
    r.src[lev].setVal(src0);
    for (amrex::MFIter mfi(r.src[lev], info); mfi.isValid(); ++mfi) {
      rad.calcRadSource(mfi, r.src[lev].array(mfi), lev);
    }
  }
  return r;
}

// Relative max difference of a MultiFab from a value
Real
rel_dev(amrex::MultiFab const& mf, Real v)
{
  return std::max(std::abs(mf.max(0) - v), std::abs(mf.min(0) - v)) /
         std::abs(v);
}

void
check_uniform_periodic(std::string const& kppath)
{
  amrex::Print() << "Uniform gas, periodic domain\n";
  Real const T = 1500.0, P = 2.0e5;
  Mesh m(32, 32, 0.1, true);
  Gas gas(m);
  gas.uniform(0.1, 0.2, 0.05, T, P);
  Rad r = solve("eq", m, gas, 1.0);

  Real const kap = kappa_uniform(kppath, 0.1, 0.2, 0.05, T, P);
  Real const Eb = 4.0 * sigma * std::pow(T, 4);
  amrex::Print() << "    kappa = " << kap << " m^-1, 4 sigma T^4 = " << Eb
                 << " W/m^2\n";
  check(
    rel_dev(r.kappa[0], kap) < 1e-13,
    "kappa = 100 P/1e5 sum X_i kp_i(T), in m^-1");
  check(
    rel_dev(r.emis[0], kap * Eb) < 1e-13, "the emission is 4 kappa sigma T^4");
  Real const dG = rel_dev(r.G[0], Eb);
  check(dG < 1e-8, "G = 4 sigma T^4 (relative error " + num(dG) + ")");
  // calcRadSource adds to the source, which was 1
  r.src[0].plus(-1.0, 0, 1);
  Real const ds = r.src[0].norm0() / (kap * Eb);
  check(
    ds < 1e-8,
    "calcRadSource adds a zero source (" + num(ds) + " of the emission)");
}

// Gas varying in x and y, periodic over the domain on every level
void
set_wavy(Mesh const& m, Gas& gas)
{
  for (int lev = 0; lev < static_cast<int>(m.geom.size()); ++lev) {
    Real const lx = m.geom[lev].ProbLength(0);
    Real const ly = m.geom[lev].ProbLength(1);
    auto const dx = m.geom[lev].CellSizeArray();
    for (amrex::MFIter mfi(gas.T[lev]); mfi.isValid(); ++mfi) {
      auto const& c = gas.xco2[lev].array(mfi);
      auto const& h = gas.xh2o[lev].array(mfi);
      auto const& o = gas.xco[lev].array(mfi);
      auto const& T = gas.T[lev].array(mfi);
      auto const& P = gas.P[lev].array(mfi);
      amrex::ParallelFor(
        mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          Real const x = (i + 0.5) * dx[0] / lx;
          Real const y = (j + 0.5) * dx[1] / ly;
          Real const s =
            std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * (y + 0.1));
          T(i, j, k) = 1300.0 + 700.0 * s;
          c(i, j, k) = 0.1 + 0.05 * s;
          h(i, j, k) = 0.12 - 0.04 * s;
          o(i, j, k) = 0.02;
          P(i, j, k) = 1.0e5;
        });
    }
  }
}

void
check_nonuniform_periodic()
{
  amrex::Print() << "Gas varying across the boundaries, periodic domain\n";
  Mesh m(64, 64, 0.5, true);
  Gas gas(m);
  set_wavy(m, gas);
  Rad r = solve("eq", m, gas);

  Real const Ebmin = 4.0 * sigma * std::pow(600.0, 4);
  Real const Ebmax = 4.0 * sigma * std::pow(2000.0, 4);
  check(
    r.G[0].min(0) > Ebmin && r.G[0].max(0) < Ebmax,
    "4 sigma T^4 at the coldest and hottest points bounds G");
  Real const imb = std::abs(r.src[0].sum(0)) / r.src[0].norm1();
  check(
    imb < 1e-8,
    "the source integrates to zero (" + num(imb) +
      " of its integrated magnitude)",
    "#731");
  check(
    r.src[0].min(0) < 0.0 && r.src[0].max(0) > 0.0,
    "the hot gas cools and the cold gas heats");
}

// The slab: G and the source against the 1D P1 solution
struct SlabErr
{
  Real G = 0.0;   // L1(G - Ga) / L1(Ga)
  Real src = 0.0; // L1(S - Sa) / L1(Sa)
};

SlabErr
slab_error(Mesh const& m, Rad const& r, int lev, Real kap, Real L)
{
  Real const Eb = slabEb();
  Real const C = marshakC(kap, L);
  Real const b = std::sqrt(3.0) * kap;
  auto const plo = m.geom[lev].ProbLoArray();
  auto const dx = m.geom[lev].CellSizeArray();
  amrex::MultiFab eG(m.grids[lev], m.dmap[lev], 4, 0);
  for (amrex::MFIter mfi(eG); mfi.isValid(); ++mfi) {
    auto const& e = eG.array(mfi);
    auto const& G = r.G[lev].const_array(mfi);
    auto const& S = r.src[lev].const_array(mfi);
    amrex::ParallelFor(
      mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        Real const x = plo[0] + (i + 0.5) * dx[0];
        Real const Ga = Eb * (1.0 - C * std::cosh(b * x));
        e(i, j, k, 0) = std::abs(G(i, j, k) - Ga);
        e(i, j, k, 1) = Ga;
        e(i, j, k, 2) = std::abs(S(i, j, k) - kap * (Ga - Eb));
        e(i, j, k, 3) = kap * (Eb - Ga);
      });
  }
  return {eG.sum(0) / eG.sum(1), eG.sum(2) / eG.sum(3)};
}

void
check_slab(std::string const& kppath)
{
  Real const Eb = slabEb();
  struct Case
  {
    std::string name;
    Real L, P;
    int nx;
    Real tol;
  };
  for (auto const& c :
       {Case{"thin", 0.002, 1.0e5, 64, 1e-5},
        Case{"intermediate", midL, midP, 64, 3e-4},
        Case{"thick", 0.5, 1.0e6, 256, 2e-3}}) {
    Real const kap = kappa_uniform(kppath, slabXc, slabXh, 0.0, slabT, c.P);
    amrex::Print() << "Slab between cold walls, " << c.name
                   << ": kappa L = " << kap * c.L << "\n";
    Mesh m(c.nx, 8, c.L, false);
    Gas gas(m);
    gas.uniform(slabXc, slabXh, 0.0, slabT, c.P);
    Rad r = solve("slab", m, gas);
    SlabErr e = slab_error(m, r, 0, kap, c.L);
    check(
      e.G < c.tol && e.src < c.tol,
      "G and the source match the P1 solution (L1 errors " + num(e.G) + ", " +
        num(e.src) + ")");
    check(r.src[0].max(0) < 0.0, "the gas cools everywhere");
    check(r.G[0].min(0) > 0.0 && r.G[0].max(0) < Eb, "0 < G < 4 sigma T^4");
    // Energy: the integrated source is minus the losses to the walls
    Real const a = std::sqrt(3.0) * kap * c.L;
    Real const loss =
      2.0 * Eb * marshakC(kap, c.L) * std::sinh(a) / std::sqrt(3.0);
    auto const dx = m.geom[0].CellSizeArray();
    Real const ly = m.geom[0].ProbLength(1);
    Real const integral = r.src[0].sum(0) * dx[0] * dx[1] / ly;
    check(
      std::abs(integral + loss) < c.tol * loss,
      "the integrated source is the wall loss " + num(loss) +
        " W/m^2 (relative error " + num(std::abs(integral + loss) / loss) +
        ")");
    if (c.name == "thin") {
      Real const ratio = -r.src[0].max(0) / (kap * Eb);
      check(
        ratio > 0.99,
        "the source is -4 kappa sigma T^4 within 1% (" + num(ratio) + ")");
    } else if (c.name == "thick") {
      Real const ratio = r.G[0].max(0) / Eb;
      check(
        ratio > 0.999 && ratio < 1.0,
        "G = 4 sigma T^4 inside the slab within 0.1% (1 - G/Eb = " +
          num(1.0 - ratio) + ")");
    }
  }
}

void
check_convergence(std::string const& kppath)
{
  amrex::Print() << "Convergence of the intermediate slab\n";
  Real const kap = kappa_uniform(kppath, slabXc, slabXh, 0.0, slabT, midP);
  Real err[3];
  int n = 16;
  for (auto& e : err) {
    Mesh m(n, 8, midL, false);
    Gas gas(m);
    gas.uniform(slabXc, slabXh, 0.0, slabT, midP);
    Rad r = solve("slab", m, gas);
    e = slab_error(m, r, 0, kap, midL).G;
    amrex::Print() << "    nx = " << n << ": L1 error of G " << e << "\n";
    n *= 2;
  }
  Real const o1 = std::log2(err[0] / err[1]);
  Real const o2 = std::log2(err[1] / err[2]);
  check(
    o1 > 1.8 && o2 > 1.8,
    "second order in dx (orders " + num(o1) + ", " + num(o2) + ")");
}

void
check_amr(std::string const& kppath)
{
  amrex::Print() << "Intermediate slab on two levels, the fine one at a wall\n";
  Real const kap = kappa_uniform(kppath, slabXc, slabXh, 0.0, slabT, midP);
  Mesh m(32, 8, midL, false, 2);
  Gas gas(m);
  gas.uniform(slabXc, slabXh, 0.0, slabT, midP);
  Rad r = solve("slab", m, gas);
  SlabErr e0 = slab_error(m, r, 0, kap, midL);
  SlabErr e1 = slab_error(m, r, 1, kap, midL);
  check(
    e0.G < 5e-4 && e1.G < 5e-4,
    "G matches the P1 solution on both levels (L1 errors " + num(e0.G) + ", " +
      num(e1.G) + ")");
  check(
    r.src[0].max(0) < 0.0 && r.src[1].max(0) < 0.0,
    "the gas cools everywhere on both levels");
}

void
check_tiling()
{
  amrex::Print() << "Tiled loops, gas varying across boxes and walls\n";
  Mesh m(64, 64, 0.5, false);
  Gas gas(m);
  set_wavy(m, gas);
  Rad r0 = solve("slab", m, gas);
  Rad r1 = solve("slab", m, gas, 0.0, 8);
  amrex::MultiFab::Subtract(r1.G[0], r0.G[0], 0, 0, 1, 0);
  amrex::MultiFab::Subtract(r1.src[0], r0.src[0], 0, 0, 1, 0);
  Real const dG = r1.G[0].norm0() / r0.G[0].norm0();
  Real const ds = r1.src[0].norm0() / r0.src[0].norm0();
  check(
    dG < 1e-12 && ds < 1e-12,
    "8x8 tiles give the untiled G and source "
    "(relative differences " +
      num(dG) + ", " + num(ds) + ")",
    "#731");
}

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    std::string kppath = "kpDB";
    amrex::ParmParse("slab").query("kppath", kppath);

    check_uniform_periodic(kppath);
    check_nonuniform_periodic();
    check_slab(kppath);
    check_convergence(kppath);
    check_amr(kppath);
    check_tiling();

    status = check.summary("RadCoupling");
  }
  amrex::Finalize();
  return status;
}
