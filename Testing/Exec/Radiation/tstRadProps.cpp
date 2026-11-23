// Checks of the radiative properties: PlanckMean.H and SpectralModels.H.
// 1. The kpDB tables: PlanckMean reads every entry of the six files, and the
//    values are positive and finite.
// 2. Units: the gas tables are Planck means per unit partial pressure in
//    cm^-1 atm^-1 or bar^-1 (within a factor of 3 of the TNF fits for H2O and
//    CO2), and the soot table is kp/fv in cm^-1, roughly proportional to T.
// 3. interpT/interpk: the table values at the nodes, values between the
//    neighbours elsewhere, continuity at the nodes, and clamping outside
//    [300 K, 2800 K], including T = 2800 K exactly, and a NaN T gives a NaN
//    kp (both #731: checks tagged #731 are expected failures until it lands).
// 4. getRadPropGas: kappa = P/1e5 * sum_i X_i kp_i(T) in cm^-1, with X_i the
//    mole fractions, so it is linear in each species' X and in P, the species
//    contributions add up, and with no radiating species it is the 0.001
//    floor. The PeleC
//    convention (P in dyn/cm^2, kappa scaled by 0.1) gives the same kappa as
//    the PeleLMeX one (P in Pa).
// 5. getRadPropSoot: adds fv * kp_soot(T) to kappa, linear in fv.

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <PlanckMean.H>
#include <SpectralModels.H>

#include "TestCheck.H"

namespace {

using amrex::Real;
// The tables: kp at nT temperatures from 300 K to 2800 K
constexpr int nT = 126;
using Table = amrex::GpuArray<Real, nT>;

Real
Tnode(int n)
{
  return 300.0 + 20.0 * n;
}

pele::testing::Checker check;

bool
close(Real a, Real b, Real rtol)
{
  return std::abs(a - b) <= rtol * std::max(std::abs(a), std::abs(b));
}

Real
interp(Real T, Table const& k)
{
  int iL = 0;
  Real w = 0.0;
  PeleRad::RadProp::interpT(T, iL, w);
  return PeleRad::RadProp::interpk(iL, w, k);
}

// TNF workshop Planck-mean fits, converted from atm^-1 m^-1 to atm^-1 cm^-1
Real
tnf_h2o(Real T)
{
  Real const x = 1000.0 / T;
  Real const kp =
    -0.23093 +
    x * (-1.12390 +
         x * (9.41530 + x * (-2.99880 + x * (0.51382 - x * 1.86840e-5))));
  return kp / 100.0;
}

Real
tnf_co2(Real T)
{
  Real const x = 1000.0 / T;
  Real const kp =
    18.741 +
    x * (-121.310 + x * (273.500 + x * (-194.050 + x * (56.310 - x * 5.8169))));
  return kp / 100.0;
}

// A one-cell Array4 view of a scalar
template <typename T>
amrex::Array4<T>
view(T& v)
{
  return amrex::Array4<T>(&v, amrex::Dim3{0, 0, 0}, amrex::Dim3{1, 1, 1}, 1);
}

// The six tables, by name
std::vector<std::pair<std::string, Table const*>>
tables(PeleRad::PlanckMean const& pm)
{
  return {{"co2", &pm.kpco2()}, {"h2o", &pm.kph2o()},   {"co", &pm.kpco()},
          {"ch4", &pm.kpch4()}, {"c2h4", &pm.kpc2h4()}, {"soot", &pm.kpsoot()}};
}

// kappa [cm^-1] of one cell from getRadPropGas, then getRadPropSoot if fv > 0
Real
kappa(
  PeleRad::PlanckMean const& pm,
  Real xco2,
  Real xh2o,
  Real xco,
  Real T,
  Real P,
  Real fv = 0.0)
{
  Real k = -1.0;
  PeleRad::RadProp::getRadPropGas(
    0, 0, 0, view<Real const>(xco2), view<Real const>(xh2o),
    view<Real const>(xco), view<Real const>(T), view<Real const>(P), view(k),
    pm.kpco2(), pm.kph2o(), pm.kpco());
  if (fv > 0.0) {
    PeleRad::RadProp::getRadPropSoot(
      0, 0, 0, view<Real const>(fv), view<Real const>(T), view(k), pm.kpsoot());
  }
  return k;
}

void
check_tables(PeleRad::PlanckMean const& pm, std::string const& path)
{
  amrex::Print() << "kpDB tables\n";
  for (auto const& [name, tab] : tables(pm)) {
    // Read the file independently of PlanckMean
    std::ifstream f(path + "/kpl_" + name + ".dat");
    std::vector<Real> T, k;
    for (Real a, b; f >> a >> b;) {
      T.push_back(a);
      k.push_back(b);
    }
    bool same = static_cast<int>(T.size()) == nT;
    bool positive = true;
    for (int n = 0; n < static_cast<int>(T.size()) && n < nT; ++n) {
      same = same && T[n] == Tnode(n) && (*tab)[n] == k[n];
      positive = positive && std::isfinite((*tab)[n]) && (*tab)[n] > 0.0;
    }
    check(same, name + ": 126 entries from 300 K to 2800 K, read exactly");
    check(positive, name + ": kp is positive and finite");
  }

  // Units. The TNF fits are not exact, but kp in m^-1 or bar^-1 Pa would be
  // off by orders of magnitude.
  bool gas_units = true;
  for (Real T = 500.0; T <= 2000.0; T += 100.0) {
    Real const rh = interp(T, pm.kph2o()) / tnf_h2o(T);
    Real const rc = interp(T, pm.kpco2()) / tnf_co2(T);
    gas_units =
      gas_units && rh > 1.0 / 3.0 && rh < 3.0 && rc > 1.0 / 3.0 && rc < 3.0;
  }
  amrex::Print() << "    kp(1000 K) [cm^-1 atm^-1]: H2O "
                 << interp(1000.0, pm.kph2o()) << " (TNF " << tnf_h2o(1000.0)
                 << "), CO2 " << interp(1000.0, pm.kpco2()) << " (TNF "
                 << tnf_co2(1000.0) << ")\n";
  check(
    gas_units, "H2O and CO2 kp within a factor of 3 of the TNF fits in cm^-1 "
               "atm^-1, 500-2000 K");
  // Soot: kp/fv ~ C T, with C ~ 1200 m^-1 K^-1 (Kent & Honnery)
  Real cmin = 1e30, cmax = 0.0;
  for (int n = 0; n < nT; ++n) {
    Real const c = pm.kpsoot()[n] / Tnode(n);
    cmin = std::min(cmin, c);
    cmax = std::max(cmax, c);
  }
  amrex::Print() << "    soot kp/(fv T) [cm^-1 K^-1]: " << cmin << " to "
                 << cmax << "\n";
  check(
    cmin > 8.0 && cmax < 16.0,
    "soot kp/(fv T) is within 8-16 cm^-1 K^-1 over the table");
}

void
check_interp(PeleRad::PlanckMean const& pm)
{
  amrex::Print() << "Interpolation in T\n";
  bool nodes = true, between = true, midpoint = true, continuous = true;
  for (auto const& [name, tab] : tables(pm)) {
    auto const& k = *tab;
    // The last node is checked below through the clamp
    for (int n = 0; n < nT - 1; ++n) {
      Real const T = Tnode(n);
      int iL = -1;
      Real w = -1.0;
      PeleRad::RadProp::interpT(T, iL, w);
      nodes = nodes && iL == n && w == 0.0 && interp(T, k) == k[n];
      for (Real f : {0.1, 0.37, 0.5, 0.9}) {
        Real const v = interp(T + 20.0 * f, k);
        between = between && v >= std::min(k[n], k[n + 1]) &&
                  v <= std::max(k[n], k[n + 1]);
      }
      midpoint =
        midpoint && close(interp(T + 10.0, k), 0.5 * (k[n] + k[n + 1]), 1e-12);
      if (n > 0) {
        continuous = continuous && close(interp(T - 1e-9, k), k[n], 1e-9) &&
                     close(interp(T + 1e-9, k), k[n], 1e-9);
      }
    }
  }
  check(nodes, "interpT/interpk reproduce the table at the nodes");
  check(between, "interpolated values lie between the neighbouring nodes");
  check(midpoint, "the interpolation is linear (midpoint = mean)");
  check(continuous, "the interpolation is continuous at the nodes");

  bool clamped = true;
  for (auto const& [name, tab] : tables(pm)) {
    auto const& k = *tab;
    clamped = clamped && interp(200.0, k) == k[0] && interp(299.0, k) == k[0] &&
              interp(2801.0, k) == k[nT - 1] &&
              interp(4000.0, k) == k[nT - 1] &&
              close(interp(2800.0 - 1e-9, k), k[nT - 1], 1e-9);
  }
  check(clamped, "T is clamped to [300 K, 2800 K]");

  // The interval index is checked before reading the table, so that a bad
  // index does not read outside it.
  int iL = -1;
  Real w = -1.0;
  PeleRad::RadProp::interpT(2800.0, iL, w);
  bool exact = iL >= 0 && iL <= nT - 2;
  for (auto const& [name, tab] : tables(pm)) {
    exact = exact && PeleRad::RadProp::interpk(iL, w, *tab) == (*tab)[nT - 1];
  }
  check(exact, "T = 2800 K gives the last table entry", "#731");

  // A NaN known only at run time: a compile-time NaN lets the compiler fold
  // the undefined NaN-to-int conversion before #731 into anything.
  volatile Real const nan_v = std::numeric_limits<Real>::quiet_NaN();
  Real const nan = nan_v;
  iL = -1;
  PeleRad::RadProp::interpT(nan, iL, w);
  check(
    iL >= 0 && iL <= nT - 2 &&
      std::isnan(PeleRad::RadProp::interpk(iL, w, pm.kph2o())),
    "a NaN temperature gives a NaN kp", "#731");
}

void
check_gas(PeleRad::PlanckMean const& p)
{
  amrex::Print() << "getRadPropGas\n";
  Real const T = 1234.0, P = 1.0e5;
  Real const xc = 0.08, xh = 0.15, xo = 0.03;
  Real const k = kappa(p, xc, xh, xo, T, P);
  Real const kc = interp(T, p.kpco2()), kh = interp(T, p.kph2o()),
             ko = interp(T, p.kpco());
  amrex::Print() << "    kappa(X_CO2=0.08, X_H2O=0.15, X_CO=0.03, 1234 K, 1 "
                    "bar) = "
                 << k << " cm^-1\n";
  check(
    close(k, P / 1.0e5 * (xc * kc + xh * kh + xo * ko), 1e-13),
    "kappa = P/1e5 * sum X_i kp_i(T) (P in Pa, kappa in cm^-1)");

  // Linear in each species and additive
  bool const linear =
    close(kappa(p, 2 * xc, xh, xo, T, P) - k, P / 1.0e5 * xc * kc, 1e-12) &&
    close(kappa(p, xc, 2 * xh, xo, T, P) - k, P / 1.0e5 * xh * kh, 1e-12) &&
    close(kappa(p, xc, xh, 2 * xo, T, P) - k, P / 1.0e5 * xo * ko, 1e-12);
  check(linear, "doubling one species' X doubles its contribution");
  Real const sum = kappa(p, xc, 0, 0, T, P) + kappa(p, 0, xh, 0, T, P) +
                   kappa(p, 0, 0, xo, T, P);
  check(close(sum, k, 1e-13), "the species contributions add up");
  check(
    close(kappa(p, 2 * xc, 2 * xh, 2 * xo, T, P), 2 * k, 1e-13) &&
      close(kappa(p, xc, xh, xo, T, 3 * P), 3 * k, 1e-13),
    "kappa is linear in the composition and in P");
  check(
    kappa(p, 0, 0, 0, T, P) == 0.001 &&
      kappa(p, 9e-6, 9e-6, 9e-6, 2000.0, 1e7) == 0.001,
    "with no radiating species kappa is the 0.001 cm^-1 floor");
  // PeleCRad passes P in dyn/cm^2 and scales kappa by 0.1
  check(
    close(0.1 * kappa(p, xc, xh, xo, T, 10.0 * P), k, 1e-13),
    "the PeleC (CGS) pressure convention gives the same kappa");
}

void
check_soot(PeleRad::PlanckMean const& p)
{
  amrex::Print() << "getRadPropSoot\n";
  Real const T = 1600.0, P = 1.0e5, fv = 2.0e-6;
  Real const kg = kappa(p, 0.1, 0.1, 0.0, T, P);
  Real const k1 = kappa(p, 0.1, 0.1, 0.0, T, P, fv) - kg;
  Real const k2 = kappa(p, 0.1, 0.1, 0.0, T, P, 2 * fv) - kg;
  amrex::Print() << "    soot kappa(fv=2e-6, 1600 K) = " << k1
                 << " cm^-1, kappa/(fv T) = " << k1 / (fv * T) << "\n";
  check(
    close(k1, fv * interp(T, p.kpsoot()), 1e-12),
    "the soot kappa fv kp_soot(T) is added to the gas kappa");
  check(close(k2, 2 * k1, 1e-12), "the soot kappa is linear in fv");
  Real const r = (kappa(p, 0, 0, 0, 2 * T / 2.5, P, fv) - 0.001) /
                 (kappa(p, 0, 0, 0, T / 2.5, P, fv) - 0.001);
  check(
    r > 1.6 && r < 2.4, "the soot kappa roughly doubles when T doubles (" +
                          std::to_string(r) + ")");
}

} // namespace

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    amrex::ParmParse pp("pelerad");
    std::string path = "kpDB";
    pp.query("kppath", path);

    PeleRad::PlanckMean pm(path);

    check_tables(pm, path);
    check_interp(pm);
    check_gas(pm);
    check_soot(pm);

    status = check.summary("RadProps");
  }
  amrex::Finalize();
  return status;
}
