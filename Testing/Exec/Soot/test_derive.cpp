// Derived plot variables (SootModel_derive) for known moments:
//   soot_largeparticledata  number, mean volume and mean surface of the
//                           second mode (the particles larger than nuclei)
//   soot_genvars            soot mass rho_soot fv and total mass rho + that

#include <AMReX_Geometry.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

#include "SootTest.H"
#include "SootModel_derive.H"

namespace soottest {

void
testDerive(const SootModel& soot, Checker& check)
{
  // soot_largeparticledata uses the nucleus size of the last SootModel
  // defined (after #732): define one from the inputs file
  SootModel sm;
  defineSoot(sm);
  const SootData& sd = *soot.getSootData();
  const SootConst sc;
  const amrex::Real Avna = pele::physics::Constants::Avna;
  const amrex::Real tol = 1.e-12;
  const amrex::Box bx = rowBox(0, 2);
  amrex::FArrayBox dfab(bx, NMOM, amrex::The_Managed_Arena());
  amrex::FArrayBox lfab(bx, 3, amrex::The_Managed_Arena());
  auto const& d = dfab.array();
  auto const& l = lfab.const_array();
  dfab.setVal<amrex::RunOn::Host>(0.);

  // Cell 0: the bimodal state (CGS). Its second mode is two populations of
  // n particles of volume V V0 and surface S S0 (see bimodalMoments)
  Moments m = bimodalMoments(sd);
  const amrex::Real N0_mol = m[NUM_SOOT_MOMENTS];
  sd.convertFromMol(m.data());
  for (int mom = 0; mom < NMOM; ++mom) {
    d(0, 0, 0, mom) = m[mom];
  }
  const amrex::Real n = 5.e-15 * Avna;
  const amrex::Real NL = 2. * n;
  const amrex::Real VL = n * (5.e5 + 2.e6) * sc.V0 / NL;
  const amrex::Real SL =
    n * (3. * std::pow(5.e5, 2. / 3.) + 4. * std::pow(2.e6, 2. / 3.)) * sc.S0 /
    NL;
  // Cell 1: N0 nuclei plus N_L particles of 50 nuclei
  const amrex::Real V_nucl = sd.nuclVol * sc.V0;
  const amrex::Real S_nucl = sd.nuclSurf * sc.S0;
  const amrex::Real N0 = 1.e8;
  const amrex::Real N_L = 1.e6;
  const amrex::Real V_L = 50. * V_nucl;
  const amrex::Real S_L = std::pow(50., 2. / 3.) * S_nucl;
  d(1, 0, 0, 0) = N0 + N_L;
  d(1, 0, 0, 1) = N0 * V_nucl + N_L * V_L;
  d(1, 0, 0, 2) = N0 * S_nucl + N_L * S_L;
  d(1, 0, 0, NUM_SOOT_MOMENTS) = N0;
  // Cell 2: nuclei and less than one large particle per cm^3
  d(2, 0, 0, 0) = N0 + 0.5;
  d(2, 0, 0, 1) = (N0 + 0.5) * V_nucl;
  d(2, 0, 0, 2) = (N0 + 0.5) * S_nucl;
  d(2, 0, 0, NUM_SOOT_MOMENTS) = N0;

  soot_largeparticledata(
    bx, lfab, 0, 3, dfab, amrex::Geometry(), 0., nullptr, 0);
  amrex::Gpu::streamSynchronize();
  amrex::Print() << "Large particles of the bimodal state: N_L = "
                 << l(0, 0, 0, 0) << " /cm^3, d_V = "
                 << std::cbrt(6. * l(0, 0, 0, 1) / M_PI) * 1.e7
                 << " nm (nuclei: " << N0_mol * Avna << " /cm^3)\n";
  check(
    relErr(l(0, 0, 0, 0), NL) <= tol && relErr(l(0, 0, 0, 1), VL) <= tol &&
      relErr(l(0, 0, 0, 2), SL) <= tol,
    "large-particle derive of the bimodal state: N_L, mean V and S", "#732");
  check(
    relErr(l(1, 0, 0, 0), N_L) <= tol && relErr(l(1, 0, 0, 1), V_L) <= tol &&
      relErr(l(1, 0, 0, 2), S_L) <= tol,
    "large-particle derive subtracts the nuclei", "#732");
  check(
    relErr(l(2, 0, 0, 0), 0.5) <= 1.e-6 && l(2, 0, 0, 1) == 0. &&
      l(2, 0, 0, 2) == 0.,
    "large-particle derive: no mean size below one particle per cm^3");

  // soot_genvars reads density (component 0) and fv (component 2)
  amrex::FArrayBox gfab(bx, 3, amrex::The_Managed_Arena());
  amrex::FArrayBox ofab(bx, 2, amrex::The_Managed_Arena());
  auto const& g = gfab.array();
  auto const& o = ofab.const_array();
  const amrex::Real rho[3] = {1.e-4, 2.e-4, 3.e-4};
  const amrex::Real fv[3] = {0., 1.e-7, 3.e-6};
  for (int i = 0; i < 3; ++i) {
    g(i, 0, 0, 0) = rho[i];
    g(i, 0, 0, 1) = std::numeric_limits<amrex::Real>::quiet_NaN();
    g(i, 0, 0, 2) = fv[i];
  }
  soot_genvars(bx, ofab, 0, 2, gfab, amrex::Geometry(), 0., nullptr, 0);
  amrex::Gpu::streamSynchronize();
  bool gen = true;
  for (int i = 0; i < 3; ++i) {
    const amrex::Real ms = sc.SootDensity * fv[i];
    gen = gen && relErr(o(i, 0, 0, 0), ms) <= tol &&
          relErr(o(i, 0, 0, 1), rho[i] + ms) <= tol;
  }
  check(gen, "soot mass = rho_soot fv, total mass = rho + soot mass");
  sm.cleanup();
}

} // namespace soottest
