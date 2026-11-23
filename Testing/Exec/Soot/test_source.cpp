// SootModel::computeSootSourceTerm with the inputs-file options, on a box of
// fresh (nuclei only) and well-developed soot states from 1400 K to 2000 K:
// the source adds into its destination, is finite, balances mass (with
// soot.conserve_mass) and energy, forms soot, is measured from the clipped
// state, and is zero below soot.temp_cutoff

#include <AMReX_Print.H>

#include <cmath>

#include <PelePhysics.H>
#include "SootTest.H"

namespace soottest {

void
testSource(const SootModel& soot, Checker& check)
{
  const amrex::Real dt = 1.e-4;
  const amrex::Real tol = 1.e-10;
  const amrex::Box bx = stateBox();
  Fabs f(bx);
  amrex::FArrayBox s1fab(bx, nS, amrex::The_Managed_Arena());
  fillStates(*soot.getSootData(), f.q);
  auto const& q = f.q.const_array();
  auto const& s = f.s.const_array();
  auto const& s1 = s1fab.const_array();

  // The source adds into its destination
  sourceOn(soot, f.q, f.mu, f.s, dt);
  s1fab.copy<amrex::RunOn::Host>(f.s);
  sourceOn(soot, f.q, f.mu, f.s, dt, true, false);
  amrex::Real err_add = 0.;
  amrex::Real ref_add = 0.;
  amrex::LoopOnCpu(bx, nS, [&](int i, int j, int k, int c) {
    err_add =
      amrex::max(err_add, std::abs(s(i, j, k, c) - 2. * s1(i, j, k, c)));
    ref_add = amrex::max(ref_add, std::abs(s1(i, j, k, c)));
  });
  check(
    err_add <= tol * ref_add, "the source adds into its destination", "#732");

  // Finite; mass balance; species sum; soot forms in the fresh cells
  sourceOn(soot, f.q, f.mu, f.s, dt);
  check(
    !f.s.contains_nan<amrex::RunOn::Host>() &&
      !f.s.contains_inf<amrex::RunOn::Host>(),
    "the sources are finite");
  const auto [err_soot, ref_soot] = massBalance(bx, s);
  amrex::Real err_spec = 0.;
  amrex::Real ref_spec = 0.;
  bool forms = true;
  amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
    const amrex::Real srho = s(i, j, k, sRho);
    amrex::Real sspec = 0.;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      sspec += s(i, j, k, sSpec + n);
    }
    err_spec = amrex::max(err_spec, std::abs(sspec - srho));
    ref_spec = amrex::max(ref_spec, std::abs(srho));
    if (j == 0) {
      forms = forms && s(i, j, k, sSoot + 1) > 0. &&
              s(i, j, k, sSoot + NUM_SOOT_MOMENTS) > 0.;
    }
  });
  amrex::Print() << "Mass balance: max |S_rho + rho_soot S_fv| = " << err_soot
                 << " (max |rho_soot S_fv| = " << ref_soot
                 << "), max |sum S_k - S_rho| = " << err_spec << "\n";
  check(
    err_soot <= tol * ref_soot,
    "conserve_mass: the gas density source balances the soot mass source",
    "#732");
  check(err_spec <= tol * ref_spec, "the species sources sum to S_rho");
  check(forms, "fresh states nucleate soot (S_fv > 0, S_N0 > 0)");

  // Energy: without the pressure term S_E = sum h_k S_k, and the pressure
  // term is -RT sum S_k / W_k
  sourceOn(soot, f.q, f.mu, s1fab, dt, false);
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real mw[NUM_SPECIES];
  eos.molecular_weight(mw);
  amrex::Real err_eng = 0.;
  amrex::Real ref_eng = 0.;
  amrex::Real err_p = 0.;
  amrex::Real ref_p = 0.;
  amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
    const amrex::Real T = q(i, j, k, qTemp);
    amrex::Real h[NUM_SPECIES];
    eos.T2Hi(T, h);
    amrex::Real sum_h = 0.;
    amrex::Real abs_h = 0.;
    amrex::Real sum_p = 0.;
    amrex::Real abs_p = 0.;
    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
      const amrex::Real Sk = s1(i, j, k, sSpec + sp);
      sum_h += Sk * h[sp];
      abs_h += std::abs(Sk * h[sp]);
      const amrex::Real pk =
        -pele::physics::Constants::RU * T * s(i, j, k, sSpec + sp) / mw[sp];
      sum_p += pk;
      abs_p += std::abs(pk);
    }
    err_eng = amrex::max(err_eng, std::abs(s1(i, j, k, sEng) - sum_h));
    ref_eng = amrex::max(ref_eng, abs_h);
    err_p =
      amrex::max(err_p, std::abs(s(i, j, k, sEng) - s1(i, j, k, sEng) - sum_p));
    ref_p = amrex::max(ref_p, abs_p);
  });
  amrex::Print() << "Energy: max |S_E - sum h_k S_k| = " << err_eng
                 << " (scale " << ref_eng
                 << "), pressure term error = " << err_p << " (scale " << ref_p
                 << ")\n";
  check(
    std::isfinite(err_eng) && err_eng <= tol * ref_eng,
    "the energy source is sum h_k S_k");
  check(
    std::isfinite(err_p) && err_p <= tol * ref_p,
    "the pressure term is -RT sum S_k / W_k");

  // Clipping the incoming state is not a soot source: unrealizable moments
  // (ten times more particles than nuclei of all the volume) and the same
  // moments clipped give the same source
  const amrex::Box bx2 = rowBox(0, 1);
  amrex::FArrayBox q2fab(bx2, nQ, amrex::The_Managed_Arena());
  amrex::FArrayBox s2fab(bx2, nS, amrex::The_Managed_Arena());
  auto const& q2 = q2fab.array();
  auto const& s2 = s2fab.const_array();
  const auto Y = sootingMixture();
  const SootData* sd_d = soot.getSootData_d();
  const Moments m_mol = bimodalMoments(*soot.getSootData());
  // Gas state; the moments are set on the device below
  setCell(q2, amrex::IntVect(AMREX_D_DECL(0, 0, 0)), 1800., Y.data(), m_mol);
  setCell(q2, amrex::IntVect(AMREX_D_DECL(1, 0, 0)), 1800., Y.data(), m_mol);
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) noexcept {
    Moments m = m_mol;
    m[0] = 10. * m[1] / sd_d->nuclVol;
    Moments mc = m;
    sd_d->clipMoments(mc.data());
    sd_d->convertFromMol(m.data());
    sd_d->convertFromMol(mc.data());
    for (int mom = 0; mom < NMOM; ++mom) {
      q2(0, 0, 0, qSoot + mom) = m[mom];
      q2(1, 0, 0, qSoot + mom) = mc[mom];
    }
  });
  amrex::Gpu::streamSynchronize();
  sourceOn(soot, q2fab, f.mu, s2fab, dt);
  amrex::Real err_clip = 0.;
  amrex::Real ref_clip = 0.;
  for (int c = 0; c < nS; ++c) {
    err_clip = amrex::max(err_clip, std::abs(s2(0, 0, 0, c) - s2(1, 0, 0, c)));
    ref_clip = amrex::max(ref_clip, std::abs(s2(1, 0, 0, c)));
  }
  check(
    err_clip <= tol * ref_clip,
    "the source is measured from the clipped incoming moments", "#732");

  // No source below soot.temp_cutoff (290 K in the inputs)
  q2(0, 0, 0, qTemp) = 280.;
  sourceOn(soot, q2fab, f.mu, s2fab, dt);
  bool cold_zero = true;
  for (int c = 0; c < nS; ++c) {
    cold_zero = cold_zero && s2(0, 0, 0, c) == 0.;
  }
  check(
    cold_zero && s2(1, 0, 0, sSoot + 1) != 0.,
    "no source below soot.temp_cutoff");

  // Names (gasSpeciesName is not const)
  auto& sm = const_cast<SootModel&>(soot); // NOLINT
  bool names = sm.gasSpeciesName(SootGasSpecIndx::indxPAH) == "A2" &&
               sm.gasSpeciesName(SootGasSpecIndx::indxC2H2) == "C2H2" &&
               soot.sootVariableName(0) == "soot_N" &&
               soot.sootVariableName(1) == "soot_fv" &&
               soot.sootVariableName(2) == "soot_S" &&
               soot.sootVariableName(NUM_SOOT_MOMENTS) == "soot_N0";
  check(names, "gas species and soot variable names");
}

} // namespace soottest
