// Soot model options, each on a SootModel re-defined in the same run with
// ParmParse overrides:
//   soot.conserve_mass = false   only the H2, density and energy sources
//                                change, by the soot mass imbalance
//   soot.incept_pah = A2, A3, A4 (soot.pah_name = A2: the mechanism has
//                                only A2) set the sticking coefficient and
//                                nucleus size, and nucleate only with PAH
//   soot.num_subcycles,          one subcycle is one explicit step of the
//   soot.max_subcycles           source terms; on a step of estSootDt
//                                more subcycles converge at first order and
//                                species stay non-negative

#include <AMReX_Print.H>

#include <cmath>
#include <vector>

#include <PelePhysics.H>
#include "SootTest.H"

namespace soottest {

namespace {

void
checkConserveMass(const SootModel& soot, Checker& check)
{
  const SootConst sc;
  const amrex::Real dt = 1.e-4;
  SootModel sf;
  defineSoot(sf, {.conserve_mass = false});
  const amrex::Box bx = stateBox();
  Fabs f(bx);
  amrex::FArrayBox sffab(bx, nS, amrex::The_Managed_Arena());
  fillStates(*soot.getSootData(), f.q);
  sourceOn(soot, f.q, f.mu, f.s, dt);
  sourceOn(sf, f.q, f.mu, sffab, dt);
  auto const& q = f.q.const_array();
  auto const& st = f.s.const_array();
  auto const& sfa = sffab.const_array();
  // conserve_mass moves the imbalance into H2
  const auto [imbalance, ref_soot] = massBalance(bx, sfa);

  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real mw[NUM_SPECIES];
  eos.molecular_weight(mw);
  const amrex::Real tol = 1.e-10;
  amrex::Real err_spec = 0.;
  amrex::Real ref_rho = 0.;
  amrex::Real err_same = 0.;
  amrex::Real ref_same = 0.;
  amrex::Real err_shift = 0.;
  amrex::Real err_eng = 0.;
  amrex::Real ref_eng = 0.;
  amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
    amrex::Real sspec = 0.;
    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
      sspec += sfa(i, j, k, sSpec + sp);
    }
    err_spec = amrex::max(err_spec, std::abs(sspec - sfa(i, j, k, sRho)));
    ref_rho = amrex::max(ref_rho, std::abs(sfa(i, j, k, sRho)));
    // The soot and all other species sources are the same
    for (int c = sSpec; c < nS; ++c) {
      ref_same = amrex::max(ref_same, std::abs(st(i, j, k, c)));
      if (c != sSpec + H2_ID) {
        err_same =
          amrex::max(err_same, std::abs(st(i, j, k, c) - sfa(i, j, k, c)));
      }
    }
    const amrex::Real del =
      sfa(i, j, k, sRho) + sfa(i, j, k, sSoot + 1) * sc.SootDensity;
    err_shift = amrex::max(
      err_shift,
      amrex::max(
        std::abs(st(i, j, k, sRho) - sfa(i, j, k, sRho) + del),
        std::abs(
          st(i, j, k, sSpec + H2_ID) - sfa(i, j, k, sSpec + H2_ID) + del)));
    const amrex::Real T = q(i, j, k, qTemp);
    amrex::Real h[NUM_SPECIES];
    eos.T2Hi(T, h);
    const amrex::Real de =
      -del * (h[H2_ID] - pele::physics::Constants::RU * T / mw[H2_ID]);
    err_eng = amrex::max(
      err_eng, std::abs(st(i, j, k, sEng) - sfa(i, j, k, sEng) - de));
    ref_eng = amrex::max(ref_eng, std::abs(de));
  });
  amrex::Print() << "conserve_mass = false: max |S_rho + rho_soot S_fv| = "
                 << imbalance << " (max |rho_soot S_fv| = " << ref_soot
                 << ")\n";
  check(
    err_spec <= tol * ref_rho,
    "conserve_mass = false: the species sources sum to S_rho");
  check(
    imbalance > 1.e-3 * ref_soot,
    "conserve_mass = false: gas mass lost != soot mass gained (H is lost)");
  check(
    err_same <= 1.e-14 * ref_same,
    "conserve_mass only changes the H2 source among species and moments");
  check(
    err_shift <= tol * imbalance && err_eng <= tol * ref_eng,
    "conserve_mass adds the imbalance to H2, rho and energy", "#732");
  sf.cleanup();
}

void
checkPAH(const SootModel& soot, Checker& check)
{
  const amrex::Real dt = 1.e-4;
  const auto Y = sootingMixture();
  auto Y0 = Y;
  Y0[N2_ID] += Y0[A2_ID];
  Y0[A2_ID] = 0.;
  Fabs f(rowBox(0, 1));
  auto const& q = f.q.array();
  auto const& s = f.s.const_array();
  const amrex::IntVect c0(AMREX_D_DECL(0, 0, 0));
  const amrex::IntVect c1(AMREX_D_DECL(1, 0, 0));

  const std::string names[] = {"A2", "A3", "A4"};
  const amrex::Real gamma[] = {0.002, 0.015, 0.025};
  const amrex::Real numC[] = {10., 14., 16.};
  const amrex::Real A_ref = soot.m_sootReact->A_f[NUM_SOOT_REACT - 1];
  for (int p = 0; p < 3; ++p) {
    SootModel sp;
    defineSoot(sp, {.incept_pah = names[p]});
    const SootData& sd = *sp.getSootData();
    // Sticking coefficient and nucleus (two dimers of two PAH) size; the
    // dimerization prefactor scales with gamma (numC)^(1/6)
    const amrex::Real A_ratio = sp.m_sootReact->A_f[NUM_SOOT_REACT - 1] / A_ref;
    const amrex::Real A_expect =
      gamma[p] / gamma[0] * std::pow(numC[p] / numC[0], 1. / 6.);
    const bool params = sp.m_PAHindx == A2_ID && sp.m_gammaStick == gamma[p] &&
                        sd.nuclVol == 4. * numC[p] &&
                        relErr(A_ratio, A_expect) <= 1.e-14;
    // Fresh soot with and without the PAH
    const Moments fresh = freshMoments(sd);
    setCell(q, c0, 1800., Y.data(), fresh);
    setCell(q, c1, 1800., Y0.data(), fresh);
    sourceOn(sp, f.q, f.mu, f.s, dt);
    const bool nucleates = s(c0, sSoot + NUM_SOOT_MOMENTS) > 0. &&
                           s(c0, sSoot + 1) > 0. && s(c0, sSpec + A2_ID) < 0.;
    const bool none = s(c1, sSpec + A2_ID) == 0. && s(c1, sSoot + 1) == 0. &&
                      s(c1, sSoot + NUM_SOOT_MOMENTS) <= 0.;
    amrex::Print() << "incept_pah = " << names[p]
                   << ": nucleus volume = " << sd.nuclVol
                   << " C, S_N0 = " << s(c0, sSoot + NUM_SOOT_MOMENTS)
                   << ", without PAH S_N0 = " << s(c1, sSoot + NUM_SOOT_MOMENTS)
                   << "\n";
    check(params, "incept_pah = " + names[p] + ": gamma, nucleus size, rate");
    check(nucleates, "incept_pah = " + names[p] + ": PAH nucleates soot");
    check(none, "incept_pah = " + names[p] + ": no inception without PAH");
    sp.cleanup();
  }
}

// One subcycle is one explicit step of computeSrcTerms from the clipped
// state, followed by a clip
void
checkOneSubcycle(Checker& check)
{
  const amrex::Real dt = 1.e-4;
  SootModel s1;
  defineSoot(
    s1, {.conserve_mass = false, .num_subcycles = 1, .max_subcycles = 1});
  const amrex::Box bx = cellBox(0);
  Fabs f(bx);
  amrex::FArrayBox efab(bx, nS, amrex::The_Managed_Arena());
  efab.setVal<amrex::RunOn::Host>(0.);
  Moments m = bimodalMoments(*s1.getSootData());
  s1.getSootData()->convertFromMol(m.data());
  const auto Y = sootingMixture();
  setCell(f.q.array(), bx.smallEnd(), 1800., Y.data(), m);
  sourceOn(s1, f.q, f.mu, f.s, dt);

  const SootData* sd = s1.getSootData_d();
  const SootReaction* sr = s1.d_sootReact;
  const amrex::Real betaNF = s1.m_betaNuclFact;
  auto const& q = f.q.const_array();
  auto const& e = efab.array();
  const SootConst sc;
  // A copy: MU is passed by reference, so the device needs its own
  const amrex::Real mu = MU;
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real mw[NUM_SPECIES];
    eos.molecular_weight(mw);
    const amrex::Real rho = q(i, j, k, qRho);
    const amrex::Real T = q(i, j, k, qTemp);
    amrex::Real rhoY[NUM_SPECIES];
    amrex::Real inv_mm = 0.;
    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
      rhoY[sp] = rho * q(i, j, k, qSpec + sp);
      inv_mm += rhoY[sp] / mw[sp];
    }
    amrex::Real xi[NUM_SOOT_GS];
    amrex::Real om[NUM_SOOT_GS] = {0.};
    for (int g = 0; g < NUM_SOOT_GS; ++g) {
      xi[g] = rhoY[sd->refIndx[g]] / mw[sd->refIndx[g]];
    }
    Moments m0;
    amrex::Real src[NMOM] = {0.};
    amrex::Real fv[NMOM + 1];
    for (int mom = 0; mom < NMOM; ++mom) {
      m0[mom] = q(i, j, k, qSoot + mom);
    }
    sd->convertToMol(m0.data());
    sd->clipMoments(m0.data());
    Moments m1 = m0;
    const amrex::Real convT = std::sqrt(sc.colFact * T);
    const amrex::Real colConst =
      convT * sc.colFactPi23 * sc.colFact16 * pele::physics::Constants::Avna;
    sd->computeSrcTerms(
      T, mu, rho, rho / inv_mm, convT, convT * betaNF, colConst, xi, om,
      m1.data(), src, fv, sr);
    for (int mom = 0; mom < NMOM; ++mom) {
      m1[mom] += dt * src[mom];
    }
    sd->clipMoments(m1.data());
    for (int mom = 0; mom < NMOM; ++mom) {
      e(i, j, k, sSoot + mom) = (m1[mom] - m0[mom]) * sd->unitConv[mom] / dt;
    }
    for (int g = 0; g < NUM_SOOT_GS; ++g) {
      const int sp = sd->refIndx[g];
      e(i, j, k, sSpec + sp) = ((xi[g] + dt * om[g]) * mw[sp] - rhoY[sp]) / dt;
    }
  });
  amrex::Gpu::streamSynchronize();
  auto const& s = f.s.const_array();
  auto const& ec = efab.const_array();
  amrex::Real err = 0.;
  amrex::Real ref = 0.;
  const amrex::IntVect iv = bx.smallEnd();
  for (int c = sSpec; c < nS; ++c) {
    err = amrex::max(err, std::abs(s(iv, c) - ec(iv, c)));
    ref = amrex::max(ref, std::abs(ec(iv, c)));
  }
  amrex::Print() << "One subcycle vs one explicit step: max |dS| = " << err
                 << " (max |S| = " << ref << ")\n";
  check(
    err <= 1.e-12 * ref,
    "num_subcycles = max_subcycles = 1 is one explicit step");
  s1.cleanup();
}

// Fixed subcycles (num_subcycles = max_subcycles = n) converge at first
// order (on a tenth of estSootDt), and on a step of estSootDt (the step a
// solver takes) the
// inputs-file and adaptive (num_subcycles = 1) subcycling keep every species
// non-negative and balance mass. Longer steps with few fixed subcycles can
// drive species negative: the subcycle step is limited by the moments only.
void
checkSubcycling(const SootModel& soot, Checker& check)
{
  const amrex::Box bx = stateBox();
  Fabs f(bx);
  fillStates(*soot.getSootData(), f.q);
  auto const& q = f.q.const_array();
  auto const& s = f.s.const_array();
  const amrex::Real dt = soot.estSootDt(bx, q);

  // Bimodal soot at 2000 K
  const amrex::IntVect iv(AMREX_D_DECL(bx.bigEnd(0), 1, 0));
  const int nsubs[] = {1, 10, 100, 1000};
  std::vector<amrex::Real> sfv;
  for (const int n : nsubs) {
    SootModel sm;
    defineSoot(sm, {.num_subcycles = n, .max_subcycles = n});
    sourceOn(sm, f.q, f.mu, f.s, 0.1 * dt);
    sfv.push_back(s(iv, sSoot + 1));
    sm.cleanup();
  }
  const amrex::Real e1 = std::abs(sfv[0] - sfv[3]);
  const amrex::Real e10 = std::abs(sfv[1] - sfv[3]);
  const amrex::Real e100 = std::abs(sfv[2] - sfv[3]);
  amrex::Print() << "dt = estSootDt = " << dt
                 << " s; S_fv over dt/10 with 1, 10, 100, 1000 subcycles:";
  for (const amrex::Real v : sfv) {
    amrex::Print() << " " << v;
  }
  amrex::Print() << "\n  differences from 1000: " << e1 << ", " << e10 << ", "
                 << e100 << "\n";
  // First order: e100 / e10 = (1/100 - 1/1000) / (1/10 - 1/1000) = 0.09
  check(
    e10 < e1 && e100 > 0. && e100 < 0.2 * e10,
    "fixed subcycles converge at first order");

  // Non-negative species and mass balance after a step of dt
  auto positive = [&](const SootModel& sm, const std::string& what) {
    sourceOn(sm, f.q, f.mu, f.s, dt);
    amrex::Real worst = 1.;
    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
      for (int sp = 0; sp < NUM_SPECIES; ++sp) {
        const amrex::Real rY = q(i, j, k, qRho) * q(i, j, k, qSpec + sp);
        if (rY > 0.) {
          worst = amrex::min(worst, (rY + dt * s(i, j, k, sSpec + sp)) / rY);
        }
      }
    });
    const auto [err, ref] = massBalance(bx, s);
    amrex::Print() << what << ": min (rho Y_k)_new / (rho Y_k) = " << worst
                   << "\n";
    check(
      worst >= -1.e-12 && !f.s.contains_nan<amrex::RunOn::Host>(),
      what + " on a step of estSootDt: species >= 0");
    check(
      err <= 1.e-10 * ref, what + " on a step of estSootDt: mass balanced",
      "#732");
  };
  positive(soot, "inputs subcycling");
  SootModel sm;
  defineSoot(sm, {.num_subcycles = 1, .max_subcycles = 1000});
  positive(sm, "adaptive subcycling (num_subcycles = 1)");
  sm.cleanup();
}

} // namespace

void
testOptions(const SootModel& soot, Checker& check)
{
  checkConserveMass(soot, check);
  checkPAH(soot, check);
  checkOneSubcycle(check);
  checkSubcycling(soot, check);
}

} // namespace soottest
