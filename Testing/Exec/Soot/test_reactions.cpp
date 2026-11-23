// The moment source pieces (SootData) and the surface chemistry
// (SootReaction) on a bimodal soot state at 1800 K. Each piece is checked
// for its sign and against the gas consumption that feeds it:
//   nucleation    adds nuclei only, at 0.5 betaNucl [DIMER]^2
//   condensation  adds betaCond [DIMER] dimer volumes, no particles
//   dimers        nucleation + condensation volume = dimer volume * rate
//   coagulation   conserves volume, removes particles
//   growth        adds 2 C per C2H2 consumed, no particles
//   oxidation     removes 1 C per CO formed
//   fragmentation conserves volume, adds particles
//   all           computeSrcTerms is the sum of the pieces, and the soot
//                 carbon source is the gas carbon sink

#include <AMReX_Print.H>

#include <cmath>
#include <vector>

#include <PelePhysics.H>
#include "SootTest.H"

namespace soottest {

namespace {

enum ReactResult {
  dimerRate,
  dimerRate2X,       // dimerRate with twice the PAH / dimerRate
  dimerRate4T,       // dimerRate at four times T / dimerRate
  dimerResidual,     // (betaNucl D^2 + betaCond D - rate) / rate
  dimerResidualCond, // the same where condensation dominates (b^2 = 1e12 4ac)
  dimerCond,         // that dimer concentration
  nuclN0,
  nuclN0Expect, // 0.5 betaNucl D^2
  nuclShape,    // max_i |S_i - momFact_i S_N0| / (momFact_i S_N0)
  nuclScale,    // nucleation source with 2 D / with D
  condM0,
  condM1,
  condM1Expect, // betaCond D dimerVol
  condN0,
  dimerVolSrc, // nucleation + condensation volume source
  coagM0,
  coagM1,
  coagM01,
  coagN0,
  coagM20, // 6 moments only
  kSG,
  kOx,
  kO2,
  sgM0,
  sgM1,
  sgM1Expect, // 2 * C2H2 consumption
  sgN0,
  oxM0,
  oxM1,
  oxM1Expect, // -CO formation
  oxN0,
  frM0,
  frM1,
  frN0,
  noSurfPAH,   // PAH source with nuclei only
  noSurfOther, // max |other gas sources| with nuclei only
  piecesErr,   // max_i |computeSrcTerms_i - sum of pieces_i| / sum |pieces_i|
  carbonSoot,  // soot carbon source (mol C/cm^3/s)
  carbonGas,   // gas carbon sink
  carbonScale, // sum of |carbon flows|
  numReactResults
};

} // namespace

void
testReactions(const SootModel& soot, Checker& check)
{
  const SootData& sd = *soot.getSootData();
  const SootData* sd_d = soot.getSootData_d();
  const SootReaction* sr_d = soot.d_sootReact;
  const amrex::Real betaNF = soot.m_betaNuclFact;

  // Gas state at 1800 K
  const amrex::Real T = 1800.;
  const auto Y = sootingMixture();
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real rho;
  eos.PYT2R(P, Y.data(), T, rho);
  amrex::Real mw[NUM_SPECIES];
  eos.molecular_weight(mw);
  amrex::Real inv_mm = 0.;
  for (int sp = 0; sp < NUM_SPECIES; ++sp) {
    inv_mm += Y[sp] / mw[sp];
  }
  const amrex::Real molarMass = 1. / inv_mm;
  // Soot gas species: concentrations and carbon atoms
  amrex::Vector<std::string> enames;
  CKSYME_STR(enames);
  int iC = -1;
  for (int e = 0; e < static_cast<int>(enames.size()); ++e) {
    if (enames[e] == "C") {
      iC = e;
    }
  }
  std::vector<int> ncf(NUM_ELEMENTS * NUM_SPECIES);
  CKNCF(ncf.data());
  amrex::GpuArray<amrex::Real, NUM_SOOT_GS> xi;
  amrex::GpuArray<amrex::Real, NUM_SOOT_GS> nC;
  for (int g = 0; g < NUM_SOOT_GS; ++g) {
    const int k = sd.refIndx[g];
    xi[g] = rho * Y[k] / mw[k];
    nC[g] = ncf[k * NUM_ELEMENTS + iC];
  }
  const Moments m_mol = bimodalMoments(sd);

  amrex::Gpu::ManagedVector<amrex::Real> res(numReactResults, 0.);
  amrex::Real* r = res.data();
  const SootConst sc;
  // A copy: MU is passed by reference, so the device needs its own
  const amrex::Real mu = MU;
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) noexcept {
    constexpr int N = NUM_SOOT_MOMENTS;
    constexpr int NG = NUM_SOOT_GS;
    const amrex::Real convT = std::sqrt(sc.colFact * T);
    const amrex::Real colConst =
      convT * sc.colFactPi23 * sc.colFact16 * pele::physics::Constants::Avna;
    const amrex::Real betaNucl = convT * betaNF;
    const amrex::Real dimerVol = 0.5 * sd_d->nuclVol;
    Moments m = m_mol;
    amrex::Real fv[N + 2];
    sd_d->computeFracMomVect(m.data(), fv);
    const amrex::Real xiPAH = xi[SootGasSpecIndx::indxPAH];

    // Dimerization: rate ~ [PAH]^2 T^(1/2), and the dimer concentration
    // solves betaNucl D^2 + betaCond D = rate
    const amrex::Real rate = sr_d->dimerRate(T, xiPAH);
    r[dimerRate] = rate;
    r[dimerRate2X] = sr_d->dimerRate(T, 2. * xiPAH) / rate;
    r[dimerRate4T] = sr_d->dimerRate(4. * T, xiPAH) / rate;
    const amrex::Real D = sd_d->dimerization(convT, betaNucl, rate, fv);
    const amrex::Real betaCond = sd_d->getBetaCond(convT, fv);
    r[dimerResidual] = (betaNucl * D * D + betaCond * D - rate) / rate;
    {
      // Unit betaNucl, condensation dominating: b^2 = 1e12 * 4ac
      const amrex::Real c = 0.25e-12 * betaCond * betaCond;
      const amrex::Real x = sd_d->dimerization(convT, 1., c, fv);
      r[dimerResidualCond] = (x * x + betaCond * x - c) / c;
      r[dimerCond] = x;
    }

    // Nucleation
    amrex::Real sn[N + 1] = {0.};
    amrex::Real sn2[N + 1] = {0.};
    sd_d->nucleationMomSrc(betaNucl, D, sn);
    sd_d->nucleationMomSrc(betaNucl, 2. * D, sn2);
    r[nuclN0] = sn[N];
    r[nuclN0Expect] = 0.5 * betaNucl * D * D;
    for (int i = 0; i < N; ++i) {
      r[nuclShape] =
        amrex::max(r[nuclShape], relErr(sn[i], sd_d->momFact[i] * sn[N]));
    }
    r[nuclScale] = sn2[N] / sn[N];

    // Condensation
    amrex::Real scd[N + 1] = {0.};
    sd_d->condensationMomSrc(colConst, D, fv, scd);
    r[condM0] = scd[0];
    r[condM1] = scd[1];
    r[condM1Expect] = betaCond * D * dimerVol;
    r[condN0] = scd[N];
    r[dimerVolSrc] = sn[1] + scd[1];

    // Coagulation
    amrex::Real sco[N + 1] = {0.};
    sd_d->coagulationMomSrc(colConst, T, mu, rho, molarMass, fv, sco);
    r[coagM0] = sco[0];
    r[coagM1] = sco[1];
    r[coagM01] = sco[2];
    r[coagN0] = sco[N];
#if NUM_SOOT_MOMENTS == 6
    r[coagM20] = sco[3];
#endif

    // Surface chemistry: rates and gas sources
    const amrex::Real surf = sc.S0 * sd_d->fracMom(0., 1., fv);
    amrex::Real k_sg = 0.;
    amrex::Real k_ox = 0.;
    amrex::Real k_o2 = 0.;
    amrex::Real om[NG] = {0.};
    auto xin = xi;
    sr_d->chemicalSrc(T, surf, xin.data(), m.data(), k_sg, k_ox, k_o2, om);
    r[kSG] = k_sg;
    r[kOx] = k_ox;
    r[kO2] = k_o2;

    // Surface growth
    amrex::Real ssg[N + 1] = {0.};
    sd_d->surfaceGrowthMomSrc(k_sg, fv, ssg);
    r[sgM0] = ssg[0];
    r[sgM1] = ssg[1];
    r[sgM1Expect] = -2. * om[SootGasSpecIndx::indxC2H2];
    r[sgN0] = ssg[N];

    // Oxidation alone, fragmentation alone
    amrex::Real sox[N + 1] = {0.};
    amrex::Real sfr[N + 1] = {0.};
    sd_d->oxidFragMomSrc(k_ox, 0., fv, sox);
    sd_d->oxidFragMomSrc(0., k_o2, fv, sfr);
    r[oxM0] = sox[0];
    r[oxM1] = sox[1];
    r[oxM1Expect] = -om[SootGasSpecIndx::indxCO];
    r[oxN0] = sox[N];
    r[frM0] = sfr[0];
    r[frM1] = sfr[1];
    r[frN0] = sfr[N];

    // Nuclei only (fv ~ 1e-24): too little soot for the surface reactions,
    // so only the dimerization consumes gas
    {
      Moments mf = sd_d->smallMoms;
      sd_d->clipMoments(mf.data());
      amrex::Real sf[N + 1] = {0.};
      amrex::Real omf[NG] = {0.};
      amrex::Real fvf[N + 2];
      auto xif = xi;
      sd_d->computeSrcTerms(
        T, mu, rho, molarMass, convT, betaNucl, colConst, xif.data(), omf,
        mf.data(), sf, fvf, sr_d);
      r[noSurfPAH] = omf[SootGasSpecIndx::indxPAH];
      for (int g = 0; g < NG; ++g) {
        if (g != SootGasSpecIndx::indxPAH) {
          r[noSurfOther] = amrex::max(r[noSurfOther], amrex::Math::abs(omf[g]));
        }
      }
    }

    // All the pieces together
    amrex::Real st[N + 1] = {0.};
    amrex::Real omt[NG] = {0.};
    amrex::Real fvt[N + 2];
    sd_d->computeSrcTerms(
      T, mu, rho, molarMass, convT, betaNucl, colConst, xin.data(), omt,
      m.data(), st, fvt, sr_d);
    amrex::Real sbo[N + 1] = {0.};
    sd_d->oxidFragMomSrc(k_ox, k_o2, fv, sbo);
    for (int i = 0; i <= N; ++i) {
      const amrex::Real pieces[5] = {sn[i], scd[i], sco[i], ssg[i], sbo[i]};
      amrex::Real sum = 0.;
      amrex::Real sum_abs = 0.;
      for (const amrex::Real p : pieces) {
        sum += p;
        sum_abs += amrex::Math::abs(p);
      }
      r[piecesErr] =
        amrex::max(r[piecesErr], amrex::Math::abs(st[i] - sum) / sum_abs);
    }
    for (int g = 0; g < NG; ++g) {
      r[carbonGas] -= nC[g] * omt[g];
      r[carbonScale] += amrex::Math::abs(nC[g] * omt[g]);
    }
    r[carbonSoot] = st[1];
  });
  amrex::Gpu::streamSynchronize();

  const amrex::Real tol = 1.e-12;
  amrex::Print() << "Dimerization rate = " << r[dimerRate] << " mol/cm^3/s; "
                 << "k_sg = " << r[kSG] << ", k_ox = " << r[kOx]
                 << ", k_o2 = " << r[kO2] << "\n"
                 << "Coagulation: S_M00 = " << r[coagM0]
                 << ", S_M01 = " << r[coagM01] << ", S_N0 = " << r[coagN0]
                 << "\n"
                 << "Carbon: soot source = " << r[carbonSoot]
                 << ", gas sink = " << r[carbonGas] << "\n";
  check(
    r[dimerRate] > 0. && relErr(r[dimerRate2X], 4.) <= tol &&
      relErr(r[dimerRate4T], 2.) <= tol,
    "dimerization rate > 0, ~ [PAH]^2 and ~ T^(1/2)");
  check(
    std::abs(r[dimerResidual]) <= tol,
    "the dimer concentration solves the dimer balance");
  check(
    r[dimerCond] > 0. && std::abs(r[dimerResidualCond]) <= 1.e-10,
    "the dimer balance where condensation dominates", "#732");
  check(
    relErr(r[nuclN0], r[nuclN0Expect]) <= tol && r[nuclShape] <= tol &&
      relErr(r[nuclScale], 4.) <= tol,
    "nucleation adds nuclei at 0.5 betaNucl [DIMER]^2");
  check(
    r[condM0] == 0. && r[condN0] < 0. &&
      relErr(r[condM1], r[condM1Expect]) <= tol,
    "condensation adds betaCond [DIMER] dimer volumes, no particles");
  check(
    relErr(r[dimerVolSrc], 0.5 * sd.nuclVol * r[dimerRate]) <= 1.e-10,
    "nucleation + condensation volume = dimer volume * dimerization rate");
  check(
    r[coagM1] == 0. && r[coagM0] < 0. && r[coagN0] < 0.,
    "coagulation conserves volume and removes particles");
#if NUM_SOOT_MOMENTS == 6
  check(r[coagM20] > 0., "coagulation raises the volume variance (M20)");
#endif
  check(
    r[kSG] > 0. && r[kOx] > 0. && r[kO2] > 0.,
    "surface growth, oxidation and fragmentation rates > 0");
  check(
    r[sgM0] == 0. && r[sgN0] < 0. && r[sgM1] > 0. &&
      relErr(r[sgM1], r[sgM1Expect]) <= tol,
    "surface growth adds 2 C per C2H2, no particles");
  check(
    r[oxM1] < 0. && r[oxM0] <= 0. && r[oxN0] <= 0. &&
      relErr(r[oxM1], r[oxM1Expect]) <= tol,
    "oxidation removes 1 C per CO formed");
  check(
    r[frM1] == 0. && r[frM0] > 0.,
    "fragmentation conserves volume and adds particles");
  check(
    relErr(r[noSurfPAH], -2. * r[dimerRate]) <= tol && r[noSurfOther] == 0.,
    "nuclei only: only the dimerization consumes gas", "#732");
  check(r[piecesErr] <= tol, "computeSrcTerms is the sum of the source pieces");
  check(
    std::abs(r[carbonSoot] - r[carbonGas]) <= tol * r[carbonScale],
    "soot carbon source = gas carbon sink");
}

} // namespace soottest
