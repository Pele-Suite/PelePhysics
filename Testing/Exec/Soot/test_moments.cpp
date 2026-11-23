// SootData moment handling: unit conversions, the lookup tables of the
// nucleus and dimer sizes, clipping, and the moment interpolation
// (computeFracMomVect, fracMom, fracMomLarge)

#include <AMReX_Print.H>

#include <cmath>

#include "SootTest.H"

namespace soottest {

namespace {

// Results of the device checks
enum MomResult {
  clipNoop,       // max rel change when clipping realizable moments
  clipRealizable, // clipped unrealizable moments are realizable
  clipIdempotent, // max |clip(clip(m)) - clip(m)|
  clipZero,       // clipped zero and negative moments are realizable
  clipWeight,     // M00 - (M00 + smallWeight) after adding nuclei
  clipWeightN0,   // and their weight
  convClipConv,   // max |momConvClipConv - (toMol, clip, fromMol)|
  modeBimodal,    // modeCoef of the bimodal state
  fracInt,        // max rel error of fracMom at the integer orders
  fracLargeInt,   // max rel error of fracMomLarge at the integer orders
  modeSmall,      // modeCoef of the smallest particles
  fracIntSmall,   // max rel error of fracMom at integer orders, one mode
  fracLargeSmall, // fracMomLarge(1, 0) / (nuclVol M00) with one mode
  numMomResults
};

} // namespace

void
testMoments(const SootModel& soot, Checker& check)
{
  const SootData& sd = *soot.getSootData();
  const SootData* sd_d = soot.getSootData_d();
  const SootConst sc;
  const amrex::Real Avna = pele::physics::Constants::Avna;
  const amrex::Real tol = 1.e-14;

  // Unit conversions (CGS): the number moments are in mol, the volume
  // moment is in mol of C (fv = M10 * M_C / rho_soot)
  check(
    relErr(sd.unitConv[0], Avna) <= tol &&
      relErr(sd.unitConv[NUM_SOOT_MOMENTS], Avna) <= tol &&
      relErr(sd.unitConv[1], sc.SootMolarMass / sc.SootDensity) <= tol,
    "unitConv: number moments in mol, fv = M10 * M_C / rho_soot");
  bool unit_ok = true;
  for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
    const amrex::Real expect = std::pow(sc.V0, sc.MomOrderV[i]) *
                               std::pow(sc.S0, sc.MomOrderS[i]) * Avna;
    unit_ok = unit_ok && relErr(sd.unitConv[i], expect) <= tol;
  }
  check(unit_ok, "unitConv of every moment is V0^x S0^y Avna");

  // Round trip through convertFromMol and convertToMol
  const Moments m_mol = bimodalMoments(sd);
  Moments m = m_mol;
  sd.convertFromMol(m.data());
  sd.convertToMol(m.data());
  amrex::Real err_trip = 0.;
  for (int i = 0; i < NMOM; ++i) {
    err_trip = amrex::max(err_trip, relErr(m[i], m_mol[i]));
  }
  // The two-argument convertFromMol converts both arrays
  Moments a = m_mol;
  Moments b = m_mol;
  Moments c = m_mol;
  sd.convertFromMol(a.data(), b.data());
  sd.convertFromMol(c.data());
  bool two_arg = true;
  for (int i = 0; i < NMOM; ++i) {
    two_arg = two_arg && a[i] == c[i] && b[i] == c[i];
  }
  amrex::Print() << "Unit round trip: max rel error = " << err_trip << "\n";
  check(err_trip <= 4.e-16, "convertToMol(convertFromMol(m)) = m");
  check(two_arg, "convertFromMol(src, mom) converts both like convertFromMol");

  // Smallest particles: smallWeight nuclei
  const Moments ms = freshMoments(sd);
  bool small_ok = relErr(
                    ms[NUM_SOOT_MOMENTS] / sd.unitConv[NUM_SOOT_MOMENTS],
                    sc.smallWeight) <= tol;
  for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
    small_ok =
      small_ok &&
      relErr(ms[i] / sd.unitConv[i], sc.smallWeight * sd.momFact[i]) <= tol;
  }
  check(small_ok, "initialSmallMomVals is smallWeight nuclei");

  // Nucleus size and the lookup tables of fractional powers
  const amrex::Real dimerVol = 0.5 * sd.nuclVol;
  bool nucl_ok = relErr(sd.nuclSurf, std::pow(sd.nuclVol, 2. / 3.)) <= tol;
  for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
    nucl_ok = nucl_ok &&
              relErr(
                sd.momFact[i], std::pow(sd.nuclVol, sc.MomOrderV[i]) *
                                 std::pow(sd.nuclSurf, sc.MomOrderS[i])) <= tol;
  }
  check(
    nucl_ok, "nucleus surface nuclVol^(2/3), momFact = nuclVol^x nuclSurf^y");
  bool table_ok = true;
  for (int e = -3; e <= 13; e += 2) {
    table_ok =
      table_ok && relErr(sd.getDimerExp6(e), std::pow(dimerVol, e / 6.)) <= tol;
  }
  for (int e = -3; e <= 7; ++e) {
    table_ok = table_ok &&
               relErr(sd.getNuclExp3(e), std::pow(sd.nuclVol, e / 3.)) <= tol &&
               relErr(sd.getNuclExp6(e), std::pow(sd.nuclVol, e / 6.)) <= tol;
  }
  check(table_ok, "getDimerExp6, getNuclExp3 and getNuclExp6 tables");

  // Device checks on the bimodal state (mol of C)
  amrex::Gpu::ManagedVector<amrex::Real> res(numMomResults, 0.);
  amrex::Real* r = res.data();
  amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE(int /*n*/) noexcept {
    constexpr int N = NUM_SOOT_MOMENTS;
    const amrex::Real slack = 1. - 1.e-14;
    auto realizable = [=](const amrex::Real mm[]) {
      bool ok = mm[N] > 0. && mm[N] <= mm[0] && mm[0] > 0. &&
                mm[1] >= slack * sd_d->nuclVol * mm[0] &&
                mm[2] >= slack * sd_d->nuclSurf * mm[0];
#if NUM_SOOT_MOMENTS == 6
      // Variances only: V and S may be negatively correlated
      ok = ok && mm[3] >= slack * mm[1] * mm[1] / mm[0] &&
           mm[5] >= slack * mm[2] * mm[2] / mm[0];
#endif
      for (int i = 0; i <= N; ++i) {
        ok = ok && std::isfinite(mm[i]);
      }
      return ok;
    };

    // Clipping realizable moments changes nothing
    Moments mc = m_mol;
    sd_d->clipMoments(mc.data());
    for (int i = 0; i <= N; ++i) {
      r[clipNoop] = amrex::max(r[clipNoop], relErr(mc[i], m_mol[i]));
    }

    // Ten times more particles than nuclei of all the volume would make:
    // clipped to realizable,
    // and clipping again changes nothing
    mc = m_mol;
    mc[0] = 10. * mc[1] / sd_d->nuclVol;
    sd_d->clipMoments(mc.data());
    r[clipRealizable] = realizable(mc.data()) ? 1. : 0.;
    Moments mc2 = mc;
    sd_d->clipMoments(mc2.data());
    for (int i = 0; i <= N; ++i) {
      r[clipIdempotent] =
        amrex::max(r[clipIdempotent], amrex::Math::abs(mc2[i] - mc[i]));
    }

    // Zero and negative moments
    amrex::Real mz[N + 1];
    amrex::Real mn[N + 1];
    for (int i = 0; i <= N; ++i) {
      mz[i] = 0.;
      mn[i] = -m_mol[i];
    }
    sd_d->clipMoments(mz);
    sd_d->clipMoments(mn);
    r[clipZero] = (realizable(mz) && realizable(mn)) ? 1. : 0.;

    // Without nuclei, clipping adds smallWeight of them
    mc = m_mol;
    mc[N] = 0.;
    sd_d->clipMoments(mc.data());
    r[clipWeight] = mc[0] - (m_mol[0] + sc.smallWeight);
    r[clipWeightN0] = mc[N];

    // momConvClipConv is convertToMol, clipMoments, convertFromMol
    Moments ca;
    for (int i = 0; i <= N; ++i) {
      ca[i] = m_mol[i] * sd_d->unitConv[i];
    }
    ca[0] =
      10. * ca[1] * sd_d->unitConv[0] / (sd_d->nuclVol * sd_d->unitConv[1]);
    Moments cb = ca;
    sd_d->momConvClipConv(ca.data());
    sd_d->convertToMol(cb.data());
    sd_d->clipMoments(cb.data());
    sd_d->convertFromMol(cb.data());
    for (int i = 0; i <= N; ++i) {
      r[convClipConv] =
        amrex::max(r[convClipConv], amrex::Math::abs(ca[i] - cb[i]));
    }

    // Interpolation reproduces the integer moments, and fracMomLarge the
    // second mode (the moments less the nuclei)
    amrex::Real fv[N + 2];
    sd_d->computeFracMomVect(m_mol.data(), fv);
    r[modeBimodal] = fv[N + 1];
    for (int i = 0; i < N; ++i) {
      const amrex::Real x = sc.MomOrderV[i];
      const amrex::Real y = sc.MomOrderS[i];
      r[fracInt] =
        amrex::max(r[fracInt], relErr(sd_d->fracMom(x, y, fv), m_mol[i]));
      const amrex::Real large = m_mol[i] - m_mol[N] * sd_d->momFact[i];
      r[fracLargeInt] = amrex::max(
        r[fracLargeInt], relErr(sd_d->fracMomLarge(x, y, fv), large));
    }

    // Only nuclei: one mode, still exact at the integer orders, and no
    // second mode
    Moments m1 = sd_d->smallMoms;
    sd_d->clipMoments(m1.data());
    sd_d->computeFracMomVect(m1.data(), fv);
    r[modeSmall] = fv[N + 1];
    for (int i = 0; i < N; ++i) {
      r[fracIntSmall] = amrex::max(
        r[fracIntSmall],
        relErr(sd_d->fracMom(sc.MomOrderV[i], sc.MomOrderS[i], fv), m1[i]));
    }
    r[fracLargeSmall] =
      sd_d->fracMomLarge(1., 0., fv) / (sd_d->nuclVol * m1[0]);
  });
  amrex::Gpu::streamSynchronize();

  amrex::Print() << "Interpolation at the integer orders: max rel error "
                 << r[fracInt] << " (fracMom), " << r[fracLargeInt]
                 << " (fracMomLarge), " << r[fracIntSmall] << " (one mode)\n";
  check(r[clipNoop] == 0., "clipping realizable moments changes nothing");
  check(
    r[clipRealizable] == 1., "clipping makes unrealizable moments realizable");
  check(r[clipIdempotent] == 0., "clipping is idempotent");
  check(r[clipZero] == 1., "clipping zero or negative moments is realizable");
  check(
    r[clipWeight] == 0. && r[clipWeightN0] == sc.smallWeight,
    "clipping adds smallWeight nuclei to a state without them");
  check(
    r[convClipConv] == 0., "momConvClipConv converts, clips, converts back");
  check(r[modeBimodal] == 1., "the bimodal state uses both modes");
  check(r[fracInt] <= 1.e-12, "fracMom reproduces the integer moments");
  check(
    r[fracLargeInt] <= 1.e-12,
    "fracMomLarge reproduces the second-mode integer moments");
  check(r[modeSmall] == 0., "nuclei alone use one mode");
  check(
    r[fracIntSmall] <= 1.e-12,
    "fracMom reproduces the integer moments with one mode");
  check(
    r[fracLargeSmall] < 1.e-30,
    "fracMomLarge is negligible without a second mode");
}

} // namespace soottest
