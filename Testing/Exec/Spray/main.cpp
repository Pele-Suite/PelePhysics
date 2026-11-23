// Spray test suite (PeleC conventions: CGS units, conservative state). Each
// file covers one part of Source/Spray:
//   properties.cpp  liquid properties and spraySetup
//   drag.cpp        droplet sources of calculateSpraySource
//   film.cpp        wall film sources of calculateFilmSource
//   boundaries.cpp  check_bounds, interpolation at walls, walls, outflows
//                   and periodic sides in the particle move
//   container.cpp   SprayParticleContainer source deposition, box layout,
//                   estTimestep, computeDerivedVars, two AMR levels
//   breakup.cpp     TAB and KH-RT breakup, splash and wall film creation
//   injection.cpp   SprayJet transforms, sprayInjection, uniformSprayInit
// Checks are balances and invariants, not stored reference values. The
// program prints PASS/FAIL lines (XFAIL/XPASS for checks tagged as known
// bugs) and returns non-zero on any failure or unexpected pass.

#include "SprayTest.H"

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  // Particles are inserted into grids without checking which rank owns them
  if (amrex::ParallelDescriptor::NProcs() > 1) {
    amrex::Abort("The spray tests run on one rank");
  }
  int status = 0;
  {
    using namespace spraytest;
    Checker check;
    amrex::InitRandom(42);

    int verbose = 0;
    SprayParticleContainer::readSprayParams(verbose);
    const Real body_force[AMREX_SPACEDIM] = {0.};
    SprayParticleContainer::spraySetup(body_force);
    SprayComps scomps;
    scomps.rhoIndx = cRho;
    scomps.momIndx = cMom;
    scomps.engIndx = cEng;
    scomps.utempIndx = cTemp;
    scomps.specIndx = cRhoY;
    scomps.rhoSrcIndx = sRho;
    scomps.momSrcIndx = sMom;
    scomps.engSrcIndx = sEng;
    scomps.specSrcIndx = sSpec;
    SprayParticleContainer::AssignSprayComps(scomps);
    pele::physics::PeleParams<TransParmType> trans_parms;
    trans_parms.initialize();

    Env env;
    env.base = *SprayParticleContainer::getSprayData();
    env.tp = trans_parms.device_parm();
    env.scratch = static_cast<SprayData*>(
      amrex::The_Managed_Arena()->alloc(sizeof(SprayData)));
    new (env.scratch) SprayData(env.base);
    env.cBoilT = boilT(env.base, Gas().p);
    // The inputs turn on splashing and TAB breakup to read their
    // properties; the tests switch them on where needed
    env.base.do_splash = false;
    env.base.do_breakup = 0;
    setSprayData(env.base);

    testProperties(check, env);
    testDrag(check, env);
    testFilm(check, env);
    testBoundaries(check, env);
    testContainer(check, env);
    testBreakupSplash(check, env);
    testInjection(check, env);

    env.scratch->~SprayData();
    amrex::The_Managed_Arena()->free(env.scratch);
    trans_parms.deallocate();
    SprayParticleContainer::SprayCleanUp();
    status = check.summary("Spray");
  }
  amrex::Finalize();
  return status;
}
