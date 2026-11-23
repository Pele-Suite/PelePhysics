// Unit tests of the soot model (Source/Soot) on its own: AMReX and
// PelePhysics only, PeleC conventions (CGS units, primitive state with mass
// fractions). Each component has its own file:
//   test_moments.cpp   unit conversions, lookup tables, clipping, moment
//                      interpolation (SootData)
//   test_reactions.cpp each moment source piece, the surface chemistry and
//                      the carbon balance (SootData, SootReaction)
//   test_source.cpp    computeSootSourceTerm contract: mass, species and
//                      energy balances, cutoff, clipping
//   test_options.cpp   conserve_mass, the PAH inception choices, subcycling
//   test_timestep.cpp  estSootDt
//   test_derive.cpp    the derived plot variables
//   test_reactor.cpp   closed reactor: conservation and realizability
// Prints a line per check (PASS, FAIL, or XFAIL/XPASS for checks tagged with
// the fix they need) and returns non-zero on any FAIL or XPASS.

#include <AMReX.H>
#include <AMReX_Print.H>

#include <utility>

#include "SootTest.H"

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    using namespace soottest;
    SootModel soot;
    defineSoot(soot);
    Checker check;
    const std::pair<const char*, void (*)(const SootModel&, Checker&)> tests[] =
      {{"Moments", testMoments},    {"Reactions", testReactions},
       {"Source term", testSource}, {"Options", testOptions},
       {"Time step", testTimeStep}, {"Derive", testDerive},
       {"Reactor", testReactor}};
    for (const auto& [name, test] : tests) {
      amrex::Print() << "\n-- " << name << " --\n";
      test(soot, check);
    }
    soot.cleanup();
    status = check.summary("Soot");
  }
  amrex::Finalize();
  return status;
}
