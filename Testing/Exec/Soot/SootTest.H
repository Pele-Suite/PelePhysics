#ifndef SOOTTEST_H
#define SOOTTEST_H

// Shared layout, state builders and helpers of the soot unit tests (PeleC
// conventions: CGS units, primitive state with mass fractions)

#include <AMReX_FArrayBox.H>
#include <AMReX_Array.H>
#include <AMReX_GpuContainers.H>

#include <array>
#include <string>
#include <utility>

#include "mechanism.H"
#include "SootModel.H"
#include "TestCheck.H"

namespace soottest {

// Component layout of the primitive state and of the source
constexpr int qRho = 0;
constexpr int qTemp = 1;
constexpr int qSpec = 2;
constexpr int qSoot = qSpec + NUM_SPECIES;
constexpr int nQ = qSoot + NUM_SOOT_MOMENTS + 1;
constexpr int sRho = 0;
constexpr int sEng = 1;
constexpr int sSpec = 2;
constexpr int sSoot = sSpec + NUM_SPECIES;
constexpr int nS = sSoot + NUM_SOOT_MOMENTS + 1;
constexpr int NMOM = NUM_SOOT_MOMENTS + 1;

// Pressure of every state (dyn/cm^2) and the gas viscosity (g/(cm s))
constexpr amrex::Real P = 1013250.;
constexpr amrex::Real MU = 5.e-4;

using Moments = amrex::GpuArray<amrex::Real, NMOM>;

using pele::testing::Checker;
using pele::testing::relErr;

// The soot inputs that the model variants change. The inputs file gives the
// others.
struct SootOptions
{
  std::string incept_pah = "A2";
  std::string pah_name = "A2";
  bool conserve_mass = true;
  int num_subcycles = 10;
  int max_subcycles = 1000;
};

// Define with these options and the inputs file (ParmParse "soot")
void defineSoot(SootModel& soot, const SootOptions& opt = {});

// Mass fractions of a rich post-flame mixture with soot precursors (A1, A2)
std::array<amrex::Real, NUM_SPECIES> sootingMixture();

// Moments (mol of C) of N0 nuclei plus two populations of aggregates in the
// second mode, with fv of about 1e-7: a well-developed soot state
Moments bimodalMoments(const SootData& sd);

// Moments (CGS) of the smallest particles, as a fresh state starts
Moments freshMoments(const SootData& sd);

// Fill one cell of the primitive state; the density follows from P, Y, T
void setCell(
  amrex::Array4<amrex::Real> const& q,
  const amrex::IntVect& iv,
  amrex::Real T,
  const amrex::Real* Y,
  const Moments& mom_cgs);

// Box of states: row 0 fresh soot (nuclei only), row 1 bimodal soot, with
// T rising from 1400 K to 2000 K along each row
amrex::Box stateBox();
void fillStates(const SootData& sd, amrex::FArrayBox& qfab);

// Row (i0, 0) to (i1, 0), and the single cell (i, 0)
amrex::Box rowBox(int i0, int i1);
inline amrex::Box
cellBox(const int i)
{
  return rowBox(i, i);
}

// State, viscosity (MU) and source of a box, in managed memory
struct Fabs
{
  amrex::FArrayBox q, mu, s;
  explicit Fabs(const amrex::Box& bx);
};

// The model's source on the box of q into s, which is zeroed first unless
// zero is false (the source adds into it)
void sourceOn(
  const SootModel& sm,
  const amrex::FArrayBox& q,
  const amrex::FArrayBox& mu,
  amrex::FArrayBox& s,
  amrex::Real dt,
  bool pres_term = true,
  bool zero = true);

// Mass balance of a source with conserve_mass: max |S_rho + rho_soot S_fv|
// and max |rho_soot S_fv| over the box
std::pair<amrex::Real, amrex::Real>
massBalance(const amrex::Box& bx, amrex::Array4<const amrex::Real> const& s);

// Model and state checks, by component
void testMoments(const SootModel& soot, Checker& check);
void testReactions(const SootModel& soot, Checker& check);
void testSource(const SootModel& soot, Checker& check);
void testOptions(const SootModel& soot, Checker& check);
void testTimeStep(const SootModel& soot, Checker& check);
void testDerive(const SootModel& soot, Checker& check);
void testReactor(const SootModel& soot, Checker& check);

} // namespace soottest

#endif
