#ifndef SPRAYTEST_H
#define SPRAYTEST_H

// Shared helpers of the spray test suite. PeleC conventions: CGS units and
// the conservative gas state rho, rho*u, rho*E, T, rho*Y.

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mechanism.H"
#include <PelePhysics.H>
#include "SprayParticles.H"
#include "TestCheck.H"

namespace spraytest {

using amrex::Real;
using amrex::RealVect;
using PType = SprayParticleContainer::ParticleType;
using TransParmType = pele::physics::transport::
  TransParm<pele::physics::EosType, pele::physics::TransportType>;
using SC = SprayComps;

// Conservative state components
constexpr int cRho = 0;
constexpr int cMom = 1;
constexpr int cEng = cMom + AMREX_SPACEDIM;
constexpr int cTemp = cEng + 1;
constexpr int cRhoY = cTemp + 1;
constexpr int nComp = cRhoY + NUM_SPECIES;
// Spray source components, as in PeleC: rho, momentum, energy, then one per
// phase change species
constexpr int sRho = 0;
constexpr int sMom = 1;
constexpr int sEng = sMom + AMREX_SPACEDIM;
constexpr int sSpec = sEng + 1;
constexpr int fuel = NC7H16_ID;

using pele::testing::Checker;
using pele::testing::relErr;

// Uniform gas (optionally with a temperature ramp in x)
struct Gas
{
  Real T = 300.;
  Real p = 1013250.;
  Real X_fuel = 0.; // fuel mole fraction, the rest air
  RealVect vel = RealVect::TheZeroVector();
  Real dTdx = 0.; // per cm from x = 0
};

// Everything the tests share: the spray data as set up from the inputs, the
// transport parameters and a scratch copy of the spray data on the device
struct Env
{
  SprayData base;          // spray data as read and set up
  const TransParmType* tp; // device transport parameters
  SprayData* scratch;      // managed memory, for single particle tests
  Real cBoilT = 0.;        // boiling temperature at 1 atm
  Real tol = 1.e-10;       // round-off tolerance of exact balances
};

// Common helpers (common.cpp)
// The gas of the single particle tests: cells of fabDx from the origin, ten
// per direction and a ghost cell around them, in managed memory
constexpr Real fabDx = 0.1;
amrex::FArrayBox gasFab(const Gas& g);
void fillState(amrex::MultiFab& s, const Gas& g, const amrex::Geometry& geom);
PType
makeDrop(const RealVect& pos, const RealVect& vel, Real T, Real dia, Real nppp);
Real liqRho(const SprayData& fd, Real T, Real cBoilT);
Real dropMass(const SprayData& fd, const PType& p, Real cBoilT);
Real dropEnergy(const SprayData& fd, const PType& p, Real cBoilT);
Real filmMass(const SprayData& fd, const PType& p, Real cBoilT);
RealVect dropVel(const PType& p);
Real boilT(const SprayData& fd, Real p);
bool finiteParticle(const PType& p);
// Set the spray data of the container, on host and device
void setSprayData(const SprayData& sd);

// Sets the spray data for a scope, then puts back the spray data of the
// inputs
class ScopedSprayData
{
public:
  ScopedSprayData(const Env& env, const SprayData& sd) : m_base(env.base)
  {
    setSprayData(sd);
  }
  ~ScopedSprayData() { setSprayData(m_base); }
  ScopedSprayData(const ScopedSprayData&) = delete;
  ScopedSprayData& operator=(const ScopedSprayData&) = delete;

private:
  const SprayData& m_base;
};

// Index of a derived spray variable, by name
int deriveIndex(const std::string& name);

// A single level (or two level) AmrCore with fixed grids, for the particle
// container. Level 1, if any, covers fine_region (level 0 index space).
class TestAmr : public amrex::AmrCore
{
public:
  TestAmr(
    const amrex::Geometry& geom,
    const amrex::AmrInfo& info,
    const amrex::Box& fine_region);

protected:
  void MakeNewLevelFromScratch(
    int lev,
    Real time,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm) override;
  void MakeNewLevelFromCoarse(
    int lev,
    Real time,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm) override
  {
    MakeNewLevelFromScratch(lev, time, ba, dm);
  }
  void RemakeLevel(
    int lev,
    Real time,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm) override
  {
    MakeNewLevelFromScratch(lev, time, ba, dm);
  }
  void ClearLevel(int lev) override;
  void
  ErrorEst(int lev, amrex::TagBoxArray& tags, Real time, int ngrow) override;

private:
  amrex::Box m_fine;
};

// Mesh, container and fields of a container test
struct Mesh
{
  std::unique_ptr<TestAmr> amr;
  std::unique_ptr<SprayParticleContainer> pc;
  amrex::Vector<amrex::MultiFab> state; // gas state with ghost cells
  amrex::Vector<amrex::MultiFab> tmp;   // spray source, with ghost cells
  amrex::Vector<amrex::MultiFab> act;   // source on the state layout
  int state_ghosts = 0;
  int source_ghosts = 0;
  Real vol(int lev) const; // cell volume (2D: Lz = dx)
};

// Boundary conditions per direction: 0 periodic, else a PhysBCType
struct MeshSpec
{
  int ncell = 32;
  int max_grid_size = 8;
  int max_level = 0;
  amrex::Box fine_region;
  RealVect plo = RealVect(AMREX_D_DECL(-0.25, 0.5, 0.));
  Real length = 1.;
  amrex::IntVect bc_lo = amrex::IntVect(AMREX_D_DECL(0, 4, 0));
  amrex::IntVect bc_hi = amrex::IntVect(AMREX_D_DECL(0, 2, 0));
  Real cfl = 0.5; // particle CFL the ghost cells are sized for
};

Mesh makeMesh(const MeshSpec& spec, const Gas& gas);
// Add particles and Redistribute
void addParticles(Mesh& m, const std::vector<PType>& parts);
// Add a particle to its tile without Redistribute
void insertParticle(Mesh& m, const PType& p, int lev = 0);
// All particles with id > 0, by id, and their level
std::map<int, PType>
collect(const Mesh& m, std::map<int, int>* level = nullptr);
// One spray update of every level: moveKickDrift (do_move) or moveKick, then
// transferSource; the sources accumulate in act
void sprayStep(Mesh& m, const Env& env, Real dt, bool do_move, Real cfl = -1.);

// Particle totals
struct Totals
{
  Real mass = 0.;
  RealVect mom = RealVect::TheZeroVector();
  Real energy = 0.;
  Real film_mass = 0.;
  int count = 0;
  int films = 0;
};
Totals totals(const std::map<int, PType>& parts, const Env& env);
// Sum of components over the valid cells of all levels, times the cell volume
std::vector<Real>
sumSource(const Mesh& m, const amrex::Vector<amrex::MultiFab>& mf);

// The test sections
void testProperties(Checker& check, Env& env);
void testDrag(Checker& check, Env& env);
void testFilm(Checker& check, Env& env);
void testBoundaries(Checker& check, Env& env);
void testContainer(Checker& check, Env& env);
void testBreakupSplash(Checker& check, Env& env);
void testInjection(Checker& check, Env& env);

// Advance one particle by dt in the gas of gasFab, as updateParticles does
// (drag.cpp)
struct Result
{
  PType p;
  GasPhaseVals gpv;
  Real Reyn = 0.;
  Real cBoilT = 0.;
  Real dtp = 0.; // the particle's step, dtmod * dt
};
Result advance(
  Env& env,
  const Gas& gas,
  const SprayData& fd,
  const PType& p0,
  Real dt,
  bool film = false);
// Relative error of the mass source of r against the change of mass of nppp
// droplets (or films) from m0 to m1
Real massErr(const Result& r, Real m0, Real m1, Real nppp = 1.);
bool finite(const GasPhaseVals& gpv);
bool speciesOk(const GasPhaseVals& gpv, Real tol);

} // namespace spraytest

#endif
