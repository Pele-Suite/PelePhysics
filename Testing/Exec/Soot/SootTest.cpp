#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <PelePhysics.H>
#include "SootTest.H"

namespace soottest {

namespace {

SootComps
sootIndices()
{
  SootComps idx;
  idx.qRhoIndx = qRho;
  idx.qTempIndx = qTemp;
  idx.qSpecIndx = qSpec;
  idx.qSootIndx = qSoot;
  idx.rhoIndx = sRho;
  idx.engIndx = sEng;
  idx.specIndx = sSpec;
  idx.sootIndx = sSoot;
  return idx;
}

} // namespace

void
defineSoot(SootModel& soot, const SootOptions& opt)
{
  // The options as inputs, removed again for the models defined later
  amrex::ParmParse pp("soot");
  const std::pair<const char*, std::string> keys[] = {
    {"incept_pah", opt.incept_pah},
    {"pah_name", opt.pah_name},
    {"conserve_mass", opt.conserve_mass ? "true" : "false"},
    {"num_subcycles", std::to_string(opt.num_subcycles)},
    {"max_subcycles", std::to_string(opt.max_subcycles)}};
  for (const auto& [key, value] : keys) {
    pp.add(key, value);
  }
  soot.readSootParams();
  soot.setIndices(sootIndices());
  soot.define();
  for (const auto& kv : keys) {
    pp.remove(kv.first);
  }
}

std::array<amrex::Real, NUM_SPECIES>
sootingMixture()
{
  std::array<amrex::Real, NUM_SPECIES> X = {0.};
  const std::vector<std::pair<int, amrex::Real>> mix = {
    {N2_ID, 0.60},  {H2O_ID, 0.12},  {CO_ID, 0.10}, {H2_ID, 0.08},
    {CO2_ID, 0.03}, {C2H2_ID, 0.05}, {H_ID, 5.e-3}, {OH_ID, 1.e-3},
    {O2_ID, 1.e-3}, {A1_ID, 1.e-4},  {A2_ID, 1.e-4}};
  amrex::Real xsum = 0.;
  for (const auto& [id, x] : mix) {
    X[id] = x;
    xsum += x;
  }
  for (amrex::Real& x : X) {
    x /= xsum;
  }
  std::array<amrex::Real, NUM_SPECIES> Y;
  auto eos = pele::physics::PhysicsType::eos();
  eos.X2Y(X.data(), Y.data());
  return Y;
}

Moments
bimodalMoments(const SootData& sd)
{
  const SootConst sc;
  // Number (mol/cm^3), volume (in V0) and surface (in S0) of each population
  const amrex::Real N0 = 1.e-14;
  const amrex::Real large[2][3] = {
    {5.e-15, 5.e5, 3. * std::pow(5.e5, 2. / 3.)},
    {5.e-15, 2.e6, 4. * std::pow(2.e6, 2. / 3.)}};
  Moments m;
  for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
    m[i] = N0 * sd.momFact[i];
    for (const auto& p : large) {
      m[i] += p[0] * std::pow(p[1], sc.MomOrderV[i]) *
              std::pow(p[2], sc.MomOrderS[i]);
    }
  }
  m[NUM_SOOT_MOMENTS] = N0;
  return m;
}

Moments
freshMoments(const SootData& sd)
{
  Moments m;
  sd.initialSmallMomVals(m.data());
  return m;
}

void
setCell(
  amrex::Array4<amrex::Real> const& q,
  const amrex::IntVect& iv,
  const amrex::Real T,
  const amrex::Real* Y,
  const Moments& mom_cgs)
{
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real rho;
  eos.PYT2R(P, Y, T, rho);
  q(iv, qRho) = rho;
  q(iv, qTemp) = T;
  for (int sp = 0; sp < NUM_SPECIES; ++sp) {
    q(iv, qSpec + sp) = Y[sp];
  }
  for (int m = 0; m < NMOM; ++m) {
    q(iv, qSoot + m) = mom_cgs[m];
  }
}

amrex::Box
stateBox()
{
  return {
    amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
    amrex::IntVect(AMREX_D_DECL(3, 1, 0))};
}

void
fillStates(const SootData& sd, amrex::FArrayBox& qfab)
{
  const auto Y = sootingMixture();
  const Moments fresh = freshMoments(sd);
  Moments bimodal = bimodalMoments(sd);
  sd.convertFromMol(bimodal.data());
  const amrex::Box& bx = qfab.box();
  const amrex::Real nx = bx.length(0);
  auto const& q = qfab.array();
  amrex::LoopOnCpu(bx, [&](int i, int j, int /*k*/) {
    const amrex::Real T = 1400. + 600. * i / (nx - 1.);
    setCell(
      q, amrex::IntVect(AMREX_D_DECL(i, j, 0)), T, Y.data(),
      (j == 0) ? fresh : bimodal);
  });
}

amrex::Box
rowBox(const int i0, const int i1)
{
  return {
    amrex::IntVect(AMREX_D_DECL(i0, 0, 0)),
    amrex::IntVect(AMREX_D_DECL(i1, 0, 0))};
}

Fabs::Fabs(const amrex::Box& bx)
  : q(bx, nQ, amrex::The_Managed_Arena()),
    mu(bx, 1, amrex::The_Managed_Arena()),
    s(bx, nS, amrex::The_Managed_Arena())
{
  mu.setVal<amrex::RunOn::Host>(MU);
}

void
sourceOn(
  const SootModel& sm,
  const amrex::FArrayBox& q,
  const amrex::FArrayBox& mu,
  amrex::FArrayBox& s,
  const amrex::Real dt,
  const bool pres_term,
  const bool zero)
{
  if (zero) {
    s.setVal<amrex::RunOn::Host>(0.);
  }
  sm.computeSootSourceTerm(
    q.box(), q.const_array(), mu.const_array(), s.array(), 0., dt, pres_term);
  amrex::Gpu::streamSynchronize();
}

std::pair<amrex::Real, amrex::Real>
massBalance(const amrex::Box& bx, amrex::Array4<const amrex::Real> const& s)
{
  const SootConst sc;
  amrex::Real err = 0.;
  amrex::Real ref = 0.;
  amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
    const amrex::Real ssoot = s(i, j, k, sSoot + 1) * sc.SootDensity;
    err = amrex::max(err, std::abs(s(i, j, k, sRho) + ssoot));
    ref = amrex::max(ref, std::abs(ssoot));
  });
  return {err, ref};
}

} // namespace soottest
