// Helpers shared by the spray tests: gas states, droplets, a minimal AmrCore
// and particle container bookkeeping.

#include "SprayTest.H"

#include <AMReX_TagBox.H>

namespace spraytest {

namespace {

void
massFractions(const Real X_fuel, Real Y[NUM_SPECIES])
{
  Real X[NUM_SPECIES] = {0.};
  X[fuel] = X_fuel;
  X[O2_ID] = 0.21 * (1. - X_fuel);
  X[N2_ID] = 0.79 * (1. - X_fuel);
  pele::physics::PhysicsType::eos().X2Y(X, Y);
}

void
fillBox(
  const amrex::Box& bx,
  amrex::Array4<Real> const& a,
  const Gas& g,
  const Real dx,
  const Real xlo)
{
  Real Y[NUM_SPECIES];
  massFractions(g.X_fuel, Y);
  amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
    const Real T = g.T + g.dTdx * (xlo + (i + 0.5) * dx);
    Real rho, e;
    pele::physics::PhysicsType::eos().PYT2RE(g.p, Y, T, rho, e);
    a(i, j, k, cRho) = rho;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      a(i, j, k, cMom + dir) = rho * g.vel[dir];
    }
    a(i, j, k, cEng) = rho * (e + 0.5 * g.vel.radSquared());
    a(i, j, k, cTemp) = T;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      a(i, j, k, cRhoY + n) = rho * Y[n];
    }
  });
}
// Append host particles to a tile
void
appendToTile(
  SprayParticleContainer::ParticleTileType& tile,
  const amrex::Gpu::HostVector<PType>& h)
{
  const auto old_size = tile.GetArrayOfStructs().size();
  tile.resize(old_size + h.size());
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, h.begin(), h.end(),
    tile.GetArrayOfStructs().begin() + old_size);
}

} // namespace

amrex::FArrayBox
gasFab(const Gas& g)
{
  constexpr int ncell = 10;
  amrex::FArrayBox s(
    amrex::Box(
      amrex::IntVect(-1), amrex::IntVect(AMREX_D_DECL(ncell, ncell, ncell))),
    nComp, amrex::The_Managed_Arena());
  fillBox(s.box(), s.array(), g, fabDx, 0.);
  return s;
}

void
fillState(amrex::MultiFab& s, const Gas& g, const amrex::Geometry& geom)
{
  // Filled on the host, including the ghost cells
  for (amrex::MFIter mfi(s); mfi.isValid(); ++mfi) {
    const amrex::Box bx = mfi.fabbox();
    amrex::FArrayBox h(bx, nComp, amrex::The_Pinned_Arena());
    fillBox(bx, h.array(), g, geom.CellSize(0), geom.ProbLo(0));
    s[mfi].copy<amrex::RunOn::Device>(h);
  }
  amrex::Gpu::streamSynchronize();
}

PType
makeDrop(
  const RealVect& pos,
  const RealVect& vel,
  const Real T,
  const Real dia,
  const Real nppp)
{
  PType p;
  p.id() = PType::NextID();
  p.cpu() = amrex::ParallelDescriptor::MyProc();
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    p.pos(dir) = pos[dir];
  }
  for (int n = 0; n < SC::pstateNum; ++n) {
    p.rdata(n) = 0.;
  }
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    p.rdata(SC::pstateVel + dir) = vel[dir];
  }
  p.rdata(SC::pstateT) = T;
  p.rdata(SC::pstateDia) = dia;
  p.rdata(SC::pstateY) = 1.;
  p.rdata(SC::pstateNumDens) = nppp;
  p.rdata(SC::pstateN0) = nppp;
  return p;
}

Real
liqRho(const SprayData& fd, const Real T, Real cBoilT)
{
  const amrex::GpuArray<Real, SPRAY_FUEL_NUM> Y = {1.};
  return fd.liqprops.rho_mix(Y, T, &cBoilT);
}

// Mass of one droplet of the parcel
Real
dropMass(const SprayData& fd, const PType& p, const Real cBoilT)
{
  return M_PI / 6. * liqRho(fd, p.rdata(SC::pstateT), cBoilT) *
         std::pow(p.rdata(SC::pstateDia), 3);
}

RealVect
dropVel(const PType& p)
{
  return {AMREX_D_DECL(
    p.rdata(SC::pstateVel), p.rdata(SC::pstateVel + 1),
    p.rdata(SC::pstateVel + 2))};
}

// Liquid enthalpy (h_gas - latent heat) and kinetic energy of one droplet
Real
dropEnergy(const SprayData& fd, const PType& p, const Real cBoilT)
{
  const Real T = p.rdata(SC::pstateT);
  const Real h_liq = fd.liqprops.cp_i(T, 0) * (T - fd.liqprops.ref_T) -
                     fd.liqprops.latentRef_minus_gasRefH_i[0];
  return dropMass(fd, p, cBoilT) * (h_liq + 0.5 * dropVel(p).radSquared());
}

// Film mass, as a cylinder of the film diameter and height
Real
filmMass(const SprayData& fd, const PType& p, const Real cBoilT)
{
  const Real r = 0.5 * p.rdata(SC::pstateDia);
  return liqRho(fd, p.rdata(SC::pstateT), cBoilT) *
         p.rdata(SC::pstateFilmHght) * M_PI * r * r;
}

Real
boilT(const SprayData& fd, const Real p)
{
  GasPhaseVals gpv;
  gpv.p_fluid = p;
  Real cBoilT[SPRAY_FUEL_NUM];
  fd.calcBoilT(gpv, cBoilT);
  return cBoilT[0];
}

bool
finiteParticle(const PType& p)
{
  bool ok = true;
  for (int n = 0; n < SC::pstateNum; ++n) {
    ok = ok && std::isfinite(p.rdata(n));
  }
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    ok = ok && std::isfinite(p.pos(dir));
  }
  return ok;
}

void
setSprayData(const SprayData& sd)
{
  *SprayParticleContainer::m_sprayData = sd;
  amrex::Gpu::htod_memcpy(
    SprayParticleContainer::d_sprayData, &sd, sizeof(SprayData));
  amrex::Gpu::streamSynchronize();
}

int
deriveIndex(const std::string& name)
{
  const auto& names = SprayParticleContainer::DeriveVarNames();
  for (int n = 0; n < SprayParticleContainer::NumDeriveVars(); ++n) {
    if (names[n] == name) {
      return n;
    }
  }
  amrex::Abort("No derived spray variable " + name);
  return -1;
}

TestAmr::TestAmr(
  const amrex::Geometry& geom,
  const amrex::AmrInfo& info,
  const amrex::Box& fine_region)
  : amrex::AmrCore(geom, info), m_fine(fine_region)
{
}

void
TestAmr::MakeNewLevelFromScratch(
  int lev,
  Real /*time*/,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm)
{
  SetBoxArray(lev, ba);
  SetDistributionMap(lev, dm);
}

void
TestAmr::ClearLevel(int lev)
{
  ClearBoxArray(lev);
  ClearDistributionMap(lev);
}

void
TestAmr::ErrorEst(
  int lev, amrex::TagBoxArray& tags, Real /*time*/, int /*ngrow*/)
{
  if (lev > 0 || m_fine.isEmpty()) {
    return;
  }
  for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
    const amrex::Box bx = mfi.validbox() & m_fine;
    if (!bx.isEmpty()) {
      tags[mfi].setVal<amrex::RunOn::Device>(amrex::TagBox::SET, bx);
    }
  }
}

Real
Mesh::vol(const int lev) const
{
  const Real dx = amr->Geom(lev).CellSize(0);
  // Square cells; in 2D the spray assumes a depth of dx
  return dx * dx * dx;
}

Mesh
makeMesh(const MeshSpec& spec, const Gas& gas)
{
  Mesh m;
  const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(spec.ncell - 1));
  amrex::RealBox rb(
    spec.plo.begin(), (spec.plo + RealVect(spec.length)).begin());
  amrex::Array<int, AMREX_SPACEDIM> is_per;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    is_per[dir] = (spec.bc_lo[dir] == 0) ? 1 : 0;
  }
  const amrex::Geometry geom(domain, rb, 0, is_per);
  amrex::AmrInfo info;
  info.max_level = spec.max_level;
  const int nlev = spec.max_level + 1;
  info.ref_ratio = amrex::Vector<amrex::IntVect>(nlev, amrex::IntVect(2));
  info.max_grid_size =
    amrex::Vector<amrex::IntVect>(nlev, amrex::IntVect(spec.max_grid_size));
  info.blocking_factor = amrex::Vector<amrex::IntVect>(
    nlev, amrex::IntVect(std::min(8, spec.max_grid_size)));
  info.n_error_buf = amrex::Vector<amrex::IntVect>(nlev, amrex::IntVect(0));
  info.grid_eff = 1.;
  m.amr = std::make_unique<TestAmr>(geom, info, spec.fine_region);
  m.amr->InitFromScratch(0.);

  amrex::BCRec bc;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    bc.setLo(dir, spec.bc_lo[dir]);
    bc.setHi(dir, spec.bc_hi[dir]);
  }
  m.pc = std::make_unique<SprayParticleContainer>(m.amr.get(), bc);

  const int finest = m.amr->finestLevel();
  m.state_ghosts =
    SprayParticleContainer::getStateGhostCells(0, finest, 1, spec.cfl);
  m.source_ghosts =
    SprayParticleContainer::getSourceGhostCells(0, finest, 1, spec.cfl);
  const int nsrc = sSpec + SprayParticleContainer::m_sprayData->N_pc;
  for (int lev = 0; lev <= finest; ++lev) {
    const auto& ba = m.amr->boxArray(lev);
    const auto& dm = m.amr->DistributionMap(lev);
    m.state.emplace_back(ba, dm, nComp, m.state_ghosts);
    m.tmp.emplace_back(ba, dm, nsrc, m.source_ghosts);
    m.act.emplace_back(ba, dm, nComp, 0);
    fillState(m.state[lev], gas, m.amr->Geom(lev));
    m.tmp[lev].setVal(0.);
    m.act[lev].setVal(0.);
  }
  return m;
}

void
insertParticle(Mesh& m, const PType& p, const int lev)
{
  // The grid holding the particle's cell; no tiling
  const auto& geom = m.amr->Geom(lev);
  amrex::IntVect cell;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    cell[dir] = static_cast<int>(
      std::floor((p.pos(dir) - geom.ProbLo(dir)) * geom.InvCellSize(dir)));
  }
  const amrex::BoxArray& ba = m.amr->boxArray(lev);
  int grid = -1;
  for (int i = 0; i < static_cast<int>(ba.size()); ++i) {
    if (ba[i].contains(cell)) {
      grid = i;
    }
  }
  if (grid < 0) {
    amrex::Abort("insertParticle: particle outside the grids");
  }
  appendToTile(
    m.pc->GetParticles(lev)[std::make_pair(grid, 0)],
    amrex::Gpu::HostVector<PType>(1, p));
}

void
addParticles(Mesh& m, const std::vector<PType>& parts)
{
  // All into the tile of grid 0, and Redistribute sorts them out
  amrex::Gpu::HostVector<PType> h(parts.size());
  std::copy(parts.begin(), parts.end(), h.begin());
  appendToTile(m.pc->GetParticles(0)[std::make_pair(0, 0)], h);
  m.pc->Redistribute();
}

std::map<int, PType>
collect(const Mesh& m, std::map<int, int>* level)
{
  std::map<int, PType> parts;
  for (int lev = 0; lev <= m.pc->finestLevel(); ++lev) {
    for (MyParConstIter pti(*m.pc, lev); pti.isValid(); ++pti) {
      const auto& aos = pti.GetArrayOfStructs();
      amrex::Gpu::HostVector<PType> h(aos.size());
      amrex::Gpu::copy(
        amrex::Gpu::deviceToHost, aos.begin(), aos.end(), h.begin());
      for (const auto& p : h) {
        if (p.id() > 0) {
          parts[static_cast<int>(p.id())] = p;
          if (level != nullptr) {
            (*level)[static_cast<int>(p.id())] = lev;
          }
        }
      }
    }
  }
  return parts;
}

void
sprayStep(
  Mesh& m, const Env& env, const Real dt, const bool do_move, const Real cfl)
{
  for (int lev = 0; lev <= m.amr->finestLevel(); ++lev) {
    if (do_move) {
      m.pc->moveKickDrift(
        m.state[lev], m.tmp[lev], lev, dt, 0., false, false, m.state_ghosts,
        m.source_ghosts, true, env.tp, cfl);
    } else {
      m.pc->moveKick(
        m.state[lev], m.tmp[lev], lev, dt, 0., false, false, m.state_ghosts,
        m.source_ghosts, env.tp, cfl);
    }
    m.pc->transferSource(m.source_ghosts, lev, m.tmp[lev], m.act[lev]);
  }
}

Totals
totals(const std::map<int, PType>& parts, const Env& env)
{
  Totals t;
  for (const auto& kv : parts) {
    const PType& p = kv.second;
    if (p.rdata(SC::pstateFilmHght) > 0.) {
      t.film_mass += filmMass(env.base, p, env.cBoilT);
      ++t.films;
      continue;
    }
    const Real n = p.rdata(SC::pstateNumDens);
    const Real mass = n * dropMass(env.base, p, env.cBoilT);
    t.mass += mass;
    t.mom += mass * dropVel(p);
    t.energy += n * dropEnergy(env.base, p, env.cBoilT);
    ++t.count;
  }
  return t;
}

std::vector<Real>
sumSource(const Mesh& m, const amrex::Vector<amrex::MultiFab>& mf)
{
  const int nc = mf[0].nComp();
  std::vector<Real> sum(nc, 0.);
  for (int lev = 0; lev <= m.amr->finestLevel(); ++lev) {
    for (int n = 0; n < nc; ++n) {
      sum[n] += mf[lev].sum(n) * m.vol(lev);
    }
  }
  return sum;
}

} // namespace spraytest
