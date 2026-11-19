#include "DiagBase.H"
#include "AMReX_ParmParse.H"
#include "AMReX_MultiFabUtil.H"
#ifdef AMREX_USE_EB
#include "AMReX_EB2.H"
#endif

#include <cmath>
#include <limits>

void
DiagBase::init(const std::string& a_prefix, std::string_view a_diagName)
{
  amrex::ParmParse pp(a_prefix);

  // IO
  pp.query("int", m_interval);
  pp.query("per", m_per);
  m_diagfile = a_diagName;
  pp.query("file", m_diagfile);
  AMREX_ASSERT(m_interval > 0 || m_per > 0.0);

  // Filters
  int nFilters = 0;
  nFilters = pp.countval("filters");
  amrex::Vector<std::string> filtersName;
  if (nFilters > 0) {
    m_filters.resize(nFilters);
    filtersName.resize(nFilters);
  }
  for (int n = 0; n < nFilters; ++n) {
    pp.get("filters", filtersName[n], n);
    const std::string filter_prefix = a_prefix + "." + filtersName[n];
    m_filters[n].init(filter_prefix);
  }
}

void
DiagBase::prepare(
  int a_nlevels,
  const amrex::Vector<amrex::Geometry>& a_geoms,
  const amrex::Vector<amrex::BoxArray>& /*a_grids*/,
  const amrex::Vector<amrex::DistributionMapping>& /*a_dmap*/,
  const amrex::Vector<std::string>& a_varNames)
{
  if (first_time) {
    int nFilters = static_cast<int>(m_filters.size());
    // Move the filter data to the device
    for (int n = 0; n < nFilters; ++n) {
      m_filters[n].setup(a_varNames);
    }
    amrex::Vector<DiagFilterData> hostFilterData;
    for (int n = 0; n < nFilters; ++n) {
      hostFilterData.push_back(m_filters[n].m_fdata);
    }
    m_filterData.resize(nFilters);
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, hostFilterData.begin(), hostFilterData.end(),
      m_filterData.begin());
  }

  m_geoms.resize(a_nlevels);
  m_refRatio.resize(a_nlevels - 1);
  for (int lev = 0; lev < a_nlevels; lev++) {
    m_geoms[lev] = a_geoms[lev];
    if (lev > 0) {
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        m_refRatio[lev - 1][d] = static_cast<int>(
          std::lround(a_geoms[lev - 1].CellSize(d) / a_geoms[lev].CellSize(d)));
      }
    }
  }
}

namespace {
// Fill a_vfrac (and its ghost cells) with the EB volume fraction, or with 1
// without EB. Returns whether an EB is present.
bool
fillVolumeFraction(const amrex::Geometry& a_geom, amrex::MultiFab& a_vfrac)
{
#ifdef AMREX_USE_EB
  if (const auto* ebis = amrex::EB2::TopIndexSpaceIfPresent()) {
    ebis->getLevel(a_geom).fillVolFrac(a_vfrac, a_geom);
    return true;
  }
#else
  amrex::ignore_unused(a_geom);
#endif
  a_vfrac.setVal(1.0);
  return false;
}
} // namespace

void
DiagBase::checkLevel(int a_lev) const
{
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    a_lev >= 0 && a_lev < static_cast<int>(m_geoms.size()),
    "Diagnostics: level not set up by DiagBase::prepare");
}

amrex::iMultiFab
DiagBase::coveredMask(
  int a_lev,
  const amrex::BoxArray& a_grids,
  const amrex::DistributionMapping& a_dmap,
  const amrex::BoxArray* a_fineGrids,
  int a_nGrow,
  const amrex::MultiFab* a_vfrac) const
{
  checkLevel(a_lev);
  const amrex::Geometry& geom = m_geoms[a_lev];
  amrex::iMultiFab mask;
  if (a_fineGrids != nullptr) {
    checkLevel(a_lev + 1);
    mask = amrex::makeFineMask(
      a_grids, a_dmap, amrex::IntVect(a_nGrow), *a_fineGrids, m_refRatio[a_lev],
      geom.periodicity(), 1, 0);
  } else {
    mask.define(a_grids, a_dmap, 1, a_nGrow);
    mask.setVal(1);
  }

  amrex::MultiFab vfracLocal;
  if (a_vfrac == nullptr) {
    vfracLocal.define(a_grids, a_dmap, 1, a_nGrow);
    if (!fillVolumeFraction(geom, vfracLocal)) {
      return mask;
    }
    a_vfrac = &vfracLocal;
  }
  AMREX_ALWAYS_ASSERT(a_vfrac->nGrowVect().allGE(amrex::IntVect(a_nGrow)));
  auto const& marrs = mask.arrays();
  auto const& vfarrs = a_vfrac->const_arrays();
  amrex::ParallelFor(
    mask, amrex::IntVect(a_nGrow),
    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      if (vfarrs[box_no](i, j, k) <= 0.0) {
        marrs[box_no](i, j, k) = 0;
      }
    });
  amrex::Gpu::streamSynchronize();
  return mask;
}

void
DiagBase::applyFilters(
  amrex::iMultiFab& a_mask,
  const amrex::MultiFab& a_data,
  int a_firstFilterComp) const
{
  const int nFilters = static_cast<int>(m_filters.size());
  if (nFilters == 0) {
    return;
  }
  AMREX_ALWAYS_ASSERT(a_data.nGrowVect().allGE(a_mask.nGrowVect()));
  AMREX_ALWAYS_ASSERT(
    a_firstFilterComp < 0 || a_firstFilterComp + nFilters <= a_data.nComp());
  auto const& marrs = a_mask.arrays();
  auto const& darrs = a_data.const_arrays();
  const auto* fdata_p = m_filterData.data();
  amrex::ParallelFor(
    a_mask, a_mask.nGrowVect(),
    [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
      for (int f = 0; f < nFilters; ++f) {
        const int comp = a_firstFilterComp >= 0 ? a_firstFilterComp + f
                                                : fdata_p[f].m_filterVarIdx;
        const amrex::Real v = darrs[box_no](i, j, k, comp);
        if (v < fdata_p[f].m_low_val || v > fdata_p[f].m_high_val) {
          marrs[box_no](i, j, k) = 0;
        }
      }
    });
  amrex::Gpu::streamSynchronize();
}

amrex::Vector<DiagBase::LevelCells>
DiagBase::levelCells(const amrex::Vector<const amrex::MultiFab*>& a_state) const
{
  const int nlev = static_cast<int>(a_state.size());
  amrex::Vector<LevelCells> cells(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    checkLevel(lev);
    const amrex::MultiFab& state = *a_state[lev];
    const amrex::BoxArray& ba = state.boxArray();
    const amrex::DistributionMapping& dm = state.DistributionMap();
    amrex::MultiFab vfrac(ba, dm, 1, 0);
    const bool hasEB = fillVolumeFraction(m_geoms[lev], vfrac);

    cells[lev].mask = coveredMask(
      lev, ba, dm, lev + 1 < nlev ? &a_state[lev + 1]->boxArray() : nullptr, 0,
      &vfrac);
    applyFilters(cells[lev].mask, state);

    // Geometric volume (accounts for 2D-RZ) times the volume fraction
    cells[lev].volume.define(ba, dm, 1, 0);
    m_geoms[lev].GetVolume(cells[lev].volume);
    if (hasEB) {
      amrex::MultiFab::Multiply(cells[lev].volume, vfrac, 0, 0, 1, 0);
    }
  }
  return cells;
}

std::pair<amrex::Real, amrex::Real>
DiagBase::maskedMinMax(
  const amrex::Vector<const amrex::MultiFab*>& a_state,
  const amrex::Vector<LevelCells>& a_cells,
  int a_comp)
{
  amrex::Real vmin = std::numeric_limits<amrex::Real>::max();
  amrex::Real vmax = std::numeric_limits<amrex::Real>::lowest();
  for (int lev = 0; lev < static_cast<int>(a_state.size()); ++lev) {
    auto const& sarrs = a_state[lev]->const_arrays();
    auto const& marrs = a_cells[lev].mask.const_arrays();
    const auto mm = amrex::ParReduce(
      amrex::TypeList<amrex::ReduceOpMin, amrex::ReduceOpMax>{},
      amrex::TypeList<amrex::Real, amrex::Real>{}, *a_state[lev],
      amrex::IntVect(0),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept
        -> amrex::GpuTuple<amrex::Real, amrex::Real> {
        if (marrs[box_no](i, j, k) == 0) {
          return {
            std::numeric_limits<amrex::Real>::max(),
            std::numeric_limits<amrex::Real>::lowest()};
        }
        const amrex::Real v = sarrs[box_no](i, j, k, a_comp);
        return {v, v};
      });
    vmin = amrex::min(vmin, amrex::get<0>(mm));
    vmax = amrex::max(vmax, amrex::get<1>(mm));
  }
  amrex::ParallelDescriptor::ReduceRealMin(vmin);
  amrex::ParallelDescriptor::ReduceRealMax(vmax);
  return {vmin, vmax};
}

bool
DiagBase::doDiag(const amrex::Real& a_time, int a_nstep)
{
  bool willDo = false;
  if (m_interval > 0 && (a_nstep % m_interval == 0)) {
    willDo = true;
  }

  // TODO: using a_time
  amrex::ignore_unused(a_time);

  return willDo;
}

void
DiagBase::addVars(amrex::Vector<std::string>& a_varList)
{
  int nFilters = static_cast<int>(m_filters.size());
  for (int n = 0; n < nFilters; ++n) {
    a_varList.push_back(m_filters[n].m_filterVar);
  }
}

int
DiagBase::getFieldIndex(
  const std::string& a_field, const amrex::Vector<std::string>& a_varList)
{
  int index = -1;
  for (int n{0}; n < a_varList.size(); ++n) {
    if (a_varList[n] == a_field) {
      index = n;
      break;
    }
  }
  if (index < 0) {
    amrex::Abort("Field : " + a_field + " wasn't found in available fields");
  }
  return index;
}
