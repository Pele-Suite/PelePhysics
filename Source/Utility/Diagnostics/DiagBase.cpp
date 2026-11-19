#include "DiagBase.H"
#include "AMReX_ParmParse.H"
#include "AMReX_MultiFabUtil.H"
#ifdef AMREX_USE_EB
#include "AMReX_EB2.H"
#endif

#include <cmath>

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

amrex::iMultiFab
DiagBase::coveredMask(
  int a_lev,
  const amrex::BoxArray& a_grids,
  const amrex::DistributionMapping& a_dmap,
  const amrex::BoxArray* a_fineGrids,
  int a_nGrow) const
{
  const amrex::Geometry& geom = m_geoms[a_lev];
  amrex::iMultiFab mask;
  if (a_fineGrids != nullptr) {
    mask = amrex::makeFineMask(
      a_grids, a_dmap, amrex::IntVect(a_nGrow), *a_fineGrids, m_refRatio[a_lev],
      geom.periodicity(), 1, 0);
  } else {
    mask.define(a_grids, a_dmap, 1, a_nGrow);
    mask.setVal(1);
  }

#ifdef AMREX_USE_EB
  if (const auto* ebis = amrex::EB2::TopIndexSpaceIfPresent()) {
    amrex::MultiFab vfrac(a_grids, a_dmap, 1, a_nGrow);
    ebis->getLevel(geom).fillVolFrac(vfrac, geom);
    auto const& marrs = mask.arrays();
    auto const& vfarrs = vfrac.const_arrays();
    amrex::ParallelFor(
      mask, amrex::IntVect(a_nGrow),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        if (vfarrs[box_no](i, j, k) <= 0.0) {
          marrs[box_no](i, j, k) = 0;
        }
      });
    amrex::Gpu::streamSynchronize();
  }
#endif
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

amrex::iMultiFab
DiagBase::cellMask(
  int a_lev,
  const amrex::MultiFab& a_state,
  const amrex::MultiFab* a_fineState) const
{
  amrex::iMultiFab mask = coveredMask(
    a_lev, a_state.boxArray(), a_state.DistributionMap(),
    a_fineState != nullptr ? &a_fineState->boxArray() : nullptr, 0);
  applyFilters(mask, a_state);
  return mask;
}

amrex::MultiFab
DiagBase::fluidVolume(
  int a_lev,
  const amrex::BoxArray& a_grids,
  const amrex::DistributionMapping& a_dmap) const
{
  const amrex::Geometry& geom = m_geoms[a_lev];
  amrex::MultiFab volume(a_grids, a_dmap, 1, 0);
  geom.GetVolume(volume);
#ifdef AMREX_USE_EB
  if (const auto* ebis = amrex::EB2::TopIndexSpaceIfPresent()) {
    amrex::MultiFab vfrac(a_grids, a_dmap, 1, 0);
    ebis->getLevel(geom).fillVolFrac(vfrac, geom);
    amrex::MultiFab::Multiply(volume, vfrac, 0, 0, 1, 0);
  }
#endif
  return volume;
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
