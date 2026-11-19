#ifndef DIAGBASE_H
#define DIAGBASE_H

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

#include <utility>
#include "DiagFilter.H"
#include "Factory.H"

class DiagBase : public pele::physics::Factory<DiagBase>
{
public:
  static std::string base_identifier() { return "DiagBase"; }

  virtual void init(const std::string& a_prefix, std::string_view a_diagName);

  virtual void close() = 0;

  bool needUpdate() const { return need_update; }

  virtual bool doDiag(const amrex::Real& a_time, int a_nstep);

  virtual void prepare(
    int a_nlevels,
    const amrex::Vector<amrex::Geometry>& a_geoms,
    const amrex::Vector<amrex::BoxArray>& a_grids,
    const amrex::Vector<amrex::DistributionMapping>& a_dmap,
    const amrex::Vector<std::string>& a_varNames);

  virtual void processDiag(
    int a_nstep,
    const amrex::Real& a_time,
    const amrex::Vector<const amrex::MultiFab*>& a_state,
    const amrex::Vector<std::string>& a_varNames) = 0;

  virtual void addVars(amrex::Vector<std::string>& a_varList);

  static int getFieldIndex(
    const std::string& a_field, const amrex::Vector<std::string>& a_varList);

  // Helpers for the derived diagnostics (public: CUDA extended lambdas
  // cannot be enclosed in protected member functions). They need the
  // geometries set by DiagBase::prepare.

  // Mask of the cells of level a_lev to process, with a_nGrow ghost cells:
  // 0 where the cell is covered by level a_lev + 1 (a_fineGrids, nullptr on
  // the finest level) or by the EB, 1 elsewhere. a_vfrac, if given, holds
  // the volume fraction with at least a_nGrow ghost cells.
  amrex::iMultiFab coveredMask(
    int a_lev,
    const amrex::BoxArray& a_grids,
    const amrex::DistributionMapping& a_dmap,
    const amrex::BoxArray* a_fineGrids,
    int a_nGrow,
    const amrex::MultiFab* a_vfrac = nullptr) const;

  // Zero a_mask where a filter rejects the cell, including in ghost cells.
  // Filter f is read from component a_firstFilterComp + f of a_data, or from
  // its state index when a_firstFilterComp < 0.
  void applyFilters(
    amrex::iMultiFab& a_mask,
    const amrex::MultiFab& a_data,
    int a_firstFilterComp = -1) const;

  // Per level, the cells to process (coveredMask and applyFilters) and their
  // fluid volume: the geometric volume (which accounts for 2D-RZ) times the
  // EB volume fraction.
  struct LevelCells
  {
    amrex::iMultiFab mask;
    amrex::MultiFab volume;
  };
  amrex::Vector<LevelCells>
  levelCells(const amrex::Vector<const amrex::MultiFab*>& a_state) const;

  // Min and max of component a_comp over the cells kept by the masks
  static std::pair<amrex::Real, amrex::Real> maskedMinMax(
    const amrex::Vector<const amrex::MultiFab*>& a_state,
    const amrex::Vector<LevelCells>& a_cells,
    int a_comp);

protected:
  // Output name for the current call: the file prefix with the step number
  // (int) or the time (per)
  std::string outputName(int a_nstep, amrex::Real a_time) const;

  // Indices in a_varList of the fields a_names, on the device
  static amrex::Gpu::DeviceVector<int> fieldIndices(
    const amrex::Vector<std::string>& a_names,
    const amrex::Vector<std::string>& a_varList);

  void checkLevel(int a_lev) const;

  std::string m_diagfile;
  int m_verbose{0};
  amrex::Real m_per{-1.0};
  int m_interval{-1};
  bool need_update{true};
  bool first_time{true};
  amrex::Vector<DiagFilter> m_filters{};
  amrex::Gpu::DeviceVector<DiagFilterData> m_filterData;
  // Set by prepare
  amrex::Vector<amrex::Geometry> m_geoms;
  amrex::Vector<amrex::IntVect> m_refRatio;
};

// Bins of a PDF or of conditional statistics: nBins bins of equal width over
// [low, high], given by the `range` option or, by default, by the min/max of
// the binned variable over the processed cells (fromData).
struct DiagBins
{
  int nBins{-1};
  amrex::Real low{0.0};
  amrex::Real high{0.0};
  bool fromData{true};

  // Read the nBins and range options
  void init(const amrex::ParmParse& a_pp);

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real width() const noexcept
  {
    return (high - low) / nBins;
  }

  amrex::Real center(int a_n) const noexcept
  {
    return low + (a_n + 0.5) * width();
  }

  // Bin of a_v (the upper bound belongs to the last bin), -1 when a_v is
  // outside the range
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
  index(amrex::Real a_v) const noexcept
  {
    const amrex::Real x = (a_v - low) / width();
    if (!(x >= 0.0 && x <= static_cast<amrex::Real>(nBins))) {
      return -1;
    }
    return amrex::min(static_cast<int>(x), nBins - 1);
  }
};

#endif
