#ifndef DIAGBASE_H
#define DIAGBASE_H

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
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
  // cannot be enclosed in protected member functions).

  // Mask of the cells of level a_lev to process, with a_nGrow ghost cells:
  // 0 where the cell is covered by level a_lev + 1 (a_fineGrids, nullptr on
  // the finest level) or by the EB, 1 elsewhere.
  amrex::iMultiFab coveredMask(
    int a_lev,
    const amrex::BoxArray& a_grids,
    const amrex::DistributionMapping& a_dmap,
    const amrex::BoxArray* a_fineGrids,
    int a_nGrow) const;

  // Zero a_mask where a filter rejects the cell, including in ghost cells.
  // Filter f is read from component a_firstFilterComp + f of a_data, or from
  // its state index when a_firstFilterComp < 0.
  void applyFilters(
    amrex::iMultiFab& a_mask,
    const amrex::MultiFab& a_data,
    int a_firstFilterComp = -1) const;

  // coveredMask and applyFilters on a state without ghost cells
  amrex::iMultiFab cellMask(
    int a_lev,
    const amrex::MultiFab& a_state,
    const amrex::MultiFab* a_fineState) const;

  // Fluid volume of the cells of level a_lev: the geometric volume (which
  // accounts for 2D-RZ) times the EB volume fraction.
  amrex::MultiFab fluidVolume(
    int a_lev,
    const amrex::BoxArray& a_grids,
    const amrex::DistributionMapping& a_dmap) const;

protected:
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

#endif
