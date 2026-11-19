#ifndef DIAGISOSURFACE_H
#define DIAGISOSURFACE_H

#include "DiagBase.H"
#include <AMReX_Enum.H>

#include <vector>

AMREX_ENUM(IsoSurfFormat, None, XDMF, MEF);

// Extract an iso-surface (3D: triangles via marching cubes) or an iso-line
// (2D: segments via marching squares) of a cell-centered field across the
// AMR hierarchy. Ported from PeleAnalysis' isosurface tool.
//
// The marching is done on the dual grid: each "cube" has cell centers as
// corners, and is owned by the box holding, as a valid cell, the first of its
// corners that belongs to the level. Every cube is thus processed exactly
// once per level; cubes touching a fine-covered, out-of-domain, filtered-out
// or unfilled corner are skipped.
//
// Each node carries its coordinates plus the value of every requested field,
// linearly interpolated along the cut edge. The total area (3D) or length
// (2D) is appended to an ASCII file at every call, and the surface itself can
// optionally be written in XDMF (ParaView/VisIt) or MEF (PeleAnalysis) format.
class DiagIsosurface : public DiagBase::Register<DiagIsosurface>
{
public:
  static std::string identifier() { return "DiagIsosurface"; }

  void init(const std::string& a_prefix, std::string_view a_diagName) override;

  void prepare(
    int a_nlevels,
    const amrex::Vector<amrex::Geometry>& a_geoms,
    const amrex::Vector<amrex::BoxArray>& a_grids,
    const amrex::Vector<amrex::DistributionMapping>& a_dmap,
    const amrex::Vector<std::string>& a_varNames) override;

  void processDiag(
    int a_nstep,
    const amrex::Real& a_time,
    const amrex::Vector<const amrex::MultiFab*>& a_state,
    const amrex::Vector<std::string>& a_varNames) override;

  void addVars(amrex::Vector<std::string>& a_varList) override;

  void close() override {}

private:
  // Values stored per node: coordinates, then m_fieldNames
  int nNodeComp() const
  {
    return AMREX_SPACEDIM + static_cast<int>(m_fieldNames.size());
  }

  // Gather the rank-local surfaces on the IO rank, merge shared nodes and
  // write the result in the requested format.
  void collateAndWrite(
    const std::vector<amrex::Real>& a_nodes,
    const std::vector<int>& a_keys,
    const std::vector<int>& a_elts,
    int a_nstep,
    const amrex::Real& a_time) const;

  void writeXDMF(
    const std::string& a_base,
    const amrex::Real& a_time,
    const std::vector<amrex::Real>& a_nodes,
    const std::vector<int>& a_elts) const;

  void writeMEF(
    const std::string& a_base,
    const amrex::Real& a_time,
    const std::vector<amrex::Real>& a_nodes,
    const std::vector<int>& a_elts) const;

  void writeMeasure(
    int a_nstep,
    const amrex::Real& a_time,
    amrex::Real a_measure,
    amrex::Long a_nElts);

  std::string outputBase(int a_nstep, const amrex::Real& a_time) const;

  // Iso-surface definition
  std::string m_isoFieldName;
  amrex::Real m_isoValue{0.0};

  // Fields interpolated onto the surface; m_fieldNames[0] is the iso field
  amrex::Vector<std::string> m_fieldNames;

  IsoSurfFormat m_format{IsoSurfFormat::XDMF};
  bool m_writeMeasure{true};
  bool m_measureHeaderDone{false};

  amrex::Vector<amrex::Geometry> m_geoms;
  amrex::Vector<amrex::IntVect> m_refRatio;
};

#endif
