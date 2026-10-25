#ifndef DIAGPROBE_H
#define DIAGPROBE_H

#include "DiagBase.H"
#include <AMReX_VisMF.H>

class DiagProbe : public DiagBase::Register<DiagProbe>
{
public:
  static std::string identifier() { return "DiagProbe"; }

  enum InterpType { CellCenter, Linear };

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
  amrex::Vector<std::string> m_fieldNames;
  amrex::Vector<int> m_fieldIndices;
  amrex::Vector<amrex::Real> m_values_at_probe;
  amrex::Gpu::DeviceVector<int> m_fieldIndices_d;
  amrex::Gpu::DeviceVector<amrex::Real> m_values_at_probe_d;

  int m_finest_level_probe; // Finest level at which probe is located
  int m_box_probe_num; // The box number at lev=m_finest_level_probe in which
  int m_probe_idx[AMREX_SPACEDIM]; // The cell index of the cell in which probe

  // Probe definition
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
    m_probe_loc; // probe location given by the user
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
    dx_finest_lev_probe; // grid size at lev=m_finest_level_probe
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
    x_low_cell; // lowerst corner of the cell in which probe is located
  amrex::GpuArray<int, AMREX_SPACEDIM> low_cell_idx;

  // Interpolation data
  InterpType m_interpType;
  amrex::Array3D<amrex::Real, 0, 1, 0, 1, 0, 1> cell_data{
    0.0}; // Neighbour cell solution values

  // Probe file name
  std::ofstream tmpProbeFile; // Ofstream object for writing probe file data
  std::string
    tmpProbeFileName; // File name in which probe data is written. Hard coded to

  // 2D-plane boxArray vector
  amrex::Vector<amrex::BoxArray> m_probebox;     // probe box array (single box)
  amrex::Vector<amrex::Vector<int>> m_dmConvert; // Distribution mapping
  amrex::Vector<amrex::DistributionMapping> m_probeboxDM; // Distribution
};

#endif
