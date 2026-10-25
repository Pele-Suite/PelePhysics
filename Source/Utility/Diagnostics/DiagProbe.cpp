#include "DiagProbe.H"
#include <AMReX_VisMF.H>
#include <AMReX_FPC.H>
#include <AMReX_PlotFileUtil.H>
#include <regex>
#include <cstdio>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
LinearInterpolate(
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> xp,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> x_low,
  amrex::Array3D<amrex::Real, 0, 1, 0, 1, 0, 1> neighbour_cells,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx)
{

  amrex::Real value = 0.0;
  amrex::Real slp[3] = {0.0};

  for (int cnt = 0; cnt < AMREX_SPACEDIM; cnt++) {
    slp[cnt] = (xp[cnt] - x_low[cnt]) / dx[cnt];
  }

  value +=
    (1.0 - slp[0]) * (1 - slp[1]) * (1 - slp[2]) * neighbour_cells(0, 0, 0);
  value += slp[0] * (1 - slp[1]) * (1 - slp[2]) * neighbour_cells(0 + 1, 0, 0);
  value +=
    (1.0 - slp[0]) * slp[1] * (1 - slp[2]) * neighbour_cells(0, 0 + 1, 0);
  value += slp[0] * slp[1] * (1 - slp[2]) * neighbour_cells(0 + 1, 0 + 1, 0);

  value +=
    (1.0 - slp[0]) * (1 - slp[1]) * slp[2] * neighbour_cells(0, 0, 0 + 1);
  value += slp[0] * (1 - slp[1]) * slp[2] * neighbour_cells(0 + 1, 0, 0 + 1);
  value += (1.0 - slp[0]) * slp[1] * slp[2] * neighbour_cells(0, 0 + 1, 0 + 1);
  value += slp[0] * slp[1] * slp[2] * neighbour_cells(0 + 1, 0 + 1, 0 + 1);
  return (value);
}

void
DiagProbe::init(const std::string& a_prefix, std::string_view a_diagName)
{
  DiagBase::init(a_prefix, a_diagName);

#ifdef AMREX_USE_EB
  amrex::Abort("\nProbe implementation not yet done for EBs!\n");
#endif

  // Warn about filters
  if (m_filters.empty()) {
    amrex::Print() << " Filters are not available on DiagFrameProbe and will "
                      "be discarded \n";
  }

  // read from inputs file
  amrex::ParmParse pp(a_prefix);

  // Read probe location
  amrex::Vector<amrex::Real> probe_loc;
  pp.getarr("probe_location", probe_loc, 0, pp.countval("probe_location"));
  if (probe_loc.size() >= AMREX_SPACEDIM) {
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
      m_probe_loc[idim] = probe_loc[idim];
    }
  } else {
    amrex::Abort("\nProvide probe location array with same dimension as "
                 "problem dimension");
  }

  // Read field names and initialize probe variables to zero
  int nOutFields = pp.countval("field_names");
  AMREX_ASSERT(nOutFields > 0);
  m_values_at_probe.resize(nOutFields);
  m_fieldNames.resize(nOutFields);
  m_fieldIndices.resize(nOutFields);
  m_fieldIndices_d.resize(nOutFields);
  m_values_at_probe_d.resize(nOutFields);

  for (int f{0}; f < nOutFields; ++f) {
    pp.get("field_names", m_fieldNames[f], f);
    m_values_at_probe[f] = 0.0;
  }

  // Read interpolation type. Check interpolation type values
  std::string intType = "CellCenter";
  pp.query("interpolation", intType);
  if (intType == "Linear") {
    m_interpType = Linear;
  } else if (intType == "CellCenter") {
    m_interpType = CellCenter;
  } else {
    amrex::Abort(
      "Unknown interpolation type for " + a_prefix +
      ". Allowed values are Linear or CellCenter.\n");
  }

  // Set output file properties
  // Output files to temporals directory (hardcoded)
  amrex::UtilCreateDirectory("temporals", 0755);
  std::string tmpProbeFileName =
    "temporals/Probe_" + std::string(a_diagName) + ".out";
  if (amrex::ParallelDescriptor::IOProcessor()) {
    tmpProbeFile.open(
      tmpProbeFileName.c_str(),
      std::ios::out | std::ios::app | std::ios_base::binary);
    tmpProbeFile.precision(12);
  }
}

void
DiagProbe::addVars(amrex::Vector<std::string>& a_varList)
{
  DiagBase::addVars(a_varList);
  for (const auto& v : m_fieldNames) {
    a_varList.push_back(v);
  }
}

void
DiagProbe::prepare(
  int a_nlevels,
  const amrex::Vector<amrex::Geometry>& a_geoms,
  const amrex::Vector<amrex::BoxArray>& a_grids,
  const amrex::Vector<amrex::DistributionMapping>& a_dmap,
  const amrex::Vector<std::string>& a_varNames)
{
  // Check if probe lies within the geometry
  if (!(a_geoms[0].insideRoundoffDomain(
        AMREX_D_DECL(m_probe_loc[0], m_probe_loc[1], m_probe_loc[2])))) {
    amrex::Abort("\nProbe " + m_diagfile + " lies outside problem domain");
  }

  // If first time (including restart), write the file header.
  if (first_time) {
    tmpProbeFile << "time,iter";
    int nOutFields = static_cast<int>(m_fieldIndices.size());
    for (int f{0}; f < nOutFields; ++f) {
      m_fieldIndices[f] = getFieldIndex(m_fieldNames[f], a_varNames);
      m_values_at_probe[f] = 0.0;
      tmpProbeFile << "," << m_fieldNames[f];
    }
    tmpProbeFile << "\n";
    tmpProbeFile.flush();

    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, m_fieldIndices.begin(), m_fieldIndices.end(),
      m_fieldIndices_d.begin());
    first_time = false;
  }

  // Search for finest level and location of the probe in index space.
  bool probe_found = false;

  for (int lev = a_nlevels - 1; lev >= 0; lev--) {
    const amrex::Real* dx = a_geoms[lev].CellSize();
    const amrex::Real* problo = a_geoms[lev].ProbLo();
    amrex::Real dist[AMREX_SPACEDIM];

    // Calculate distance of probe from the domain low values
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
      dist[idim] = (m_probe_loc[idim] - (problo[idim])) / dx[idim];
    }

    // index of the cell where probe is located.
    amrex::IntVect idx_lev(AMREX_D_DECL(
      static_cast<int>(dist[0]), static_cast<int>(dist[1]),
      static_cast<int>(dist[2])));

    // loop through all boxes in lev to find which box the cell identified above
    // is located.
    for (int i = 0; i < a_grids[lev].size(); i++) {
      auto cBox = a_grids[lev][i];
      if (cBox.contains(idx_lev) && !probe_found) {
        // box found. store the level and  box number. set
        // probe_found to true to stop searching any further
        m_finest_level_probe = lev;
        m_box_probe_num = i;
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
          // Store grid size, cell corner and probe index
          dx_finest_lev_probe[idim] = dx[idim];
          m_probe_idx[idim] = idx_lev[idim];
          x_low_cell[idim] =
            std::floor(
              (m_probe_loc[idim] - (problo[idim] + dx[idim] * 0.5)) /
              dx[idim]) *
              dx[idim] +
            (problo[idim] + dx[idim] * 0.5);
          low_cell_idx[idim] =
            static_cast<int>((x_low_cell[idim] - (problo[idim])) / dx[idim]);
        }
        probe_found = true;
      }
    }
  }

  if (!probe_found) {
    amrex::Abort(
      "\nUnable to find the probe location. There seems to be something wrong");
  }

  // We are storing only the single box which contains the probe
  m_probebox.resize(1);
  m_probeboxDM.resize(1);
  m_dmConvert.resize(1);

  amrex::Vector<int> pmap;
  amrex::BoxList bl(a_grids[m_finest_level_probe].ixType());
  bl.reserve(1);
  amrex::Vector<int> dmConvertLev;
  auto cBox = a_grids[m_finest_level_probe][m_box_probe_num];
  bl.push_back(cBox);
  pmap.push_back(a_dmap[m_finest_level_probe][m_box_probe_num]);
  dmConvertLev.push_back(m_box_probe_num);
  m_dmConvert[0] = dmConvertLev;
  m_probebox[0].define(bl);
  m_probeboxDM[0].define(pmap);
}

void
DiagProbe::processDiag(
  int a_nstep,
  const amrex::Real& a_time,
  const amrex::Vector<const amrex::MultiFab*>& a_state,
  const amrex::Vector<std::string>& a_varNames)
{
  // since we are only taking the single box, just create a single multifab
  amrex::Vector<amrex::MultiFab> planeData(1);
  planeData[0].define(
    m_probebox[0], m_probeboxDM[0], static_cast<int>(m_fieldNames.size()), 0);
  m_values_at_probe.resize(m_fieldIndices.size());
  std::fill(m_values_at_probe.begin(), m_values_at_probe.end(), 0.0);
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, m_values_at_probe.begin(),
    m_values_at_probe.end(), m_values_at_probe_d.begin());

  for (amrex::MFIter mfi(planeData[0], amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const auto& bx = mfi.tilebox();
    const int state_idx = m_dmConvert[0][mfi.index()];
    auto const& state =
      a_state[m_finest_level_probe]->const_array(state_idx, 0);
    auto const& plane = planeData[0].array(mfi);
    auto* idx_d_p = m_fieldIndices_d.dataPtr();
    amrex::Real* tmp_values_d = m_values_at_probe_d.data();
    auto const& m_probe_idx_d = m_probe_idx;
    auto const& low_cell_idx_d = low_cell_idx;

    auto& m_probe_loc_d = m_probe_loc;
    auto& x_low_cell_d = x_low_cell;
    auto& dx_finest_lev_probe_d = dx_finest_lev_probe;
    auto& m_interpType_d = m_interpType;

    amrex::ParallelFor(
      m_fieldIndices_d.size(), [=] AMREX_GPU_DEVICE(int n) noexcept {
        amrex::Array3D<amrex::Real, 0, 1, 0, 1, 0, 1> neighbour_cells{
          0.0}; // Neighbour cell solution values
        int stIdx = idx_d_p[n];

        if (m_interpType_d == Linear) {

#if (AMREX_SPACEDIM == 1)
          neighbour_cells(0, 0, 0) = state(low_cell_idx_d[0], 0, 0, stIdx);
          neighbour_cells(1, 0, 0) = state(low_cell_idx_d[0] + 1, 0, 0, stIdx);
#elif (AMREX_SPACEDIM == 2)
          neighbour_cells(0, 0, 0) =
            state(low_cell_idx_d[0], low_cell_idx_d[1], 0, stIdx);
          neighbour_cells(1, 0, 0) =
            state(low_cell_idx_d[0] + 1, low_cell_idx_d[1], 0, stIdx);
          neighbour_cells(0, 1, 0) =
            state(low_cell_idx_d[0], low_cell_idx_d[1] + 1, 0, stIdx);
          neighbour_cells(1, 1, 0) =
            state(low_cell_idx_d[0] + 1, low_cell_idx_d[1] + 1, 0, stIdx);
#else
          neighbour_cells(0, 0, 0) = state(
            low_cell_idx_d[0], low_cell_idx_d[1], low_cell_idx_d[2], stIdx);
          neighbour_cells(1, 0, 0) = state(
            low_cell_idx_d[0] + 1, low_cell_idx_d[1], low_cell_idx_d[2], stIdx);
          neighbour_cells(0, 1, 0) = state(
            low_cell_idx_d[0], low_cell_idx_d[1] + 1, low_cell_idx_d[2], stIdx);
          neighbour_cells(1, 1, 0) = state(
            low_cell_idx_d[0] + 1, low_cell_idx_d[1] + 1, low_cell_idx_d[2],
            stIdx);

          neighbour_cells(0, 0, 1) = state(
            low_cell_idx_d[0], low_cell_idx_d[1], low_cell_idx_d[2] + 1, stIdx);
          neighbour_cells(1, 0, 1) = state(
            low_cell_idx_d[0] + 1, low_cell_idx_d[1], low_cell_idx_d[2] + 1,
            stIdx);
          neighbour_cells(0, 1, 1) = state(
            low_cell_idx_d[0], low_cell_idx_d[1] + 1, low_cell_idx_d[2] + 1,
            stIdx);
          neighbour_cells(1, 1, 1) = state(
            low_cell_idx_d[0] + 1, low_cell_idx_d[1] + 1, low_cell_idx_d[2] + 1,
            stIdx);
#endif
          tmp_values_d[n] = LinearInterpolate(
            m_probe_loc_d, x_low_cell_d, neighbour_cells,
            dx_finest_lev_probe_d);
        } else if (m_interpType_d == CellCenter) {
#if (AMREX_SPACEDIM == 1)
          tmp_values_d[n] = state(m_probe_idx_d[0], 0, 0, stIdx);
#elif (AMREX_SPACEDIM == 2)
          tmp_values_d[n] = state(m_probe_idx_d[0], m_probe_idx_d[1], 0, stIdx);
#else
          tmp_values_d[n] =
            state(m_probe_idx_d[0], m_probe_idx_d[1], m_probe_idx_d[2], stIdx);
#endif
        }
      });
  }

  amrex::ParallelDescriptor::ReduceRealSum(
    m_values_at_probe_d.data(), static_cast<int>(m_values_at_probe_d.size()));

  amrex::Gpu::copy(
    amrex::Gpu::deviceToHost, m_values_at_probe_d.begin(),
    m_values_at_probe_d.end(), m_values_at_probe.begin());

  // Write probe values to file
  if (amrex::ParallelDescriptor::IOProcessor()) {
    tmpProbeFile << a_time << "," << a_nstep;
    for (int f{0}; f < m_values_at_probe.size(); ++f) {
      tmpProbeFile << "," << m_values_at_probe[f];
    }
    tmpProbeFile << "\n";
    tmpProbeFile.flush();
  }
}
