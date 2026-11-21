#include "DiagConditional.H"

#include <tuple>

void
DiagConditional::init(const std::string& a_prefix, std::string_view a_diagName)
{
  DiagBase::init(a_prefix, a_diagName);

  amrex::ParmParse pp(a_prefix);

  std::string condType;
  pp.get("conditional_type", condType);
  if (condType == "Average") {
    m_condType = Average;
  } else if (condType == "Integral") {
    m_condType = Integral;
  } else if (condType == "Sum") {
    m_condType = Sum;
  } else {
    amrex::Abort("Unknown conditional_type: " + condType);
  }
  m_bins.init(pp);
  pp.get("condition_field_name", m_cFieldName);
  pp.getarr("field_names", m_fieldNames);
  AMREX_ASSERT(!m_fieldNames.empty());
}

void
DiagConditional::addVars(amrex::Vector<std::string>& a_varList)
{
  DiagBase::addVars(a_varList);
  a_varList.push_back(m_cFieldName);
  for (const auto& v : m_fieldNames) {
    a_varList.push_back(v);
  }
}

void
DiagConditional::processDiag(
  int a_nstep,
  const amrex::Real& a_time,
  const amrex::Vector<const amrex::MultiFab*>& a_state,
  const amrex::Vector<std::string>& a_stateVar)
{
  // Cells to process (not covered by a finer level or the EB, passing the
  // filters) and their fluid volume
  const int cFieldIdx = getFieldIndex(m_cFieldName, a_stateVar);
  const auto fieldIdx_d = fieldIndices(m_fieldNames, a_stateVar);
  const auto cells = levelCells(a_state);
  if (m_bins.fromData) {
    std::tie(m_bins.low, m_bins.high) = maskedMinMax(a_state, cells, cFieldIdx);
  }

  // Per bin, in one buffer: the weighted sums of the fields (cond) and of
  // their squares (condSq), of the condition field (condAbs) and of the
  // weights (condVol). The weight is the fluid volume, or 1 for Sum.
  const int nBins = m_bins.nBins;
  const int nFields = static_cast<int>(m_fieldNames.size());
  const int condOff = 0;
  const int condSqOff = nFields * nBins;
  const int condAbsOff = 2 * nFields * nBins;
  const int condVolOff = condAbsOff + nBins;
  amrex::Gpu::DeviceVector<amrex::Real> data_d(condVolOff + nBins, 0.0);
  auto* data_p = data_d.dataPtr();
  const auto* idx_p = fieldIdx_d.dataPtr();
  const bool weighted = m_condType != Sum;
  const bool average = m_condType == Average;

  for (int lev = 0; lev < a_state.size(); ++lev) {
    auto const& sarrs = a_state[lev]->const_arrays();
    auto const& marrs = cells[lev].mask.const_arrays();
    auto const& varrs = cells[lev].volume.const_arrays();
    amrex::ParallelFor(
      *a_state[lev], amrex::IntVect(0),
      [=, bins =
            m_bins] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        if (marrs[box_no](i, j, k) == 0) {
          return;
        }
        const amrex::Real c = sarrs[box_no](i, j, k, cFieldIdx);
        const int cbin = bins.index(c);
        if (cbin < 0) {
          return;
        }
        const amrex::Real w =
          weighted ? varrs[box_no](i, j, k) : amrex::Real(1.0);
        for (int f{0}; f < nFields; ++f) {
          const amrex::Real v = sarrs[box_no](i, j, k, idx_p[f]);
          amrex::HostDevice::Atomic::Add(
            &(data_p[condOff + f * nBins + cbin]), w * v);
          if (average) {
            amrex::HostDevice::Atomic::Add(
              &(data_p[condSqOff + f * nBins + cbin]), w * v * v);
          }
        }
        if (average) {
          amrex::HostDevice::Atomic::Add(&(data_p[condVolOff + cbin]), w);
        }
        amrex::HostDevice::Atomic::Add(&(data_p[condAbsOff + cbin]), w * c);
      });
  }

  amrex::Vector<amrex::Real> data(data_d.size());
  amrex::Gpu::copy(
    amrex::Gpu::deviceToHost, data_d.begin(), data_d.end(), data.begin());
  amrex::ParallelDescriptor::ReduceRealSum(
    data.data(), static_cast<int>(data.size()));
  amrex::Real* cond = data.data() + condOff;
  amrex::Real* condSq = data.data() + condSqOff;
  amrex::Real* condAbs = data.data() + condAbsOff;
  const amrex::Real* condVol = data.data() + condVolOff;

  const std::string file = outputName(a_nstep, a_time) + ".dat";
  if (m_condType == Average) {
    for (int n{0}; n < nBins; ++n) {
      if (condVol[n] != 0.0) {
        for (int f{0}; f < nFields; ++f) {
          cond[f * nBins + n] /= condVol[n];
          condSq[f * nBins + n] /= condVol[n];
        }
        condAbs[n] /= condVol[n];
      }
    }
    writeAverageDataToFile(file, condAbs, cond, condSq, condVol);
  } else {
    writeTotalDataToFile(
      file, condAbs, cond, m_condType == Integral ? "_Int" : "_Sum");
  }
}

void
DiagConditional::writeAverageDataToFile(
  const std::string& a_file,
  const amrex::Real* a_condAbs,
  const amrex::Real* a_cond,
  const amrex::Real* a_condSq,
  const amrex::Real* a_condVol)
{
  if (!amrex::ParallelDescriptor::IOProcessor()) {
    return;
  }
  std::ofstream condFile(a_file, std::ios::out);
  const int prec = 8;
  const int width = 16;
  const int nFields = static_cast<int>(m_fieldNames.size());
  amrex::Vector<int> widths(3 + 2 * nFields, width);

  condFile << std::left << std::setw(widths[0]) << "BinCenter" << " ";
  widths[1] =
    amrex::max<int>(width, static_cast<int>(m_cFieldName.length()) + 1);
  condFile << std::setw(widths[1]) << m_cFieldName << " ";
  condFile << std::setw(widths[2]) << "Volume" << " ";
  for (int f{0}; f < nFields; ++f) {
    widths[3 + 2 * f] =
      amrex::max<int>(width, static_cast<int>(m_fieldNames[f].length()) + 5);
    condFile << std::setw(widths[3 + 2 * f]) << m_fieldNames[f] + "_Avg" << " ";
    widths[4 + 2 * f] =
      amrex::max<int>(width, static_cast<int>(m_fieldNames[f].length()) + 7);
    condFile << std::setw(widths[4 + 2 * f]) << m_fieldNames[f] + "_StdDev"
             << " ";
  }
  condFile << "\n";

  const int nBins = m_bins.nBins;
  condFile << std::setprecision(prec) << std::scientific;
  for (int n{0}; n < nBins; ++n) {
    condFile << std::setw(widths[0]) << m_bins.center(n) << " ";
    condFile << std::setw(widths[1]) << a_condAbs[n] << " ";
    condFile << std::setw(widths[2]) << a_condVol[n] << " ";
    for (int f{0}; f < nFields; ++f) {
      const amrex::Real avg = a_cond[f * nBins + n];
      condFile << std::setw(widths[3 + 2 * f]) << avg << " "
               << std::setw(widths[4 + 2 * f])
               << std::sqrt(std::abs(a_condSq[f * nBins + n] - avg * avg))
               << " ";
    }
    condFile << "\n";
  }
}

void
DiagConditional::writeTotalDataToFile(
  const std::string& a_file,
  const amrex::Real* a_condAbs,
  const amrex::Real* a_cond,
  const std::string& a_suffix)
{
  if (!amrex::ParallelDescriptor::IOProcessor()) {
    return;
  }
  std::ofstream condFile(a_file, std::ios::out | std::ios::app);
  const int prec = 8;
  const int width = 16;
  const int nFields = static_cast<int>(m_fieldNames.size());
  amrex::Vector<int> widths(1 + nFields, width);

  widths[0] =
    amrex::max<int>(width, static_cast<int>(m_cFieldName.length()) + 1);
  condFile << std::left << std::setw(widths[0]) << m_cFieldName << " ";
  for (int f{0}; f < nFields; ++f) {
    widths[1 + f] =
      amrex::max<int>(width, static_cast<int>(m_fieldNames[f].length()) + 5);
    condFile << std::setw(widths[1 + f]) << m_fieldNames[f] + a_suffix << " ";
  }
  condFile << "\n";

  const int nBins = m_bins.nBins;
  condFile << std::setprecision(prec) << std::scientific;
  for (int n{0}; n < nBins; ++n) {
    condFile << std::setw(widths[0]) << a_condAbs[n] << " ";
    for (int f{0}; f < nFields; ++f) {
      condFile << std::setw(widths[1 + f]) << a_cond[f * nBins + n] << " ";
    }
    condFile << "\n";
  }
}
