#include "DiagPDF.H"

#include <numeric>
#include <tuple>

void
DiagPDF::init(const std::string& a_prefix, std::string_view a_diagName)
{
  DiagBase::init(a_prefix, a_diagName);

  amrex::ParmParse pp(a_prefix);
  pp.get("field_name", m_fieldName);
  m_bins.init(pp);
  pp.query("normalized", m_normalized);
  pp.query("volume_weighted", m_volWeighted);
}

void
DiagPDF::addVars(amrex::Vector<std::string>& a_varList)
{
  DiagBase::addVars(a_varList);
  a_varList.push_back(m_fieldName);
}

void
DiagPDF::processDiag(
  int a_nstep,
  const amrex::Real& a_time,
  const amrex::Vector<const amrex::MultiFab*>& a_state,
  const amrex::Vector<std::string>& a_stateVar)
{
  // Cells to process (not covered by a finer level or the EB, passing the
  // filters) and their fluid volume
  const int fieldIdx = getFieldIndex(m_fieldName, a_stateVar);
  const auto cells = levelCells(a_state);
  if (m_bins.fromData) {
    std::tie(m_bins.low, m_bins.high) = maskedMinMax(a_state, cells, fieldIdx);
  }

  // Populate the data from each level on each proc
  amrex::Gpu::DeviceVector<amrex::Real> pdf_d(m_bins.nBins, 0.0);
  auto* pdf_d_p = pdf_d.dataPtr();
  for (int lev = 0; lev < a_state.size(); ++lev) {
    auto const& sarrs = a_state[lev]->const_arrays();
    auto const& marrs = cells[lev].mask.const_arrays();
    auto const& varrs = cells[lev].volume.const_arrays();
    amrex::ParallelFor(
      *a_state[lev], amrex::IntVect(0),
      [=, bins = m_bins, volWeighted = m_volWeighted] AMREX_GPU_DEVICE(
        int box_no, int i, int j, int k) noexcept {
        if (marrs[box_no](i, j, k) != 0) {
          const int cbin = bins.index(sarrs[box_no](i, j, k, fieldIdx));
          if (cbin >= 0) {
            amrex::HostDevice::Atomic::Add(
              &(pdf_d_p[cbin]),
              volWeighted != 0 ? varrs[box_no](i, j, k) : amrex::Real(1.0));
          }
        }
      });
  }

  amrex::Vector<amrex::Real> pdf(m_bins.nBins);
  amrex::Gpu::copy(
    amrex::Gpu::deviceToHost, pdf_d.begin(), pdf_d.end(), pdf.begin());
  amrex::ParallelDescriptor::ReduceRealSum(
    pdf.data(), static_cast<int>(pdf.size()));
  auto sum =
    std::accumulate(pdf.begin(), pdf.end(), decltype(pdf)::value_type(0));

  writePDFToFile(a_nstep, a_time, pdf, sum);
}

void
DiagPDF::writePDFToFile(
  int a_nstep,
  const amrex::Real& a_time,
  const amrex::Vector<amrex::Real>& a_pdf,
  const amrex::Real& a_sum)
{
  if (!amrex::ParallelDescriptor::IOProcessor()) {
    return;
  }
  std::ofstream pdfFile(outputName(a_nstep, a_time) + ".dat", std::ios::out);
  const int prec = 8;
  const int width = 16;
  const int width0 =
    amrex::max<int>(width, static_cast<int>(m_fieldName.length()) + 1);
  const int width1 =
    amrex::max<int>(width, static_cast<int>(m_fieldName.length()) + 5);
  pdfFile << std::setw(width0) << m_fieldName << " " << std::setw(width1)
          << m_fieldName + "_PDF" << "\n";

  const amrex::Real binWidth = m_bins.width();
  pdfFile << std::setprecision(prec) << std::scientific;
  for (int i{0}; i < a_pdf.size(); ++i) {
    pdfFile << std::setw(width0) << m_bins.center(i) << " " << std::setw(width1)
            << a_pdf[i] / (m_normalized != 0 ? a_sum : 1.0) / binWidth << "\n";
  }
}
