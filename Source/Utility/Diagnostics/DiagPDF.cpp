#include "DiagPDF.H"
#include "AMReX_MultiFabUtil.H"

#include <numeric>
#include <tuple>

void
DiagPDF::init(const std::string& a_prefix, std::string_view a_diagName)
{
  DiagBase::init(a_prefix, a_diagName);

  amrex::ParmParse pp(a_prefix);
  pp.get("field_name", m_fieldName);
  pp.get("nBins", m_nBins);
  AMREX_ASSERT(m_nBins > 0);
  pp.query("normalized", m_normalized);
  pp.query("volume_weighted", m_volWeighted);

  if (pp.countval("range") != 0) {
    amrex::Vector<amrex::Real> range{0.0};
    pp.getarr("range", range, 0, 2);
    m_lowBnd = amrex::min<amrex::Real>(range[0], range[1]);
    m_highBnd = amrex::max<amrex::Real>(range[0], range[1]);
    m_useFieldMinMax = false;
  }
}

void
DiagPDF::addVars(amrex::Vector<std::string>& a_varList)
{
  DiagBase::addVars(a_varList);
  a_varList.push_back(m_fieldName);
}

void
DiagPDF::prepare(
  int a_nlevels,
  const amrex::Vector<amrex::Geometry>& a_geoms,
  const amrex::Vector<amrex::BoxArray>& a_grids,
  const amrex::Vector<amrex::DistributionMapping>& a_dmap,
  const amrex::Vector<std::string>& a_varNames)
{
  DiagBase::prepare(a_nlevels, a_geoms, a_grids, a_dmap, a_varNames);
  if (first_time) {
    first_time = false;
  }
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

  // Set PDF range
  if (m_useFieldMinMax) {
    std::tie(m_lowBnd, m_highBnd) = maskedMinMax(a_state, cells, fieldIdx);
  }
  const amrex::Real binWidth = (m_highBnd - m_lowBnd) / m_nBins;

  // Data holders
  amrex::Gpu::DeviceVector<amrex::Real> pdf_d(m_nBins, 0.0);
  amrex::Vector<amrex::Real> pdf(m_nBins, 0.0);

  // Populate the data from each level on each proc
  for (int lev = 0; lev < a_state.size(); ++lev) {
    auto const& sarrs = a_state[lev]->const_arrays();
    auto const& marrs = cells[lev].mask.const_arrays();
    auto const& varrs = cells[lev].volume.const_arrays();
    auto* pdf_d_p = pdf_d.dataPtr();
    amrex::ParallelFor(
      *a_state[lev], amrex::IntVect(0),
      [=, nBins = m_nBins, lowBnd = m_lowBnd,
       volWeighted =
         m_volWeighted] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        if (marrs[box_no](i, j, k) != 0) {
          const int cbin =
            binIndex(sarrs[box_no](i, j, k, fieldIdx), lowBnd, binWidth, nBins);
          if (cbin >= 0) {
            amrex::HostDevice::Atomic::Add(
              &(pdf_d_p[cbin]),
              volWeighted != 0 ? varrs[box_no](i, j, k) : amrex::Real(1.0));
          }
        }
      });
    amrex::Gpu::streamSynchronize();
  }

  amrex::Gpu::copy(
    amrex::Gpu::deviceToHost, pdf_d.begin(), pdf_d.end(), pdf.begin());
  amrex::Gpu::streamSynchronize();
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
  std::string diagfile;
  if (m_interval > 0) {
    diagfile = amrex::Concatenate(m_diagfile, a_nstep, 6);
  }
  if (m_per > 0.0) {
    diagfile = m_diagfile + std::to_string(a_time);
  }
  diagfile = diagfile + ".dat";

  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::ofstream pdfFile;
    pdfFile.open(diagfile.c_str(), std::ios::out);
    const int prec = 8;
    const int width = 16;
    amrex::Vector<int> widths(2, width);

    amrex::Real binWidth = (m_highBnd - m_lowBnd) / (m_nBins);

    widths[0] =
      amrex::max<int>(width, static_cast<int>(m_fieldName.length()) + 1);
    widths[1] =
      amrex::max<int>(width, static_cast<int>(m_fieldName.length()) + 5);
    pdfFile << std::setw(widths[0]) << m_fieldName << " "
            << std::setw(widths[1]) << m_fieldName + "_PDF" << "\n";

    for (int i{0}; i < a_pdf.size(); ++i) {
      pdfFile << std::setw(widths[0]) << std::setprecision(prec)
              << std::scientific
              << m_lowBnd + (static_cast<amrex::Real>(i) + 0.5) * binWidth
              << " " << std::setw(widths[1]) << std::setprecision(prec)
              << std::scientific
              << a_pdf[i] / (m_normalized != 0 ? a_sum : 1.0) / binWidth
              << "\n";
    }

    pdfFile.flush();
    pdfFile.close();
  }
}
