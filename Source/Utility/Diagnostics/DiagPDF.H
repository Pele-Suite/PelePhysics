#ifndef DIAGPDF_H
#define DIAGPDF_H

#include "DiagBase.H"

class DiagPDF : public DiagBase::Register<DiagPDF>
{
public:
  static std::string identifier() { return "DiagPDF"; }

  void init(const std::string& a_prefix, std::string_view a_diagName) override;

  void processDiag(
    int a_nstep,
    const amrex::Real& a_time,
    const amrex::Vector<const amrex::MultiFab*>& a_state,
    const amrex::Vector<std::string>& a_stateVar) override;

  void addVars(amrex::Vector<std::string>& a_varList) override;

  void close() override {}

private:
  void writePDFToFile(
    int a_nstep,
    const amrex::Real& a_time,
    const amrex::Vector<amrex::Real>& a_pdf,
    const amrex::Real& a_sum);

  std::string m_fieldName; // Field name
  DiagBins m_bins;         // Bins of the field
  int m_normalized{1};     // Normalized ?
  int m_volWeighted{1};    // Volume weighted ?
};

#endif
