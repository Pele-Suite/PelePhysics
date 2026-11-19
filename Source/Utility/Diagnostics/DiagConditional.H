#ifndef DIAGCOND_H
#define DIAGCOND_H

#include "DiagBase.H"

class DiagConditional : public DiagBase::Register<DiagConditional>
{
public:
  static std::string identifier() { return "DiagConditional"; }

  enum conditionalType { Average, Integral, Sum };

  void init(const std::string& a_prefix, std::string_view a_diagName) override;

  void processDiag(
    int a_nstep,
    const amrex::Real& a_time,
    const amrex::Vector<const amrex::MultiFab*>& a_state,
    const amrex::Vector<std::string>& a_stateVar) override;

  void addVars(amrex::Vector<std::string>& a_varList) override;

  void close() override {}

private:
  void writeAverageDataToFile(
    const std::string& a_file,
    const amrex::Real* a_condAbs,
    const amrex::Real* a_cond,
    const amrex::Real* a_condSq,
    const amrex::Real* a_condVol);
  // Integral or Sum: the per-bin totals, field columns suffixed a_suffix
  void writeTotalDataToFile(
    const std::string& a_file,
    const amrex::Real* a_condAbs,
    const amrex::Real* a_cond,
    const std::string& a_suffix);

  conditionalType m_condType;              // Type of conditional calculation
  std::string m_cFieldName;                // Condition Field name
  amrex::Vector<std::string> m_fieldNames; // Processed Field names
  DiagBins m_bins;                         // Bins of the condition field
};

#endif
