#include "DiagFilter.H"
#include "DiagBase.H"

bool
DiagFilter::queryRange(
  const amrex::ParmParse& a_pp,
  const std::string& a_key,
  amrex::Real& a_low,
  amrex::Real& a_high)
{
  if (a_pp.countval(a_key.c_str()) == 0) {
    return false;
  }
  amrex::Vector<amrex::Real> range;
  a_pp.getarr(a_key.c_str(), range, 0, 2);
  a_low = amrex::min<amrex::Real>(range[0], range[1]);
  a_high = amrex::max<amrex::Real>(range[0], range[1]);
  return true;
}

void
DiagFilter::init(const std::string& a_prefix)
{
  amrex::ParmParse pp(a_prefix);

  pp.query("field_name", m_filterVar);
  if (m_filterVar.empty()) {
    amrex::Abort("filter: " + a_prefix + " is missing a field_name !");
  }

  if (pp.countval("value_greater") != 0) {
    pp.get("value_greater", m_fdata.m_low_val);
    m_fdata.m_high_val = AMREX_REAL_MAX;
  } else if (pp.countval("value_less") != 0) {
    pp.get("value_less", m_fdata.m_high_val);
    m_fdata.m_low_val = AMREX_REAL_LOWEST;
  } else if (!queryRange(
               pp, "value_inrange", m_fdata.m_low_val, m_fdata.m_high_val)) {
    amrex::Abort("filter: " + a_prefix + " is missing a range definition !");
  }
}

void
DiagFilter::setup(const amrex::Vector<std::string>& a_varNames)
{
  m_fdata.m_filterVarIdx = DiagBase::getFieldIndex(m_filterVar, a_varNames);
}
