#ifndef DIAGFILTERS_H
#define DIAGFILTERS_H

#include <AMReX.H>
#include <AMReX_Utility.H>
#include <AMReX_ParmParse.H>

struct DiagFilterData
{
  int m_filterVarIdx{-1};
  amrex::Real m_low_val{0.0};
  amrex::Real m_high_val{0.0};
};

struct DiagFilter
{
  std::string m_filterVar;

  void init(const std::string& a_prefix);
  void setup(const amrex::Vector<std::string>& a_varNames);

  // Read the two values of a_key, if given, as a range [a_low, a_high]
  static bool queryRange(
    const amrex::ParmParse& a_pp,
    const std::string& a_key,
    amrex::Real& a_low,
    amrex::Real& a_high);

  DiagFilterData m_fdata;
};
#endif
