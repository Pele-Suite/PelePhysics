#ifndef PONECHECK_H
#define PONECHECK_H

// Shared by the radiation tests: the Stefan-Boltzmann constant and the check
// of the P1 flame solutions (tstPOneSingleAF, tstPOneMultiAF)

#include <AMReX_MultiFab.H>

#include <cmath>
#include <string>

#include "TestCheck.H"

namespace pele::testing {

// W/(m^2 K^4), as in PeleLMRad.H
constexpr amrex::Real sigma = 5.67e-8;

// The P1 solution for a flame between cold walls: G finite, positive and
// below 4 sigma T^4 at the hottest point, and, given the source, the gas
// loses energy overall.
inline void
check_solution(
  Checker& check,
  amrex::MultiFab const& G,
  amrex::MultiFab const* rad_src,
  amrex::MultiFab const& T,
  std::string const& label)
{
  amrex::Real const Gmax_bound = 4.0 * sigma * std::pow(T.max(0), 4.0);
  bool const finite = !G.contains_nan(0, 1, 0) && !G.contains_inf(0, 1, 0);
  amrex::Real const Gmin = G.min(0), Gmax = G.max(0);
  check(finite, label + "G is finite");
  check(
    Gmin > 0.0 && Gmax <= Gmax_bound,
    label + "0 < G <= 4 sigma Tmax^4 (G in [" + std::to_string(Gmin) + ", " +
      std::to_string(Gmax) + "], bound " + std::to_string(Gmax_bound) + ")");
  if (rad_src != nullptr) {
    check(
      rad_src->sum(0) < 0.0 && rad_src->min(0) < 0.0,
      label + "the flame loses energy (net source " +
        std::to_string(rad_src->sum(0)) + ")");
  }
}

} // namespace pele::testing

#endif
