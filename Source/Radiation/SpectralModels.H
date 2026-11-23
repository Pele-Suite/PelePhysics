#ifndef SPECTRAL_MODELS_H
#define SPECTRAL_MODELS_H

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>

namespace PeleRad::RadProp {
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
interpT(amrex::Real const& T, int& TindexL, amrex::Real& weight)
{
  if (T < 300) {
    TindexL = 0;
    weight = 0.0;
    return;
  }
  // The tables run from 300 K to 2800 K in 20 K steps (126 entries); at
  // T = 2800 K exactly the general branch would read k[126].
  if (T >= 2800) {
    TindexL = 124;
    weight = 1.0;
    return;
  }

  amrex::Real TindexReal = (T - 300.0) / 20.0;
  amrex::Real TindexInte = floor((T - 300.0) / 20.0);
  TindexL = static_cast<int>(TindexInte);
  weight = TindexReal - TindexInte;

  AMREX_ASSERT(weight <= 1 && weight >= 0);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
interpk(
  int const& TindexL,
  amrex::Real const& weight,
  amrex::GpuArray<amrex::Real, 126UL> const& k)
{
  return (1.0 - weight) * k[TindexL] + weight * k[TindexL + 1];
}

// Planck-mean absorption coefficient of the gas, in 1/cm:
// kappa = P / 1e5 * sum_i X_i kp_i(T), with X_i the mole fractions of CO2,
// H2O and CO and P in Pa. Dividing by 1e5 treats the kp tables as
// 1/(cm bar); their source (bar or atm) is not recorded.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getRadPropGas(
  int i,
  int j,
  int k,
  amrex::Array4<const amrex::Real> const& xco2,
  amrex::Array4<const amrex::Real> const& xh2o,
  amrex::Array4<const amrex::Real> const& xco,
  amrex::Array4<const amrex::Real> const& temp,
  amrex::Array4<const amrex::Real> const& pressure,
  amrex::Array4<amrex::Real> const& absc,
  amrex::GpuArray<amrex::Real, 126UL> const& kdataco2,
  amrex::GpuArray<amrex::Real, 126UL> const& kdatah2o,
  amrex::GpuArray<amrex::Real, 126UL> const& kdataco)
{
  if (
    xco2(i, j, k) < 1.0e-5 && xh2o(i, j, k) < 1.0e-5 && xco(i, j, k) < 1.0e-5) {
    absc(i, j, k) = 0.001;
    return;
  }
  int TindexL = 0;
  amrex::Real weight = 1.0;
  interpT(temp(i, j, k), TindexL, weight);

  amrex::Real kp_co2 = interpk(TindexL, weight, kdataco2);
  amrex::Real kp_h2o = interpk(TindexL, weight, kdatah2o);
  amrex::Real kp_co = interpk(TindexL, weight, kdataco);

  absc(i, j, k) =
    xco2(i, j, k) * kp_co2 + xh2o(i, j, k) * kp_h2o + xco(i, j, k) * kp_co;

  // absc(i, j, k) *= pressure(i, j, k) / 1.0e5 * 100.0; //si, in m-1
  absc(i, j, k) *=
    pressure(i, j, k) /
    1.0e5; // cgs, in cm-1; if P is in bar(PeleLM), no correction
           // is needed, otherwise the absc needs to be corrected.
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
getRadPropSoot(
  int i,
  int j,
  int k,
  amrex::Array4<const amrex::Real> const& fv,
  amrex::Array4<const amrex::Real> const& temp,
  amrex::Array4<amrex::Real> const& absc,
  amrex::GpuArray<amrex::Real, 126UL> const& kdatasoot)
{
  int TindexL = 0;
  amrex::Real weight = 1.0;

  interpT(temp(i, j, k), TindexL, weight);

  amrex::Real kp_soot = interpk(TindexL, weight, kdatasoot);

  // absc(i, j, k) += fv(i, j, k) * kp_soot * 100.0; //si, in m-1
  absc(i, j, k) += fv(i, j, k) * kp_soot; // cgs, in cm-1
}

} // namespace PeleRad::RadProp
#endif
