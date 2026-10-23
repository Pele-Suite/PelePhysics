#ifndef TRANSPORTUTILS_H
#define TRANSPORTUTILS_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <cmath>
#include "mechanism.H"

namespace pele::physics::transport {

// Number of species pairs (i,j) with i <= j, i.e. the packed storage used by
// the mechanism's binary diffusion fits (egtransetCOFD)
constexpr int NUM_BINARY_PAIRS = NUM_SPECIES * (NUM_SPECIES + 1) / 2;

// Packed index of the unordered species pair (i,j), matching egtransetCOFD
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
constexpr int
binary_pair_index(const int i, const int j)
{
  return (i < j) ? (j * (j + 1) / 2 + i) : (i * (i + 1) / 2 + j);
}

// Thermal conductivity models of SimpleTransport (transport.conductivity_model)
struct ConductivityModel
{
  static constexpr int Mixture = 0; // empirical mixture rule (default)
  static constexpr int LCT1 = 1;    // EGlib EGSLCT1: Lambda_[e], N unknowns
  static constexpr int LCT3 = 3;    // EGlib EGSLCT3: Lambda, 2N unknowns
};

// Parker's rotational relaxation temperature dependence
//   F(x) = 1 + pi^{3/2}/2 sqrt(x) + (2 + pi^2/4) x + pi^{3/2} x^{3/2},
// with x = eps/T (same expression as the bulk viscosity in Simple.H)
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
parker_factor(const amrex::Real x)
{
  const amrex::Real pi3_2 = std::pow(M_PI, 1.5);
  const amrex::Real sqrtx = std::sqrt(x);
  return 1.0 + 0.50 * pi3_2 * sqrtx + (2.0 + .50 * M_PI * M_PI) * x +
         pi3_2 * sqrtx * x;
}

// In-place projection Dmat <- P Dmat P with P = I - Y u^T / sum(Y), where u is
// the vector of ones (EGlib EGSPRD). Dmat is row-major NUM_SPECIES x
// NUM_SPECIES. Afterwards sum_i Dmat_ij = 0 for every j, so the species
// fluxes F_i = -sum_j Dmat_ij grad(X_j) sum to zero exactly.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
project_diffusion_matrix(const amrex::Real* Yloc, amrex::Real* Dmat)
{
  amrex::Real r[NUM_SPECIES] = {0.0}; // (Dmat Y)_i / s
  amrex::Real t[NUM_SPECIES] = {0.0}; // (u^T Dmat)_j
  amrex::Real s = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i) {
    s += Yloc[i];
  }
  const amrex::Real rs = 1.0 / s;
  for (int i = 0; i < NUM_SPECIES; ++i) {
    const amrex::Real* row = Dmat + i * NUM_SPECIES;
    amrex::Real acc = 0.0;
    for (int j = 0; j < NUM_SPECIES; ++j) {
      acc += row[j] * Yloc[j];
      t[j] += row[j];
    }
    r[i] = acc * rs;
  }
  amrex::Real c = 0.0;
  for (int i = 0; i < NUM_SPECIES; ++i) {
    c += r[i];
  }
  c *= rs;
  for (int i = 0; i < NUM_SPECIES; ++i) {
    amrex::Real* row = Dmat + i * NUM_SPECIES;
    const amrex::Real yi = Yloc[i];
    for (int j = 0; j < NUM_SPECIES; ++j) {
      row[j] = row[j] - r[i] - yi * (t[j] * rs - c);
    }
  }
}

// Diagonal flux diffusion matrix with a uniform coefficient, projected so
// that it satisfies the same mass-conservation contract as the
// multicomponent matrix. For a uniform coefficient the projected fluxes
// F_i = -sum_j Dmat_ij grad(X_j) reduce to -rhoD grad(X_i) because the mole
// fraction gradients sum to zero.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
uniform_diffusion_matrix(
  const amrex::Real rhoD, const amrex::Real* Yloc, amrex::Real* Dmat)
{
  for (int n = 0; n < NUM_SPECIES * NUM_SPECIES; ++n) {
    Dmat[n] = 0.0;
  }
  for (int i = 0; i < NUM_SPECIES; ++i) {
    Dmat[i * NUM_SPECIES + i] = rhoD;
  }
  if (Yloc != nullptr) {
    project_diffusion_matrix(Yloc, Dmat);
  }
}

} // namespace pele::physics::transport
#endif
