#ifndef TRANSPORTUTILS_H
#define TRANSPORTUTILS_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include "mechanism.H"

namespace pele::physics::transport {

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
