#ifndef SOOTREACTIONS_H
#define SOOTREACTIONS_H

#include "Constants_Soot.H"

struct SootReaction
{
  amrex::Real SootDensityC;
  amrex::Real SootChi;
  // Vectors of the forward and backward pre-exponential rates,
  // temperature exponents, and activation energies/R
  amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> A_f = {{0.}};
  amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> n_f = {{0.}};
  amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> ER_f = {{0.}};
  amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> A_b = {{0.}};
  amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> n_b = {{0.}};
  amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> ER_b = {{0.}};
  // Vector of number of reactants and products
  amrex::GpuArray<int, NUM_SOOT_REACT> rNum = {{0}};
  amrex::GpuArray<int, NUM_SOOT_REACT> pNum = {{0}};
  // Vector of species indices
  // Maximum 3 species per reaction side
  amrex::GpuArray<int, 3 * NUM_SOOT_REACT> nIndx_f = {{0}};
  amrex::GpuArray<int, 3 * NUM_SOOT_REACT> nIndx_b = {{0}};
  // Soot surface reaction reference indexing
  // Forward and backward reaction soot indices
  // Note: Assumes only 1 mole of soot is on each side
  amrex::GpuArray<int, NUM_SOOT_REACT> sIndx_f = {{0}};
  amrex::GpuArray<int, NUM_SOOT_REACT> sIndx_b = {{0}};
  // Vector of stoichiometric coefficients
  amrex::GpuArray<amrex::Real, 3 * NUM_SOOT_REACT> nu_f = {{0.}};
  amrex::GpuArray<amrex::Real, 3 * NUM_SOOT_REACT> nu_b = {{0.}};

  // Compute the dimerization rate
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
  dimerRate(const amrex::Real& T, const amrex::Real& xi_PAH) const
  {
    // The effective rate of dimerization
    // This should coincide with the last reaction in the reaction lists
    const int fr = NUM_SOOT_REACT - 1;
    return amrex::max<amrex::Real>(
      0., A_f[fr] * std::pow(T, n_f[fr]) * std::exp(-ER_f[fr] / T) * xi_PAH *
            xi_PAH);
  }

  // Compute the surface and gas phase chemistry rates
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void chemicalSrc(
    const amrex::Real& T,
    const amrex::Real& surf,
    const amrex::Real xi_n[],
    const amrex::Real moments[],
    amrex::Real& k_sg,
    amrex::Real& k_ox,
    amrex::Real& k_o2,
    amrex::Real omega_src[],
    const bool surface) const
  {
    // Number of surface reactions
    const int nsr = NUM_SOOT_REACT;
    amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> k_fwd = {0.0};
    amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> k_bkwd = {0.0};
    amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> w_fwd = {0.0};
    amrex::GpuArray<amrex::Real, NUM_SOOT_REACT> w_bkwd = {0.0};
    const amrex::Real invT = 1. / T;
    // Without the surface reactions only the last (dimerization) reaction
    // contributes
    const int i0 = surface ? 0 : nsr - 1;
    // Loop over reactions
    for (int i = i0; i < nsr; ++i) {
      k_fwd[i] = A_f[i] * std::pow(T, n_f[i]) * std::exp(-ER_f[i] * invT);
      k_bkwd[i] = A_b[i] * std::pow(T, n_b[i]) * std::exp(-ER_b[i] * invT);
      amrex::Real fwdM = 1.;
      for (int j = 0; j < rNum[i]; ++j) {
        // Reactant gas species index
        const int rIndx = nIndx_f[3 * i + j];
        fwdM *= std::pow(xi_n[rIndx], nu_f[3 * i + j]);
      }
      w_fwd[i] = k_fwd[i] * fwdM;
      amrex::Real bkwdM = 1.;
      for (int j = 0; j < pNum[i]; ++j) {
        // Product gas species index
        const int pIndx = nIndx_b[3 * i + j];
        bkwdM *= std::pow(xi_n[pIndx], nu_b[3 * i + j]);
      }
      w_bkwd[i] = k_bkwd[i] * bkwdM;
    }
    if (surface) {
      // Note: This will depend on the surface reactions, currently hardcoded
      amrex::Real fSootStar =
        computeRadSiteConc(xi_n, k_fwd.data(), k_bkwd.data());
      computeSurfRates(
        w_fwd.data(), w_bkwd.data(), fSootStar, k_sg, k_ox, k_o2);
      // Determine the concentration of hydrogenated and radical soot surface
      // sites Quasi-steady state for surface radical sites on soot
      amrex::GpuArray<amrex::Real, SootIndx::numSootSpecs> C_Soot = {0.0};
      C_Soot[SootIndx::indxSootS] = fSootStar * SootDensityC * moments[2];
      C_Soot[SootIndx::indxSootH] =
        (1. - fSootStar) * SootDensityC * moments[2];
      // Note: The last two reaction (6 and 7) are special
      // they are not treated the same in this loop
      // Currently assumes soot is involved in every reaction on both sides
      for (int i = 0; i < nsr - 2; ++i) {
        int sootIndx = sIndx_f[i];
        w_fwd[i] *= C_Soot[sootIndx];
        sootIndx = sIndx_b[i];
        w_bkwd[i] *= C_Soot[sootIndx];
      }
      // Note: This is specific to the hardcoded 6th reaction, not universal
      w_fwd[5] *= surf;
    }
    // Loop over gas species and solve for omegas
    for (int i = i0; i < nsr; ++i) {
      // Creation/destruction rates for the i-th reaction
      amrex::Real rate = w_fwd[i] - w_bkwd[i];
      // Loop over reactant species and subtract nu'_(i,r)*rate
      for (int r = 0; r < rNum[i]; ++r) {
        const int rIndx = nIndx_f[3 * i + r];
        amrex::Real nup = nu_f[3 * i + r];
        omega_src[rIndx] -= nup * rate;
      }
      // Loop over product species and add nu"_(i,r)*rate
      for (int p = 0; p < pNum[i]; ++p) {
        const int pIndx = nIndx_b[3 * i + p];
        amrex::Real nupp = nu_b[3 * i + p];
        omega_src[pIndx] += nupp * rate;
      }
    }
  }

  // Return fSootStar, fraction of hydrogenated sites
  // that are radical sites
  static AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real computeRadSiteConc(
    const amrex::Real xi_n[],
    const amrex::Real k_fwd[],
    const amrex::Real k_bkwd[])
  {
    amrex::Real C_OH = xi_n[SootGasSpecIndx::indxOH];
    amrex::Real C_H = xi_n[SootGasSpecIndx::indxH];
    amrex::Real C_H2 = xi_n[SootGasSpecIndx::indxH2];
    amrex::Real C_H2O = xi_n[SootGasSpecIndx::indxH2O];
    amrex::Real C_C2H2 = xi_n[SootGasSpecIndx::indxC2H2];
    // Factor r for the quasi-steady state concentration of radical sites, r1/r2
    amrex::Real r1 = (k_fwd[0] * C_OH + k_fwd[1] * C_H + k_fwd[2]);
    amrex::Real r2 =
      (k_bkwd[0] * C_H2O + k_bkwd[1] * C_H2 + k_bkwd[2] * C_H +
       k_fwd[3] * C_C2H2);
    return r1 / (r2 + r1);
  }

  // Compute the surface chemistry rates (1/s)
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void computeSurfRates(
    const amrex::Real w_fwd[],
    const amrex::Real* /*w_bkwd*/,
    const amrex::Real& fSootStar,
    amrex::Real& k_sg,
    amrex::Real& k_ox,
    amrex::Real& k_o2) const
  {
    k_sg = w_fwd[3] * fSootStar;
    k_ox = w_fwd[4] * fSootStar + w_fwd[5] * 0.5 / SootChi;
    k_o2 = w_fwd[4] * fSootStar;
  }
};

#endif
