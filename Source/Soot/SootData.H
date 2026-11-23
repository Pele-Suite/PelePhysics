#ifndef SOOTDATA_H
#define SOOTDATA_H

#include "AMReX_REAL.H"
#include "Constants_Soot.H"
#include "SootReactions.H"

using namespace amrex::literals;

struct SootData
{
  SootConst sc;
  amrex::Real nuclVol;
  amrex::Real nuclSurf;
  amrex::Real condFact;
  amrex::Real lambdaCF;
  amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS + 1> unitConv;
  amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS + 1> smallMoms;
  amrex::GpuArray<int, NUM_SOOT_GS> refIndx;
  amrex::GpuArray<amrex::Real, 9> dime6;
  amrex::GpuArray<amrex::Real, 11> nve3;
  amrex::GpuArray<amrex::Real, 11> nve6;
  amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS> momFact;
  amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS + 1> ssfmCF;
  amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS> sscnCF;
  amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS + 1> smallOF;
  amrex::GpuArray<amrex::Real, NUM_SOOT_MOMENTS> fragFact;

  // Return corresponding PelePhysics species index
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int
  getPPIndx(SootGasSpecIndx soot_indx) const
  {
    return refIndx[soot_indx];
  }

  // Convert moments from CGS/SI to mol of C
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
  convertToMol(amrex::Real moments[]) const
  {
    for (int i = 0; i < NUM_SOOT_MOMENTS + 1; ++i) {
      moments[i] /= unitConv[i];
    }
  }

  // Convert moments from mol of C to CGS/SI
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
  convertFromMol(amrex::Real mom_src[]) const
  {
    for (int i = 0; i < NUM_SOOT_MOMENTS + 1; ++i) {
      mom_src[i] *= unitConv[i];
    }
  }

  // Convert moments from mol of C to CGS/SI
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
  convertFromMol(amrex::Real mom_src[], amrex::Real moments[]) const
  {
    for (int i = 0; i < NUM_SOOT_MOMENTS + 1; ++i) {
      mom_src[i] *= unitConv[i];
      moments[i] *= unitConv[i];
    }
  }

  // Whether there is enough soot (volume fraction above 1e-12) for the
  // surface reactions; moments in mol of C
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
  surfaceActive(const amrex::Real moments[]) const
  {
    return moments[1] * sc.V0 * pele::physics::Constants::Avna > 1.E-12;
  }

  // Initialize moment values to very small
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
  initialSmallMomVals(amrex::Real moments[]) const
  {
    for (int i = 0; i < NUM_SOOT_MOMENTS + 1; ++i) {
      moments[i] = smallMoms[i];
    }
    convertFromMol(moments);
  }

  // Given an exponent numerator, return dimer_vol^(exp/6)
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
  getDimerExp6(const int exp) const
  {
    int indx = (exp + 3) / 2;
    return dime6[indx];
  }

  // Given an exponent numerator, return (nucl_vol)^(exp/3)
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
  getNuclExp3(const int exp) const
  {
    int indx = exp + 3;
    return nve3[indx];
  }

  // Given an exponent numerator, return (nucl_vol)^(exp/6)
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
  getNuclExp6(const int exp) const
  {
    int indx = exp + 3;
    return nve6[indx];
  }

  /********************************************************************
    Moment source terms
  ********************************************************************/
  // Compute the soot moment and species source terms
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void computeSrcTerms(
    const amrex::Real& T,
    const amrex::Real& mu,
    const amrex::Real& rho,
    const amrex::Real& molarMass,
    const amrex::Real& convT,
    const amrex::Real& betaNucl,
    const amrex::Real& colConst,
    amrex::Real xi_n[],
    amrex::Real omega_src[],
    amrex::Real moments[],
    amrex::Real mom_src[],
    amrex::Real mom_fv[],
    const SootReaction* sr) const
  {
    // Molar concentration of the PAH inception species
    amrex::Real xi_PAH = xi_n[SootGasSpecIndx::indxPAH];
    // Compute the vector of factors used for moment interpolation
    computeFracMomVect(moments, mom_fv);
    // Compute the dimerization rate
    const amrex::Real dimerRate = sr->dimerRate(T, xi_PAH);
    // Estimate [DIMER]
    amrex::Real dimerConc = dimerization(convT, betaNucl, dimerRate, mom_fv);
    // Add the nucleation source term to mom_src
    nucleationMomSrc(betaNucl, dimerConc, mom_src);
    // Add the condensation source term to mom_src
    condensationMomSrc(colConst, dimerConc, mom_fv, mom_src);
    // Add the coagulation source term to mom_src
    coagulationMomSrc(colConst, T, mu, rho, molarMass, mom_fv, mom_src);
    amrex::Real surf = sc.S0 * fracMom(0., 1., mom_fv);
    // Reaction rates for surface growth (k_sg), oxidation (k_ox),
    // and fragmentation (k_o2)
    amrex::Real k_sg = 0.;
    amrex::Real k_ox = 0.;
    amrex::Real k_o2 = 0.;
    // Without enough soot the surface reactions are left out of both the
    // moment and the gas sources
    const bool surface = surfaceActive(moments);
    // Compute the species reaction source terms into omega_src
    sr->chemicalSrc(
      T, surf, xi_n, moments, k_sg, k_ox, k_o2, omega_src, surface);
    if (surface) {
      // Add the surface growth source to mom_src
      surfaceGrowthMomSrc(k_sg, mom_fv, mom_src);
      oxidFragMomSrc(k_ox, k_o2, mom_fv, mom_src);
    }
  }

  // Nucleation source term
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void nucleationMomSrc(
    const amrex::Real& betaNucl,
    const amrex::Real& dimerConc,
    amrex::Real mom_src[]) const
  {
    const amrex::Real dimerConc2 = dimerConc * dimerConc;
    for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
      mom_src[i] += 0.5 * betaNucl * dimerConc2 * momFact[i];
    }
    mom_src[NUM_SOOT_MOMENTS] += 0.5 * betaNucl * dimerConc2;
  }

  /** Compute condensation source values
      @param colConst Constant for free molecular collisions
      @param dimerConc Concentration of dimer
      @param momFV Vector of factors used in moment interpolation
      @param mom_src Moment source values
  */
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void condensationMomSrc(
    const amrex::Real& colConst,
    const amrex::Real& dimerConc,
    const amrex::Real momFV[],
    amrex::Real mom_src[]) const
  {
    amrex::Real weightDelta = momFV[NUM_SOOT_MOMENTS];
    for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
      const amrex::Real momV = sc.MomOrderV[i];
      const amrex::Real momS = sc.MomOrderS[i];
      const amrex::Real vv1 = momV + 2. * sc.SootAv;
      const amrex::Real vs1 = momS + 2. * sc.SootAs;
      const amrex::Real vv2 = momV + sc.SootAv;
      const amrex::Real vs2 = momS + sc.SootAs;
      amrex::Real volTerm =
        fracMom(vv1 - 1., vs1, momFV) * getDimerExp6(3) +
        2. * fracMom(vv2 - 1., vs2, momFV) * getDimerExp6(5) +
        fracMom(momV - 1., momS, momFV) * getDimerExp6(7) +
        0.5 * fracMom(vv1 - 2., vs1, momFV) * getDimerExp6(9) +
        fracMom(vv2 - 2., vs2, momFV) * getDimerExp6(11) +
        0.5 * fracMom(momV - 2., momS, momFV) * getDimerExp6(13);
      const amrex::Real ss3 = momS + 3. * sc.SootFitE;
      const amrex::Real sv3 = momV - 2. * sc.SootFitE;
      const amrex::Real ss2 = ss3 + sc.SootAs;
      const amrex::Real sv2 = sv3 + sc.SootAv;
      const amrex::Real ss1 = ss3 + 2. * sc.SootAs;
      const amrex::Real sv1 = sv3 + 2. * sc.SootAv;
      const amrex::Real surfTerm =
        fracMom(sv1 - 1., ss1, momFV) * getDimerExp6(3) +
        2. * fracMom(sv2 - 1., ss2, momFV) * getDimerExp6(5) +
        fracMom(sv3 - 1., ss3, momFV) * getDimerExp6(7) +
        0.5 * fracMom(sv1 - 2., ss1, momFV) * getDimerExp6(9) +
        fracMom(sv2 - 2., ss2, momFV) * getDimerExp6(11) +
        0.5 * fracMom(sv3 - 2., ss3, momFV) * getDimerExp6(13);
      mom_src[i] +=
        colConst * (momV * volTerm + sc.SootFitC * momS * surfTerm) * dimerConc;
    }
    // Source for the weight of the delta function
    mom_src[NUM_SOOT_MOMENTS] -= condFact * colConst * dimerConc * weightDelta;
  }

  // Surface growth source term
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void surfaceGrowthMomSrc(
    const amrex::Real& k_sg,
    const amrex::Real momFV[],
    amrex::Real mom_src[]) const
  {
    // Index of the weight of the delta function
    const int dwIndx = NUM_SOOT_MOMENTS;
    const amrex::Real weightDelta = momFV[dwIndx];
    const amrex::Real factor = sc.SootDensityC * sc.dVol * k_sg;
    for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
      amrex::Real fact1 =
        fracMom(sc.MomOrderV[i] - 1., sc.MomOrderS[i] + 1., momFV);
      amrex::Real fact2 = fracMom(
        sc.MomOrderV[i] - 1. - 2. * sc.SootFitE,
        sc.MomOrderS[i] + 1. + 3. * sc.SootFitE, momFV);
      mom_src[i] +=
        (sc.MomOrderV[i] * fact1 + sc.MomOrderS[i] * sc.SootFitC * fact2) *
        factor;
    }
    // Weight of the delta function
    mom_src[dwIndx] -= nuclSurf * k_sg * sc.SootDensityC * weightDelta;
  }

  // Oxidation and fragmentation source terms
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void oxidFragMomSrc(
    const amrex::Real& k_ox,
    const amrex::Real& k_o2,
    const amrex::Real momFV[],
    amrex::Real mom_src[]) const
  {
    // Index of the weight of the delta function
    const int dwIndx = NUM_SOOT_MOMENTS;
    const amrex::Real weightDelta = momFV[dwIndx];
    const amrex::Real factOx = k_ox * sc.dVol * sc.SootDensityC;
    const amrex::Real factO2 = 2. * k_o2 * sc.dVol * sc.SootDensityC;
    for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
      // Oxidation of the small particles
      amrex::Real small = -factOx * smallOF[i] * weightDelta;
      // Oxidation of the larger particles
      amrex::Real fracLarge =
        fracMomLarge(sc.MomOrderV[i] - 1., sc.MomOrderS[i] + 1., momFV);
      amrex::Real large =
        -factOx * (sc.MomOrderV[i] + 2. / 3. * sc.MomOrderS[i]) * fracLarge;
      // Add oxidation source
      mom_src[i] += large + small;
      // Add fragmentation source
      mom_src[i] += fragFact[i] * factO2 * fracLarge;
    }
    amrex::Real fracLarge = fracMomLarge(-1., 1., momFV);
    amrex::Real small = -factOx * smallOF[dwIndx] * weightDelta;
    amrex::Real inter =
      nuclVol / (fracMomLarge(1., 0., momFV) / fracMomLarge(0., 0., momFV));
    amrex::Real large = factOx * inter * fracLarge;
    // Add oxidation source for weight of delta function
    mom_src[dwIndx] += (small + large);
    // Add fragmentation source for weight of delta function
    mom_src[dwIndx] += inter * factO2 * fracLarge;
  }

  // Return the dimer concentration
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real dimerization(
    const amrex::Real& convT,
    const amrex::Real& betaNucl,
    const amrex::Real& dimerRate,
    const amrex::Real momFV[]) const
  {
    // Collision coefficient for condensation
    const amrex::Real betaCond = getBetaCond(convT, momFV);
    // Positive root of betaNucl*[DIMER]^2 + betaCond*[DIMER] - dimerRate = 0,
    // as 2c / (b + sqrt(b^2 + 4ac)), which does not cancel when b^2 >> 4ac;
    // denom is zero only when both coefficients are
    const amrex::Real delta = betaCond * betaCond + 4. * betaNucl * dimerRate;
    const amrex::Real denom = betaCond + std::sqrt(delta);
    return (denom == 0.) ? 0. : 2. * dimerRate / denom;
  }

  // Clip moment values
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
  clipMoments(amrex::Real moments[]) const
  {
    amrex::Real weightDelta = moments[NUM_SOOT_MOMENTS];
    const amrex::Real tolV = sc.smallWeight * nuclVol;
    const amrex::Real tolS = sc.smallWeight * nuclSurf;
    // Check for globally small moments
    if (moments[0] < sc.smallWeight || moments[1] < tolV || moments[2] < tolS) {
      moments[1] = amrex::max<amrex::Real>(moments[1], tolV);
      moments[0] = moments[1] / nuclVol;
      moments[2] = nuclSurf * moments[0];
      weightDelta = moments[0];
    }
    // Check for size of second mode
    if (
      moments[1] < nuclVol * moments[0] || moments[2] < nuclSurf * moments[0]) {
      moments[0] = moments[1] / nuclVol;
      moments[2] = moments[0] * nuclSurf;
    }
#if NUM_SOOT_MOMENTS == 6
    // Check for (co)variance of second mode
    moments[3] =
      amrex::max<amrex::Real>(moments[3], moments[1] * moments[1] / moments[0]);
    moments[4] =
      amrex::max<amrex::Real>(moments[4], moments[1] * moments[2] / moments[0]);
    moments[5] =
      amrex::max<amrex::Real>(moments[5], moments[2] * moments[2] / moments[0]);
#endif
    if (weightDelta < sc.smallWeight) {
      for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
        moments[i] += (sc.smallWeight - weightDelta) *
                      std::pow(nuclVol, sc.MomOrderV[i]) *
                      std::pow(nuclSurf, sc.MomOrderS[i]);
      }
      weightDelta = sc.smallWeight;
    }
    weightDelta = amrex::min<amrex::Real>(weightDelta, moments[0]);
    moments[NUM_SOOT_MOMENTS] = weightDelta;
  }

  // Convert from CGS/SI to mol, clip moments, and convert back
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
  momConvClipConv(amrex::Real mom[]) const
  {
    convertToMol(mom);
    clipMoments(mom);
    convertFromMol(mom);
  }

  // Compute the coagulation source term
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void coagulationMomSrc(
    const amrex::Real& colConst,
    const amrex::Real& T,
    const amrex::Real& mu,
    const amrex::Real& rho,
    const amrex::Real& molMass,
    const amrex::Real momFV[],
    amrex::Real mom_src[]) const
  {
    const amrex::Real tol = 0.;
    // Index of the weight of the delta function
    const int dwIndx = NUM_SOOT_MOMENTS;
    // Free molecular collision coefficient with van der Waals enhancements
    const amrex::Real C_fm = 2.2 * colConst;
    // Continuum collision coefficient
    const amrex::Real C_cn = 8. * pele::physics::Constants::RU * T / (3. * mu);
    // Mean free path for finite nudsen number correction in continuum regimes
    const amrex::Real lambda =
      3. * mu / rho *
      std::sqrt(M_PI * molMass / (8. * pele::physics::Constants::RU * T)) *
      lambdaCF;
    amrex::Real weightDelta2 = std::pow(momFV[dwIndx], 2);
    for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
      // Coagulation does not change total volume
      if (i == 1) {
        continue;
      }
      // Collisions between two first mode particles
      // Collision model: pure coalescence
      // S_(0+0) = (144*pi)^(1/3)*V0^(2/3)
      // Free molecular regime
      amrex::Real ss_fm = C_fm * ssfmCF[i] * weightDelta2;
      // Continuum regime
      amrex::Real ss_cn = 4. * C_cn * (1. + 1.257 * lambda * getNuclExp3(-1)) *
                          sscnCF[i] * weightDelta2;
      amrex::Real prodss = ss_fm * ss_cn;
      // Harmonic mean for transitional regime
      amrex::Real ss =
        (amrex::Math::abs(prodss) == tol) ? 0. : prodss / (ss_fm + ss_cn);

      // Collision between a particle in each mode
      // Collision model: "Splashing"
      // S_(i+0) = S_i + delta S
      // delta S = S*delta V / V *2/3*n_p^(-0.2043)
      // delta V = 2*W_C/rho_soot
      // Free molecular regime
      amrex::Real sl_fm = C_fm * FMCoagSL(i, momFV);
      // Continuum regime
      amrex::Real sl_cn = C_cn * CNCoagSL(i, lambda, momFV);
      amrex::Real prodsl = sl_fm * sl_cn;
      // Harmonic mean for transitional regime
      amrex::Real sl =
        (amrex::Math::abs(prodsl) == tol) ? 0. : prodsl / (sl_fm + sl_cn);

      // Collision between two second mode particles
      // Collision model: Pure aggregation
      // S_(i+j) = S_i + S_j
      // Free molecular regime
      amrex::Real ll_fm = C_fm * FMCoagLL(i, momFV);
      // Continuum regime
      amrex::Real ll_cn = C_cn * CNCoagLL(i, lambda, momFV);
      amrex::Real prodll = ll_fm * ll_cn;
      // Harmonic mean for transitional regime
      amrex::Real ll =
        (amrex::Math::abs(prodll) == tol) ? 0. : prodll / (ll_fm + ll_cn);
      mom_src[i] += (ss + sl + ll);
    }
    // Free molecular regime
    amrex::Real ss_fm = -C_fm * weightDelta2 * ssfmCF[dwIndx];
    // Continuum regime
    amrex::Real ss_cn =
      -4. * C_cn * (1. + 1.257 * lambda * getNuclExp3(-1)) * weightDelta2;
    // Harmonic mean for transitional regime
    amrex::Real prodss = ss_fm * ss_cn;
    amrex::Real ss =
      (amrex::Math::abs(prodss) == tol) ? 0. : prodss / (ss_fm + ss_cn);
    // Free molecular regime
    amrex::Real sl_fm = C_fm * FMCoagSL(dwIndx, momFV);
    // Continuum regime
    amrex::Real sl_cn = C_cn * CNCoagSL(dwIndx, lambda, momFV);
    // Harmonic mean for transitional regime
    amrex::Real prodsl = sl_fm * sl_cn;
    amrex::Real sl =
      (amrex::Math::abs(prodsl) == tol) ? 0. : prodsl / (sl_fm + sl_cn);
    mom_src[dwIndx] += (ss + sl);
  }

  /*********************************************************************
    Moment interpolation functions
  *********************************************************************/

  // Compute the moment interpolation array
  /*
    momFV contains factors for interpolating the moments
    It is ordered as the following
    momFV[0-NUM_SOOT_MOMENTS-1] - Corresponding factor for moment interpolation
    momFV[NUM_SOOT_MOMENTS] - Weight of the delta function
    momFV[NUM_SOOT_MOMENTS+1] - modeCoef
    modeCoef signifies the number of modes to be used
    If the moments are effectively zero, modeCoef = 0. and only 1 mode is used
    Otherwise, modeCoef = 1. and both modes are used
  */
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
  computeFracMomVect(const amrex::Real moments[], amrex::Real momFV[]) const
  {
    // See above for description of modeCoef
    amrex::Real modeCoef;
    // Copy over the weight of the delta function
    momFV[NUM_SOOT_MOMENTS] = moments[NUM_SOOT_MOMENTS];
#if NUM_SOOT_MOMENTS == 3
    const amrex::Real M00 = moments[0] - momFact[0] * moments[3];
    const amrex::Real M10 = moments[1] - momFact[1] * moments[3];
    const amrex::Real M01 = moments[2] - momFact[2] * moments[3];
    // If moments are effectively zero, only use one mode
    if (M00 < 1.E-25 || M10 < 1.E-25 || M01 < 1.E-25) {
      // Contribution from only one mode
      momFV[0] = moments[0];
      momFV[1] = moments[1];
      momFV[2] = moments[2];
      modeCoef = 0.;
    } else {
      // Contribution from both modes
      momFV[0] = M00;
      momFV[1] = M10;
      momFV[2] = M01;
      modeCoef = 1.;
    }
#elif NUM_SOOT_MOMENTS == 6
    const amrex::Real M00 = moments[0] - momFact[0] * moments[6];
    const amrex::Real M10 = moments[1] - momFact[1] * moments[6];
    const amrex::Real M01 = moments[2] - momFact[2] * moments[6];
    const amrex::Real M20 = moments[3] - momFact[3] * moments[6];
    const amrex::Real M11 = moments[4] - momFact[4] * moments[6];
    const amrex::Real M02 = moments[5] - momFact[5] * moments[6];
    amrex::Real minMom =
      amrex::min<amrex::Real>(M00, amrex::min<amrex::Real>(M10, M01));
    minMom = amrex::min<amrex::Real>(
      minMom, amrex::min<amrex::Real>(M20, amrex::min<amrex::Real>(M11, M02)));
    // If moments are effectively zero, only use one mode
    if (minMom < 1.E-25) {
      const amrex::Real c1 = std::pow(moments[0], -1.5);
      const amrex::Real c2 = std::pow(moments[0], 0.5);
      momFV[0] = moments[0];
      momFV[1] = std::pow(moments[1], 2.) * c1 * std::pow(moments[3], -0.5);
      momFV[2] = std::pow(moments[2], 2.) * c1 * std::pow(moments[5], -0.5);
      momFV[3] = std::pow(moments[3], 0.5) * c2 * std::pow(moments[1], -1.);
      momFV[4] = moments[4] * moments[0] / (moments[1] * moments[2]);
      momFV[5] = std::pow(moments[5], 0.5) * c2 * std::pow(moments[2], -1.);
      modeCoef = 0.;
    } else {
      const amrex::Real c1 = std::pow(M00, -1.5);
      const amrex::Real c2 = std::pow(M00, 0.5);
      momFV[0] = M00;
      momFV[1] = std::pow(M10, 2.) * c1 * std::pow(M20, -0.5);
      momFV[2] = std::pow(M01, 2.) * c1 * std::pow(M02, -0.5);
      momFV[3] = std::pow(M20, 0.5) * c2 * std::pow(M10, -1.);
      momFV[4] = M11 * M00 / (M10 * M01);
      momFV[5] = std::pow(M02, 0.5) * c2 * std::pow(M01, -1.);
      modeCoef = 1.;
    }
#endif
    momFV[NUM_SOOT_MOMENTS + 1] = modeCoef;
  }

  // Moment interpolation
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real fracMomLarge(
    const amrex::Real volOrd,
    const amrex::Real surfOrd,
    const amrex::Real momFV[]) const
  {
    // Weight of the delta function
    amrex::Real dwVal = momFV[NUM_SOOT_MOMENTS];
    amrex::Real factor =
      std::pow(nuclVol, volOrd) * std::pow(nuclSurf, surfOrd);
    if (momFV[NUM_SOOT_MOMENTS + 1] == 0.) {
      return factor * 1.E-60;
    }
    // Remove the contribution from the first mode
    amrex::Real outMom = fracMom(volOrd, surfOrd, momFV) - dwVal * factor;
    return outMom;
  }

  // Moment interpolation
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real fracMom(
    const amrex::Real volOrd,
    const amrex::Real surfOrd,
    const amrex::Real momFV[]) const
  {
    // If modeCoef = 0.; only first mode is used
    // If modeCoef = 1.; both modes are used
    const amrex::Real modeCoef = momFV[NUM_SOOT_MOMENTS + 1];
    amrex::Real bothPFact = momFV[NUM_SOOT_MOMENTS] *
                            std::pow(nuclVol, volOrd) *
                            std::pow(nuclSurf, surfOrd) * modeCoef;
#if NUM_SOOT_MOMENTS == 3
    amrex::Real peak = std::pow(momFV[0], 1. - volOrd - surfOrd) *
                       std::pow(momFV[1], volOrd) * std::pow(momFV[2], surfOrd);
    return bothPFact + peak;
#elif NUM_SOOT_MOMENTS == 6
    amrex::Real prod = momFV[0];
    prod *= std::pow(momFV[1], volOrd);
    prod *= std::pow(momFV[2], surfOrd);
    prod *= std::pow(momFV[3], volOrd * volOrd);
    prod *= std::pow(momFV[4], volOrd * surfOrd);
    prod *= std::pow(momFV[5], surfOrd * surfOrd);
    return bothPFact + prod;
#endif
  }

  // Interpolation for the reduced mass term (square root of sum) in the
  // collision kernel for collision between a particle in each mode
  // Only two grid functions used for all moments
  // Limited sensitivity to increasing the number of grid functions
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real psiSL(
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real a,
    const amrex::Real b,
    const amrex::Real momFV[]) const
  {
    const amrex::Real weightDelta = momFV[NUM_SOOT_MOMENTS];
    const amrex::Real factor = weightDelta * std::pow(nuclVol, a + 2. / 3. * b);
    const amrex::Real VF[3] = {2.0_rt * sc.SootAv + x, sc.SootAv + x, x};
    const amrex::Real SF[3] = {2.0_rt * sc.SootAs + y, sc.SootAs + y, y};
    const amrex::Real FML_1 = fracMomLarge(VF[0] - 0.5, SF[0], momFV);
    const amrex::Real FML_2 = fracMomLarge(VF[1] - 0.5, SF[1], momFV);
    const amrex::Real FML_3 = fracMomLarge(VF[2] - 0.5, SF[2], momFV);
    // nuclVolExp6[i] = nuclVol^(2*i - 3)/6
    amrex::Real psi1 =
      factor * (getNuclExp6(-3) * FML_1 + 2. * getNuclExp6(-1) * FML_2 +
                getNuclExp6(1) * FML_3);
    const amrex::Real FPL_1 = fracMomLarge(VF[0] + 0.5, SF[0], momFV);
    const amrex::Real FPL_2 = fracMomLarge(VF[1] + 0.5, SF[1], momFV);
    const amrex::Real FPL_3 = fracMomLarge(VF[2] + 0.5, SF[2], momFV);
    amrex::Real psi2_1 =
      factor * (getNuclExp6(-3) * FPL_1 + 2. * getNuclExp6(-1) * FPL_2 +
                getNuclExp6(1) * FPL_3);
    amrex::Real psi2_2 =
      factor * (getNuclExp6(3) * FML_1 + 2. * getNuclExp6(5) * FML_2 +
                getNuclExp6(7) * FML_3);
    return std::sqrt(psi1 * (psi2_1 + psi2_2));
  }

  // Interpolation for the reduced mass term (square root of sum) in the
  // collision kernel for collision between two particles in the second mode
  // Only two grid functions used for all moments
  // Limited sensitivity to increasing the number of grid functions
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real psiLL(
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real a,
    const amrex::Real b,
    const amrex::Real momFV[]) const
  {
    const amrex::Real VF_xy[3] = {2.0_rt * sc.SootAv + x, sc.SootAv + x, x};
    const amrex::Real SF_xy[3] = {2.0_rt * sc.SootAs + y, sc.SootAs + y, y};
    const amrex::Real VF_ab[3] = {a, sc.SootAv + a, 2.0_rt * sc.SootAv + a};
    const amrex::Real SF_ab[3] = {b, sc.SootAs + b, 2.0_rt * sc.SootAs + b};
    const amrex::Real xy_M[3] = {
      fracMomLarge(VF_xy[0] - 0.5, SF_xy[0], momFV),
      fracMomLarge(VF_xy[1] - 0.5, SF_xy[1], momFV),
      fracMomLarge(VF_xy[2] - 0.5, SF_xy[2], momFV)};
    const amrex::Real xy_P[3] = {
      fracMomLarge(VF_xy[0] + 0.5, SF_xy[0], momFV),
      fracMomLarge(VF_xy[1] + 0.5, SF_xy[1], momFV),
      fracMomLarge(VF_xy[2] + 0.5, SF_xy[2], momFV)};
    const amrex::Real ab_M[3] = {
      fracMomLarge(VF_ab[0] - 0.5, SF_ab[0], momFV),
      fracMomLarge(VF_ab[1] - 0.5, SF_ab[1], momFV),
      fracMomLarge(VF_ab[2] - 0.5, SF_ab[2], momFV)};
    const amrex::Real ab_P[3] = {
      fracMomLarge(VF_ab[0] + 0.5, SF_ab[0], momFV),
      fracMomLarge(VF_ab[1] + 0.5, SF_ab[1], momFV),
      fracMomLarge(VF_ab[2] + 0.5, SF_ab[2], momFV)};
    const amrex::Real psi1 =
      xy_M[0] * ab_M[0] + 2. * xy_M[1] * ab_M[1] + xy_M[2] * ab_M[2];
    const amrex::Real psi2_1 =
      xy_P[0] * ab_M[0] + 2. * xy_P[1] * ab_M[1] + xy_P[2] * ab_M[2];
    const amrex::Real psi2_2 =
      xy_M[0] * ab_P[0] + 2. * xy_M[1] * ab_P[1] + xy_M[2] * ab_P[2];
    return std::sqrt(psi1 * (psi2_1 + psi2_2));
  }

  // Free molecular coagulation source term
  // Small-Large: "Splashing"
  // -Generalized grid function follows terms
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
  FMCoagSL(const int i, const amrex::Real momFV[]) const
  {
    // Weight of delta function N0 and M00
    if (i == NUM_SOOT_MOMENTS || i == 0) {
      return -psiSL(0., 0., 0., 0., momFV);
    }
    const amrex::Real fact1 = -2. * sc.SootFitE;
    const amrex::Real fact2 = 3. * sc.SootFitE;
    switch (i) {
    case 1: // M10
      return 0.;
    case 2: // M01
      return sc.SootFitC * psiSL(fact1 - 1., fact2 + 1., 1., 0., momFV) -
             psiSL(0., 0., 0., 1., momFV);
    case 3: // M20
      return 2. * psiSL(1., 0., 1., 0., momFV);
    case 4: // M11
      return sc.SootFitC * psiSL(fact1, fact2 + 1., 1., 0., momFV) +
             psiSL(0., 1., 1., 0., momFV) +
             sc.SootFitC * psiSL(fact1 - 1., fact2 + 1., 2., 0., momFV) -
             psiSL(0., 0., 1., 1., momFV);
    case 5: // M02
      return 2. * sc.SootFitC * psiSL(fact1 - 1., fact2 + 2., 1., 0., momFV) +
             sc.SootFitC * sc.SootFitC *
               psiSL(2. * fact1 - 2., -3. * fact1 + 2., 2., 0., momFV) -
             psiSL(0., 0., 0., 2., momFV);
    default:
      amrex::Abort(
        "SootModel::FMCoagSL: Moment not contained in number of moments!");
      break;
    }
    return 0.;
  }

  // Free molecular coagulation source term
  // Large-Large: Pure aggregation
  // -Generalized grid function follows terms
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
  FMCoagLL(const int i, const amrex::Real momFV[]) const
  {
    switch (i) {
    case 0: // M00
      return -0.5 * psiLL(0., 0., 0., 0., momFV);
    case 1:
    case 2: // M10 M01
      return 0.;
    case 3: // M20
      return psiLL(1., 0., 1., 0., momFV);
    case 4: // M11
      return psiLL(1., 0., 0., 1., momFV);
    case 5: // M02
      return psiLL(0., 1., 0., 1., momFV);
    default:
      amrex::Abort(
        "SootModel::FMCoagLL: Moment not contained in number of moments!");
    }
    return 0.;
  }

  // Continuum coagulation source terms
  // Small-Large: "Splashing"
  // Large-Large: Pure aggregation
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real CNCoagSL(
    const int i, const amrex::Real& lambda, const amrex::Real momFV[]) const
  {
    const amrex::Real weightDelta = momFV[NUM_SOOT_MOMENTS];
    // Mean free path for finite Knudsen number correction in continuum regime
    if (i == NUM_SOOT_MOMENTS || i == 0) { // N0 or M00
      int n[] = {0, 1, -1, -2};
      amrex::Real x = 0.;
      amrex::Real y = 0.;
      return -weightDelta * CNCoagSLFunc(n, x, y, lambda, momFV);
    }
    switch (i) {
    case 1: // M10
      return 0.;
    case 2: // M01
    {
      amrex::Real p1, p2;
      {
        int n[] = {3, 4, 2, 1};
        amrex::Real x = -2. * sc.SootFitE - 1.;
        amrex::Real y = 3. * sc.SootFitE + 1.;
        p1 = sc.SootFitC * CNCoagSLFunc(n, x, y, lambda, momFV);
      }
      {
        int n[] = {2, 3, 1, 0};
        p2 = -CNCoagSLFunc(n, 0., 0., lambda, momFV);
      }
      return weightDelta * (p1 + p2);
    }
    case 3: // M20
    {
      int n[] = {3, 4, 2, 1};
      return 2. * weightDelta * CNCoagSLFunc(n, 1., 0., lambda, momFV);
    }
    case 4: // M11
    {
      amrex::Real p1, p2, p3, p4;
      {
        int n[] = {3, 4, 2, 1};
        amrex::Real x = -2. * sc.SootFitE;
        amrex::Real y = 3. * sc.SootFitE + 1.;
        p1 = sc.SootFitC * CNCoagSLFunc(n, x, y, lambda, momFV);
      }
      {
        int n[] = {3, 4, 2, 1};
        p2 = CNCoagSLFunc(n, 0., 1., lambda, momFV);
      }
      {
        int n[] = {6, 7, 5, 4};
        amrex::Real x = -2. * sc.SootFitE - 1.;
        amrex::Real y = 3. * sc.SootFitE + 1.;
        p3 = sc.SootFitC * CNCoagSLFunc(n, x, y, lambda, momFV);
      }
      {
        int n[] = {5, 6, 4, 3};
        p4 = -CNCoagSLFunc(n, 0., 0., lambda, momFV);
      }
      return weightDelta * (p1 + p2 + p3 + p4);
    }
    case 5: // M02
    {
      amrex::Real p1, p2, p3;
      {
        int n[] = {3, 4, 2, 1};
        amrex::Real x = -2. * sc.SootFitE - 1.;
        amrex::Real y = 3. * sc.SootFitE + 2.;
        p1 = 2. * sc.SootFitC * CNCoagSLFunc(n, x, y, lambda, momFV);
      }
      {
        int n[] = {6, 7, 5, 4};
        amrex::Real x = -4. * sc.SootFitE - 2.;
        amrex::Real y = 6. * sc.SootFitE + 2.;
        p2 = sc.SootFitC * sc.SootFitC * CNCoagSLFunc(n, x, y, lambda, momFV);
      }
      {
        int n[] = {4, 5, 3, 2};
        p3 = -CNCoagSLFunc(n, 0., 0., lambda, momFV);
      }
      return 2. * weightDelta * (p1 + p2 + p3);
    }
    default:
      return 0.;
    }
    return 0.;
  }

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real CNCoagSLFunc(
    int n[4],
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real& lambda,
    const amrex::Real momFV[]) const
  {
    amrex::Real xy_1 = fracMomLarge(x, y, momFV);
    amrex::Real xy_2 = fracMomLarge(x - sc.SootAv, y - sc.SootAs, momFV);
    amrex::Real xy_3 = fracMomLarge(x + sc.SootAv, y + sc.SootAs, momFV);
    amrex::Real xy_4 =
      fracMomLarge(x - 2. * sc.SootAv, y - 2. * sc.SootAs, momFV);
    amrex::Real n_1 = getNuclExp3(n[0]);
    amrex::Real n_2 = getNuclExp3(n[1]);
    amrex::Real n_3 = getNuclExp3(n[2]);
    amrex::Real n_4 = getNuclExp3(n[3]);
    return 2. * xy_1 * n_1 + xy_2 * n_2 + xy_3 * n_3 +
           1.257 * lambda * (xy_1 * n_3 + xy_2 * n_1 + xy_3 * n_4 + xy_4 * n_2);
  }

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real CNCoagLL(
    const int i, const amrex::Real& lambda, const amrex::Real momFV[]) const
  {
    switch (i) {
    case 0: // M00
      return -0.5 * CNCoagLLFunc(0., 0., lambda, momFV);
    case 1:
    case 2: // M10 M01
      return 0.;
    case 3: // M20
      return CNCoagLLFunc(1., 0., lambda, momFV);
    case 4: // M11
      return CNCoagLLFunc(1., 0., 0., 1., lambda, momFV);
    case 5: // M02
      return CNCoagLLFunc(0., 1., lambda, momFV);
    default:
      return 0.;
    }
    return 0.;
  }

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real CNCoagLLFunc(
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real& lambda,
    const amrex::Real momFV[]) const
  {
    const amrex::Real stav = sc.SootAv;
    const amrex::Real stas = sc.SootAs;
    amrex::Real xy_1 = fracMomLarge(x, y, momFV);
    amrex::Real xy_2 = fracMomLarge(x - stav, y - stas, momFV);
    amrex::Real xy_3 = fracMomLarge(x + stav, y + stas, momFV);
    amrex::Real xy_4 = fracMomLarge(x - 2. * stav, y - 2. * stas, momFV);
    return 2. * xy_1 * xy_1 + xy_2 * xy_3 + xy_3 * xy_2 +
           1.257 * lambda *
             (xy_1 * xy_2 + xy_2 * xy_1 + xy_3 * xy_4 + xy_4 * xy_3);
  }

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real CNCoagLLFunc(
    const amrex::Real x,
    const amrex::Real y,
    const amrex::Real a,
    const amrex::Real b,
    const amrex::Real& lambda,
    const amrex::Real momFV[]) const
  {
    const amrex::Real stav = sc.SootAv;
    const amrex::Real stas = sc.SootAs;
    amrex::Real xy_1 = fracMomLarge(x, y, momFV);
    amrex::Real xy_2 = fracMomLarge(x - stav, y - stas, momFV);
    amrex::Real xy_3 = fracMomLarge(x + stav, y + stas, momFV);
    amrex::Real xy_4 = fracMomLarge(x - 2. * stav, y - 2. * stas, momFV);
    amrex::Real ab_1 = fracMomLarge(a, b, momFV);
    amrex::Real ab_2 = fracMomLarge(a - stav, b - stas, momFV);
    amrex::Real ab_3 = fracMomLarge(a + stav, b + stas, momFV);
    amrex::Real ab_4 = fracMomLarge(a - 2. * stav, b - 2. * stas, momFV);
    return 2. * ab_1 * xy_1 + ab_2 * xy_3 + ab_3 * xy_2 +
           1.257 * lambda *
             (ab_1 * xy_2 + ab_2 * xy_1 + ab_3 * xy_4 + ab_4 * xy_3);
  }

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
  getBetaCond(const amrex::Real& convT, const amrex::Real momFV[]) const
  {
    // Collision frequency between two dimer in the free
    // molecular regime WITHOUT van der Waals enhancement
    // Units: 1/s
    const amrex::Real stav = sc.SootAv;
    const amrex::Real stas = sc.SootAs;
    const amrex::Real Cfm =
      sc.colFactPi23 * convT * sc.colFact16 * pele::physics::Constants::Avna;
    const amrex::Real SN =
      fracMom(2. * stav, 2. * stas, momFV) * getDimerExp6(-3) +
      2. * fracMom(stav, stas, momFV) * getDimerExp6(-1) +
      fracMom(0., 0., momFV) * getDimerExp6(1) +
      0.5 * fracMom(2. * stav - 1., 2. * stas, momFV) * getDimerExp6(3) +
      fracMom(stav - 1., stas, momFV) * getDimerExp6(5) +
      0.5 * fracMom(-1., 0., momFV) * getDimerExp6(7);
    return Cfm * SN;
  }
};

#endif
