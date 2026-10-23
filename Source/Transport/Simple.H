#ifndef SIMPLETRANSPORT_H
#define SIMPLETRANSPORT_H

#include "TransportParams.H"
#include "TransportUtils.H"
#include "SimpleConductivity.H"
#include "PhysicsConstants.H"

namespace pele::physics::transport {

template <typename EOSType>
struct NonIdealChungCorrections
{
  template <class... Args>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void operator()(Args... /*unused*/)
  {
  }
};

template <>
struct NonIdealChungCorrections<eos::SRK>
{
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  void operator()(
    const bool wtr_get_mu,
    const bool wtr_get_lam,
    const amrex::Real Temp,
    const amrex::Real* Xloc,
    const amrex::Real rholoc,
    const amrex::Real wbar,
    amrex::Real& mu,
    amrex::Real& lam,
    TransParm<eos::SRK, SimpleTransport> const* trans_parm)
  {
    amrex::Real sigma_M_3 = 0.0;
    amrex::Real Epsilon_M = 0.0;
    amrex::Real Omega_M = 0.0;
    amrex::Real MW_m = 0.0;
    amrex::Real DP_m_4 = 0.0;
    amrex::Real KappaM = 0.0;

    // Note: all the square roots are precalculated for efficiency

    for (int i = 0; i < NUM_SPECIES; ++i) {
      for (int j = 0; j < NUM_SPECIES; ++j) {
        const amrex::Real Xij = Xloc[i] * Xloc[j];
        const int idx = i * NUM_SPECIES + j;

        const amrex::Real sqrtT2 = trans_parm->sqrtT2ij[idx];
        const amrex::Real T2 = sqrtT2 * sqrtT2;
        const amrex::Real T3 = T2 * sqrtT2;
        sigma_M_3 += Xij * T3;

        const amrex::Real Epsilon_ij = trans_parm->sqrtEpsilonij[idx];
        Epsilon_M += Xij * Epsilon_ij * T3;

        const amrex::Real Omega_ij =
          0.5 * (trans_parm->omega[i] + trans_parm->omega[j]);
        Omega_M += Xij * Omega_ij * T3;

        MW_m += Xij * Epsilon_ij * T2 * trans_parm->sqrtMWij[idx];

        DP_m_4 += Xij * trans_parm->dip[i] * trans_parm->dip[i] *
                  trans_parm->dip[j] * trans_parm->dip[j] / (T3 * Epsilon_ij);

        KappaM += Xij * trans_parm->sqrtKappaij[idx];
      }
    }

    MW_m *= MW_m;
    const amrex::Real sigma_M = std::cbrt(sigma_M_3);
    const amrex::Real InvSigma1 = 1.0 / sigma_M;
    const amrex::Real InvSigma3 = 1.0 / sigma_M_3;
    Epsilon_M *= InvSigma3;
    const amrex::Real Tstar = Temp / Epsilon_M;
    const amrex::Real Tcm = 1.2593 * Epsilon_M;
    const amrex::Real Vcm = 1.8887 * sigma_M_3;
    Omega_M *= InvSigma3;
    MW_m = MW_m * InvSigma3 * InvSigma1 / (Epsilon_M * Epsilon_M);
    DP_m_4 = DP_m_4 * sigma_M_3 * Epsilon_M;
    const amrex::Real DP_red_4 =
      297.2069113e6 * DP_m_4 / (Vcm * Vcm * Tcm * Tcm);
    const amrex::Real y = Vcm * rholoc / (6.0 * wbar);
    const amrex::Real G1 =
      (1.0 - 0.5 * y) / ((1.0 - y) * (1.0 - y) * (1.0 - y));

    // Set nonideal viscosity
    if (wtr_get_mu) {
      amrex::Real A[10];
      for (int i = 0; i < 10; ++i) {
        A[i] = trans_parm->Afac[4 * i] + trans_parm->Afac[4 * i + 1] * Omega_M +
               trans_parm->Afac[4 * i + 2] * DP_red_4 +
               trans_parm->Afac[4 * i + 3] * KappaM;
      }
      const amrex::Real G2 = (A[0] * (1.0 - std::exp(-A[3] * y)) / y +
                              A[1] * G1 * std::exp(A[4] * y) + A[2] * G1) /
                             (A[0] * A[3] + A[1] + A[2]);
      const amrex::Real eta_P =
        (36.344e-6 * std::sqrt(MW_m * Tcm) / std::pow(Vcm, 2.0 / 3.0)) * A[6] *
        y * y * G2 * std::exp(A[7] + A[8] / Tstar + A[9] / (Tstar * Tstar));

      const amrex::Real updated_mu = mu * (1.0 / G2 + A[5] * y) + eta_P;
      if (updated_mu > 0) {
        mu = updated_mu;
      }
    }

    // Set nonideal conductivity
    if (wtr_get_lam) {
      amrex::Real B[7];
      for (int i = 0; i < 7; ++i) {
        B[i] = trans_parm->Bfac[i * 4] + trans_parm->Bfac[i * 4 + 1] * Omega_M +
               trans_parm->Bfac[i * 4 + 2] * DP_red_4 +
               trans_parm->Bfac[i * 4 + 3] * KappaM;
      }
      amrex::Real H2 = (B[0] * (1.0 - std::exp(-B[3] * y)) / y +
                        B[1] * G1 * std::exp(B[4] * y) + B[2] * G1) /
                       (B[0] * B[3] + B[1] + B[2]);
      amrex::Real lambda_p = 3.039e-4 * std::sqrt(Tcm / MW_m) /
                             std::pow(Vcm, 2.0 / 3.0) * B[6] * y * y * H2 *
                             std::sqrt(Tstar);

      lambda_p *= 4.184e+7; // erg/(cm s K)
      const amrex::Real beta = 1.0 / H2 + B[5] * y;

      const amrex::Real updated_lam = lam * beta + lambda_p;
      if (updated_lam > 0) {
        lam = updated_lam;
      }
    }
  }
};

// Visit every species pair (i,j), i<j, in the fixed order used historically
// by the mixture-averaged diffusivities (outer i ascending, inner j
// descending), calling f(i, j, b_atm, b_eff) with b_atm the reciprocal of the
// dilute-gas binary diffusion coefficient at 1 atm and b_eff the effective
// value used for species diffusion (identical for an ideal gas; density
// corrected for SRK, evaluated only when need_eff). Returns the factor that
// converts W_i * (...) / sum_j X_j b_eff_ij into rho * D (CGS).
template <typename EOSType>
struct BinaryDiffInv
{
  template <typename F>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
    const amrex::Real* /*Yloc*/,
    const amrex::Real* logT,
    const amrex::Real /*rholoc*/,
    const amrex::Real Tloc,
    const bool /*need_eff*/,
    TransParm<EOSType, SimpleTransport> const* tparm,
    F&& f) const
  {
    for (int i = 0; i < NUM_SPECIES; ++i) {
      for (int j = NUM_SPECIES - 1; j > i; --j) {
        const int four_idx_ij = 4 * (j * (j + 1) / 2 + i);
        amrex::Real dbintemp = tparm->fitdbin[four_idx_ij] +
                               tparm->fitdbin[four_idx_ij + 1] * logT[0] +
                               tparm->fitdbin[four_idx_ij + 2] * logT[1] +
                               tparm->fitdbin[four_idx_ij + 3] * logT[2];
        dbintemp = std::exp(-dbintemp);
        f(i, j, dbintemp, dbintemp);
      }
    }
    return Constants::PATM / (Constants::RU * Tloc);
  }
};

template <>
struct BinaryDiffInv<eos::SRK>
{
  template <typename F>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
    const amrex::Real* Yloc,
    const amrex::Real* logT,
    const amrex::Real rholoc,
    const amrex::Real Tloc,
    const bool need_eff,
    TransParm<eos::SRK, SimpleTransport> const* tparm,
    F&& f) const
  {
    for (int i = 0; i < NUM_SPECIES; ++i) {
      for (int j = NUM_SPECIES - 1; j > i; --j) {
        const int idx_ij = 4 * (j * (j + 1) / 2 + i);
        const int idx_ijk = (i + NUM_SPECIES * j) * NUM_SPECIES;
        amrex::Real dbintemp = tparm->fitdbin[idx_ij] +
                               tparm->fitdbin[idx_ij + 1] * logT[0] +
                               tparm->fitdbin[idx_ij + 2] * logT[1] +
                               tparm->fitdbin[idx_ij + 3] * logT[2];
        dbintemp = std::exp(-dbintemp);

        amrex::Real beff = dbintemp;
        if (need_eff) {
          amrex::Real Upsilonij = 0.0;
          for (int k = 0; k < NUM_SPECIES; ++k) {
            Upsilonij += tparm->Upsilonijk[idx_ijk + k] * Yloc[k];
          }
          // here we have a symmetry
          Upsilonij = Upsilonij * rholoc * Constants::Avna * M_PI / 12.0 + 1.0;
          beff =
            dbintemp * (Constants::RU * Tloc * Upsilonij) / Constants::PATM;
        }
        f(i, j, dbintemp, beff);
      }
    }
    return 1.0;
  }
};

// Finish the mixture-averaged (Hirschfelder-Curtiss) diffusivities once
// Ddiag[i] holds sum_{j != i} X_j binv_ij:
//   Ddiag[i] = rho * (W_i / Wbar) * (sum_{j != i} Y_j) / (sum_{j != i} X_j /
//   D_ij)
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
finalize_mixture_averaged_diffusivities(
  const amrex::Real* Yloc,
  const amrex::Real scale,
  const amrex::Real* wt,
  amrex::Real* Ddiag)
{
  for (int i = 0; i < NUM_SPECIES; ++i) {
    amrex::Real term1 = 0.0;
    for (int j = NUM_SPECIES - 1; j > i; --j) {
      term1 += Yloc[j];
    }
    for (int j = 0; j < i; ++j) {
      term1 += Yloc[j];
    }
    Ddiag[i] = wt[i] * term1 / Ddiag[i] * scale;
  }
}

// Multicomponent flux diffusion matrix (Ern & Giovangigli; EGlib EGSDR1).
//
// On exit Dmat[i*NUM_SPECIES + j] = rho * Y_i * D_ij in CGS (g/cm/s), where
// D_ij is the symmetric multicomponent diffusion matrix, such that the species
// mass fluxes are F_i = -sum_j Dmat_ij grad(X_j). The result is pressure
// independent and satisfies sum_i Dmat_ij = 0.
//
// Inputs are the trace-adjusted mole and mass fractions (all strictly
// positive, each summing to ~1), the packed reciprocal binary diffusion
// coefficients binv[binary_pair_index(i,j)], the density scale returned by
// BinaryDiffInv, the number of projected standard iterations on the
// Stefan-Maxwell matrix L_[00] (niter = 1 is Hirschfelder-Curtiss, niter = 2
// is EGlib's choice) and the species molecular weights.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
multicomponent_diffusion_matrix(
  const amrex::Real* Xloc,
  const amrex::Real* Yloc,
  const amrex::Real* binv,
  const amrex::Real scale,
  const int niter,
  const amrex::Real* wt,
  amrex::Real* Dmat)
{
  constexpr int N = NUM_SPECIES;

  // Diagonal of L_[00]: Gd_i = X_i sum_{j != i} X_j binv_ij, and
  // a_i = sum_{j != i} Y_j (summed rather than 1 - Y_i to stay accurate for
  // the dominant species). The splitting L_[00] = M - Z with
  // M = diag(Gd_i / a_i) makes the first iterate Hirschfelder-Curtiss.
  amrex::Real Gd[N] = {0.0};
  amrex::Real a[N] = {0.0};
  amrex::Real m[N] = {0.0};
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      const amrex::Real b = binv[binary_pair_index(i, j)];
      Gd[i] += Xloc[j] * b;
      Gd[j] += Xloc[i] * b;
      a[i] += Yloc[j];
      a[j] += Yloc[i];
    }
  }
  for (int i = 0; i < N; ++i) {
    Gd[i] *= Xloc[i];
    m[i] = a[i] / Gd[i];
  }

  // Symmetric iterate S^(k) = sum_{l<k} (M^-1 Z)^l M^-1, written into Dmat.
  // k = 1: diag(m); k = 2 (closed form, EGlib EGSSI1):
  //   S_ii = m_i (1 + Y_i),  S_ij = m_i m_j X_i X_j binv_ij
  if (niter <= 1) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        Dmat[i * N + j] = 0.0;
      }
      Dmat[i * N + i] = m[i];
    }
  } else {
    for (int i = 0; i < N; ++i) {
      Dmat[i * N + i] = m[i] * (1.0 + Yloc[i]);
      for (int j = i + 1; j < N; ++j) {
        const amrex::Real sij =
          m[i] * m[j] * Xloc[i] * Xloc[j] * binv[binary_pair_index(i, j)];
        Dmat[i * N + j] = sij;
        Dmat[j * N + i] = sij;
      }
    }
  }

  // Further iterations: S^(k) = diag(m) (I + Z S^(k-1)) with
  // Z_ij = X_i X_j binv_ij (i != j), Z_ii = Gd_i Y_i / a_i.
  if (niter > 2) {
    amrex::Real S[N * N];
    amrex::Real* cur = Dmat;
    amrex::Real* nxt = S;
    for (int it = 3; it <= niter; ++it) {
      for (int i = 0; i < N; ++i) {
        const amrex::Real zii = Gd[i] * Yloc[i] / a[i];
        for (int j = 0; j < N; ++j) {
          amrex::Real acc = (i == j) ? 1.0 : 0.0;
          acc += zii * cur[i * N + j];
          for (int k = 0; k < N; ++k) {
            if (k != i) {
              acc += Xloc[i] * Xloc[k] * binv[binary_pair_index(i, k)] *
                     cur[k * N + j];
            }
          }
          nxt[i * N + j] = m[i] * acc;
        }
      }
      amrex::Real* tmp = cur;
      cur = nxt;
      nxt = tmp;
    }
    if (cur != Dmat) {
      for (int n = 0; n < N * N; ++n) {
        Dmat[n] = S[n];
      }
    }
  }

  // Flux form: rho * Y_i * S_ij with rho * Y_i = scale * X_i * W_i, then the
  // mass-conserving projection P D P.
  for (int i = 0; i < N; ++i) {
    const amrex::Real fac = scale * Xloc[i] * wt[i];
    for (int j = 0; j < N; ++j) {
      Dmat[i * N + j] *= fac;
    }
  }
  project_diffusion_matrix(Yloc, Dmat);
}

struct SimpleTransport
{
  using transport_type = SimpleTransport;

  static std::string identifier() { return "SimpleTransport"; }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void comp_pure_bulk(
    amrex::Real Tloc,
    const amrex::Real* muloc,
    const amrex::Real* cvk,
    amrex::Real* xiloc,
    TransParm<EosType, transport_type> const* tparm)
  {
    amrex::Real cvkint[NUM_SPECIES] = {0.0};
    amrex::Real cvkrot[NUM_SPECIES] = {0.0};
    amrex::Real FofT[NUM_SPECIES] = {0.0};
    amrex::Real Fnorm[NUM_SPECIES] = {0.0};

    for (int i = 0; i < NUM_SPECIES; ++i) {
      if (tparm->nlin[i] == 0) {
        cvkint[i] = 0.0;
        cvkrot[i] = 0.0;
      } else if (tparm->nlin[i] == 1) {
        cvkint[i] = cvk[i] * tparm->wt[i] / Constants::RU - 1.50;
        cvkrot[i] = 1.0;
      } else {
        cvkint[i] = cvk[i] * tparm->wt[i] / Constants::RU - 1.50;
        cvkrot[i] = 1.5;
      }
    }

    for (int i = 0; i < NUM_SPECIES; ++i) {
      Fnorm[i] = parker_factor(tparm->eps[i] / 298.0);
      FofT[i] = parker_factor(tparm->eps[i] / Tloc);
    }

    for (int i = 0; i < NUM_SPECIES; ++i) {
      if (tparm->nlin[i] == 0) {
        xiloc[i] = 0.0;
      } else {
        //   zrot/crot approximately zint / cint by assuming vibrational
        //   internal energy is small cvkrot is scaled by wk / Ru = mk / kb
        //   relative to standard specific cv
        xiloc[i] =
          0.250 * M_PI *
          (cvkint[i] / (cvkint[i] + 1.50) * cvkint[i] / (cvkint[i] + 1.50)) *
          tparm->zrot[i] / cvkrot[i] * Fnorm[i] / FofT[i] * muloc[i];
      }
    }
  }

  // Shared implementation. DoMatrix = false compiles out the multicomponent
  // diffusion matrix and the rescaled thermal diffusion ratios (including
  // their N^2 work arrays), so callers of the historical API are only
  // affected by the runtime thermal conductivity model.
  template <bool DoMatrix>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void transport_impl(
    const bool wtr_get_xi,
    const bool wtr_get_mu,
    const bool wtr_get_lam,
    const bool wtr_get_Ddiag,
    const bool wtr_get_chi,
    const bool wtr_get_Dmat,
    const bool wtr_get_chit,
    const amrex::Real Tloc,
    const amrex::Real rholoc,
    const amrex::Real* Yloc,
    amrex::Real* Ddiag,
    amrex::Real* chi_mix,
    amrex::Real* Dmat,
    amrex::Real* chit,
    amrex::Real& mu,
    amrex::Real& xi,
    amrex::Real& lam,
    TransParm<EosType, transport_type> const* tparm)
  {
    const amrex::Real trace = tparm->trace_epsilon;
    amrex::Real Xloc[NUM_SPECIES] = {0.0};
    amrex::Real Yloc_mod[NUM_SPECIES] = {0.0};
    amrex::Real muloc[NUM_SPECIES] = {0.0};
    amrex::Real logT[NUM_FIT - 1] = {0.0};

    // Multicomponent (EGlib LCT) thermal conductivity requested?
#ifndef PELE_TRANSPORT_NO_LCT
    constexpr bool lct_compiled = true;
    const bool lct =
      (tparm->conductivity_model != ConductivityModel::Mixture) &&
      (wtr_get_lam || wtr_get_chit);
#else
    constexpr bool lct_compiled = false;
    const bool lct = false;
#endif
    if (
      wtr_get_chit && tparm->conductivity_model == ConductivityModel::Mixture) {
      amrex::Abort(
        "Rescaled thermal diffusion ratios (chit) require "
        "transport.conductivity_model = lct1 or lct3");
    }

    logT[0] = std::log(Tloc);
    logT[1] = logT[0] * logT[0];
    logT[2] = logT[0] * logT[1];

    // Modify Yloc to avoid potential divide by 0 for pure components
    amrex::Real sum = 0.0;
    for (int i = 0; i < NUM_SPECIES; ++i) {
      sum += Yloc[i];
    }
    for (int i = 0; i < NUM_SPECIES; ++i) {
      Yloc_mod[i] = Yloc[i] + trace * (sum / NUM_SPECIES - Yloc[i]);
    }

    amrex::Real wbar = 0.0;
    for (int i = 0; i < NUM_SPECIES; ++i) {
      wbar += Yloc_mod[i] * tparm->iwt[i];
    }
    wbar = 1.0 / wbar;
    for (int i = 0; i < NUM_SPECIES; ++i) {
      Xloc[i] = Yloc_mod[i] * wbar * tparm->iwt[i];
    }

    // Pure species viscosities: needed for mu, xi and the LCT systems
    if (wtr_get_mu || wtr_get_xi || lct) {
      for (int i = 0; i < NUM_SPECIES; ++i) {
        muloc[i] = tparm->fitmu[4 * i] + tparm->fitmu[1 + 4 * i] * logT[0] +
                   tparm->fitmu[2 + 4 * i] * logT[1] +
                   tparm->fitmu[3 + 4 * i] * logT[2];
        muloc[i] = std::exp(muloc[i]);
      }
    }
    if (wtr_get_mu) {
      mu = 0.0;
      for (int i = 0; i < NUM_SPECIES; ++i) {
        mu += Xloc[i] *
              (muloc[i] * muloc[i] * muloc[i] * muloc[i] * muloc[i] * muloc[i]);
      }
      mu = std::cbrt(std::sqrt(mu));
    }

    // Species heat capacities: needed for the bulk viscosity and the LCT
    // systems
    amrex::Real cvk[NUM_SPECIES] = {0.0};
    if (wtr_get_xi || lct) {
      CKCVMS(Tloc, cvk);
    }
    if (wtr_get_xi) {
      amrex::Real xiloc[NUM_SPECIES] = {0.0};
      comp_pure_bulk(Tloc, muloc, cvk, xiloc, tparm);
      xi = 0.0;
      for (int i = 0; i < NUM_SPECIES; ++i) {
        xi += Xloc[i] * (std::sqrt(std::sqrt(xiloc[i] * xiloc[i] * xiloc[i])));
      }
      xi = std::cbrt(xi * xi * xi * xi);
    }

    // Thermal conductivity, empirical mixture rule (default model)
    if (wtr_get_lam && !lct) {
      amrex::Real lamloc[NUM_SPECIES] = {0.0};
      for (int i = 0; i < NUM_SPECIES; ++i) {
        lamloc[i] = tparm->fitlam[4 * i] + tparm->fitlam[1 + 4 * i] * logT[0] +
                    tparm->fitlam[2 + 4 * i] * logT[1] +
                    tparm->fitlam[3 + 4 * i] * logT[2];
        lamloc[i] = std::exp(lamloc[i]);
      }

      lam = 0.0;
      for (int i = 0; i < NUM_SPECIES; ++i) {
        lam += Xloc[i] * std::sqrt(std::sqrt(lamloc[i]));
      }
      lam = lam * lam * lam * lam;
    }

    // Species diffusion and LCT conductivity: the binary coefficients are
    // evaluated once and shared. b_eff feeds Ddiag/Dmat, b_atm the LCT
    // systems.
    const bool need_eff = wtr_get_Ddiag || (DoMatrix && wtr_get_Dmat);
    if constexpr (DoMatrix) {
      if (need_eff || lct) {
        amrex::Real binv_eff[NUM_BINARY_PAIRS] = {0.0};
        amrex::Real binv_atm[NUM_BINARY_PAIRS] = {0.0};
        if (wtr_get_Ddiag) {
          for (int i = 0; i < NUM_SPECIES; ++i) {
            Ddiag[i] = 0.0;
          }
        }
        const amrex::Real scale = BinaryDiffInv<EosType>()(
          Yloc_mod, logT, rholoc, Tloc, need_eff, tparm,
          [&](
            const int i, const int j, const amrex::Real batm,
            const amrex::Real beff) {
            const int p = binary_pair_index(i, j);
            binv_eff[p] = beff;
            binv_atm[p] = batm;
            if (wtr_get_Ddiag) {
              Ddiag[i] += Xloc[j] * beff;
              Ddiag[j] += Xloc[i] * beff;
            }
          });
        if (wtr_get_Ddiag) {
          finalize_mixture_averaged_diffusivities(
            Yloc_mod, scale, tparm->wt.data(), Ddiag);
        }
        if (wtr_get_Dmat) {
          multicomponent_diffusion_matrix(
            Xloc, Yloc_mod, binv_eff, scale, tparm->matrix_diffusion_niter,
            tparm->wt.data(), Dmat);
        }
        if constexpr (lct_compiled) {
          if (lct) {
            amrex::Real lam_lct = 0.0;
            multicomponent_conductivity(
              tparm->conductivity_model, tparm->conductivity_niter, Tloc,
              logT[0], Xloc, muloc, cvk, binv_atm, tparm, lam_lct,
              wtr_get_chit ? chit : nullptr);
            if (wtr_get_lam) {
              lam = lam_lct;
            }
          }
        } else {
          amrex::ignore_unused(binv_atm, cvk);
        }
      }
    } else {
      amrex::ignore_unused(wtr_get_Dmat, Dmat, wtr_get_chit, chit);
      bool done_Ddiag = false;
      if constexpr (lct_compiled) {
        if (lct) {
          amrex::Real binv[NUM_BINARY_PAIRS] = {0.0};
          if (wtr_get_Ddiag) {
            for (int i = 0; i < NUM_SPECIES; ++i) {
              Ddiag[i] = 0.0;
            }
          }
          const amrex::Real scale = BinaryDiffInv<EosType>()(
            Yloc_mod, logT, rholoc, Tloc, need_eff, tparm,
            [&](
              const int i, const int j, const amrex::Real batm,
              const amrex::Real beff) {
              binv[binary_pair_index(i, j)] = batm;
              if (wtr_get_Ddiag) {
                Ddiag[i] += Xloc[j] * beff;
                Ddiag[j] += Xloc[i] * beff;
              }
            });
          if (wtr_get_Ddiag) {
            finalize_mixture_averaged_diffusivities(
              Yloc_mod, scale, tparm->wt.data(), Ddiag);
          }
          amrex::Real lam_lct = 0.0;
          multicomponent_conductivity(
            tparm->conductivity_model, tparm->conductivity_niter, Tloc, logT[0],
            Xloc, muloc, cvk, binv, tparm, lam_lct, nullptr);
          if (wtr_get_lam) {
            lam = lam_lct;
          }
          done_Ddiag = true;
        }
      } else {
        amrex::ignore_unused(cvk);
      }
      if (wtr_get_Ddiag && !done_Ddiag) {
        for (int i = 0; i < NUM_SPECIES; ++i) {
          Ddiag[i] = 0.0;
        }
        const amrex::Real scale = BinaryDiffInv<EosType>()(
          Yloc_mod, logT, rholoc, Tloc, true, tparm,
          [&](
            const int i, const int j, const amrex::Real /*batm*/,
            const amrex::Real beff) {
            Ddiag[i] += Xloc[j] * beff;
            Ddiag[j] += Xloc[i] * beff;
          });
        finalize_mixture_averaged_diffusivities(
          Yloc_mod, scale, tparm->wt.data(), Ddiag);
      }
    }

    // Add nonideal corrections if necessary (mu and lam are final here)
    NonIdealChungCorrections<EosType>()(
      wtr_get_mu, wtr_get_lam, Tloc, Xloc, rholoc, wbar, mu, lam, tparm);

    if (wtr_get_chi) {
      for (int n = 0; n < tparm->numLite; ++n) {
        const int i = tparm->liteSpec[n];
        for (int j = 0; j < NUM_SPECIES; ++j) {
          if (j != i) {
            const int four_idx_nj = 4 * (j + NUM_SPECIES * n);
            const amrex::Real chi =
              tparm->fittdrat[four_idx_nj] +
              Tloc * (tparm->fittdrat[1 + four_idx_nj] +
                      Tloc * (tparm->fittdrat[2 + four_idx_nj] +
                              Tloc * (tparm->fittdrat[3 + four_idx_nj])));
            chi_mix[i] += Xloc[j] * chi;
          }
        }
        chi_mix[i] *= Xloc[i];
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void transport(
    const bool wtr_get_xi,
    const bool wtr_get_mu,
    const bool wtr_get_lam,
    const bool wtr_get_Ddiag,
    const bool wtr_get_chi,
    const amrex::Real Tloc,
    const amrex::Real rholoc,
    const amrex::Real* Yloc,
    amrex::Real* Ddiag,
    amrex::Real* chi_mix,
    amrex::Real& mu,
    amrex::Real& xi,
    amrex::Real& lam,
    TransParm<EosType, transport_type> const* tparm)
  {
    transport_impl<false>(
      wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi, false,
      false, Tloc, rholoc, Yloc, Ddiag, chi_mix, nullptr, nullptr, mu, xi, lam,
      tparm);
  }

  // Extended API: evaluates the quantities selected in req and writes them
  // to the non-null members of out. In addition to the historical outputs it
  // provides the multicomponent flux diffusion matrix
  // Dmat[i*NUM_SPECIES + j] = rho * Y_i * D_ij (see
  // multicomponent_diffusion_matrix) and, for an LCT conductivity model, the
  // rescaled thermal diffusion ratios chit (see SimpleConductivity.H).
  // out.chi must be zeroed by the caller.
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void transport(
    const TransportRequest& req,
    const amrex::Real Tloc,
    const amrex::Real rholoc,
    const amrex::Real* Yloc,
    const TransportOutputs& out,
    TransParm<EosType, transport_type> const* tparm)
  {
    amrex::Real mu_l = 0.0, xi_l = 0.0, lam_l = 0.0;
    transport_impl<true>(
      req.xi, req.mu, req.lam, req.Ddiag && out.Ddiag != nullptr,
      req.chi && out.chi != nullptr, req.Dmat && out.Dmat != nullptr,
      req.chit && out.chit != nullptr, Tloc, rholoc, Yloc, out.Ddiag, out.chi,
      out.Dmat, out.chit, mu_l, xi_l, lam_l, tparm);
    if (out.mu != nullptr) {
      *out.mu = mu_l;
    }
    if (out.xi != nullptr) {
      *out.xi = xi_l;
    }
    if (out.lam != nullptr) {
      *out.lam = lam_l;
    }
  }

  template <bool DoMatrix>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
  get_transport_coeffs_impl(
    amrex::Box const& bx,
    const TransportRequest& req,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& Rho_in,
    amrex::Array4<amrex::Real> const& D_out,
    amrex::Array4<amrex::Real> const& chi_out,
    amrex::Array4<amrex::Real> const& Dmat_out,
    amrex::Array4<amrex::Real> const& chit_out,
    amrex::Array4<amrex::Real> const& mu_out,
    amrex::Array4<amrex::Real> const& xi_out,
    amrex::Array4<amrex::Real> const& lam_out,
    TransParm<EosType, transport_type> const* tparm)
  {
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {

          const amrex::Real T = T_in(i, j, k);
          const amrex::Real rho = Rho_in(i, j, k);
          amrex::Real massloc[NUM_SPECIES] = {0.0};
          for (int n = 0; n < NUM_SPECIES; ++n) {
            massloc[n] = Y_in(i, j, k, n);
          }

          amrex::Real muloc = 0.0, xiloc = 0.0, lamloc = 0.0;
          amrex::Real Ddiag[NUM_SPECIES] = {0.0};
          amrex::Real chi_mix[NUM_SPECIES] = {0.0};
          if constexpr (DoMatrix) {
            amrex::Real Dmat[NUM_SPECIES * NUM_SPECIES] = {0.0};
            amrex::Real chit[NUM_SPECIES] = {0.0};
            transport_impl<true>(
              req.xi, req.mu, req.lam, req.Ddiag, req.chi, req.Dmat, req.chit,
              T, rho, massloc, Ddiag, chi_mix, Dmat, chit, muloc, xiloc, lamloc,
              tparm);
            if (req.Dmat) {
              for (int n = 0; n < NUM_SPECIES * NUM_SPECIES; ++n) {
                Dmat_out(i, j, k, n) = Dmat[n];
              }
            }
            if (req.chit) {
              for (int n = 0; n < NUM_SPECIES; ++n) {
                chit_out(i, j, k, n) = chit[n];
              }
            }
          } else {
            amrex::ignore_unused(Dmat_out, chit_out);
            transport_impl<false>(
              req.xi, req.mu, req.lam, req.Ddiag, req.chi, false, false, T, rho,
              massloc, Ddiag, chi_mix, nullptr, nullptr, muloc, xiloc, lamloc,
              tparm);
          }

          //   mu, xi and lambda are stored after D in the diffusion multifab
          for (int n = 0; n < NUM_SPECIES; ++n) {
            if (req.Ddiag) {
              D_out(i, j, k, n) = Ddiag[n];
            }
            if (req.chi) {
              chi_out(i, j, k, n) = chi_mix[n];
            }
          }
          if (req.mu) {
            mu_out(i, j, k) = muloc;
          }
          if (req.xi) {
            xi_out(i, j, k) = xiloc;
          }
          if (req.lam) {
            lam_out(i, j, k) = lamloc;
          }
        }
      }
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void get_transport_coeffs(
    amrex::Box const& bx,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& Rho_in,
    amrex::Array4<amrex::Real> const& D_out,
    amrex::Array4<amrex::Real> const& chi_out,
    amrex::Array4<amrex::Real> const& mu_out,
    amrex::Array4<amrex::Real> const& xi_out,
    amrex::Array4<amrex::Real> const& lam_out,
    TransParm<EosType, transport_type> const* tparm)
  {
    TransportRequest req;
    req.xi = true;
    req.mu = true;
    req.lam = true;
    req.Ddiag = true;
    req.chi = tparm->use_soret;
    get_transport_coeffs_impl<false>(
      bx, req, Y_in, T_in, Rho_in, D_out, chi_out, D_out, D_out, mu_out, xi_out,
      lam_out, tparm);
  }

  // Extended API (see the TransportRequest overload of transport()).
  // Dmat_out needs NUM_SPECIES*NUM_SPECIES components (component
  // i*NUM_SPECIES + j receives rho * Y_i * D_ij) and chit_out NUM_SPECIES
  // components; outputs that are not requested may be any Array4.
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void get_transport_coeffs(
    amrex::Box const& bx,
    const TransportRequest& req,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& Rho_in,
    amrex::Array4<amrex::Real> const& D_out,
    amrex::Array4<amrex::Real> const& chi_out,
    amrex::Array4<amrex::Real> const& Dmat_out,
    amrex::Array4<amrex::Real> const& chit_out,
    amrex::Array4<amrex::Real> const& mu_out,
    amrex::Array4<amrex::Real> const& xi_out,
    amrex::Array4<amrex::Real> const& lam_out,
    TransParm<EosType, transport_type> const* tparm)
  {
    AMREX_ASSERT(!req.Ddiag || D_out.nComp() >= NUM_SPECIES);
    AMREX_ASSERT(!req.chi || chi_out.nComp() >= NUM_SPECIES);
    AMREX_ASSERT(!req.Dmat || Dmat_out.nComp() >= NUM_SPECIES * NUM_SPECIES);
    AMREX_ASSERT(!req.chit || chit_out.nComp() >= NUM_SPECIES);
    get_transport_coeffs_impl<true>(
      bx, req, Y_in, T_in, Rho_in, D_out, chi_out, Dmat_out, chit_out, mu_out,
      xi_out, lam_out, tparm);
  }

  template <class... Args>
  AMREX_GPU_HOST_DEVICE explicit SimpleTransport(Args... /*unused*/)
  {
  }
};
} // namespace pele::physics::transport
#endif
