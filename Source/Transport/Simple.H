#ifndef SIMPLETRANSPORT_H
#define SIMPLETRANSPORT_H

#include "TransportParams.H"
#include "TransportUtils.H"
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

// Visit every species pair (i,j), i<j, in the fixed order used historically
// by the mixture-averaged diffusivities (outer i ascending, inner j
// descending), calling f(i, j, binv_ij) with binv_ij the reciprocal of the
// binary diffusion coefficient of the pair. Returns the factor that converts
// W_i * (...) / sum_j X_j binv_ij into rho * D (CGS).
template <typename EOSType>
struct BinaryDiffInv
{
  template <typename F>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real operator()(
    const amrex::Real* /*Yloc*/,
    const amrex::Real* logT,
    const amrex::Real /*rholoc*/,
    const amrex::Real Tloc,
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
        f(i, j, dbintemp);
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

        amrex::Real Upsilonij = 0.0;
        for (int k = 0; k < NUM_SPECIES; ++k) {
          Upsilonij += tparm->Upsilonijk[idx_ijk + k] * Yloc[k];
        }
        // here we have a symmetry
        Upsilonij = Upsilonij * rholoc * Constants::Avna * M_PI / 12.0 + 1.0;
        dbintemp *= (Constants::RU * Tloc * Upsilonij) / Constants::PATM;
        f(i, j, dbintemp);
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
    amrex::Real* xiloc,
    TransParm<EosType, transport_type> const* tparm)
  {
    amrex::Real cvk[NUM_SPECIES] = {0.0};
    amrex::Real cvkint[NUM_SPECIES] = {0.0};
    amrex::Real cvkrot[NUM_SPECIES] = {0.0};
    amrex::Real FofT[NUM_SPECIES] = {0.0};
    amrex::Real Fnorm[NUM_SPECIES] = {0.0};
    amrex::Real pi3_2 = std::pow(M_PI, 1.5);

    CKCVMS(Tloc, cvk);

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
      amrex::Real epskoverTstd = tparm->eps[i] / 298.0;
      amrex::Real epskoverT = tparm->eps[i] / Tloc;
      Fnorm[i] = 1.0 + 0.50 * pi3_2 * std::sqrt(epskoverTstd) +
                 (2.0 + .50 * M_PI * M_PI) * epskoverTstd +
                 pi3_2 * std::sqrt(epskoverTstd) * epskoverTstd;

      FofT[i] = 1.0 + 0.50 * pi3_2 * std::sqrt(epskoverT) +
                (2.0 + .50 * M_PI * M_PI) * epskoverT +
                pi3_2 * std::sqrt(epskoverT) * epskoverT;
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

  // Shared implementation. DoMatrix = false compiles out every trace of the
  // multicomponent diffusion matrix (including its N^2 work arrays), so
  // callers of the historical API are unaffected.
  template <bool DoMatrix>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void transport_impl(
    const bool wtr_get_xi,
    const bool wtr_get_mu,
    const bool wtr_get_lam,
    const bool wtr_get_Ddiag,
    const bool wtr_get_chi,
    const bool wtr_get_Dmat,
    const amrex::Real Tloc,
    const amrex::Real rholoc,
    const amrex::Real* Yloc,
    amrex::Real* Ddiag,
    amrex::Real* chi_mix,
    amrex::Real* Dmat,
    amrex::Real& mu,
    amrex::Real& xi,
    amrex::Real& lam,
    TransParm<EosType, transport_type> const* tparm)
  {
    const amrex::Real trace = tparm->trace_epsilon;
    amrex::Real Xloc[NUM_SPECIES] = {0.0};
    amrex::Real Yloc_mod[NUM_SPECIES] = {0.0};
    amrex::Real muloc[NUM_SPECIES] = {0.0};
    amrex::Real xiloc[NUM_SPECIES] = {0.0};
    amrex::Real logT[NUM_FIT - 1] = {0.0};

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
    if (wtr_get_mu) {
      for (int i = 0; i < NUM_SPECIES; ++i) {
        muloc[i] = tparm->fitmu[4 * i] + tparm->fitmu[1 + 4 * i] * logT[0] +
                   tparm->fitmu[2 + 4 * i] * logT[1] +
                   tparm->fitmu[3 + 4 * i] * logT[2];
        muloc[i] = std::exp(muloc[i]);
      }

      mu = 0.0;

      for (int i = 0; i < NUM_SPECIES; ++i) {
        mu += Xloc[i] *
              (muloc[i] * muloc[i] * muloc[i] * muloc[i] * muloc[i] * muloc[i]);
      }
      mu = std::cbrt(std::sqrt(mu));

      //  assumption that we only get bulk viscosity if we are already getting
      //  shear viscosity

      if (wtr_get_xi) {
        comp_pure_bulk(Tloc, muloc, xiloc, tparm);
        xi = 0.0;
        for (int i = 0; i < NUM_SPECIES; ++i) {
          xi +=
            Xloc[i] * (std::sqrt(std::sqrt(xiloc[i] * xiloc[i] * xiloc[i])));
        }
        xi = std::cbrt(xi * xi * xi * xi);
      }
    }

    if (wtr_get_lam) {
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
    // Add nonideal corrections if necessary
    NonIdealChungCorrections<EosType>()(
      wtr_get_mu, wtr_get_lam, Tloc, Xloc, rholoc, wbar, mu, lam, tparm);

    // Species diffusion: the binary coefficients are evaluated once and used
    // for the mixture-averaged diffusivities and/or the full matrix.
    if constexpr (DoMatrix) {
      if (wtr_get_Ddiag || wtr_get_Dmat) {
        amrex::Real binv[NUM_BINARY_PAIRS] = {0.0};
        if (wtr_get_Ddiag) {
          for (int i = 0; i < NUM_SPECIES; ++i) {
            Ddiag[i] = 0.0;
          }
        }
        const amrex::Real scale = BinaryDiffInv<EosType>()(
          Yloc_mod, logT, rholoc, Tloc, tparm,
          [&](const int i, const int j, const amrex::Real b) {
            binv[binary_pair_index(i, j)] = b;
            if (wtr_get_Ddiag) {
              Ddiag[i] += Xloc[j] * b;
              Ddiag[j] += Xloc[i] * b;
            }
          });
        if (wtr_get_Ddiag) {
          finalize_mixture_averaged_diffusivities(
            Yloc_mod, scale, tparm->wt.data(), Ddiag);
        }
        if (wtr_get_Dmat) {
          multicomponent_diffusion_matrix(
            Xloc, Yloc_mod, binv, scale, tparm->matrix_diffusion_niter,
            tparm->wt.data(), Dmat);
        }
      }
    } else {
      amrex::ignore_unused(wtr_get_Dmat, Dmat);
      if (wtr_get_Ddiag) {
        for (int i = 0; i < NUM_SPECIES; ++i) {
          Ddiag[i] = 0.0;
        }
        const amrex::Real scale = BinaryDiffInv<EosType>()(
          Yloc_mod, logT, rholoc, Tloc, tparm,
          [&](const int i, const int j, const amrex::Real b) {
            Ddiag[i] += Xloc[j] * b;
            Ddiag[j] += Xloc[i] * b;
          });
        finalize_mixture_averaged_diffusivities(
          Yloc_mod, scale, tparm->wt.data(), Ddiag);
      }
    }

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
      Tloc, rholoc, Yloc, Ddiag, chi_mix, nullptr, mu, xi, lam, tparm);
  }

  // Extended API: additionally returns the multicomponent flux diffusion
  // matrix Dmat[i*NUM_SPECIES + j] = rho * Y_i * D_ij (see
  // multicomponent_diffusion_matrix). chi_mix must be zeroed by the caller.
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void transport(
    const bool wtr_get_xi,
    const bool wtr_get_mu,
    const bool wtr_get_lam,
    const bool wtr_get_Ddiag,
    const bool wtr_get_chi,
    const bool wtr_get_Dmat,
    const amrex::Real Tloc,
    const amrex::Real rholoc,
    const amrex::Real* Yloc,
    amrex::Real* Ddiag,
    amrex::Real* chi_mix,
    amrex::Real* Dmat,
    amrex::Real& mu,
    amrex::Real& xi,
    amrex::Real& lam,
    TransParm<EosType, transport_type> const* tparm)
  {
    transport_impl<true>(
      wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi,
      wtr_get_Dmat, Tloc, rholoc, Yloc, Ddiag, chi_mix, Dmat, mu, xi, lam,
      tparm);
  }

  template <bool DoMatrix>
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
  get_transport_coeffs_impl(
    amrex::Box const& bx,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& Rho_in,
    amrex::Array4<amrex::Real> const& D_out,
    amrex::Array4<amrex::Real> const& chi_out,
    amrex::Array4<amrex::Real> const& Dmat_out,
    amrex::Array4<amrex::Real> const& mu_out,
    amrex::Array4<amrex::Real> const& xi_out,
    amrex::Array4<amrex::Real> const& lam_out,
    TransParm<EosType, transport_type> const* tparm)
  {
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    const bool wtr_get_xi = true;
    const bool wtr_get_mu = true;
    const bool wtr_get_lam = true;
    const bool wtr_get_Ddiag = true;
    const bool wtr_get_chi = tparm->use_soret;

    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {

          const amrex::Real T = T_in(i, j, k);
          const amrex::Real rho = Rho_in(i, j, k);
          amrex::Real massloc[NUM_SPECIES] = {0.0};
          for (int n = 0; n < NUM_SPECIES; ++n) {
            massloc[n] = Y_in(i, j, k, n);
          }

          amrex::Real muloc, xiloc, lamloc;
          amrex::Real Ddiag[NUM_SPECIES] = {0.0};
          amrex::Real chi_mix[NUM_SPECIES] = {0.0};
          if constexpr (DoMatrix) {
            amrex::Real Dmat[NUM_SPECIES * NUM_SPECIES] = {0.0};
            transport_impl<true>(
              wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi,
              true, T, rho, massloc, Ddiag, chi_mix, Dmat, muloc, xiloc, lamloc,
              tparm);
            for (int n = 0; n < NUM_SPECIES * NUM_SPECIES; ++n) {
              Dmat_out(i, j, k, n) = Dmat[n];
            }
          } else {
            amrex::ignore_unused(Dmat_out);
            transport_impl<false>(
              wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi,
              false, T, rho, massloc, Ddiag, chi_mix, nullptr, muloc, xiloc,
              lamloc, tparm);
          }

          //   mu, xi and lambda are stored after D in the diffusion multifab
          for (int n = 0; n < NUM_SPECIES; ++n) {
            D_out(i, j, k, n) = Ddiag[n];
            if (wtr_get_chi) {
              chi_out(i, j, k, n) = chi_mix[n];
            }
          }
          mu_out(i, j, k) = muloc;
          xi_out(i, j, k) = xiloc;
          lam_out(i, j, k) = lamloc;
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
    get_transport_coeffs_impl<false>(
      bx, Y_in, T_in, Rho_in, D_out, chi_out, D_out, mu_out, xi_out, lam_out,
      tparm);
  }

  // Extended API: Dmat_out must have NUM_SPECIES*NUM_SPECIES components;
  // component i*NUM_SPECIES + j receives rho * Y_i * D_ij.
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void get_transport_coeffs(
    amrex::Box const& bx,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& Rho_in,
    amrex::Array4<amrex::Real> const& D_out,
    amrex::Array4<amrex::Real> const& chi_out,
    amrex::Array4<amrex::Real> const& Dmat_out,
    amrex::Array4<amrex::Real> const& mu_out,
    amrex::Array4<amrex::Real> const& xi_out,
    amrex::Array4<amrex::Real> const& lam_out,
    TransParm<EosType, transport_type> const* tparm)
  {
    AMREX_ASSERT(D_out.nComp() >= NUM_SPECIES);
    AMREX_ASSERT(Dmat_out.nComp() >= NUM_SPECIES * NUM_SPECIES);
    get_transport_coeffs_impl<true>(
      bx, Y_in, T_in, Rho_in, D_out, chi_out, Dmat_out, mu_out, xi_out, lam_out,
      tparm);
  }

  template <class... Args>
  AMREX_GPU_HOST_DEVICE explicit SimpleTransport(Args... /*unused*/)
  {
  }
};
} // namespace pele::physics::transport
#endif
