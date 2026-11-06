#ifndef SIMPLECONDUCTIVITY_H
#define SIMPLECONDUCTIVITY_H

#include <cmath>
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Algorithm.H>
#include "mechanism.H"
#include "PhysicsConstants.H"
#include "TransportUtils.H"

// Multicomponent thermal conductivity and rescaled thermal diffusion ratios
// (Ern & Giovangigli; EGlib EGSLCT1 / EGSLCT3). Opt-in: compiled with
// -DPELE_TRANSPORT_LCT (GNU make: USE_LCT_TRANSPORT = TRUE), because the
// runtime model switch puts the LCT work arrays into the local-memory
// footprint of every kernel that evaluates transport.
#ifdef PELE_TRANSPORT_LCT

namespace pele::physics::transport {

// Collision integral ratios A*, B*, C* as functions of s = ln(T / eps_ij):
// the universal degree-6 fits of CHEMKIN-III / EGlib (EGini.f FITA0/B0/C0).
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
collision_ratios(
  const amrex::Real s,
  amrex::Real& astar,
  amrex::Real& bstar,
  amrex::Real& cstar)
{
  astar = 0.1106910525e+01 +
          s * (-0.7065517161e-02 +
               s * (-0.1671975393e-01 +
                    s * (0.1188708609e-01 +
                         s * (0.7569367323e-03 + s * (-0.1313998345e-02 +
                                                      s * 0.1720853282e-03)))));
  bstar =
    0.1199673577e+01 +
    s * (-0.1140928763e+00 +
         s * (-0.2147636665e-02 +
              s * (0.2512965407e-01 +
                   s * (-0.3030372973e-02 +
                        s * (-0.1445009039e-02 + s * 0.2492954809e-03)))));
  cstar =
    0.8386993788e+00 +
    s * (0.4748325276e-01 +
         s * (0.3250097527e-01 +
              s * (-0.1625859588e-01 +
                   s * (-0.2260153363e-02 +
                        s * (0.1844922811e-02 + s * (-0.2115417788e-03))))));
}

// Species quantities of the LCT systems (EGlib LSPAR with IFLAG = 7):
//   cint_i = c_int,i / R      (cv_i W_i / R - 3/2; 0 for atoms)
//   cxi_i  = c_rot,i / (pi xi_i), xi_i = max(1, Z_rot,i) F(eps/298) / F(eps/T)
//   fac_i  = (5 P_atm / (3 T)) W_i / (R eta_i)   (= 2 A*_ii / D_ii)
//   binv_ii = 5 P_atm W_i / (6 R T A*_ii eta_i) (1 + 2985 / T^1.5 if polar),
// the "internal" self diffusion term, written into the diagonal slot of the
// packed binary coefficient array.
template <typename TParm>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
lct_species_setup(
  const amrex::Real T,
  const amrex::Real lnT,
  const amrex::Real* eta,
  const amrex::Real* cvk,
  TParm const* tparm,
  amrex::Real* binv,
  amrex::Real* cint,
  amrex::Real* cxi,
  amrex::Real* fac)
{
  const amrex::Real polar_fac = 1.0 + 2.985e3 / (T * std::sqrt(T));
  const amrex::Real fac0 = 5.0 * Constants::PATM / (3.0 * T * Constants::RU);
  for (int i = 0; i < NUM_SPECIES; ++i) {
    const amrex::Real wt = tparm->wt[i];
    amrex::Real crot = 0.0;
    if (tparm->nlin[i] == 0) {
      cint[i] = 0.0;
    } else {
      cint[i] = cvk[i] * wt / Constants::RU - 1.5;
      crot = (tparm->nlin[i] == 1) ? 1.0 : 1.5;
    }
    const amrex::Real xi_i = amrex::max<amrex::Real>(1.0, tparm->zrot[i]) *
                             parker_factor(tparm->eps[i] / 298.0) /
                             parker_factor(tparm->eps[i] / T);
    cxi[i] = crot / (xi_i * M_PI);
    fac[i] = fac0 * wt / eta[i];
    amrex::Real astar, bstar, cstar;
    collision_ratios(
      lnT - tparm->log_epsij[binary_pair_index(i, i)], astar, bstar, cstar);
    amrex::Real bii =
      5.0 * Constants::PATM * wt / (6.0 * Constants::RU * T * astar * eta[i]);
    if (tparm->dip[i] != 0.0) {
      bii *= polar_fac;
    }
    binv[binary_pair_index(i, i)] = bii;
  }
}

// Thermal conductivity from the solution a of the transport linear system
// with right-hand side beta: lambda = (P_atm / T) sum_k a_k beta_k
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
lct_conductivity(
  const int n,
  const amrex::Real* a,
  const amrex::Real* beta,
  const amrex::Real T)
{
  amrex::Real tc = 0.0;
  for (int k = 0; k < n; ++k) {
    tc += a[k] * beta[k];
  }
  return tc * Constants::PATM / T;
}

// LCT1: lambda from Lambda_[e] (one combined energy mode per species, EGlib
// EGSEMAE) with niter Jacobi-preconditioned CG iterations (EGSCG1). The
// matrix is never stored: the diagonal is accumulated once and the
// off-diagonal entries are recomputed in every matrix-vector product.
// On exit a[0..N) holds the solution.
template <typename TParm>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
conductivity_lct1(
  const int niter,
  const amrex::Real T,
  const amrex::Real lnT,
  const amrex::Real* X,
  const amrex::Real* binv,
  const amrex::Real* cint,
  const amrex::Real* cxi,
  const amrex::Real* fac,
  TParm const* tparm,
  amrex::Real* a)
{
  constexpr int N = NUM_SPECIES;
  const amrex::Real* wt = tparm->wt.data();
  amrex::Real diag[N], beta[N];
  for (int i = 0; i < N; ++i) {
    diag[i] = fac[i] * X[i] * X[i] * (1.0 + 8.0 * cxi[i] / 15.0) +
              X[i] * X[i] * cint[i] * binv[binary_pair_index(i, i)];
    beta[i] = (2.5 + cint[i]) * X[i];
  }
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      amrex::Real astar, bstar, cstar;
      const int p = binary_pair_index(i, j);
      collision_ratios(lnT - tparm->log_epsij[p], astar, bstar, cstar);
      const amrex::Real aaa = X[i] * X[j] * binv[p];
      const amrex::Real wi = wt[i] / (wt[i] + wt[j]);
      const amrex::Real wj = wt[j] / (wt[i] + wt[j]);
      const amrex::Real wioj = 3.0 * wt[i] / wt[j] - 2.0;
      const amrex::Real wjoi = 3.0 * wt[j] / wt[i] - 2.0;
      const amrex::Real bbj =
        7.5 * wi * wi + (6.25 - 3.0 * bstar) * wj * wj + 4.0 * astar * wi * wj +
        (4.0 / 15.0) * astar * cxi[i] * wi * wj * wioj * wioj +
        (20.0 / 3.0) * astar * cxi[j] * wi * wj;
      const amrex::Real bbi =
        7.5 * wj * wj + (6.25 - 3.0 * bstar) * wi * wi + 4.0 * astar * wi * wj +
        (4.0 / 15.0) * astar * cxi[j] * wi * wj * wjoi * wjoi +
        (20.0 / 3.0) * astar * cxi[i] * wi * wj;
      diag[i] += aaa * (bbj + cint[i]);
      diag[j] += aaa * (bbi + cint[j]);
    }
  }

  // y = Lambda_[e] z
  auto matvec = [&](const amrex::Real* z, amrex::Real* y) {
    for (int i = 0; i < N; ++i) {
      y[i] = diag[i] * z[i];
    }
    for (int i = 0; i < N; ++i) {
      for (int j = i + 1; j < N; ++j) {
        amrex::Real astar, bstar, cstar;
        const int p = binary_pair_index(i, j);
        collision_ratios(lnT - tparm->log_epsij[p], astar, bstar, cstar);
        const amrex::Real aaa = X[i] * X[j] * binv[p];
        const amrex::Real wi = wt[i] / (wt[i] + wt[j]);
        const amrex::Real wj = wt[j] / (wt[i] + wt[j]);
        const amrex::Real wioj = 3.0 * wt[i] / wt[j] - 2.0;
        const amrex::Real wjoi = 3.0 * wt[j] / wt[i] - 2.0;
        const amrex::Real off =
          -aaa * wi * wj *
          (13.75 - 3.0 * bstar - 4.0 * astar +
           (4.0 / 3.0) * astar * (wioj * cxi[i] + wjoi * cxi[j]));
        y[i] += off * z[j];
        y[j] += off * z[i];
      }
    }
  };

  // Jacobi-preconditioned conjugate gradient from a zero initial guess
  amrex::Real r[N], z[N], t[N];
  amrex::Real rMr = 0.0;
  for (int i = 0; i < N; ++i) {
    a[i] = 0.0;
    z[i] = 0.0;
    r[i] = beta[i];
    rMr += r[i] * r[i] / diag[i];
  }
  amrex::Real betan = 0.0;
  for (int it = 0; it < niter; ++it) {
    for (int i = 0; i < N; ++i) {
      z[i] = r[i] / diag[i] + betan * z[i];
    }
    matvec(z, t);
    amrex::Real zAz = 0.0;
    for (int i = 0; i < N; ++i) {
      zAz += z[i] * t[i];
    }
    if (zAz <= 0.0 || rMr <= 0.0) {
      break; // exact convergence (or loss of positivity): stop
    }
    const amrex::Real alpha = rMr / zAz;
    amrex::Real rMr_new = 0.0;
    for (int i = 0; i < N; ++i) {
      a[i] += alpha * z[i];
      r[i] -= alpha * t[i];
      rMr_new += r[i] * r[i] / diag[i];
    }
    betan = rMr_new / rMr;
    rMr = rMr_new;
  }
  return lct_conductivity(N, a, beta, T);
}

// LCT3: lambda from the full Lambda system (separate translational "10" and
// internal "01" energy modes per species, EGlib EGSEMA) with niter CG
// iterations preconditioned by the per-species 2x2 diagonal blocks
// (EGSCG2). Matrix-free as in conductivity_lct1. Unknown ordering:
// a[i] translational, a[N+i] internal. On exit a[0..2N) holds the solution.
template <typename TParm>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
conductivity_lct3(
  const int niter,
  const amrex::Real T,
  const amrex::Real lnT,
  const amrex::Real* X,
  const amrex::Real* binv,
  const amrex::Real* cint,
  const amrex::Real* cxi,
  const amrex::Real* fac,
  TParm const* tparm,
  amrex::Real* a)
{
  constexpr int N = NUM_SPECIES;
  constexpr int N2 = 2 * NUM_SPECIES;
  const amrex::Real* wt = tparm->wt.data();
  amrex::Real g11[N], g12[N], g22[N], beta[N2];
  for (int i = 0; i < N; ++i) {
    const amrex::Real fx2 = fac[i] * X[i] * X[i];
    g11[i] = fx2 * (1.0 + 10.0 * cxi[i] / 3.0);
    g12[i] = -fx2 * 2.0 * cxi[i];
    g22[i] = fx2 * 1.2 * cxi[i] +
             X[i] * X[i] * cint[i] * binv[binary_pair_index(i, i)];
    beta[i] = 2.5 * X[i];
    beta[N + i] = cint[i] * X[i];
  }
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      amrex::Real astar, bstar, cstar;
      const int p = binary_pair_index(i, j);
      collision_ratios(lnT - tparm->log_epsij[p], astar, bstar, cstar);
      const amrex::Real aaa = X[i] * X[j] * binv[p];
      const amrex::Real wi = wt[i] / (wt[i] + wt[j]);
      const amrex::Real wj = wt[j] / (wt[i] + wt[j]);
      const amrex::Real bb1 = (20.0 / 3.0) * astar * (cxi[i] + cxi[j]);
      const amrex::Real bbj = 7.5 * wi * wi + (6.25 - 3.0 * bstar) * wj * wj +
                              4.0 * astar * wi * wj + wi * wj * bb1;
      const amrex::Real bbi = 7.5 * wj * wj + (6.25 - 3.0 * bstar) * wi * wi +
                              4.0 * astar * wi * wj + wi * wj * bb1;
      g11[i] += bbj * aaa;
      g11[j] += bbi * aaa;
      g12[i] -= wi * 4.0 * astar * cxi[i] * aaa;
      g12[j] -= wj * 4.0 * astar * cxi[j] * aaa;
      g22[i] += aaa * (cint[i] + 2.4 * astar * (wt[i] / wt[j]) * cxi[i]);
      g22[j] += aaa * (cint[j] + 2.4 * astar * (wt[j] / wt[i]) * cxi[j]);
    }
  }
  for (int i = 0; i < N; ++i) {
    if (tparm->nlin[i] == 0) {
      g22[i] = 1.0; // atoms carry no internal energy
    }
  }

  // y = Lambda z
  auto matvec = [&](const amrex::Real* z, amrex::Real* y) {
    for (int i = 0; i < N; ++i) {
      y[i] = g11[i] * z[i] + g12[i] * z[N + i];
      y[N + i] = g12[i] * z[i] + g22[i] * z[N + i];
    }
    for (int i = 0; i < N; ++i) {
      for (int j = i + 1; j < N; ++j) {
        amrex::Real astar, bstar, cstar;
        const int p = binary_pair_index(i, j);
        collision_ratios(lnT - tparm->log_epsij[p], astar, bstar, cstar);
        const amrex::Real aaa = X[i] * X[j] * binv[p];
        const amrex::Real wi = wt[i] / (wt[i] + wt[j]);
        const amrex::Real wj = wt[j] / (wt[i] + wt[j]);
        const amrex::Real bb1 = (20.0 / 3.0) * astar * (cxi[i] + cxi[j]);
        const amrex::Real l1010 =
          -aaa * wi * wj * (13.75 - 3.0 * bstar - 4.0 * astar - bb1);
        const amrex::Real cci = -aaa * wi * 4.0 * astar * cxi[i]; // (N+i, j)
        const amrex::Real ccj = -aaa * wj * 4.0 * astar * cxi[j]; // (i, N+j)
        y[i] += l1010 * z[j] + ccj * z[N + j];
        y[j] += l1010 * z[i] + cci * z[N + i];
        y[N + i] += cci * z[j];
        y[N + j] += ccj * z[i];
      }
    }
  };

  // Inverse of the 2x2 diagonal blocks (fallback to the diagonal if the
  // block is not positive definite)
  amrex::Real m11[N], m12[N], m22[N];
  for (int i = 0; i < N; ++i) {
    const amrex::Real det = g11[i] * g22[i] - g12[i] * g12[i];
    if (det > 0.0) {
      m11[i] = g22[i] / det;
      m12[i] = -g12[i] / det;
      m22[i] = g11[i] / det;
    } else {
      m11[i] = 1.0 / g11[i];
      m12[i] = 0.0;
      m22[i] = 1.0 / g22[i];
    }
  }
  auto precond_dot = [&](const amrex::Real* r) {
    amrex::Real s = 0.0;
    for (int i = 0; i < N; ++i) {
      s += r[i] * (m11[i] * r[i] + m12[i] * r[N + i]) +
           r[N + i] * (m12[i] * r[i] + m22[i] * r[N + i]);
    }
    return s;
  };

  amrex::Real r[N2], z[N2], t[N2];
  for (int k = 0; k < N2; ++k) {
    a[k] = 0.0;
    z[k] = 0.0;
    r[k] = beta[k];
  }
  amrex::Real rMr = precond_dot(r);
  amrex::Real betan = 0.0;
  for (int it = 0; it < niter; ++it) {
    for (int i = 0; i < N; ++i) {
      z[i] = m11[i] * r[i] + m12[i] * r[N + i] + betan * z[i];
      z[N + i] = m12[i] * r[i] + m22[i] * r[N + i] + betan * z[N + i];
    }
    matvec(z, t);
    amrex::Real zAz = 0.0;
    for (int k = 0; k < N2; ++k) {
      zAz += z[k] * t[k];
    }
    if (zAz <= 0.0 || rMr <= 0.0) {
      break;
    }
    const amrex::Real alpha = rMr / zAz;
    for (int k = 0; k < N2; ++k) {
      a[k] += alpha * z[k];
      r[k] -= alpha * t[k];
    }
    const amrex::Real rMr_new = precond_dot(r);
    betan = rMr_new / rMr;
    rMr = rMr_new;
  }
  return lct_conductivity(N2, a, beta, T);
}

// Rescaled thermal diffusion ratios chit_i (EGlib EGSEVCT) from the
// translational components a[0..N) of the LCT solution:
//   chit_i = sum_{j != i} X_j (1/2) binv_ij (6 C*_ij - 5)
//            (W_i a_j - W_j a_i) / (W_i + W_j).
// The thermal diffusion ratios are chi_i = X_i chit_i and sum to zero.
template <typename TParm>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
lct_thermal_diffusion_ratios(
  const amrex::Real lnT,
  const amrex::Real* X,
  const amrex::Real* binv,
  const amrex::Real* a,
  TParm const* tparm,
  amrex::Real* chit)
{
  constexpr int N = NUM_SPECIES;
  const amrex::Real* wt = tparm->wt.data();
  for (int i = 0; i < N; ++i) {
    chit[i] = 0.0;
  }
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      amrex::Real astar, bstar, cstar;
      const int p = binary_pair_index(i, j);
      collision_ratios(lnT - tparm->log_epsij[p], astar, bstar, cstar);
      const amrex::Real term = 0.5 * binv[p] * (6.0 * cstar - 5.0) *
                               (wt[i] * a[j] - wt[j] * a[i]) / (wt[i] + wt[j]);
      chit[i] += X[j] * term;
      chit[j] -= X[i] * term;
    }
  }
}

// Entry point: thermal conductivity (erg/cm/s/K) and, if chit != nullptr,
// the rescaled thermal diffusion ratios, for conductivity model `model`
// (ConductivityModel::LCT1 or LCT3) with `niter` CG iterations.
//   X    trace-adjusted mole fractions
//   eta  pure species viscosities at T
//   cvk  species heat capacities at constant volume (erg/g/K, CKCVMS)
//   binv packed reciprocal binary diffusion coefficients at 1 atm; the
//        diagonal slots are overwritten with the internal self-diffusion term
template <typename TParm>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
multicomponent_conductivity(
  const int model,
  const int niter,
  const amrex::Real T,
  const amrex::Real lnT,
  const amrex::Real* X,
  const amrex::Real* eta,
  const amrex::Real* cvk,
  amrex::Real* binv,
  TParm const* tparm,
  amrex::Real& lam,
  amrex::Real* chit)
{
  amrex::Real cint[NUM_SPECIES], cxi[NUM_SPECIES], fac[NUM_SPECIES];
  amrex::Real a[2 * NUM_SPECIES];
  lct_species_setup(T, lnT, eta, cvk, tparm, binv, cint, cxi, fac);
  if (model == ConductivityModel::LCT3) {
    lam = conductivity_lct3(niter, T, lnT, X, binv, cint, cxi, fac, tparm, a);
  } else {
    lam = conductivity_lct1(niter, T, lnT, X, binv, cint, cxi, fac, tparm, a);
  }
  if (chit != nullptr) {
    lct_thermal_diffusion_ratios(lnT, X, binv, a, tparm, chit);
  }
}

} // namespace pele::physics::transport

#else // !PELE_TRANSPORT_LCT

namespace pele::physics::transport {
// Stub so that the (discarded) call sites in Simple.H still compile; the
// runtime option that would reach it is rejected at initialization.
template <typename TParm>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
multicomponent_conductivity(
  int /*model*/,
  int /*niter*/,
  amrex::Real /*T*/,
  amrex::Real /*lnT*/,
  const amrex::Real* /*X*/,
  const amrex::Real* /*eta*/,
  const amrex::Real* /*cvk*/,
  amrex::Real* /*binv*/,
  TParm const* /*tparm*/,
  amrex::Real& /*lam*/,
  amrex::Real* /*chit*/)
{
  amrex::Abort(
    "Multicomponent conductivity not compiled in "
    "(build with USE_LCT_TRANSPORT = TRUE / -DPELE_TRANSPORT_LCT)");
}
} // namespace pele::physics::transport

#endif // PELE_TRANSPORT_LCT
#endif
