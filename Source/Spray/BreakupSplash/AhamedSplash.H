
#ifndef AHAMEDSPLASH_H
#define AHAMEDSPLASH_H

#include "SprayFuelData.H"
#include "SBData.H"

/****************************************************************
 Functions for applying the splash model developed by Ahamed et al. Atomization
 and Sprays, 32(4): 1-23 (2022)
 ***************************************************************/

enum splash_type {
  rebound = 0,
  deposit,
  splash_dry,
  thermal_breakup,
  spread,
  splash_wet
};

// Find tangents along surface
AMREX_GPU_HOST_DEVICE
AMREX_INLINE
void
find_tangents(
  const amrex::RealVect& pvel,
  amrex::RealVect& tanPsi,
  const amrex::RealVect& norm,
  amrex::RealVect& tanBeta)
{
#if AMREX_SPACEDIM == 3
  amrex::RealVect testvec = -pvel;
  // Check if directions of norm and velocity are the same
  if (testvec.crossProduct(norm).vectorLength() < 1.E-5) {
    // If so, pick an arbitrary direction
    testvec += {1., 1., 1.};
  }
  tanPsi = testvec.crossProduct(norm);
  tanPsi /= tanPsi.vectorLength();
  tanBeta = tanPsi.crossProduct(norm);
  tanBeta /= tanBeta.vectorLength();
#else
  amrex::ignore_unused(pvel, tanPsi);
  tanBeta[0] = -norm[1];
  tanBeta[1] = norm[0];
#endif
}

// According to the reference, four splashed droplets are formed
inline void
get_splash_vels(
  const amrex::Real U0norm,
  const amrex::Real U0tan,
  const amrex::Real Kv,
  const amrex::Real del_film,
  amrex::Real& uBeta_0,
  amrex::Real& uBeta_half,
  amrex::Real& uBeta_pi,
  amrex::Real& uPsi_coeff,
  amrex::Real& usNorm)
{
  amrex::Real Un_mag = std::abs(U0norm);
  if (del_film < 0.01) {
    amrex::Real Ut_splash = 0.0065 * Kv * std::exp(-0.004 * Kv) * Un_mag;
    uBeta_0 = U0tan + Ut_splash;
    uBeta_half = std::cos(M_PI / 6.) * U0tan;
    uBeta_pi = std::cos(M_PI / 3.) * U0tan - Ut_splash;
    uPsi_coeff = 0.2 * U0tan + Ut_splash;
    usNorm = 0.0028 * Kv * std::exp(-0.0062 * Kv) * Un_mag;
  } else if (del_film < 0.1) {
    amrex::Real Ut_splash;
    if (del_film > 0.05) {
      Ut_splash = (5.E-5 * Kv + 0.0203) * Un_mag;
      usNorm = 0.03 * U0tan + (1.32E-4 * Kv + 0.0158) * Un_mag;
    } else {
      Ut_splash = (3.45E-5 * Kv + 0.006) * Un_mag;
      usNorm = 0.03 * U0tan + (3.57E-5 * Kv - 0.0057) * Un_mag;
    }
    uBeta_0 = 0.3 * U0tan + Ut_splash;
    uBeta_half = 0.1 * U0tan;
    uBeta_pi = 0.033 * U0tan - Ut_splash;
    uPsi_coeff = 0.06 * U0tan + Ut_splash;
  } else {
    uBeta_0 = 0.1 * U0tan;
    uBeta_half = 0.05 * U0tan;
    uBeta_pi = 0.01 * U0tan;
    uPsi_coeff = 0.02 * U0tan;
    usNorm = 0.03 * U0tan;
  }
}

inline void
get_ms_theta(
  const amrex::Real alpha,
  const amrex::Real ms,
  const amrex::Real del_film,
  amrex::Real* ms_thetas)
{
  // Indices: 0: psi = -pi/2, 1: psi = 0, 2: psi = pi/2, 3: psi = pi
  if (del_film > 0.) {
    amrex::Real alpha_d = alpha * 180. / M_PI;
    amrex::Real ms_psi_0, ms_psi_pih;
    if (del_film > 0.1) {
      ms_psi_0 = (1.9E-4 * alpha_d * alpha_d + 0.0202 * alpha_d - 0.0107) * ms;
      ms_psi_pih =
        (6.4E-5 * alpha_d * alpha_d - 7.6E-3 * alpha_d + 0.4176) * ms;
    } else {
      ms_psi_0 = (-2.3E-4 * alpha_d * alpha_d + 0.0236 * alpha_d) * ms;
      ms_psi_pih = (6.E-5 * alpha_d * alpha_d - 6.8E-3 * alpha_d + 0.3619) * ms;
    }
    ms_thetas[0] = ms_psi_pih;
    ms_thetas[1] = ms_psi_0;
    ms_thetas[2] = ms_psi_pih;
    ms_thetas[3] = ms - ms_psi_0 - 2. * ms_psi_pih;
  } else {
    amrex::Real tanterm = 0.4 * std::tan(M_PI / 2. - alpha);
    ms_thetas[0] = 0.25 * ms;
    ms_thetas[1] = 0.25 * ms * (1. + tanterm);
    ms_thetas[2] = 0.25 * ms;
    ms_thetas[3] = 0.25 * ms * (1. - tanterm);
  }
}

// Determine the type of particle impact at the wall
AMREX_GPU_DEVICE
AMREX_INLINE
splash_type
splash_criteria(
  const amrex::Real Kv,
  const amrex::Real Tstar,
  const amrex::Real alpha,
  const amrex::Real Kcrit,
  const amrex::Real del_film,
  splash_breakup& N_SB,
  amrex::Real& ms,
  amrex::Real& dm)
{
  splash_type crit = splash_type::thermal_breakup;
  if (del_film < 0.01) {
    if (Tstar < 1.1) {
      if (Kv < Kcrit) {
        crit = splash_type::deposit;
        ms = 0.; // No splashing mass
        dm = 1.; // All mass is deposited
      } else {
        crit = splash_type::splash_dry;
        ms = 0.75 * (1. - std::exp(-3.E-3 * (Kv - Kcrit)));
        dm = 1. - ms;
        N_SB = splash_breakup::splash_dry_splash;
      }
    } else {
      if (Kv < Kcrit) {
        crit = splash_type::rebound;
      } else {
        crit = splash_type::thermal_breakup;
        ms = 0.75 * (1. - std::exp(-0.0175 * (Kv - Kcrit)));
        dm = 0.;
        N_SB = splash_breakup::splash_thermal_breakup;
      }
    }
  } else {
    amrex::Real Kcrit_wet = Kcrit + 226. * std::pow(del_film, 0.72);
    if (Kv > Kcrit_wet) {
      crit = splash_type::splash_wet;
      amrex::Real A = 4.5;
      amrex::Real B = 0.2;
      if (del_film < 0.5) {
        A = 10.42 * del_film + 0.32;
        B = 8.75 * del_film - 0.107;
      }
      amrex::Real ms0 = A * (1. - std::exp(-0.003 * (Kv - Kcrit_wet))) + B;
      amrex::Real alpha_d = alpha * 180. / M_PI;
      if (del_film > 0.1) {
        if (alpha_d > 45.) {
          ms = ms0 * (-0.0305 * alpha_d + 3.7496);
        } else {
          ms = ms0 * 0.0525 * alpha_d;
        }
      } else {
        if (alpha_d > 45.) {
          ms = ms0 * (-7.2E-3 * alpha_d + 1.6389);
        } else {
          ms = ms0 * (-4.6E-4 * alpha_d * alpha_d + 0.05 * alpha_d);
        }
      }
      // Do not allow mass to exceed one for now, should be fixed
      ms = amrex::min<amrex::Real>(ms, 1.);
      // TODO: get_ms_theta splits a wet splash into negative masses (at
      // normal impact the psi = 0 share is 3.35 ms, leaving 1 - 3.35 - 0.5
      // ms at psi = pi), so the splashed drops are not created yet and the
      // whole droplet joins the film rather than being lost
      ms = 0.;
      dm = 1.;
    } else {
      crit = splash_type::spread;
      // All mass becomes part of the wall film
      ms = 0.;
      dm = 1.;
    }
  }
  return crit;
}

// Determine if droplet is splashing, fill SB data if it is
AMREX_GPU_DEVICE
AMREX_INLINE
void
droplet_splashing(
  SprayParticleContainer::ParticleType& p,
  int pid,
  const amrex::RealVect& /*dx*/,
  const amrex::RealVect& /*plo*/,
  const SprayData& fdat,
  const amrex::Real p_gas,
  const amrex::RealVect& vel_part,
  const amrex::Real par_dot,
  const amrex::RealVect& normal,
  const amrex::Real* cBoilT,
  splash_breakup* N_SB,
  const SBPtrs& rf,
  const amrex::Real film_h)
{
  // Velocity magnitude of droplet
  amrex::Real U0mag = vel_part.vectorLength();
  // Normal velocity magnitude of droplet relative to wall
  amrex::Real U0norm = normal.dotProduct(vel_part);
  // Magnitude of tangential velocity relative to wall
  amrex::Real U0tan = std::sqrt(U0mag * U0mag - U0norm * U0norm);
  amrex::Real T_part = p.rdata(SprayComps::pstateT);
  amrex::Real num_dens = p.rdata(SprayComps::pstateNumDens);
  amrex::Real Y_part[SPRAY_FUEL_NUM] = {0.0};
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    Y_part[spf] = p.rdata(SprayComps::pstateY + spf);
  }
  // TODO: Determine correct method for handling multi-component liquids
  const amrex::Real Tboil = fdat.liqprops.boilT_mix(Y_part, cBoilT);
  const amrex::Real rho_part = fdat.liqprops.rho_mix(Y_part, T_part, cBoilT);
  const amrex::Real mu_part = fdat.liqprops.mu_mix(Y_part, T_part, cBoilT);
  const amrex::Real sigma_part =
    fdat.liqprops.sigma_mix(Y_part, T_part, cBoilT);
  amrex::Real Tstar = fdat.wall_T / Tboil;
  const amrex::Real dia_part = p.rdata(SprayComps::pstateDia);
  const amrex::Real pmass = M_PI / 6. * rho_part * std::pow(dia_part, 3);
  // Weber number
  const amrex::Real We = rho_part * dia_part * U0norm * U0norm / sigma_part;
  const amrex::Real Re_L = std::abs(U0norm) * dia_part * rho_part / mu_part;
  const amrex::Real Kv = std::sqrt(We * std::sqrt(Re_L));
  // Inclination of on-coming particle to surface in radians
  amrex::Real alpha = amrex::max<amrex::Real>(
    M_PI / 6., std::asin(amrex::Math::abs(U0norm) / U0mag));
#ifdef PELELM_USE_SPRAY
  const amrex::Real p_bar = p_gas * 1.E-5;
#else
  const amrex::Real p_bar = p_gas * 1.E-6;
#endif
  const amrex::Real Kcrit = (26.788 * std::log(p_bar) + 33.546) *
                              std::exp(-6.E-3 * (fdat.wall_T - 260.)) +
                            36.;
  const amrex::Real del_film = film_h / dia_part;
  // Ratio of mass that splashes
  amrex::Real ms = 0.;
  // Ratio of mass that deposits into wall film
  amrex::Real dm = 0.;
  amrex::RealVect impact_loc = p.pos() - par_dot * vel_part / U0norm;
  splash_type splash_flag =
    splash_criteria(Kv, Tstar, alpha, Kcrit, del_film, N_SB[pid], ms, dm);
  if (splash_flag == splash_type::rebound) {
    amrex::Real rbound = 0.823 * std::exp(-1.9835E-3 * std::pow(Kv, 1.6));
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      p.pos(dir) -= (1. + rbound) * par_dot * normal[dir];
      p.rdata(SprayComps::pstateVel + dir) -=
        U0norm * (1. + rbound) * normal[dir];
    }
  } else if (ms > 0.) {
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      // Save location where droplet hit wall
      rf.loc[AMREX_SPACEDIM * pid + dir] = impact_loc[dir];
      rf.norm[AMREX_SPACEDIM * pid + dir] = normal[dir];
      rf.vel[AMREX_SPACEDIM * pid + dir] = vel_part[dir];
    }
    rf.ref_dia[pid] = p.rdata(SprayComps::pstateDia);
    rf.phi1[pid] = Kv;
    rf.phi2[pid] = pmass * ms;
    rf.phi3[pid] = del_film;
    rf.T0[pid] = T_part;
    rf.num_dens[pid] = num_dens;
#if SPRAY_FUEL_NUM > 1
    for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
      rf.Y0[SPRAY_FUEL_NUM * pid + spf] = p.rdata(SprayComps::pstateY + spf);
    }
#endif
    // Droplet reflects in the case of thermal breakup
    if (splash_flag == splash_type::thermal_breakup) {
      amrex::Real rem_mass = pmass * (1. - ms);
      amrex::Real new_dia = std::cbrt(6. * rem_mass / (M_PI * rho_part));
      amrex::RealVect tanBeta, tanPsi;
      find_tangents(vel_part, tanPsi, normal, tanBeta);
      amrex::Real Un_splash =
        0.0028 * Kv * std::exp(-0.0062 * Kv) * std::abs(U0mag);
      // This is the average of the splashed droplet velocities, 3 sqrt(3) / (2
      // pi) is the integral of cos(theta/3) from -pi to pi
      amrex::Real Ut_splash = 1.5 * std::sqrt(3.) * U0tan / M_PI;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        p.pos(dir) = impact_loc[dir];
        p.rdata(SprayComps::pstateVel + dir) =
          Ut_splash * tanBeta[dir] + Un_splash * normal[dir];
      }
      p.rdata(SprayComps::pstateDia) = new_dia;
      p.rdata(SprayComps::pstateBM1) = 0.;
      p.rdata(SprayComps::pstateBM2) = 0.;
    }
  }
  if (dm > 0.) {
    // Convert parcel to wall film
    // Assume all remaining droplets in parcel coalesce to form the wall film
    amrex::Real film_vol = num_dens * pmass * dm / rho_part;
    // Combined droplet diameter
    amrex::Real comb_dia = std::cbrt(film_vol * 6. / M_PI);
    amrex::Real comb_We = We * comb_dia / dia_part;
    amrex::Real comb_Re = Re_L * comb_dia / dia_part;
    // Contact angle
    amrex::Real theta_c = fdat.theta_c;
    // Find the maximum diameter of the wall film using a modified version of
    // the formula from Pasandideh-Fard et al. 1996 See documentation for more
    // info
    amrex::Real film_dia =
      comb_dia * std::sqrt(
                   (12. + comb_We) / (6. * (1. - std::cos(theta_c)) /
                                        std::pow(std::sin(theta_c), 2) +
                                      4. * comb_We / std::sqrt(comb_Re)));
    p.rdata(SprayComps::pstateDia) = film_dia;
    // Assume film is a cylinder
    amrex::Real film_hght = film_vol / (0.25 * M_PI * film_dia * film_dia);
    p.rdata(SprayComps::pstateFilmHght) = film_hght;
    p.rdata(SprayComps::pstateN0) = 1.;
    p.rdata(SprayComps::pstateNumDens) = 1.;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      p.pos(dir) = impact_loc[dir];
      p.rdata(SprayComps::pstateVel + dir) = 0.;
    }
  }
}

#endif
