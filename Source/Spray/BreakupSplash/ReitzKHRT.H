
#ifndef REITZKHRT_H
#define REITZKHRT_H

// This is the implementation of the wave model detailed by Reitz (1987)

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
updateBreakupKHRT(
  const int pid,
  SprayParticleContainer::ParticleType& p,
  const amrex::Real& Reyn_d,
  const amrex::Real& dt,
  const amrex::Real* cBoilT,
  const amrex::Real& avg_inject_mass,
  const amrex::Real B0,
  const amrex::Real B1,
  const amrex::Real C3,
  GasPhaseVals& gpv,
  const SprayData& fdat,
  splash_breakup* N_SB,
  const SBPtrs& rf,
  const bool make_new_drops)
{
  // Threshold to make new droplets normally
  const amrex::Real mthresh_jet = 0.03;
  // Threshold for when shedding mass is too high
  const amrex::Real mthresh_drop = 0.9;
  const amrex::Real We_crit = 6.;
  // RT model constants
  const amrex::Real A1 = 0.188;
  // Retrieve particle data
  amrex::RealVect vel_part(AMREX_D_DECL(
    p.rdata(SprayComps::pstateVel), p.rdata(SprayComps::pstateVel + 1),
    p.rdata(SprayComps::pstateVel + 2)));
  amrex::Real T_part = p.rdata(SprayComps::pstateT);
  amrex::Real dia_part = p.rdata(SprayComps::pstateDia);
  amrex::Real num_dens = p.rdata(SprayComps::pstateNumDens);
  amrex::Real Y_part[SPRAY_FUEL_NUM] = {0.0};
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    Y_part[spf] = p.rdata(SprayComps::pstateY + spf);
  }
  const amrex::Real rho_part = fdat.liqprops.rho_mix(Y_part, T_part, cBoilT);
  const amrex::Real mu_part = fdat.liqprops.mu_mix(Y_part, T_part, cBoilT);
  // Minimum droplet radius for child droplets
  // Should not create child droplets that are near to or less than the minimum
  // allowable mass to avoid unphysical evaporation of mass
  amrex::Real min_rad =
    4. * std::cbrt(SprayUnits::min_mass * 3. / (4. * M_PI * rho_part));
  amrex::Real shed_mass = p.rdata(SprayComps::pstateBM1);
  amrex::Real N0 = p.rdata(SprayComps::pstateN0);
  // To prolong RT breakup, rt_time is set to -1 on injection. RT breakup can
  // then only occur after KH breakup happens at least once
  amrex::Real rt_time = p.rdata(SprayComps::pstateBM2);
  const amrex::Real sigma_part =
    fdat.liqprops.sigma_mix(Y_part, T_part, cBoilT);
  amrex::Real rad_part = 0.5 * dia_part;
  if (rad_part < min_rad) {
    return;
  }
  amrex::Real rad_part3 = std::pow(rad_part, 3);
  amrex::RealVect diff_vel = gpv.vel_fluid - vel_part;
  amrex::Real vel_mag = diff_vel.vectorLength();
  amrex::Real vel_sq = vel_mag * vel_mag;
  amrex::Real Reyn = Reyn_d;
  amrex::Real C_D = 0.;
  if (Reyn > 1000.) {
    C_D = 0.424;
  } else if (Reyn > 1.) {
    C_D = 24. / Reyn * (1. + std::cbrt(Reyn * Reyn) / 6.);
  } else if (Reyn > 0.) {
    C_D = 24. / Reyn;
  }
  amrex::Real We_g = gpv.rho_fluid * vel_sq * rad_part / sigma_part;
  amrex::Real Z = mu_part / std::sqrt(rho_part * rad_part * sigma_part);
  amrex::Real T = Z * std::sqrt(We_g);
  bool breakupRT = false;
  // Check of RT breakup occurs
  if (rt_time >= 0.) {
    // Add gravity here when applicable
    amrex::Real a =
      3. / 8. * C_D * gpv.rho_fluid * vel_sq / (rho_part * rad_part);
    amrex::Real momdiff = a * (rho_part - gpv.rho_fluid);
    amrex::Real omega_RT = std::sqrt(
      2. / (3. * std::sqrt(3. * sigma_part)) * momdiff * std::sqrt(momdiff) /
      (rho_part + gpv.rho_fluid));
    amrex::Real KRT = std::sqrt(momdiff / (3. * sigma_part));
    amrex::Real lam_RT = C3 * M_PI / KRT;
    if (rad_part < lam_RT) {
      rt_time = 0.;
    } else {
      rt_time += dt;
    }
    amrex::Real tau_RT = 1. / omega_RT;
    // RT breakup occurs
    if (rt_time > tau_RT && lam_RT < rad_part) {
      rt_time = 0.;
      num_dens *= std::pow(rad_part / lam_RT, 3);
      rad_part = lam_RT;
      breakupRT = true;
    }
  }
  if (!breakupRT) {
    // Check if KH breakup occurs
    amrex::Real lam_KH = rad_part * 9.02 * (1. + 0.45 * std::sqrt(Z)) *
                         (1. + 0.4 * std::pow(T, 0.7)) /
                         std::pow(1. + 0.865 * std::pow(We_g, 1.67), 0.6);
    amrex::Real omega_KH = (0.34 + 0.385 * We_g * std::sqrt(We_g)) /
                           ((1. + Z) * (1. + 1.4 * std::pow(T, 0.6))) *
                           std::sqrt(sigma_part / (rho_part * rad_part3));
    amrex::Real tau_KH = 3.726 * B1 * rad_part / (lam_KH * omega_KH);
    amrex::Real rs = B0 * lam_KH;
    // KH breakup occurs
    if (rs < rad_part && We_g > We_crit) {
      amrex::Real rp = (dt / tau_KH * rs + rad_part) / (1. + dt / tau_KH);
      amrex::Real rp3 = rp * rp * rp;
      amrex::Real rs3 = rs * rs * rs;
      amrex::Real np = num_dens * rad_part3 / rp3;
      rt_time = 0.;
      shed_mass += N0 * (rad_part3 - rp3);
      amrex::Real ns = shed_mass / rs3;
      bool create_child_drops = false;
      if (make_new_drops) {
        // Percent of shed mass relative to total mass
        amrex::Real rel_shed_drop = shed_mass / (num_dens * rad_part3);
        // Percent of shed mass relative to injected droplet mass
        amrex::Real rel_shed_jet =
          8. * shed_mass /
          amrex::max<amrex::Real>(
            avg_inject_mass, std::numeric_limits<amrex::Real>::epsilon());
        // Too much mass has been shed, must split droplet up and delete droplet
        if (rel_shed_drop > mthresh_drop) {
          create_child_drops = true;
          shed_mass = num_dens * rad_part3;
          rs = amrex::max<amrex::Real>(rs, min_rad);
          rs3 = rs * rs * rs;
          ns = shed_mass / rs3;
          p.id() = -1;
        } else if (
          rel_shed_jet > mthresh_jet && rs > min_rad && ns > num_dens) {
          create_child_drops = true;
          np = (np * rp3 - ns * rs3) / rp3;
          N0 = np;
          shed_mass = 0.;
        }
      }
      if (create_child_drops) {
        amrex::Real Utan = A1 * omega_KH * lam_KH;
        // Unit vector along the droplet velocity, as for TAB
        const amrex::Real part_vel_mag = vel_part.vectorLength();
        N_SB[pid] = splash_breakup::breakup_KH;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
          rf.loc[AMREX_SPACEDIM * pid + dir] = p.pos(dir);
          rf.vel[AMREX_SPACEDIM * pid + dir] = vel_part[dir];
          rf.norm[AMREX_SPACEDIM * pid + dir] = vel_part[dir] / part_vel_mag;
        }
        rf.num_dens[pid] = ns;
        rf.ref_dia[pid] = 2. * rs;
        rf.T0[pid] = T_part;
        rf.phi1[pid] = Utan;
        rf.phi2[pid] = 0.; // Unused
        rf.phi3[pid] = 0.; // Unused
#if SPRAY_FUEL_NUM > 1
        for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
          rf.Y0[SPRAY_FUEL_NUM * pid + spf] =
            p.rdata(SprayComps::pstateY + spf);
        }
#endif
      } // if (create_child_drops)...
      rad_part = rp;
      num_dens = np;
    } // if (rs < rad_part && We_g > We_crit)...
  } // if (!breakupRT)...
  p.rdata(SprayComps::pstateDia) = 2. * rad_part;
  p.rdata(SprayComps::pstateNumDens) = num_dens;
  p.rdata(SprayComps::pstateN0) = N0;
  p.rdata(SprayComps::pstateBM1) = shed_mass;
  p.rdata(SprayComps::pstateBM2) = rt_time;
}

#endif
