
#ifndef WALLFILM_H
#define WALLFILM_H

#include "Drag.H"

// Determine the wall film thickness within the cell by summing film volume
// divided by the face area
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fillFilmFab(
  amrex::Array4<amrex::Real> const& wf_arr,
  SprayParticleContainer::ParticleType& p,
  const amrex::Real& face_area,
  const amrex::RealVect& plo,
  const amrex::RealVect& dx)
{
  amrex::RealVect lxc = (p.pos() - plo) / dx;
  amrex::IntVect ijkc = lxc.floor(); // Cell with particle
  amrex::Real film_hght = p.rdata(SprayComps::pstateFilmHght);
  amrex::Real film_dia = p.rdata(SprayComps::pstateDia);
  amrex::Real film_vol =
    M_PI * film_hght * (0.75 * film_dia * film_dia + film_hght * film_hght);
  amrex::Gpu::Atomic::Add(&wf_arr(ijkc, 0), film_vol / face_area);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
calculateFilmSource(
  const amrex::Real flow_dt,
  GasPhaseVals& gpv,
  SprayData fdat,
  SprayParticleContainer::ParticleType& p,
  amrex::Real* cBoilT,
  pele::physics::transport::TransParm<
    pele::physics::EosType,
    pele::physics::TransportType> const* trans_parm)
{
  auto eos = pele::physics::PhysicsType::eos();
  const amrex::Real dt = flow_dt * fdat.dtmod;
  const amrex::Real rule = 1. / 2.;
  amrex::Real C_eps = 1.E-15;
  amrex::Real min_height = 1.E-5 * SprayUnits::len_conv;
  bool get_xi = false;
  bool get_Ddiag = true;
  bool get_lambda = true;
  bool get_mu = false;
  bool get_chi = false;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> Y_skin;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> h_film;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> cp_i;
  amrex::GpuArray<amrex::Real, NUM_SPECIES> rhoD_ri;
  amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> L_n{};
  amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> mn_dot = {};
  amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> Y_film{};    // Liquid mass fracs
  amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> oldY_film{}; // Prev. mass fracs
  amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> X_vn{};
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    Y_film[spf] = p.rdata(SprayComps::pstateY + spf);
    oldY_film[spf] = Y_film[spf];
  }
  amrex::Real T_film = p.rdata(SprayComps::pstateT);
  const amrex::Real rho_film = fdat.liqprops.rho_mix(Y_film, T_film, cBoilT);
  const amrex::Real Tcrit = fdat.liqprops.critT_mix(Y_film);
  const amrex::Real cp_film = fdat.liqprops.cp_mix(Y_film, T_film, cBoilT);
  const amrex::Real lambda_film =
    fdat.liqprops.lambda_mix(Y_film, T_film, cBoilT);
  const amrex::Real mw_film = fdat.liqprops.mw_mix(Y_film);
  T_film = amrex::min<amrex::Real>(0.999 * Tcrit, T_film);
  amrex::Real film_height = p.rdata(SprayComps::pstateFilmHght);
  amrex::Real film_dia = p.rdata(SprayComps::pstateDia);
  amrex::Real rad2 = std::pow(0.5 * film_dia, 2);
  // Surface area assuming film is a cylinder
  amrex::Real film_area = M_PI * rad2;
  amrex::Real film_mass = rho_film * film_height * film_area;
  amrex::Real start_mass = film_mass;
  // Model the fuel vapor using the one-third rule
  amrex::Real delT = amrex::max<amrex::Real>(gpv.T_fluid - T_film, 0.);
  amrex::Real T_skin = T_film + rule * delT;

  // Calculate the C_p at the skin temperature for each species
  eos.T2Cpi(T_skin, cp_i.data());
  eos.T2Hi(T_film, h_film.data());
  for (int n = 0; n < NUM_SPECIES; ++n) {
    Y_skin[n] = 0.;
    h_film[n] *= SprayUnits::eng_conv;
    cp_i[n] *= SprayUnits::eng_conv;
  }
  amrex::Real cp_skin = 0.; // Average C_p in modeled skin phase
  amrex::Real mw_skin = 0.; // Average molar mass of skin phase
  amrex::Real B_M = 0.;     // Mass Spalding number
  amrex::Real sumXVap = 0.; // Sum of X_v
  calcVaporState<pele::physics::PhysicsType::eos_type>()(
    fdat, gpv, rule, T_film, C_eps, mw_film, Y_film.data(), T_skin, cBoilT,
    Y_skin.data(), X_vn.data(), L_n.data(), B_M, sumXVap, cp_skin, mw_skin);
  amrex::Real lambda_skin = 0.;
  amrex::Real mu_skin = 0.;
  amrex::Real xi_skin = 0.;
  amrex::Real rho_skin =
    mw_skin * gpv.p_fluid /
    (pele::physics::Constants::RU * T_skin * SprayUnits::ru_conv);
  amrex::Real rho_cgs = rho_skin / SprayUnits::rho_conv;
  auto trans = pele::physics::PhysicsType::transport();
  trans.transport(
    get_xi, get_mu, get_lambda, get_Ddiag, get_chi, T_skin, rho_cgs,
    Y_skin.data(), rhoD_ri.data(), nullptr, mu_skin, xi_skin, lambda_skin,
    trans_parm);
  lambda_skin *= SprayUnits::lambda_conv;
  // If gas phase is not saturated
  if (sumXVap > 0.) {
    // Convert mass diffusion coefficient from mixture average to binary for
    // fuel only, not concerned with other species
    for (int pc = 0; pc < fdat.N_pc; ++pc) {
      const int pcspec = fdat.pc_indx[pc];
      rhoD_ri[pcspec] *= mw_skin / gpv.mw[pcspec] * SprayUnits::rhod_conv;
      amrex::Real X_vi =
        fdat.binary_csr_rowdotprod(pcspec, fdat.L_row, fdat.L_col, X_vn.data());
      amrex::Real X_norm = X_vi / sumXVap;
      rhoD_ri[pcspec] *= X_norm;
    }
    // Estimate convective heat transfer using O'Rourke and Amsden 1996 assuming
    // y+ < y+c
    // TODO: This is very rudimentary and should be improved to account for gas
    // phase velocity
    amrex::Real h_heat = lambda_skin / film_height;
    amrex::Real q_conv = film_area * h_heat * (gpv.T_fluid - T_film);
    amrex::Real q_cond =
      -film_area * lambda_film * 2. * (T_film - fdat.wall_T) / film_height;
    // Use Chilton-Colburn analogy to find the convective mass transfer
    // coefficient; like droplet evaporation, this uses mixture averaged
    // values for lambda, cp, and B_M but uses normalized species mass
    // diffusivity
    amrex::Real h_mass = film_area * h_heat * std::cbrt(lambda_skin / cp_skin) *
                         std::log1p(B_M) / lambda_skin;
    amrex::Real sumL = 0.;
    amrex::Real m_dot = 0.;
    for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
      // Species index
      const int fdspec = fdat.dep_indx[spf];
      if (X_vn[spf] > 0.) {
        amrex::Real X_vi =
          fdat.binary_csr_rowdotprod(fdspec, fdat.L_row, fdat.L_col, X_vn);
        amrex::Real Xnorm = 1.;
        if (X_vi > X_vn[spf]) {
          // Multiple species deposit to fdspec, scale rhoD_{r,n} accordingly
          Xnorm = X_vn[spf] / X_vi;
        }
        mn_dot[spf] = -amrex::max<amrex::Real>(
          h_mass * std::pow(Xnorm * rhoD_ri[fdspec], 2. / 3.), 0.);
        m_dot += mn_dot[spf];
        sumL += mn_dot[spf] * L_n[spf];
      }
    }
    amrex::Real part_temp_src =
      (q_conv + q_cond + sumL) / (film_mass * cp_film);
    gpv.fluid_eng_src += q_conv;
    T_film += dt * part_temp_src;
    amrex::Real new_mass = film_mass + dt * m_dot;
    // Assumes mass is only lost in the wall normal direction and the diameter
    // remains constant; this is not physically correct
    film_height = new_mass / (rho_film * film_area);
    if (film_height > min_height) {
      if (SPRAY_FUEL_NUM > 1) {
        amrex::Real sumY = 0.;
        for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
          Y_film[spf] = amrex::min<amrex::Real>(
            1., amrex::max<amrex::Real>(
                  (Y_film[spf] * film_mass + mn_dot[spf] * dt) / new_mass, 0.));
          if (Y_film[spf] < 1.E-12) {
            Y_film[spf] = 0.;
          }
          sumY += Y_film[spf];
        }
        if (sumY > 0.) {
          for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
            Y_film[spf] /= sumY;
          }
        } else {
          for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
            Y_film[spf] = 0.;
          }
        }
      }
    } else {
      film_height = 0.;
      new_mass = 0.;
      p.id() = -1;
    }
    film_mass = new_mass;
  }

  p.rdata(SprayComps::pstateT) = T_film;
  amrex::Real mdot_total = (film_mass - start_mass) / dt;
  gpv.fluid_mass_src = mdot_total;
  for (int pc = 0; pc < fdat.N_pc; ++pc) {
    const int pcspec = fdat.pc_indx[pc];
    amrex::Real oldY_film_pc =
      fdat.binary_csr_rowdotprod(pcspec, fdat.L_row, fdat.L_col, oldY_film);
    amrex::Real newY =
      fdat.binary_csr_rowdotprod(pcspec, fdat.L_row, fdat.L_col, Y_film);
    amrex::Real midot = (newY * film_mass - oldY_film_pc * start_mass) / dt;
    gpv.fluid_Y_dot[pcspec] = midot;
    gpv.fluid_eng_src += midot * h_film[pcspec];
  }
  p.rdata(SprayComps::pstateDia) = film_dia;
  p.rdata(SprayComps::pstateFilmHght) = film_height;
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    p.rdata(SprayComps::pstateY + spf) = Y_film[spf];
  }
}
#endif
