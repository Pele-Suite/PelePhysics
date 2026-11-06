#ifndef SPRAYINTERPOLATION_H
#define SPRAYINTERPOLATION_H

#include "AMReX_REAL.H"
#ifdef AMREX_USE_EB
#include "AMReX_EBMultiFabUtil.H"
#endif

using namespace amrex::literals;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
InterpolateFaceVelocity(
  const amrex::RealVect& len,
  const amrex::IntVect& dom_lo,
  const amrex::IntVect& dom_hi,
  amrex::GpuArray<amrex::Array4<amrex::Real const>, AMREX_SPACEDIM> const& umac,
  amrex::RealVect& vel_fluid)
{
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    amrex::RealVect face_len(AMREX_D_DECL(
      len[0] + (dir == 0) * 0.5, len[1] + (dir == 1) * 0.5,
      len[2] + (dir == 2) * 0.5));
    // Compute distances for trilinear interpolation
    amrex::IntVect indx(AMREX_D_DECL(
      static_cast<int>(amrex::Math::floor(face_len[0])),
      static_cast<int>(amrex::Math::floor(face_len[1])),
      static_cast<int>(amrex::Math::floor(face_len[2]))));
    amrex::RealVect delL(AMREX_D_DECL(
      face_len[0] - static_cast<amrex::Real>(indx[0]),
      face_len[1] - static_cast<amrex::Real>(indx[1]),
      face_len[2] - static_cast<amrex::Real>(indx[2])));
    const int iin[2] = {
      amrex::max<int>(indx[0] - 1, dom_lo[0]),
      amrex::min<int>(indx[0], dom_hi[0])};
    const amrex::Real ssx[2] = {1.0_rt - delL[0], delL[0]};
    const int jjn[2] = {
      amrex::max<int>(indx[1] - 1, dom_lo[1]),
      amrex::min<int>(indx[1], dom_hi[1])};
    const amrex::Real ssy[2] = {1.0_rt - delL[1], delL[1]};
#if AMREX_SPACEDIM == 3
    const int kkn[2] = {
      amrex::max<int>(indx[2] - 1, dom_lo[2]),
      amrex::min<int>(indx[2], dom_hi[2])};
    const amrex::Real ssz[2] = {1.0_rt - delL[2], delL[2]};
#endif
    int ke = (AMREX_SPACEDIM == 3) ? 2 : 1;
    for (int kk = 0; kk < ke; ++kk) {
      for (int jj = 0; jj < 2; ++jj) {
        const int yindx = jjn[jj];
        for (int ii = 0; ii < 2; ++ii) {
          const int xindx = iin[ii];
          const amrex::IntVect cijk = {AMREX_D_DECL(xindx, yindx, kkn[kk])};
          vel_fluid[dir] +=
            (umac[dir])(cijk, 0) * AMREX_D_TERM(ssx[ii], *ssy[jj], *ssz[kk]);
        }
      }
    }
  }
}

/* Check if outside of bounds of Cartesian grid
 Return IntVect that where the flags are per direction
 0 - Not outside domain or at periodic boundary
 -1 - Outside lower reflective boundary
 1 - Outside upper reflective boundary
 -2 - Adjacent to lower reflective boundary
 2 - Adjacent to upper reflective boundary
 And true is returned if outside non-reflective domain
*/
AMREX_GPU_DEVICE AMREX_INLINE bool
check_bounds(
  const amrex::RealVect pos,
  const amrex::RealVect plo,
  const amrex::RealVect phi,
  const amrex::RealVect dx,
  const amrex::IntVect bndry_lo,
  const amrex::IntVect bndry_hi,
  amrex::IntVect& bflags)
{
  for (int hilo = 0; hilo < 2; ++hilo) {
    int fact = (hilo == 0) ? 1 : -1;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      amrex::Real dom_loc = (hilo == 0) ? plo[dir] : phi[dir];
      int bflag = (hilo == 0) ? bndry_lo[dir] : bndry_hi[dir];
      if (bflag != 0) { // Not periodic
        amrex::Real diff =
          static_cast<amrex::Real>(fact) * (pos[dir] - dom_loc);
        if (diff < 0.) {
          // Outside domain
          if (bflag == 1) { // Outside reflective BC
            bflags[dir] = -fact;
          } else { // Outside non-reflective BC
            return true;
          }
          // Adjacent to non-periodic boundary
        } else if (diff < 0.5 * dx[dir]) {
          bflags[dir] = -2 * fact;
        }
      }
    }
  }
  return false;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
InterpolateGasPhase(
  GasPhaseVals& gpv,
  const amrex::Box& state_box,
  amrex::Array4<const amrex::Real> const& rhoarr,
  amrex::Array4<const amrex::Real> const& rhoYarr,
  amrex::Array4<const amrex::Real> const& Tarr,
  amrex::Array4<const amrex::Real> const& momarr,
  amrex::Array4<const amrex::Real> const& engarr,
  const amrex::IntVect* indx_array,
  const amrex::Real* weights,
  const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
    eosparm = nullptr,
  const SprayData* sprayData = nullptr)
{
  auto eos = pele::physics::PhysicsType::eos(eosparm);
#ifdef PELELM_USE_SPRAY
  amrex::ignore_unused(engarr);
#endif

#ifndef USE_MANIFOLD_EOS
  amrex::ignore_unused(sprayData);
#endif

  amrex::GpuArray<amrex::Real, NUM_SPECIES> mass_frac;
  for (int aindx = 0.; aindx < AMREX_D_PICK(2, 4, 8); ++aindx) {
    amrex::Real cw = weights[aindx];
    if (cw > 0.) {
      amrex::IntVect cur_indx = indx_array[aindx];
      if (!state_box.contains(cur_indx)) {
        amrex::Abort(
          "SprayParticleContainer::updateParticles() -- state box too small");
      }
      amrex::Real cur_rho = rhoarr(cur_indx);
      gpv.rho_fluid += cw * cur_rho;
      amrex::Real inv_rho = 1. / cur_rho;
      for (int n = 0; n < NUM_SPECIES; ++n) {
        amrex::Real cur_mf = rhoYarr(cur_indx, n) * inv_rho;
        gpv.Y_fluid[n] += cw * cur_mf;
        mass_frac[n] = cur_mf;
      }
#ifdef PELELM_USE_SPRAY
      inv_rho = 1.;
#endif
#ifndef PELELM_USE_SPRAY
      amrex::Real ke = 0.;
#endif
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        amrex::Real vel = momarr(cur_indx, dir) * inv_rho;
        gpv.vel_fluid[dir] += cw * vel;
#ifndef PELELM_USE_SPRAY
        ke += vel * vel / 2.;
#endif
      }
      amrex::Real T_i = Tarr(cur_indx);
#ifndef PELELM_USE_SPRAY
      amrex::Real intEng = engarr(cur_indx) * inv_rho - ke;
      eos.EY2T(intEng, mass_frac.data(), T_i);
#endif
      gpv.T_fluid += cw * T_i;
    }
  }

  // In the case of  manifold model simulations, the objective is to get the
  // mass fractions of all the phase-change gas species, at the spray location,
  // read from the manifold table. We already have the interpolated value of
  // manifold variables at the spray particle location stored in 'mass_frac'.
#ifdef USE_MANIFOLD_EOS
  for (int n = 0; n < SPRAY_FUEL_NUM; n++) {
    amrex::Real tempval;
    eos.Y2ChemSpecies(mass_frac.data(), sprayData->dep_indx[n], tempval);
    gpv.Yspec_fluid[n] = tempval;
  }
#endif

  // Fill in some other data with EOS calls
  eos.Y2WBAR(gpv.Y_fluid.data(), gpv.mw_mix);
  gpv.mw_mix *= SprayUnits::mass_conv;

  eos.RTY2P(
    gpv.rho_fluid / SprayUnits::rho_conv, gpv.T_fluid, gpv.Y_fluid.data(),
    gpv.p_fluid);
  gpv.p_fluid *= SprayUnits::pres_conv;
}

// Slightly modified from MFIX code

/****************************************************************
 Functions for interpolation on non-EB mesh
 ***************************************************************/

AMREX_GPU_DEVICE AMREX_INLINE void
trilinear_interp(
  const amrex::IntVect& ijk,
  const amrex::RealVect& lx,
  amrex::IntVect* indx_array,
  amrex::Real* weights,
  const amrex::IntVect& bflags)
{
  // Note: if near a reflective boundary, ijk has been shifted in check_bounds
  AMREX_D_TERM(int i = ijk[0];, int j = ijk[1];, int k = ijk[2];)

  const amrex::RealVect sx_hi = lx - amrex::RealVect(ijk);
  const amrex::RealVect sx_lo = 1. - sx_hi;
  amrex::GpuArray<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, 2> ssv;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    // Check if adjacent to upper boundary
    // or lower boundary and make it zeroth-order interpolation
    if (bflags[dir] > 1) {
      ssv[0][dir] = 1.;
      ssv[1][dir] = 0.;
    } else if (bflags[dir] < -1) {
      ssv[0][dir] = 0.;
      ssv[1][dir] = 1.;
    } else {
      ssv[0][dir] = sx_lo[dir];
      ssv[1][dir] = sx_hi[dir];
    }
  }
  int cc = 0;
  int ks = (AMREX_SPACEDIM == 3) ? -1 : 0;
  for (int kk = ks; kk < 1; kk++) {
    for (int jj = -1; jj < 1; jj++) {
      for (int ii = -1; ii < 1; ii++) {
        AMREX_D_TERM(
          indx_array[cc][0] = i + ii;, indx_array[cc][1] = j + jj;
          , indx_array[cc][2] = k + kk;)
        weights[cc] =
          AMREX_D_TERM(ssv[ii + 1][0], *ssv[jj + 1][1], *ssv[kk + 1][2]);
        cc++;
      }
    }
  }
}

#ifdef AMREX_USE_EB

/****************************************************************
 Functions for the Newtons solver
 ***************************************************************/

AMREX_GPU_DEVICE AMREX_INLINE amrex::Real
f(const int dir,
  const amrex::RealVect& pos,
  const amrex::GpuArray<
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>,
    AMREX_D_PICK(2, 4, 8)>& nodes,
  const amrex::Real& xi,
  const amrex::Real& eta,
  const amrex::Real& zeta)
{
  amrex::Real a0 = (nodes[0][dir] - pos[dir]);
  amrex::Real a1 = (nodes[1][dir] - nodes[0][dir]);
  amrex::Real a2 = (nodes[2][dir] - nodes[0][dir]);
  amrex::Real a4 =
    (nodes[0][dir] - nodes[1][dir] + nodes[3][dir] - nodes[2][dir]);
#if AMREX_SPACEDIM == 3
  amrex::Real a3 = (nodes[4][dir] - nodes[0][dir]);
  amrex::Real a5 =
    (nodes[0][dir] - nodes[1][dir] - nodes[4][dir] + nodes[5][dir]);
  amrex::Real a6 =
    (nodes[0][dir] - nodes[2][dir] - nodes[4][dir] + nodes[6][dir]);
  amrex::Real a7 =
    (nodes[1][dir] - nodes[3][dir] + nodes[2][dir] + nodes[4][dir] -
     nodes[5][dir] + nodes[7][dir] - nodes[6][dir] - nodes[0][dir]);

  return a0 + a1 * xi + a2 * eta + a3 * zeta + a4 * xi * eta + a5 * xi * zeta +
         a6 * eta * zeta + a7 * xi * eta * zeta;
#else
  amrex::ignore_unused(zeta);
  return a0 + a1 * xi + a2 * eta + a4 * xi * eta;
#endif
}

AMREX_GPU_DEVICE AMREX_INLINE amrex::Real
dfdxi(
  const int dir,
  const amrex::GpuArray<
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>,
    AMREX_D_PICK(2, 4, 8)>& nodes,
  const amrex::Real& /*xi*/,
  const amrex::Real& eta,
  const amrex::Real& zeta)
{
  amrex::Real a1 = (nodes[1][dir] - nodes[0][dir]);
  amrex::Real a4 =
    (nodes[0][dir] - nodes[1][dir] + nodes[3][dir] - nodes[2][dir]);
#if AMREX_SPACEDIM == 3
  amrex::Real a5 =
    (nodes[0][dir] - nodes[1][dir] - nodes[4][dir] + nodes[5][dir]);
  amrex::Real a7 =
    (nodes[1][dir] - nodes[3][dir] + nodes[2][dir] + nodes[4][dir] -
     nodes[5][dir] + nodes[7][dir] - nodes[6][dir] - nodes[0][dir]);

  return a1 + a4 * eta + a5 * zeta + a7 * eta * zeta;
#else
  amrex::ignore_unused(zeta);
  return a1 + a4 * eta;
#endif
}

AMREX_GPU_DEVICE AMREX_INLINE amrex::Real
dfdeta(
  const int dir,
  const amrex::GpuArray<
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>,
    AMREX_D_PICK(2, 4, 8)>& nodes,
  const amrex::Real& xi,
  const amrex::Real& /*eta*/,
  const amrex::Real& zeta)
{

  amrex::Real a2 = (nodes[2][dir] - nodes[0][dir]);
  amrex::Real a4 =
    (nodes[0][dir] - nodes[1][dir] + nodes[3][dir] - nodes[2][dir]);
#if AMREX_SPACEDIM == 3
  amrex::Real a6 =
    (nodes[0][dir] - nodes[2][dir] - nodes[4][dir] + nodes[6][dir]);
  amrex::Real a7 =
    (nodes[1][dir] - nodes[3][dir] + nodes[2][dir] + nodes[4][dir] -
     nodes[5][dir] + nodes[7][dir] - nodes[6][dir] - nodes[0][dir]);

  return a2 + a4 * xi + a6 * zeta + a7 * xi * zeta;
#else
  amrex::ignore_unused(zeta);
  return a2 + a4 * xi;
#endif
}

#if AMREX_SPACEDIM == 3
AMREX_GPU_DEVICE AMREX_INLINE amrex::Real
dfdzeta(
  const int dir,
  const amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 8>& nodes,
  const amrex::Real& xi,
  const amrex::Real& eta,
  const amrex::Real& /*zeta*/)
{

  amrex::Real a3 = (nodes[4][dir] - nodes[0][dir]);
  amrex::Real a5 =
    (nodes[0][dir] - nodes[1][dir] - nodes[4][dir] + nodes[5][dir]);
  amrex::Real a6 =
    (nodes[0][dir] - nodes[2][dir] - nodes[4][dir] + nodes[6][dir]);
  amrex::Real a7 =
    (nodes[1][dir] - nodes[3][dir] + nodes[2][dir] + nodes[4][dir] -
     nodes[5][dir] + nodes[7][dir] - nodes[6][dir] - nodes[0][dir]);

  return a3 + a5 * xi + a6 * eta + a7 * xi * eta;
}
#endif

AMREX_GPU_DEVICE AMREX_INLINE void
get_interp_mapping(
  const amrex::RealVect& pos,
  const amrex::GpuArray<
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>,
    AMREX_D_PICK(2, 4, 8)>& nodes,
  amrex::Real& xi,
  amrex::Real& eta,
  amrex::Real& zeta)
{
  int lc = 0;
  amrex::Real err = 1.0;

  while (err > 1.0e-3 && lc < 10) {

    amrex::Real f0 = f(0, pos, nodes, xi, eta, zeta);
    amrex::Real f1 = f(1, pos, nodes, xi, eta, zeta);

    amrex::Real df0dxi = dfdxi(0, nodes, xi, eta, zeta);
    amrex::Real df0deta = dfdeta(0, nodes, xi, eta, zeta);

    amrex::Real df1dxi = dfdxi(1, nodes, xi, eta, zeta);
    amrex::Real df1deta = dfdeta(1, nodes, xi, eta, zeta);

#if AMREX_SPACEDIM == 3
    amrex::Real f2 = f(2, pos, nodes, xi, eta, zeta);

    amrex::Real df0dzeta = dfdzeta(0, nodes, xi, eta, zeta);

    amrex::Real df1dzeta = dfdzeta(1, nodes, xi, eta, zeta);

    amrex::Real df2dxi = dfdxi(2, nodes, xi, eta, zeta);
    amrex::Real df2deta = dfdeta(2, nodes, xi, eta, zeta);
    amrex::Real df2dzeta = dfdzeta(2, nodes, xi, eta, zeta);

    amrex::Real detJ = df0dxi * (df1deta * df2dzeta - df1dzeta * df2deta) -
                       df0deta * (df1dxi * df2dzeta - df1dzeta * df2dxi) +
                       df0dzeta * (df1dxi * df2deta - df1deta * df2dxi);

    amrex::Real detJ_xi = f0 * (df1deta * df2dzeta - df1dzeta * df2deta) -
                          df0deta * (f1 * df2dzeta - df1dzeta * f2) +
                          df0dzeta * (f1 * df2deta - df1deta * f2);

    amrex::Real detJ_eta = df0dxi * (f1 * df2dzeta - df1dzeta * f2) -
                           f0 * (df1dxi * df2dzeta - df1dzeta * df2dxi) +
                           df0dzeta * (df1dxi * f2 - f1 * df2dxi);

    amrex::Real detJ_zeta = df0dxi * (df1deta * f2 - f1 * df2deta) -
                            df0deta * (df1dxi * f2 - f1 * df2dxi) +
                            f0 * (df1dxi * df2deta - df1deta * df2dxi);
    amrex::Real new_xi = xi - detJ_xi / detJ;
    amrex::Real new_eta = eta - detJ_eta / detJ;
    amrex::Real new_zeta = zeta - detJ_zeta / detJ;
#else
    amrex::ignore_unused(zeta);
    amrex::Real detJ = df0dxi * df1deta - df0deta * df1dxi;

    amrex::Real detJ_xi = f0 * df1deta - f1 * df0deta;

    amrex::Real detJ_eta = df0dxi * f1 - f0 * df1dxi;

    amrex::Real new_xi = xi - detJ_xi / detJ;
    amrex::Real new_eta = eta - detJ_eta / detJ;
    amrex::Real new_zeta = 0.;
#endif

    lc += 1;
    err = amrex::max<amrex::Real>(
      amrex::Math::abs(xi - new_xi), amrex::Math::abs(eta - new_eta),
      amrex::Math::abs(zeta - new_zeta));

    xi = new_xi;
    eta = new_eta;
    zeta = new_zeta;
  }
}

AMREX_GPU_DEVICE AMREX_INLINE bool
eb_interp(
  SprayParticleContainer::ParticleType& p,
  amrex::IntVect& ijkc,
  const amrex::IntVect& ijk,
  const amrex::RealVect& dx,
  const amrex::RealVect& dxi,
  const amrex::RealVect& lx,
  const amrex::RealVect& plo,
  const amrex::IntVect& bflags,
  amrex::Array4<const amrex::EBCellFlag> const& flags,
  amrex::Array4<const amrex::Real> const& ccent,
  amrex::Array4<const amrex::Real> const& bcent,
  amrex::Array4<const amrex::Real> const& bnorm,
  amrex::Array4<const amrex::Real> const& vfrac,
  const amrex::Real min_eb_vfrac,
  amrex::IntVect* indx_array,
  amrex::Real* weights)
{
  bool do_fe_interp = false;
  int ip = 0, jp = 0, kp = 0;
  // Cell containing particle centroid
  AMREX_D_TERM(ip = ijkc[0];, jp = ijkc[1];, kp = ijkc[2];);
  // Get position normalized by dx
  amrex::RealVect normpos = (p.pos() - plo) * dxi;
  amrex::RealVect normposp = normpos - 0.5;
  // Upper cell relative to particle position, could be cell that contains
  // particle as well
  int ic = 0, jc = 0, kc = 0;
  AMREX_D_TERM(ic = ijk[0];, jc = ijk[1];, kc = ijk[2]);
  int ks = (AMREX_SPACEDIM == 3) ? kc - 1 : 0;
  int ke = (AMREX_SPACEDIM == 3) ? kc : 0;
  // All cells in the stencil are regular. Use
  // traditional trilinear interpolation
  for (int kk = ks; kk <= ke; kk++) {
    for (int jj = jc - 1; jj <= jc; jj++) {
      for (int ii = ic - 1; ii <= ic; ii++) {
        if (!flags(ii, jj, kk).isRegular()) {
          do_fe_interp = true;
        }
      }
    }
  }
  if (!do_fe_interp) {
    trilinear_interp(ijk, lx, indx_array, weights, bflags);
    return false;
  }
  const amrex::Real tolerance = std::numeric_limits<amrex::Real>::epsilon();

  amrex::Real par_dot_EB = 2.;
  amrex::Real cent_dot_EB = 1.;
  amrex::RealVect normal;

  // If the particle is inside a cut-cell, verify that it is on the correct
  // side of the EB before trying to interpolate.
  if (flags(ip, jp, kp).isSingleValued()) {
    normal = {AMREX_D_DECL(
      -bnorm(ip, jp, kp, 0), -bnorm(ip, jp, kp, 1), -bnorm(ip, jp, kp, 2))};

    // Projection of vector pointing from EB centroid to particle onto EB
    // normal
    par_dot_EB = AMREX_D_TERM(
      (normposp[0] - (ip + bcent(ip, jp, kp, 0))) * normal[0],
      +(normposp[1] - (jp + bcent(ip, jp, kp, 1))) * normal[1],
      +(normposp[2] - (kp + bcent(ip, jp, kp, 2))) * normal[2]);

    // Projection of vector pointing from EB centroid to cell centroid onto EB
    // normal
    cent_dot_EB = AMREX_D_TERM(
      (ccent(ip, jp, kp, 0) - bcent(ip, jp, kp, 0)) * normal[0],
      +(ccent(ip, jp, kp, 1) - bcent(ip, jp, kp, 1)) * normal[1],
      +(ccent(ip, jp, kp, 2) - bcent(ip, jp, kp, 2)) * normal[2]);
    // Temporary sanity check
    AMREX_ASSERT_WITH_MESSAGE(
      cent_dot_EB > tolerance,
      "cent_dot_EB < tolerance ... this makes no sense!");
  }

  // Count the number of non-connected cells in the stencil
  int covered = 0;
  for (int kk = ks; kk <= ke; kk++) {
    for (int jj = jc - 1; jj <= jc; jj++) {
      for (int ii = ic - 1; ii <= ic; ii++) {
        if (!flags(ip, jp, kp).isConnected(ii - ip, jj - jp, kk - kp)) {
          covered += 1;
        }
      }
    }
  }
  // Check if particle is in a covered cell
  bool in_covered = flags(ip, jp, kp).isCovered();
  // Check if particle is behind the EB cut
  bool behind_EB = (par_dot_EB < tolerance);
  // These checks mean the particle must be moved
  bool bad_part = in_covered || behind_EB;

  // Virtual particles can inadvertently be placed behind EB walls on lower
  // levels. For these situations, simply move the particle to the other side
  // of EB. Otherwise, something has gone wrong

  // Check if one or more adjacent cells are covered
  bool near_covered = (covered > 0);
  // Check particle is between the EB and the cell centroid
  bool near_EB = (par_dot_EB < cent_dot_EB);
  // Check if particle is in a cell with a small volume fraction
  bool in_small_cell = (vfrac(ip, jp, kp) < min_eb_vfrac);

  // These checks mean an inverse weighting method will be used
  bool use_invw = near_covered || near_EB || in_small_cell || bad_part;

  if (use_invw) {
    // The index for weights and indx_array
    int aindx = 0;
    // aindx index of closest cell
    int closest_aindx = -1;
    // Total weighting
    amrex::Real sum_wt = 0.;
    // Minimum distance to nearest acceptable cell center (meaning non-covered
    // with large enough volume fraction)
    amrex::Real min_dis = 1.E12;
    bool at_cent = false;
    // Loop over neighboring cells
    for (int kk = ks; kk <= ke; kk++) {
      for (int jj = jc - 1; jj <= jc; jj++) {
        auto y = static_cast<amrex::Real>(jj);
        for (int ii = ic - 1; ii <= ic; ii++) {
          auto x = static_cast<amrex::Real>(ii);
          indx_array[aindx] = {AMREX_D_DECL(ii, jj, kk)};
          if (
            !flags(ii, jj, kk).isCovered() &&
            vfrac(ii, jj, kk) > min_eb_vfrac) {
            // Distance to cell center
            AMREX_D_TERM(
              amrex::Real dlx = normposp[0] - (x + ccent(ii, jj, kk, 0));
              , amrex::Real dly = normposp[1] - (y + ccent(ii, jj, kk, 1));
              , auto dlz = normposp[2] - (static_cast<amrex::Real>(kk) +
                                          ccent(ii, jj, kk, 2)););
            amrex::Real len =
              std::sqrt(AMREX_D_TERM(dlx * dlx, +dly * dly, +dlz * dlz));
            if (len == 0.) {
              weights[aindx] = 1.;
              at_cent = true;
            } else {
              weights[aindx] = 1. / len;
              sum_wt += 1. / len;
            }
            if (len < min_dis) {
              closest_aindx = aindx;
              min_dis = len;
            }
          } else {
            weights[aindx] = 0.;
          }
          aindx++;
        }
      }
    }
    // Normalize weighting
    if (at_cent) {
      for (aindx = 1; aindx < AMREX_D_PICK(2, 4, 8); ++aindx) {
        weights[aindx] = 0.;
      }
    } else {
      for (aindx = 0; aindx < AMREX_D_PICK(2, 4, 8); ++aindx) {
        weights[aindx] /= sum_wt;
      }
    }
    ijkc = indx_array[closest_aindx];
    // If bad particle, move to near closest boundary face. Should only be
    // necessary for virtual particles but could be applied to all if current
    // implementation doesn't work
    if (bad_part) {
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        p.pos(dir) =
          (static_cast<amrex::Real>(ijkc[dir]) + 0.5 + bcent(ijkc, dir)) *
            dx[dir] +
          plo[dir];
      }
    }
  } else {
    // If not using inverse distance weighting, do the following
    amrex::GpuArray<
      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, AMREX_D_PICK(2, 4, 8)>
      nodes;

    // Here we pack the local 2x2x2 stencil into a local array. Note that
    // the node ordering is not consistent with the previous version.
    // Nodes 2 and 3 have been swapped as well as nodes 6 and 7. This was
    // to allow for a more compact for-loop filling.

    int lc = 0;
    for (int kk = ks; kk <= ke; kk++) {
      for (int jj = jc - 1; jj <= jc; jj++) {
        for (int ii = ic - 1; ii <= ic; ii++) {
          AMREX_D_TERM(
            nodes[lc][0] = (ii + 0.5 + ccent(ii, jj, kk, 0));
            , nodes[lc][1] = (jj + 0.5 + ccent(ii, jj, kk, 1));
            , nodes[lc][2] = (kk + 0.5 + ccent(ii, jj, kk, 2));)
          lc += 1;
        }
      }
    }

    amrex::Real xi = 0., eta = 0., zeta = 0.;
    AMREX_D_TERM(
      xi = normpos[0] - nodes[0][0];, eta = normpos[1] - nodes[0][1];
      , zeta = normpos[2] - nodes[0][2];);

    get_interp_mapping(normpos, nodes, xi, eta, zeta);
#if AMREX_SPACEDIM == 3
    indx_array[0] = {ic - 1, jc - 1, kc - 1};
    indx_array[1] = {ic, jc - 1, kc - 1};
    indx_array[2] = {ic, jc, kc - 1};
    indx_array[3] = {ic - 1, jc, kc - 1};
    indx_array[4] = {ic - 1, jc - 1, kc};
    indx_array[5] = {ic, jc - 1, kc};
    indx_array[6] = {ic, jc, kc};
    indx_array[7] = {ic - 1, jc, kc};
    weights[0] = 1. - xi - eta - zeta + xi * eta + xi * zeta + eta * zeta -
                 xi * eta * zeta;
    weights[1] = xi * (1. - eta - zeta + eta * zeta);
    weights[2] = xi * eta - xi * eta * zeta;
    weights[3] = eta * (1. - xi - zeta + xi * zeta);
    weights[4] = zeta * (1. - xi - eta + xi * eta);
    weights[5] = xi * zeta - xi * eta * zeta;
    weights[6] = xi * eta * zeta;
    weights[7] = eta * zeta - xi * eta * zeta;
#else
    indx_array[0] = {ic - 1, jc - 1};
    indx_array[1] = {ic, jc - 1};
    indx_array[2] = {ic, jc};
    indx_array[3] = {ic - 1, jc};
    weights[0] = 1. - xi - eta + xi * eta;
    weights[1] = xi * (1. - eta);
    weights[2] = xi * eta;
    weights[3] = eta * (1. - xi);
#endif
  }
  return do_fe_interp;
}

#endif // AMREX_USE_EB
#endif // SPRAYINTERPOLATION_H
