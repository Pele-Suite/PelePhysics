#ifndef SUTHERLANDTRANSPORT_H
#define SUTHERLANDTRANSPORT_H

#include "TransportParams.H"
#include "TransportUtils.H"

namespace pele::physics::transport {
struct SutherlandTransport
{
  using transport_type = SutherlandTransport;

  static std::string identifier() { return "Sutherland"; }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void transport(
    const bool wtr_get_xi,
    const bool wtr_get_mu,
    const bool wtr_get_lam,
    const bool wtr_get_Ddiag,
    const bool wtr_get_chi,
    const amrex::Real Tloc,
    const amrex::Real /*rholoc*/,
    amrex::Real* Yloc,
    amrex::Real* Ddiag,
    amrex::Real* chi_mix,
    amrex::Real& mu,
    amrex::Real& xi,
    amrex::Real& lam,
    TransParm<EosType, transport_type> const* tparm)
  {

    if (wtr_get_mu) {
      mu = tparm->viscosity_mu_ref *
           std::pow((Tloc / tparm->viscosity_T_ref), 1.5) *
           (tparm->viscosity_T_ref + tparm->viscosity_S) /
           (Tloc + tparm->viscosity_S);
    }

    if (wtr_get_lam) {
      amrex::Real muloc = tparm->viscosity_mu_ref *
                          std::pow((Tloc / tparm->viscosity_T_ref), 1.5) *
                          (tparm->viscosity_T_ref + tparm->viscosity_S) /
                          (Tloc + tparm->viscosity_S);

      amrex::Real Cpmix = 0.0, Tloc_cpy = Tloc;
      CKCPBS(Tloc_cpy, Yloc, Cpmix);
      lam = muloc * Cpmix / tparm->Prandtl_number;
    }

    if (wtr_get_xi) {
      xi = tparm->const_bulk_viscosity;
    }

    if (wtr_get_Ddiag) {
      for (int i = 0; i < NUM_SPECIES; ++i) {
        Ddiag[i] = tparm->const_diffusivity;
      }
    }

    if (wtr_get_chi) {
      for (int i = 0; i < NUM_SPECIES; ++i) {
        chi_mix[i] = 0.0;
      }
    }
  }

  // Extended API: additionally returns the flux diffusion matrix
  // Dmat[i*NUM_SPECIES + j] (uniform diffusivity, projected for mass
  // conservation).
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
    amrex::Real* Yloc,
    amrex::Real* Ddiag,
    amrex::Real* chi_mix,
    amrex::Real* Dmat,
    amrex::Real& mu,
    amrex::Real& xi,
    amrex::Real& lam,
    TransParm<EosType, transport_type> const* tparm)
  {
    transport(
      wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi, Tloc,
      rholoc, Yloc, Ddiag, chi_mix, mu, xi, lam, tparm);
    if (wtr_get_Dmat) {
      uniform_diffusion_matrix(tparm->const_diffusivity, Yloc, Dmat);
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void get_transport_coeffs(
    amrex::Box const& bx,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& /*Rho_in*/,
    amrex::Array4<amrex::Real> const& D_out,
    amrex::Array4<amrex::Real> const& chi_out,
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

          amrex::Real T = T_in(i, j, k);
          amrex::Real massloc[NUM_SPECIES] = {0.0};
          for (int n = 0; n < NUM_SPECIES; ++n) {
            massloc[n] = Y_in(i, j, k, n);
          }

          amrex::Real muloc, xiloc, lamloc;
          amrex::Real Ddiag[NUM_SPECIES] = {0.0};
          amrex::Real chi_mix[NUM_SPECIES] = {0.0};
          transport(
            wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi, T,
            0.0, massloc, Ddiag, chi_mix, muloc, xiloc, lamloc, tparm);

          // mu, xi and lambda are stored after D in the diffusion multifab
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
  // Extended API: Dmat_out must have NUM_SPECIES*NUM_SPECIES components
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void get_transport_coeffs(
    amrex::Box const& bx,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& /*Rho_in*/,
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

          amrex::Real T = T_in(i, j, k);
          amrex::Real massloc[NUM_SPECIES] = {0.0};
          for (int n = 0; n < NUM_SPECIES; ++n) {
            massloc[n] = Y_in(i, j, k, n);
          }
          amrex::Real muloc, xiloc, lamloc;
          amrex::Real Ddiag[NUM_SPECIES] = {0.0};
          amrex::Real chi_mix[NUM_SPECIES] = {0.0};
          amrex::Real Dmat[NUM_SPECIES * NUM_SPECIES] = {0.0};
          transport(
            wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi,
            true, T, 0.0, massloc, Ddiag, chi_mix, Dmat, muloc, xiloc, lamloc,
            tparm);

          for (int n = 0; n < NUM_SPECIES; ++n) {
            D_out(i, j, k, n) = Ddiag[n];
            if (wtr_get_chi) {
              chi_out(i, j, k, n) = chi_mix[n];
            }
          }
          for (int n = 0; n < NUM_SPECIES * NUM_SPECIES; ++n) {
            Dmat_out(i, j, k, n) = Dmat[n];
          }
          mu_out(i, j, k) = muloc;
          xi_out(i, j, k) = xiloc;
          lam_out(i, j, k) = lamloc;
        }
      }
    }
  }

  template <class... Args>
  AMREX_GPU_HOST_DEVICE explicit SutherlandTransport(Args... /*unused*/)
  {
  }
};

} // namespace pele::physics::transport
#endif
