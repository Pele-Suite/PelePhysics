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
    const amrex::Real* Yloc,
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

  // Extended API: evaluates the quantities selected in req and writes them
  // to the corresponding non-null members of out (see TransportTypes.H);
  // outputs that are not requested are left untouched.
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void transport(
    const TransportRequest& req,
    const amrex::Real Tloc,
    const amrex::Real rholoc,
    const amrex::Real* Yloc,
    const TransportOutputs& out,
    TransParm<EosType, transport_type> const* tparm)
  {
    if (req.chit) {
      amrex::Abort(
        "Rescaled thermal diffusion ratios (chit) are not available for "
        "SutherlandTransport");
    }
    if (req.Dmat && out.Dmat != nullptr) {
      uniform_diffusion_matrix(tparm->const_diffusivity, Yloc, out.Dmat);
    }
    amrex::Real mu_l = 0.0, xi_l = 0.0, lam_l = 0.0;
    transport(
      req.xi, req.mu, req.lam, req.Ddiag && out.Ddiag != nullptr,
      req.chi && out.chi != nullptr, Tloc, rholoc, Yloc, out.Ddiag, out.chi,
      mu_l, xi_l, lam_l, tparm);
    if (req.mu && out.mu != nullptr) {
      *out.mu = mu_l;
    }
    if (req.xi && out.xi != nullptr) {
      *out.xi = xi_l;
    }
    if (req.lam && out.lam != nullptr) {
      *out.lam = lam_l;
    }
  }

  // Extended API: Dmat_out needs NUM_SPECIES*NUM_SPECIES components when
  // requested; outputs that are not requested may be any Array4.
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE static void get_transport_coeffs(
    amrex::Box const& bx,
    const TransportRequest& req,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& /*Rho_in*/,
    amrex::Array4<amrex::Real> const& D_out,
    amrex::Array4<amrex::Real> const& chi_out,
    amrex::Array4<amrex::Real> const& Dmat_out,
    amrex::Array4<amrex::Real> const& /*chit_out*/,
    amrex::Array4<amrex::Real> const& mu_out,
    amrex::Array4<amrex::Real> const& xi_out,
    amrex::Array4<amrex::Real> const& lam_out,
    TransParm<EosType, transport_type> const* tparm)
  {
    AMREX_ASSERT(!req.Ddiag || D_out.nComp() >= NUM_SPECIES);
    AMREX_ASSERT(!req.chi || chi_out.nComp() >= NUM_SPECIES);
    AMREX_ASSERT(!req.Dmat || Dmat_out.nComp() >= NUM_SPECIES * NUM_SPECIES);
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {
          const amrex::Real T = T_in(i, j, k);
          amrex::Real massloc[NUM_SPECIES] = {0.0};
          for (int n = 0; n < NUM_SPECIES; ++n) {
            massloc[n] = Y_in(i, j, k, n);
          }
          amrex::Real muloc = 0.0, xiloc = 0.0, lamloc = 0.0;
          amrex::Real Ddiag[NUM_SPECIES] = {0.0};
          amrex::Real chi_mix[NUM_SPECIES] = {0.0};
          amrex::Real Dmat[NUM_SPECIES * NUM_SPECIES] = {0.0};
          TransportOutputs out;
          out.mu = &muloc;
          out.xi = &xiloc;
          out.lam = &lamloc;
          out.Ddiag = Ddiag;
          out.chi = chi_mix;
          out.Dmat = Dmat;
          transport(req, T, 0.0, massloc, out, tparm);

          for (int n = 0; n < NUM_SPECIES; ++n) {
            if (req.Ddiag) {
              D_out(i, j, k, n) = Ddiag[n];
            }
            if (req.chi) {
              chi_out(i, j, k, n) = chi_mix[n];
            }
          }
          if (req.Dmat) {
            for (int n = 0; n < NUM_SPECIES * NUM_SPECIES; ++n) {
              Dmat_out(i, j, k, n) = Dmat[n];
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

  template <class... Args>
  AMREX_GPU_HOST_DEVICE explicit SutherlandTransport(Args... /*unused*/)
  {
  }
};

} // namespace pele::physics::transport
#endif
