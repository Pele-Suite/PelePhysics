#ifndef MANIFOLDTRANSPORT_H
#define MANIFOLDTRANSPORT_H

#include "TransportParams.H"

namespace pele::physics::transport {

struct ManifoldTransport
{
  using transport_type = ManifoldTransport;

  static std::string identifier() { return "Manifold"; }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void transport(
    const bool wtr_get_xi,
    const bool wtr_get_mu,
    const bool wtr_get_lam,
    const bool wtr_get_Ddiag,
    const bool wtr_get_chi,
    const amrex::Real /*Tloc*/,
    const amrex::Real /*rholoc*/,
    const amrex::Real* Yloc,
    amrex::Real* Ddiag,
    amrex::Real* /*chi_mix*/,
    amrex::Real& mu,
    amrex::Real& /*xi*/,
    amrex::Real& lam,
    TransParm<EosType, transport_type> const* tparm)
  {
    BlackBoxFunctionFactory<eos::ManifoldFunctionType> manfunc(
      tparm->manf_data);

    if (wtr_get_mu) {
      manfunc.get_func()->get_value(tparm->idx_mu, Yloc, mu);
    }

    if (wtr_get_lam) {
      // Under the Lewis number=1 assumption, lambda=rho*Cp*D
      amrex::Real cp = 0.0;
      amrex::Real rhoD = 0.0;
      if (tparm->idx_cp > 0) {
        manfunc.get_func()->get_value(tparm->idx_cp, Yloc, cp);
      }
      manfunc.get_func()->get_value(tparm->idx_rhoD, Yloc, rhoD);
      lam = cp * rhoD;
    }

    if (wtr_get_xi) {
      // could be added to table if desired
      amrex::Abort(
        "Use of bulk viscosity is not supported for ManifoldTransport");
    }

    if (wtr_get_chi) {
      // could be added to table if desired
      amrex::Abort(
        "Use of soret diffusion is not supported for ManifoldTransport");
    }

    if (wtr_get_Ddiag) {
      amrex::Real Dtemp{0.0};
      manfunc.get_func()->get_value(tparm->idx_rhoD, Yloc, Dtemp);
      for (int i = 0; i < NUM_SPECIES - 1; ++i) {
        Ddiag[i] = Dtemp;
      }
      Ddiag[NUM_SPECIES - 1] = 0.0; // Placeholder for density
    }
  }

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void get_transport_coeffs(
    amrex::Box const& bx,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& /*T_in*/,
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

    const bool wtr_get_xi = false;
    const bool wtr_get_mu = true;
    const bool wtr_get_lam = false;
    const bool wtr_get_Ddiag = true;
    const bool wtr_get_chi = false;

    for (int k = lo.z; k <= hi.z; ++k) {
      for (int j = lo.y; j <= hi.y; ++j) {
        for (int i = lo.x; i <= hi.x; ++i) {

          amrex::Real massloc[NUM_SPECIES] = {0.0};
          for (int n = 0; n < NUM_SPECIES; ++n) {
            massloc[n] = Y_in(i, j, k, n);
          }

          amrex::Real muloc = 0.0, xiloc = 0.0, lamloc = 0.0;
          amrex::Real Ddiag[NUM_SPECIES] = {0.0};
          amrex::Real chi_loc[NUM_SPECIES] = {0.0};
          transport(
            wtr_get_xi, wtr_get_mu, wtr_get_lam, wtr_get_Ddiag, wtr_get_chi,
            0.0, 0.0, massloc, Ddiag, chi_loc, muloc, xiloc, lamloc, tparm);

          // mu, xi and lambda are stored after D in the diffusion multifab
          for (int n = 0; n < NUM_SPECIES; ++n) {
            D_out(i, j, k, n) = Ddiag[n];
            chi_out(i, j, k, n) = chi_loc[n];
          }

          mu_out(i, j, k) = muloc;
          xi_out(i, j, k) = xiloc;
          lam_out(i, j, k) = lamloc;
        }
      }
    }
  }
  // Extended API: the multicomponent diffusion matrix and the rescaled
  // thermal diffusion ratios are not available for manifold transport.
  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void transport(
    const TransportRequest& req,
    const amrex::Real Tloc,
    const amrex::Real rholoc,
    const amrex::Real* Yloc,
    const TransportOutputs& out,
    TransParm<EosType, transport_type> const* tparm)
  {
    if (req.Dmat || req.chit) {
      amrex::Abort(
        "Multicomponent diffusion matrix and rescaled thermal "
        "diffusion ratios are not supported for ManifoldTransport");
    }
    amrex::Real mu_l = 0.0, xi_l = 0.0, lam_l = 0.0;
    transport(
      req.xi, req.mu, req.lam, req.Ddiag && out.Ddiag != nullptr,
      req.chi && out.chi != nullptr, Tloc, rholoc, Yloc, out.Ddiag, out.chi,
      mu_l, xi_l, lam_l, tparm);
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

  AMREX_GPU_HOST_DEVICE
  AMREX_FORCE_INLINE
  static void get_transport_coeffs(
    amrex::Box const& bx,
    const TransportRequest& req,
    amrex::Array4<const amrex::Real> const& Y_in,
    amrex::Array4<const amrex::Real> const& T_in,
    amrex::Array4<const amrex::Real> const& Rho_in,
    amrex::Array4<amrex::Real> const& D_out,
    amrex::Array4<amrex::Real> const& chi_out,
    amrex::Array4<amrex::Real> const& /*Dmat_out*/,
    amrex::Array4<amrex::Real> const& /*chit_out*/,
    amrex::Array4<amrex::Real> const& mu_out,
    amrex::Array4<amrex::Real> const& xi_out,
    amrex::Array4<amrex::Real> const& lam_out,
    TransParm<EosType, transport_type> const* tparm)
  {
    if (req.Dmat || req.chit) {
      amrex::Abort(
        "Multicomponent diffusion matrix and rescaled thermal "
        "diffusion ratios are not supported for ManifoldTransport");
    }
    get_transport_coeffs(
      bx, Y_in, T_in, Rho_in, D_out, chi_out, mu_out, xi_out, lam_out, tparm);
  }

  template <class... Args>
  AMREX_GPU_HOST_DEVICE ManifoldTransport(Args... /*args*/)
  {
  }
};

} // namespace pele::physics::transport
#endif
