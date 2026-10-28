#ifndef TRANSPORT_PARAMS_H
#define TRANSPORT_PARAMS_H

#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>
#include "TransportTypes.H"
#include "TransportUtils.H"
#include "PeleParamsGeneric.H"
#include "EOS.H"

namespace pele::physics {
namespace transport {

struct ConstTransport;
struct SimpleTransport;
struct SutherlandTransport;

template <typename EOSType, typename TransportType>
struct TransParm
{
};

template <typename EOSType>
struct TransParm<EOSType, ConstTransport>
{
  amrex::Real const_viscosity{0.0};
  amrex::Real const_bulk_viscosity{0.0};
  amrex::Real const_diffusivity{0.0};
  amrex::Real const_conductivity{0.0};
  amrex::Real const_thermal_diffusion_ratio{0.0};
  bool use_soret = false;
};

template <typename EOSType>
struct TransParm<EOSType, SutherlandTransport>
{
  amrex::Real Prandtl_number{0.7};
  amrex::Real viscosity_mu_ref{17.16e-5};
  amrex::Real viscosity_T_ref{273.15};
  amrex::Real viscosity_S{110.4};
  amrex::Real const_bulk_viscosity{0.0};
  amrex::Real const_diffusivity{1.0};
  bool use_soret = false;
};

template <typename EOSType>
struct TransParm<EOSType, ManifoldTransport>
{
  const BlackBoxFunctionData* manf_data;
  int idx_rhoD{-1};
  int idx_mu{-1};
  int idx_cp{-1};
};

template <typename EOSType>
struct TransParm<EOSType, SimpleTransport>
{
  amrex::GpuArray<amrex::Real, NUM_SPECIES> wt = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> iwt = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> eps = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> sig = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> dip = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> pol = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> zrot = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_FIT> fitmu = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_FIT> fitlam = {0.0};
  amrex::
    GpuArray<amrex::Real, ((NUM_SPECIES * (NUM_SPECIES + 1) * NUM_FIT) / 2)>
      fitdbin = {0.0};
  bool use_soret = false;
  int numLite = 3;
  amrex::GpuArray<int, 3> liteSpec = {0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_FIT * 3> fittdrat = {0.0};
  amrex::GpuArray<int, NUM_SPECIES> nlin = {0};
  // Shift applied to mass/mole fractions so that vanishing species keep a
  // positive "trace" value: Y_tr = Y + eps * (sum(Y)/N - Y). The default is
  // the value that was hard-coded before the option existed.
  amrex::Real trace_epsilon{1.e-15};
  // Number of projected standard iterations for the multicomponent
  // diffusion matrix (2 reproduces EGlib's EGSDR1)
  int matrix_diffusion_niter{2};
  // Thermal conductivity model (ConductivityModel::Mixture/LCT1/LCT3) and the
  // number of conjugate-gradient iterations for the LCT models (3 = EGlib)
  int conductivity_model{ConductivityModel::Mixture};
  int conductivity_niter{3};
  // ln of the Lennard-Jones well depth of every species pair, including the
  // polar/non-polar correction (EGlib LEVEPS), packed as binary_pair_index
  amrex::GpuArray<amrex::Real, NUM_BINARY_PAIRS> log_epsij = {0.0};
};

template <>
struct TransParm<eos::SRK, SimpleTransport>
{
  amrex::GpuArray<amrex::Real, NUM_SPECIES> wt = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> iwt = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> eps = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> sig = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> dip = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> pol = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> zrot = {0.0};

  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_FIT> fitmu = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_FIT> fitlam = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES*(NUM_SPECIES + 1) / 2 * NUM_FIT>
    fitdbin = {0.0};
  bool use_soret = false;
  int numLite = 10;
  amrex::GpuArray<int, 10> liteSpec = {0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_FIT * 10> fittdrat = {0.0};
  amrex::GpuArray<int, NUM_SPECIES> nlin = {0};
  amrex::GpuArray<amrex::Real, 10 * 4> Afac = {0.0};
  amrex::GpuArray<amrex::Real, 7 * 4> Bfac = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_SPECIES> sqrtT2ij = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_SPECIES> sqrtEpsilonij = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_SPECIES> sqrtMWij = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_SPECIES> sqrtKappaij = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES * NUM_SPECIES * NUM_SPECIES>
    Upsilonijk = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> Kappai = {0.0};
  amrex::GpuArray<amrex::Real, NUM_SPECIES> omega = {0.0};
  amrex::Real trace_epsilon{1.e-15};
  int matrix_diffusion_niter{2};
  int conductivity_model{ConductivityModel::Mixture};
  int conductivity_niter{3};
  amrex::GpuArray<amrex::Real, NUM_BINARY_PAIRS> log_epsij = {0.0};
};

// ln(eps_ij) for every species pair: geometric mean of the well depths with
// the polar/non-polar correction of EGlib's LEVEPS (dipoles in Debye,
// polarizabilities in A^3, sigma in A, eps in K). Must be called after wt,
// eps, sig, dip and pol have been loaded.
template <typename TParm>
inline void
simple_transport_pair_epsilon(TParm* tparm)
{
  constexpr amrex::Real FAC = 1.0e-12;
  constexpr amrex::Real BOLTZ = 1.38056e-16;
  constexpr amrex::Real DIPMIN = 1.0e-20;
  for (int j = 0; j < NUM_SPECIES; ++j) {
    for (int i = 0; i <= j; ++i) {
      const amrex::Real epsi = tparm->eps[i];
      const amrex::Real epsj = tparm->eps[j];
      amrex::Real epsij = std::sqrt(epsi * epsj);
      int polar = -1, nonpolar = -1;
      if (tparm->dip[j] < DIPMIN && tparm->dip[i] > DIPMIN) {
        polar = i;
        nonpolar = j;
      } else if (tparm->dip[i] < DIPMIN && tparm->dip[j] > DIPMIN) {
        polar = j;
        nonpolar = i;
      }
      if (polar >= 0) {
        const amrex::Real sig3p = std::pow(tparm->sig[polar], 3);
        const amrex::Real sig3n = std::pow(tparm->sig[nonpolar], 3);
        const amrex::Real xi =
          1.0 + 0.25 * (tparm->pol[nonpolar] / sig3n) * (FAC / BOLTZ) *
                  (tparm->dip[polar] * tparm->dip[polar] /
                   (tparm->eps[polar] * sig3p)) *
                  std::sqrt(tparm->eps[polar] / tparm->eps[nonpolar]);
        epsij *= xi * xi;
      }
      tparm->log_epsij[transport::binary_pair_index(i, j)] = std::log(epsij);
    }
  }
}

// Read and validate the runtime options shared by all SimpleTransport
// specializations
template <typename TParm>
inline void
simple_transport_query_common(TParm* tparm)
{
  amrex::ParmParse pp("transport");
  pp.query("trace_epsilon", tparm->trace_epsilon);
  pp.query("matrix_diffusion_niter", tparm->matrix_diffusion_niter);
  if (tparm->trace_epsilon < 1.e-30 || tparm->trace_epsilon > 1.e-6) {
    amrex::Abort(
      "transport.trace_epsilon must be in [1e-30, 1e-6] (default 1e-15)");
  }
  if (tparm->matrix_diffusion_niter < 1) {
    amrex::Abort("transport.matrix_diffusion_niter must be >= 1 (default 2)");
  }
  std::string model = "mixture";
  pp.query("conductivity_model", model);
  if (model == "mixture") {
    tparm->conductivity_model = ConductivityModel::Mixture;
  } else if (model == "lct1") {
    tparm->conductivity_model = ConductivityModel::LCT1;
  } else if (model == "lct3") {
    tparm->conductivity_model = ConductivityModel::LCT3;
  } else {
    amrex::Abort(
      "transport.conductivity_model must be mixture (default), lct1 or lct3");
  }
#ifndef PELE_TRANSPORT_LCT
  if (tparm->conductivity_model != ConductivityModel::Mixture) {
    amrex::Abort(
      "transport.conductivity_model = lct1/lct3 requested but the code was "
      "built without the LCT models: build with USE_LCT_TRANSPORT = TRUE "
      "(GNU make) or -DPELE_TRANSPORT_LCT");
  }
#endif
  pp.query("conductivity_niter", tparm->conductivity_niter);
  if (tparm->conductivity_niter < 1) {
    amrex::Abort("transport.conductivity_niter must be >= 1 (default 3)");
  }
  simple_transport_pair_epsilon(tparm);
}

} // namespace transport

template <typename EOSType>
struct InitParm<transport::TransParm<EOSType, transport::ConstTransport>>
{
  static void host_initialize(
    PeleParams<transport::TransParm<EOSType, transport::ConstTransport>>*
      parm_in)
  {
    transport::TransParm<EOSType, transport::ConstTransport>* tparm =
      &(parm_in->m_h_parm);
    amrex::ParmParse pp("transport");
    pp.query("const_viscosity", tparm->const_viscosity);
    pp.query("const_bulk_viscosity", tparm->const_bulk_viscosity);
    pp.query("const_conductivity", tparm->const_conductivity);
    pp.query("const_diffusivity", tparm->const_diffusivity);
    pp.query(
      "const_thermal_diffusion_ratio", tparm->const_thermal_diffusion_ratio);
    if (std::abs(tparm->const_thermal_diffusion_ratio) > 1e-24) {
      tparm->use_soret = true;
    }
    std::string units = "CGS";
    pp.query("units", units);
    if (units == "MKS") {
      // If input in MKS, convert to CGS
      tparm->const_viscosity *= amrex::Real(1.0e1);
      tparm->const_bulk_viscosity *= amrex::Real(1.0e1);
      tparm->const_diffusivity *= amrex::Real(1.0e1);
      tparm->const_conductivity *= amrex::Real(1.0e5);
    } else if (units != "CGS") {
      amrex::Abort(
        "Unable to initialize transport using unknown units " + units +
        ". Use CGS (def) or MKS");
    }
  }

  static void host_deallocate(
    PeleParams<transport::TransParm<EOSType, transport::ConstTransport>>*
    /*parm_in*/)
  {
  }
};

template <typename EOSType>
struct InitParm<transport::TransParm<EOSType, transport::SutherlandTransport>>
{

  static void host_initialize(
    PeleParams<transport::TransParm<EOSType, transport::SutherlandTransport>>*
      parm_in)
  {
    transport::TransParm<EOSType, transport::SutherlandTransport>* tparm =
      &(parm_in->m_h_parm);
    amrex::ParmParse pp("transport");
    pp.query("Prandtl_number", tparm->Prandtl_number);
    pp.query("viscosity_mu_ref", tparm->viscosity_mu_ref);
    pp.query("viscosity_T_ref", tparm->viscosity_T_ref);
    pp.query("viscosity_S", tparm->viscosity_S);
    pp.query("const_bulk_viscosity", tparm->const_bulk_viscosity);
    pp.query("const_diffusivity", tparm->const_diffusivity);
  }

  static void host_deallocate(
    PeleParams<transport::TransParm<EOSType, transport::SutherlandTransport>>*
    /*parm_in*/)
  {
  }
};

template <typename EOSType>
struct HostOnlyParm<transport::TransParm<EOSType, transport::ManifoldTransport>>
{
  std::shared_ptr<PeleParamsGeneric<BlackBoxFunctionData>> manfunc_par;
  bool use_eos_manifold = true;
};

template <typename EOSType>
struct InitParm<transport::TransParm<EOSType, transport::ManifoldTransport>>
{
  static void host_initialize(
    PeleParams<transport::TransParm<EOSType, transport::ManifoldTransport>>*
      parm_in)
  {
    amrex::ParmParse pp("manifold_transport");
    pp.query("use_eos_manifold", parm_in->m_host_only_parm.use_eos_manifold);

    if (
      parm_in->m_host_only_parm.manfunc_par == nullptr &&
      parm_in->m_host_only_parm.use_eos_manifold) {
      amrex::Abort(
        "For manifold transport tabulated or neural network data must "
        "be provided to transport parameters by setting the host_only_parm"
        "unless you set manifold_transport.use_eos_manifold = false");
    } else if (!parm_in->m_host_only_parm.use_eos_manifold) {
      std::string manifold_model;
      pp.get("model", manifold_model);
      parm_in->m_host_only_parm.manfunc_par =
        PeleParamsGeneric<BlackBoxFunctionData>::create(manifold_model);
      parm_in->m_host_only_parm.manfunc_par->host_only_parm()
        .parm_parse_prefix = "manifold_transport";
      parm_in->m_host_only_parm.manfunc_par->initialize();
    }

    transport::TransParm<EOSType, transport::ManifoldTransport>* tparm =
      &(parm_in->m_h_parm);
    tparm->manf_data = parm_in->m_host_only_parm.manfunc_par->device_parm();
    tparm->idx_rhoD = get_var_index(
      "DIFF", &parm_in->m_host_only_parm.manfunc_par->host_parm());
    tparm->idx_mu = get_var_index(
      "VISC", &parm_in->m_host_only_parm.manfunc_par->host_parm());
#ifdef PELE_USE_SPRAY
    tparm->idx_cp =
      get_var_index("CP", &parm_in->m_host_only_parm.manfunc_par->host_parm());
#endif
  }

  static void host_deallocate(
    PeleParams<transport::TransParm<EOSType, transport::ManifoldTransport>>*
      parm_in)
  {
    if (!parm_in->m_host_only_parm.use_eos_manifold) {
      parm_in->m_host_only_parm.manfunc_par->deallocate();
    }
    parm_in->m_host_only_parm.manfunc_par = nullptr;
  }
};

template <typename EOSType>
struct InitParm<transport::TransParm<EOSType, transport::SimpleTransport>>
{
  static void host_initialize(
    PeleParams<transport::TransParm<EOSType, transport::SimpleTransport>>*
      parm_in)
  {
    transport::TransParm<EOSType, transport::SimpleTransport>* tparm =
      &(parm_in->m_h_parm);
    egtransetWT(tparm->wt.data());
    egtransetEPS(tparm->eps.data());
    egtransetSIG(tparm->sig.data());
    egtransetDIP(tparm->dip.data());
    egtransetPOL(tparm->pol.data());
    egtransetZROT(tparm->zrot.data());
    egtransetNLIN(tparm->nlin.data());
    egtransetCOFETA(tparm->fitmu.data());
    egtransetCOFLAM(tparm->fitlam.data());
    egtransetCOFD(tparm->fitdbin.data());
    amrex::ParmParse pp("transport");
    pp.query("use_soret", tparm->use_soret);
    if (tparm->use_soret) {
      egtransetNLITE(&tparm->numLite);
      egtransetKTDIF(tparm->liteSpec.data());
      egtransetCOFTD(tparm->fittdrat.data());
    }
    transport::simple_transport_query_common(tparm);
    for (int i = 0; i < NUM_SPECIES; ++i) {
      tparm->iwt[i] = 1. / tparm->wt[i];
    }
  }

  static void host_deallocate(
    PeleParams<transport::TransParm<EOSType, transport::SimpleTransport>>*
    /*parm_in*/)
  {
  }
};

template <>
struct InitParm<transport::TransParm<eos::SRK, transport::SimpleTransport>>
{
  static void host_initialize(
    PeleParams<transport::TransParm<eos::SRK, transport::SimpleTransport>>*
      parm_in)
  {
    transport::TransParm<eos::SRK, transport::SimpleTransport>* tparm =
      &(parm_in->m_h_parm);
    egtransetWT(tparm->wt.data());
    egtransetEPS(tparm->eps.data());
    egtransetSIG(tparm->sig.data());
    egtransetDIP(tparm->dip.data());
    egtransetPOL(tparm->pol.data());
    egtransetZROT(tparm->zrot.data());
    egtransetNLIN(tparm->nlin.data());
    egtransetCOFETA(tparm->fitmu.data());
    egtransetCOFLAM(tparm->fitlam.data());
    egtransetCOFD(tparm->fitdbin.data());
    egtransetNLITE(&tparm->numLite);
    egtransetKTDIF(tparm->liteSpec.data());
    egtransetCOFTD(tparm->fittdrat.data());
    transport::simple_transport_query_common(tparm);

    for (int i = 0; i < NUM_SPECIES; ++i) {
      tparm->iwt[i] = 1. / tparm->wt[i];
    }

    // Nonideal transport coefficients computed using Chung's method:
    // Chung, T.H., Ajlan, M., Lee, L.L. and Starling, K.E., 1988.
    // Generalized multiparameter correlation for nonpolar and polar
    // fluid transport properties. Industrial & engineering chemistry
    // research, 27(4), pp.671-679.

    // Initialize coefficients of model
    {
      amrex::Real tmp1[NUM_SPECIES], tmp2[NUM_SPECIES], tmp3[NUM_SPECIES];
      GET_CRITPARAMS(tmp1, tmp2, tmp3, tparm->omega.data());
    }

    tparm->Afac[0] = 6.32402;
    tparm->Afac[1] = 50.41190;
    tparm->Afac[2] = -51.68010;
    tparm->Afac[3] = 1189.020;
    tparm->Afac[4] = 0.12102e-2;
    tparm->Afac[5] = -0.11536e-2;
    tparm->Afac[6] = -0.62571e-2;
    tparm->Afac[7] = 0.37283e-1;
    tparm->Afac[8] = 5.28346;
    tparm->Afac[9] = 254.209;
    tparm->Afac[10] = -168.481;
    tparm->Afac[11] = 3898.27;
    tparm->Afac[12] = 6.62263;
    tparm->Afac[13] = 38.09570;
    tparm->Afac[14] = -8.46414;
    tparm->Afac[15] = 31.41780;
    tparm->Afac[16] = 19.74540;
    tparm->Afac[17] = 7.63034;
    tparm->Afac[18] = -14.35440;
    tparm->Afac[19] = 31.52670;
    tparm->Afac[20] = -1.89992;
    tparm->Afac[21] = -12.5367;
    tparm->Afac[22] = 4.98529;
    tparm->Afac[23] = -18.15070;
    tparm->Afac[24] = 24.27450;
    tparm->Afac[25] = 3.44945;
    tparm->Afac[26] = -11.29130;
    tparm->Afac[27] = 69.34660;
    tparm->Afac[28] = 0.79716;
    tparm->Afac[29] = 1.11764;
    tparm->Afac[30] = 0.12348e-1;
    tparm->Afac[31] = -4.11661;
    tparm->Afac[32] = -0.23816;
    tparm->Afac[33] = 0.67695e-1;
    tparm->Afac[34] = -0.81630;
    tparm->Afac[35] = 4.02528;
    tparm->Afac[36] = 0.68629e-1;
    tparm->Afac[37] = 0.34793;
    tparm->Afac[38] = 0.59256;
    tparm->Afac[39] = -0.72663;

    tparm->Bfac[0] = 2.41657;
    tparm->Bfac[1] = 0.74824;
    tparm->Bfac[2] = -0.91858;
    tparm->Bfac[3] = 121.721;
    tparm->Bfac[4] = -0.50924;
    tparm->Bfac[5] = -1.50936;
    tparm->Bfac[6] = -49.99120;
    tparm->Bfac[7] = 69.9834;
    tparm->Bfac[8] = 6.61069;
    tparm->Bfac[9] = 5.62073;
    tparm->Bfac[10] = 64.75990;
    tparm->Bfac[11] = 27.0389;
    tparm->Bfac[12] = 14.54250;
    tparm->Bfac[13] = -8.91387;
    tparm->Bfac[14] = -5.63794;
    tparm->Bfac[15] = 74.3435;
    tparm->Bfac[16] = 0.79274;
    tparm->Bfac[17] = 0.82019;
    tparm->Bfac[18] = -0.69369;
    tparm->Bfac[19] = 6.31734;
    tparm->Bfac[20] = -5.86340;
    tparm->Bfac[21] = 12.80050;
    tparm->Bfac[22] = 9.58926;
    tparm->Bfac[23] = -65.5292;
    tparm->Bfac[24] = 81.17100;
    tparm->Bfac[25] = 114.15800;
    tparm->Bfac[26] = -60.84100;
    tparm->Bfac[27] = 466.7750;

    // Initialize Kappa, which has nonzero values only for specific polar
    // species
    for (int i = 0; i < NUM_SPECIES; ++i) {
      tparm->Kappai[i] = 0.0;
    }
    {
      amrex::Vector<std::string> spec_names_kappa;
      spec_names_kappa.resize(1);
      spec_names_kappa[0] = "Null";
      CKSYMS_STR(spec_names_kappa);
      for (int i = 0; i < NUM_SPECIES; i++) {
        if (spec_names_kappa[i] == "H2O") {
          tparm->Kappai[i] = 0.075908;
        } else if (spec_names_kappa[i] == "CH3OH") {
          tparm->Kappai[i] = 0.215175;
        } else if (spec_names_kappa[i] == "CH3CH2OH") {
          tparm->Kappai[i] = 0.174823;
        }
      }
    }
    for (int i = 0; i < NUM_SPECIES; ++i) {
      for (int j = 0; j < NUM_SPECIES; ++j) {
        const int idx = i * NUM_SPECIES + j;
        tparm->sqrtT2ij[idx] = std::sqrt(tparm->sig[i] * tparm->sig[j]);
        tparm->sqrtEpsilonij[idx] = std::sqrt(tparm->eps[i] * tparm->eps[j]);
        tparm->sqrtMWij[idx] = std::sqrt(2.0 / (tparm->iwt[i] + tparm->iwt[j]));
        tparm->sqrtKappaij[idx] =
          std::sqrt(tparm->Kappai[i] * tparm->Kappai[j]);
      }
    }

    for (int i = 0; i < NUM_SPECIES; ++i) {
      for (int j = 0; j < NUM_SPECIES; ++j) {
        // cppcheck-suppress knownConditionTrueFalse
        if (i != j) {
          const int idx_ij = i + NUM_SPECIES * j;
          const amrex::Real S_ij =
            0.5 * (tparm->sig[i] + tparm->sig[j]) * 1e-8; // converted to cm
          const amrex::Real S_ij_inv = 1.0 / S_ij;
          for (int k = 0; k < NUM_SPECIES; ++k) {
            const amrex::Real S_ik =
              0.5 * (tparm->sig[i] + tparm->sig[k]) * 1e-8; // converted to cm
            const amrex::Real S_jk =
              0.5 * (tparm->sig[j] + tparm->sig[k]) * 1e-8; // converted to cm
            tparm->Upsilonijk[idx_ij * NUM_SPECIES + k] =
              tparm->iwt[k] *
              (8.0 * (S_ik * S_ik * S_ik + S_jk * S_jk * S_jk) -
               6.0 * (S_ik * S_ik + S_jk * S_jk) * S_ij -
               3.0 *
                 ((S_ik * S_ik - S_jk * S_jk) * (S_ik * S_ik - S_jk * S_jk)) *
                 S_ij_inv +
               S_ij * S_ij * S_ij);
          }
        }
      }
    }
  }

  static void host_deallocate(
    PeleParams<transport::TransParm<eos::SRK, transport::SimpleTransport>>*
    /*parm_in*/)
  {
  }
};

} // namespace pele::physics
#endif
