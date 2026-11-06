#ifndef EOS_PARAMS_H
#define EOS_PARAMS_H

#include <AMReX_REAL.H>
#include <AMReX_ParmParse.H>

#include "mechanism.H"
#include "BlackBoxFunction.H"
#include "PhysicsConstants.H"
#include "PeleParamsGeneric.H"

namespace pele::physics {
namespace eos {

enum class density_lookup_type { linear, log, inverse };

struct GammaLaw;
struct Fuego;
struct SRK;
struct Manifold;

template <typename EOSType>
struct EosParm
{
};

template <>
struct EosParm<GammaLaw>
{
  amrex::Real gamma{Constants::gamma};
};

#ifndef MANIFOLD_DIM
#define MANIFOLD_DIM 1
#endif

template <>
struct EosParm<Manifold>
{
  const BlackBoxFunctionData* manf_data;
  amrex::Real Pnom_cgs;
  density_lookup_type dens_lookup;
  bool compute_temperature{false};
  int idx_density{0};
  int idx_Wbar{0};
  int idx_T{0};
  int idx_Wdot[MANIFOLD_DIM];
  int idx_Cp{-1};
  int idx_Cp_fuel{-1};
  int is_variance_of[MANIFOLD_DIM];
  int verbose{1};
  bool has_mani_src{true};
  bool has_species_mw{false};
  amrex::Real* chemSpecies_Mwt;
  int* idx_chemspecies;
  int num_chemspecies;
};

} // namespace eos

template <>
struct InitParm<eos::EosParm<eos::GammaLaw>>
{
  static void host_initialize(PeleParams<eos::EosParm<eos::GammaLaw>>* parm_in)
  {
    amrex::ParmParse pp("eos");
    pp.query("gamma", parm_in->m_h_parm.gamma);
    if (parm_in->m_h_parm.gamma != Constants::gamma) {
      amrex::Warning(
        "Runtime gamma does not match compile time gamma, use caution because "
        "runtime specification of gamma may not yet be supported everywhere");
    }
  }

  static void
  host_deallocate(PeleParams<eos::EosParm<eos::GammaLaw>>* /*parm_in*/)
  {
  }
};

template <>
struct HostOnlyParm<eos::EosParm<eos::Manifold>>
{
  std::shared_ptr<PeleParamsGeneric<BlackBoxFunctionData>> manfunc_par;
};

template <>
struct InitParm<eos::EosParm<eos::Manifold>>
{
  static void host_initialize(PeleParams<eos::EosParm<eos::Manifold>>* parm_in)
  {
    amrex::ParmParse pp("manifold");
    std::string metadata_filename;
    pp.query("metadata_filename", metadata_filename);
    if (!metadata_filename.empty()) {
      pp.addfile(metadata_filename);
    }

    std::string manifold_model;
    pp.get("model", manifold_model);
    parm_in->m_host_only_parm.manfunc_par =
      PeleParamsGeneric<BlackBoxFunctionData>::create(manifold_model);
    parm_in->m_host_only_parm.manfunc_par->host_only_parm().parm_parse_prefix =
      "manifold";
    parm_in->m_host_only_parm.manfunc_par->initialize();
    const BlackBoxFunctionData* d_manf_data_in =
      parm_in->m_host_only_parm.manfunc_par->device_parm();
    BlackBoxFunctionData* h_manf_data_in =
      &(parm_in->m_host_only_parm.manfunc_par->host_parm());

    parm_in->m_h_parm.manf_data = d_manf_data_in;
    // First (N-1) species are table dimensions (last species corresponds to
    // density)
    AMREX_ALWAYS_ASSERT(h_manf_data_in->Ndim == NUM_SPECIES - 1);

    pp.query("v", parm_in->m_h_parm.verbose);
    int verbose = parm_in->m_h_parm.verbose;
    pp.get("nominal_pressure_cgs", parm_in->m_h_parm.Pnom_cgs);
    pp.query("has_mani_src", parm_in->m_h_parm.has_mani_src);
    pp.get("compute_temperature", parm_in->m_h_parm.compute_temperature);

    bool required = false;
#ifdef PELE_USE_SPRAY
    required = true;
    amrex::Vector<std::string> spray_species;
    {
      amrex::ParmParse ppp("particles");
      if (ppp.contains("dep_fuel_species")) {
        ppp.getarr("dep_fuel_species", spray_species);
      } else {
        ppp.getarr("fuel_species", spray_species);
      }
    }
#endif
    parm_in->m_h_parm.idx_Cp = get_var_index("CP", h_manf_data_in, required);
    parm_in->m_h_parm.idx_Cp_fuel =
      get_var_index("CP_fuel", h_manf_data_in, required);

    // Molecular weights and indices of chemical species if requested
    pp.query("has_species_mw", parm_in->m_h_parm.has_species_mw);
    if (parm_in->m_h_parm.has_species_mw) {
      const auto* mani_data = h_manf_data_in;
      // First find the total number of chemical species in the table
      amrex::Vector<int> spec_idx_vect;
      for (int n = 0; n < mani_data->Nvar; n++) {
        std::string nametmp = std::string(
          &(mani_data->varnames)[n * mani_data->len_str], mani_data->len_str);
        if (nametmp.substr(0, 2) == "Y-") {
          // A bit of a hack - if doing spray we only take the species involved
          // in spray phase change
#ifdef PELE_USE_SPRAY
          std::string find_spec =
            amrex::trim(nametmp.substr(2U, std::string::npos));
          if (
            std::find(spray_species.begin(), spray_species.end(), find_spec) !=
            spray_species.end()) {
            spec_idx_vect.emplace_back(n);
          }
#else
          spec_idx_vect.emplace_back(n);
#endif
        }
      }
      const int num_of_chemspecies = static_cast<int>(spec_idx_vect.size());
      parm_in->m_h_parm.num_chemspecies = num_of_chemspecies;

      // Dynamically allocate memory for chemSpecies_Mwt;
      constexpr std::streamsize real_size = sizeof(amrex::Real);
      parm_in->m_h_parm.chemSpecies_Mwt = static_cast<amrex::Real*>(
        amrex::The_Pinned_Arena()->alloc(num_of_chemspecies * real_size));
      constexpr std::streamsize int_size = sizeof(int);
      parm_in->m_h_parm.idx_chemspecies = static_cast<int*>(
        amrex::The_Pinned_Arena()->alloc(num_of_chemspecies * int_size));

      // Loop through table to find species names. Search these species names in
      // the metadata file and read molecular weights from the metadata file
      for (int i = 0; i < num_of_chemspecies; i++) {
        const int n = spec_idx_vect[i];
        parm_in->m_h_parm.idx_chemspecies[i] = n;
        std::string nametmp = std::string(
          &(mani_data->varnames)[n * mani_data->len_str], mani_data->len_str);
        std::string speciesname_search =
          amrex::trim(nametmp.substr(2U, std::string::npos)) + "_mw";
        pp.get(speciesname_search, parm_in->m_h_parm.chemSpecies_Mwt[i]);
      }
    }

    // Setup density lookups
    std::string density_lookup_type_string{"linear"};
    pp.query("density_lookup_type", density_lookup_type_string);
    if (density_lookup_type_string == "linear") {
      parm_in->m_h_parm.dens_lookup = eos::density_lookup_type::linear;
      parm_in->m_h_parm.idx_density = get_var_index("RHO", h_manf_data_in);
      if (verbose > 0) {
        amrex::Print()
          << "Manifold EOS: Using linear density lookups : index = "
          << parm_in->m_h_parm.idx_density << std::endl;
      }
    } else if (density_lookup_type_string == "log") {
      parm_in->m_h_parm.dens_lookup = eos::density_lookup_type::log;
      parm_in->m_h_parm.idx_density =
        get_var_index("lnRHO", h_manf_data_in, false);
      if (parm_in->m_h_parm.idx_density < 0) {
        parm_in->m_h_parm.idx_density = get_var_index("logRHO", h_manf_data_in);
      }
      if (verbose > 0) {
        amrex::Print()
          << "Manifold EOS: Using logarithmic density lookups : index = "
          << parm_in->m_h_parm.idx_density << std::endl;
      }
    } else if (density_lookup_type_string == "inverse") {
      parm_in->m_h_parm.dens_lookup = eos::density_lookup_type::inverse;
      parm_in->m_h_parm.idx_density = get_var_index("invRHO", h_manf_data_in);
      if (verbose > 0) {
        amrex::Print()
          << "Manifold EOS: Using inverse density lookups : index = "
          << parm_in->m_h_parm.idx_density << std::endl;
      }
    } else {
      amrex::Abort("Invalid density lookup type supplied");
    }

    // Get important indices
    parm_in->m_h_parm.idx_T = get_var_index("T", h_manf_data_in);
#ifdef PELE_USE_SPRAY
    parm_in->m_h_parm.idx_Wbar = get_var_index("WBAR", h_manf_data_in);
#else
    parm_in->m_h_parm.idx_Wbar = get_var_index("WBAR", h_manf_data_in, false);
#endif

    // For manifold table parameter source terms, assume if index not found,
    // source term is 0 For neural net, require a definition to be supplied for
    // each manifold parameter
    if (parm_in->m_h_parm.has_mani_src) {
      for (int idim = 0; idim < h_manf_data_in->Ndim; idim++) {
        const std::string dimname = std::string(
          &h_manf_data_in->dimnames[idim * h_manf_data_in->len_str],
          h_manf_data_in->len_str);
        const std::string dim_src =
          (dimname.rfind("Y-") == 0)
            ? "SRC_" + amrex::trim(dimname).substr(2U, std::string::npos)
            : "SRC_" + amrex::trim(dimname);
        parm_in->m_h_parm.idx_Wdot[idim] =
          get_var_index(dim_src.c_str(), h_manf_data_in, false);
        if ((parm_in->m_h_parm.idx_Wdot[idim] < 0) && verbose > 0) {
          amrex::Print() << "Manifold EOS: Warning: No source term found for "
                            "manifold parameter "
                         << amrex::trim(dimname) << ", assuming SRC_"
                         << amrex::trim(dimname) << " = 0" << std::endl;
        }
      }
    } else {
      for (int idim = 0; idim < h_manf_data_in->Ndim; idim++) {
        // The info file / metadata loader will ensure we have a definition for
        // each
        parm_in->m_h_parm.idx_Wdot[idim] = -1;
        amrex::Abort(
          "Computation of manifold src terms from species src terms "
          "needs to be reimplemented");
      }
    }

    // Variances - any manifold parameter ending in "VAR" is assumed to be
    // the variance of another parameter - e.g. ZMIXVAR would be the variance
    // of ZMIX
    for (int idim = 0; idim < h_manf_data_in->Ndim; idim++) {
      const std::string dimname = amrex::trim(
        std::string(
          &h_manf_data_in->dimnames[idim * h_manf_data_in->len_str],
          h_manf_data_in->len_str));
      size_t suffix_start = amrex::max<size_t>(dimname.length() - 3, size_t(0));
      if (dimname.substr(suffix_start) == "VAR") {
        const std::string variance_of = dimname.substr(0, suffix_start);
        int jdim;
        for (jdim = 0; jdim < h_manf_data_in->Ndim; jdim++) {
          const std::string test_dimname = amrex::trim(
            std::string(
              &h_manf_data_in->dimnames[jdim * h_manf_data_in->len_str],
              h_manf_data_in->len_str));
          if (variance_of == test_dimname) {
            parm_in->m_h_parm.is_variance_of[idim] = jdim;
            if (verbose > 0) {
              amrex::Print() << "Manifold EOS: using " << dimname
                             << " as variance of " << variance_of << std::endl;
            }
            break;
          }
        }
        if (jdim == h_manf_data_in->Ndim) {
          amrex::Abort(
            "Manifold EOS: variance " + dimname + " is used but " +
            variance_of + "does not exist");
        }
      } else {
        parm_in->m_h_parm.is_variance_of[idim] = -1;
      }
    }
    if (verbose > 0) {
      amrex::Print() << "Completed initialization of Manifold EOS\n";
    }
  }

  static void host_deallocate(PeleParams<eos::EosParm<eos::Manifold>>* parm_in)
  {
    parm_in->m_host_only_parm.manfunc_par->deallocate();
    parm_in->m_host_only_parm.manfunc_par = nullptr;
    if (parm_in->m_h_parm.has_species_mw) {
      amrex::The_Pinned_Arena()->free(parm_in->m_h_parm.chemSpecies_Mwt);
      amrex::The_Pinned_Arena()->free(parm_in->m_h_parm.idx_chemspecies);
    }
  }
};

} // namespace pele::physics
#endif
