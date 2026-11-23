#ifndef MLMG_PARAM_H
#define MLMG_PARAM_H

#include <AMReX_ParmParse.H>

namespace PeleRad {

struct MLMGParam
{
public:
  amrex::ParmParse pp_;

  int verbose_ = 0;
  int bottom_verbose_ = 0;
  int max_iter_ = 50;
  int max_fmg_iter_ = 0;
  int max_bottom_iter_ = 20;
  amrex::Real reltol_ = 1.e-4;
  amrex::Real abstol_ = 1.e-4;
  amrex::Real bottom_reltol_ = 1.e-6;
  amrex::Real bottom_abstol_ = 1.e-6;
  int linop_maxorder_ = 3;
  int max_coarsening_level_ = 20;
  int agg_grid_size_ = -1;
  int con_grid_size_ = -1;
  int agglomeration_ = 0;
  int consolidation_ = 1;
  int composite_solve_ = 1;
  int fine_level_solve_only_ = 0;
  bool use_hypre_ = false;
  amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> lobc_;
  amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> hibc_;
  int ebbc_type_ = 2;
  std::string kppath_;

  AMREX_GPU_HOST
  MLMGParam() = default;

  AMREX_GPU_HOST
  MLMGParam(const amrex::ParmParse pp) : pp_(std::move(pp))
  {
    pp_.query("kppath", kppath_);
    pp_.query("verbose", verbose_);
    pp_.query("bottom_verbose", bottom_verbose_);
    pp_.query("max_iter", max_iter_);
    pp_.query("max_fmg_iter", max_fmg_iter_);
    pp_.query("max_bottom_iter", max_bottom_iter_);
    pp_.query("reltol", reltol_);
    pp_.query("abstol", abstol_);
    pp_.query("bottom_reltol", bottom_reltol_);
    pp_.query("bottom_abstol", bottom_abstol_);
    pp_.query("linop_maxorder", linop_maxorder_);
    pp_.query("max_coarsening_level", max_coarsening_level_);
    pp_.query("agg_grid_size", agg_grid_size_);
    pp_.query("con_grid_size", con_grid_size_);
    pp_.query("agglomeration", agglomeration_);
    pp_.query("consolidation", consolidation_);
    pp_.query("composite_solve", composite_solve_);
    pp_.query("fine_level_solve_only", fine_level_solve_only_);
    pp_.query("use_hypre", use_hypre_);

    amrex::Vector<std::string> lo_bc_char_(AMREX_SPACEDIM);
    amrex::Vector<std::string> hi_bc_char_(AMREX_SPACEDIM);
    pp_.getarr("lo_bc", lo_bc_char_, 0, AMREX_SPACEDIM);
    pp_.getarr("hi_bc", hi_bc_char_, 0, AMREX_SPACEDIM);

    std::unordered_map<std::string, amrex::LinOpBCType> bctype;
    bctype["Dirichlet"] = amrex::LinOpBCType::Dirichlet;
    bctype["Periodic"] = amrex::LinOpBCType::Periodic;
    bctype["Neumann"] = amrex::LinOpBCType::Neumann;
    bctype["Robin"] = amrex::LinOpBCType::Robin;

    pp_.query("ebbc_type", ebbc_type_);

    amrex::Print() << "Define radiation BC:"
                   << "\n";

    auto const toBC = [&bctype](std::string const& name) {
      auto const it = bctype.find(name);
      if (it == bctype.end()) {
        amrex::Abort(
          "PeleRad: unknown pelerad.lo_bc/hi_bc type " + name +
          ", must be Dirichlet, Periodic, Neumann or Robin");
      }
      return it->second;
    };

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
      lobc_[idim] = toBC(lo_bc_char_[idim]);
      hibc_[idim] = toBC(hi_bc_char_[idim]);

      amrex::Print() << "lobc[" << idim << "]=" << lobc_[idim] << ", ";
      amrex::Print() << "hibc[" << idim << "]=" << hibc_[idim] << "\n";
    }
  }
};

} // namespace PeleRad

#endif
