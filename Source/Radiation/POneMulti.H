#ifndef PONEMULTI_H
#define PONEMULTI_H

#include <AMRParam.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <MLMGParam.H>

namespace PeleRad {

class POneMulti
{
private:
  MLMGParam mlmgpp_;

  amrex::LPInfo info_;

public:
  amrex::Vector<amrex::Geometry>& geom_;
  amrex::Vector<amrex::BoxArray>& grids_;
  amrex::Vector<amrex::DistributionMapping>& dmap_;

  amrex::Vector<amrex::MultiFab>& solution_;
  amrex::Vector<amrex::MultiFab> const& rhs_;
  amrex::Vector<amrex::MultiFab> const& acoef_;
  amrex::Vector<amrex::MultiFab> const& bcoef_;

  amrex::Vector<amrex::MultiFab> const& robin_a_;
  amrex::Vector<amrex::MultiFab> const& robin_b_;
  amrex::Vector<amrex::MultiFab> const& robin_f_;

  amrex::Real const ascalar_ = 1.0;
  amrex::Real const bscalar_ = 1.0 / 3.0;

  POneMulti() = delete;

  // constructor
  POneMulti(
    MLMGParam const mlmgpp,
    amrex::Vector<amrex::Geometry>& geom,
    amrex::Vector<amrex::BoxArray>& grids,
    amrex::Vector<amrex::DistributionMapping>& dmap,
    amrex::Vector<amrex::MultiFab>& solution,
    amrex::Vector<amrex::MultiFab> const& rhs,
    amrex::Vector<amrex::MultiFab> const& acoef,
    amrex::Vector<amrex::MultiFab> const& bcoef,
    amrex::Vector<amrex::MultiFab> const& robin_a,
    amrex::Vector<amrex::MultiFab> const& robin_b,
    amrex::Vector<amrex::MultiFab> const& robin_f)
    : mlmgpp_(std::move(mlmgpp)),
      geom_(geom),
      grids_(grids),
      dmap_(dmap),
      solution_(solution),
      rhs_(rhs),
      acoef_(acoef),
      bcoef_(bcoef),
      robin_a_(robin_a),
      robin_b_(robin_b),
      robin_f_(robin_f)
  {
    auto const max_coarsening_level = mlmgpp_.max_coarsening_level_;
    auto const agglomeration = mlmgpp_.agglomeration_;
    auto const consolidation = mlmgpp_.consolidation_;
    info_.setAgglomeration(agglomeration != 0);
    info_.setConsolidation(consolidation != 0);
    info_.setMaxCoarseningLevel(max_coarsening_level);
  }

  void solve()
  {
    auto const linop_maxorder = mlmgpp_.linop_maxorder_;

    auto const& lobc = mlmgpp_.lobc_;
    auto const& hibc = mlmgpp_.hibc_;

    amrex::MLABecLaplacian mlabec(geom_, grids_, dmap_, info_);
    // geom_ may hold levels with no grids yet (finest_level < max_level);
    // the operator keeps only the levels that have grids.
    int const nlevels = mlabec.NAMRLevels();

    mlabec.setDomainBC(lobc, hibc);
    mlabec.setScalars(ascalar_, bscalar_);
    mlabec.setMaxOrder(linop_maxorder);

    auto const max_iter = mlmgpp_.max_iter_;
    auto const max_fmg_iter = mlmgpp_.max_fmg_iter_;
    auto const verbose = mlmgpp_.verbose_;
    auto const bottom_verbose = mlmgpp_.bottom_verbose_;
    auto const use_hypre = mlmgpp_.use_hypre_;
    auto const bottom_reltol = mlmgpp_.bottom_reltol_;
    auto const bottom_abstol = mlmgpp_.bottom_abstol_;

    amrex::MLMG mlmg(mlabec);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    mlmg.setBottomSolver(amrex::BottomSolver::bicgstab);
    if (use_hypre) {
      mlmg.setBottomSolver(amrex::MLMG::BottomSolver::hypre);
    }
    mlmg.setBottomTolerance(bottom_reltol);
    mlmg.setBottomToleranceAbs(bottom_abstol);

    for (int ilev = 0; ilev < nlevels; ++ilev) {
      auto const& geom = geom_[ilev];
      auto& solution = solution_[ilev];
      auto const& acoef = acoef_[ilev];
      auto const& bcoef = bcoef_[ilev];
      auto const& robin_a = robin_a_[ilev];
      auto const& robin_b = robin_b_[ilev];
      auto const& robin_f = robin_f_[ilev];

      mlabec.setLevelBC(ilev, &solution, &robin_a, &robin_b, &robin_f);

      mlabec.setACoeffs(ilev, acoef);

      amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> face_bcoef;
      for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        amrex::BoxArray const& ba = amrex::convert(
          bcoef.boxArray(), amrex::IntVect::TheDimensionVector(idim));
        face_bcoef[idim].define(ba, bcoef.DistributionMap(), 1, 0);
      }
      amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoef), bcoef, geom);
      mlabec.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoef));
    }

    auto const tol_rel = mlmgpp_.reltol_;
    auto const tol_abs = mlmgpp_.abstol_;

    mlmg.solve(
      GetVecOfPtrs(solution_), GetVecOfConstPtrs(rhs_), tol_rel, tol_abs);
  }
};

} // namespace PeleRad

#endif
