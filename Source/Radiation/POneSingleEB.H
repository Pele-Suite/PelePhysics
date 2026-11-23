#ifndef PONESINGLEEB_H
#define PONESINGLEEB_H

#include <AMRParam.H>
#include <AMReX_EB2.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MLEBABecLap.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <MLMGParam.H>

namespace PeleRad {

class POneSingleEB
{
private:
  MLMGParam mlmgpp_;

public:
  amrex::Geometry const& geom_;
  amrex::BoxArray const& grids_;
  amrex::DistributionMapping const& dmap_;

  std::unique_ptr<amrex::EBFArrayBoxFactory> const& factory_;

  amrex::MultiFab& solution_;
  amrex::MultiFab const& rhs_;
  amrex::MultiFab const& acoef_;
  amrex::MultiFab const& bcoef_;

  amrex::MultiFab const& robin_a_;
  amrex::MultiFab const& robin_b_;
  amrex::MultiFab const& robin_f_;

  amrex::Real const ascalar = 1.0;
  amrex::Real const bscalar = 1.0 / 3.0;

  // constructor
  POneSingleEB(
    MLMGParam const& mlmgpp,
    amrex::Geometry const& geom,
    amrex::BoxArray const& grids,
    amrex::DistributionMapping const& dmap,
    std::unique_ptr<amrex::EBFArrayBoxFactory> const& factory,
    amrex::MultiFab& solution,
    amrex::MultiFab const& rhs,
    amrex::MultiFab const& acoef,
    amrex::MultiFab const& bcoef,
    amrex::MultiFab const& robin_a,
    amrex::MultiFab const& robin_b,
    amrex::MultiFab const& robin_f)
    : mlmgpp_(mlmgpp),
      geom_(geom),
      grids_(grids),
      dmap_(dmap),
      factory_(factory),
      solution_(solution),
      rhs_(rhs),
      acoef_(acoef),
      bcoef_(bcoef),
      robin_a_(robin_a),
      robin_b_(robin_b),
      robin_f_(robin_f) {};

  void solve()
  {
    auto const max_coarsening_level = mlmgpp_.max_coarsening_level_;
    auto const max_iter = mlmgpp_.max_iter_;
    auto const max_fmg_iter = mlmgpp_.max_fmg_iter_;
    auto const verbose = mlmgpp_.verbose_;
    auto const bottom_verbose = mlmgpp_.bottom_verbose_;
    auto const tol_rel = mlmgpp_.reltol_;
    auto const tol_abs = mlmgpp_.abstol_;
    auto const use_hypre = mlmgpp_.use_hypre_;

    auto const& lobc = mlmgpp_.lobc_;
    auto const& hibc = mlmgpp_.hibc_;

    auto const& geom = geom_;
    auto const& grids = grids_;
    auto const& dmap = dmap_;
    auto const& factory = factory_;

    auto& solution = solution_;
    auto const& rhs = rhs_;
    auto const& acoef = acoef_;
    auto const& bcoef = bcoef_;
    auto const& robin_a = robin_a_;
    auto const& robin_b = robin_b_;
    auto const& robin_f = robin_f_;

    amrex::MLEBABecLap mlabec(
      {geom}, {grids}, {dmap},
      amrex::LPInfo().setMaxCoarseningLevel(max_coarsening_level),
      {factory.get()});

    mlabec.setDomainBC(lobc, hibc);

    mlabec.setLevelBC(0, &solution, &robin_a, &robin_b, &robin_f);

    mlabec.setScalars(ascalar, bscalar);

    mlabec.setACoeffs(0, acoef);

    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> face_bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      amrex::BoxArray const& ba = amrex::convert(
        bcoef.boxArray(), amrex::IntVect::TheDimensionVector(idim));
      face_bcoef[idim].define(ba, bcoef.DistributionMap(), 1, 0);
    }

    amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoef), bcoef, geom);

    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));

    if (mlmgpp_.ebbc_type_ == 1) {
      mlabec.setEBDirichlet(0, solution, 1.0);
    }

    amrex::MLMG mlmg(mlabec);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);

    if (use_hypre)
      mlmg.setBottomSolver(amrex::MLMG::BottomSolver::hypre);

    mlmg.solve({&solution}, {&rhs}, tol_rel, tol_abs);
  }

  void calcRadSource(amrex::MultiFab& rad_src)
  {
    for (amrex::MFIter mfi(rad_src); mfi.isValid(); ++mfi) {
      amrex::Box const& bx = mfi.validbox();

      auto const& rhsfab = rhs_.array(mfi);
      auto const& solfab = solution_.array(mfi);
      auto const& acfab = acoef_.array(mfi);

      auto radfab = rad_src.array(mfi);

      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          radfab(i, j, k, 4) =
            acfab(i, j, k) * solfab(i, j, k) - rhsfab(i, j, k);
        });
    }
  }
};

} // namespace PeleRad

#endif
