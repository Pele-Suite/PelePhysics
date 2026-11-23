#ifndef RADIATION_H
#define RADIATION_H

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <Constants.H>

#ifdef AMREX_USE_EB
#include <POneMultiEB.H>
#else
#include <POneMulti.H>
#endif

#include <PlanckMean.H>
#include <SpectralModels.H>

namespace PeleRad {

class Radiation
{
private:
  PlanckMean radprop;

  amrex::Vector<amrex::Geometry>& geom_;
  amrex::Vector<amrex::BoxArray>& grids_;
  amrex::Vector<amrex::DistributionMapping>& dmap_;

  RadComps rc_;

  MLMGParam mlmgpp_;

  amrex::Vector<amrex::MultiFab> solution_;
  amrex::Vector<amrex::MultiFab> rhs_;
  amrex::Vector<amrex::MultiFab> acoef_;
  amrex::Vector<amrex::MultiFab> bcoef_;
  amrex::Vector<amrex::MultiFab> robin_a_;
  amrex::Vector<amrex::MultiFab> robin_b_;
  amrex::Vector<amrex::MultiFab> robin_f_;

  amrex::Vector<amrex::MultiFab> absc_;

  //    bool composite_solve_;

#ifdef AMREX_USE_EB
  std::unique_ptr<POneMultiEB> rte_;
#else
  std::unique_ptr<POneMulti> rte_;
#endif

  //    std::unique_ptr<POneMultiLevbyLev> rtelevbylev_;

#ifdef AMREX_USE_EB
  amrex::Vector<amrex::EBFArrayBoxFactory const*> ebfactVec_;
#endif

  // Size the level data to the levels that have grids (the level vectors
  // from AmrMesh have max_level + 1 entries); a regrid that removes a level
  // frees its data.
  int resizeLevels()
  {
    int const nlev = numLevelsWithGrids(grids_);
    for (auto* mfs :
         {&solution_, &rhs_, &acoef_, &bcoef_, &robin_a_, &robin_b_, &robin_f_,
          &absc_}) {
      mfs->resize(nlev);
    }
    return nlev;
  }

  // Define and initialise one level's data; all of it is set before the first
  // solve, since plot_init_state writes it.
  void defineLevel(int ilev, amrex::FabFactory<amrex::FArrayBox> const& fact)
  {
    auto const& ba = grids_[ilev];
    auto const& dm = dmap_[ilev];
    amrex::IntVect const ng{1};
    amrex::MFInfo const info;
    solution_[ilev].define(ba, dm, 1, ng, info, fact);
    rhs_[ilev].define(ba, dm, 1, 0, info, fact);
    acoef_[ilev].define(ba, dm, 1, 0, info, fact);
    bcoef_[ilev].define(ba, dm, 1, ng, info, fact);
    robin_a_[ilev].define(ba, dm, 1, ng, info, fact);
    robin_b_[ilev].define(ba, dm, 1, ng, info, fact);
    robin_f_[ilev].define(ba, dm, 1, ng, info, fact);
    absc_[ilev].define(ba, dm, 1, 0, info, fact);

    for (auto* mf :
         {&solution_[ilev], &rhs_[ilev], &acoef_[ilev], &robin_a_[ilev],
          &robin_b_[ilev], &robin_f_[ilev], &absc_[ilev]}) {
      mf->setVal(0.0, 0, 1, mf->nGrowVect());
    }
    bcoef_[ilev].setVal(1.0, 0, 1, ng);
  }

public:
  AMREX_GPU_HOST
  Radiation(
    amrex::Vector<amrex::Geometry>& geom,
    amrex::Vector<amrex::BoxArray>& grids,
    amrex::Vector<amrex::DistributionMapping>& dmap,
    RadComps rc,
    amrex::ParmParse const& mlmgpp
#ifdef AMREX_USE_EB
    ,
    amrex::Vector<std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>>> const&
      factory
#endif
    )
    : geom_(geom), grids_(grids), dmap_(dmap), rc_(rc), mlmgpp_(mlmgpp)
  {
    if (amrex::ParallelDescriptor::IOProcessor()) {
      rc_.checkIndices();
    }

#ifdef AMREX_USE_EB
    initVars(grids, dmap, factory);
#else
    initVars(grids, dmap);
#endif

    loadSpecModel();

    //        composite_solve_ = mlmgpp_.composite_solve_;

    //        if (composite_solve_)
    //        {

#ifdef AMREX_USE_EB
    rte_ = std::make_unique<POneMultiEB>(
      mlmgpp_, geom_, grids_, dmap_, ebfactVec_, solution_, rhs_, acoef_,
      bcoef_, robin_a_, robin_b_, robin_f_);
#else
    rte_ = std::make_unique<POneMulti>(
      mlmgpp_, geom_, grids_, dmap_, solution_, rhs_, acoef_, bcoef_, robin_a_,
      robin_b_, robin_f_);
#endif
    //        }
    //        else
    //        {
    // level by level option is not ready
    //            rtelevbylev_ =
    //            std::make_unique<POneMultiLevbyLev>(mlmgpp_, 2,
    //                geom_, grids_, dmap_, solution_, rhs_, acoef_, bcoef_,
    //                robin_a_, robin_b_, robin_f_);
    //        }
  }

  AMREX_GPU_HOST
  void loadSpecModel()
  {
    // path to spectral database
    auto data_path = mlmgpp_.kppath_;

    radprop.load(data_path);

    amrex::Print() << "The radiative property database is loaded."
                   << "\n"
                   << "kp path: " << data_path << "\n";
  }

  // Xco2, Xh2o, Xco: mole fractions; T in K; P in Pa; fv: soot volume
  // fraction.
  void updateSpecProp(
    amrex::MFIter const& mfi,
    amrex::Array4<const amrex::Real> const& Xco2,
    amrex::Array4<const amrex::Real> const& Xh2o,
    amrex::Array4<const amrex::Real> const& Xco,
    amrex::Array4<const amrex::Real> const& T,
    amrex::Array4<const amrex::Real> const& P
#ifdef PELELM_USE_SOOT
    ,
    amrex::Array4<const amrex::Real> const& fv
#endif
    ,
    int ilev)
  {
    //  amrex::Print() << "update radiative properties \n";

    auto const& kpco2 = radprop.kpco2();
    auto const& kph2o = radprop.kph2o();
    auto const& kpco = radprop.kpco();

    // The caller may tile: work on this tile only. growntilebox(1) grows the
    // tile only across its valid box faces, so the boundary cells below read
    // kappa from cells of the same tile.
    amrex::Box const bx = mfi.tilebox();
    amrex::Box const gbx = mfi.growntilebox(1);

    auto const dlo = amrex::lbound(geom_[ilev].Domain());
    auto const dhi = amrex::ubound(geom_[ilev].Domain());

    auto const& kappa = absc_[ilev].array(mfi);
    auto const& rhsfab = rhs_[ilev].array(mfi);
    auto const& alphafab = acoef_[ilev].array(mfi);
    auto const& betafab = bcoef_[ilev].array(mfi);
    auto const& robin_a_fab = robin_a_[ilev].array(mfi);
    auto const& robin_b_fab = robin_b_[ilev].array(mfi);
    auto const& robin_f_fab = robin_f_[ilev].array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      RadProp::getRadPropGas(
        i, j, k, Xco2, Xh2o, Xco, T, P, kappa, kpco2, kph2o, kpco);
    });

#ifdef PELELM_USE_SOOT
    auto const& kpsoot = radprop.kpsoot();
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      RadProp::getRadPropSoot(i, j, k, fv, T, kappa, kpsoot);
    });
#endif

    auto const blo = amrex::lbound(bx);
    auto const bhi = amrex::ubound(bx);
    amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // The nearest cell of this tile: the cell itself, or for a ghost cell
      // the adjacent valid cell (growntilebox grows only across valid-box
      // faces), which on the domain boundary is the boundary cell.
      amrex::Real const kappa_m = // in 1/m
        kappa(
          amrex::Clamp(i, blo.x, bhi.x), amrex::Clamp(j, blo.y, bhi.y),
          amrex::Clamp(k, blo.z, bhi.z)) *
        100;

      if (bx.contains(i, j, k)) {
        amrex::Real const ka = amrex::max<amrex::Real>(0.01, kappa_m);
        betafab(i, j, k) = 1.0 / ka;
        rhsfab(i, j, k) = 4.0 * ka * 5.67e-8 * std::pow(T(i, j, k), 4.0); // SI
        alphafab(i, j, k) = ka;
        return;
      }

      // Ghost cells outside the domain in exactly one direction carry the
      // Marshak (Robin) condition. The others start from the adjacent cell's
      // value: FillBoundary overwrites those next to other boxes and periodic
      // images, and coarse/fine ghosts keep it.
      int const nout = static_cast<int>(i < dlo.x || i > dhi.x) +
                       static_cast<int>(j < dlo.y || j > dhi.y) +
                       static_cast<int>(k < dlo.z || k > dhi.z);
      if (nout == 1) {
        betafab(i, j, k) = 1.0 / amrex::max<amrex::Real>(1.0, kappa_m);
        robin_a_fab(i, j, k) = -1.0 / betafab(i, j, k);
        robin_b_fab(i, j, k) = -2.0 / 3.0;
        robin_f_fab(i, j, k) = 0.0;
      } else {
        betafab(i, j, k) = 1.0 / amrex::max<amrex::Real>(0.01, kappa_m);
      }
    });
  }

  void initVars(
    amrex::Vector<amrex::BoxArray> const& grids,
    amrex::Vector<amrex::DistributionMapping> const& dmap)
  {
    grids_ = grids;
    dmap_ = dmap;
    int const nlev = resizeLevels();
    for (int ilev = 0; ilev < nlev; ++ilev) {
      defineLevel(ilev, amrex::FArrayBoxFactory());
    }
  }

#ifdef AMREX_USE_EB
  void initVars(
    amrex::Vector<amrex::BoxArray> const& grids,
    amrex::Vector<amrex::DistributionMapping> const& dmap,
    amrex::Vector<std::unique_ptr<amrex::FabFactory<amrex::FArrayBox>>> const&
      factory)
  {
    grids_ = grids;
    dmap_ = dmap;
    int const nlev = resizeLevels();
    // Rebuilt on every call: after a regrid the old factories are gone.
    ebfactVec_.resize(nlev);
    for (int ilev = 0; ilev < nlev; ++ilev) {
      ebfactVec_[ilev] =
        &(static_cast<amrex::EBFArrayBoxFactory const&>(*factory[ilev]));
      defineLevel(ilev, *factory[ilev]);
    }
  }
#endif

  void evaluateRad()
  {
    // std::cout << "begin of evaluateRad() \n";

    int const nlev = numLevelsWithGrids(grids_);
    for (int ilev = 0; ilev < nlev; ++ilev) {
      bcoef_[ilev].FillBoundary(geom_[ilev].periodicity());
      // The solution's ghost cells carry the Dirichlet boundary values (zero
      // incident radiation), but the previous solve overwrote them with
      // extrapolated values: reset them.
      solution_[ilev].setBndry(0.0);
    }
    //        if (composite_solve_)
    rte_->solve();
    //        else
    //            rtelevbylev_->solve();
  }

  void calcRadSource(
    amrex::MFIter const& mfi,
    amrex::Array4<amrex::Real> const& radfab,
    int ilev)
  {
    amrex::Box const bx = mfi.tilebox();
    auto const& rhsfab = rhs_[ilev].array(mfi);
    auto const& solfab = solution_[ilev].array(mfi);
    auto const& acfab = acoef_[ilev].array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      radfab(i, j, k) += acfab(i, j, k) * solfab(i, j, k) - rhsfab(i, j, k);
    });
  }

  RadComps readRadIndices() const { return rc_; }

  amrex::Vector<amrex::MultiFab> const& G() { return solution_; }

  amrex::Vector<amrex::MultiFab> const& kappa() { return acoef_; }

  amrex::Vector<amrex::MultiFab> const& emis() { return rhs_; }

  amrex::Vector<amrex::BoxArray> const& grids() { return grids_; }
};

} // namespace PeleRad

#endif
