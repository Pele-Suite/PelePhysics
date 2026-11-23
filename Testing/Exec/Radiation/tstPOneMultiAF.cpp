#include <PlanckMean.H>
#include <SpectralModels.H>
#include <AMReX.H>
#include <AMReX_PlotFileUtil.H>
#include <POneMulti.H>
#include <POneMultiLevbyLev.H>

#include "POneCheck.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
initGasField(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& y_co2,
  amrex::Array4<amrex::Real> const& y_h2o,
  amrex::Array4<amrex::Real> const& y_co,
  amrex::Array4<amrex::Real> const& fv_soot,
  amrex::Array4<amrex::Real> const& temp,
  amrex::Array4<amrex::Real> const& pressure,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& phi)
{
  amrex::Real coef = 100.0;
  amrex::Real xc = (phi[0] + plo[0]) * 0.5;
  amrex::Real yc = (phi[1] + plo[1]) * 0.5;

  amrex::Real x = plo[0] + (i + 0.5) * dx[0];
  amrex::Real y = plo[1] + (j + 0.5) * dx[1];
#if (AMREX_SPACEDIM == 3)
  amrex::Real z = plo[2] + (k + 0.5) * dx[2];
  z /= coef;
#else
  // A cross-section of the 3D flame near its peak temperature
  amrex::Real z = 0.3;
#endif

  amrex::Real r = std::sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc));

  r /= coef;

  amrex::Real expr = std::exp(
    -(4.0 * r / (0.05 + 0.1 * 4.0 * z)) * (4.0 * r / (0.05 + 0.1 * 4.0 * z)));
  amrex::Real expTz = std::exp(
    -((4.0 * z - 1.3) / (0.7 + 0.5 * 4.0 * z)) *
    ((4.0 * z - 1.3) / (0.7 + 0.5 * 4.0 * z)));

  temp(i, j, k) = 300.0 + 1700.0 * expr * expTz;

  pressure(i, j, k) = 1.0e5; // Pa

  amrex::Real expSoot =
    std::exp(-((4.0 * z - 1.0) / 0.7) * ((4.0 * z - 1.0) / 0.7));

  fv_soot(i, j, k) = 1e-6 * expr * expSoot;

  amrex::Real expCO2z = std::exp(
    -((4.0 * z - 1.1) / (0.6 + 0.5 * 4.0 * z)) *
    ((4.0 * z - 1.1) / (0.6 + 0.5 * 4.0 * z)));

  y_co2(i, j, k) = 0.1 * expr * expCO2z;

  amrex::Real expH2Oz = std::exp(
    -((4.0 * z - 1.0) / (0.7 + 0.5 * 4.0 * z)) *
    ((4.0 * z - 1.0) / (0.7 + 0.5 * 4.0 * z)));

  y_h2o(i, j, k) = 0.2 * expr * expH2Oz;

  amrex::Real expCOz =
    std::exp(-((4.0 * z - 1.0) / 0.7) * ((4.0 * z - 1.0) / 0.7));

  y_co(i, j, k) = 0.09 * expr * expCOz;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
actual_init_coefs(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& rhs,
  amrex::Array4<amrex::Real> const& alpha,
  amrex::Array4<amrex::Real> const& beta,
  amrex::Dim3 const& dlo,
  amrex::Dim3 const& dhi,
  amrex::Box const& vbx,
  amrex::Array4<amrex::Real> const& absc,
  amrex::Array4<amrex::Real> const& T)
{
  beta(i, j, k) = 1.0;
  if (vbx.contains(i, j, k)) {
    amrex::Real ka = amrex::max<amrex::Real>(0.01, absc(i, j, k) * 100);
    beta(i, j, k) = 1.0 / ka;
    rhs(i, j, k) = 4.0 * ka * 5.67e-8 * std::pow(T(i, j, k), 4.0); // SI
    alpha(i, j, k) = ka;
  }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
actual_init_bc_coefs(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& alpha,
  amrex::Array4<amrex::Real> const& beta,
  amrex::Array4<amrex::Real> const& robin_a,
  amrex::Array4<amrex::Real> const& robin_b,
  amrex::Array4<amrex::Real> const& robin_f,
  amrex::Dim3 const& dlo,
  amrex::Dim3 const& dhi,
  amrex::Array4<amrex::Real> const& T)
{
  // Robin BC
  bool robin_cell = false;
  if (j >= dlo.y && j <= dhi.y && k >= dlo.z && k <= dhi.z) {
    if (i > dhi.x || i < dlo.x) {
      robin_cell = true;
    }
  } else if (i >= dlo.x && i <= dhi.x && k >= dlo.z && k <= dhi.z) {
    if (j > dhi.y || j < dlo.y) {
      robin_cell = true;
    }
  }

  if (robin_cell) {
    robin_a(i, j, k) = -1.0 / beta(i, j, k);
    robin_b(i, j, k) = -2.0 / 3.0;
    robin_f(i, j, k) = 0.0;
  }
}

void
initProbABecLaplacian(
  amrex::Vector<amrex::Geometry>& geom,
  amrex::Vector<amrex::MultiFab>& solution,
  amrex::Vector<amrex::MultiFab>& rhs,
  amrex::Vector<amrex::MultiFab>& acoef,
  amrex::Vector<amrex::MultiFab>& bcoef,
  amrex::Vector<amrex::MultiFab>& robin_a,
  amrex::Vector<amrex::MultiFab>& robin_b,
  amrex::Vector<amrex::MultiFab>& robin_f,
  amrex::Vector<amrex::MultiFab>& y_co2,
  amrex::Vector<amrex::MultiFab>& y_h2o,
  amrex::Vector<amrex::MultiFab>& y_co,
  amrex::Vector<amrex::MultiFab>& soot_fv_rad,
  amrex::Vector<amrex::MultiFab>& temperature,
  amrex::Vector<amrex::MultiFab>& pressure,
  amrex::Vector<amrex::MultiFab>& absc)
{
  using PeleRad::PlanckMean;
  using PeleRad::RadProp::getRadPropGas;
  using PeleRad::RadProp::getRadPropSoot;

  std::string data_path("kpDB/");
  PlanckMean radprop(data_path);
  auto const& kpco2 = radprop.kpco2();
  auto const& kph2o = radprop.kph2o();
  auto const& kpco = radprop.kpco();
  auto const& kpsoot = radprop.kpsoot();

  auto nlevels = geom.size();

  for (int ilev = 0; ilev < nlevels; ++ilev) {
    auto const prob_lo = geom[ilev].ProbLoArray();
    auto const prob_hi = geom[ilev].ProbHiArray();
    auto const dx = geom[ilev].CellSizeArray();
    auto const dlo = amrex::lbound(geom[ilev].Domain());
    auto const dhi = amrex::ubound(geom[ilev].Domain());

    for (amrex::MFIter mfi(rhs[ilev]); mfi.isValid(); ++mfi) {
      amrex::Box const& bx = mfi.validbox();
      amrex::Box const& gbx = amrex::grow(bx, 1);
      auto const& Yco2 = y_co2[ilev].array(mfi);
      auto const& Yh2o = y_h2o[ilev].array(mfi);
      auto const& Yco = y_co[ilev].array(mfi);
      auto const& fv = soot_fv_rad[ilev].array(mfi);
      auto const& T = temperature[ilev].array(mfi);
      auto const& P = pressure[ilev].array(mfi);
      auto const& kappa = absc[ilev].array(mfi);

      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          initGasField(
            i, j, k, Yco2, Yh2o, Yco, fv, T, P, dx, prob_lo, prob_hi);
          getRadPropGas(
            i, j, k, Yco2, Yh2o, Yco, T, P, kappa, kpco2, kph2o, kpco);
        });

      // if soot exists
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          getRadPropSoot(i, j, k, fv, T, kappa, kpsoot);
        });

      auto const& rhsfab = rhs[ilev].array(mfi);
      auto const& acfab = acoef[ilev].array(mfi);
      auto const& bcfab = bcoef[ilev].array(mfi);
      auto const& rafab = robin_a[ilev].array(mfi);
      auto const& rbfab = robin_b[ilev].array(mfi);
      auto const& rffab = robin_f[ilev].array(mfi);
      amrex::ParallelFor(
        gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          actual_init_coefs(
            i, j, k, rhsfab, acfab, bcfab, dlo, dhi, bx, kappa, T);
        });

      amrex::ParallelFor(
        gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          actual_init_bc_coefs(
            i, j, k, acfab, bcfab, rafab, rbfab, rffab, dlo, dhi, T);
        });
    }
    solution[ilev].setVal(0.0, 0, 1, amrex::IntVect(0));
  }
}

void
initMeshandData(
  PeleRad::AMRParam const& amrpp,
  amrex::Vector<amrex::Geometry>& geom,
  amrex::Vector<amrex::BoxArray>& grids,
  amrex::Vector<amrex::DistributionMapping>& dmap,
  amrex::Vector<amrex::MultiFab>& solution,
  amrex::Vector<amrex::MultiFab>& rhs,
  amrex::Vector<amrex::MultiFab>& acoef,
  amrex::Vector<amrex::MultiFab>& bcoef,
  amrex::Vector<amrex::MultiFab>& robin_a,
  amrex::Vector<amrex::MultiFab>& robin_b,
  amrex::Vector<amrex::MultiFab>& robin_f,
  amrex::Vector<amrex::MultiFab>& y_co2,
  amrex::Vector<amrex::MultiFab>& y_h2o,
  amrex::Vector<amrex::MultiFab>& y_co,
  amrex::Vector<amrex::MultiFab>& soot_fv_rad,
  amrex::Vector<amrex::MultiFab>& temperature,
  amrex::Vector<amrex::MultiFab>& pressure,
  amrex::Vector<amrex::MultiFab>& absc)
{
  int const nlevels = amrpp.max_level_ + 1;
  int const ref_ratio = amrpp.ref_ratio_;
  int const n_cell = amrpp.n_cell_;
  int const max_grid_size = amrpp.max_grid_size_;

  // initialize mesh
  geom.resize(nlevels);
  grids.resize(nlevels);
  dmap.resize(nlevels);

  amrex::RealBox rb(
    {AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(12.5, 12.5, 75)});
  amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
  amrex::Geometry::Setup(&rb, 0, is_periodic.data());

  std::vector<int> npts{n_cell, n_cell, 6 * n_cell};
  amrex::Box domain0(
    amrex::IntVect{AMREX_D_DECL(0, 0, 0)},
    amrex::IntVect{AMREX_D_DECL(npts[0] - 1, npts[1] - 1, npts[2] - 1)});

  amrex::Box domain = domain0;

  for (int ilev = 0; ilev < nlevels; ++ilev) {
    geom[ilev].define(domain);
    domain.refine(ref_ratio);
  }

  domain = domain0;

  for (int ilev = 0; ilev < nlevels; ++ilev) {
    grids[ilev].define(domain);
    grids[ilev].maxSize(max_grid_size);
    domain.refine(ref_ratio);
  }

  // initialize variables
  solution.resize(nlevels);
  rhs.resize(nlevels);
  acoef.resize(nlevels);
  bcoef.resize(nlevels);
  robin_a.resize(nlevels);
  robin_b.resize(nlevels);
  robin_f.resize(nlevels);

  y_co2.resize(nlevels);
  y_h2o.resize(nlevels);
  y_co.resize(nlevels);
  soot_fv_rad.resize(nlevels);
  temperature.resize(nlevels);
  pressure.resize(nlevels);
  absc.resize(nlevels);

  amrex::IntVect ng = amrex::IntVect{1};

  for (int ilev = 0; ilev < nlevels; ++ilev) {
    dmap[ilev].define(grids[ilev]);
    solution[ilev].define(grids[ilev], dmap[ilev], 1, ng);
    rhs[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    acoef[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    bcoef[ilev].define(grids[ilev], dmap[ilev], 1, ng);
    robin_a[ilev].define(grids[ilev], dmap[ilev], 1, ng);
    robin_b[ilev].define(grids[ilev], dmap[ilev], 1, ng);
    robin_f[ilev].define(grids[ilev], dmap[ilev], 1, ng);

    y_co2[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    y_h2o[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    y_co[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    soot_fv_rad[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    temperature[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    pressure[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    absc[ilev].define(grids[ilev], dmap[ilev], 1, 0);
  }

  initProbABecLaplacian(
    geom, solution, rhs, acoef, bcoef, robin_a, robin_b, robin_f, y_co2, y_h2o,
    y_co, soot_fv_rad, temperature, pressure, absc);

  for (int ilev = 0; ilev < nlevels; ++ilev) {
    bcoef[ilev].FillBoundary();
  }
}

int
main(int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    pele::testing::Checker check;
    amrex::ParmParse pp;
    PeleRad::AMRParam amrpp(pp);
    PeleRad::MLMGParam mlmgpp(pp);

    bool const write = false;
    int const nlevels = amrpp.max_level_ + 1;
    int const ref_ratio = amrpp.ref_ratio_;
    int const composite_solve = mlmgpp.composite_solve_;

    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;

    amrex::Vector<amrex::MultiFab> solution;
    amrex::Vector<amrex::MultiFab> rhs;

    amrex::Vector<amrex::MultiFab> acoef;
    amrex::Vector<amrex::MultiFab> bcoef;
    amrex::Vector<amrex::MultiFab> robin_a;
    amrex::Vector<amrex::MultiFab> robin_b;
    amrex::Vector<amrex::MultiFab> robin_f;

    amrex::Vector<amrex::MultiFab> y_co2;
    amrex::Vector<amrex::MultiFab> y_h2o;
    amrex::Vector<amrex::MultiFab> y_co;
    amrex::Vector<amrex::MultiFab> soot_fv_rad;
    amrex::Vector<amrex::MultiFab> temperature;
    amrex::Vector<amrex::MultiFab> pressure;
    amrex::Vector<amrex::MultiFab> absc;

    std::cout << "initialize data ... \n";
    initMeshandData(
      amrpp, geom, grids, dmap, solution, rhs, acoef, bcoef, robin_a, robin_b,
      robin_f, y_co2, y_h2o, y_co, soot_fv_rad, temperature, pressure, absc);

    std::cout << "construct the PDE ... \n";

    if (composite_solve) {
      PeleRad::POneMulti rte(
        mlmgpp, geom, grids, dmap, solution, rhs, acoef, bcoef, robin_a,
        robin_b, robin_f);
      rte.solve();
    } else {
      PeleRad::POneMultiLevbyLev rte(
        mlmgpp, ref_ratio, geom, grids, dmap, solution, rhs, acoef, bcoef,
        robin_a, robin_b, robin_f);
      rte.solve();
    }

    for (int ilev = 0; ilev < nlevels; ++ilev) {
      pele::testing::check_solution(
        check, solution[ilev], nullptr, temperature[ilev],
        "level " + std::to_string(ilev) + ": ");
    }

    // plot results
    if (write) {
      amrex::Vector<amrex::MultiFab> plotmf(nlevels);

      for (int ilev = 0; ilev < nlevels; ++ilev) {
        std::cout << "write the results ... \n";
        plotmf[ilev].define(grids[ilev], dmap[ilev], 9, 0);
        amrex::MultiFab::Copy(plotmf[ilev], solution[ilev], 0, 0, 1, 0);
        amrex::MultiFab::Copy(plotmf[ilev], rhs[ilev], 0, 1, 1, 0);
        amrex::MultiFab::Copy(plotmf[ilev], acoef[ilev], 0, 2, 1, 0);
        amrex::MultiFab::Copy(plotmf[ilev], bcoef[ilev], 0, 3, 1, 0);

        amrex::MultiFab::Copy(plotmf[ilev], y_co2[ilev], 0, 4, 1, 0);
        amrex::MultiFab::Copy(plotmf[ilev], y_h2o[ilev], 0, 5, 1, 0);
        amrex::MultiFab::Copy(plotmf[ilev], y_co[ilev], 0, 6, 1, 0);
        amrex::MultiFab::Copy(plotmf[ilev], soot_fv_rad[ilev], 0, 7, 1, 0);
        amrex::MultiFab::Copy(plotmf[ilev], temperature[ilev], 0, 8, 1, 0);
      }

      auto const plot_file_name = amrpp.plot_file_name_;
      amrex::WriteMultiLevelPlotfile(
        plot_file_name, nlevels, amrex::GetVecOfConstPtrs(plotmf),
        {"solution", "rhs", "acoef", "bcoef", "Y_co2", "Y_h2o", "Y_co",
         "Soot_fv_rad", "Temperature"},
        geom, 0.0, amrex::Vector<int>(nlevels, 0),
        amrex::Vector<amrex::IntVect>(nlevels, amrex::IntVect{ref_ratio}));
    }
    status = check.summary("POneMultiAF");
  }
  amrex::Finalize();
  return status;
}
