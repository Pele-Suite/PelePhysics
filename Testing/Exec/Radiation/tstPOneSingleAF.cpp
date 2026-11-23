#include <AMReX.H>
#include <PlanckMean.H>
#include <SpectralModels.H>
#include <AMReX_PlotFileUtil.H>
#include <POneSingle.H>

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
    amrex::Real ka = amrex::max<amrex::Real>(0.001, absc(i, j, k) * 100);
    beta(i, j, k) = 1.0 / ka;
    rhs(i, j, k) = 4.0 * ka * 5.67e-8 * std::pow(T(i, j, k), 4.0);
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
  amrex::Geometry& geom,
  amrex::MultiFab& solution,
  amrex::MultiFab& rhs,
  amrex::MultiFab& acoef,
  amrex::MultiFab& bcoef,
  amrex::MultiFab& robin_a,
  amrex::MultiFab& robin_b,
  amrex::MultiFab& robin_f,
  amrex::MultiFab& y_co2,
  amrex::MultiFab& y_h2o,
  amrex::MultiFab& y_co,
  amrex::MultiFab& soot_fv_rad,
  amrex::MultiFab& temperature,
  amrex::MultiFab& pressure,
  amrex::MultiFab& absc)
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

  auto const prob_lo = geom.ProbLoArray();
  auto const prob_hi = geom.ProbHiArray();
  auto const dx = geom.CellSizeArray();
  auto const dlo = amrex::lbound(geom.Domain());
  auto const dhi = amrex::ubound(geom.Domain());

  for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
    amrex::Box const& bx = mfi.validbox();
    amrex::Box const& gbx = amrex::grow(bx, 1);
    auto const& Yco2 = y_co2.array(mfi);
    auto const& Yh2o = y_h2o.array(mfi);
    auto const& Yco = y_co.array(mfi);
    auto const& fv = soot_fv_rad.array(mfi);
    auto const& T = temperature.array(mfi);
    auto const& P = pressure.array(mfi);
    auto const& kappa = absc.array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      initGasField(i, j, k, Yco2, Yh2o, Yco, fv, T, P, dx, prob_lo, prob_hi);
      getRadPropGas(i, j, k, Yco2, Yh2o, Yco, T, P, kappa, kpco2, kph2o, kpco);
    });

    // if soot exists
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      getRadPropSoot(i, j, k, fv, T, kappa, kpsoot);
    });

    auto const& rhsfab = rhs.array(mfi);
    auto const& acfab = acoef.array(mfi);
    auto const& bcfab = bcoef.array(mfi);
    auto const& rafab = robin_a.array(mfi);
    auto const& rbfab = robin_b.array(mfi);
    auto const& rffab = robin_f.array(mfi);

    amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      actual_init_coefs(i, j, k, rhsfab, acfab, bcfab, dlo, dhi, bx, kappa, T);
    });

    amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      actual_init_bc_coefs(
        i, j, k, acfab, bcfab, rafab, rbfab, rffab, dlo, dhi, T);
    });
  }

  solution.setVal(0.0, 0, 1, amrex::IntVect(0));
}

void
initMeshandData(
  PeleRad::AMRParam const& amrpp,
  amrex::Geometry& geom,
  amrex::BoxArray& grids,
  amrex::DistributionMapping& dmap,
  amrex::MultiFab& solution,
  amrex::MultiFab& rhs,
  amrex::MultiFab& rad_src,
  amrex::MultiFab& acoef,
  amrex::MultiFab& bcoef,
  amrex::MultiFab& robin_a,
  amrex::MultiFab& robin_b,
  amrex::MultiFab& robin_f,
  amrex::MultiFab& y_co2,
  amrex::MultiFab& y_h2o,
  amrex::MultiFab& y_co,
  amrex::MultiFab& soot_fv_rad,
  amrex::MultiFab& temperature,
  amrex::MultiFab& pressure,
  amrex::MultiFab& absc)
{
  int const n_cell = amrpp.n_cell_;
  int const max_grid_size = amrpp.max_grid_size_;

  amrex::RealBox rb(
    {AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(12.5, 12.5, 75)});
  amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
  amrex::Geometry::Setup(&rb, 0, is_periodic.data());

  std::vector<int> npts{n_cell, n_cell, 6 * n_cell};
  amrex::Box domain0(
    amrex::IntVect{AMREX_D_DECL(0, 0, 0)},
    amrex::IntVect{AMREX_D_DECL(npts[0] - 1, npts[1] - 1, npts[2] - 1)});
  geom.define(domain0);

  grids.define(domain0);
  grids.maxSize(max_grid_size);

  amrex::IntVect ng = amrex::IntVect{1};

  dmap.define(grids);
  solution.define(grids, dmap, 1, ng);
  rhs.define(grids, dmap, 1, 0);
  rad_src.define(grids, dmap, 1, 0);
  acoef.define(grids, dmap, 1, 0);
  bcoef.define(grids, dmap, 1, ng);
  robin_a.define(grids, dmap, 1, ng);
  robin_b.define(grids, dmap, 1, ng);
  robin_f.define(grids, dmap, 1, ng);

  y_co2.define(grids, dmap, 1, 0);
  y_h2o.define(grids, dmap, 1, 0);
  y_co.define(grids, dmap, 1, 0);
  soot_fv_rad.define(grids, dmap, 1, 0);
  temperature.define(grids, dmap, 1, 0);
  pressure.define(grids, dmap, 1, 0);
  absc.define(grids, dmap, 1, 0);

  initProbABecLaplacian(
    geom, solution, rhs, acoef, bcoef, robin_a, robin_b, robin_f, y_co2, y_h2o,
    y_co, soot_fv_rad, temperature, pressure, absc);

  bcoef.FillBoundary();
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

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab solution;
    amrex::MultiFab rhs;
    amrex::MultiFab rad_src;

    amrex::MultiFab acoef;
    amrex::MultiFab bcoef;
    amrex::MultiFab robin_a;
    amrex::MultiFab robin_b;
    amrex::MultiFab robin_f;

    amrex::MultiFab y_co2;
    amrex::MultiFab y_h2o;
    amrex::MultiFab y_co;
    amrex::MultiFab soot_fv_rad;
    amrex::MultiFab temperature;
    amrex::MultiFab pressure;
    amrex::MultiFab absc;

    // std::cout << "initialize data ... \n";
    initMeshandData(
      amrpp, geom, grids, dmap, solution, rhs, rad_src, acoef, bcoef, robin_a,
      robin_b, robin_f, y_co2, y_h2o, y_co, soot_fv_rad, temperature, pressure,
      absc);

    // std::cout << "construct the PDE ... \n";

    PeleRad::POneSingle rte(
      mlmgpp, geom, grids, dmap, solution, rhs, acoef, bcoef, robin_a, robin_b,
      robin_f);
    // std::cout << "solve the PDE ... \n";
    rte.solve();

    // calculate radiative heat source
    for (amrex::MFIter mfi(rad_src); mfi.isValid(); ++mfi) {
      amrex::Box const& bx = mfi.validbox();

      auto const& rhsfab = rhs.array(mfi);
      auto const& solfab = solution.array(mfi);
      auto const& acfab = acoef.array(mfi);

      auto radfab = rad_src.array(mfi);

      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          radfab(i, j, k) = acfab(i, j, k) * solfab(i, j, k) - rhsfab(i, j, k);
        });
    }

    pele::testing::check_solution(check, solution, &rad_src, temperature, "");

    // plot results
    if (write) {
      // std::cout << "write the results ... \n";
      amrex::MultiFab plotmf(grids, dmap, 10, 0);
      amrex::MultiFab::Copy(plotmf, solution, 0, 0, 1, 0);
      amrex::MultiFab::Copy(plotmf, rhs, 0, 1, 1, 0);
      amrex::MultiFab::Copy(plotmf, acoef, 0, 2, 1, 0);
      amrex::MultiFab::Copy(plotmf, bcoef, 0, 3, 1, 0);
      amrex::MultiFab::Copy(plotmf, rad_src, 0, 4, 1, 0);

      amrex::MultiFab::Copy(plotmf, y_co2, 0, 5, 1, 0);
      amrex::MultiFab::Copy(plotmf, y_h2o, 0, 6, 1, 0);
      amrex::MultiFab::Copy(plotmf, y_co, 0, 7, 1, 0);
      amrex::MultiFab::Copy(plotmf, soot_fv_rad, 0, 8, 1, 0);
      amrex::MultiFab::Copy(plotmf, temperature, 0, 9, 1, 0);

      auto const plot_file_name = amrpp.plot_file_name_;
      amrex::WriteSingleLevelPlotfile(
        plot_file_name, plotmf,
        {"G", "Emis", "kappa", "bcoef", "RadSrc", "Y_co2", "Y_h2o", "Y_co",
         "Soot_fv", "Temperature"},
        geom, 0.0, 0);
    }
    status = check.summary("POneSingleAF");
  }
  amrex::Finalize();
  return status;
}
