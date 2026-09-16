#include <main.H>

using namespace amrex;

void
turbinflow_from_diag_frame_planes(std::ofstream& ifsd, std::ofstream& ifsh)
{

  ParmParse pp;

  int nf = pp.countval("ifiles");
  AMREX_ALWAYS_ASSERT(nf >= 4);
  Vector<std::string> ifiles(nf);
  pp.getarr("ifiles", ifiles, 0, nf);

  int save_level = 0;
  pp.query("level", save_level);

  Array<int, AMREX_SPACEDIM> perio{1, 1, 1};
  int np = pp.countval("periodicity");
  if (np == 0) {
    amrex::Print() << "periodcity not specified, assuming fully periodic domain"
                   << std::endl;
  } else if (np != AMREX_SPACEDIM) {
    amrex::Error("Must provide 3 values for periodicity");
  }
  pp.query("periodicity", perio);

  int normal;
  pp.get("normal", normal);
  amrex::Vector<std::string> varnames{3};
  if (normal == 0) {
    varnames[0] = "y_velocity";
    varnames[1] = "z_velocity";
    varnames[2] = "x_velocity";
  } else if (normal == 1) {
    varnames[0] = "x_velocity";
    varnames[1] = "z_velocity";
    varnames[2] = "y_velocity";
  } else if (normal == 2) {
    varnames[0] = "x_velocity";
    varnames[1] = "y_velocity";
    varnames[2] = "z_velocity";
  } else {
    amrex::Abort("invalid normal");
  }
  amrex::Vector<int> varidx;
  varidx.resize(varnames.size());

  Box domain;
  RealBox probDomain;
  int coord_sys;
  amrex::Vector<std::string> varlist;
  {
    pele::physics::pltfilemanager::PltFileManager firstfile(ifiles[0]);
    int Nlevels = firstfile.getNlev();
    AMREX_ALWAYS_ASSERT(save_level >= 0 && save_level < Nlevels);
    domain = firstfile.getGeom(save_level).Domain();
    probDomain = firstfile.getGeom(save_level).ProbDomain();
    coord_sys = firstfile.getGeom(save_level).CoordInt();
    varlist = firstfile.getVariableList();
  }

  // Find the variables we need in the plt file
  for (int i = 0; i < varidx.size(); ++i) {
    int j;
    for (j = 0; j < varlist.size(); ++j) {
      if (varnames[i] == varlist[j]) {
        varidx[i] = j;
        break;
      }
    }
    if (j >= varlist.size()) {
      amrex::Abort(
        "Requested variable " + varnames[i] + " not found in plt file");
    }
  }

  //
  // Write the first part of the header.
  //
  IntVect bg;
  bg[0] = domain.length(0);
  bg[1] = domain.length(1);
  bg[2] = nf;
  ifsh << bg[0] + 3 << ' ' << bg[1] + 3 << ' ' << bg[2] << '\n';

  Vector<Real> planeSize(2), dx(2);
  planeSize[0] = probDomain.length(0);
  planeSize[1] = probDomain.length(1);

  dx[0] = planeSize[0] / bg[0];
  dx[1] = planeSize[1] / bg[1];
  // Full precision: TurbInflow derives the file's spacing from these, and
  // for a mesh-mapped file they must agree with the trailer's Xi bounds.
  ifsh << std::setprecision(17) << planeSize[0] + 2 * dx[0] << ' '
       << planeSize[1] + 2 * dx[1] << ' ' << nf << '\n';

  ifsh << perio[0] << ' ' << perio[1] << ' ' << 0 << '\n';

  Vector<Real> fileTimes(nf);
  IntVect periodicity{perio[0], perio[1], 0};
  IntVect dim_map{0, 1, 2};
  for (int d = 0; d < varidx.size(); ++d) {
    std::cout << "Loading component " << d << " ... " << std::endl;

    for (int k = 0; k < nf; ++k) {
      pele::physics::pltfilemanager::PltFileManager pltfile(ifiles[k]);
      if (d == 0) {
        fileTimes[k] = pltfile.getTime();
        int Nlevels = pltfile.getNlev();
        AMREX_ALWAYS_ASSERT(save_level >= 0 && save_level < Nlevels);
      }

      // Dummy MultiFab with 1 Fab and 2 ghost cells
      BoxArray bxs{domain};
      DistributionMapping dm{bxs, 1};
      MultiFab mf{bxs, dm, 1, {0, 0, 0}};
      Geometry geom{domain, probDomain, coord_sys, perio};
      pltfile.fillPatchFromPlt(save_level, geom, varidx[d], 0, 1, mf);
      Box tbx{{0, 0, 0}, {bg[0] + 2, bg[1] + 2, 0}};
      FArrayBox TMP(tbx, 1);
      TMP.setVal(1.2e34);

      const auto& pltarr = mf[0].const_array();
      const auto& planearr = TMP.array();

      ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const IntVect idx_plane{i, j, k};
        const IntVect idx_plt =
          index_mapper(idx_plane, dim_map, domain, periodicity);
        planearr(idx_plane) = pltarr(idx_plt);
      });
      ifsh << ifsd.tellp() << std::endl;
      TMP.writeOn(ifsd);
      amrex::Print() << "for d = " << d << " and k = " << k
                     << ": minval = " << TMP.min(0)
                     << ", maxval = " << TMP.max(0) << std::endl;
    }
    std::cout << "done" << std::endl;
  }
  // Write plane times
  for (int k = 0; k < nf; ++k) {
    ifsh << std::setprecision(17) << fileTimes[k] << std::endl;
  }

  // Optional mesh-mapping trailer; must come after the plane times.
  write_meshmap_trailer(ifsh, probDomain, dim_map);
}
