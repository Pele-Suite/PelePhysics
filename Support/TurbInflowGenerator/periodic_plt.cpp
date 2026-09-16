#include <main.H>

using namespace amrex;

void
turbinflow_from_periodic_plt(std::ofstream& ifsd, std::ofstream& ifsh)
{

  ParmParse pp;
  std::string ifile;
  pp.get("ifile", ifile);

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
  if (perio[normal] == 0) {
    amrex::Abort("input plot file must be periodic in normal direction");
  }
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

  pele::physics::pltfilemanager::PltFileManager pltfile(ifile);
  int Nlevels = pltfile.getNlev();
  AMREX_ALWAYS_ASSERT(save_level >= 0 && save_level < Nlevels);
  const auto& domain = pltfile.getGeom(save_level).Domain();
  const auto& probDomain = pltfile.getGeom(save_level).ProbDomain();
  const auto& coord_sys = pltfile.getGeom(save_level).CoordInt();
  const auto& varlist = pltfile.getVariableList();

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

  // Construct dim_map from turbinflow plane dims to plt file dims
  IntVect dim_map;
  int cdim = 0;
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
    if (i != normal) {
      dim_map[cdim] = i;
      cdim += 1;
    }
    dim_map[2] = normal;
  }

  // Write the Header
  IntVect bg;
  bg[0] = domain.length(dim_map[0]);
  bg[1] = domain.length(dim_map[1]);
  bg[2] = domain.length(dim_map[2]);
  ifsh << bg[0] + 3 << ' ' << bg[1] + 3 << ' ' << bg[2] << '\n';

  Vector<Real> planeSize(3), dx(3);
  planeSize[0] = probDomain.length(dim_map[0]);
  planeSize[1] = probDomain.length(dim_map[1]);
  planeSize[2] = probDomain.length(dim_map[2]);
  dx[0] = planeSize[0] / bg[0];
  dx[1] = planeSize[1] / bg[1];
  dx[2] = planeSize[2] / bg[2];
  // Full precision: TurbInflow derives the file's spacing from these, and
  // for a mesh-mapped file they must agree with the trailer's Xi bounds.
  ifsh << std::setprecision(17) << planeSize[0] + 2 * dx[0] << ' '
       << planeSize[1] + 2 * dx[1] << ' ' << planeSize[2] << '\n';
  ifsh << perio[dim_map[0]] << ' ' << perio[dim_map[1]] << ' ' << 1 << '\n';

  IntVect periodicity{perio[0], perio[1], perio[2]};
  // Output FABS
  for (int d = 0; d < varidx.size(); ++d) {
    std::cout << "Loading component " << d << " ... " << std::flush;

    // Construct a FAB on the level we want from the plt file data
    BoxArray bxs{domain};
    DistributionMapping dm{bxs, 1};
    MultiFab mf{bxs, dm, 1, 0};
    Geometry geom{domain, probDomain, coord_sys, perio};
    pltfile.fillPatchFromPlt(save_level, geom, varidx[d], 0, 1, mf);
    const auto& pltarr = mf[0].const_array();

    // Loop over normal direction dumping slices
    // reverse order (assume flow in positive direction, this works backwards
    // from outflow with Taylor's hypothesis)
    for (int k = bg[2] - 1; k >= 0; --k) {
      Box tbx{{0, 0, k}, {bg[0] + 2, bg[1] + 2, k}};
      FArrayBox TMP(tbx, 1);
      const auto& planearr = TMP.array();
      ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const IntVect idx_plane{i, j, k};
        const IntVect idx_plt =
          index_mapper(idx_plane, dim_map, domain, periodicity);
        planearr(idx_plane) = pltarr(idx_plt);
      });
      TMP.shift({0, 0, -k});
      ifsh << ifsd.tellp() << std::endl;
      TMP.writeOn(ifsd);
      amrex::Print() << "for d = " << d << " and k = " << k
                     << ": minval = " << TMP.min(0)
                     << ", maxval = " << TMP.max(0) << std::endl;
    }
    std::cout << "done" << std::endl;
  }

  // Optional mesh-mapping trailer; must come after the plane offsets (this
  // mode writes no plane times).
  write_meshmap_trailer(ifsh, probDomain, dim_map);
}
