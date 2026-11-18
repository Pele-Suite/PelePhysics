#include <main.H>

using namespace amrex;

void
readHIT(HITData* a_data)
{
  ParmParse pp;
  std::string hit_file("IC");
  pp.query("hit_file", hit_file);

  pp.query("input_ncell", a_data->input_ncell);
  if (a_data->input_ncell == 0) {
    amrex::Abort("for HIT, the input ncell cannot be 0 !");
  }

  int binfmt = 0; // Default is ASCII format
  pp.query("input_binaryformat", binfmt);
  pp.query("urms0", a_data->urms0);

  amrex::Real lambda0 = 0.5;
  amrex::Real tau = lambda0 / a_data->urms0;

  // Output initial conditions
  std::ofstream ofs("initialConditions.txt", std::ofstream::out);
  amrex::Print(ofs) << "lambda0, urms0, tau \n";
  amrex::Print(ofs).SetPrecision(17)
    << lambda0 << "," << a_data->urms0 << "," << tau << std::endl;
  ofs.close();

  // Read initial velocity field
  const size_t nx = a_data->input_ncell;
  const size_t ny = a_data->input_ncell;
  const size_t nz = a_data->input_ncell;
  amrex::Vector<amrex::Real> data(
    nx * ny * nz * 6); /* this needs to be double */
  if (binfmt) {
    read_binary(hit_file, nx, ny, nz, 6, data);
  } else {
    read_csv(hit_file, nx, ny, nz, data);
  }

  // Extract position and velocities
  amrex::Vector<amrex::Real> xinput;
  amrex::Vector<amrex::Real> uinput;
  amrex::Vector<amrex::Real> vinput;
  amrex::Vector<amrex::Real> winput;
  amrex::Vector<amrex::Real> xdiff;
  amrex::Vector<amrex::Real> xarray;

  xinput.resize(nx * ny * nz);
  uinput.resize(nx * ny * nz);
  vinput.resize(nx * ny * nz);
  winput.resize(nx * ny * nz);

  for (long i = 0; i < xinput.size(); i++) {
    xinput[i] = data[0 + i * 6];
    uinput[i] = data[3 + i * 6] * a_data->urms0 / a_data->uin_norm;
    vinput[i] = data[4 + i * 6] * a_data->urms0 / a_data->uin_norm;
    winput[i] = data[5 + i * 6] * a_data->urms0 / a_data->uin_norm;
  }

  // Get the xarray table and the differences.
  xarray.resize(nx);
  for (long i = 0; i < xarray.size(); i++) {
    xarray[i] = xinput[i];
  }
  xdiff.resize(nx);
  std::adjacent_difference(xarray.begin(), xarray.end(), xdiff.begin());
  xdiff[0] = xdiff[1];

  // Make sure the search array is increasing
  if (not std::is_sorted(xarray.begin(), xarray.end())) {
    amrex::Abort("Error: non ascending x-coordinate array.");
  }
  if (std::abs(xarray[0] - 0.5 * xdiff[0]) > 1e-12 * xdiff[0]) {
    amrex::Abort("Error: domain must start at 0");
  }
  for (const amrex::Real& xd : xdiff) {
    if (std::abs(xd - xdiff[0]) / xdiff[0] > 1e-12) {
      amrex::Abort("Error: grid must be uniformly spaced");
    }
  }

  // Pass data to the prob_parm
  a_data->Linput = xarray[nx - 1] + 0.5 * xdiff[nx - 1];

  a_data->d_xarray =
    (amrex::Real*)amrex::The_Arena()->alloc(nx * sizeof(amrex::Real));
  a_data->d_xdiff =
    (amrex::Real*)amrex::The_Arena()->alloc(nx * sizeof(amrex::Real));
  a_data->d_uinput =
    (amrex::Real*)amrex::The_Arena()->alloc(nx * ny * nz * sizeof(amrex::Real));
  a_data->d_vinput =
    (amrex::Real*)amrex::The_Arena()->alloc(nx * ny * nz * sizeof(amrex::Real));
  a_data->d_winput =
    (amrex::Real*)amrex::The_Arena()->alloc(nx * ny * nz * sizeof(amrex::Real));

  // The_Arena is device memory on GPU builds
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, xarray.begin(), xarray.end(), a_data->d_xarray);
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, xdiff.begin(), xdiff.end(), a_data->d_xdiff);
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, uinput.begin(), uinput.end(), a_data->d_uinput);
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, vinput.begin(), vinput.end(), a_data->d_vinput);
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, winput.begin(), winput.end(), a_data->d_winput);
}

void
turbinflow_from_turb_box(std::ofstream& ifsd, std::ofstream& ifsh)
{

  HITData data;
  readHIT(&data);

  int ncell = data.input_ncell;
  Real xlo = 0.0;
  Real xhi = data.Linput;

  Box box_turb(
    IntVect(AMREX_D_DECL(0, 0, 0)),
    IntVect(AMREX_D_DECL(ncell - 1, ncell - 1, ncell - 1)));
  RealBox rb_turb({AMREX_D_DECL(xlo, xlo, xlo)}, {AMREX_D_DECL(xhi, xhi, xhi)});
  int coord_turb(0);
  Array<int, AMREX_SPACEDIM> per_turb = {AMREX_D_DECL(1, 1, 1)};
  Geometry geom_turb(box_turb, rb_turb, coord_turb, per_turb);
  const Real* dx_turb = geom_turb.CellSize();

  // Fill the velocity FAB with HIT
  FArrayBox vel_turb(box_turb, AMREX_SPACEDIM);
  Array4<Real> const& fab = vel_turb.array();
  auto geomdata = geom_turb.data();
  AMREX_PARALLEL_FOR_3D(
    box_turb, i, j, k, { fillVelFab(i, j, k, fab, geomdata, data); });

  //
  // Write the first part of the Turb header.
  // Note that this is solely for periodic style inflow files.
  //
  Box box_turb_io(box_turb);
  box_turb_io.setBig(0, box_turb.bigEnd(0) + 3);
  box_turb_io.setBig(1, box_turb.bigEnd(1) + 3);
  box_turb_io.setBig(2, box_turb.bigEnd(2));

  ifsh << box_turb_io.length(0) << ' ' << box_turb_io.length(1) << ' '
       << box_turb_io.length(2) << '\n';

  ifsh << rb_turb.length(0) + 2 * dx_turb[0] << ' '
       << rb_turb.length(1) + 2 * dx_turb[1] << ' ' << rb_turb.length(2)
       << '\n';

  ifsh << per_turb[0] << ' ' << per_turb[1] << ' ' << per_turb[2] << '\n';

  // Dump field as a "turbulence file"
  IntVect sm = box_turb_io.smallEnd();
  IntVect bg = box_turb_io.bigEnd();
  IntVect dim_map{0, 1, 2};
  FArrayBox TMP;
  //
  // Iterate through Z-planes and save
  //
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    for (int k = 0; k <= box_turb.bigEnd(2); k++) {
      sm[2] = k;
      bg[2] = k;
      Box tbx(sm, bg);
      TMP.resize(tbx, 1);
      const auto& planearr = TMP.array();
      ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int kp) noexcept {
        const IntVect idx_plane{i, j, kp};
        const IntVect idx_plt =
          index_mapper(idx_plane, dim_map, box_turb, {1, 1, 1});
        planearr(idx_plane) = fab(idx_plt, dim_map[d]);
      });
      ifsh << ifsd.tellp() << std::endl;
      TMP.writeOn(ifsd);
      amrex::Print() << "for d = " << d << " and k = " << k
                     << ": minval = " << TMP.min<RunOn::Device>(0)
                     << ", maxval = " << TMP.max<RunOn::Device>(0) << std::endl;
    }
  }

  amrex::The_Arena()->free(data.d_xarray);
  amrex::The_Arena()->free(data.d_xdiff);
  amrex::The_Arena()->free(data.d_uinput);
  amrex::The_Arena()->free(data.d_vinput);
  amrex::The_Arena()->free(data.d_winput);
}
