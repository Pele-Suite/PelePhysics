#include <turbinflow.H>

#include <algorithm>
#include <fstream>
#include <limits>

namespace pele::physics::turbinflow {

namespace {
// Optional HDR trailers.  Everything a legacy reader consumes ends with the
// plane times, so anything appended after them is invisible to older code.
// Each trailer starts with a keyword token.
//
// Mesh mapping.  A file without it is, by declaration, uniform in physical
// position.  Format (one line per transverse direction, in the file's own
// transverse order):
//
//   MESHMAP_V2
//   <kind> <p> <p2> <p3> <q> <xi_lo> <xi_hi>
//   <kind> <p> <p2> <p3> <q> <xi_lo> <xi_hi>
//
// kind/p/p2/p3/q are the MeshMapEvaluator payload for that axis (0 identity,
// 1 constant, 2 exp stretch, 3 tanh stretch, 4 interior stretch);
// xi_lo/xi_hi are the precursor's computational-domain bounds along that
// axis, which the inverse map needs.  The earlier MESHMAP_V1 form,
// <kind> <p> <q> <xi_lo> <xi_hi>, is read with p2 = p3 = 0.  A trailer marks
// the file as uniform in the precursor's Xi coordinate rather than in
// physical position; add_turb() inverts the map before indexing the file.
//
// Content this reader does not recognise (a trailer from a newer generator,
// or stray tokens) is skipped up to the next keyword it knows, so a known
// trailer is found whatever precedes it.  One warning is printed per
// unrecognised run.

// Read the body of the MESHMAP trailer introduced by keyword token
void
read_meshmap_trailer(
  std::istream& is,
  const std::string& token,
  const std::string& turb_header,
  const std::string& name,
  TurbParm& a_tp)
{
  TurbFileGeom& g = a_tp.geom;
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    !g.has_map,
    "TurbInflow::init(): duplicate MESHMAP trailer in " + turb_header);
  const bool v2 = (token == "MESHMAP_V2");
  int kind[2] = {0, 0};
  for (int idim = 0; idim < 2; ++idim) {
    amrex::Real p = 0.0;
    amrex::Real p2 = 0.0;
    amrex::Real p3 = 0.0;
    int q = -1;
    if (v2) {
      is >> kind[idim] >> p >> p2 >> p3 >> q;
    } else {
      is >> kind[idim] >> p >> q;
    }
    is >> g.map_xi_lo[idim] >> g.map_xi_hi[idim];
    g.map.m_p[idim] = p;
    g.map.m_p2[idim] = p2;
    g.map.m_p3[idim] = p3;
    g.map.m_q[idim] = q;
  }
  // !fail() rather than good(): a trailer that ends the file without a
  // final newline sets eofbit on its last value, and is still valid.
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    !is.fail(),
    "TurbInflow::init(): malformed " + token + " trailer in " + turb_header);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    kind[0] == kind[1], "TurbInflow::init(): " + token + " trailer in " +
                          turb_header +
                          " names two different map kinds; a precursor has "
                          "one map");
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    kind[0] >= static_cast<int>(MeshMapEvaluator::Kind::Identity) &&
      kind[0] <= static_cast<int>(MeshMapEvaluator::Kind::InteriorStretch),
    "TurbInflow::init(): unknown map kind in " + token + " trailer of " +
      turb_header);
  for (int idim = 0; idim < 2; ++idim) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      g.map_xi_hi[idim] > g.map_xi_lo[idim], "TurbInflow::init(): " + token +
                                               " trailer in " + turb_header +
                                               " has xi_hi <= xi_lo");
  }
  g.map.m_kind = static_cast<MeshMapEvaluator::Kind>(kind[0]);
  g.has_map = true;
  // The trailer's Xi bounds are exact while the header's probsize may have
  // been written with limited precision; derive the file's Xi spacing from
  // the trailer so that the inverse map and the index lookup agree to
  // round-off, after checking the two are consistent.
  for (int idim = 0; idim < 2; ++idim) {
    const amrex::Real L = g.map_xi_hi[idim] - g.map_xi_lo[idim];
    const amrex::Real dx_tr = L / static_cast<amrex::Real>(g.npboxcells[idim]);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      std::abs(dx_tr - g.dx[idim]) <= 1.0e-5 * g.dx[idim],
      "TurbInflow::init(): " + token + " trailer of " + turb_header +
        " gives a Xi extent inconsistent with the header's probsize");
    g.dx[idim] = dx_tr;
    g.dxinv[idim] = 1.0 / dx_tr;
    g.pboxsize[idim] = L;
  }
  if (a_tp.verbose > 0) {
    amrex::Print() << "   " << name << " carries a " << token
                   << " trailer: file is uniform in the precursor's Xi "
                      "coordinate (map kind "
                   << kind[0] << ")\n";
  }
}

// Read the HDR of a_tp's file: its geometry, the plane offsets and times,
// and the optional trailers
void
read_header(TurbParm& a_tp, const std::string& name)
{
  const std::string turb_header = a_tp.m_turb_file + "/HDR";
  std::ifstream is(turb_header.c_str());
  if (!is.is_open()) {
    amrex::Abort("Unable to open input file " + turb_header);
  }
  TurbFileGeom& g = a_tp.geom;

  // The following is relative to the injection face: 0 and 1 are the
  // transverse directions, 2 is normal
  amrex::Array<int, 3> npts = {{0}};
  amrex::Array<amrex::Real, 3> probsize = {{0}};
  amrex::Array<int, 3> periodicity = {{1, 1, 1}};
  is >> npts[0] >> npts[1] >> npts[2];
  is >> probsize[0] >> probsize[1] >> probsize[2];
  // Normal periodicity tells whether the file holds time planes
  is >> periodicity[0] >> periodicity[1] >> periodicity[2];

  a_tp.istimeplanes = periodicity[2] == 0;
  if (periodicity[0] == 0 || periodicity[1] == 0) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      g.lininterp, "Linear interpolation required for turbinflow with "
                   "nonperiodic directions");
  }

  // One ghost point on each side, tangential to the inflow face
  for (int idim = 0; idim < 2; ++idim) {
    g.periodicity[idim] = periodicity[idim];
    g.dx[idim] = probsize[idim] / static_cast<amrex::Real>(npts[idim] - 1);
    g.dxinv[idim] = 1.0 / g.dx[idim];
    g.pboxsize[idim] = probsize[idim] - 2.0 * g.dx[idim];
    g.npboxcells[idim] = npts[idim] - 3;
  }
  a_tp.dz = probsize[2] / static_cast<amrex::Real>(npts[2]);
  a_tp.dzinv = 1.0 / a_tp.dz;
  a_tp.kmax = npts[2];

  // Swirl type: we can't load more planes than are available
  if (a_tp.istimeplanes) {
    a_tp.nplane = amrex::min<int>(a_tp.nplane, npts[2]);
  }

  // Offset for each plane in Binary TurbFile
  a_tp.offset.resize(static_cast<long>(a_tp.kmax) * a_tp.ncomp);
  for (auto& off : a_tp.offset) {
    is >> off;
  }

  if (a_tp.istimeplanes) {
    a_tp.planeTimes.resize(a_tp.kmax);
    for (auto& t : a_tp.planeTimes) {
      is >> t; // Time for each plane
    }
  }

  std::string token;
  bool skipping = false;
  while (is >> token) {
    if (token != "MESHMAP_V1" && token != "MESHMAP_V2") {
      if (!skipping) {
        amrex::Print() << "TurbInflow: WARNING skipping unrecognised "
                          "trailing content in "
                       << turb_header << " starting at '" << token << "'\n";
      }
      skipping = true;
      continue;
    }
    skipping = false;
    read_meshmap_trailer(is, token, turb_header, name, a_tp);
  }
}

// Place a_tp's file on the injection face (pboxlo) from turb_center, given
// (has_turb_center) or not
void
place_file(
  TurbParm& a_tp,
  const std::string& name,
  amrex::Vector<amrex::Real> turb_center,
  const bool has_turb_center)
{
  TurbFileGeom& g = a_tp.geom;
  // For a physically-uniform file turb_center is in case units and the file
  // is placed around it.  For a mesh-mapped file the sampling coordinate is
  // the precursor's Xi (add_turb() inverts the map), so the natural
  // placement is the precursor's own: pboxlo = xi_lo, i.e. turb_center =
  // (xi_lo+xi_hi)/2 in the file's Xi units, and that is the default; a
  // supplied turb_center is taken in those same Xi units (unscaled) and
  // shifts the pattern within the plane.
  if (g.has_map) {
    for (int idim = 0; idim < 2; ++idim) {
      if (!has_turb_center) {
        turb_center[idim] = 0.5 * (g.map_xi_lo[idim] + g.map_xi_hi[idim]);
      } else {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          turb_center[idim] >= g.map_xi_lo[idim] &&
            turb_center[idim] <= g.map_xi_hi[idim],
          "TurbInflow::init(): turb_center for a mesh-mapped turbulence "
          "file is in the file's Xi units and must lie within "
          "[xi_lo, xi_hi] of its MESHMAP trailer");
      }
    }
    if (a_tp.verbose > 0) {
      amrex::Print() << "   turb_center of " << name
                     << " in file Xi units: " << turb_center[0] << " "
                     << turb_center[1] << (has_turb_center ? "" : " (default)")
                     << ", physical (case units): "
                     << g.x_phys(0, turb_center[0]) / g.scale_loc << " "
                     << g.x_phys(1, turb_center[1]) / g.scale_loc << "\n";
    }
  } else {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      has_turb_center,
      "turbinflow." + name +
        ".turb_center is required for a turbulence file that is uniform "
        "in physical position (no MESHMAP trailer)");
    for (amrex::Real& tc : turb_center) {
      tc *= g.scale_loc;
    }
  }
  for (int idim = 0; idim < 2; ++idim) {
    g.pboxlo[idim] = turb_center[idim] - 0.5 * g.pboxsize[idim];
  }
}

// For a time-plane file with time_periodic, bring plane time z into
// [t_0, t_{kmax-2}): the last plane is only there to interpolate towards.
// Any other z is returned unchanged.
amrex::Real
wrap_plane_time(const TurbParm& a_tp, const amrex::Real z)
{
  if (!a_tp.istimeplanes || !a_tp.time_periodic || a_tp.kmax < 2) {
    return z;
  }
  const amrex::Real t_start = a_tp.planeTimes[0];
  const amrex::Real period = a_tp.planeTimes[a_tp.kmax - 2] - t_start;
  if (period <= 0.0) {
    return z;
  }
  const amrex::Real from_start = z - t_start;
  return t_start + from_start - std::floor(from_start / period) * period;
}

// Source of TurbParm::chunk_generation: one counter for all files and all
// init() calls, so that a sampler never matches a chunk it did not come from
int last_chunk_generation = 0;
} // namespace

void
TurbInflow::init(amrex::Geometry const& /*geom*/)
{
  amrex::ParmParse ppr;

  const int n_tp = ppr.countval("turbinflows");
  if (n_tp > 0) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      AMREX_SPACEDIM == 3, "TurbInflow::init(): TurbInflows are only supported "
                           "for 3 dimensional simulations for now");
    tp.resize(n_tp);
  }

  for (int n = 0; n < n_tp; n++) {
    std::string name;
    ppr.get("turbinflows", name, n);
    amrex::ParmParse pp("turbinflow." + name);
    if (pp.countval("turb_file") == 0) {
      continue;
    }
    TurbParm& t = tp[n];

    // Query data
    pp.query("turb_file", t.m_turb_file);
    pp.query("dir", t.dir);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      t.dir >= 0 && t.dir < AMREX_SPACEDIM,
      "Injection direction is needed: 0, 1 or 2");
    std::string side;
    pp.query("side", side);
    if (side == "low") {
      t.side = amrex::Orientation::low;
    } else if (side == "high") {
      t.side = amrex::Orientation::high;
    } else {
      amrex::Abort("turbinflow.side can only be low or high");
    }
    pp.query("time_offset", t.time_shift);
    pp.query("turb_scale_loc", t.geom.scale_loc);
    pp.query("turb_scale_vel", t.turb_scale_vel);
    pp.query("verbose", t.verbose);
    pp.query("extrap_nonperiodic", t.geom.extrap_nonperiodic);
    pp.query("tile_periodic", t.geom.tile_periodic);
    pp.query("time_periodic", t.time_periodic);
    // linear required for non-periodic directions
    TurbInterpType interp_type = TurbInterpType::quadratic;
    pp.query("interp_type", interp_type);
    t.geom.lininterp = interp_type == TurbInterpType::linear;
    if (t.verbose > 0) {
      amrex::Print() << "Initializing turbInflow " << name << " with file "
                     << t.m_turb_file
                     << " (location coordinates in will be scaled by "
                     << t.geom.scale_loc << " and velocity out to be scaled by "
                     << t.turb_scale_vel << ") \n";
    }
    if (t.geom.scale_loc == 0.0) {
      amrex::Abort(
        "TurbInflow::init(): turb_scale_loc must be non-zero for " + name);
    }

    // Get the turbcenter on the injection face.  Required for a file that
    // is uniform in physical position; optional for a mesh-mapped file
    // (see the MESHMAP trailer), whose default centre follows from the
    // trailer.
    amrex::Vector<amrex::Real> turb_center(AMREX_SPACEDIM - 1, 0);
    const bool has_turb_center = pp.queryarr("turb_center", turb_center) != 0;
    if (has_turb_center) {
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        turb_center.size() == AMREX_SPACEDIM - 1,
        "turb_center must have AMREX_SPACEDIM-1 elements");
    }

    pp.query("turb_nplane", t.nplane);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      t.nplane > 3, "need at least 4 turb planes for 3 point "
                    "interpolation stencil + 1 extra");
    pp.query("turb_conv_vel", t.turb_conv_vel);

    // The file holds the three velocity components
    t.ncomp = AMREX_SPACEDIM;
    read_header(t, name);

    const amrex::Box sbx(
      amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
      amrex::IntVect(AMREX_D_DECL(
        t.geom.npboxcells[0] + 2, t.geom.npboxcells[1] + 2, t.nplane - 1)));
    t.sdata = make_plane_chunk(sbx, t.ncomp);

    place_file(t, name, turb_center, has_turb_center);

    if (t.verbose > 0) {
      // Report the file's transverse spacing in case units so that it can
      // be compared against the target grid's spacing on the injection
      // face -- see PeleLMeX's turbInflow resolution check.  A mesh-mapped
      // file is uniform in Xi only, so give its physical range.
      amrex::Real dmin[2] = {0.0, 0.0};
      amrex::Real dmax[2] = {0.0, 0.0};
      transverse_dx_range(t, dmin, dmax);
      amrex::Print() << "   transverse spacing of " << name
                     << " in case units: ";
      for (int idim = 0; idim < 2; ++idim) {
        if (t.geom.has_map) {
          amrex::Print() << "[" << dmin[idim] << ", " << dmax[idim] << "]";
        } else {
          amrex::Print() << dmin[idim];
        }
        amrex::Print() << (idim == 0 ? " x " : "\n");
      }
    }
  }
  if (n_tp > 0) {
    turbinflow_initialized = true;
  }
}

std::shared_ptr<amrex::FArrayBox>
TurbInflow::make_plane_chunk(const amrex::Box& bx, const int ncomp)
{
  auto fab =
    std::make_shared<amrex::FArrayBox>(bx, ncomp, amrex::The_Async_Arena());
  // Runs before the arenas are torn down.  A weak reference, so that the
  // hook neither keeps the chunk alive nor touches it once its owner has
  // freed or replaced it.
  amrex::ExecOnFinalize([w = std::weak_ptr<amrex::FArrayBox>(fab)]() {
    if (auto f = w.lock()) {
      f->clear();
    }
  });
  return fab;
}

TurbFileSampler
TurbInflow::file_sampler(TurbParm& a_tp, const amrex::Real z)
{
  TurbFileSampler s;
  static_cast<TurbFileGeom&>(s) = a_tp.geom;
  prepare_planes(a_tp, z, s.cz, s.k0);
  s.sd = a_tp.sdata->const_array();
  s.generation = a_tp.chunk_generation;
  return s;
}

amrex::Real
TurbInflow::sampling_position(const TurbParm& a_tp, const amrex::Real time)
{
  if (a_tp.istimeplanes) {
    return time + a_tp.time_shift;
  }
  if (a_tp.convected_distance >= 0.0) {
    // Through-plane position supplied as a physical convected distance
    // (e.g. the time-integrated inlet velocity); turb_conv_vel is bypassed.
    return a_tp.convected_distance * a_tp.geom.scale_loc;
  }
  return (time + a_tp.time_shift) * a_tp.turb_conv_vel * a_tp.geom.scale_loc;
}

namespace {
bool
on_face(
  const TurbParm& tpn, const int dir, const amrex::Orientation::Side& side)
{
  return tpn.dir == dir && tpn.side == side;
}
} // namespace

int
TurbInflow::n_files(const int dir, const amrex::Orientation::Side& side) const
{
  return static_cast<int>(
    std::count_if(tp.begin(), tp.end(), [dir, side](const TurbParm& tpn) {
      return on_face(tpn, dir, side);
    }));
}

const TurbParm*
TurbInflow::find_turbparm(
  const int dir, const amrex::Orientation::Side& side, const int ifile) const
{
  int i = 0;
  for (const auto& tpn : tp) {
    if (on_face(tpn, dir, side) && i++ == ifile) {
      return &tpn;
    }
  }
  return nullptr;
}

bool
TurbInflow::file_has_map(
  const int dir, const amrex::Orientation::Side& side) const
{
  // Any of the files acting on the face
  return std::any_of(tp.begin(), tp.end(), [dir, side](const TurbParm& tpn) {
    return on_face(tpn, dir, side) && tpn.geom.has_map;
  });
}

void
TurbInflow::transverse_dx_range(
  const TurbParm& a_tp, amrex::Real dx_min[2], amrex::Real dx_max[2])
{
  // The file's lengths are in file units; queried coordinates are multiplied
  // by scale_loc before the lookup, so the equivalent spacing in case units
  // is dx / scale_loc.  A mesh-mapped file is uniform in Xi: difference the
  // physical face positions of each interior cell instead.
  const TurbFileGeom& g = a_tp.geom;
  for (int idim = 0; idim < 2; ++idim) {
    if (!g.has_map) {
      dx_min[idim] = g.dx[idim] / g.scale_loc;
      dx_max[idim] = dx_min[idim];
      continue;
    }
    amrex::Real dmin = std::numeric_limits<amrex::Real>::max();
    amrex::Real dmax = 0.0;
    for (int i = 0; i < g.npboxcells[idim]; ++i) {
      const amrex::Real d = g.map.dx_phys_cc(
        idim, i, g.map_xi_lo[idim], g.map_xi_hi[idim], g.dx[idim]);
      dmin = amrex::min(dmin, d);
      dmax = amrex::max(dmax, d);
    }
    dx_min[idim] = dmin / g.scale_loc;
    dx_max[idim] = dmax / g.scale_loc;
  }
}

bool
TurbInflow::file_transverse_dx(
  const int dir,
  const amrex::Orientation::Side& side,
  amrex::Real& dx_tdir1,
  amrex::Real& dx_tdir2,
  const int ifile) const
{
  const TurbParm* tpn = find_turbparm(dir, side, ifile);
  if (tpn == nullptr) {
    return false;
  }
  // In file units: the uniform spacing, or for a mesh-mapped file the mean
  // physical spacing, i.e. the file's physical transverse extent over its
  // interior cell count.  See transverse_dx_range() for the case units.
  const TurbFileGeom& g = tpn->geom;
  amrex::Real d[2];
  for (int idim = 0; idim < 2; ++idim) {
    d[idim] = g.has_map ? (g.x_phys(idim, g.map_xi_hi[idim]) -
                           g.x_phys(idim, g.map_xi_lo[idim])) /
                            static_cast<amrex::Real>(g.npboxcells[idim])
                        : g.dx[idim];
  }
  dx_tdir1 = d[0] / g.scale_loc;
  dx_tdir2 = d[1] / g.scale_loc;
  return true;
}

bool
TurbInflow::file_transverse_dx_range(
  const int dir,
  const amrex::Orientation::Side& side,
  amrex::Real dx_min[2],
  amrex::Real dx_max[2],
  const int ifile) const
{
  const TurbParm* tpn = find_turbparm(dir, side, ifile);
  if (tpn == nullptr) {
    return false;
  }
  transverse_dx_range(*tpn, dx_min, dx_max);
  return true;
}

int
TurbInflow::file_map(
  const int dir,
  const amrex::Orientation::Side& side,
  pele::physics::MeshMapEvaluator& map,
  amrex::Real xi_lo[2],
  amrex::Real xi_hi[2],
  const int ifile) const
{
  const TurbParm* tpn = find_turbparm(dir, side, ifile);
  if (tpn == nullptr) {
    return -1;
  }
  const TurbFileGeom& g = tpn->geom;
  if (!g.has_map) {
    return 0;
  }
  map = g.map;
  for (int idim = 0; idim < 2; ++idim) {
    xi_lo[idim] = g.map_xi_lo[idim];
    xi_hi[idim] = g.map_xi_hi[idim];
  }
  return 1;
}

namespace {
// Add scale times file components [0, ncomp) of s, at the given case-unit
// positions, to the components [0, ncomp) of dst over its box.  dst is a
// plane in the file's orientation (its first two directions are the file's
// transverse ones); x_phys and y_phys hold the positions of its cells along
// them.  Cells outside the file are left untouched.  (Free functions rather
// than private members: nvcc does not allow extended device lambdas in
// private member functions.)
void
sample_plane(
  const TurbFileSampler& s,
  const amrex::Vector<amrex::Real>& x_phys,
  const amrex::Vector<amrex::Real>& y_phys,
  amrex::FArrayBox& dst,
  const int ncomp,
  const amrex::Real scale)
{
  // File coordinates of the positions: scaled to file units and, for a
  // mesh-mapped file, inverted through the file's map into the precursor's
  // Xi coordinate, which is what indexes the file.
  amrex::Vector<amrex::Real> x(x_phys.size());
  amrex::Vector<amrex::Real> y(y_phys.size());
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[i] = s.file_coordinate(0, x_phys[i] * s.scale_loc);
  }
  for (int j = 0; j < static_cast<int>(y.size()); ++j) {
    y[j] = s.file_coordinate(1, y_phys[j] * s.scale_loc);
  }

  // Freed by a stream callback once the kernel below is done with them, so
  // no synchronization is needed before they go out of scope.  The kernel
  // reads the plane chunk asynchronously: read_turb_planes() synchronizes
  // before it overwrites the chunk.
  amrex::Gpu::AsyncArray<amrex::Real> x_dev(x.data(), x.size());
  amrex::Gpu::AsyncArray<amrex::Real> y_dev(y.data(), y.size());
  const amrex::Real* xd = x_dev.data();
  const amrex::Real* yd = y_dev.data();

  // The coordinates span the plane (add_turb), so empty coordinate vectors
  // mean an empty plane box with nothing to fill, and null device copies
  if (xd == nullptr || yd == nullptr) {
    return;
  }

  const auto& bx = dst.box();
  const auto& da = dst.array();
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    TurbFileSampler::Stencil st;
    if (s.locate_file(xd[i - bx.smallEnd(0)], yd[j - bx.smallEnd(1)], st)) {
      for (int n = 0; n < ncomp; ++n) {
        da(i, j, k, n) += scale * s.component(st, n);
      }
    }
  });
}

// Copy the velocity plane v (file orientation) into data, from component
// dcomp, in the solver's orientation
void
set_turb(
  int normDir,
  int transDir1,
  int transDir2,
  amrex::FArrayBox& v,
  amrex::FArrayBox& data,
  const int dcomp)
{
  // copy velocity fluctuations from plane into data
  const auto& box = v.box(); // z-normal plane
  const auto& v_in = v.array();
  const auto& v_out = data.array(dcomp);

  amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    // From z-normal box index to data box index
    amrex::GpuArray<int, 3> idx{};
    idx[transDir1] = i;
    idx[transDir2] = j;
    idx[normDir] = k;
    v_out(idx[0], idx[1], idx[2], transDir1) =
      v_in(i, j, k, 0); // transverse velocity 1
    v_out(idx[0], idx[1], idx[2], transDir2) =
      v_in(i, j, k, 1); // transverse velocity 2
    v_out(idx[0], idx[1], idx[2], normDir) =
      v_in(i, j, k, 2); // normal velocity
  });
}
} // namespace

void
TurbInflow::add_turb(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const int dir,
  const amrex::Orientation::Side& side)
{
  AMREX_ALWAYS_ASSERT(turbinflow_initialized);

  // Uniform grid: build the cell-centre positions and defer to the
  // coordinate-based overload, which holds the single implementation.
  int tdir1 = 0;
  int tdir2 = 0;
  transverseDirs(dir, tdir1, tdir2);

  amrex::Vector<amrex::Real> x(bx.length(tdir1));
  amrex::Vector<amrex::Real> y(bx.length(tdir2));
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[i] = geom.CellCenter(bx.smallEnd(tdir1) + i, tdir1);
  }
  for (int j = 0; j < static_cast<int>(y.size()); ++j) {
    y[j] = geom.CellCenter(bx.smallEnd(tdir2) + j, tdir2);
  }

  add_turb(bx, data, dcomp, geom.Domain(), x, y, time, dir, side);
}

void
TurbInflow::add_turb(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  amrex::Box const& domain,
  const amrex::Vector<amrex::Real>& x_phys,
  const amrex::Vector<amrex::Real>& y_phys,
  const amrex::Real time,
  const int dir,
  const amrex::Orientation::Side& side)
{
  AMREX_ALWAYS_ASSERT(turbinflow_initialized);

  // The ghost plane being filled
  const int planeLoc =
    (side == amrex::Orientation::low ? domain.smallEnd(dir) - 1
                                     : domain.bigEnd(dir) + 1);

  int tdir1 = 0;
  int tdir2 = 0;
  transverseDirs(dir, tdir1, tdir2);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    static_cast<int>(x_phys.size()) == bx.length(tdir1) &&
      static_cast<int>(y_phys.size()) == bx.length(tdir2),
    "TurbInflow::add_turb(): supplied coordinate vectors must span the "
    "transverse extents of bx");

  // The plane in the file's orientation: transverse directions first
  // and the normal one last
  amrex::Box turbBox(
    amrex::IntVect(AMREX_D_DECL(bx.smallEnd(tdir1), bx.smallEnd(tdir2), 0)),
    amrex::IntVect(AMREX_D_DECL(bx.bigEnd(tdir1), bx.bigEnd(tdir2), 0)));
  turbBox.setRange(AMREX_SPACEDIM - 1, planeLoc);
  amrex::FArrayBox v(turbBox, AMREX_SPACEDIM, amrex::The_Async_Arena());
  v.setVal<amrex::RunOn::Device>(0);

  // Add turbulence from all the tp acting on this face
  for (auto& tpn : tp) {
    if (!on_face(tpn, dir, side)) {
      continue;
    }
    const amrex::Real velScale = (side == amrex::Orientation::high)
                                   ? -tpn.turb_scale_vel
                                   : tpn.turb_scale_vel;
    sample_plane(
      file_sampler(tpn, sampling_position(tpn, time)), x_phys, y_phys, v,
      AMREX_SPACEDIM, velScale);
  }

  // Moving it into data
  set_turb(dir, tdir1, tdir2, v, data, dcomp);
}

void
TurbInflow::read_turb_planes(TurbParm& a_tp, const amrex::Real z)
{
  if (a_tp.istimeplanes) {
    if (z < a_tp.planeTimes[0] || z >= a_tp.planeTimes[a_tp.kmax - 2]) {
      amrex::Error(
        "TurbInflow::read_turb_planes(): Requested time (" + std::to_string(z) +
        ") is outside bounds of turbulence data [" +
        std::to_string(a_tp.planeTimes[0]) + ", " +
        std::to_string(a_tp.planeTimes[a_tp.kmax - 2]) +
        ")"); // Need one turbplane forward for interpolation
    }
    for (a_tp.izlo = 0; (a_tp.izlo <= (a_tp.kmax - a_tp.nplane + 1)) &&
                        (a_tp.planeTimes[a_tp.izlo] <= z);
         ++a_tp.izlo) {
    } // Stop when first plane later than time=z
    a_tp.izlo -= 2; // read one extra earlier tplane to prevent rereading
    a_tp.izlo = amrex::max<int>(a_tp.izlo, 0);
    a_tp.izhi = a_tp.izlo + a_tp.nplane - 1;
    a_tp.szlo = a_tp.planeTimes[a_tp.izlo];
    a_tp.szhi = a_tp.planeTimes[a_tp.izhi - 1]; // need one plane forward in
                                                // time for interpolating
  } else {
    a_tp.izlo = static_cast<int>(std::floor(z * a_tp.dzinv - 0.5)) -
                1; // read one extra earlier tplane to prevent rereading
    a_tp.izhi = a_tp.izlo + a_tp.nplane - 1;
    a_tp.szlo = (static_cast<amrex::Real>(a_tp.izlo) + 0.5) * a_tp.dz;
    a_tp.szhi = (static_cast<amrex::Real>(a_tp.izhi) - 0.5) *
                a_tp.dz; // need one plane forward in time for interpolating
  }
  if (a_tp.verbose > 1) {
    amrex::Print() << "read_turb_planes filling " << a_tp.izlo << " to "
                   << a_tp.izhi << std::endl
                   << " --> now have interp data for range [" << a_tp.szlo
                   << ", " << a_tp.szhi << ") with current "
                   << (a_tp.istimeplanes ? "t" : "z") << " = " << z
                   << std::endl;
  }

  // DAT holds kmax planes of component 0, then kmax planes of component 1,
  // and so on; a_tp.offset gives where each starts.  Read the chunk on the
  // host, then move it to the device in one copy.
  const std::string turb_data = a_tp.m_turb_file + "/DAT";
  std::ifstream ifs(turb_data.c_str());
  if (!ifs.is_open()) {
    amrex::Abort("Unable to open input file " + turb_data);
  }
#ifdef AMREX_USE_GPU
  amrex::FArrayBox host(
    a_tp.sdata->box(), a_tp.ncomp, amrex::The_Pinned_Arena());
#else
  amrex::FArrayBox& host = *a_tp.sdata;
#endif

  for (int iplane = 0; iplane < a_tp.nplane; ++iplane) {
    int k = a_tp.izlo + iplane;
    if (!a_tp.istimeplanes) {
      k = (a_tp.kmax + k) % a_tp.kmax; // "wrap" planes if data is periodic
    }
    amrex::Box dstBox = host.box();
    dstBox.setRange(AMREX_SPACEDIM - 1, iplane);
    for (int n = 0; n < a_tp.ncomp; ++n) {
      const long offset_idx = k + (static_cast<long>(n) * a_tp.kmax);
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        offset_idx < a_tp.offset.size(), "Bad turb fab offset idx");
      ifs.seekg(a_tp.offset[offset_idx], std::ios::beg);
      if (!ifs.good()) {
        amrex::Abort(
          "TurbInflow::read_turb_planes(): seekg() failed in " + turb_data);
      }

      amrex::FArrayBox tmp(amrex::The_Pinned_Arena());
      tmp.readFrom(ifs);
      if (a_tp.verbose > 2) {
        amrex::Print() << "   for d = " << n << " and k = " << k
                       << ": minval = " << tmp.min<amrex::RunOn::Host>(0)
                       << ", maxval = " << tmp.max<amrex::RunOn::Host>(0)
                       << std::endl;
      }
      host.copy<amrex::RunOn::Host>(tmp, tmp.box(), 0, dstBox, n, 1);
    }
  }

  // Kernels launched by earlier add_turb() calls may still be reading the
  // chunk, possibly on other streams (callers loop over boxes inside an
  // MFIter, which switches streams); let them finish before overwriting it.
  amrex::Gpu::synchronize();
#ifdef AMREX_USE_GPU
  amrex::Gpu::htod_memcpy_async(
    a_tp.sdata->dataPtr(), host.dataPtr(), host.nBytes());
#endif
  a_tp.chunk_generation = ++last_chunk_generation;
  // The chunk may next be read from another stream, and the host buffer
  // must outlive the copy.
  amrex::Gpu::streamSynchronize();
}

void
TurbInflow::prepare_planes(
  TurbParm& a_tp, amrex::Real z, amrex::GpuArray<amrex::Real, 3>& cz, int& k0)
{
  // Wrapped once, here, so that the chunk loaded and the stencil computed
  // below use the same z
  z = wrap_plane_time(a_tp, z);

  if ((z < a_tp.szlo) || (z >= a_tp.szhi)) {
    if (a_tp.verbose > 1) {
      amrex::Print() << "Reading new data because "
                     << (a_tp.istimeplanes ? "t" : "z") << " = " << z
                     << " is outside the range [" << a_tp.szlo << ", "
                     << a_tp.szhi << ")" << std::endl;
    }
    read_turb_planes(a_tp, z);
  }

  k0 = -1;
  if (a_tp.istimeplanes) {
    AMREX_ALWAYS_ASSERT(
      z >= a_tp.planeTimes[a_tp.izlo] && z <= a_tp.planeTimes[a_tp.izhi]);
    for (k0 = 1; k0 < a_tp.nplane - 2 && a_tp.planeTimes[a_tp.izlo + k0] <= z;
         ++k0) {
    } // Stop when first plane later than time=z, then go back one
    k0 -= 1;
    const auto& t0 = a_tp.planeTimes[a_tp.izlo + k0];
    const auto& t1 = a_tp.planeTimes[a_tp.izlo + k0 + 1];
    const auto& t2 = a_tp.planeTimes[a_tp.izlo + k0 + 2];
    AMREX_ALWAYS_ASSERT(z >= t0 && z <= t2);
    cz[0] = (z - t1) * (z - t2) / ((t0 - t1) * (t0 - t2));
    cz[1] = (z - t0) * (z - t2) / ((t1 - t0) * (t1 - t2));
    cz[2] = (z - t0) * (z - t1) / ((t2 - t0) * (t2 - t1));
  } else {
    // How many dz away from the chunk's first plane
    amrex::Real zz = (z - a_tp.szlo) * a_tp.dzinv;
    k0 = static_cast<int>(std::floor(zz));
    stencil_weights(zz - static_cast<amrex::Real>(k0), a_tp.geom.lininterp, cz);
  }
}

} // namespace pele::physics::turbinflow
