#include <turbinflow.H>

#include <algorithm>

namespace pele::physics::turbinflow {
void
TurbInflow::init(amrex::Geometry const& /*geom*/)
{
  amrex::ParmParse ppr;

  int n_tp = 0;
  n_tp = ppr.countval("turbinflows");
  amrex::Vector<std::string> tp_list;
  if (n_tp > 0) {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      AMREX_SPACEDIM == 3, "TurbInflow::init(): TurbInflows are only supported "
                           "for 3 dimensional simulations for now");
    tp.resize(n_tp);
    tp_list.resize(n_tp);
    for (int n = 0; n < n_tp; n++) {
      ppr.get("turbinflows", tp_list[n], n);
    }
  }

  for (int n = 0; n < n_tp; n++) {

    amrex::ParmParse pp("turbinflow." + tp_list[n]);
    if (pp.countval("turb_file") > 0) {

      // Query data
      pp.query("turb_file", tp[n].m_turb_file);
      tp[n].dir = -1;
      pp.query("dir", tp[n].dir);
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        tp[n].dir >= 0 && tp[n].dir < AMREX_SPACEDIM,
        "Injection direction is needed: 0, 1 or 2");
      std::string side;
      pp.query("side", side);
      if (side == "low") {
        tp[n].side = amrex::Orientation::low;
      } else if (side == "high") {
        tp[n].side = amrex::Orientation::high;
      } else {
        amrex::Abort("turbinflow.side can only be low or high");
      }
      pp.query("time_offset", tp[n].time_shift);
      pp.query("turb_scale_loc", tp[n].turb_scale_loc);
      pp.query("turb_scale_vel", tp[n].turb_scale_vel);
      pp.query("verbose", tp[n].verbose);
      pp.query("extrap_nonperiodic", tp[n].extrap_nonperiodic);
      pp.query("tile_periodic", tp[n].tile_periodic);
      pp.query("time_periodic", tp[n].time_periodic);
      pp.query("interp_type", tp[n].interp_type);
      if (tp[n].verbose > 0) {
        amrex::Print() << "Initializing turbInflow " << tp_list[n]
                       << " with file " << tp[n].m_turb_file
                       << " (location coordinates in will be scaled by "
                       << tp[n].turb_scale_loc
                       << " and velocity out to be scaled by "
                       << tp[n].turb_scale_vel << ") \n";
      }

      // Get the turbcenter on the injection face.  Required for a file that
      // is uniform in physical position; optional for a mesh-mapped file
      // (see the MESHMAP trailer below), whose default centre follows from
      // the trailer.
      amrex::Vector<amrex::Real> turb_center(AMREX_SPACEDIM - 1, 0);
      const bool has_turb_center = pp.queryarr("turb_center", turb_center) != 0;
      if (has_turb_center) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          turb_center.size() == AMREX_SPACEDIM - 1,
          "turb_center must have AMREX_SPACEDIM-1 elements");
      }

      pp.query("turb_nplane", tp[n].nplane);
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        tp[n].nplane > 3, "need at least 4 turb planes for 3 point "
                          "interpolation stencil + 1 extra");
      pp.query("turb_conv_vel", tp[n].turb_conv_vel);

      // Set other stuff
      std::string turb_header = tp[n].m_turb_file + "/HDR";
      std::ifstream is(turb_header.c_str());
      if (!is.is_open()) {
        amrex::Abort("Unable to open input file " + turb_header);
      }
      amrex::Array<int, AMREX_SPACEDIM> npts = {{0}};
      amrex::Array<amrex::Real, AMREX_SPACEDIM> probsize = {{0}};

      AMREX_D_TERM(is >> npts[0], >> npts[1], >> npts[2]);
      AMREX_D_TERM(is >> probsize[0], >> probsize[1], >> probsize[2]);
      AMREX_D_TERM(
        is >> tp[n].periodicity[0], >> tp[n].periodicity[1],
        >> tp[n].periodicity[2]); // Will use zperiodicity to single whether
                                  // using periodic or time per plane mode

      tp[n].istimeplanes = AMREX_D_PICK(, false, tp[n].periodicity[2] == 0);
      if (tp[n].periodicity[0] == 0 || tp[n].periodicity[1] == 0) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          tp[n].interp_type == TurbInterpType::linear,
          "Linear interpolation required for turbinflow with nonperiodic "
          "directions");
      }

      for (int idim = 0; idim < 2; ++idim) {
        tp[n].dx[idim] = probsize[idim] / amrex::Real(npts[idim] - 1);
        tp[n].dxinv[idim] = 1.0 / tp[n].dx[idim];
      }
      AMREX_D_TERM(, , tp[n].dx[2] = probsize[2] / amrex::Real(npts[2]);)
      AMREX_D_TERM(, , tp[n].dxinv[2] = 1.0 / tp[n].dx[2];)

      // The following is relative to the injection face:
      // 0 and 1 are transverse directions, 2 is normal
      // one ghost point on each side, tangential to inflow face
      AMREX_D_TERM(
        tp[n].pboxsize[0] = probsize[0] - 2.0 * tp[n].dx[0];
        , tp[n].pboxsize[1] = probsize[1] - 2.0 * tp[n].dx[1];
        , tp[n].pboxsize[2] = probsize[2];)

      AMREX_D_TERM(
        tp[n].npboxcells[0] = npts[0] - 3;, tp[n].npboxcells[1] = npts[1] - 3;
        , tp[n].npboxcells[2] = npts[2];)

      if (tp[n].turb_scale_loc == 0.0) {
        amrex::Abort(
          "TurbInflow::init(): turb_scale_loc must be non-zero for " +
          tp_list[n]);
      }

      // Swirl type: we can't load more planes than are available
      if (tp[n].istimeplanes) {
        tp[n].nplane = AMREX_D_PICK(
          tp[n].nplane, tp[n].nplane, amrex::min<int>(tp[n].nplane, npts[2]));
      }

      amrex::Box sbx(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(
          AMREX_D_DECL(npts[0] - 1, npts[1] - 1, tp[n].nplane - 1)));

      tp[n].sdata = new amrex::FArrayBox(sbx, 3, amrex::The_Async_Arena());

      AMREX_D_TERM(, , tp[n].kmax = npts[2];)

      // Offset for each plane in Binary TurbFile
      tp[n].offset.resize(tp[n].kmax * AMREX_SPACEDIM);
      for (auto& off : tp[n].offset) {
        is >> off;
      }

      if (tp[n].istimeplanes) {
        tp[n].planeTimes.resize(tp[n].kmax);
        for (int i = 0; i < tp[n].kmax; i++) {
          is >> tp[n].planeTimes[i]; // Time for each plane
        }
      }

      // Optional trailer.  Everything a legacy reader consumes ends with the
      // plane times, so anything appended after them is invisible to older
      // code and a file without it is, by declaration, uniform in physical
      // position.  Format (one line per transverse direction, in the file's
      // own transverse order):
      //
      //   MESHMAP_V2
      //   <kind> <p> <p2> <p3> <q> <xi_lo> <xi_hi>
      //   <kind> <p> <p2> <p3> <q> <xi_lo> <xi_hi>
      //
      // kind/p/p2/p3/q are the MeshMapEvaluator payload for that axis
      // (0 identity, 1 constant, 2 exp stretch, 3 tanh stretch, 4 interior
      // stretch); xi_lo/xi_hi are the precursor's computational-domain
      // bounds along that axis, which the inverse map needs.  The earlier
      // MESHMAP_V1 form, <kind> <p> <q> <xi_lo> <xi_hi>, is read with
      // p2 = p3 = 0.  A trailer marks the file as uniform in the precursor's
      // Xi coordinate rather than in physical position; add_turb() inverts
      // the map before indexing the file.
      std::string token;
      if ((is >> token) && (token == "MESHMAP_V1" || token == "MESHMAP_V2")) {
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
          is >> tp[n].map_xi_lo[idim] >> tp[n].map_xi_hi[idim];
          tp[n].map.m_p[idim] = p;
          tp[n].map.m_p2[idim] = p2;
          tp[n].map.m_p3[idim] = p3;
          tp[n].map.m_q[idim] = q;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          is.good(), "TurbInflow::init(): malformed " + token + " trailer in " +
                       turb_header);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          kind[0] == kind[1],
          "TurbInflow::init(): " + token + " trailer in " + turb_header +
            " names two different map kinds; a precursor has one map");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          kind[0] >= static_cast<int>(MeshMapEvaluator::Kind::Identity) &&
            kind[0] <=
              static_cast<int>(MeshMapEvaluator::Kind::InteriorStretch),
          "TurbInflow::init(): unknown map kind in " + token + " trailer of " +
            turb_header);
        for (int idim = 0; idim < 2; ++idim) {
          AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            tp[n].map_xi_hi[idim] > tp[n].map_xi_lo[idim],
            "TurbInflow::init(): " + token + " trailer in " + turb_header +
              " has xi_hi <= xi_lo");
        }
        tp[n].map.m_kind = static_cast<MeshMapEvaluator::Kind>(kind[0]);
        tp[n].has_map = true;
        // The trailer's Xi bounds are exact while the header's probsize may
        // have been written with limited precision; derive the file's Xi
        // spacing from the trailer so that the inverse map and the index
        // lookup agree to round-off, after checking the two are consistent.
        for (int idim = 0; idim < 2; ++idim) {
          const amrex::Real L = tp[n].map_xi_hi[idim] - tp[n].map_xi_lo[idim];
          const amrex::Real dx_tr =
            L / static_cast<amrex::Real>(tp[n].npboxcells[idim]);
          AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(dx_tr - tp[n].dx[idim]) <= 1.0e-5 * tp[n].dx[idim],
            "TurbInflow::init(): " + token + " trailer of " + turb_header +
              " gives a Xi extent inconsistent with the header's probsize");
          tp[n].dx[idim] = dx_tr;
          tp[n].dxinv[idim] = 1.0 / dx_tr;
          tp[n].pboxsize[idim] = L;
        }
        if (tp[n].verbose > 0) {
          amrex::Print() << "   " << tp_list[n] << " carries a " << token
                         << " trailer: file is uniform in the precursor's Xi "
                            "coordinate (map kind "
                         << kind[0] << ")\n";
        }
      } else if (!token.empty() && !is.eof()) {
        amrex::Print() << "TurbInflow: WARNING ignoring unrecognised trailing "
                          "content in "
                       << turb_header << " starting at '" << token << "'\n";
      }

      // Center the turbulence.  For a physically-uniform file turb_center
      // is in case units and the file is placed around it.  For a
      // mesh-mapped file the sampling coordinate is the precursor's Xi
      // (add_turb() inverts the map), so the natural placement is the
      // precursor's own: pboxlo = xi_lo, i.e. turb_center = (xi_lo+xi_hi)/2
      // in the file's Xi units, and that is the default; a supplied
      // turb_center is taken in those same Xi units (unscaled) and shifts
      // the pattern within the plane.
      if (tp[n].has_map) {
        for (int idim = 0; idim < 2; ++idim) {
          const amrex::Real xi_mid =
            0.5 * (tp[n].map_xi_lo[idim] + tp[n].map_xi_hi[idim]);
          if (!has_turb_center) {
            turb_center[idim] = xi_mid;
          } else {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
              turb_center[idim] >= tp[n].map_xi_lo[idim] &&
                turb_center[idim] <= tp[n].map_xi_hi[idim],
              "TurbInflow::init(): turb_center for a mesh-mapped turbulence "
              "file is in the file's Xi units and must lie within "
              "[xi_lo, xi_hi] of its MESHMAP trailer");
          }
        }
        if (tp[n].verbose > 0) {
          amrex::Print() << "   turb_center of " << tp_list[n]
                         << " in file Xi units: " << turb_center[0] << " "
                         << turb_center[1]
                         << (has_turb_center ? "" : " (default)")
                         << ", physical (case units): "
                         << tp[n].map.x_phys_from_xi(
                              0, turb_center[0], tp[n].map_xi_lo[0],
                              tp[n].map_xi_hi[0]) /
                              tp[n].turb_scale_loc
                         << " "
                         << tp[n].map.x_phys_from_xi(
                              1, turb_center[1], tp[n].map_xi_lo[1],
                              tp[n].map_xi_hi[1]) /
                              tp[n].turb_scale_loc
                         << "\n";
        }
      } else {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
          has_turb_center,
          "turbinflow." + tp_list[n] +
            ".turb_center is required for a turbulence file that is uniform "
            "in physical position (no MESHMAP trailer)");
        for (amrex::Real& tc : turb_center) {
          tc *= tp[n].turb_scale_loc;
        }
      }
      AMREX_D_TERM(
        tp[n].pboxlo[0] = turb_center[0] - 0.5 * tp[n].pboxsize[0];
        , tp[n].pboxlo[1] = turb_center[1] - 0.5 * tp[n].pboxsize[1];
        , tp[n].pboxlo[2] = 0.0;)

      if (tp[n].verbose > 0) {
        // Report the file's transverse spacing in case units so that it can
        // be compared against the target grid's spacing on the injection
        // face -- see PeleLMeX's turbInflow resolution check.  A
        // mesh-mapped file is uniform in Xi only, so give its physical
        // range.
        amrex::Real dmin[2] = {0.0, 0.0};
        amrex::Real dmax[2] = {0.0, 0.0};
        transverse_dx_range(tp[n], dmin, dmax);
        amrex::Print() << "   transverse spacing of " << tp_list[n]
                       << " in case units: ";
        for (int idim = 0; idim < 2; ++idim) {
          if (tp[n].has_map) {
            amrex::Print() << "[" << dmin[idim] << ", " << dmax[idim] << "]";
          } else {
            amrex::Print() << dmin[idim];
          }
          amrex::Print() << (idim == 0 ? " x " : "\n");
        }
      }
      is.close();
    }
    turbinflow_initialized = true;
  }
}

bool
TurbInflow::file_has_map(
  const int dir, const amrex::Orientation::Side& side) const
{
  bool found = false;
  bool any_has_map = false;
  for (const auto& tpn : tp) {
    if (tpn.dir == dir && tpn.side == side) {
      found = true;
      any_has_map = any_has_map || tpn.has_map;
    }
  }
  return found ? any_has_map : false;
}

void
TurbInflow::transverse_dx_range(
  const TurbParm& a_tp, amrex::Real dx_min[2], amrex::Real dx_max[2])
{
  // tp.dx lives in turb-file units; queried coordinates are multiplied by
  // turb_scale_loc before the lookup, so the equivalent spacing in case
  // units is dx / turb_scale_loc.  A mesh-mapped file is uniform in Xi:
  // difference the physical face positions of each interior cell instead.
  for (int idim = 0; idim < 2; ++idim) {
    if (!a_tp.has_map) {
      dx_min[idim] = a_tp.dx[idim] / a_tp.turb_scale_loc;
      dx_max[idim] = dx_min[idim];
      continue;
    }
    amrex::Real dmin = std::numeric_limits<amrex::Real>::max();
    amrex::Real dmax = 0.0;
    for (int i = 0; i < a_tp.npboxcells[idim]; ++i) {
      const amrex::Real d = a_tp.map.dx_phys_cc(
        idim, i, a_tp.map_xi_lo[idim], a_tp.map_xi_hi[idim], a_tp.dx[idim]);
      dmin = amrex::min(dmin, d);
      dmax = amrex::max(dmax, d);
    }
    dx_min[idim] = dmin / a_tp.turb_scale_loc;
    dx_max[idim] = dmax / a_tp.turb_scale_loc;
  }
}

bool
TurbInflow::file_transverse_dx(
  const int dir,
  const amrex::Orientation::Side& side,
  amrex::Real& dx_tdir1,
  amrex::Real& dx_tdir2) const
{
  const TurbParm* tpn = find_turbparm(dir, side);
  if (tpn == nullptr) {
    return false;
  }
  if (!tpn->has_map) {
    // tp.dx lives in turb-file units; queried coordinates are multiplied
    // by turb_scale_loc before the lookup, so the equivalent spacing in
    // case units is dx / turb_scale_loc.
    dx_tdir1 = tpn->dx[0] / tpn->turb_scale_loc;
    dx_tdir2 = tpn->dx[1] / tpn->turb_scale_loc;
    return true;
  }
  // Mean physical spacing: the file's physical transverse extent over its
  // interior cell count.
  amrex::Real mean[2];
  for (int idim = 0; idim < 2; ++idim) {
    const amrex::Real xlo = tpn->map.x_phys_from_xi(
      idim, tpn->map_xi_lo[idim], tpn->map_xi_lo[idim], tpn->map_xi_hi[idim]);
    const amrex::Real xhi = tpn->map.x_phys_from_xi(
      idim, tpn->map_xi_hi[idim], tpn->map_xi_lo[idim], tpn->map_xi_hi[idim]);
    mean[idim] = (xhi - xlo) / static_cast<amrex::Real>(tpn->npboxcells[idim]) /
                 tpn->turb_scale_loc;
  }
  dx_tdir1 = mean[0];
  dx_tdir2 = mean[1];
  return true;
}

const TurbParm*
TurbInflow::find_turbparm(
  const int dir, const amrex::Orientation::Side& side) const
{
  auto it =
    std::find_if(tp.begin(), tp.end(), [dir, side](const TurbParm& tpn) {
      return tpn.dir == dir && tpn.side == side;
    });
  return (it != tp.end()) ? &(*it) : nullptr;
}

bool
TurbInflow::file_transverse_dx_range(
  const int dir,
  const amrex::Orientation::Side& side,
  amrex::Real dx_min[2],
  amrex::Real dx_max[2]) const
{
  const TurbParm* tpn = find_turbparm(dir, side);
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
  amrex::Real xi_hi[2]) const
{
  const TurbParm* tpn = find_turbparm(dir, side);
  if (tpn == nullptr) {
    return -1;
  }
  if (!tpn->has_map) {
    return 0;
  }
  map = tpn->map;
  xi_lo[0] = tpn->map_xi_lo[0];
  xi_lo[1] = tpn->map_xi_lo[1];
  xi_hi[0] = tpn->map_xi_hi[0];
  xi_hi[1] = tpn->map_xi_hi[1];
  return 1;
}

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

  // Uniform grid: cell-centre positions are affine in the index.  Build
  // them and defer to the coordinate-based overload so that there is only
  // one copy of the interpolation logic.
  int tdir1 = 0;
  int tdir2 = 0;
  transverseDirs(dir, tdir1, tdir2);

  amrex::Vector<amrex::Real> x(bx.length(tdir1));
  amrex::Vector<amrex::Real> y(bx.length(tdir2));
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    const int idx = bx.smallEnd(tdir1) + i;
    x[i] = geom.ProbLo()[tdir1] +
           (static_cast<amrex::Real>(idx) + 0.5) * geom.CellSize(tdir1);
  }
  for (int j = 0; j < static_cast<int>(y.size()); ++j) {
    const int idx = bx.smallEnd(tdir2) + j;
    y[j] = geom.ProbLo()[tdir2] +
           (static_cast<amrex::Real>(idx) + 0.5) * geom.CellSize(tdir2);
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

  // Box on which we will access data
  amrex::Box bvalsBox = bx;
  int planeLoc =
    (side == amrex::Orientation::low ? domain.smallEnd()[dir] - 1
                                     : domain.bigEnd()[dir] + 1);
  bvalsBox.setSmall(dir, planeLoc);
  bvalsBox.setBig(dir, planeLoc);

  // Define box that we will fill with turb: need to be z-normal
  // Get transverse directions
  int tdir1 = 0;
  int tdir2 = 0;
  transverseDirs(dir, tdir1, tdir2);
  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    static_cast<int>(x_phys.size()) == bvalsBox.length(tdir1) &&
      static_cast<int>(y_phys.size()) == bvalsBox.length(tdir2),
    "TurbInflow::add_turb(): supplied coordinate vectors must span the "
    "transverse extents of bx");
  int tr1Lo = bvalsBox.smallEnd()[tdir1];
  int tr1Hi = bvalsBox.bigEnd()[tdir1];
  int tr2Lo = bvalsBox.smallEnd()[tdir2];
  int tr2Hi = bvalsBox.bigEnd()[tdir2];
  const amrex::IntVect lo(AMREX_D_DECL(tr1Lo, tr2Lo, planeLoc));
  const amrex::IntVect hi(AMREX_D_DECL(tr1Hi, tr2Hi, planeLoc));
  amrex::Box turbBox(lo, hi);
  amrex::FArrayBox v(turbBox, 3, amrex::The_Async_Arena());
  v.setVal<amrex::RunOn::Device>(0);

  // Add turbulence from all the tp acting on this face
  for (auto& tpn : tp) {

    if (tpn.dir == dir && tpn.side == side) {

      // 0 and 1 are the two transverse directions.  turb_scale_loc is a
      // per-TurbParm quantity, so the scaling is applied here rather than
      // once by the caller.  For a mesh-mapped file the scaled physical
      // position is then inverted through the file's map into the
      // precursor's Xi coordinate, which is what indexes the file.
      amrex::Vector<amrex::Real> x(turbBox.size()[0]), y(turbBox.size()[1]);
      for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        x[i] = file_coordinate(tpn, 0, x_phys[i] * tpn.turb_scale_loc);
      }
      for (int j = 0; j < static_cast<int>(y.size()); ++j) {
        y[j] = file_coordinate(tpn, 1, y_phys[j] * tpn.turb_scale_loc);
      }

      // Get the turbulence
      amrex::Real z;
      if (tpn.istimeplanes) {
        z = time + tpn.time_shift;
      } else if (tpn.convected_distance >= 0.0) {
        // Through-plane position supplied as a physical convected distance
        // (e.g. the time-integrated inlet velocity); turb_conv_vel is bypassed.
        z = tpn.convected_distance * tpn.turb_scale_loc;
      } else {
        z = (time + tpn.time_shift) * tpn.turb_conv_vel * tpn.turb_scale_loc;
      }
      fill_turb_plane(tpn, x, y, z, v);
    }
  }

  // Moving it into data
  set_turb(dir, tdir1, tdir2, v, data, dcomp);
}

amrex::Real
TurbInflow::file_coordinate(
  const TurbParm& a_tp, int idim, amrex::Real x_file_phys)
{
  if (!a_tp.has_map) {
    return x_file_phys;
  }
  const amrex::Real xi_lo = a_tp.map_xi_lo[idim];
  const amrex::Real xi_hi = a_tp.map_xi_hi[idim];
  const amrex::Real x_lo = a_tp.map.x_phys_from_xi(idim, xi_lo, xi_lo, xi_hi);
  const amrex::Real x_hi = a_tp.map.x_phys_from_xi(idim, xi_hi, xi_lo, xi_hi);

  amrex::Real x = x_file_phys;
  if (a_tp.tile_periodic && a_tp.periodicity[idim] != 0) {
    // Tiled: bring x into the file's physical period before inverting.
    // The inverse clamps to [x_lo, x_hi], which would otherwise collapse
    // every tiled copy onto the edge of the file.
    const amrex::Real period = x_hi - x_lo;
    x -= std::floor((x - x_lo) / period) * period;
  } else if (x < x_lo || x >= x_hi) {
    // Outside the file's physical extent.  A physically-uniform file leaves
    // such positions unfilled (fill_turb_plane's range test); keep that
    // behaviour rather than let the clamped inverse pin them to the edge
    // row, by returning a Xi coordinate that is outside the box whatever
    // turb_center shifted pboxlo to (|shift| <= pboxsize/2).
    const amrex::Real away = a_tp.pboxsize[idim];
    return (x < x_lo) ? xi_lo - away : xi_hi + away;
  }

  amrex::Real xi = a_tp.map.xi_from_x_phys(idim, x, xi_lo, xi_hi);

  // When the target's map and Xi grid coincide with the file's, xi lands on
  // a file cell centre up to the round-off of the inverse (log/atanh, or
  // the Newton tolerance for InteriorStretch).  Snap it so that the
  // interpolation weights collapse exactly and the same-map case
  // reproduces the file bit for bit.
  const amrex::Real idx = (xi - a_tp.pboxlo[idim]) * a_tp.dxinv[idim] + 0.5;
  const amrex::Real idx_round = std::round(idx);
  if (std::abs(idx - idx_round) < amrex::Real(1.0e-8)) {
    xi = a_tp.pboxlo[idim] + (idx_round - 0.5) * a_tp.dx[idim];
  }
  return xi;
}

void
TurbInflow::set_turb(
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
    int idx[3] = {0};
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

void
TurbInflow::read_one_turb_plane(TurbParm& a_tp, int iplane, int k)
{
  // There are AMREX_SPACEDIM * kmax planes of FABs.
  // The first component are in the first kmax planes,
  // the second component in the next kmax planes, ....
  // Note also that both (*plane) and (*ncomp) start from
  // 1 not 0 since they're passed from Fortran.

  std::string turb_data = a_tp.m_turb_file + "/DAT";
  std::ifstream ifs(turb_data.c_str());
  if (!ifs.is_open()) {
    amrex::Abort("Unable to open input file " + turb_data);
  }

  amrex::Box dstBox = a_tp.sdata->box();
  dstBox.setSmall(AMREX_SPACEDIM - 1, iplane);
  dstBox.setBig(AMREX_SPACEDIM - 1, iplane);

  for (int n = 0; n < AMREX_SPACEDIM; ++n) {

    const long offset_idx = k + (n * a_tp.kmax);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
      offset_idx < a_tp.offset.size(), "Bad turb fab offset idx");

    const long start = a_tp.offset[offset_idx];
    ifs.seekg(start, std::ios::beg);

    if (!ifs.good()) {
      amrex::Abort("getplane(): seekg() failed");
    }

    amrex::FArrayBox tmp;
    tmp.readFrom(ifs);
    if (a_tp.verbose > 2) {
      amrex::Print() << "   for d = " << n << " and k = " << k
                     << ": minval = " << tmp.min<amrex::RunOn::Device>(0)
                     << ", maxval = " << tmp.max<amrex::RunOn::Device>(0)
                     << std::endl;
    }
    amrex::Box srcBox = tmp.box();
    a_tp.sdata->copy<amrex::RunOn::Device>(tmp, srcBox, 0, dstBox, n, 1);
  }
  ifs.close();
}

void
TurbInflow::read_turb_planes(TurbParm& a_tp, amrex::Real z)
{
  if (a_tp.istimeplanes) {
    // If time_periodic is enabled, wrap the time value to be within bounds
    if (a_tp.time_periodic) {
      const amrex::Real t_start = a_tp.planeTimes[0];
      const amrex::Real t_end = a_tp.planeTimes[a_tp.kmax - 2];
      const amrex::Real period = t_end - t_start;
      if (period > 0.0) {
        // Wrap z to be within [t_start, t_end)
        amrex::Real z_wrapped = z - t_start;
        z_wrapped = z_wrapped - std::floor(z_wrapped / period) * period;
        z = z_wrapped + t_start;
      }
    }

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
    a_tp.izlo = (int)(floor(z * a_tp.dxinv[2] - 0.5)) -
                1; // read one extra earlier tplane to prevent rereading
    a_tp.izhi = a_tp.izlo + a_tp.nplane - 1;
    a_tp.szlo = (static_cast<amrex::Real>(a_tp.izlo) + 0.5) * a_tp.dx[2];
    a_tp.szhi = (static_cast<amrex::Real>(a_tp.izhi) - 0.5) *
                a_tp.dx[2]; // need one plane forward in time for interpolating
  }
  if (a_tp.verbose > 1) {
    std::string varname = a_tp.istimeplanes ? "t" : "z";
    amrex::Print() << "read_turb_planes filling " << a_tp.izlo << " to "
                   << a_tp.izhi << std::endl
                   << " --> now have interp data for range [" << a_tp.szlo
                   << ", " << a_tp.szhi << ") with current " << varname << " = "
                   << z << std::endl;
  }

  for (int iplane = 0; iplane < a_tp.nplane; ++iplane) {
    int k = a_tp.izlo + iplane;
    if (!a_tp.istimeplanes) {
      k = (a_tp.npboxcells[2] + k) %
          a_tp.npboxcells[2]; // "wrap" planes if data is periodic
    }
    read_one_turb_plane(a_tp, iplane, k);
  }
}

void
TurbInflow::fill_turb_plane(
  TurbParm& a_tp,
  const amrex::Vector<amrex::Real>& x,
  const amrex::Vector<amrex::Real>& y,
  amrex::Real z,
  amrex::FArrayBox& v)
{
  // If time_periodic is enabled and istimeplanes, wrap the time value
  if (a_tp.istimeplanes && a_tp.time_periodic && a_tp.kmax > 0) {
    const amrex::Real t_start = a_tp.planeTimes[0];
    const amrex::Real t_end = a_tp.planeTimes[a_tp.kmax - 2];
    const amrex::Real period = t_end - t_start;
    if (period > 0.0) {
      // Wrap z to be within [t_start, t_end)
      amrex::Real z_wrapped = z - t_start;
      z_wrapped = z_wrapped - std::floor(z_wrapped / period) * period;
      z = z_wrapped + t_start;
    }
  }

  const amrex::Real tplanes_lo = a_tp.szlo;
  const amrex::Real tplanes_hi = a_tp.szhi;

  if ((z < tplanes_lo) || (z >= tplanes_hi)) {
    if (a_tp.verbose > 1) {
      std::string varname = a_tp.istimeplanes ? "t" : "z";
      amrex::Print() << "Reading new data because " << varname << " = " << z
                     << " is outside the range [" << tplanes_lo << ", "
                     << tplanes_hi << ")" << std::endl;
    }
    read_turb_planes(a_tp, z);
  }

  const auto& bx = v.box();
  const auto& vd = v.array();

  amrex::Gpu::DeviceVector<amrex::Real> x_dev(x.size());
  amrex::Gpu::DeviceVector<amrex::Real> y_dev(y.size());
  amrex::Gpu::copyAsync(
    amrex::Gpu::hostToDevice, x.begin(), x.end(), x_dev.begin());
  amrex::Gpu::copyAsync(
    amrex::Gpu::hostToDevice, y.begin(), y.end(), y_dev.begin());
  amrex::Real* xd = x_dev.data();
  amrex::Real* yd = y_dev.data();

  amrex::Real velScale = (a_tp.side == amrex::Orientation::high)
                           ? -a_tp.turb_scale_vel
                           : a_tp.turb_scale_vel;
  const auto& npboxcells = a_tp.npboxcells;
  const auto& pboxlo = a_tp.pboxlo;
  const auto& pboxsize = a_tp.pboxsize;
  const auto& szlo = a_tp.szlo;
  const auto& dxinv = a_tp.dxinv;
  const auto& dx = a_tp.dx;
  const auto& sd = a_tp.sdata->array();
  const auto& ext_nonper = a_tp.extrap_nonperiodic;
  const auto& tile_per = a_tp.tile_periodic;
  const auto& periodicity = a_tp.periodicity;
  const bool lininterp = a_tp.interp_type == TurbInterpType::linear;
  amrex::Real cz[3];
  int k0 = -1;
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
    amrex::Real zz =
      (z - szlo) * dxinv[2];    // How many dz away from the left side ?
    k0 = (int)(std::floor(zz)); // What's the closest point ?
    zz -= amrex::Real(k0);
    cz[0] =
      lininterp ? 1.0 - zz : 0.5 * (zz - 1.0) * (zz - 2.0); // Weight of k0
    cz[1] = lininterp ? zz : zz * (2.0 - zz);               // Weight of k0 + 1
    cz[2] = lininterp ? 0.0 : 0.5 * zz * (zz - 1.0);        // Weight of k0 + 2
  }

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    amrex::Real cx[3], cy[3], ydata[3];
    amrex::Real zdata[3][3];

    const amrex::Real x_from_pboxlo = xd[i - bx.smallEnd(0)] - pboxlo[0];
    amrex::Real xx =
      (x_from_pboxlo)*dxinv[0] +
      0.5; // from 1st cell center (ghost cell so 1/2 dx outside of pbox)
    const amrex::Real y_from_pboxlo = yd[j - bx.smallEnd(1)] - pboxlo[1];
    amrex::Real yy =
      (y_from_pboxlo)*dxinv[1] +
      0.5; // from 1st cell center (ghost cell so 1/2 dx outside of pbox)
    int i0 = (int)(std::floor(xx));
    int j0 = (int)(std::floor(yy));
    xx -= amrex::Real(i0);
    yy -= amrex::Real(j0);
    // Wrap if needed
    if (tile_per) {
      i0 = (i0 % npboxcells[0] + npboxcells[0]) % npboxcells[0];
      j0 = (j0 % npboxcells[1] + npboxcells[1]) % npboxcells[1];
    }
    cx[0] = lininterp ? 1.0 - xx : 0.5 * (xx - 1.0) * (xx - 2.0);
    cy[0] = lininterp ? 1.0 - yy : 0.5 * (yy - 1.0) * (yy - 2.0);
    cx[1] = lininterp ? xx : xx * (2.0 - xx);
    cy[1] = lininterp ? yy : yy * (2.0 - yy);
    cx[2] = lininterp ? 0.0 : 0.5 * xx * (xx - 1.0);
    cy[2] = lininterp ? 0.0 : 0.5 * yy * (yy - 1.0);

    if (
      (x_from_pboxlo >= 0.0 && x_from_pboxlo < pboxsize[0]) ||
      (tile_per && periodicity[0] != 0)) {
      if (
        (x_from_pboxlo < 0.5 * dx[0] || x_from_pboxlo > pboxsize[0] - dx[0]) &&
        (periodicity[0] == 0) && !ext_nonper) {
        amrex::Error(
          "TurbInflow interp stencil touches ghost cell for nonperiodic "
          "direction and extrap_nonperiodic option not used");
      }
      if (
        (y_from_pboxlo >= 0.0 && y_from_pboxlo < pboxsize[1]) ||
        (tile_per && periodicity[1] != 0)) {
        if (
          (y_from_pboxlo < 0.5 * dx[1] ||
           y_from_pboxlo > pboxsize[1] - dx[1]) &&
          (periodicity[1] == 0) && !ext_nonper) {
          amrex::Error(
            "TurbInflow interp stencil touches ghost cell for nonperiodic "
            "direction and extrap_nonperiodic option not used");
        }

        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
          for (int ii = 0; ii <= 2; ++ii) {
            for (int jj = 0; jj <= 2; ++jj) {
              zdata[ii][jj] = cz[0] * sd(i0 + ii, j0 + jj, k0, n) +
                              cz[1] * sd(i0 + ii, j0 + jj, k0 + 1, n) +
                              cz[2] * sd(i0 + ii, j0 + jj, k0 + 2, n);
            }
          }
          for (int ii = 0; ii <= 2; ++ii) {
            ydata[ii] = cy[0] * zdata[ii][0] + cy[1] * zdata[ii][1] +
                        cy[2] * zdata[ii][2];
          }
          vd(i, j, k, n) +=
            velScale * (cx[0] * ydata[0] + cx[1] * ydata[1] + cx[2] * ydata[2]);
        }
      }
    }
  });
  amrex::Gpu::synchronize(); // Ensure that DeviceVector's don't leave scope
                             // early
}
} // namespace pele::physics::turbinflow
