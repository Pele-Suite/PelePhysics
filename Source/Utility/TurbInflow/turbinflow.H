#ifndef TURBINFLOW_H
#define TURBINFLOW_H

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuAsyncArray.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>

#include <cmath>
#include <memory>

// Relative to this header so that a build which lists TurbInflow but not
// Source/Utility/MeshMap among its include directories still compiles.
#include "../MeshMap/MeshMapEvaluator.H"

// Feature macro: this PelePhysics provides the TurbInflow::add_turb()
// overload that takes the physical positions of the inflow-face cell
// centres explicitly, rather than deriving them from an amrex::Geometry.
//
// A solver whose AMReX grid is a uniform computational (Xi) grid carrying a
// coordinate mapping -- e.g. PeleLMeX's geometry.mesh_mapping -- cannot use
// the Geometry-based overload: the turbulence file is indexed by PHYSICAL
// position, while ProbLo + (i+0.5)*CellSize is the Xi position.  Consumers
// #ifdef on this macro so that the PelePhysics and solver-side changes can
// land independently.
#define PELEPHYSICS_TURBINFLOW_HAS_COORD_ADDTURB 1

// Feature macro: the HDR reader understands the optional MESHMAP trailer
// (V1 or V2) that tags a turbulence file as uniform in a precursor's
// computational (Xi) coordinate rather than in physical position.  Files
// without the trailer are, by declaration, uniform in physical position.
#define PELEPHYSICS_TURBINFLOW_HAS_MESHMAP_HDR 1

// Feature macro: add_turb() samples a file carrying a MESHMAP trailer by
// inverting the file's map (interpolation in the precursor's Xi), and the
// file_map() / file_transverse_dx_range() accessors exist.  Consumers that
// previously refused, or warned about, mapped files can #ifdef on this.
#define PELEPHYSICS_TURBINFLOW_SAMPLES_MESHMAP 1

namespace pele::physics::turbinflow {

AMREX_ENUM(TurbInterpType, linear, quadratic);

//! Weights of the three-point stencil at fractional offset \p f from its
//! first point: linear (the third weight is then 0) or quadratic Lagrange
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
stencil_weights(
  const amrex::Real f,
  const bool lininterp,
  amrex::GpuArray<amrex::Real, 3>& c) noexcept
{
  c[0] = lininterp ? 1.0 - f : 0.5 * (f - 1.0) * (f - 2.0);
  c[1] = lininterp ? f : f * (2.0 - f);
  c[2] = lininterp ? 0.0 : 0.5 * f * (f - 1.0);
}

/**
 * \brief Transverse geometry of a turbulence file and how it is placed on
 *        the injection face: everything needed to turn a position into a
 *        file stencil.
 *
 * Index 0 and 1 are the file's two transverse directions.  Lengths are in
 * file units (case units times scale_loc); for a mesh-mapped file
 * (has_map) the grid, and so pboxlo, pboxsize and dx, are in the
 * precursor's Xi coordinate.
 */
struct TurbFileGeom
{
  amrex::GpuArray<int, 2> npboxcells = {{0, 0}};
  amrex::GpuArray<amrex::Real, 2> pboxlo = {{0.0, 0.0}};
  amrex::GpuArray<amrex::Real, 2> pboxsize = {{0.0, 0.0}};
  amrex::GpuArray<amrex::Real, 2> dx = {{0.0, 0.0}};
  amrex::GpuArray<amrex::Real, 2> dxinv = {{0.0, 0.0}};
  amrex::GpuArray<int, 2> periodicity = {{1, 1}};
  // Patch the whole face with a periodic tiling of the file
  bool tile_periodic = false;
  // Allow stencils that touch ghost cells in nonperiodic directions
  bool extrap_nonperiodic = false;
  bool lininterp = false; // Linear rather than quadratic interpolation
  // File units per case unit
  amrex::Real scale_loc = 1.0;

  // Optional coordinate-mapping descriptor, read from the MESHMAP trailer
  // of HDR (see the file-format notes in turbinflow.cpp).  When present,
  // the file's transverse grid is uniform in the computational (Xi)
  // coordinate of the precursor that produced it, and a physical position
  // is inverted through this map before being turned into a file index.
  // The evaluator's axis slots 0 and 1 hold the payload for the file's two
  // transverse directions, and map_xi_lo / map_xi_hi the precursor's Xi
  // bounds along them, i.e. the file's own transverse extent before the
  // two ghost layers.
  bool has_map = false;
  pele::physics::MeshMapEvaluator map{};
  amrex::GpuArray<amrex::Real, 2> map_xi_lo = {{0.0, 0.0}};
  amrex::GpuArray<amrex::Real, 2> map_xi_hi = {{0.0, 0.0}};

  //! Physical position (file units) of Xi coordinate \p xi along \p d of a
  //! mesh-mapped file
  [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
  x_phys(const int d, const amrex::Real xi) const noexcept
  {
    return map.x_phys_from_xi(d, xi, map_xi_lo[d], map_xi_hi[d]);
  }
};

/**
 * \brief Device-callable view of the plane chunk of one turbulence file at
 *        one through-plane position: the interpolation of the file.
 *
 * Obtained from TurbInflow::file_sampler() right before use: it points at the
 * plane chunk currently loaded in memory and holds the through-plane weights
 * for the requested position.  Trivially copyable, so it can be captured by
 * value in a kernel or a boundary-fill functor.
 *
 * It is only valid until that file's planes are next loaded: any later
 * file_sampler() call on the same TurbParm may reload them, and a sampler
 * kept across it then reads the new planes with its old weights, which
 * gives wrong values without any error.  Get the sampler, launch the
 * kernels that use it, and get a new one for the next position; on the
 * host, TurbInflow::sampler_is_current() tells whether one is still valid.
 */
struct TurbFileSampler : TurbFileGeom
{
  amrex::Array4<const amrex::Real> sd;

  // Through-plane stencil: planes k0, k0+1, k0+2 with weights cz
  int k0 = 0;
  amrex::GpuArray<amrex::Real, 3> cz = {{0.0, 0.0, 0.0}};

  //! TurbParm::chunk_generation of the chunk this sampler points at
  int generation = -1;

  //! Transverse interpolation stencil of one position, shared by all the
  //! components sampled there
  struct Stencil
  {
    amrex::GpuArray<int, 2> i0{};
    amrex::GpuArray<amrex::GpuArray<amrex::Real, 3>, 2> c{};
  };

  /**
   * \brief Locate a position given in file coordinates (see
   *        file_coordinate()).
   *
   * Returns false when it lies outside the file (and the file is not tiled
   * there), so that the caller can fall back to its own boundary value.
   */
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool locate_file(
    const amrex::Real xf1, const amrex::Real xf2, Stencil& st) const noexcept
  {
    const amrex::GpuArray<amrex::Real, 2> xf = {xf1, xf2};
    for (int d = 0; d < 2; ++d) {
      const amrex::Real from_lo = xf[d] - pboxlo[d];
      const bool inside = (from_lo >= 0.0 && from_lo < pboxsize[d]) ||
                          (tile_periodic && periodicity[d] != 0);
      if (!inside) {
        return false;
      }
      if (
        (from_lo < 0.5 * dx[d] || from_lo > pboxsize[d] - dx[d]) &&
        (periodicity[d] == 0) && !extrap_nonperiodic) {
        amrex::Error(
          "TurbInflow interp stencil touches ghost cell for nonperiodic "
          "direction and extrap_nonperiodic option not used");
      }
      // From the first cell center (ghost cell, 1/2 dx outside of pbox)
      amrex::Real ff = from_lo * dxinv[d] + 0.5;
      int i0 = static_cast<int>(std::floor(ff));
      ff -= static_cast<amrex::Real>(i0);
      if (tile_periodic) {
        i0 = (i0 % npboxcells[d] + npboxcells[d]) % npboxcells[d];
      }
      st.i0[d] = i0;
      stencil_weights(ff, lininterp, st.c[d]);
    }
    return true;
  }

  //! Coordinate in the file of a position (already scaled to file units)
  //! along transverse direction \p d: the identity for a file that is uniform
  //! in physical position, the inverse of the file's map for a mesh-mapped
  //! one (with the tile_periodic wrap applied first).
  [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
  file_coordinate(const int d, const amrex::Real x_file_phys) const noexcept
  {
    if (!has_map) {
      return x_file_phys;
    }
    const amrex::Real xi_lo = map_xi_lo[d];
    const amrex::Real xi_hi = map_xi_hi[d];
    const amrex::Real x_lo = x_phys(d, xi_lo);
    const amrex::Real x_hi = x_phys(d, xi_hi);

    amrex::Real x = x_file_phys;
    if (tile_periodic && periodicity[d] != 0) {
      // Tiled: bring x into the file's physical period before inverting.
      // The inverse clamps to [x_lo, x_hi], which would otherwise collapse
      // every tiled copy onto the edge of the file.
      const amrex::Real period = x_hi - x_lo;
      x -= std::floor((x - x_lo) / period) * period;
    } else if (x < x_lo || x >= x_hi) {
      // Outside the file's physical extent.  A physically-uniform file leaves
      // such positions unfilled (the range test of the interpolation); keep
      // that behaviour rather than let the clamped inverse pin them to the
      // edge row, by returning a Xi coordinate that is outside the box
      // whatever turb_center shifted pboxlo to (|shift| <= pboxsize/2).
      return (x < x_lo) ? xi_lo - pboxsize[d] : xi_hi + pboxsize[d];
    }

    amrex::Real xi = map.xi_from_x_phys(d, x, xi_lo, xi_hi);

    // When the target's map and Xi grid coincide with the file's, xi lands on
    // a file cell centre up to the round-off of the inverse (log/atanh, or
    // the Newton tolerance for InteriorStretch).  Snap it so that the
    // interpolation weights collapse exactly and the same-map case
    // reproduces the file bit for bit.
    const amrex::Real idx = (xi - pboxlo[d]) * dxinv[d] + 0.5;
    const amrex::Real idx_round = std::round(idx);
    constexpr amrex::Real snap_tol = 1.0e-8;
    if (std::abs(idx - idx_round) < snap_tol) {
      xi = pboxlo[d] + (idx_round - 0.5) * dx[d];
    }
    return xi;
  }

  //! Value of chunk component \p n at a located position
  [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
  component(const Stencil& st, const int n) const noexcept
  {
    amrex::GpuArray<amrex::Real, 3> ydata{};
    for (int ii = 0; ii <= 2; ++ii) {
      amrex::GpuArray<amrex::Real, 3> zdata{};
      for (int jj = 0; jj <= 2; ++jj) {
        const int i = st.i0[0] + ii;
        const int j = st.i0[1] + jj;
        zdata[jj] = cz[0] * sd(i, j, k0, n) + cz[1] * sd(i, j, k0 + 1, n) +
                    cz[2] * sd(i, j, k0 + 2, n);
      }
      ydata[ii] =
        st.c[1][0] * zdata[0] + st.c[1][1] * zdata[1] + st.c[1][2] * zdata[2];
    }
    return st.c[0][0] * ydata[0] + st.c[0][1] * ydata[1] +
           st.c[0][2] * ydata[2];
  }
};

struct TurbParm
{
  // Move-only: copies would share one plane chunk (sdata) but track its
  // loaded planes separately
  TurbParm() = default;
  TurbParm(const TurbParm&) = delete;
  TurbParm& operator=(const TurbParm&) = delete;
  TurbParm(TurbParm&&) = default;
  TurbParm& operator=(TurbParm&&) = default;
  ~TurbParm() = default;

  // Turbulence data file
  std::string m_turb_file;

  // Verbosity
  int verbose = 1;

  // Transverse geometry and placement of the file
  TurbFileGeom geom;

  // Through-plane direction: plane spacing (positions are in file units)
  // and number of planes in the file
  amrex::Real dz = 0.0;
  amrex::Real dzinv = 0.0;
  int kmax = 0;

  int dir = -1;                  // Direction where to use this TurbParm
  amrex::Orientation::Side side; // Side on which to use this TurbParm
  amrex::Real time_shift =
    0.0; // Enable to offset the location in the turb file

  int ncomp = 0;   // Number of components in the file
  int nplane = 32; // Number of turb planes stored at once in memory
  // Currently loaded chunk of turb data; shared only for make_plane_chunk()
  std::shared_ptr<amrex::FArrayBox> sdata;
  int izlo = -1;              // Currently loaded low plane
  int izhi = -1;              // Currently loaded high plane
  amrex::Real szlo = -1.0e12; // Position of the current chunk low plane
  amrex::Real szhi = -1.0e11; // Position of the current chunk high plane
  int chunk_generation = 0;   // Changed whenever the chunk is reloaded

  bool istimeplanes = false;  // false indicates turb data is periodic normal to
                              // flow dir, true requires time array in turb data
  bool time_periodic = false; // if true, turbulence data is looped in time

  amrex::Real turb_scale_vel = 1.; // Velocity scaling
  amrex::Real turb_conv_vel =
    1.; // Mean inlet velocity, used to move through data planes

  amrex::Real convected_distance =
    -1.; // If >= 0, used directly as the through-plane sampling position
         // (physical convected distance, e.g. the time integral of the inlet
         // velocity), bypassing turb_conv_vel * time. Negative => disabled.

  amrex::Vector<amrex::Real> planeTimes;

  // Offset in DAT of plane k of component n, at index k + n * kmax
  amrex::Vector<long> offset;
};

struct TurbInflow
{
public:
  TurbInflow() = default;

  TurbInflow(const TurbInflow&) = delete;
  TurbInflow& operator=(const TurbInflow&) = delete;
  TurbInflow(TurbInflow&&) = default;
  TurbInflow& operator=(TurbInflow&&) = default;

  void init(amrex::Geometry const& geom);

  //! Uniform-grid entry point: the physical position of a cell centre is
  //! ProbLo + (i+0.5)*CellSize.  Builds those coordinates and forwards to
  //! the overload below, which holds the single implementation.
  void add_turb(
    amrex::Box const& bx,
    amrex::FArrayBox& data,
    const int dcomp,
    amrex::Geometry const& geom,
    const amrex::Real time,
    const int dir,
    const amrex::Orientation::Side& side);

  /**
   * \brief Add turbulence to \p bx using caller-supplied physical
   *        coordinates for the two transverse directions.
   *
   * For a solver on a non-uniform or mapped grid this is the correct entry
   * point: the turbulence file is sampled by physical position, which the
   * caller (who owns the mapping) is the only one able to compute.
   *
   * \p x_phys and \p y_phys are the physical cell-centre coordinates along
   * the transverse directions \c tdir1 and \c tdir2 given by
   * transverseDirs(dir, ...). They must have lengths bx.length(tdir1) and
   * bx.length(tdir2), with x_phys[0] corresponding to index
   * bx.smallEnd(tdir1) (and similarly for y_phys/tdir2), and must span the
   * corresponding extents of \p bx.  They are in case units; the per-TurbParm
   * \c turb_scale_loc is applied internally, exactly as for the
   * Geometry-based overload.
   *
   * \p domain is the (Xi-space) problem domain Box, used only to locate the
   * ghost plane being filled.
   */
  void add_turb(
    amrex::Box const& bx,
    amrex::FArrayBox& data,
    const int dcomp,
    amrex::Box const& domain,
    const amrex::Vector<amrex::Real>& x_phys,
    const amrex::Vector<amrex::Real>& y_phys,
    const amrex::Real time,
    const int dir,
    const amrex::Orientation::Side& side);

  /**
   * \brief The two transverse directions associated with injection
   *        direction \p dir, in the order the turbulence data uses them.
   *
   * Exposed so that callers building coordinate vectors for the overload
   * above cannot disagree with the internal convention.
   */
  static AMREX_FORCE_INLINE void
  transverseDirs(const int dir, int& tdir1, int& tdir2) noexcept
  {
    tdir1 = (dir != 0) ? 0 : 1;
    tdir2 = (dir == 2) ? 1 : 2;
  }

  /**
   * \brief Transverse cell spacings of the turbulence file feeding the
   *        (\p dir, \p side) face, expressed in case units.
   *
   * Returns false when no turb file acts on that face.  Intended for
   * resolution-consistency diagnostics: the ratio of the target grid's
   * local spacing to this value is what decides whether the injected
   * field can fill the grid's resolved scales.  For a mesh-mapped file
   * (\c has_map) the physical spacing varies across the plane and this
   * reports its mean; see \c file_transverse_dx_range for the extremes.
   */
  bool file_transverse_dx(
    const int dir,
    const amrex::Orientation::Side& side,
    amrex::Real& dx_tdir1,
    amrex::Real& dx_tdir2) const;

  /**
   * \brief Smallest and largest physical transverse cell spacing of the
   *        turbulence file feeding the (\p dir, \p side) face, in case
   *        units, for each of the two transverse directions.
   *
   * Equal to \c file_transverse_dx for a physically-uniform file.  Returns
   * false when no turb file acts on that face.
   */
  bool file_transverse_dx_range(
    const int dir,
    const amrex::Orientation::Side& side,
    amrex::Real dx_min[2],
    amrex::Real dx_max[2]) const;

  /**
   * \brief Whether the turbulence file feeding the (\p dir, \p side) face
   *        carries a MESHMAP trailer, i.e. is uniform in a precursor's
   *        computational coordinate rather than in physical position.
   *
   * Returns false when no turb file acts on that face.
   */
  bool file_has_map(const int dir, const amrex::Orientation::Side& side) const;

  /**
   * \brief The mesh-mapping descriptor of the turbulence file feeding the
   *        (\p dir, \p side) face.
   *
   * Returns -1 when no turb file acts on that face, 0 when the file is
   * uniform in physical position (\p map is then left untouched), and 1
   * when it carries a MESHMAP trailer, in which case \p map holds the
   * precursor's map with the file's two transverse directions in axis
   * slots 0 and 1 and \p xi_lo / \p xi_hi their Xi-space bounds.  Lets a
   * solver compare the file's map with its own: equal maps and bounds
   * mean the file is injected by exact index, anything else by
   * interpolation in the file's Xi coordinate.
   */
  int file_map(
    const int dir,
    const amrex::Orientation::Side& side,
    pele::physics::MeshMapEvaluator& map,
    amrex::Real xi_lo[2],
    amrex::Real xi_hi[2]) const;

  bool is_initialized() const { return turbinflow_initialized; }

  // Override the through-plane sampling position with a supplied physical
  // convected distance (e.g. the time-integrated inlet velocity). Applies to
  // all loaded turb planes. Pass a negative value to revert to the default
  // turb_conv_vel * time behavior.
  void set_convected_distance(amrex::Real dist)
  {
    for (auto& t : tp) {
      t.convected_distance = dist;
    }
  }

  //! Whether \p s still points at the plane chunk loaded in \p a_tp, i.e.
  //! no reload has happened since file_sampler() returned it.
  static bool
  sampler_is_current(const TurbParm& a_tp, const TurbFileSampler& s) noexcept
  {
    return s.generation == a_tp.chunk_generation;
  }

private:
  /**
   * \brief Allocate an arena-backed buffer of a TurbParm (the plane chunk).
   *
   * A TurbInflow may outlive amrex::Finalize (PeleC holds a static one), so
   * the buffer is freed at Finalize, while its arena still exists, if it is
   * alive then.  Its owner is left with an empty FArrayBox, which is safe to
   * destroy, move or replace afterwards.  Every arena-backed member of
   * TurbParm must be allocated through here.
   */
  static std::shared_ptr<amrex::FArrayBox>
  make_plane_chunk(const amrex::Box& bx, int ncomp);

  //! Load the chunk of planes around \p z, which for a time_periodic file
  //! must already be wrapped into the file's period (prepare_planes() does
  //! so).
  static void read_turb_planes(TurbParm& a_tp, amrex::Real z);

  //! Through-plane sampling position of \p a_tp at \p time: the plane time
  //! for time-plane files, else the convected distance (file units).
  static amrex::Real sampling_position(const TurbParm& a_tp, amrex::Real time);

  //! Make sure the planes around \p z are loaded and return the
  //! through-plane stencil (planes k0..k0+2 of sdata, weights cz).
  static void prepare_planes(
    TurbParm& a_tp,
    amrex::Real z,
    amrex::GpuArray<amrex::Real, 3>& cz,
    int& k0);

  //! Load the planes around through-plane position \p z and return the
  //! sampler of \p a_tp's file there.  Host only; used on the device.  The
  //! sampler is invalidated by the next reload of \p a_tp's planes (see
  //! TurbFileSampler).
  static TurbFileSampler file_sampler(TurbParm& a_tp, amrex::Real z);

  //! Smallest and largest physical transverse spacing of \p a_tp's file in
  //! case units, per transverse direction.
  static void transverse_dx_range(
    const TurbParm& a_tp, amrex::Real dx_min[2], amrex::Real dx_max[2]);

  //! The TurbParm acting on the (\p dir, \p side) face, or nullptr.
  const TurbParm*
  find_turbparm(const int dir, const amrex::Orientation::Side& side) const;

  amrex::Vector<TurbParm> tp;
  bool turbinflow_initialized = false;
};
} // namespace pele::physics::turbinflow

#endif
