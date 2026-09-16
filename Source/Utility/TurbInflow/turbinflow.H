#ifndef TURBINFLOW_H
#define TURBINFLOW_H

#include <AMReX_FArrayBox.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>

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

struct TurbParm
{
  // Turbulence data file
  std::string m_turb_file;

  // Verbosity
  int verbose = 1;

  // Geometry of the turbulent data
  amrex::GpuArray<int, 3> npboxcells = {{0}};
  amrex::GpuArray<amrex::Real, 3> pboxlo = {{0.0}};
  amrex::GpuArray<amrex::Real, 3> dx = {{0.0}};
  amrex::GpuArray<amrex::Real, 3> dxinv = {{0.0}};
  amrex::GpuArray<amrex::Real, 3> pboxsize = {{0.0}};
  int dir;                       // Direction where to use this TurbParm
  amrex::Orientation::Side side; // Side on which to use this TurbParm
  amrex::Real time_shift =
    0.0; // Enable to offset the location in the turb file

  int nplane = 32; // Number of turb planes stored at once in memory
  amrex::FArrayBox* sdata = nullptr; // Currently loaded chunk of turb data
  int izlo = -1;                     // Currently loaded low plane
  int izhi = -1;                     // Currently loaded high plane
  amrex::Real szlo = -1.0e12;        // Position of the current chunk low plane
  amrex::Real szhi = -1.0e11;        // Position of the current chunk high plane

  amrex::IntVect periodicity{AMREX_D_DECL(1, 1, 1)};
  bool istimeplanes = false; // false indicates turb data is periodic normal to
                             // flow dir, true requires time array in turb data

  bool extrap_nonperiodic =
    false; // if true, allow interpolation where stencil touches ghost cells for
           // nonperiodic directions
  bool tile_periodic = false; // if true, turbulence is patched over entire face
                              // with a periodic tiled pattern
  bool time_periodic = false; // if true, turbulence data is looped in time
  TurbInterpType interp_type =
    TurbInterpType::quadratic; // linear required for non-periodic directions

  amrex::Real turb_scale_loc =
    1.; // Spatial scale relating turbfile geom -> actual case scale
  amrex::Real turb_scale_vel = 1.; // Velocity scaling
  amrex::Real turb_conv_vel =
    1.; // Mean inlet velocity, used to move through data planes

  amrex::Real convected_distance =
    -1.; // If >= 0, used directly as the through-plane sampling position
         // (physical convected distance, e.g. the time integral of the inlet
         // velocity), bypassing turb_conv_vel * time. Negative => disabled.

  amrex::Vector<amrex::Real> planeTimes;

  amrex::Vector<long> offset;
  int kmax; // Number of plane in Turbfile

  // Optional coordinate-mapping descriptor of the turbulence file, read
  // from the MESHMAP trailer of HDR (see the file-format notes in
  // turbinflow.cpp).  When present, the file's transverse grid is uniform
  // in the *computational* (Xi) coordinate of the precursor that produced
  // it, and a physical position must be inverted through this map before
  // being turned into a file index -- add_turb() does so.  Absent trailer
  // => has_map == false and the file is uniform in physical position.
  //
  // The evaluator's axis slots 0 and 1 hold the payload for the file's
  // two transverse directions (in the file's own order), and map_xi_lo /
  // map_xi_hi the precursor's Xi-space bounds along them, i.e. the file's
  // own transverse extent before the two ghost layers.
  bool has_map = false;
  pele::physics::MeshMapEvaluator map{};
  amrex::GpuArray<amrex::Real, 2> map_xi_lo = {{0.0, 0.0}};
  amrex::GpuArray<amrex::Real, 2> map_xi_hi = {{0.0, 0.0}};
};

struct TurbInflow
{
public:
  TurbInflow() = default;

  ~TurbInflow() = default;

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
    tdir2 = (dir != 0) ? ((dir == 2) ? 1 : 2) : 2;
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

  static void read_turb_planes(TurbParm& a_tp, amrex::Real z);

  static void read_one_turb_plane(TurbParm& a_tp, int iplane, int k);

  static void fill_turb_plane(
    TurbParm& a_tp,
    const amrex::Vector<amrex::Real>& x,
    const amrex::Vector<amrex::Real>& y,
    amrex::Real z,
    amrex::FArrayBox& v);

  //! Turbulence-file coordinate of a physical position (already in file
  //! units) along transverse direction \p idim of \p a_tp: the identity
  //! for a physically-uniform file, the inverse of the file's map for a
  //! mesh-mapped one (with the tile_periodic wrap applied first).
  static amrex::Real
  file_coordinate(const TurbParm& a_tp, int idim, amrex::Real x_file_phys);

  //! Smallest and largest physical transverse spacing of \p a_tp's file in
  //! case units, per transverse direction.
  static void transverse_dx_range(
    const TurbParm& a_tp, amrex::Real dx_min[2], amrex::Real dx_max[2]);

  static void set_turb(
    int normDir,
    int transDir1,
    int transDir2,
    amrex::FArrayBox& v,
    amrex::FArrayBox& data,
    const int dcomp);

private:
  //! The TurbParm acting on the (\p dir, \p side) face, or nullptr.
  const TurbParm*
  find_turbparm(const int dir, const amrex::Orientation::Side& side) const;

  amrex::Vector<TurbParm> tp;
  bool turbinflow_initialized = false;
};
} // namespace pele::physics::turbinflow

#endif
