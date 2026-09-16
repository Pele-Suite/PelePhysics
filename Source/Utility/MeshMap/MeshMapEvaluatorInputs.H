#ifndef PELE_PHYSICS_MESHMAP_EVALUATOR_INPUTS_H
#define PELE_PHYSICS_MESHMAP_EVALUATOR_INPUTS_H

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>

#include "MeshMapEvaluator.H"

#include <cmath>
#include <limits>
#include <string>

namespace pele::physics {

/**
 * \brief Build a MeshMapEvaluator from the same input-file block PeleLMeX
 *        reads for \c geometry.mesh_mapping.
 *
 * Host-only.  Recognised (all names and defaults as in PeleLMeX's
 * \c Source/Mesh/ map classes, so a precursor's lines can be copied
 * verbatim into a utility's inputs):
 *
 *   geometry.mesh_mapping          = ConstantMap | ExpStretchMap |
 *                                    TanhStretchMap | InteriorStretchMap
 *   ConstantMap.scaling_factor     = fx fy fz            (default 1 1 1)
 *   ExpStretchMap.direction        = d                   (default 1)
 *   ExpStretchMap.wall_end         = lo | hi             (default lo)
 *   ExpStretchMap.beta             = beta                (default 0)
 *   TanhStretchMap.beta            = bx by bz            (default 0 0 0)
 *   InteriorStretchMap.beta_lo     = bx by bz            (default 0 0 0)
 *   InteriorStretchMap.beta_hi     = bx by bz            (default 0 0 0)
 *   InteriorStretchMap.center_frac = cx cy cz            (default .5 .5 .5)
 *   InteriorStretchMap.center      = x y z   physical; overrides center_frac
 *
 * \p probDomain is the Xi-space problem domain, needed only to turn an
 * InteriorStretchMap physical \c center into a fraction of the axis.  When
 * \c geometry.mesh_mapping is absent the result is \c Kind::Identity, so
 * callers can test \c e.m_kind against it to mean "no mapping".
 *
 * Validation mirrors PeleLMeX's constructors (non-negative betas, positive
 * scaling factors, interior clustering point strictly inside the domain)
 * and aborts with the same intent on failure.
 */
inline MeshMapEvaluator
make_evaluator_from_inputs(const amrex::RealBox& probDomain)
{
  using amrex::Real;
  MeshMapEvaluator e;

  amrex::ParmParse ppg("geometry");
  std::string name;
  if (ppg.query("mesh_mapping", name) == 0) {
    return e; // Kind::Identity
  }

  if (name == "ConstantMap") {
    amrex::ParmParse pp("ConstantMap");
    amrex::Vector<Real> fac(AMREX_SPACEDIM, Real(1.0));
    pp.queryarr("scaling_factor", fac, 0, AMREX_SPACEDIM);
    e.m_kind = MeshMapEvaluator::Kind::Constant;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      if (fac[d] <= std::numeric_limits<Real>::epsilon()) {
        amrex::Abort(
          "ConstantMap: All scaling factors must be strictly positive.");
      }
      e.m_p[d] = fac[d];
      e.m_q[d] = -1;
    }
    return e;
  }

  if (name == "ExpStretchMap") {
    amrex::ParmParse pp("ExpStretchMap");
    int dir = 1;
    pp.query("direction", dir);
    if (dir < 0 || dir >= AMREX_SPACEDIM) {
      amrex::Abort(
        "ExpStretchMap: 'direction' must be in [0, AMREX_SPACEDIM).  "
        "Set ExpStretchMap.direction to 0, 1, or 2.");
    }
    std::string wall{"lo"};
    pp.query("wall_end", wall);
    int wall_end = -1;
    if (wall == "lo") {
      wall_end = 0;
    } else if (wall == "hi") {
      wall_end = 1;
    } else {
      amrex::Abort(
        "ExpStretchMap: 'wall_end' must be 'lo' or 'hi' (got '" + wall + "').");
    }
    Real beta = Real(0.0);
    pp.query("beta", beta);
    if (beta < Real(0.0)) {
      amrex::Abort(
        "ExpStretchMap: 'beta' must be non-negative (got " +
        std::to_string(beta) + ").");
    }
    e.m_kind = MeshMapEvaluator::Kind::ExpStretch;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      e.m_p[d] = (d == dir) ? beta : Real(0.0);
      e.m_q[d] = (d == dir) ? wall_end : -1;
    }
    return e;
  }

  if (name == "TanhStretchMap") {
    amrex::ParmParse pp("TanhStretchMap");
    amrex::Vector<Real> beta(AMREX_SPACEDIM, Real(0.0));
    pp.queryarr("beta", beta, 0, AMREX_SPACEDIM);
    e.m_kind = MeshMapEvaluator::Kind::TanhStretch;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      if (beta[d] < Real(0.0)) {
        amrex::Abort(
          "TanhStretchMap: beta values must be non-negative (got " +
          std::to_string(beta[d]) + " on axis " + std::to_string(d) + ").");
      }
      e.m_p[d] = beta[d];
      e.m_q[d] = -1;
    }
    return e;
  }

  if (name == "InteriorStretchMap") {
    // Mirrors PeleLMeX's InteriorStretchMap constructor: the clustering
    // point is a fraction of the axis (physical `center` wins), and the
    // Xi-space split s_c that places it there follows in closed form.
    constexpr Real SPLIT_EPS = Real(1.0e-12);
    amrex::ParmParse pp("InteriorStretchMap");
    amrex::Vector<Real> blo(AMREX_SPACEDIM, Real(0.0));
    amrex::Vector<Real> bhi(AMREX_SPACEDIM, Real(0.0));
    pp.queryarr("beta_lo", blo, 0, AMREX_SPACEDIM);
    pp.queryarr("beta_hi", bhi, 0, AMREX_SPACEDIM);
    amrex::Vector<Real> cfrac(AMREX_SPACEDIM, Real(0.5));
    pp.queryarr("center_frac", cfrac, 0, AMREX_SPACEDIM);
    if (pp.contains("center")) {
      amrex::Vector<Real> cphys(AMREX_SPACEDIM, Real(0.0));
      pp.getarr("center", cphys, 0, AMREX_SPACEDIM);
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        cfrac[d] = (cphys[d] - probDomain.lo(d)) / probDomain.length(d);
      }
    }
    e.m_kind = MeshMapEvaluator::Kind::InteriorStretch;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      if (blo[d] < Real(0.0) || bhi[d] < Real(0.0)) {
        amrex::Abort(
          "InteriorStretchMap: beta_lo/beta_hi must be non-negative (got " +
          std::to_string(blo[d]) + " / " + std::to_string(bhi[d]) +
          " on axis " + std::to_string(d) + ").");
      }
      const Real beta_lo =
        (blo[d] < MeshMapEvaluator::BETA_EPS) ? Real(0.0) : blo[d];
      const Real beta_hi =
        (bhi[d] < MeshMapEvaluator::BETA_EPS) ? Real(0.0) : bhi[d];
      const bool stretched = (beta_lo > Real(0.0)) || (beta_hi > Real(0.0));
      if (stretched && (cfrac[d] <= Real(0.0) || cfrac[d] >= Real(1.0))) {
        amrex::Abort(
          "InteriorStretchMap: the clustering point must lie strictly "
          "inside the domain on axis " +
          std::to_string(d) + " (got a fraction of " +
          std::to_string(cfrac[d]) +
          ").  Use ExpStretchMap for clustering at a boundary.");
      }
      const Real cf =
        std::fmin(std::fmax(cfrac[d], SPLIT_EPS), Real(1.0) - SPLIT_EPS);
      const Real s_c = std::fmin(
        std::fmax(
          MeshMapEvaluator::interior_split(cf, beta_lo, beta_hi), SPLIT_EPS),
        Real(1.0) - SPLIT_EPS);
      e.m_p[d] = beta_lo;
      e.m_p2[d] = beta_hi;
      e.m_p3[d] = s_c;
      e.m_q[d] = -1;
    }
    return e;
  }

  amrex::Abort(
    "make_evaluator_from_inputs(): unrecognised geometry.mesh_mapping '" +
    name +
    "'.  Supported: ConstantMap, ExpStretchMap, TanhStretchMap, "
    "InteriorStretchMap.");
  return e;
}

} // namespace pele::physics

#endif // PELE_PHYSICS_MESHMAP_EVALUATOR_INPUTS_H
