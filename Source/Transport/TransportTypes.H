#ifndef TRANSPORTTYPES_H
#define TRANSPORTTYPES_H

#include <AMReX_REAL.H>

// Feature macro for downstream codes: the request/output transport API below
// (TransportRequest / TransportOutputs with the multicomponent diffusion
// matrix) is available. The LCT conductivity models and the rescaled thermal
// diffusion ratios additionally need the build option PELE_TRANSPORT_LCT
// (GNU make: USE_LCT_TRANSPORT = TRUE).
#define PELE_PHYSICS_TRANSPORT_EXTENDED_API 1

namespace pele::physics {
namespace transport {
struct ConstTransport;
struct SimpleTransport;
struct SutherlandTransport;
struct ManifoldTransport;

// Which quantities an extended transport() call should evaluate
struct TransportRequest
{
  bool xi{false};    // bulk viscosity
  bool mu{false};    // shear viscosity
  bool lam{false};   // thermal conductivity
  bool Ddiag{false}; // mixture-averaged diffusivities rho*D_i*W_i/Wbar
  bool chi{false};   // mixture-averaged Soret coefficients (light species)
  bool Dmat{false};  // multicomponent flux diffusion matrix rho*Y_i*D_ij
  bool chit{false};  // rescaled thermal diffusion ratios (needs an LCT model)
};

// Where an extended transport() call writes its results. Every member is a
// nullable pointer so the struct can be captured by value in device lambdas.
// Array sizes: Ddiag, chi, chit: NUM_SPECIES; Dmat: NUM_SPECIES*NUM_SPECIES.
// chi is accumulated into (+=) and must be zeroed by the caller.
struct TransportOutputs
{
  amrex::Real* mu{nullptr};
  amrex::Real* xi{nullptr};
  amrex::Real* lam{nullptr};
  amrex::Real* Ddiag{nullptr};
  amrex::Real* chi{nullptr};
  amrex::Real* Dmat{nullptr};
  amrex::Real* chit{nullptr};
};
} // namespace transport
#ifdef USE_CONSTANT_TRANSPORT
using TransportType = transport::ConstTransport;
#elif USE_SIMPLE_TRANSPORT
using TransportType = transport::SimpleTransport;
#elif USE_SUTHERLAND_TRANSPORT
using TransportType = transport::SutherlandTransport;
#elif USE_MANIFOLD_TRANSPORT
using TransportType = transport::ManifoldTransport;
#else
static_assert(false, "Invalid Transport specified");
#endif
} // namespace pele::physics
#endif
