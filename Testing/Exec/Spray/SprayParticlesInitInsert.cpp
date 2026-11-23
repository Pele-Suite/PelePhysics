// The application provides these (PeleC, PeleLMeX). The container gets the
// jets named by spraytest.jet_names.
#include "SprayParticles.H"
#include "SprayInjection.H"

bool
SprayParticleContainer::injectParticles(
  amrex::Real time,
  amrex::Real dt,
  int /*nstep*/,
  int level,
  int /*finest_level*/)
{
  for (auto& jet : m_sprayJets) {
    sprayInjection(time, jet.get(), dt, level);
  }
  return !m_sprayJets.empty();
}

void
SprayParticleContainer::InitSprayParticles(const bool init_parts)
{
  if (!init_parts) {
    return;
  }
  amrex::ParmParse pp("spraytest");
  std::vector<std::string> names;
  pp.queryarr("jet_names", names);
  for (const auto& name : names) {
    m_sprayJets.push_back(std::make_unique<SprayJet>(name, Geom(0)));
  }
}
