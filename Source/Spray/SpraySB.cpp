
#include "SprayParticles.H"
#include "SBData.H"
#include "AhamedSplash.H"
#include "Distributions.H"

using namespace amrex;

void
SprayParticleContainer::CreateSBDroplets(
  const int Np,
  const Real sub_dt,
  const splash_breakup* N_SB_h,
  const SBPtrs& rfh,
  const int level)
{
  ParticleLocData pld;
  const SprayData* fdat = m_sprayData;
  std::map<std::pair<int, int>, Gpu::HostVector<ParticleType>> host_particles;
  std::pair<int, int> ind(pld.m_grid, pld.m_tile);
  for (int n = 0; n < Np; n++) {
    if (N_SB_h[n] != splash_breakup::no_change) {
      RealVect normal;
      RealVect loc0;
      RealVect vel0;
      const int vn = AMREX_SPACEDIM * n;
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        normal[dir] = rfh.norm[vn + dir];
        loc0[dir] = rfh.loc[vn + dir];
        vel0[dir] = rfh.vel[vn + dir];
      }
      Real ref_dia = rfh.ref_dia[n];

      Real num_dens0 = rfh.num_dens[n];
      // These values differ depending on breakup or splashing
      // Splashing: Kv
      // Breakup: Utan
      Real phi1 = rfh.phi1[n];

      // Splashing: ms, splash amount
      // TAB Breakup: TAB y value
      // KH-RT Breakup: Unused
      Real phi2 = rfh.phi2[n];

      // Splashing: film thickness / droplet diameter
      // TAB Breakup: TAB y dot value
      // KH-RT Breakup: Unused
      Real phi3 = rfh.phi3[n];
      Real T0 = rfh.T0[n];
      Array<Real, SPRAY_FUEL_NUM> Y0 = {{0.0}};
#if SPRAY_FUEL_NUM > 1
      Real rho_part = 0.;
      const int vy = SPRAY_FUEL_NUM * n;
      for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
        Y0[spf] = rfh.Y0[vy + spf];
      }
      rho_part = fdat->liqprops.rho_mix(Y0, T0);
#else
      Real rho_part = fdat->liqprops.rho_i(T0, 0);
      Y0[0] = 1.;
#endif
      // Real pmass = M_PI / 6. * rho_part * std::pow(ref_dia, 3);
      Real U0mag = vel0.vectorLength();

      // Splashing
      if (N_SB_h[n] >= splash_breakup::splash_dry_splash) {
        // tanPsi: parallel with wall, perpendicular to velocity
        // tanBeta: parallel with wall, in plane with velocity
        RealVect tanPsi, tanBeta;
        find_tangents(vel0, tanPsi, normal, tanBeta);
        Real Kv = phi1;
        Real ms = phi2;
        Real del_film = phi3;
        Real U0norm = normal.dotProduct(vel0);
        Real alpha = amrex::max<amrex::Real>(
          M_PI / 6., std::asin(amrex::Math::abs(U0norm) / U0mag));
        // Real alpha_d = alpha * 180. / M_PI;
        Real U0tan = std::sqrt(U0mag * U0mag - U0norm * U0norm);
        Real uBeta_0, uBeta_half, uBeta_pi, uPsi_coeff, usNorm;
        get_splash_vels(
          U0norm, U0tan, Kv, del_film, uBeta_0, uBeta_half, uBeta_pi,
          uPsi_coeff, usNorm);
        const int Nsint = 4;
        // Secondary mass for drops in each direction, -pi/2, 0, pi/2, and pi
        Real ms_thetas[Nsint];
        get_ms_theta(alpha, ms, del_film, ms_thetas);
        // Note: Must be -pi/2 < psi < pi, not 0 < psi < pi for symmetry
        for (int new_parts = 0; new_parts < Nsint; ++new_parts) {
          ParticleType p;
          p.id() = ParticleType::NextID();
          p.cpu() = ParallelDescriptor::MyProc();
          Real new_mass = ms_thetas[new_parts];
          Real dia_part = std::cbrt(6. * new_mass / (M_PI * rho_part));
          p.rdata(SprayComps::pstateDia) = dia_part;
          Real utBeta = uBeta_half;
          if (new_parts == 1) {
            utBeta = uBeta_0;
          } else if (new_parts == 3) {
            utBeta = uBeta_pi;
          }
          AMREX_D_PICK(
            , , Real psi = 0.5 * M_PI * (static_cast<Real>(new_parts) - 1.);
            Real utPsi = uPsi_coeff * std::sin(psi);)
          for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            Real pvel = AMREX_D_TERM(
              usNorm * normal[dir], +utBeta * tanBeta[dir],
              +utPsi * tanPsi[dir]);
            p.pos(dir) = loc0[dir] + dia_part * normal[dir];
            p.rdata(SprayComps::pstateVel + dir) = pvel;
          }
          for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
            p.rdata(SprayComps::pstateY + spf) = Y0[spf];
          }
          p.rdata(SprayComps::pstateBM1) = 0.;
          p.rdata(SprayComps::pstateBM2) = 0.;
          p.rdata(SprayComps::pstateFilmHght) = 0.;
          p.rdata(SprayComps::pstateN0) = num_dens0;
          p.rdata(SprayComps::pstateNumDens) = num_dens0;
          bool where = Where(p, pld);
          if (!where) {
            amrex::Abort("Bad reflected particle");
          }
          p.rdata(SprayComps::pstateT) = T0;
          host_particles[ind].push_back(p);
        }
        // Breakup
      } else {
        Real Utan = phi1;
        Real dmean = ref_dia;
        // num_dens0 = N_s N_d, where N_d - number of newly created parcels and
        // N_s - number density of newly created parcels There is no one way to
        // do this
        Real N_s = std::pow(num_dens0, m_breakupPPPFact);
        int N_d = amrex::max<int>(1, static_cast<int>(num_dens0 / N_s));
        N_s = num_dens0 / static_cast<Real>(N_d);
        // Real new_mass = M_PI / 6. * rho_part * std::pow(dmean, 3);
#if AMREX_SPACEDIM == 3
        RealVect testvec(1., 0., 0.);
        if (testvec.crossProduct(normal).vectorLength() < 1.E-5) {
          testvec = {0., 1., 0.};
        }
        RealVect tanPsi = testvec.crossProduct(normal);
        tanPsi /= tanPsi.vectorLength();
        RealVect tanBeta = tanPsi.crossProduct(normal);
        tanBeta /= tanBeta.vectorLength();
#else
        RealVect tanBeta(-normal[1], normal[0]);
#endif
        for (int new_parts = 0; new_parts < N_d; ++new_parts) {
          Real rand = amrex::Random();
          ParticleType p;
          p.id() = ParticleType::NextID();
          p.cpu() = ParallelDescriptor::MyProc();
          p.rdata(SprayComps::pstateDia) = dmean;
          p.rdata(SprayComps::pstateT) = T0;
          for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
            p.rdata(SprayComps::pstateY + spf) = Y0[spf];
          }
          if (m_sprayData->do_breakup == 2) {
            p.rdata(SprayComps::pstateBM1) = 0.;
            p.rdata(SprayComps::pstateBM2) = 0.;
          } else {
            p.rdata(SprayComps::pstateBM1) = phi2;
            p.rdata(SprayComps::pstateBM2) = phi3;
          }

          p.rdata(SprayComps::pstateFilmHght) = 0.;
          p.rdata(SprayComps::pstateN0) = N_s;
          p.rdata(SprayComps::pstateNumDens) = N_s;
          for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
#if AMREX_SPACEDIM == 3
            Real psi = rand * 2. * M_PI;
            Real pvel = vel0[dir] + Utan * (std::sin(psi) * tanPsi[dir] +
                                            std::cos(psi) * tanBeta[dir]);
#else
            Real sgn = std::copysign(1., 0.5 - rand);
            Real pvel = vel0[dir] + sgn * Utan * tanBeta[dir];
#endif
            p.pos(dir) = loc0[dir] + sub_dt * pvel;
            p.rdata(SprayComps::pstateVel + dir) = pvel;
          }
          bool where = Where(p, pld);
          if (!where) {
            amrex::Abort("Bad breakup particle");
          }
          host_particles[ind].push_back(p);
        }
      }
    }
  }
  for (auto& kv : host_particles) {
    auto grid = kv.first.first;
    auto tile = kv.first.second;
    const auto& src_tile = kv.second;
    auto& dst_tile = GetParticles(level)[std::make_pair(grid, tile)];
    auto old_size = dst_tile.GetArrayOfStructs().size();
    auto new_size = old_size + src_tile.size();
    dst_tile.resize(new_size);
    // Copy the AoS part of the host particles to the GPU
    Gpu::copy(
      Gpu::hostToDevice, src_tile.begin(), src_tile.end(),
      dst_tile.GetArrayOfStructs().begin() + old_size);
  }
}
