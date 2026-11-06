#ifndef SPRAYINJECTION_H
#define SPRAYINJECTION_H
#include "SprayParticles.H"

/*
 These are generalized initialization and injection routines to be called in a
 problem-specific SprayParticlesInitInsert.cpp file, when necessary.
*/

std::ifstream&
GotoLine(std::ifstream& file, long int num)
{
  file.seekg(std::ios::beg);
  for (int i = 0; i < num - 1; ++i) {
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return file;
}

void
SprayParticleContainer::sprayInjectionfromDPMFile(
  const amrex::Real time,
  SprayJet* spray_jet,
  const amrex::Real sim_dt,
  const int level)
{

  if (!spray_jet->jet_active(time) || sim_dt <= 0.) {
    if (m_verbose > 0) {
      amrex::Print() << "Warning! Spray jet " << spray_jet->jet_name()
                     << " is inactive and hence 0 particles are injected.\n";
    }
    return;
  }

  if (amrex::ParallelDescriptor::IOProcessor()) {
    amrex::Real dt = sim_dt;
    const amrex::Real smallnum = 1e-15;

    while ((std::fabs(time - spray_jet->m_nxt_inj_flw_time) < smallnum ||
            time >= spray_jet->m_nxt_inj_flw_time) &&
           spray_jet->jet_active(time)) {
      // Line numbers in the dpm file
      long int lineno_current_dpm_time_begin = 0;
      long int lineno_current_dpm_time_end = 0;
      long int lineno_next_dpm_time_begin = 0;
      long int lineno_first_dpm_time_begin;
      long int lineno_last_dpm_time_end = 0;

      long int ln; // iterating index
      amrex::Real dummy_time = -1;
      amrex::Real part_loc[3];
      amrex::Real part_vel[3];
      amrex::Real dia, temperature, parcel_mass, particle_mass, nppp;
      const amrex::Real verysmallnum = 1.e-15;

      long int dummy_line_no;
      std::ifstream is_dpm(spray_jet->m_FluentDPMFile, std::ios::in);
      is_dpm >> dummy_line_no; // Reading total number of time instants to a
                               // dummy variable

      long int ctr = 0;

      // Locate the current injection dpm time in the dpm file
      while (std::fabs(dummy_time - spray_jet->m_cur_inj_dpm_time) >
               verysmallnum &&
             ctr <= spray_jet->m_dpm_num_of_time_instants) {
        if (ctr == 0) {
          is_dpm >> dummy_time >> lineno_first_dpm_time_begin;
          dummy_line_no = lineno_first_dpm_time_begin;
        } else {
          is_dpm >> dummy_time >> dummy_line_no;
        }
        ctr++;
      }
      // If after one sweep, current injection dpm time is not found
      if (ctr > spray_jet->m_dpm_num_of_time_instants) {
        amrex::Abort(
          "Error! Unable to find the current injection dpm time in file.\n");
      }

      // If at the last dpm time
      if (m_verbose > 0) {
        amrex::Print() << "\nCollecting particle data at DPM time "
                       << dummy_time << std::endl;
      }
      if (
        dummy_time == spray_jet->m_cur_inj_dpm_time &&
        spray_jet->m_cur_inj_dpm_time ==
          spray_jet->dpm_final_time()) // The last time in the dpm file
      {
        if (m_verbose > 0) {
          amrex::Print() << " (last dpm time) \n";
        }
        is_dpm >> lineno_last_dpm_time_end; // one past the last line no
        lineno_current_dpm_time_begin = dummy_line_no;
        lineno_current_dpm_time_end = lineno_last_dpm_time_end - 1;

        // if dpm file is periodic, the next time instant is time=0
        if (spray_jet->is_dpm_periodic) {

          spray_jet->m_nxt_inj_dpm_time = spray_jet->dpm_intial_time();
          // lineno_next_dpm_time_begin = lineno_first_dpm_time_begin;
        } else {
          amrex::Print() << "\nWARNING! No more injection time found in DPM "
                            "file. To continue the same injection again, set "
                            "'is_dpm_periodic' to true.\n ";
          spray_jet->m_nxt_inj_dpm_time = -1;
          spray_jet->set_end_time(time - verysmallnum);
        }
      } else if (
        dummy_time == spray_jet->m_cur_inj_dpm_time &&
        spray_jet->m_cur_inj_dpm_time != spray_jet->dpm_final_time()) {
        lineno_current_dpm_time_begin = dummy_line_no;
        is_dpm >> spray_jet->m_nxt_inj_dpm_time >> lineno_next_dpm_time_begin;
        lineno_current_dpm_time_end = lineno_next_dpm_time_begin - 1;
        spray_jet->dt_dpm =
          spray_jet->m_nxt_inj_dpm_time - spray_jet->m_cur_inj_dpm_time;
      } else {
        amrex::Abort(
          "Error! Unable to find the current injection dpm time in "
          "the dpm file");
      }
      is_dpm.close();

      // Now reading the particle data
      std::ifstream is_dpm1(spray_jet->m_FluentDPMFile, std::ios::in);
      GotoLine(
        is_dpm1, lineno_current_dpm_time_begin); // Skip lines in dpm file till
                                                 // the current injection time
      for (ln = 0;
           ln <= lineno_current_dpm_time_end - lineno_current_dpm_time_begin;
           ln++) {
        amrex::Real pmov1 =
          amrex::Random(); // Random number to make sure particle is advanced a
                           // little bit downstream
        amrex::Real pmov = amrex::Random();
        amrex::Real Y_part[SPRAY_FUEL_NUM]; // Spray fuel mass fraction
        bool set_part_Y = false;

        // Reading parcel/particle details from dpm input file
        is_dpm1 >> dummy_time >> part_loc[0] >> part_loc[1] >> part_loc[2] >>
          part_vel[0] >> part_vel[1] >> part_vel[2] >> dia >> temperature >>
          parcel_mass >> particle_mass >> nppp;
        if (!is_dpm1) {
          amrex::Print()
            << "Warning: DPM file " << spray_jet->m_FluentDPMFile
            << " has fewer data lines than its index implies (block t = "
            << spray_jet->m_cur_inj_dpm_time << "); stopping this block\n";
          break;
        }

        amrex::ParticleLocData pld;
        std::map<std::pair<int, int>, amrex::Gpu::HostVector<ParticleType>>
          host_particles;
        ParticleType p;
        p.id() = ParticleType::NextID();
        p.cpu() = amrex::ParallelDescriptor::MyProc();

        // Setting particle properties
        // Transforming the particle coordinates to user-specified coordinate
        // system
        spray_jet->transform_to_new_cs(part_loc);

        if (
          spray_jet->get_override_injection_plane_dir() !=
          -1) // Overriding dpm input file particle location on a plane with
              // user specified location
        {
          part_loc[spray_jet->get_override_injection_plane_dir()] =
            spray_jet->get_override_injection_plane_loc();
        }

        spray_jet->transform_to_new_cs(
          part_vel); // Transforming the particle velocity to user-specified
                     // coordinate system
        AMREX_D_TERM(p.rdata(SprayComps::pstateVel) = (part_vel[0] + pmov1);
                     , // Setting the particle velocities
                     p.rdata(SprayComps::pstateVel + 1) = (part_vel[1] + pmov1);
                     , p.rdata(SprayComps::pstateVel + 2) =
                         (part_vel[2] + pmov1););

        for (int dir = 0; dir < AMREX_SPACEDIM;
             ++dir) // Setting the particle coordinates
        {
          p.pos(dir) =
            part_loc[dir] + pmov * dt * p.rdata(SprayComps::pstateVel + dir);
        }

        p.rdata(SprayComps::pstateDia) = dia; // Setting the particle density
        p.rdata(SprayComps::pstateT) =
          temperature;          // Setting the particle temperature
        set_part_Y =            // NOLINT(clang-analyzer-deadcode.DeadStores)
          spray_jet             // NOLINT(clang-analyzer-deadcode.DeadStores)
            ->get_new_particle( // NOLINT(clang-analyzer-deadcode.DeadStores)
              Y_part);          // NOLINT(clang-analyzer-deadcode.DeadStores)

        if (SPRAY_FUEL_NUM > 1 && set_part_Y) // Setting particle mass fractions
        {
          for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
            p.rdata(SprayComps::pstateY + spf) = Y_part[spf];
          }
        } else {
          p.rdata(SprayComps::pstateY) = 1.;
        }

        // If KHRT is used, BM1 is shed mass
        // If TAB is used, BM1 is y
        p.rdata(SprayComps::pstateBM1) = 0.;
        p.rdata(SprayComps::pstateBM2) = 0;
        p.rdata(SprayComps::pstateFilmHght) = 0.;
        p.rdata(SprayComps::pstateN0) = 2;
        p.rdata(SprayComps::pstateNumDens) = nppp;

        bool where = Where(p, pld);
        if (!where) {
          amrex::Print()
            << "Warning: DPM parcel outside the domain discarded (jet "
            << spray_jet->jet_name() << ", pos = " << p.pos(0) << " "
            << p.pos(1) << " " << p.pos(2) << ")\n";
          continue;
        }

        spray_jet->m_totalInjMass += parcel_mass;

        std::pair<int, int> ind(pld.m_grid, pld.m_tile);
        host_particles[ind].push_back(p);

        for (auto& kv : host_particles) {
          auto grid = kv.first.first;
          auto tile = kv.first.second;
          const auto& src_tile = kv.second;
          auto& dst_tile = GetParticles(level)[std::make_pair(grid, tile)];
          auto old_size = dst_tile.GetArrayOfStructs().size();
          auto new_size = old_size + src_tile.size();
          dst_tile.resize(new_size);
          // Copy the AoS part of the host particles to the GPU
          amrex::Gpu::copy(
            amrex::Gpu::hostToDevice, src_tile.begin(), src_tile.end(),
            dst_tile.GetArrayOfStructs().begin() + old_size);
        }
      }
      is_dpm1.close();

      if (m_verbose > 0) {
        amrex::Print() << "Injected " << ln << " particles through "
                       << spray_jet->jet_name()
                       << " at flow time = " << spray_jet->m_nxt_inj_flw_time
                       << "\n";
      }
      spray_jet->m_nxt_inj_flw_time += spray_jet->dt_dpm;
      spray_jet->m_cur_inj_dpm_time = spray_jet->m_nxt_inj_dpm_time;

      if (m_verbose > 0) {
        amrex::Print() << "\tNext injection flow time = "
                       << spray_jet->m_nxt_inj_flw_time << "\n";
        amrex::Print() << "\tNext injection DPM time = "
                       << spray_jet->m_cur_inj_dpm_time << "\n\n";
      }
    }
  }
  Redistribute();
}

void
SprayParticleContainer::sprayInjection(
  const amrex::Real time,
  SprayJet* spray_jet,
  const amrex::Real sim_dt,
  const int level)
{
  if (
    !spray_jet->jet_active(time) || sim_dt <= 0. ||
    spray_jet->jet_vel(time) < 0.) {
    return;
  }
  int curProc = amrex::ParallelDescriptor::MyProc();
  int injProc = spray_jet->Proc();
  if (curProc != injProc) {
    return;
  }
  const SprayData* fdat = m_sprayData;
  amrex::Real rho_avg = 0.;
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    rho_avg += spray_jet->get_avg_Y(spf) /
               fdat->liqprops.rho_i(spray_jet->get_avg_T(), spf);
  }
  rho_avg = 1. / rho_avg;
  const amrex::Real Pi_six = M_PI / 6.;
  // Check if mass must be injected across multiple timesteps
  const amrex::Real avg_dia = spray_jet->get_avg_dia();
  const amrex::Real avg_mass = Pi_six * rho_avg * std::pow(avg_dia, 3);
  const amrex::Real min_dia =
    std::cbrt(6. * SprayUnits::min_mass / (M_PI * rho_avg));
  if (avg_dia < min_dia || avg_mass < SprayUnits::min_mass) {
    amrex::Abort(
      "Average droplet size too small, floating point issues expected");
  }
  amrex::Real initial_bm2 = 0.;
  if (fdat->do_breakup == 2) {
    // If KHRT is used, BM2 is RT time
    initial_bm2 = -1.;
  }
  // Minimum number of parcels we want injected at a time
  amrex::Real min_inj_parcel = spray_jet->m_minParcel;
  amrex::Real dt = sim_dt;
  amrex::Real mdot = spray_jet->mass_flow_rate(time);
  amrex::Real inject_mass = mdot * sim_dt;
  // See if there is uninjected mass from previous time steps
  if (spray_jet->m_sumInjMass > 0.) {
    dt += spray_jet->m_sumInjTime;
    inject_mass += spray_jet->m_sumInjMass;
  }
  // TODO: Add check to set particle number density so that the injected
  // particle volume are less than 10% of the finest cell volume
  // This is very important
  amrex::Real num_ppp = spray_jet->num_ppp();
  // Set initial injection number density
  if (num_ppp < 0.) {
    num_ppp = inject_mass / avg_mass;
    spray_jet->set_num_ppp(num_ppp);
  }
  // If we can't get enough mass to create the desired minimum parcel injection,
  // accumulate mass injection over more time steps
  if (inject_mass / (num_ppp * avg_mass) < min_inj_parcel) {
    spray_jet->m_sumInjMass = inject_mass;
    spray_jet->m_sumInjTime = dt;
    return;
  }

  amrex::ParticleLocData pld;
  std::map<std::pair<int, int>, amrex::Gpu::HostVector<ParticleType>>
    host_particles;
  amrex::Real cur_mass = 0.;
  while (cur_mass < inject_mass) {
    // Pick random percentage from 0 to 1
    amrex::Real radp = amrex::Random();
#if AMREX_SPACEDIM == 3
    if (spray_jet->hollow_spray()) {
      radp = 1.;
    }
    amrex::Real phi_radial = amrex::Random() * 2. * M_PI;
    // This determines the radial location of the particle within the jet inlet
    amrex::Real cur_rad = radp * spray_jet->jet_dia() / 2.;
#else
    if (spray_jet->hollow_spray()) {
      if (radp <= 0.5) {
        radp = 0.;
      } else {
        radp = 1.;
      }
    }
    amrex::Real phi_radial = 0.;
    amrex::Real cur_rad = (radp - 0.5) * spray_jet->jet_dia();
#endif
    // phi_radial - azimuthal angle, called phi_j in docs
    // theta_spread - inclination angle, called theta_j in docs
    // phi_swirl - azimuthal swirling component, called phi_s in docs
    amrex::Real umag, theta_spread, phi_swirl, dia_part, T_part;
    amrex::Real Y_part[SPRAY_FUEL_NUM];
    bool make_new_part = spray_jet->get_new_particle(
      time, phi_radial, cur_rad, umag, theta_spread, phi_swirl, dia_part,
      T_part, Y_part);
    if (make_new_part && dia_part > min_dia && umag > 0.) {
      amrex::RealVect part_loc, vel_part;
      spray_jet->transform_loc_vel(
        theta_spread, phi_radial, cur_rad, umag, phi_swirl, vel_part, part_loc);
      ParticleType p;
      p.id() = ParticleType::NextID();
      p.cpu() = amrex::ParallelDescriptor::MyProc();
      AMREX_D_TERM(p.rdata(SprayComps::pstateVel) = vel_part[0];
                   , p.rdata(SprayComps::pstateVel + 1) = vel_part[1];
                   , p.rdata(SprayComps::pstateVel + 2) = vel_part[2];);
      p.rdata(SprayComps::pstateT) = T_part;
      // Never add particle with less than minimum mass
      p.rdata(SprayComps::pstateDia) = dia_part;
      amrex::Real rho_part = 0.;
      if (SPRAY_FUEL_NUM > 1) {
        for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
          p.rdata(SprayComps::pstateY + spf) = Y_part[spf];
        }
        rho_part = fdat->liqprops.rho_mix(Y_part, T_part);
      } else {
        rho_part = fdat->liqprops.rho_i(T_part, 0);
        p.rdata(SprayComps::pstateY) = 1.;
      }
      // Add particles as if they have advanced some random portion of
      // dt
      amrex::Real pmov = amrex::Random();
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        p.pos(dir) =
          part_loc[dir] + pmov * dt * p.rdata(SprayComps::pstateVel + dir);
      }
      amrex::Real pmass = Pi_six * rho_part * std::pow(dia_part, 3);
      // If KHRT is used, BM1 is shed mass
      // If TAB is used, BM1 is y
      p.rdata(SprayComps::pstateBM1) = 0.;
      p.rdata(SprayComps::pstateBM2) = initial_bm2;
      p.rdata(SprayComps::pstateFilmHght) = 0.;
      p.rdata(SprayComps::pstateN0) = num_ppp;
      p.rdata(SprayComps::pstateNumDens) = num_ppp;
      amrex::Real new_mass = cur_mass + num_ppp * pmass;
      bool where = Where(p, pld);
      if (!where) {
        amrex::Abort("Bad injection particle");
      }
      std::pair<int, int> ind(pld.m_grid, pld.m_tile);
      host_particles[ind].push_back(p);
      cur_mass = new_mass;
    }
  }
  amrex::Real est_mdot = cur_mass / dt;
  // If we are over-injecting mass, increase the minimum parcels needed for
  // injection
  if (est_mdot - mdot > 0.5 * mdot) {
    spray_jet->m_minParcel += 1.;
  }
  spray_jet->m_totalInjMass += cur_mass;
  spray_jet->m_totalInjTime += dt;
  for (auto& kv : host_particles) {
    auto grid = kv.first.first;
    auto tile = kv.first.second;
    const auto& src_tile = kv.second;
    auto& dst_tile = GetParticles(level)[std::make_pair(grid, tile)];
    auto old_size = dst_tile.GetArrayOfStructs().size();
    auto new_size = old_size + src_tile.size();
    dst_tile.resize(new_size);
    // Copy the AoS part of the host particles to the GPU
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, src_tile.begin(), src_tile.end(),
      dst_tile.GetArrayOfStructs().begin() + old_size);
  }
  spray_jet->reset_sum();
}

amrex::IntVect
unflatten_particles(const amrex::ULong idx, const amrex::IntVect& max_parts)
{
  amrex::IntVect indx;
  amrex::ULong cidx = idx;
  amrex::ULong d1 = max_parts[0];
#if AMREX_SPACEDIM > 2
  amrex::ULong d2 = max_parts[1];
  indx[2] = int(cidx / (d1 * d2));
  cidx -= amrex::ULong(indx[2]) * d1 * d2;
#endif
  indx[1] = int(cidx / d1);
  indx[0] = int(cidx % d1);
  return indx;
}

void
SprayParticleContainer::uniformSprayInit(
  const amrex::IntVect num_part,
  const amrex::RealVect vel_part,
  const amrex::Real dia_part,
  const amrex::Real T_part,
  const amrex::Real* Y_part,
  const int level,
  const int num_redist,
  const amrex::Real num_ppp)
{
  const int MyProc = amrex::ParallelDescriptor::MyProc();
  const int NProcs = amrex::ParallelDescriptor::NProcs();
  int NRedist = num_redist;
  if (NRedist <= 0) {
    NRedist = 1;
    if (NProcs <= 1024) {
      NRedist = 2;
    } else if (NProcs <= 2048) {
      NRedist = 4;
    } else if (NProcs <= 4096) {
      NRedist = 8;
    }
  }
  // Reference values for the particles
  amrex::Real part_vals[NSR_SPR];
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    part_vals[SprayComps::pstateVel + dir] = vel_part[dir];
  }
  part_vals[SprayComps::pstateT] = T_part;
  part_vals[SprayComps::pstateDia] = dia_part;
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    part_vals[SprayComps::pstateY + spf] = Y_part[spf];
  }
  const SprayData* fdat = m_sprayData;
  amrex::Real initial_bm2 = 0.;
  if (fdat->do_breakup == 2) {
    // If KHRT is used, BM2 is RT time
    initial_bm2 = -1.;
  }
  part_vals[SprayComps::pstateNumDens] = num_ppp;
  part_vals[SprayComps::pstateN0] = num_ppp;
  part_vals[SprayComps::pstateBM1] = 0.;
  part_vals[SprayComps::pstateBM2] = initial_bm2;
  part_vals[SprayComps::pstateFilmHght] = 0.;
  const amrex::RealVect dx_part(AMREX_D_DECL(
    Geom(level).ProbLength(0) / amrex::Real(num_part[0]),
    Geom(level).ProbLength(1) / amrex::Real(num_part[1]),
    Geom(level).ProbLength(2) / amrex::Real(num_part[2])));
  AMREX_D_TERM(
    amrex::ULong np0 = num_part[0];, amrex::ULong np1 = num_part[1];
    , amrex::ULong np2 = num_part[2];);
  const amrex::ULong total_part_num = AMREX_D_TERM(np0, *np1, *np2);
  amrex::ULong parts_pp = total_part_num / amrex::ULong(NProcs);
  // Number of particles per processor to be initialized
  amrex::ULong cur_parts_pp = parts_pp;
  // Give any remaining particles to the last processor
  if (MyProc == NProcs - 1) {
    cur_parts_pp += (total_part_num % amrex::ULong(NProcs));
  }
  // Starting particle for this processor
  const amrex::ULong first_part = amrex::ULong(MyProc) * parts_pp;
  amrex::Gpu::HostVector<ParticleType> nparticles;
  for (amrex::ULong prc = 0; prc < cur_parts_pp; ++prc) {
    amrex::ULong cur_part = first_part + prc;
    amrex::IntVect indx = unflatten_particles(cur_part, num_part);
    ParticleType p;
    p.id() = ParticleType::NextID();
    p.cpu() = amrex::ParallelDescriptor::MyProc();
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      p.pos(dir) = (amrex::Real(indx[dir]) + 0.5) * dx_part[dir];
    }
    for (int n = 0; n < NSR_SPR; ++n) {
      p.rdata(n) = part_vals[n];
    }
    nparticles.push_back(p);
  }
  amrex::ParticleLocData pld;
  // Only copy particle data for certain processors at a time
  int NRchunk = NProcs / NRedist;
  for (int nr = 0; nr < NRedist; ++nr) {
    std::map<std::pair<int, int>, amrex::Gpu::HostVector<ParticleType>>
      host_particles;
    if (m_verbose > 0) {
      amrex::Print() << "Redistributing from processor " << nr * NRchunk
                     << " to " << (nr + 1) * NRchunk - 1 << '\n';
    }
    for (int which = nr * NRchunk; which < (nr + 1) * NRchunk; ++which) {
      if (which == MyProc) {
        while (!nparticles.empty()) {
          // Retrieve the last particle entry and add it to host_particles
          ParticleType& p = nparticles.back();
          bool where = Where(p, pld);
          if (!where) {
            amrex::Abort("Bad particle");
          }
          std::pair<int, int> ind(pld.m_grid, pld.m_tile);
          host_particles[ind].push_back(p);
          // Remove the particle just read
          nparticles.pop_back();
        }
      } // if (which == MyProc)
    } // for (int which ...
    for (auto& kv : host_particles) {
      auto grid = kv.first.first;
      auto tile = kv.first.second;
      const auto& src_tile = kv.second;
      auto& dst_tile = GetParticles(level)[std::make_pair(grid, tile)];
      auto old_size = dst_tile.GetArrayOfStructs().size();
      auto new_size = old_size + src_tile.size();
      dst_tile.resize(new_size);

      // Copy the AoS part of the host particles to the GPU
      amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, src_tile.begin(), src_tile.end(),
        dst_tile.GetArrayOfStructs().begin() + old_size);
    }
    Redistribute();
  } // for (int nr ...
  // Now copy over any remaining processors
  for (int which = NRedist * NRchunk; which < NProcs; ++which) {
    std::map<std::pair<int, int>, amrex::Gpu::HostVector<ParticleType>>
      host_particles;
    if (m_verbose > 0) {
      amrex::Print() << "Redistributing from processor " << NRedist * NRchunk
                     << " to " << NProcs << '\n';
    }
    if (which == MyProc) {
      while (!nparticles.empty()) {
        // Retrieve the last particle entry and add it to host_particles
        ParticleType& p = nparticles.back();
        Where(p, pld);
        std::pair<int, int> ind(pld.m_grid, pld.m_tile);
        host_particles[ind].push_back(p);
        // Remove the particle just read
        nparticles.pop_back();
      }
    } // if (which == MyProc)
    for (auto& kv : host_particles) {
      auto grid = kv.first.first;
      auto tile = kv.first.second;
      const auto& src_tile = kv.second;
      auto& dst_tile = GetParticles(level)[std::make_pair(grid, tile)];
      auto old_size = dst_tile.GetArrayOfStructs().size();
      auto new_size = old_size + src_tile.size();
      dst_tile.resize(new_size);

      // Copy the AoS part of the host particles to the GPU
      amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, src_tile.begin(), src_tile.end(),
        dst_tile.GetArrayOfStructs().begin() + old_size);
    }
    Redistribute();
  } // for (int which ...
}
#endif
