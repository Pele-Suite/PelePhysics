#ifndef SPRAYJET_H
#define SPRAYJET_H

#include "DistBase.H"
#include <AMReX_RealVect.H>
#include <AMReX_Geometry.H>

class SprayJet
{
public:
  // Default constructor
  SprayJet() = default;

  // Constructor for assigning parameters directly
  SprayJet(
    const std::string& jet_name,
    const amrex::Geometry& geom,
    const amrex::RealVect jet_cent,
    const amrex::RealVect jet_norm,
    const amrex::Real spread_angle,
    const amrex::Real jet_dia,
    const amrex::Real jet_vel,
    const amrex::Real mass_flow,
    const amrex::Real jet_temp,
    const amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> jet_Y,
    const std::string& dist_type,
    const amrex::Real start_time = 0.,
    const amrex::Real end_time = 1.E8,
    const amrex::Real phi_swirl = 0.,
    bool hollow_spray = false,
    const amrex::Real hollow_spread = 0.);

  // Constructor where parameters are set from input file
  SprayJet(const std::string& jet_name, const amrex::Geometry& geom);

  void readROI(
    const std::string roi_file,
    const amrex::Real rho_part,
    const amrex::Real cd);

  virtual ~SprayJet() = default;

  const amrex::RealVect& jet_norm() const { return m_norm; }
  const amrex::RealVect& jet_cent() const { return m_cent; }
  amrex::Real jet_vel(const amrex::Real time) const
  {
    if (m_useROI) {
      return interpolateVel(time);
    }
    return m_jetVel;
  }
  const amrex::Real& max_jet_vel() const { return m_maxJetVel; }
  const amrex::Real& jet_dia() const { return m_jetDia; }
  const amrex::Real& spread_angle() const { return m_spreadAngle; }
  amrex::Real mass_flow_rate(const amrex::Real time) const
  {
    if (m_useROI) {
      return interpolateMdot(time);
    }
    return m_massFlow;
  }
  const amrex::Real& Y(const int spf) const { return m_jetY[spf]; }
  const amrex::Real& T() const { return m_jetT; }
  const amrex::Real& start_time() const { return m_startTime; }
  const amrex::Real& end_time() const { return m_endTime; }
  const amrex::Real& num_ppp() const { return m_numPPP; }
  const std::string& jet_name() const { return m_jetName; }
  int Proc() const { return m_proc; }

  // Call this before using spray jet
  bool jet_active(const amrex::Real time) const
  {

    if (!m_use_Fluentdpmfile) {
      if (m_jetDia < 0.) {
        amrex::Abort("Jet diameter is not set");
      } else if (m_avgDia < 0.) {
        amrex::Abort("Average droplet diameter is not set");
      } else if (m_jetT < 0.) {
        amrex::Abort("Droplet temperature is not set");
      }
    }
    bool check = true;
    if (time < m_startTime || time > m_endTime) {
      check = false;
    }
    return check;
  }

  bool hollow_spray() const { return m_hollowSpray; }

  /// Returns the average values given for the jet
  amrex::Real get_avg_dia() const { return m_avgDia; }

  amrex::Real get_avg_T() const { return m_jetT; }

  amrex::Real get_avg_Y(const int spf) const { return m_jetY[spf]; }

  bool use_FluentDPMFile() const { return m_use_Fluentdpmfile; }

  void set_initial_injection_flow_time(amrex::Real time)
  {
    m_flow_time_initial_injection = time;
  }

  void set_initial_injection_dpm_time(amrex::Real time)
  {
    m_dpm_time_initial_injection = time;
  }

  void set_final_injection_flow_time()
  {
    const amrex::Real largenum = 1e8;

    if (is_dpm_periodic) {
      m_flow_time_final_injection = largenum;
    } else {
      m_flow_time_final_injection = m_flow_time_initial_injection +
                                    m_dpm_time_final -
                                    m_dpm_time_initial_injection;
      m_endTime = m_flow_time_final_injection + largenum;
    }
  }

  void set_jet_vel(amrex::Real jet_vel) { m_jetVel = jet_vel; }

  void set_mass_flow(amrex::Real mass_flow) { m_massFlow = mass_flow; }

  void set_jet_norm(amrex::RealVect jet_norm)
  {
    m_norm = jet_norm;
    amrex::Real mag = m_norm.vectorLength();
    m_norm /= mag;
  }

  void set_inj_proc(int inj_proc)
  {
    m_proc = inj_proc;
    if (m_proc < 0 || m_proc > amrex::ParallelDescriptor::NProcs()) {
      amrex::Abort("SprayJet proc not valid");
    }
  }

  void set_jet_endtime(amrex::Real end_time) { m_endTime = end_time; }

  void set_jet_cent(amrex::RealVect jet_cent, const amrex::Geometry& geom)
  {
    m_cent = jet_cent;
    check_jet_cent(geom);
  }

  void check_jet_cent(const amrex::Geometry& geom) const
  {
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
      if (m_cent[dir] > geom.ProbHi(dir) || m_cent[dir] < geom.ProbLo(dir)) {
        amrex::Abort("Jet center outside of problem domain");
      }
    }
  }

  void set_mass_frac(const amrex::Real* Y_jet)
  {
    amrex::Real sumtest = 0.;
    for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
      m_jetY[spf] = Y_jet[spf];
      sumtest += m_jetY[spf];
    }
    if (amrex::Math::abs(1. - sumtest) > 1.E-8) {
      amrex::Abort("Jet mass fractions must sum to 1");
    }
  }

  void set_override_injection_plane(int dir, amrex::Real loc)
  {
    m_override_inj_plane_dir = dir;
    m_override_inj_plane_loc = loc;
  }

  void set_dpm_periodic(bool b) { is_dpm_periodic = b; }

  amrex::Real get_flow_time_initial_injection() const
  {
    return m_flow_time_initial_injection;
  }
  amrex::Real get_dpm_time_initial_injection() const
  {
    return m_dpm_time_initial_injection;
  }

  amrex::Real dpm_final_time() const { return m_dpm_time_final; }
  amrex::Real dpm_intial_time() const { return m_dpm_time_initial; }

  int get_override_injection_plane_dir() const
  {
    return m_override_inj_plane_dir;
  }
  amrex::Real get_override_injection_plane_loc() const
  {
    return m_override_inj_plane_loc;
  }

  void set_start_time(amrex::Real start_time) { m_startTime = start_time; }
  void set_end_time(amrex::Real end_time) { m_endTime = end_time; }
  void set_num_ppp(amrex::Real num_ppp) { m_numPPP = num_ppp; }

  void reset_sum()
  {
    m_sumInjMass = -1;
    m_sumInjTime = -1;
  }

  /**
     Function for creating new particle parameters based on jet location. Note
     all angles must be in radians.
     @param[in] time Current solution time
     @param[in] phi_radial Azimuthal angle of particle location in jet CS
     @param[in] cur_radius Current radial location in jet CS
     @param[out] umag Magnitude of velocity
     @param[out] theta_spread Spread angle for velocity relative to jet norm
     @param[out] phi_swirl Azimuthal angle for tangential velocity component
     @param[out] dia_part Droplet diameter
     @param[out] T_part Droplet temperature
     @param[out] Y_part Droplet mass fractions
  */
  virtual bool get_new_particle(
    const amrex::Real time,
    const amrex::Real& phi_radial,
    const amrex::Real& cur_radius,
    amrex::Real& umag,
    amrex::Real& theta_spread,
    amrex::Real& phi_swirl,
    amrex::Real& dia_part,
    amrex::Real& T_part,
    amrex::Real* Y_part);

  virtual bool get_new_particle(amrex::Real* Y_part);

  /**
     Solve for transformed location and velocity based on provided angles and
     radius.
     @param[in] theta_spread Spread angle for velocity relative to jet norm
     @param[in] phi_radial Azimuthal angle of particle location in jet CS
     @param[in] cur_radius Current radial location of particle
     @param[in] umag Particle velocity magnitude
     @param[in] phi_swirl Azimuthal angle for tangential velocity component
     @param[out] part_vel Particle velocity in jet CS to be solved for
     @param[out] part_loc Particle location in jet CS to be solved for
   */
  void transform_loc_vel(
    const amrex::Real theta_spread,
    const amrex::Real phi_radial,
    const amrex::Real cur_radius,
    const amrex::Real umag,
    const amrex::Real phi_swirl,
    amrex::RealVect& part_vel,
    amrex::RealVect& part_loc)
  {
#if AMREX_SPACEDIM == 3
    amrex::Real norm_mag = m_norm.vectorLength();
    amrex::Real theta_jet = std::acos(m_norm[2] / norm_mag);
    amrex::Real phi_jet =
      std::atan2(m_norm[1] / norm_mag, m_norm[0] / norm_mag);
    amrex::Real sp1 = std::sin(phi_jet);
    amrex::Real cp1 = std::cos(phi_jet);
    amrex::Real sp2 = std::sin(phi_radial);
    amrex::Real cp2 = std::cos(phi_radial);
#else
    amrex::Real theta_jet = std::atan2(m_norm[1], m_norm[0]) + M_PI / 2.;
    amrex::ignore_unused(phi_radial, phi_swirl);
#endif
    amrex::Real st1 = std::sin(theta_jet);
    amrex::Real ct1 = std::cos(theta_jet);
    amrex::Real st2 = std::sin(theta_spread);
    amrex::Real ct2 = std::cos(theta_spread);
#if AMREX_SPACEDIM == 3
    amrex::RealVect dp(AMREX_D_DECL(
      cp1 * cp2 * ct1 - sp1 * sp2, sp1 * cp2 * ct1 + cp1 * sp2,
      -std::sin(theta_jet) * cp2));
    // Add phi_swirl for velocity
    amrex::Real phivel = phi_radial + phi_swirl;
    sp2 = std::sin(phivel);
    cp2 = std::cos(phivel);
    amrex::Real v1 = st1 * ct2 + st2 * cp2 * ct1;
    part_vel = {
      cp1 * v1 - sp1 * sp2 * st2, sp1 * v1 + sp2 * st2 * cp1,
      ct1 * ct2 - st1 * st2 * cp2};
#else
    amrex::RealVect dp(ct1, st1);
    part_vel = {st1 * ct2 - st2 * ct1, -ct1 * ct2 - st1 * st2};
#endif
    part_loc = m_cent + cur_radius * dp;
    part_vel *= umag;
  }

  amrex::Real
  interpolateROI(const amrex::Real& time, const amrex::Real* vals) const
  {
    int i = 0;
    amrex::Real ctime = inject_time[0];
    while (ctime < time) {
      ctime = inject_time[++i];
    }
    const amrex::Real time1 = inject_time[i - 1];
    const amrex::Real time2 = inject_time[i];
    const amrex::Real invt = (time - time1) / (time2 - time1);
    const amrex::Real val1 = vals[i - 1];
    const amrex::Real val2 = vals[i];
    return val1 + (val2 - val1) * invt;
  }

  // Rotate vec into the user coordinate system, and translate it if it is a
  // position (not a velocity)
  void transform_to_new_cs(
    amrex::Real vec[AMREX_SPACEDIM], const bool is_position = true)
  {
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> trans = {0.};
    if (is_position) {
      trans = m_translation;
    }
#if AMREX_SPACEDIM == 2
    const amrex::Real x = vec[0];
    const amrex::Real y = vec[1];
    vec[0] = trans[0] + (m_trans_matrix[0] * x + m_trans_matrix[1] * y);
    vec[1] = trans[1] + (m_trans_matrix[2] * x + m_trans_matrix[3] * y);
#else
    const amrex::Real x = vec[0];
    const amrex::Real y = vec[1];
    const amrex::Real z = vec[2];
    vec[0] = trans[0] + (m_trans_matrix[0] * x + m_trans_matrix[1] * y +
                         m_trans_matrix[2] * z);
    vec[1] = trans[1] + (m_trans_matrix[3] * x + m_trans_matrix[4] * y +
                         m_trans_matrix[5] * z);
    vec[2] = trans[2] + (m_trans_matrix[6] * x + m_trans_matrix[7] * y +
                         m_trans_matrix[8] * z);
#endif
  }

  amrex::Real interpolateMdot(const amrex::Real time) const
  {
    return interpolateROI(time, inject_mass.dataPtr());
  }

  amrex::Real interpolateVel(const amrex::Real time) const
  {
    return interpolateROI(time, inject_vel.dataPtr());
  }

  amrex::Real m_sumInjMass = -1.;
  amrex::Real m_sumInjTime = -1.;
  // Minimum parcels to inject at a time
  amrex::Real m_minParcel = 1.;

  // Total injection mass and time
  amrex::Real m_totalInjMass = 0.;
  amrex::Real m_totalInjTime = 0.;

  bool m_use_Fluentdpmfile;
  bool m_rstrt_Fltdpmsim_from_nonFltDPMChckPointFile = false;
  std::string m_FluentDPMFile;
  long int m_dpm_num_of_time_instants = 0;
  bool is_dpm_periodic = true;

  amrex::Real m_cur_inj_dpm_time =
    0.0; // The current DPM time at which injection has occurred
  amrex::Real m_nxt_inj_dpm_time =
    0.0; // The next DPM time at which injection is to occur
  amrex::Real m_nxt_inj_flw_time =
    0.0;                    // The next flow time at which injection is to occur
  amrex::Real dt_dpm = 0.0; // The time between two consecutive injections. Is
                            // updated after every injection

protected:
  // Member data
  // Adding the following variables to include dpm injection from external file
  int m_override_inj_plane_dir = -1;
  amrex::Real m_override_inj_plane_loc = 0.0;

  amrex::Real m_dpm_time_initial =
    0.0; // The minimum time that appears in the Fluent DPM file.
  amrex::Real m_dpm_time_final =
    -1.0; // The maximum time that appears in the Fluent DPM file.
  amrex::Real m_dpm_time_initial_injection =
    0.0; // The initial DPM time that injection starts for a particular
         // simulation. Is constant for a simulation even after restarts
  amrex::Real m_flow_time_initial_injection =
    -1.0; // The initial Flow time that injection starts for a particular
          // simulation. Is constant for a simulation even after restarts
  amrex::Real m_flow_time_final_injection =
    -1.0; // The initial Flow time that injection starts for a particular
          // simulation. Is constant for a simulation even after restarts

  std::string m_jetName;
  amrex::RealVect m_norm = amrex::RealVect::TheZeroVector();
  amrex::RealVect m_cent = amrex::RealVect::TheZeroVector();
  amrex::Real m_spreadAngle = 0.;
  amrex::Real m_swirlAngle = 0.;
  amrex::Real m_jetDia = -1.;
  amrex::Real m_jetVel = -1.;
  amrex::Real m_maxJetVel = -1.;
  amrex::Real m_massFlow = -1.;
  amrex::Real m_jetT = -1.;
  amrex::GpuArray<amrex::Real, SPRAY_FUEL_NUM> m_jetY = {{0.}};

  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM * AMREX_SPACEDIM> m_trans_matrix =
    {AMREX_D_DECL(
      AMREX_D_DECL(1.0, 0.0, 0.0),
      AMREX_D_DECL(0.0, 1.0, 0.0),
      AMREX_D_DECL(0.0, 0.0, 1.0))};
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_translation = {
    AMREX_D_DECL(0.0, 0.0, 0.0)};

  amrex::Real m_avgDia = -1.;
  std::unique_ptr<DistBase> m_dropDist;
  amrex::Real m_startTime = 0.;
  amrex::Real m_endTime = 1.E8;
  bool m_hollowSpray = false;
  amrex::Real m_hollowSpread = 0.;
  amrex::Real m_numPPP = -1.;
  int m_proc = 0;
  bool m_useROI = false;
  amrex::Vector<amrex::Real> inject_time;
  amrex::Vector<amrex::Real> inject_mass;
  amrex::Vector<amrex::Real> inject_vel;
};

#endif
