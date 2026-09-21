#include <TurbForcing.H>

namespace pele::physics::turbforcing {
void
TurbForcing::init(amrex::GeometryData const& geomdata)
{

  // Forcing only in 3D only.
  AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM == 3);

  const amrex::Real* problo = geomdata.ProbLo();
  const amrex::Real* probhi = geomdata.ProbHi();

  constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;
  constexpr amrex::Real TwoPi = 2.0 * Pi;

  const amrex::Real Lx = probhi[0] - problo[0];
  const amrex::Real Ly = probhi[1] - problo[1];
  amrex::Real Lz = probhi[2] - problo[2];
  AMREX_ALWAYS_ASSERT(
    Lx == Ly); // Forcing requires that Lx==Ly, Lz can be longer

  // parse in variables
  amrex::ParmParse pp("turbforce");
  pp.query("v", m_tfp.m_verbose);
  // fine scale tuning of the forcing
  pp.query("force_scale_fudge", m_tfp.m_force_scale_fudge);
  // target velocity fluctuation
  pp.query("urms", m_tfp.m_urms);
  // time offset (e.g. for starting from precursor turb sim)
  pp.query("time_offset", m_tfp.m_time_offset);
  // allow peridodic reproduction in z
  // can be used as a flag i.e. =1 for a factor 2
  // or as the factor itself
  pp.query("hack_lz", m_tfp.m_hack_lz);

  // Tuned by Andrew Aspden, change at own risk!
  // fast force coarsening factor
  pp.query("ff_factor", m_tfp.m_ff_factor);
  // largest mode to force
  pp.query("nmodes", m_tfp.m_nmodes);
  // reduce amplitude of modes used for breaking symmetry
  pp.query("forcing_epsilon", m_tfp.m_forcing_epsilon);
  // shape the spectrum of the forcing
  pp.query("spectrum_type", m_tfp.m_spectrum_type);
  // reduce the impact of any zero mode
  pp.query("moderate_zero_modes", m_tfp.m_moderate_zero_modes);
  pp.query("mode_start", m_tfp.m_mode_start);

  if (m_tfp.m_hack_lz > 0) {
    if (m_tfp.m_hack_lz == 1) {
      Lz = Lz / 2.0;
    } else {
      Lz = Lz / m_tfp.m_hack_lz;
    }
  }

  if (m_tfp.m_verbose > 0) {
    amrex::Print() << "================ Init Turbulent Forcing ================"
                   << std::endl;
    amrex::Print() << "Lx = " << Lx << std::endl;
    amrex::Print() << "Ly = " << Ly << std::endl;
    amrex::Print() << "Lz = " << Lz << std::endl;
  }

  //
  // calculate turbulent forcing scales (using AJA's reference values)
  //
  if (Lx == Lz) {
    m_tfp.m_fsr = 1.47e7; // cubic reference value
  } else {
    m_tfp.m_fsr = 1.41e7; // non-cubic reference value
  }
  m_tfp.m_force_scale = m_tfp.m_force_scale_fudge * m_tfp.m_fsr *
                        pow(m_tfp.m_urms / TurbForcingParm::m_fvr, 2) *
                        pow(TurbForcingParm::m_flr / Lx, 3);
  m_tfp.m_forcing_time_scale_min =
    TurbForcingParm::m_fts_min *
    (TurbForcingParm::m_fvr / TurbForcingParm::m_flr) * (Lx / m_tfp.m_urms);
  m_tfp.m_forcing_time_scale_max =
    TurbForcingParm::m_fts_max *
    (TurbForcingParm::m_fvr / TurbForcingParm::m_flr) * (Lx / m_tfp.m_urms);

  m_tfp.m_Lmin = amrex::min<amrex::Real>(Lx, amrex::min<amrex::Real>(Ly, Lz));
  m_tfp.m_kappaMax =
    static_cast<amrex::Real>(m_tfp.m_nmodes) / m_tfp.m_Lmin + 1.0e-8;
  m_tfp.m_nxmodes =
    m_tfp.m_nmodes * static_cast<int>(std::lround(Lx / m_tfp.m_Lmin));
  m_tfp.m_nymodes =
    m_tfp.m_nmodes * static_cast<int>(std::lround(Ly / m_tfp.m_Lmin));
  m_tfp.m_nzmodes =
    m_tfp.m_nmodes * static_cast<int>(std::lround(Lz / m_tfp.m_Lmin));

  if (m_tfp.m_verbose > 0) {
    amrex::Print() << "Lmin = " << m_tfp.m_Lmin << std::endl;
    amrex::Print() << "kappaMax = " << m_tfp.m_kappaMax << std::endl;
    amrex::Print() << "nxmodes = " << m_tfp.m_nxmodes << std::endl;
    amrex::Print() << "nymodes = " << m_tfp.m_nymodes << std::endl;
    amrex::Print() << "nzmodes = " << m_tfp.m_nzmodes << std::endl;
  }

  m_tfp.m_freqMin = 1.0 / m_tfp.m_forcing_time_scale_max;
  m_tfp.m_freqMax = 1.0 / m_tfp.m_forcing_time_scale_min;
  m_tfp.m_freqDiff = m_tfp.m_freqMax - m_tfp.m_freqMin;

  if (m_tfp.m_verbose > 0) {
    amrex::Print() << "force_scale = " << m_tfp.m_force_scale << "\n";
    amrex::Print() << "forcing_time_scale_min = "
                   << m_tfp.m_forcing_time_scale_min << std::endl;
    amrex::Print() << "forcing_time_scale_max = "
                   << m_tfp.m_forcing_time_scale_max << std::endl;
    amrex::Print() << "freqMin = " << m_tfp.m_freqMin << std::endl;
    amrex::Print() << "freqMax = " << m_tfp.m_freqMax << std::endl;
    amrex::Print() << "freqDiff = " << m_tfp.m_freqDiff << std::endl;
  }

  // tmp CPU storage that holds everything in one flat array
  constexpr int num_elmts = TurbForcingParm::m_array_size *
                            TurbForcingParm::m_array_size *
                            TurbForcingParm::m_array_size;
  constexpr int tmp_buffer_size = TurbForcingParm::m_num_fdarray * num_elmts;
  amrex::Real tmp_buffer[tmp_buffer_size];
  // Separate out forcing data into individual Array4's
  int i_arr = 0;
  constexpr int fd_ncomp = 1;
  constexpr amrex::Dim3 fd_begin{0, 0, 0};
  constexpr amrex::Dim3 fd_end{
    TurbForcingParm::m_array_size, TurbForcingParm::m_array_size,
    TurbForcingParm::m_array_size};

  amrex::Array4<amrex::Real> FTX(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> TAT(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPX(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPY(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPZ(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FAX(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FAY(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FAZ(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPXX(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPXY(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPXZ(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPYX(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPYY(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPYZ(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPZX(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPZY(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPZZ(
    &tmp_buffer[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);

  // initiate the magic
  mersenne_twister::InitRandom((unsigned long)111397);

  int mode_count = 0;

  const auto xstep = static_cast<int>(std::lround(Lx / m_tfp.m_Lmin));
  const auto ystep = static_cast<int>(std::lround(Ly / m_tfp.m_Lmin));
  const auto zstep = static_cast<int>(std::lround(Lz / m_tfp.m_Lmin));

  if (m_tfp.m_verbose > 0) {
    amrex::Print() << "Mode step = " << xstep << " " << ystep << " " << zstep
                   << std::endl;
  }

  for (int kz = m_tfp.m_mode_start * zstep; kz <= m_tfp.m_nzmodes;
       kz += zstep) {
    const auto kzd = static_cast<amrex::Real>(kz);
    for (int ky = m_tfp.m_mode_start * ystep; ky <= m_tfp.m_nymodes;
         ky += ystep) {
      const auto kyd = static_cast<amrex::Real>(ky);
      for (int kx = m_tfp.m_mode_start * xstep; kx <= m_tfp.m_nxmodes;
           kx += xstep) {
        const auto kxd = static_cast<amrex::Real>(kx);
        const amrex::Real kappa = std::sqrt(
          (kxd * kxd) / (Lx * Lx) + (kyd * kyd) / (Ly * Ly) +
          (kzd * kzd) / (Lz * Lz));

        if (kappa <= m_tfp.m_kappaMax) {
          FTX(kx, ky, kz) =
            (m_tfp.m_freqMin + m_tfp.m_freqDiff * mersenne_twister::Random()) *
            TwoPi;
          mersenne_twister::Random(); // dummy FTY (don't remove)
          mersenne_twister::Random(); // dummy FTZ (don't remove)
          // Translation angles, theta=0..2Pi and phi=0..Pi
          TAT(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          mersenne_twister::Random(); // dummy TAP (don't remove)
          // Phases
          FPX(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPY(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPZ(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPXX(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPYX(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPZX(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPXY(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPYY(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPZY(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPXZ(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPYZ(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPZZ(kx, ky, kz) = mersenne_twister::Random() * TwoPi;

          // Amplitudes (alpha)
          const amrex::Real thetaTmp = mersenne_twister::Random() * TwoPi;
          const amrex::Real cosThetaTmp = cos(thetaTmp);
          const amrex::Real sinThetaTmp = sin(thetaTmp);

          const amrex::Real phiTmp = mersenne_twister::Random() * Pi;
          const amrex::Real cosPhiTmp = cos(phiTmp);
          const amrex::Real sinPhiTmp = sin(phiTmp);

          const amrex::Real px = cosThetaTmp * sinPhiTmp;
          const amrex::Real py = sinThetaTmp * sinPhiTmp;
          const amrex::Real pz = cosPhiTmp;

          const amrex::Real mp2 = px * px + py * py + pz * pz;
          if (kappa < 0.000001) {
            if (m_tfp.m_verbose > 0) {
              amrex::Print()
                << "   ZERO AMPLITUDE MODE " << kx << ky << kz << std::endl;
            }
            FAX(kx, ky, kz) = 0.;
            FAY(kx, ky, kz) = 0.;
            FAZ(kx, ky, kz) = 0.;
          } else {
            // Count modes that contribute
            mode_count++;
            // Set amplitudes
            amrex::Real Ekh;
            if (m_tfp.m_spectrum_type == 1) {
              Ekh = 1.0 / kappa;
            } else if (m_tfp.m_spectrum_type == 2) {
              Ekh = 1.0 / (kappa * kappa);
            } else {
              Ekh = 1.0;
            }
            // div_free_forcing (assumed) needs another
            Ekh /= kappa;

            if (m_tfp.m_moderate_zero_modes == 1) {
              if (kx == 0) {
                Ekh /= 2.;
              }
              if (ky == 0) {
                Ekh /= 2.;
              }
              if (kz == 0) {
                Ekh /= 2.;
              }
            }
            if (m_tfp.m_force_scale > 0.0) {
              FAX(kx, ky, kz) = m_tfp.m_force_scale * px * Ekh / mp2;
              FAY(kx, ky, kz) = m_tfp.m_force_scale * py * Ekh / mp2;
              FAZ(kx, ky, kz) = m_tfp.m_force_scale * pz * Ekh / mp2;
            } else {
              FAX(kx, ky, kz) = px * Ekh / mp2;
              FAY(kx, ky, kz) = py * Ekh / mp2;
              FAZ(kx, ky, kz) = pz * Ekh / mp2;
            }

            if (m_tfp.m_verbose > 1) {
              amrex::Print() << "   Mode";
              amrex::Print() << "   kappa = " << kx << " " << ky << " " << kz
                             << " " << kappa << " "
                             << sqrt(
                                  FAX(kx, ky, kz) * FAX(kx, ky, kz) +
                                  FAY(kx, ky, kz) * FAY(kx, ky, kz) +
                                  FAZ(kx, ky, kz) * FAZ(kx, ky, kz))
                             << std::endl;
              amrex::Print() << "   Amplitudes - A" << std::endl;
              amrex::Print() << FAX(kx, ky, kz) << " " << FAY(kx, ky, kz) << " "
                             << FAZ(kx, ky, kz) << std::endl;
              amrex::Print() << "   Frequencies" << std::endl;
              amrex::Print() << FTX(kx, ky, kz) << std::endl;
              amrex::Print() << "   TAT" << std::endl;
              amrex::Print() << TAT(kx, ky, kz) << std::endl;
              amrex::Print() << "   Amplitudes - AA" << std::endl;
              amrex::Print() << FPXX(kx, ky, kz) << " " << FPYX(kx, ky, kz)
                             << " " << FPZX(kx, ky, kz) << std::endl;
              amrex::Print() << FPXY(kx, ky, kz) << " " << FPYY(kx, ky, kz)
                             << " " << FPZY(kx, ky, kz) << std::endl;
              amrex::Print() << FPXZ(kx, ky, kz) << " " << FPYZ(kx, ky, kz)
                             << " " << FPZZ(kx, ky, kz) << std::endl;
            }
          }
        }
      }
    }
  }

  // Now let's break symmetry, have to assume high aspect ratio in z for now
  int reduced_mode_count = 0;

  for (int kz = 1; kz < zstep; ++kz) {
    const auto kzd = static_cast<amrex::Real>(kz);
    for (int ky = m_tfp.m_mode_start; ky <= m_tfp.m_nymodes; ky += ystep) {
      const auto kyd = static_cast<amrex::Real>(ky);
      for (int kx = m_tfp.m_mode_start; kx <= m_tfp.m_nxmodes; kx += xstep) {
        const auto kxd = static_cast<amrex::Real>(kx);
        const auto kappa = std::sqrt(
          (kxd * kxd) / (Lx * Lx) + (kyd * kyd) / (Ly * Ly) +
          (kzd * kzd) / (Lz * Lz));

        if (kappa <= m_tfp.m_kappaMax) {
          FTX(kx, ky, kz) =
            (m_tfp.m_freqMin + m_tfp.m_freqDiff * mersenne_twister::Random()) *
            TwoPi;
          mersenne_twister::Random(); // dummy FTY (don't remove)
          mersenne_twister::Random(); // dummy FTZ (don't remove)
          // Translation angles, theta=0..2Pi and phi=0..Pi
          TAT(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          mersenne_twister::Random(); // dummy TAP (don't remove)
          // Phases
          FPX(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPY(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPZ(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPXX(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPYX(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPZX(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPXY(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPYY(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPZY(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPXZ(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPYZ(kx, ky, kz) = mersenne_twister::Random() * TwoPi;
          FPZZ(kx, ky, kz) = mersenne_twister::Random() * TwoPi;

          // Amplitudes (alpha)
          const amrex::Real thetaTmp = mersenne_twister::Random() * TwoPi;
          const amrex::Real cosThetaTmp = cos(thetaTmp);
          const amrex::Real sinThetaTmp = sin(thetaTmp);

          const amrex::Real phiTmp = mersenne_twister::Random() * Pi;
          const amrex::Real cosPhiTmp = cos(phiTmp);
          const amrex::Real sinPhiTmp = sin(phiTmp);

          const amrex::Real px = cosThetaTmp * sinPhiTmp;
          const amrex::Real py = sinThetaTmp * sinPhiTmp;
          const amrex::Real pz = cosPhiTmp;

          const amrex::Real mp2 = px * px + py * py + pz * pz;
          if (kappa < 0.000001) {
            if (m_tfp.m_verbose > 0) {
              amrex::Print()
                << "   ZERO AMPLITUDE MODE " << kx << ky << kz << std::endl;
            }
            FAX(kx, ky, kz) = 0.;
            FAY(kx, ky, kz) = 0.;
            FAZ(kx, ky, kz) = 0.;
          } else {
            // Count modes that contribute
            reduced_mode_count++;
            // Set amplitudes
            amrex::Real Ekh;
            if (m_tfp.m_spectrum_type == 1) {
              Ekh = 1. / kappa;
            } else if (m_tfp.m_spectrum_type == 2) {
              Ekh = 1. / (kappa * kappa);
            } else {
              Ekh = 1.;
            }
            // div_free_forcing (assumed) needs another
            Ekh /= kappa;

            if (m_tfp.m_moderate_zero_modes == 1) {
              if (kx == 0) {
                Ekh /= 2.;
              }
              if (ky == 0) {
                Ekh /= 2.;
              }
              if (kz == 0) {
                Ekh /= 2.;
              }
            }
            if (m_tfp.m_force_scale > 0.) {
              FAX(kx, ky, kz) =
                m_tfp.m_forcing_epsilon * m_tfp.m_force_scale * px * Ekh / mp2;
              FAY(kx, ky, kz) =
                m_tfp.m_forcing_epsilon * m_tfp.m_force_scale * py * Ekh / mp2;
              FAZ(kx, ky, kz) =
                m_tfp.m_forcing_epsilon * m_tfp.m_force_scale * pz * Ekh / mp2;
            } else {
              FAX(kx, ky, kz) = m_tfp.m_forcing_epsilon * px * Ekh / mp2;
              FAY(kx, ky, kz) = m_tfp.m_forcing_epsilon * py * Ekh / mp2;
              FAZ(kx, ky, kz) = m_tfp.m_forcing_epsilon * pz * Ekh / mp2;
            }

            if (m_tfp.m_verbose > 1) {
              amrex::Print() << "   Mode";
              amrex::Print() << "   kappa = " << kx << " " << ky << " " << kz
                             << " " << kappa << " "
                             << std::sqrt(
                                  FAX(kx, ky, kz) * FAX(kx, ky, kz) +
                                  FAY(kx, ky, kz) * FAY(kx, ky, kz) +
                                  FAZ(kx, ky, kz) * FAZ(kx, ky, kz))
                             << std::endl;
              amrex::Print() << "   Amplitudes - A" << std::endl;
              amrex::Print() << FAX(kx, ky, kz) << " " << FAY(kx, ky, kz) << " "
                             << FAZ(kx, ky, kz) << std::endl;
              amrex::Print() << "   Frequencies" << std::endl;
              amrex::Print() << FTX(kx, ky, kz) << std::endl;
              amrex::Print() << "   TAT" << std::endl;
              amrex::Print() << TAT(kx, ky, kz) << std::endl;
              amrex::Print() << "   Amplitudes - AA" << std::endl;
              amrex::Print() << FPXX(kx, ky, kz) << " " << FPYX(kx, ky, kz)
                             << " " << FPZX(kx, ky, kz) << std::endl;
              amrex::Print() << FPXY(kx, ky, kz) << " " << FPYY(kx, ky, kz)
                             << " " << FPZY(kx, ky, kz) << std::endl;
              amrex::Print() << FPXZ(kx, ky, kz) << " " << FPYZ(kx, ky, kz)
                             << " " << FPZZ(kx, ky, kz) << std::endl;
            }
          }
        }
      }
    }
  }

  if (m_tfp.m_verbose > 0) {
    amrex::Print() << "mode_count = " << mode_count << std::endl;
    amrex::Print() << "reduced_mode_count = " << reduced_mode_count
                   << std::endl;
    if (m_tfp.m_spectrum_type == 1) {
      amrex::Print() << "Spectrum type 1" << std::endl;
    } else if (m_tfp.m_spectrum_type == 2) {
      amrex::Print() << "Spectrum type 2" << std::endl;
    } else {
      amrex::Print() << "Spectrum type OTHER" << std::endl;
    }
  }

  // Now allocate forcedata and copy in tmp array.
#ifdef AMREX_USE_GPU
  if (amrex::Gpu::inLaunchRegion()) {
    m_tfp.m_forcedata = static_cast<amrex::Real*>(
      amrex::The_Arena()->alloc(tmp_buffer_size * sizeof(amrex::Real)));
    amrex::Gpu::htod_memcpy_async(
      m_tfp.m_forcedata, tmp_buffer, tmp_buffer_size * sizeof(amrex::Real));
  } else
#endif
  {
    m_tfp.m_forcedata = static_cast<amrex::Real*>(
      amrex::The_Pinned_Arena()->alloc(tmp_buffer_size * sizeof(amrex::Real)));
    std::memcpy(
      m_tfp.m_forcedata, tmp_buffer, tmp_buffer_size * sizeof(amrex::Real));
  }

  if (m_tfp.m_verbose > 0) {
    amrex::Print() << "========================================================"
                   << std::endl;
  }
  m_turbforcing_initialized = true;
}

void
TurbForcing::addTurbVelForces(
  amrex::GeometryData const& geomdata,
  const amrex::Box& bx,
  const amrex::Real& time,
  amrex::Array4<amrex::Real> const& force,
  amrex::Array4<const amrex::Real> const& rho,
  const int a_incompressible,
  const amrex::Real a_rho_incompressible)
{
  AMREX_ALWAYS_ASSERT(m_turbforcing_initialized);

  if (a_incompressible != 0 && a_rho_incompressible <= 0.0) {
    amrex::Abort("rho_incompressible must be greater than 0 when "
                 "incompressible\n");
  }

  constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;
  constexpr amrex::Real TwoPi = 2.0 * Pi;

  const amrex::Real* problo = geomdata.ProbLo();
  const amrex::Real* probhi = geomdata.ProbHi();

  amrex::Real Lx = probhi[0] - problo[0];
  amrex::Real Ly = probhi[1] - problo[1];
  amrex::Real Lz = probhi[2] - problo[2];

  if (m_tfp.m_hack_lz > 0) {
    if (m_tfp.m_hack_lz == 1) {
      Lz = Lz / 2.0;
    } else {
      Lz = Lz / m_tfp.m_hack_lz;
    }
  }

  const int* f_lo = bx.loVect();
  const int* f_hi = bx.hiVect();

  const auto xstep = static_cast<int>(std::lround(Lx / m_tfp.m_Lmin));
  const auto ystep = static_cast<int>(std::lround(Ly / m_tfp.m_Lmin));
  const auto zstep = static_cast<int>(std::lround(Lz / m_tfp.m_Lmin));

  const amrex::Real kappaMax = m_tfp.m_nmodes / m_tfp.m_Lmin + 1.0e-8;

  const amrex::Real forcetime = time + m_tfp.m_time_offset;

  auto const& dx = geomdata.CellSize();

  // Separate out forcing data into individual Array4's
  int i_arr = 0;
  constexpr int fd_ncomp = 1;
  constexpr int num_elmts = TurbForcingParm::m_array_size *
                            TurbForcingParm::m_array_size *
                            TurbForcingParm::m_array_size;
  constexpr amrex::Dim3 fd_begin{0, 0, 0};
  constexpr amrex::Dim3 fd_end{
    TurbForcingParm::m_array_size, TurbForcingParm::m_array_size,
    TurbForcingParm::m_array_size};

  amrex::Array4<amrex::Real> FTX(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> TAT(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPX(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPY(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPZ(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FAX(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FAY(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FAZ(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPXX(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPXY(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPXZ(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPYX(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPYY(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPYZ(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPZX(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPZY(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);
  amrex::Array4<amrex::Real> FPZZ(
    &m_tfp.m_forcedata[(i_arr++) * num_elmts], fd_begin, fd_end, fd_ncomp);

  const amrex::RealBox loc =
    amrex::RealBox(bx, geomdata.CellSize(), geomdata.ProbLo());
  const auto& loc_lo = loc.lo();
  const amrex::GpuArray<const amrex::Real, AMREX_SPACEDIM> xlo = {
    AMREX_D_DECL(loc_lo[0], loc_lo[1], loc_lo[2])};

  //
  // Construct force at fewer points and then interpolate.
  // This is much faster on CPU.
  //

  const amrex::Real hx = dx[0];
  const amrex::Real hy = dx[1];
  const amrex::Real hz = dx[2];

  const int ilo = f_lo[0];
  const int jlo = f_lo[1];
  const int klo = f_lo[2];

  const int ihi = f_hi[0];
  const int jhi = f_hi[1];
  const int khi = f_hi[2];

  // coarse cell size
  const amrex::Real ff_hx = hx * m_tfp.m_ff_factor;
  const amrex::Real ff_hy = hy * m_tfp.m_ff_factor;
  const amrex::Real ff_hz = hz * m_tfp.m_ff_factor;

  // coarse bounds (without accounting for ghost cells)
  // FIXME -- think about how bx (the box we want to fill) may not be the same
  // as the force box!
  int ff_ilo = ilo / m_tfp.m_ff_factor;
  int ff_jlo = jlo / m_tfp.m_ff_factor;
  int ff_klo = klo / m_tfp.m_ff_factor;

  int ff_ihi = (ihi + 1) / m_tfp.m_ff_factor;
  int ff_jhi = (jhi + 1) / m_tfp.m_ff_factor;
  int ff_khi = (khi + 1) / m_tfp.m_ff_factor;

  // adjust for ghost cells
  if (ilo < (ff_ilo * m_tfp.m_ff_factor)) {
    ff_ilo = ff_ilo - 1;
  }
  if (jlo < (ff_jlo * m_tfp.m_ff_factor)) {
    ff_jlo = ff_jlo - 1;
  }
  if (klo < (ff_klo * m_tfp.m_ff_factor)) {
    ff_klo = ff_klo - 1;
  }
  if (ihi == (ff_ihi * m_tfp.m_ff_factor)) {
    ff_ihi = ff_ihi + 1;
  }
  if (jhi == (ff_jhi * m_tfp.m_ff_factor)) {
    ff_jhi = ff_jhi + 1;
  }
  if (khi == (ff_khi * m_tfp.m_ff_factor)) {
    ff_khi = ff_khi + 1;
  }

  // allocate coarse force array
  amrex::Box ffbx(
    amrex::IntVect(AMREX_D_DECL(ff_ilo, ff_jlo, ff_klo)),
    amrex::IntVect(AMREX_D_DECL(ff_ihi, ff_jhi, ff_khi)));
  amrex::FArrayBox ff_force(ffbx, AMREX_SPACEDIM, amrex::The_Async_Arena());
  const auto& ffarr = ff_force.array();

  // Construct node-based coarse forcing
  amrex::ParallelFor(
    ffbx,
    [=, mode_start = m_tfp.m_mode_start,
     nmodes = m_tfp.m_nmodes] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      amrex::Real z = xlo[2] + ff_hz * (k - ff_klo);
      amrex::Real y = xlo[1] + ff_hy * (j - ff_jlo);
      amrex::Real x = xlo[0] + ff_hx * (i - ff_ilo);

      for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        ffarr(i, j, k, n) = 0.0;
      }

      // forcedata (and Array4) has column-major layout
      for (int kz = mode_start * zstep; kz <= nmodes * zstep; kz += zstep) {
        for (int ky = mode_start * ystep; ky <= nmodes * ystep; ky += ystep) {
          for (int kx = mode_start * xstep; kx <= nmodes * xstep; kx += xstep) {
            const amrex::Real kappa = std::sqrt(
              (kx * kx) / (Lx * Lx) + (ky * ky) / (Ly * Ly) +
              (kz * kz) / (Lz * Lz));

            if (kappa <= kappaMax) {
              const amrex::Real xT =
                cos(FTX(kx, ky, kz) * forcetime + TAT(kx, ky, kz));

              ffarr(i, j, k, 0) +=
                xT * (FAZ(kx, ky, kz) * TwoPi * (ky / Ly) *
                        sin(TwoPi * kx * x / Lx + FPZX(kx, ky, kz)) *
                        cos(TwoPi * ky * y / Ly + FPZY(kx, ky, kz)) *
                        sin(TwoPi * kz * z / Lz + FPZZ(kx, ky, kz)) -
                      FAY(kx, ky, kz) * TwoPi * (kz / Lz) *
                        sin(TwoPi * kx * x / Lx + FPYX(kx, ky, kz)) *
                        sin(TwoPi * ky * y / Ly + FPYY(kx, ky, kz)) *
                        cos(TwoPi * kz * z / Lz + FPYZ(kx, ky, kz)));

              ffarr(i, j, k, 1) +=
                xT * (FAX(kx, ky, kz) * TwoPi * (kz / Lz) *
                        sin(TwoPi * kx * x / Lx + FPXX(kx, ky, kz)) *
                        sin(TwoPi * ky * y / Ly + FPXY(kx, ky, kz)) *
                        cos(TwoPi * kz * z / Lz + FPXZ(kx, ky, kz)) -
                      FAZ(kx, ky, kz) * TwoPi * (kx / Lx) *
                        cos(TwoPi * kx * x / Lx + FPZX(kx, ky, kz)) *
                        sin(TwoPi * ky * y / Ly + FPZY(kx, ky, kz)) *
                        sin(TwoPi * kz * z / Lz + FPZZ(kx, ky, kz)));

              ffarr(i, j, k, 2) +=
                xT * (FAY(kx, ky, kz) * TwoPi * (kx / Lx) *
                        cos(TwoPi * kx * x / Lx + FPYX(kx, ky, kz)) *
                        sin(TwoPi * ky * y / Ly + FPYY(kx, ky, kz)) *
                        sin(TwoPi * kz * z / Lz + FPYZ(kx, ky, kz)) -
                      FAX(kx, ky, kz) * TwoPi * (ky / Ly) *
                        sin(TwoPi * kx * x / Lx + FPXX(kx, ky, kz)) *
                        cos(TwoPi * ky * y / Ly + FPXY(kx, ky, kz)) *
                        sin(TwoPi * kz * z / Lz + FPXZ(kx, ky, kz)));
            }
          }
        }
      }

      //
      // For high aspect ratio domain, add more modes to break symmetry at a low
      // level. We assume Lz is longer, Lx = Ly.
      //
      for (int kz = 1; kz <= zstep - 1; ++kz) {
        for (int ky = mode_start; ky <= nmodes * ystep; ++ky) {
          for (int kx = mode_start; kx <= nmodes * xstep; ++kx) {
            const amrex::Real kappa = std::sqrt(
              (kx * kx) / (Lx * Lx) + (ky * ky) / (Ly * Ly) +
              (kz * kz) / (Lz * Lz));

            if (kappa <= kappaMax) {
              const amrex::Real xT =
                cos(FTX(kx, ky, kz) * forcetime + TAT(kx, ky, kz));

              ffarr(i, j, k, 0) +=
                xT * (FAZ(kx, ky, kz) * TwoPi * (ky / Ly) *
                        sin(TwoPi * kx * x / Lx + FPZX(kx, ky, kz)) *
                        cos(TwoPi * ky * y / Ly + FPZY(kx, ky, kz)) *
                        sin(TwoPi * kz * z / Lz + FPZZ(kx, ky, kz)) -
                      FAY(kx, ky, kz) * TwoPi * (kz / Lz) *
                        sin(TwoPi * kx * x / Lx + FPYX(kx, ky, kz)) *
                        sin(TwoPi * ky * y / Ly + FPYY(kx, ky, kz)) *
                        cos(TwoPi * kz * z / Lz + FPYZ(kx, ky, kz)));

              ffarr(i, j, k, 1) +=
                xT * (FAX(kx, ky, kz) * TwoPi * (kz / Lz) *
                        sin(TwoPi * kx * x / Lx + FPXX(kx, ky, kz)) *
                        sin(TwoPi * ky * y / Ly + FPXY(kx, ky, kz)) *
                        cos(TwoPi * kz * z / Lz + FPXZ(kx, ky, kz)) -
                      FAZ(kx, ky, kz) * TwoPi * (kx / Lx) *
                        cos(TwoPi * kx * x / Lx + FPZX(kx, ky, kz)) *
                        sin(TwoPi * ky * y / Ly + FPZY(kx, ky, kz)) *
                        sin(TwoPi * kz * z / Lz + FPZZ(kx, ky, kz)));

              ffarr(i, j, k, 2) +=
                xT * (FAY(kx, ky, kz) * TwoPi * (kx / Lx) *
                        cos(TwoPi * kx * x / Lx + FPYX(kx, ky, kz)) *
                        sin(TwoPi * ky * y / Ly + FPYY(kx, ky, kz)) *
                        sin(TwoPi * kz * z / Lz + FPYZ(kx, ky, kz)) -
                      FAX(kx, ky, kz) * TwoPi * (ky / Ly) *
                        sin(TwoPi * kx * x / Lx + FPXX(kx, ky, kz)) *
                        cos(TwoPi * ky * y / Ly + FPXY(kx, ky, kz)) *
                        sin(TwoPi * kz * z / Lz + FPXZ(kx, ky, kz)));
            }
          }
        }
      }
    });

  // Need all of ffarr filled for next lambda
  amrex::Gpu::synchronize();

  // Now interpolate onto fine grid
  if (a_incompressible == 0) {
    amrex::ParallelFor(
      bx, AMREX_SPACEDIM,
      [=, ff_factor = m_tfp.m_ff_factor] AMREX_GPU_DEVICE(
        int i, int j, int k, int n) noexcept {
        const int ff_k = k / ff_factor;
        const int ff_j = j / ff_factor;
        const int ff_i = i / ff_factor;

        const amrex::Real zd =
          (hz * (k - klo + 0.5) - ff_hz * (ff_k - ff_klo)) / ff_hz;
        const amrex::Real yd =
          (hy * (j - jlo + 0.5) - ff_hy * (ff_j - ff_jlo)) / ff_hy;
        const amrex::Real xd =
          (hx * (i - ilo + 0.5) - ff_hx * (ff_i - ff_ilo)) / ff_hx;

        const amrex::Real ff00 = ffarr(ff_i, ff_j, ff_k, n) * (1. - xd) +
                                 ffarr(ff_i + 1, ff_j, ff_k, n) * xd;
        const amrex::Real ff01 = ffarr(ff_i, ff_j, ff_k + 1, n) * (1. - xd) +
                                 ffarr(ff_i + 1, ff_j, ff_k + 1, n) * xd;
        const amrex::Real ff10 = ffarr(ff_i, ff_j + 1, ff_k, n) * (1. - xd) +
                                 ffarr(ff_i + 1, ff_j + 1, ff_k, n) * xd;
        const amrex::Real ff11 =
          ffarr(ff_i, ff_j + 1, ff_k + 1, n) * (1. - xd) +
          ffarr(ff_i + 1, ff_j + 1, ff_k + 1, n) * xd;
        const amrex::Real ff = (ff00 * (1. - yd) + ff10 * yd) * (1. - zd) +
                               (ff01 * (1. - yd) + ff11 * yd) * zd;

        force(i, j, k, n) += rho(i, j, k) * ff;
      });
  } else {
    amrex::ParallelFor(
      bx, AMREX_SPACEDIM,
      [=, ff_factor = m_tfp.m_ff_factor,
       a_rho =
         a_rho_incompressible] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
        const int ff_k = k / ff_factor;
        const int ff_j = j / ff_factor;
        const int ff_i = i / ff_factor;

        const amrex::Real zd =
          (hz * (k - klo + 0.5) - ff_hz * (ff_k - ff_klo)) / ff_hz;
        const amrex::Real yd =
          (hy * (j - jlo + 0.5) - ff_hy * (ff_j - ff_jlo)) / ff_hy;
        const amrex::Real xd =
          (hx * (i - ilo + 0.5) - ff_hx * (ff_i - ff_ilo)) / ff_hx;

        const amrex::Real ff00 = ffarr(ff_i, ff_j, ff_k, n) * (1. - xd) +
                                 ffarr(ff_i + 1, ff_j, ff_k, n) * xd;
        const amrex::Real ff01 = ffarr(ff_i, ff_j, ff_k + 1, n) * (1. - xd) +
                                 ffarr(ff_i + 1, ff_j, ff_k + 1, n) * xd;
        const amrex::Real ff10 = ffarr(ff_i, ff_j + 1, ff_k, n) * (1. - xd) +
                                 ffarr(ff_i + 1, ff_j + 1, ff_k, n) * xd;
        const amrex::Real ff11 =
          ffarr(ff_i, ff_j + 1, ff_k + 1, n) * (1. - xd) +
          ffarr(ff_i + 1, ff_j + 1, ff_k + 1, n) * xd;

        const amrex::Real ff = (ff00 * (1. - yd) + ff10 * yd) * (1. - zd) +
                               (ff01 * (1. - yd) + ff11 * yd) * zd;

        force(i, j, k, n) += a_rho * ff;
      });
  }
}

} // namespace pele::physics::turbforcing
