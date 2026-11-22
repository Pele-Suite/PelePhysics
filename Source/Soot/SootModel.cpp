
// AMReX include statements
#include <AMReX_Reduce.H>

// PelePhysics include statements
#include "PelePhysics.H"

// Soot include statements
#include "SootModel.H"
#include "SootModel_derive.H"

using namespace amrex;

// Default constructor
SootModel::SootModel()
  : m_sootVarName(NUM_SOOT_MOMENTS + 1, ""), m_gasSpecNames(NUM_SOOT_GS, "")
{
  m_sootVarName[NUM_SOOT_MOMENTS] = "soot_N0";
  m_sootVarName[0] = "soot_N";
  m_sootVarName[1] = "soot_fv";
  m_sootVarName[2] = "soot_S";
#if NUM_SOOT_MOMENTS == 6
  m_sootVarName[3] = "soot_V_V";
  m_sootVarName[4] = "soot_V_S";
  m_sootVarName[5] = "soot_S_S";
#endif
}

// Fill in moment source and reaction member data
void
SootModel::define()
{
  const int ngs = NUM_SOOT_GS;
  // This should be called after readSootParams()
  AMREX_ASSERT(m_readSootParams);
  m_sootData = new SootData{};
  // -1 marks a soot gas species not (yet) found in the mechanism
  m_sootData->refIndx.fill(-1);
  m_sootReact = new SootReaction{};
  d_sootData =
    static_cast<SootData*>(amrex::The_Arena()->alloc(sizeof(SootData)));
  d_sootReact =
    static_cast<SootReaction*>(amrex::The_Arena()->alloc(sizeof(SootReaction)));
  // Double check indices are set
  m_setIndx = m_sootIndx.checkIndices();
  if (!m_setIndx) {
    Abort("SootModel::define(): Must set indices before defining");
  }
  // Relevant species names for the surface chemistry model
  // Must correspond to SootGasSpecIndx enum in Constants_Soot.H
  m_gasSpecNames = {"H2", "H", "OH", "H2O", "CO", "C2H2", "O2", m_PAHname};
  // Number of accounted for PAH particles (hard-coded)
  const int numPAH = 3;
  // Names of PAH species. Can only handle naphthalene (C10H8), phenathrene
  // (C14H10), and pyrene (C16H10) and can only handle 1 PAH inception species
  const std::string PAH_names[] = {"A2", "A3", "A4"};
  // Corresponding sticking coefficients
  const Real PAH_gammas[] = {0.002, 0.015, 0.025};
  // Average number of C atoms per PAH species
  const Real PAH_numC[] = {10., 14., 16.};

  // Determine which PAH inception species is being used
  bool corrPAH = false;
  Real dimerVol = 0.;
  for (int cpah = 0; cpah < numPAH; ++cpah) {
    if (PAH_names[cpah] == m_inceptPAH) {
      m_gammaStick = PAH_gammas[cpah];
      dimerVol = 2. * PAH_numC[cpah];
      corrPAH = true;
    }
  }

  if (!corrPAH) {
    Abort(
      m_inceptPAH + " not recognized as PAH inception species. Must be " +
      PAH_names[0] + ", " + PAH_names[1] + ", or " + PAH_names[2]);
  }
  Vector<std::string> spec_names(NUM_SPECIES);
  pele::physics::eos::speciesNames<pele::physics::EosType>(spec_names);
  // Loop over all species
  for (int i = 0; i < NUM_SPECIES; ++i) {
    // Check if species is the PAH inceptor
    if (spec_names[i] == m_PAHname) {
      m_PAHindx = i;
    }
    // Check if species matches others for surface reactions
    for (int sootSpec = 0; sootSpec < ngs; ++sootSpec) {
      if (spec_names[i] == m_gasSpecNames[sootSpec]) {
        m_sootData->refIndx[sootSpec] = i;
      }
    }
  }
  // Return error if no PAH inception species is specified in mechanism
  if (m_PAHindx == -1) {
    Abort("PAH inception species was not found in PelePhysics mechanism");
  }
  // Return error if not all soot species are present
  for (int sootSpec = 0; sootSpec < ngs; ++sootSpec) {
    if (m_sootData->refIndx[sootSpec] == -1) {
      Abort(
        "Species " + m_gasSpecNames[sootSpec] +
        " must be present in PelePhysics mechanism");
    }
  }

  // Assign moment factor member data
  defineMemberData(dimerVol);
  // From SootModel_react.cpp
  // Initialize reaction and species member data
  initializeReactData();

  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, m_sootData, m_sootData + 1, d_sootData);
  amrex::Gpu::copy(
    amrex::Gpu::hostToDevice, m_sootReact, m_sootReact + 1, d_sootReact);

  if (m_sootVerbosity >= 1 && ParallelDescriptor::IOProcessor()) {
    Print() << "SootModel::define(): Soot model successfully defined"
            << std::endl;
  }
}

// Read soot related inputs
void
SootModel::readSootParams()
{
  ParmParse pp("soot");
  pp.get("incept_pah", m_inceptPAH);
  m_PAHname = m_inceptPAH;
  // Necessary if species names differ from A2, A3, or A4
  pp.query("pah_name", m_PAHname);
  pp.query("v", m_sootVerbosity);
  if (m_sootVerbosity >= 1) {
    Print() << "SootModel::readSootParams(): Reading input parameters"
            << std::endl;
  }
  // Set the maximum allowable change for variables during soot source terms
  pp.query("X_cutoff", m_Xcutoff);
  pp.query("temp_cutoff", m_Tcutoff);
  pp.query("max_subcycles", m_maxSubcycles);
#ifdef PELELM_USE_SOOT
  m_numSubcycles = 5;
#endif
  pp.query("num_subcycles", m_numSubcycles);
  if (m_numSubcycles < 1 || m_maxSubcycles < 1) {
    Abort("soot.num_subcycles and soot.max_subcycles must be at least 1");
  }
  if (m_numSubcycles > m_maxSubcycles) {
    Abort("soot.num_subcycles must not exceed soot.max_subcycles");
  }
  // Determines if mass is conserved by adding lost mass to H2
  pp.query("conserve_mass", m_conserveMass);
  m_readSootParams = true;
}

// Define all member data
void
SootModel::defineMemberData(const Real dimerVol)
{
  if (m_sootVerbosity >= 1) {
    Print() << "SootModel::defineMemberData(): Defining member data"
            << std::endl;
  }
  SootConst sc;
  Real nuclVol = 2. * dimerVol;
  Real nuclSurf = std::pow(nuclVol, 2. / 3.);
  m_sootData->nuclVol = nuclVol;
  m_sootData->nuclSurf = nuclSurf;
  setSootDeriveNuclSize(nuclVol, nuclSurf);
  // Compute V_nucl and V_dimer to fractional powers
  for (int i = 0; i < 9; ++i) {
    Real exponent = 2. * (Real)i - 3.;
    m_sootData->dime6[i] = std::pow(dimerVol, exponent / 6.);
  }
  for (int i = 0; i < 11; ++i) {
    Real exponent = (Real)i - 3.;
    m_sootData->nve3[i] = std::pow(nuclVol, exponent / 3.);
    m_sootData->nve6[i] = std::pow(nuclVol, exponent / 6.);
  }

  for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
    // Used to convert moments to mol of C
    m_sootData->unitConv[i] = std::pow(sc.V0, sc.MomOrderV[i]) *
                              std::pow(sc.S0, sc.MomOrderS[i]) *
                              pele::physics::Constants::Avna;
    // First convert moments from SI to CGS, only necessary for PeleLM
    const Real expFact = 3. * sc.MomOrderV[i] + 2. * sc.MomOrderS[i];
    m_sootData->unitConv[i] *= std::pow(sc.len_conv, 3. - expFact);
    // Used for computing nucleation source term
    m_sootData->momFact[i] =
      std::pow(nuclVol, sc.MomOrderV[i]) * std::pow(nuclSurf, sc.MomOrderS[i]);
    m_sootData->smallMoms[i] = m_sootData->momFact[i] * sc.smallWeight;
  }
  // and to convert the weight of the delta function
  m_sootData->unitConv[NUM_SOOT_MOMENTS] =
    pele::physics::Constants::Avna * std::pow(sc.len_conv, 3);
  m_sootData->smallMoms[NUM_SOOT_MOMENTS] = m_sootData->smallMoms[0];

  // Coagulation, oxidation, and fragmentation factors
  m_sootData->lambdaCF =
    1. / (std::pow(
           6. * sc.SootMolarMass /
             (M_PI * sc.SootDensity * pele::physics::Constants::Avna),
           1. / 3.));
  for (int i = 0; i < NUM_SOOT_MOMENTS; ++i) {
    const Real expFact = sc.MomOrderV[i] + 2. / 3. * sc.MomOrderS[i];
    const Real factor = (std::pow(2., expFact - 1.) - 1.);
    m_sootData->ssfmCF[i] =
      std::pow(2., 2.5) * factor * std::pow(nuclVol, expFact + 1. / 6.);
    m_sootData->sscnCF[i] = factor * std::pow(nuclVol, expFact);
    m_sootData->smallOF[i] = std::pow(nuclVol, expFact - 1. / 3.);
    m_sootData->fragFact[i] =
      std::pow(2., 1. - sc.MomOrderV[i] - sc.MomOrderS[i]) - 1.;
  }
  const Real ne32 = m_sootData->nve3[2 + 3];
  const Real ne3m = m_sootData->nve3[-1 + 3];
  m_sootData->ssfmCF[NUM_SOOT_MOMENTS] =
    std::pow(2., 2.5) * std::pow(1. / nuclVol, 0.5) * ne32;
  m_sootData->smallOF[NUM_SOOT_MOMENTS] = ne3m;

  // Beta and dimer factors
  // cm^0.5 mol^-1
  const Real dnfact = 4. * std::sqrt(2.) * sc.colFact16 * sc.colFactPi23 *
                      pele::physics::Constants::Avna;
  m_betaDimerFact = dnfact * std::pow(0.5 * dimerVol, 1. / 6.);
  m_betaNuclFact = 2.2 * dnfact * m_sootData->dime6[2];

  /// Condensation factor
  m_sootData->condFact =
    std::sqrt((1. / nuclVol) + (1. / dimerVol)) *
    std::pow((std::pow(nuclVol, 1. / 3.) + std::pow(dimerVol, 1. / 3.)), 2.);
  m_memberDataDefined = true;
}

// Add soot source term
void
SootModel::computeSootSourceTerm(
  const Box& vbox,
  Array4<const Real> const& Qstate,
  Array4<const Real> const& coeff_mu,
  Array4<Real> const& soot_state,
  const Real /*time*/,
  const Real dt,
  const bool pres_term) const
{
  AMREX_ASSERT(m_memberDataDefined);
  AMREX_ASSERT(m_setIndx);
  BL_PROFILE("SootModel::computeSootSourceTerm");
  if (m_sootVerbosity >= 1 && ParallelDescriptor::IOProcessor()) {
    Print() << "SootModel::computeSootSourceTerm(): Adding soot source term to "
            << vbox << std::endl;
  }
  const int nsub_init = m_numSubcycles;
  const int nsubMAX = m_maxSubcycles;
  // Primitive components
  const int qRhoIndx = m_sootIndx.qRhoIndx;
  const int qTempIndx = m_sootIndx.qTempIndx;
  const int qSpecIndx = m_sootIndx.qSpecIndx;
  const int qSootIndx = m_sootIndx.qSootIndx;
  const int rhoIndx = m_sootIndx.rhoIndx;
  const int engIndx = m_sootIndx.engIndx;
  const int specIndx = m_sootIndx.specIndx;
  const int sootIndx = m_sootIndx.sootIndx;
  const Real betaNF = m_betaNuclFact;

  const bool conserveMass = m_conserveMass;
  const Real Tcutoff = m_Tcutoff;

  // H2 absorbs the error from surface reactions
  const int absorbIndx = SootGasSpecIndx::indxH2;
  const int absorbIndxN = m_sootData->refIndx[absorbIndx];
  const int absorbIndxP = specIndx + absorbIndxN;

  const SootData* sd = d_sootData;
  const SootReaction* sr = d_sootReact;
  SootConst sc;
  amrex::ParallelFor(vbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    auto eos = pele::physics::PhysicsType::eos();
    GpuArray<Real, NUM_SPECIES> mw_fluidF;
    GpuArray<Real, NUM_SOOT_GS> mw_fluid;
    eos.molecular_weight(mw_fluidF.data());
    GpuArray<Real, NUM_SPECIES> Hi;
    GpuArray<Real, NUM_SOOT_GS> omega_src;
    GpuArray<Real, NUM_SPECIES> rho_YF;
    // Molar concentrations (mol/cm^3)
    GpuArray<Real, NUM_SOOT_GS> xi_n;
    // Array of moment values M_xy (cm^(3(x + 2/3y))cm^(-3))
    // M00, M10, M01,..., N0
    GpuArray<Real, NUM_SOOT_MOMENTS + 1> mom0;
    GpuArray<Real, NUM_SOOT_MOMENTS + 1> moments;
    Real* momentsPtr = moments.data();
    /*
      These are the values inside the terms in fracMom
      momFV[NUM_SOOT_MOMENTS] - Weight of the delta function
      momFV[NUM_SOOT_MOMENTS+1] - modeCoef
      where modeCoef signifies the number of modes to be used
      If the moments are effectively zero, modeCoef = 0 and only 1 mode is used
      Otherwise, modeCoef = 1 and both modes are used
      The rest of the momFV values are used in fracMom fact1 = momFV[0],
      fact2 = momFV[1]^volOrd, fact2 = momFV[2]^surfOrd, etc.
    */
    GpuArray<Real, NUM_SOOT_MOMENTS + 2> mom_fv;
    Real* mom_fvPtr = mom_fv.data();
    // Array of source terms for moment equations
    GpuArray<Real, NUM_SOOT_MOMENTS + 1> mom_src;
    Real* mom_srcPtr = mom_src.data();
    Real rho = Qstate(i, j, k, qRhoIndx) * sc.rho_conv;
    const Real T = Qstate(i, j, k, qTempIndx);
    if (T > Tcutoff) {
      // Dynamic viscosity
      const Real mu = coeff_mu(i, j, k) * sc.mu_conv;
      // Compute species enthalpy
      eos.T2Hi(T, Hi.data());
      // Extract mass fractions for gas phases corresponding to GasSpecIndx
      for (int sp = 0; sp < NUM_SPECIES; ++sp) {
        const int peleIndx = qSpecIndx + sp;
        // State provided by PeleLM is the concentration, rhoY
#ifdef PELELM_USE_SOOT
        rho_YF[sp] =
          amrex::max<amrex::Real>(0.0, Qstate(i, j, k, peleIndx) * sc.rho_conv);
#else
        rho_YF[sp] = amrex::max<amrex::Real>(0.0, rho * Qstate(i, j, k, peleIndx));
#endif
      }
      // Compute the average molar mass (g/mol)
      Real molarMass = 0.0;
      for (int sp = 0; sp < NUM_SPECIES; ++sp) {
        molarMass += rho_YF[sp] / mw_fluidF[sp];
      }
      molarMass = rho / molarMass;
      // Extract moment values
      for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
        const int peleIndx = qSootIndx + mom;
        moments[mom] = Qstate(i, j, k, peleIndx);
        mom0[mom] = moments[mom];
        mom_src[mom] = 0.0;
      }
      for (int sp = 0; sp < NUM_SOOT_GS; ++sp) {
        const int spcc = sd->refIndx[sp];
        mw_fluid[sp] = mw_fluidF[spcc];
        xi_n[sp] = rho_YF[spcc] / mw_fluid[sp];
        // Reset the reaction source term
        omega_src[sp] = 0.0;
      }
      // Convert moments from CGS to mol of C
      sd->convertToMol(momentsPtr);
      // Compute constant values used throughout
      // (R*T*Pi/(2*A*rho_soot))^(1/2)
      const Real convT = std::sqrt(sc.colFact * T);
      // Constant for free molecular collisions
      const Real colConst =
        convT * sc.colFactPi23 * sc.colFact16 * pele::physics::Constants::Avna;
      // Clipping the incoming state is not a soot source: start the source
      // from the clipped moments where clipping changed them
      sd->clipMoments(momentsPtr);
      for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
        if (moments[mom] != mom0[mom] / sd->unitConv[mom]) {
          mom0[mom] = moments[mom] * sd->unitConv[mom];
        }
      }
      Real RT = pele::physics::Constants::RU * T;
      // Collision frequency between two dimer in the free
      // molecular regime with van der Waals enhancement
      // Units: cm^3/mol-s
      const Real betaNucl = convT * betaNF;
      int nsub = nsub_init;
      Real mindt = dt / Real(nsubMAX);
      Real sootdt = dt / Real(nsub);
      int isub = 1;
      Real tstart = 0.0;
      // Subcycling
      while (tstart < dt && isub < nsubMAX + 1) {
        sd->computeSrcTerms(
          T, mu, rho, molarMass, convT, betaNucl, colConst, xi_n.data(),
          omega_src.data(), momentsPtr, mom_srcPtr, mom_fvPtr, sr);
        // Estimate subcycling time step size
        Real rate = 1.0;
        for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
          rate = amrex::max<amrex::Real>(
            rate, 1.05 * -sootdt * mom_src[mom] / moments[mom]);
        }
        if (rate > 1.0) {
          sootdt = amrex::max<amrex::Real>(sootdt / rate, mindt);
        }
        if (tstart + sootdt > dt) {
          sootdt = dt - tstart;
        }
        // Update species concentrations within subcycle
        for (int sp = 0; sp < NUM_SOOT_GS; ++sp) {
          xi_n[sp] += sootdt * omega_src[sp];
          rho += sootdt * omega_src[sp] * mw_fluid[sp];
          omega_src[sp] = 0.; // Reset omega source
        }
        // Update moments within subcycle
        for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
          moments[mom] += sootdt * mom_src[mom];
          mom_src[mom] = 0.0; // Reset moment source
        }
        sd->clipMoments(momentsPtr);
        tstart += sootdt;
        isub++;
      }
      // If not finished with the time step, add remaining source
      if (tstart < dt) {
        Real remdt = dt - tstart;
        sd->computeSrcTerms(
          T, mu, rho, molarMass, convT, betaNucl, colConst, xi_n.data(),
          omega_src.data(), momentsPtr, mom_srcPtr, mom_fvPtr, sr);
        // Update species concentrations within subcycle
        for (int sp = 0; sp < NUM_SOOT_GS; ++sp) {
          xi_n[sp] += remdt * omega_src[sp];
          rho += remdt * omega_src[sp] * mw_fluid[sp];
        }
        for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
          moments[mom] += remdt * mom_src[mom];
        }
        sd->clipMoments(momentsPtr);
      }
      sd->convertFromMol(momentsPtr);
      Real vol_src = 0.0; // Volume moment source of this call
      for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
        const int peleIndx = sootIndx + mom;
        const Real msrc = (moments[mom] - mom0[mom]) / dt;
        soot_state(i, j, k, peleIndx) += msrc;
        if (mom == 1) {
          vol_src = msrc;
        }
      }
      Real rho_src = 0.0;
      Real eng_src = 0.0;
      Real p_src = 0.0;
      for (int sp = 0; sp < NUM_SOOT_GS; ++sp) {
        // Convert from local gas species index to global gas species index
        const int spcc = sd->refIndx[sp];
        const int peleIndx = specIndx + spcc;
        Real newrhoY = xi_n[sp] * mw_fluid[sp];
        Real omegai = (newrhoY - rho_YF[spcc]) / dt;
        soot_state(i, j, k, peleIndx) += omegai * sc.mass_src_conv;
        rho_src += omegai;
        eng_src += omegai * Hi[spcc];
        p_src -= omegai * RT / mw_fluid[sp];
      }
      if (conserveMass) {
        // Difference between mass lost from fluid and mass gained to soot
        // (the volume moment is the soot volume fraction)
        Real del_rho_dot = rho_src + vol_src * sc.SootDensity;
        // Add that mass to H2
        soot_state(i, j, k, absorbIndxP) -= del_rho_dot * sc.mass_src_conv;
        rho_src -= del_rho_dot;
        eng_src -= del_rho_dot * Hi[absorbIndxN];
        p_src += del_rho_dot * RT / mw_fluidF[absorbIndxN];
      }
      if (pres_term) {
        eng_src += p_src;
      }
      // Add density source term
      soot_state(i, j, k, rhoIndx) += rho_src * sc.mass_src_conv;
      soot_state(i, j, k, engIndx) += eng_src * sc.eng_src_conv;
    }
  });
}

// Compute time step estimate for soot
Real
SootModel::estSootDt(const Box& vbox, Array4<const Real> const& Qstate) const
{
  // Primitive components
#ifndef PELELM_USE_SOOT
  const int qRhoIndx = m_sootIndx.qRhoIndx;
#endif
  const int qTempIndx = m_sootIndx.qTempIndx;
  const int qSpecIndx = m_sootIndx.qSpecIndx;
  const int qSootIndx = m_sootIndx.qSootIndx;
  const Real Tcutoff = m_Tcutoff;
  const Real Xcutoff = m_Xcutoff;

  const SootData* sd = d_sootData;
  const SootReaction* sr = d_sootReact;
  ReduceOps<ReduceOpMin> reduce_op;
  ReduceData<Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  Real soot_dt = std::numeric_limits<Real>::max();
  SootConst sc;
  BL_PROFILE("estSootDt()");
  reduce_op.eval(
    vbox, reduce_data,
    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
      auto eos = pele::physics::PhysicsType::eos();
      GpuArray<Real, NUM_SPECIES> mw_fluidF;
      GpuArray<Real, NUM_SOOT_GS> mw_fluid;
      eos.molecular_weight(mw_fluidF.data());
      GpuArray<Real, NUM_SOOT_GS> omega_src;
      GpuArray<Real, NUM_SOOT_GS> rho_Y;
      GpuArray<Real, NUM_SOOT_GS> xi_n;
      GpuArray<Real, NUM_SOOT_MOMENTS + 1> moments;
      Real* momentsPtr = moments.data();
      GpuArray<Real, NUM_SOOT_MOMENTS + 2> mom_fv;
      Real* mom_fvPtr = mom_fv.data();
#ifndef PELELM_USE_SOOT
      const Real rho = Qstate(i, j, k, qRhoIndx) * sc.rho_conv;
#endif
      const Real T = Qstate(i, j, k, qTempIndx);
      if (T < Tcutoff) {
        return 1.E20;
      }
      for (int mom = 0; mom < NUM_SOOT_MOMENTS + 1; ++mom) {
        const int peleIndx = qSootIndx + mom;
        moments[mom] = Qstate(i, j, k, peleIndx);
      }
      for (int sp = 0; sp < NUM_SOOT_GS; ++sp) {
        const int spcc = sd->refIndx[sp];
        const int peleIndx = qSpecIndx + spcc;
#ifdef PELELM_USE_SOOT
        rho_Y[sp] = Qstate(i, j, k, peleIndx) * sc.rho_conv;
#else
        rho_Y[sp] = rho * Qstate(i, j, k, peleIndx);
#endif
        mw_fluid[sp] = mw_fluidF[spcc];
        xi_n[sp] = rho_Y[sp] / mw_fluid[sp];
        omega_src[sp] = 0.0;
      }
      sd->convertToMol(momentsPtr);
      sd->clipMoments(momentsPtr);
      sd->computeFracMomVect(momentsPtr, mom_fvPtr);
      Real surf = sc.S0 * sd->fracMom(0.0, 1.0, mom_fvPtr);
      Real k_sg = 0.0;
      Real k_ox = 0.0;
      Real k_o2 = 0.0;
      // Compute the species reaction source terms into omega_src, with the
      // same soot threshold for the surface reactions as the source term
      sr->chemicalSrc(
        T, surf, xi_n.data(), momentsPtr, k_sg, k_ox, k_o2, omega_src.data(),
        sd->surfaceActive(momentsPtr));
      Real mindt = 1.E100;
      for (int sp = 0; sp < NUM_SOOT_GS; ++sp) {
        if (xi_n[sp] > Xcutoff) {
          Real omegai = omega_src[sp] * mw_fluid[sp] + 1.E-16;
          Real rate = -rho_Y[sp] / omegai;
          // Check if source terms cause mass to drop to 0 or less
          if (rate > 0.0 && rate < mindt) {
            mindt = rate;
          }
        }
      }
      return mindt;
    });
  ReduceTuple hv = reduce_data.value();
  Real ldt_cpu = amrex::get<0>(hv);
  soot_dt = amrex::min<amrex::Real>(soot_dt, ldt_cpu);
  return soot_dt;
}
