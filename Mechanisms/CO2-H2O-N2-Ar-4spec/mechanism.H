#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  O
1  H
2  C
3  N
4  Ar
*/

// Species
#define CO2_ID 0
#define N2_ID 1
#define AR_ID 2
#define H2O_ID 3

#define NUM_GAS_ELEMENTS 5  // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 4   // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 0 // Reactions in the homogeneous phase

#define SITE_DENSITY 0.000000E+00 // mol/cm^2

#define NUM_SURFACE_ELEMENTS 0  // Additional elements in heterogeneous phase
#define NUM_SURFACE_SPECIES 0   // Species in the heterogeneous phase
#define NUM_SURFACE_REACTIONS 0 // Reactions in the heterogeneous phase

#define NUM_ELEMENTS (NUM_GAS_ELEMENTS + NUM_SURFACE_ELEMENTS)
#define NUM_SPECIES (NUM_GAS_SPECIES + NUM_SURFACE_SPECIES)
#define NUM_REACTIONS (NUM_GAS_REACTIONS + NUM_SURFACE_REACTIONS)

#define NUM_IONS 0

#define NUM_FIT 4

//  ALWAYS on CPU stuff -- can have different def depending on if we are CPU or
//  GPU based. Defined in mechanism.cpp MISC
void CKSYME_STR(amrex::Vector<std::string>& ename);
void CKSYMS_STR(amrex::Vector<std::string>& kname);
void GET_RMAP(int* _rmap);
void CKINU(const int i, int& nspec, int* ki, int* nu);
void CKKFKR(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real* x,
  amrex::Real* q_f,
  amrex::Real* q_r);
void progressRateFR(
  amrex::Real* q_f, amrex::Real* q_r, amrex::Real* sc, amrex::Real T);
//  SPARSE INFORMATION
void SPARSITY_INFO(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST(int* nJdata, const int* consP, int NCELLS);
void SPARSITY_INFO_SYST_SIMPLIFIED(int* nJdata, const int* consP);
void
SPARSITY_PREPROC_CSC(int* rowVals, int* colPtrs, const int* consP, int NCELLS);
void SPARSITY_PREPROC_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_CSR(
  int* colVals, int* rowPtrs, const int* consP, int NCELLS, int base);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSC(
  int* rowVals, int* colPtrs, int* indx, const int* consP);
void SPARSITY_PREPROC_SYST_SIMPLIFIED_CSR(
  int* colVals, int* rowPtr, const int* consP, int base);

// A few mechanism parameters
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKINDX(int& mm, int& kk, int& ii, int& nfit)
{
  mm = 5;
  kk = 4;
  ii = 0;
  nfit = -1; // Why do you need this anyway ?
}

// save atomic weights into array
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
atomicWeight(amrex::Real* awt)
{
  awt[0] = 15.999000; // O
  awt[1] = 1.008000;  // H
  awt[2] = 12.011000; // C
  awt[3] = 14.007000; // N
  awt[4] = 39.950000; // Ar
}

// get atomic weight for all elements
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKAWT(amrex::Real* awt)
{
  atomicWeight(awt);
}

// Returns the elemental composition
// of the speciesi (mdim is num of elements)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKNCF(int* ncf)
{
  int kd = 5;
  // Zero ncf
  for (int id = 0; id < kd * 4; ++id) {
    ncf[id] = 0;
  }

  // CO2
  ncf[0 * kd + 2] = 1; // C
  ncf[0 * kd + 0] = 2; // O

  // N2
  ncf[1 * kd + 3] = 2; // N

  // AR
  ncf[2 * kd + 4] = 1; // Ar

  // H2O
  ncf[3 * kd + 1] = 2; // H
  ncf[3 * kd + 0] = 1; // O
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[4] = {
  2.2722624917630486e-02, // CO2
  3.5696437495537948e-02, // N2
  2.5031289111389236e-02, // AR
  5.5509297807382736e-02, // H2O
};
#endif
const amrex::Real h_global_imw[4] = {
  2.2722624917630486e-02, // CO2
  3.5696437495537948e-02, // N2
  2.5031289111389236e-02, // AR
  5.5509297807382736e-02, // H2O
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[4] = {
  4.4009000000000000e+01, // CO2
  2.8013999999999999e+01, // N2
  3.9950000000000003e+01, // AR
  1.8015000000000001e+01, // H2O
};
#endif
const amrex::Real h_global_mw[4] = {
  4.4009000000000000e+01, // CO2
  2.8013999999999999e+01, // N2
  3.9950000000000003e+01, // AR
  1.8015000000000001e+01, // H2O
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 2.2722624917630486e-02; // CO2
  imw_new[1] = 3.5696437495537948e-02; // N2
  imw_new[2] = 2.5031289111389236e-02; // AR
  imw_new[3] = 5.5509297807382736e-02; // H2O
}

//  inverse molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
imw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_imw[n];
#else
  return h_global_imw[n];
#endif
}
//  molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_mw(amrex::Real* mw_new)
{
  mw_new[0] = 4.4009000000000000e+01; // CO2
  mw_new[1] = 2.8013999999999999e+01; // N2
  mw_new[2] = 3.9950000000000003e+01; // AR
  mw_new[3] = 1.8015000000000001e+01; // H2O
}

//  molecular weight
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
mw(const int n)
{
#if AMREX_DEVICE_COMPILE
  return global_mw[n];
#else
  return h_global_mw[n];
#endif
}

//  Returns R, Rc, Patm
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRP(amrex::Real& ru, amrex::Real& ruc, amrex::Real& pa)
{
  ru = 8.31446261815324e+07;
  ruc = 1.98721558317399615845;
  pa = 1.01325e+06;
}

// compute Cv/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cv_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with no change across T
  // species 2: AR
  species[2] = +1.500000000000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    species[0] = +1.356773520000000e+00 + 8.984596770000000e-03 * T -
                 7.123562690000000e-06 * T2 + 2.459190220000000e-09 * T3 -
                 1.436995480000000e-13 * T4;
    // species 1: N2
    species[1] = +2.298677000000000e+00 + 1.408240400000000e-03 * T -
                 3.963222000000000e-06 * T2 + 5.641515000000000e-09 * T3 -
                 2.444854000000000e-12 * T4;
    // species 3: H2O
    species[3] = +3.198640560000000e+00 - 2.036434100000000e-03 * T +
                 6.520402110000000e-06 * T2 - 5.487970620000000e-09 * T3 +
                 1.771978170000000e-12 * T4;
  } else {
    // species 0: CO2
    species[0] = +2.857460290000000e+00 + 4.414370260000000e-03 * T -
                 2.214814040000000e-06 * T2 + 5.234901880000001e-10 * T3 -
                 4.720841640000000e-14 * T4;
    // species 1: N2
    species[1] = +1.926640000000000e+00 + 1.487976800000000e-03 * T -
                 5.684760000000000e-07 * T2 + 1.009703800000000e-10 * T3 -
                 6.753351000000000e-15 * T4;
    // species 3: H2O
    species[3] = +2.033992490000000e+00 + 2.176918040000000e-03 * T -
                 1.640725180000000e-07 * T2 - 9.704198700000000e-11 * T3 +
                 1.682009920000000e-14 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with no change across T
  // species 2: AR
  species[2] = +2.500000000000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    species[0] = +2.356773520000000e+00 + 8.984596770000000e-03 * T -
                 7.123562690000000e-06 * T2 + 2.459190220000000e-09 * T3 -
                 1.436995480000000e-13 * T4;
    // species 1: N2
    species[1] = +3.298677000000000e+00 + 1.408240400000000e-03 * T -
                 3.963222000000000e-06 * T2 + 5.641515000000000e-09 * T3 -
                 2.444854000000000e-12 * T4;
    // species 3: H2O
    species[3] = +4.198640560000000e+00 - 2.036434100000000e-03 * T +
                 6.520402110000000e-06 * T2 - 5.487970620000000e-09 * T3 +
                 1.771978170000000e-12 * T4;
  } else {
    // species 0: CO2
    species[0] = +3.857460290000000e+00 + 4.414370260000000e-03 * T -
                 2.214814040000000e-06 * T2 + 5.234901880000001e-10 * T3 -
                 4.720841640000000e-14 * T4;
    // species 1: N2
    species[1] = +2.926640000000000e+00 + 1.487976800000000e-03 * T -
                 5.684760000000000e-07 * T2 + 1.009703800000000e-10 * T3 -
                 6.753351000000000e-15 * T4;
    // species 3: H2O
    species[3] = +3.033992490000000e+00 + 2.176918040000000e-03 * T -
                 1.640725180000000e-07 * T2 - 9.704198700000000e-11 * T3 +
                 1.682009920000000e-14 * T4;
  }
}

// compute the g/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
gibbs(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 2: AR
  species[2] = -7.453750000000000e+02 * invT - 1.866000000000000e+00 -
               2.500000000000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    species[0] = -4.837196970000000e+04 * invT - 7.544278700000000e+00 -
                 2.356773520000000e+00 * logT - 4.492298385000000e-03 * T +
                 1.187260448333333e-06 * T2 - 2.049325183333333e-10 * T3 +
                 7.184977399999999e-15 * T4;
    // species 1: N2
    species[1] = -1.020899900000000e+03 * invT - 6.516950000000001e-01 -
                 3.298677000000000e+00 * logT - 7.041202000000000e-04 * T +
                 6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                 1.222427000000000e-13 * T4;
    // species 3: H2O
    species[3] = -3.029372670000000e+04 * invT + 5.047672768000000e+00 -
                 4.198640560000000e+00 * logT + 1.018217050000000e-03 * T -
                 1.086733685000000e-06 * T2 + 4.573308850000000e-10 * T3 -
                 8.859890850000000e-14 * T4;
  } else {
    // species 0: CO2
    species[0] = -4.875916600000000e+04 * invT + 1.585822230000000e+00 -
                 3.857460290000000e+00 * logT - 2.207185130000000e-03 * T +
                 3.691356733333334e-07 * T2 - 4.362418233333334e-11 * T3 +
                 2.360420820000000e-15 * T4;
    // species 1: N2
    species[1] = -9.227977000000000e+02 * invT - 3.053888000000000e+00 -
                 2.926640000000000e+00 * logT - 7.439884000000000e-04 * T +
                 9.474600000000001e-08 * T2 - 8.414198333333333e-12 * T3 +
                 3.376675500000000e-16 * T4;
    // species 3: H2O
    species[3] = -3.000429710000000e+04 * invT - 1.932777610000000e+00 -
                 3.033992490000000e+00 * logT - 1.088459020000000e-03 * T +
                 2.734541966666666e-08 * T2 + 8.086832250000000e-12 * T3 -
                 8.410049600000000e-16 * T4;
  }
}

// compute the a/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
helmholtz(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 2: AR
  species[2] = -7.453750000000000e+02 * invT - 2.866000000000000e+00 -
               2.500000000000000e+00 * logT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    species[0] = -4.837196970000000e+04 * invT - 8.544278700000000e+00 -
                 2.356773520000000e+00 * logT - 4.492298385000000e-03 * T +
                 1.187260448333333e-06 * T2 - 2.049325183333333e-10 * T3 +
                 7.184977399999999e-15 * T4;
    // species 1: N2
    species[1] = -1.020899900000000e+03 * invT - 1.651695000000000e+00 -
                 3.298677000000000e+00 * logT - 7.041202000000000e-04 * T +
                 6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                 1.222427000000000e-13 * T4;
    // species 3: H2O
    species[3] = -3.029372670000000e+04 * invT + 4.047672768000000e+00 -
                 4.198640560000000e+00 * logT + 1.018217050000000e-03 * T -
                 1.086733685000000e-06 * T2 + 4.573308850000000e-10 * T3 -
                 8.859890850000000e-14 * T4;
  } else {
    // species 0: CO2
    species[0] = -4.875916600000000e+04 * invT + 5.858222300000002e-01 -
                 3.857460290000000e+00 * logT - 2.207185130000000e-03 * T +
                 3.691356733333334e-07 * T2 - 4.362418233333334e-11 * T3 +
                 2.360420820000000e-15 * T4;
    // species 1: N2
    species[1] = -9.227977000000000e+02 * invT - 4.053888000000000e+00 -
                 2.926640000000000e+00 * logT - 7.439884000000000e-04 * T +
                 9.474600000000001e-08 * T2 - 8.414198333333333e-12 * T3 +
                 3.376675500000000e-16 * T4;
    // species 3: H2O
    species[3] = -3.000429710000000e+04 * invT - 2.932777610000000e+00 -
                 3.033992490000000e+00 * logT - 1.088459020000000e-03 * T +
                 2.734541966666666e-08 * T2 + 8.086832250000000e-12 * T3 -
                 8.410049600000000e-16 * T4;
  }
}

// compute the e/(RT) at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesInternalEnergy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 2: AR
  species[2] = +1.500000000000000e+00 - 7.453750000000000e+02 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    species[0] = +1.356773520000000e+00 + 4.492298385000000e-03 * T -
                 2.374520896666666e-06 * T2 + 6.147975550000000e-10 * T3 -
                 2.873990960000000e-14 * T4 - 4.837196970000000e+04 * invT;
    // species 1: N2
    species[1] = +2.298677000000000e+00 + 7.041202000000000e-04 * T -
                 1.321074000000000e-06 * T2 + 1.410378750000000e-09 * T3 -
                 4.889707999999999e-13 * T4 - 1.020899900000000e+03 * invT;
    // species 3: H2O
    species[3] = +3.198640560000000e+00 - 1.018217050000000e-03 * T +
                 2.173467370000000e-06 * T2 - 1.371992655000000e-09 * T3 +
                 3.543956340000000e-13 * T4 - 3.029372670000000e+04 * invT;
  } else {
    // species 0: CO2
    species[0] = +2.857460290000000e+00 + 2.207185130000000e-03 * T -
                 7.382713466666667e-07 * T2 + 1.308725470000000e-10 * T3 -
                 9.441683280000000e-15 * T4 - 4.875916600000000e+04 * invT;
    // species 1: N2
    species[1] = +1.926640000000000e+00 + 7.439884000000000e-04 * T -
                 1.894920000000000e-07 * T2 + 2.524259500000000e-11 * T3 -
                 1.350670200000000e-15 * T4 - 9.227977000000000e+02 * invT;
    // species 3: H2O
    species[3] = +2.033992490000000e+00 + 1.088459020000000e-03 * T -
                 5.469083933333333e-08 * T2 - 2.426049675000000e-11 * T3 +
                 3.364019840000000e-15 * T4 - 3.000429710000000e+04 * invT;
  }
}

// compute the h/(RT) at the given temperature (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEnthalpy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 2: AR
  species[2] = +2.500000000000000e+00 - 7.453750000000000e+02 * invT;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    species[0] = +2.356773520000000e+00 + 4.492298385000000e-03 * T -
                 2.374520896666666e-06 * T2 + 6.147975550000000e-10 * T3 -
                 2.873990960000000e-14 * T4 - 4.837196970000000e+04 * invT;
    // species 1: N2
    species[1] = +3.298677000000000e+00 + 7.041202000000000e-04 * T -
                 1.321074000000000e-06 * T2 + 1.410378750000000e-09 * T3 -
                 4.889707999999999e-13 * T4 - 1.020899900000000e+03 * invT;
    // species 3: H2O
    species[3] = +4.198640560000000e+00 - 1.018217050000000e-03 * T +
                 2.173467370000000e-06 * T2 - 1.371992655000000e-09 * T3 +
                 3.543956340000000e-13 * T4 - 3.029372670000000e+04 * invT;
  } else {
    // species 0: CO2
    species[0] = +3.857460290000000e+00 + 2.207185130000000e-03 * T -
                 7.382713466666667e-07 * T2 + 1.308725470000000e-10 * T3 -
                 9.441683280000000e-15 * T4 - 4.875916600000000e+04 * invT;
    // species 1: N2
    species[1] = +2.926640000000000e+00 + 7.439884000000000e-04 * T -
                 1.894920000000000e-07 * T2 + 2.524259500000000e-11 * T3 -
                 1.350670200000000e-15 * T4 - 9.227977000000000e+02 * invT;
    // species 3: H2O
    species[3] = +3.033992490000000e+00 + 1.088459020000000e-03 * T -
                 5.469083933333333e-08 * T2 - 2.426049675000000e-11 * T3 +
                 3.364019840000000e-15 * T4 - 3.000429710000000e+04 * invT;
  }
}

// compute the S/R at the given temperature (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
speciesEntropy(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real logT = log(T);

  // species with no change across T
  // species 2: AR
  species[2] = +2.500000000000000e+00 * logT + 4.366000000000000e+00;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    species[0] = +2.356773520000000e+00 * logT + 8.984596770000000e-03 * T -
                 3.561781345000000e-06 * T2 + 8.197300733333333e-10 * T3 -
                 3.592488700000000e-14 * T4 + 9.901052220000000e+00;
    // species 1: N2
    species[1] = +3.298677000000000e+00 * logT + 1.408240400000000e-03 * T -
                 1.981611000000000e-06 * T2 + 1.880505000000000e-09 * T3 -
                 6.112135000000000e-13 * T4 + 3.950372000000000e+00;
    // species 3: H2O
    species[3] = +4.198640560000000e+00 * logT - 2.036434100000000e-03 * T +
                 3.260201055000000e-06 * T2 - 1.829323540000000e-09 * T3 +
                 4.429945425000000e-13 * T4 - 8.490322080000000e-01;
  } else {
    // species 0: CO2
    species[0] = +3.857460290000000e+00 * logT + 4.414370260000000e-03 * T -
                 1.107407020000000e-06 * T2 + 1.744967293333334e-10 * T3 -
                 1.180210410000000e-14 * T4 + 2.271638060000000e+00;
    // species 1: N2
    species[1] = +2.926640000000000e+00 * logT + 1.487976800000000e-03 * T -
                 2.842380000000000e-07 * T2 + 3.365679333333333e-11 * T3 -
                 1.688337750000000e-15 * T4 + 5.980528000000000e+00;
    // species 3: H2O
    species[3] = +3.033992490000000e+00 * logT + 2.176918040000000e-03 * T -
                 8.203625899999999e-08 * T2 - 3.234732900000000e-11 * T3 +
                 4.205024800000000e-15 * T4 + 4.966770100000000e+00;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;

  // species with no change across T
  // species 2: AR
  species[2] = 0.0;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    species[0] = +8.984596770000000e-03 - 1.424712538000000e-05 * T +
                 7.377570659999999e-09 * T2 - 5.747981920000000e-13 * T3;
    // species 1: N2
    species[1] = +1.408240400000000e-03 - 7.926444000000000e-06 * T +
                 1.692454500000000e-08 * T2 - 9.779416000000000e-12 * T3;
    // species 3: H2O
    species[3] = -2.036434100000000e-03 + 1.304080422000000e-05 * T -
                 1.646391186000000e-08 * T2 + 7.087912680000000e-12 * T3;
  } else {
    // species 0: CO2
    species[0] = +4.414370260000000e-03 - 4.429628080000000e-06 * T +
                 1.570470564000000e-09 * T2 - 1.888336656000000e-13 * T3;
    // species 1: N2
    species[1] = +1.487976800000000e-03 - 1.136952000000000e-06 * T +
                 3.029111400000000e-10 * T2 - 2.701340400000000e-14 * T3;
    // species 3: H2O
    species[3] = +2.176918040000000e-03 - 3.281450360000000e-07 * T -
                 2.911259610000000e-10 * T2 + 6.728039680000000e-14 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[4]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 4; ++id) {
    result += x[id] * cpor[id];
  }

  cpbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CP (Eq. 34)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cpbs)
{
  amrex::Real result = 0.0;

  // compute Cp/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with no change across T
  // species 2: AR
  result += y[2] * (+2.500000000000000e+00) * 0.0250312891113892;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    result += y[0] *
              (+2.356773520000000e+00 + 8.984596770000000e-03 * T -
               7.123562690000000e-06 * T2 + 2.459190220000000e-09 * T3 -
               1.436995480000000e-13 * T4) *
              0.0227226249176305;
    // species 1: N2
    result += y[1] *
              (+3.298677000000000e+00 + 1.408240400000000e-03 * T -
               3.963222000000000e-06 * T2 + 5.641515000000000e-09 * T3 -
               2.444854000000000e-12 * T4) *
              0.0356964374955379;
    // species 3: H2O
    result += y[3] *
              (+4.198640560000000e+00 - 2.036434100000000e-03 * T +
               6.520402110000000e-06 * T2 - 5.487970620000000e-09 * T3 +
               1.771978170000000e-12 * T4) *
              0.0555092978073827;
  } else {
    // species 0: CO2
    result += y[0] *
              (+3.857460290000000e+00 + 4.414370260000000e-03 * T -
               2.214814040000000e-06 * T2 + 5.234901880000001e-10 * T3 -
               4.720841640000000e-14 * T4) *
              0.0227226249176305;
    // species 1: N2
    result += y[1] *
              (+2.926640000000000e+00 + 1.487976800000000e-03 * T -
               5.684760000000000e-07 * T2 + 1.009703800000000e-10 * T3 -
               6.753351000000000e-15 * T4) *
              0.0356964374955379;
    // species 3: H2O
    result += y[3] *
              (+3.033992490000000e+00 + 2.176918040000000e-03 * T -
               1.640725180000000e-07 * T2 - 9.704198700000000e-11 * T3 +
               1.682009920000000e-14 * T4) *
              0.0555092978073827;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[4]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 4; ++id) {
    result += x[id] * cvor[id];
  }

  cvbl = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 36)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBS(const amrex::Real T, const amrex::Real y[], amrex::Real& cvbs)
{
  amrex::Real result = 0.0;
  // compute Cv/R at the given temperature
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with no change across T
  // species 2: AR
  result += y[2] * (+1.500000000000000e+00) * 0.0250312891113892;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    result += y[0] *
              (+1.356773520000000e+00 + 8.984596770000000e-03 * T -
               7.123562690000000e-06 * T2 + 2.459190220000000e-09 * T3 -
               1.436995480000000e-13 * T4) *
              0.0227226249176305;
    // species 1: N2
    result += y[1] *
              (+2.298677000000000e+00 + 1.408240400000000e-03 * T -
               3.963222000000000e-06 * T2 + 5.641515000000000e-09 * T3 -
               2.444854000000000e-12 * T4) *
              0.0356964374955379;
    // species 3: H2O
    result += y[3] *
              (+3.198640560000000e+00 - 2.036434100000000e-03 * T +
               6.520402110000000e-06 * T2 - 5.487970620000000e-09 * T3 +
               1.771978170000000e-12 * T4) *
              0.0555092978073827;
  } else {
    // species 0: CO2
    result += y[0] *
              (+2.857460290000000e+00 + 4.414370260000000e-03 * T -
               2.214814040000000e-06 * T2 + 5.234901880000001e-10 * T3 -
               4.720841640000000e-14 * T4) *
              0.0227226249176305;
    // species 1: N2
    result += y[1] *
              (+1.926640000000000e+00 + 1.487976800000000e-03 * T -
               5.684760000000000e-07 * T2 + 1.009703800000000e-10 * T3 -
               6.753351000000000e-15 * T4) *
              0.0356964374955379;
    // species 3: H2O
    result += y[3] *
              (+2.033992490000000e+00 + 2.176918040000000e-03 * T -
               1.640725180000000e-07 * T2 - 9.704198700000000e-11 * T3 +
               1.682009920000000e-14 * T4) *
              0.0555092978073827;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[4];                        //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 4; ++id) {
    result += x[id] * hml[id];
  }

  hbml = result * RT;
}

// Returns mean enthalpy of mixture in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& hbms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 2: AR
  result += y[2] * (+2.500000000000000e+00 - 7.453750000000000e+02 * invT) *
            0.0250312891113892;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    result += y[0] *
              (+2.356773520000000e+00 + 4.492298385000000e-03 * T -
               2.374520896666666e-06 * T2 + 6.147975550000000e-10 * T3 -
               2.873990960000000e-14 * T4 - 4.837196970000000e+04 * invT) *
              0.0227226249176305;
    // species 1: N2
    result += y[1] *
              (+3.298677000000000e+00 + 7.041202000000000e-04 * T -
               1.321074000000000e-06 * T2 + 1.410378750000000e-09 * T3 -
               4.889707999999999e-13 * T4 - 1.020899900000000e+03 * invT) *
              0.0356964374955379;
    // species 3: H2O
    result += y[3] *
              (+4.198640560000000e+00 - 1.018217050000000e-03 * T +
               2.173467370000000e-06 * T2 - 1.371992655000000e-09 * T3 +
               3.543956340000000e-13 * T4 - 3.029372670000000e+04 * invT) *
              0.0555092978073827;
  } else {
    // species 0: CO2
    result += y[0] *
              (+3.857460290000000e+00 + 2.207185130000000e-03 * T -
               7.382713466666667e-07 * T2 + 1.308725470000000e-10 * T3 -
               9.441683280000000e-15 * T4 - 4.875916600000000e+04 * invT) *
              0.0227226249176305;
    // species 1: N2
    result += y[1] *
              (+2.926640000000000e+00 + 7.439884000000000e-04 * T -
               1.894920000000000e-07 * T2 + 2.524259500000000e-11 * T3 -
               1.350670200000000e-15 * T4 - 9.227977000000000e+02 * invT) *
              0.0356964374955379;
    // species 3: H2O
    result += y[3] *
              (+3.033992490000000e+00 + 1.088459020000000e-03 * T -
               5.469083933333333e-08 * T2 - 2.426049675000000e-11 * T3 +
               3.364019840000000e-15 * T4 - 3.000429710000000e+04 * invT) *
              0.0555092978073827;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[4];                        //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 4; ++id) {
    result += x[id] * uml[id];
  }

  ubml = result * RT;
}

// get mean internal energy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBMS(const amrex::Real T, const amrex::Real y[], amrex::Real& ubms)
{
  amrex::Real result = 0.0;
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;
  const amrex::Real invT = 1.0 / T;

  // species with no change across T
  // species 2: AR
  result += y[2] * (+1.500000000000000e+00 - 7.453750000000000e+02 * invT) *
            0.0250312891113892;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: CO2
    result += y[0] *
              (+1.356773520000000e+00 + 4.492298385000000e-03 * T -
               2.374520896666666e-06 * T2 + 6.147975550000000e-10 * T3 -
               2.873990960000000e-14 * T4 - 4.837196970000000e+04 * invT) *
              0.0227226249176305;
    // species 1: N2
    result += y[1] *
              (+2.298677000000000e+00 + 7.041202000000000e-04 * T -
               1.321074000000000e-06 * T2 + 1.410378750000000e-09 * T3 -
               4.889707999999999e-13 * T4 - 1.020899900000000e+03 * invT) *
              0.0356964374955379;
    // species 3: H2O
    result += y[3] *
              (+3.198640560000000e+00 - 1.018217050000000e-03 * T +
               2.173467370000000e-06 * T2 - 1.371992655000000e-09 * T3 +
               3.543956340000000e-13 * T4 - 3.029372670000000e+04 * invT) *
              0.0555092978073827;
  } else {
    // species 0: CO2
    result += y[0] *
              (+2.857460290000000e+00 + 2.207185130000000e-03 * T -
               7.382713466666667e-07 * T2 + 1.308725470000000e-10 * T3 -
               9.441683280000000e-15 * T4 - 4.875916600000000e+04 * invT) *
              0.0227226249176305;
    // species 1: N2
    result += y[1] *
              (+1.926640000000000e+00 + 7.439884000000000e-04 * T -
               1.894920000000000e-07 * T2 + 2.524259500000000e-11 * T3 -
               1.350670200000000e-15 * T4 - 9.227977000000000e+02 * invT) *
              0.0356964374955379;
    // species 3: H2O
    result += y[3] *
              (+2.033992490000000e+00 + 1.088459020000000e-03 * T -
               5.469083933333333e-08 * T2 - 2.426049675000000e-11 * T3 +
               3.364019840000000e-15 * T4 - 3.000429710000000e+04 * invT) *
              0.0555092978073827;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  ubms = result * RT;
}

// get mixture entropy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBML(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& sbml)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[4]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 4; ++id) {
    result += x[id] * (sor[id] - log((x[id] + 1e-100)) - logPratio);
  }

  sbml = result * 8.31446261815324e+07;
}

// get mixture entropy in mass units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSBMS(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& sbms)
{
  amrex::Real result = 0;
  // Log of normalized pressure in cgs units dynes/cm^2 by Patm
  amrex::Real logPratio = log(P / 1013250.0);
  amrex::Real sor[4];  //  temporary storage
  amrex::Real x[4];    //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (4.4009000000000000e+01 * YOW);
  x[1] = y[1] / (2.8013999999999999e+01 * YOW);
  x[2] = y[2] / (3.9950000000000003e+01 * YOW);
  x[3] = y[3] / (1.8015000000000001e+01 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 4; i++) {
    result += x[i] * (sor[i] - log((x[i] + 1e-100)) - logPratio);
  }
  // Scale by R/W
  sbms = result * 8.31446261815324e+07 * YOW;
}

//  get temperature given internal energy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_EY(
  const amrex::Real e, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 300.0;  // max lower bound for thermo def
  amrex::Real tmax = 3500.0; // min upper bound for thermo def
  amrex::Real e1, emin, emax, cv, t1, dt;
  CKUBMS(tmin, y, emin);
  CKUBMS(tmax, y, emax);
  if (e < emin) {
    // Linear Extrapolation below tmin
    CKCVBS(tmin, y, cv);
    t = tmin - (emin - e) / cv;
    ierr = 1;
    return;
  }
  if (e > emax) {
    // Linear Extrapolation above tmax
    CKCVBS(tmax, y, cv);
    t = tmax - (emax - e) / cv;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (e - emin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKUBMS(t1, y, e1);
    CKCVBS(t1, y, cv);
    dt = (e - e1) / cv;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

//  get temperature given enthalpy in mass units and mass fracs
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_T_GIVEN_HY(
  const amrex::Real h, const amrex::Real y[], amrex::Real& t, int& ierr)
{
#ifdef CONVERGENCE
  const int maxiter = 5000;
  const amrex::Real tol = 1.e-12;
#else
  const int maxiter = 200;
  const amrex::Real tol = 1.e-6;
#endif
  amrex::Real tmin = 300.0;  // max lower bound for thermo def
  amrex::Real tmax = 3500.0; // min upper bound for thermo def
  amrex::Real h1, hmin, hmax, cp, t1, dt;
  CKHBMS(tmin, y, hmin);
  CKHBMS(tmax, y, hmax);
  if (h < hmin) {
    // Linear Extrapolation below tmin
    CKCPBS(tmin, y, cp);
    t = tmin - (hmin - h) / cp;
    ierr = 1;
    return;
  }
  if (h > hmax) {
    // Linear Extrapolation above tmax
    CKCPBS(tmax, y, cp);
    t = tmax - (hmax - h) / cp;
    ierr = 1;
    return;
  }
  t1 = t;
  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (hmax - hmin) * (h - hmin);
  }
  for (int i = 0; i < maxiter; ++i) {
    CKHBMS(t1, y, h1);
    CKCPBS(t1, y, cp);
    dt = (h - h1) / cp;
    if (dt > 100.) {
      dt = 100.;
    } else if (dt < -100.) {
      dt = -100.;
    } else if (fabs(dt) < tol) {
      break;
    }
    t1 += dt;
  }
  t = t1;
  ierr = 0;
}

// Compute P = rhoRT/W(x)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPX(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& P)
{
  amrex::Real XW = 0;                      //  To hold mean molecular wt
  XW += x[0] * 4.4009000000000000e+01;     // CO2
  XW += x[1] * 2.8013999999999999e+01;     // N2
  XW += x[2] * 3.9950000000000003e+01;     // AR
  XW += x[3] * 1.8015000000000001e+01;     // H2O
  P = rho * 8.31446261815324e+07 * T / XW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(y)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPY(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& P)
{
  amrex::Real YOW = 0; //  for computing mean MW

  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }
  P = rho * 8.31446261815324e+07 * T * YOW; // P = rho*R*T/W
}

// Compute P = rhoRT/W(c)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKPC(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& P)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 4.4009000000000000e+01; // CO2
  W += c[1] * 2.8013999999999999e+01; // N2
  W += c[2] * 3.9950000000000003e+01; // AR
  W += c[3] * 1.8015000000000001e+01; // H2O

  for (int id = 0; id < 4; ++id) {
    sumC += c[id];
  }
  P = rho * 8.31446261815324e+07 * T * sumC / W; // P = rho*R*T/W
}

// Compute rho = PW(x)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOX(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real& rho)
{
  amrex::Real XW = 0;                        //  To hold mean molecular wt
  XW += x[0] * 4.4009000000000000e+01;       // CO2
  XW += x[1] * 2.8013999999999999e+01;       // N2
  XW += x[2] * 3.9950000000000003e+01;       // AR
  XW += x[3] * 1.8015000000000001e+01;       // H2O
  rho = P * XW / (8.31446261815324e+07 * T); // rho = P*W/(R*T)
}

// Compute rho = P*W(y)/RT
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOY(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real& rho)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }

  rho = P / (8.31446261815324e+07 * T * YOW); // rho = P*W/(R*T)
}

// Compute rho = P*W(c)/(R*T)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKRHOC(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real c[],
  amrex::Real& rho)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 4.4009000000000000e+01; // CO2
  W += c[1] * 2.8013999999999999e+01; // N2
  W += c[2] * 3.9950000000000003e+01; // AR
  W += c[3] * 1.8015000000000001e+01; // H2O

  for (int id = 0; id < 4; ++id) {
    sumC += c[id];
  }
  rho = P * W / (sumC * T * 8.31446261815324e+07); // rho = PW/(R*T)
}

// get molecular weight for all species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWT(amrex::Real wt[])
{
  get_mw(wt);
}

// given y[species]: mass fractions
// s mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWY(const amrex::Real y[], amrex::Real& wtm)
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;                  //  see Eq 4 in CK Manual
  XW += x[0] * 4.4009000000000000e+01; // CO2
  XW += x[1] * 2.8013999999999999e+01; // N2
  XW += x[2] * 3.9950000000000003e+01; // AR
  XW += x[3] * 1.8015000000000001e+01; // H2O
  wtm = XW;
}

// given c[species]: molar concentration
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWC(const amrex::Real c[], amrex::Real& wtm)
{
  // See Eq 5 in CK Manual
  amrex::Real W = 0;
  amrex::Real sumC = 0;
  W += c[0] * 4.4009000000000000e+01; // CO2
  W += c[1] * 2.8013999999999999e+01; // N2
  W += c[2] * 3.9950000000000003e+01; // AR
  W += c[3] * 1.8015000000000001e+01; // H2O

  for (int id = 0; id < 4; ++id) {
    sumC += c[id];
  }
  //  CK provides no guard against division by zero
  wtm = W / sumC;
}

// get Cp/R as a function of T
// for all species (Eq 19)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPOR(const amrex::Real T, amrex::Real cpor[])
{
  cp_R(cpor, T);
}

// get H/RT as a function of T
// for all species (Eq 20)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHORT(const amrex::Real T, amrex::Real hort[])
{
  speciesEnthalpy(hort, T);
}

// get S/R as a function of T
// for all species (Eq 21)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSOR(const amrex::Real T, amrex::Real sor[])
{
  speciesEntropy(sor, T);
}

// convert y[species] (mass fracs) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTX(const amrex::Real y[], amrex::Real x[])
{
  amrex::Real YOW = 0;

  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 4; i++) {
    x[i] = y[i] * imw(i) * YOWINV;
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real c[])
{
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 4; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 4; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 4; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }
}

// convert y[species] (mass fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKYTCR(
  const amrex::Real rho,
  amrex::Real /*T*/,
  const amrex::Real y[],
  amrex::Real c[])
{

  for (int i = 0; i < 4; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 4.4009000000000000e+01; // CO2
  XW += x[1] * 2.8013999999999999e+01; // N2
  XW += x[2] * 3.9950000000000003e+01; // AR
  XW += x[3] * 1.8015000000000001e+01; // H2O
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 4.4009000000000000e+01 * XWinv;
  y[1] = x[1] * 2.8013999999999999e+01 * XWinv;
  y[2] = x[2] * 3.9950000000000003e+01 * XWinv;
  y[3] = x[3] * 1.8015000000000001e+01 * XWinv;
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real PORT = P / (8.31446261815324e+07 * T); // P/RT

  // Compute conversion, see Eq 10
  for (int id = 0; id < 4; ++id) {
    c[id] = x[id] * PORT;
  }
}

// convert x[species] (mole fracs) to c[species] (molar conc)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTCR(
  const amrex::Real rho,
  const amrex::Real /*T*/,
  const amrex::Real x[],
  amrex::Real c[])
{
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 4.4009000000000000e+01; // CO2
  XW += x[1] * 2.8013999999999999e+01; // N2
  XW += x[2] * 3.9950000000000003e+01; // AR
  XW += x[3] * 1.8015000000000001e+01; // H2O
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 4; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 4; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 4; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 4.4009000000000000e+01; // CO2
  CW += c[1] * 2.8013999999999999e+01; // N2
  CW += c[2] * 3.9950000000000003e+01; // AR
  CW += c[3] * 1.8015000000000001e+01; // H2O
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 4.4009000000000000e+01 * CWinv;
  y[1] = c[1] * 2.8013999999999999e+01 * CWinv;
  y[2] = c[2] * 3.9950000000000003e+01 * CWinv;
  y[3] = c[3] * 1.8015000000000001e+01 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    cvml[id] *= 8.31446261815324e+07;
  }
}

// get specific heat at constant pressure as a
// function of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPML(const amrex::Real T, amrex::Real cpml[])
{
  cp_R(cpml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    cpml[id] *= 8.31446261815324e+07;
  }
}

// get internal energy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUML(const amrex::Real T, amrex::Real uml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    uml[id] *= RT;
  }
}

// get enthalpy as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHML(const amrex::Real T, amrex::Real hml[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    sml[id] *= 8.31446261815324e+07;
  }
}

// Returns the specific heats at constant volume
// in mass units (Eq. 29)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVMS(const amrex::Real T, amrex::Real cvms[])
{
  cv_R(cvms, T);
  // multiply by R/molecularweight
  cvms[0] *= 1.889264154639560e+06; // CO2
  cvms[1] *= 2.967966951578939e+06; // N2
  cvms[2] *= 2.081217176008320e+06; // AR
  cvms[3] *= 4.615299815794193e+06; // H2O
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 1.889264154639560e+06; // CO2
  cpms[1] *= 2.967966951578939e+06; // N2
  cpms[2] *= 2.081217176008320e+06; // AR
  cpms[3] *= 4.615299815794193e+06; // H2O
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 4; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 4; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 1.889264154639560e+06; // CO2
  sms[1] *= 2.967966951578939e+06; // N2
  sms[2] *= 2.081217176008320e+06; // AR
  sms[3] *= 4.615299815794193e+06; // H2O
}

// GPU version of productionRate: no more use of thermo namespace vectors
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
comp_qfqr(
  amrex::Real* /*qf*/,
  amrex::Real* /*qr*/,
  const amrex::Real* /*sc*/,
  const amrex::Real* /*sc_qss*/,
  const amrex::Real /*T*/,
  const amrex::Real /*invT*/,
  const amrex::Real /*logT*/)
{
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(
  amrex::Real* wdot, const amrex::Real* /*sc*/, const amrex::Real /*T*/)
{

  for (int i = 0; i < 4; ++i) {
    wdot[i] = 0.0;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 4; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    C[id] *= 1.0e-6;
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[4]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 4; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 4; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given P, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXP(
  const amrex::Real P,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[4]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 4; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mass fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWYR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real y[],
  amrex::Real wdot[])
{
  amrex::Real c[4]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 4; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

// Returns the molar production rate of species
// Given rho, T, and mole fractions
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWXR(
  const amrex::Real rho,
  const amrex::Real T,
  const amrex::Real x[],
  amrex::Real wdot[])
{
  amrex::Real c[4];   // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 4.4009000000000000e+01; // CO2
  XW += x[1] * 2.8013999999999999e+01; // N2
  XW += x[2] * 3.9950000000000003e+01; // AR
  XW += x[3] * 1.8015000000000001e+01; // H2O
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 4; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 4; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0; // CO2
  kcharge[1] = 0; // N2
  kcharge[2] = 0; // AR
  kcharge[3] = 0; // H2O
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[4];
  CKCHRG(kchrg);

  for (int id = 0; id < 4; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J,
  const amrex::Real* /*sc*/,
  const amrex::Real /*T*/,
  const int /*HP*/)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 25; i++) {
    J[i] = 0.0;
  }
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[4];

  for (int k = 0; k < 4; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 4; k++) {
    J[20 + k] *= 1.e-6;
    J[k * 5 + 4] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J,
  const amrex::Real* /*sc*/,
  const amrex::Real /*T*/,
  const int /*consP*/)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 25; i++) {
    J[i] = 0.0;
  }
#else
  amrex::Abort();
#endif
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT(
  amrex::Real* J,
  const amrex::Real* sc,
  const amrex::Real* Tp,
  const int* consP)
{
  amrex::Real c[4];

  for (int k = 0; k < 4; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 4; k++) {
    J[20 + k] *= 1.e-6;
    J[k * 5 + 4] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 16;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 412;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLITE(int* NLITE)
{
  *NLITE = 0;
}

// Patm in ergs/cm3
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPATM(amrex::Real* PATM)
{
  *PATM = 0.1013250000000000E+07;
}

// the molecular weights in g/mol
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetWT(amrex::Real* WT)
{
  WT[0] = 4.40090000E+01;
  WT[1] = 2.80140000E+01;
  WT[2] = 3.99500000E+01;
  WT[3] = 1.80150000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 2.44000000E+02;
  EPS[1] = 9.75300000E+01;
  EPS[2] = 1.36500000E+02;
  EPS[3] = 5.72400000E+02;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 3.76300000E+00;
  SIG[1] = 3.62100000E+00;
  SIG[2] = 3.33000000E+00;
  SIG[3] = 2.60500000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 1.84400000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 2.65000000E+00;
  POL[1] = 1.76000000E+00;
  POL[2] = 0.00000000E+00;
  POL[3] = 0.00000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 2.10000000E+00;
  ZROT[1] = 4.00000000E+00;
  ZROT[2] = 0.00000000E+00;
  ZROT[3] = 4.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 1;
  NLIN[1] = 1;
  NLIN[2] = 0;
  NLIN[3] = 2;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -2.36749638E+01;
  COFETA[1] = 4.99775518E+00;
  COFETA[2] = -5.52687718E-01;
  COFETA[3] = 2.34353338E-02;
  COFETA[4] = -1.62526676E+01;
  COFETA[5] = 2.24839597E+00;
  COFETA[6] = -2.13428438E-01;
  COFETA[7] = 9.46192413E-03;
  COFETA[8] = -1.86067352E+01;
  COFETA[9] = 3.27402596E+00;
  COFETA[10] = -3.45827972E-01;
  COFETA[11] = 1.51622680E-02;
  COFETA[12] = -1.17770995E+01;
  COFETA[13] = -8.26744826E-01;
  COFETA[14] = 3.39009392E-01;
  COFETA[15] = -2.00674472E-02;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = -1.24047472E+01;
  COFLAM[1] = 6.34783072E+00;
  COFLAM[2] = -6.37857831E-01;
  COFLAM[3] = 2.37613806E-02;
  COFLAM[4] = 1.15506870E+01;
  COFLAM[5] = -2.91452034E+00;
  COFLAM[6] = 5.55043078E-01;
  COFLAM[7] = -2.75172223E-02;
  COFLAM[8] = -2.73651502E+00;
  COFLAM[9] = 3.27402596E+00;
  COFLAM[10] = -3.45827972E-01;
  COFLAM[11] = 1.51622680E-02;
  COFLAM[12] = 2.28195672E+01;
  COFLAM[13] = -8.72278727E+00;
  COFLAM[14] = 1.49300458E+00;
  COFLAM[15] = -7.41523911E-02;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -2.10907633E+01;
  COFD[1] = 5.29211327E+00;
  COFD[2] = -4.56068366E-01;
  COFD[3] = 1.91195062E-02;
  COFD[4] = -1.77350630E+01;
  COFD[5] = 4.19328271E+00;
  COFD[6] = -3.26911461E-01;
  COFD[7] = 1.40520357E-02;
  COFD[8] = -1.47639411E+01;
  COFD[9] = 3.15955654E+00;
  COFD[10] = -1.97590757E-01;
  COFD[11] = 8.64692156E-03;
  COFD[12] = -1.90183604E+01;
  COFD[13] = 4.64763677E+00;
  COFD[14] = -3.82799418E-01;
  COFD[15] = 1.63539171E-02;
  COFD[16] = -1.57236886E+01;
  COFD[17] = 3.51447210E+00;
  COFD[18] = -2.42579007E-01;
  COFD[19] = 1.05506318E-02;
  COFD[20] = -1.68944987E+01;
  COFD[21] = 3.94346012E+00;
  COFD[22] = -2.96835271E-01;
  COFD[23] = 1.28438696E-02;
  COFD[24] = -2.11719364E+01;
  COFD[25] = 5.20775027E+00;
  COFD[26] = -4.07348285E-01;
  COFD[27] = 1.55473262E-02;
  COFD[28] = -2.07768625E+01;
  COFD[29] = 5.42470175E+00;
  COFD[30] = -4.69700438E-01;
  COFD[31] = 1.95706912E-02;
  COFD[32] = -2.10785351E+01;
  COFD[33] = 5.51573149E+00;
  COFD[34] = -4.78177665E-01;
  COFD[35] = 1.98082796E-02;
  COFD[36] = -1.31492509E+01;
  COFD[37] = 1.48004045E+00;
  COFD[38] = 1.60499956E-01;
  COFD[39] = -1.19765871E-02;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* /*KTDIF*/)
{
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* /*COFTD*/)
{
}

// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS(
  amrex::Real* Tci, amrex::Real* ai, amrex::Real* bi, amrex::Real* acentric_i)
{

  amrex::Real EPS[4];
  amrex::Real SIG[4];
  amrex::Real wt[4];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: CO2
  // Imported from NIST
  Tci[0] = 304.120000;
  ai[0] = 1e6 * 0.42748 * Rcst * Rcst * Tci[0] * Tci[0] /
          (44.009950 * 44.009950 * 73.740000);
  bi[0] = 0.08664 * Rcst * Tci[0] / (44.009950 * 73.740000);
  acentric_i[0] = 0.225000;

  // species 1: N2
  // Imported from NIST
  Tci[1] = 126.192000;
  ai[1] = 1e6 * 0.42748 * Rcst * Rcst * Tci[1] * Tci[1] /
          (28.013400 * 28.013400 * 33.958000);
  bi[1] = 0.08664 * Rcst * Tci[1] / (28.013400 * 33.958000);
  acentric_i[1] = 0.037200;

  // species 2: AR
  // Imported from NIST
  Tci[2] = 150.860000;
  ai[2] = 1e6 * 0.42748 * Rcst * Rcst * Tci[2] * Tci[2] /
          (39.948000 * 39.948000 * 48.980000);
  bi[2] = 0.08664 * Rcst * Tci[2] / (39.948000 * 48.980000);
  acentric_i[2] = -0.002000;

  // species 3: H2O
  // Imported from NIST
  Tci[3] = 647.096000;
  ai[3] = 1e6 * 0.42748 * Rcst * Rcst * Tci[3] * Tci[3] /
          (18.015340 * 18.015340 * 220.640000);
  bi[3] = 0.08664 * Rcst * Tci[3] / (18.015340 * 220.640000);
  acentric_i[3] = 0.344300;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: CO2
  // Imported from NIST
  sqrtOneOverTc[0] = 5.7342616962522e-02;
  sqrtAsti[0] = 4.3745610142919e+04;
  Bi[0] = 6.7506167841745e-01;
  Fomega[0] = 8.2653709187500e-01;

  // species 1: N2
  // Imported from NIST
  sqrtOneOverTc[1] = 8.9019282240563e-02;
  sqrtAsti[1] = 4.2022983929562e+04;
  Bi[1] = 9.5560052221563e-01;
  Fomega[1] = 5.4259343186608e-01;

  // species 2: AR
  // Imported from NIST
  sqrtOneOverTc[2] = 8.1416597423011e-02;
  sqrtAsti[2] = 2.9333346340060e+04;
  Bi[2] = 5.5540880812616e-01;
  Fomega[2] = 4.8197599354800e-01;

  // species 3: H2O
  // Imported from NIST
  sqrtOneOverTc[3] = 3.9311140369715e-02;
  sqrtAsti[3] = 1.3145468419592e+05;
  Bi[3] = 1.1727204136223e+00;
  Fomega[3] = 1.0013577274636e+00;
}

// gauss-jordan solver external routine
// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort("sgjsolve not implemented, choose a different solver ");
}

// Replace this routine with the one generated by the Gauss Jordan solver of DW
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
sgjsolve_simplified(amrex::Real* /*A*/, amrex::Real* /*x*/, amrex::Real* /*b*/)
{
  amrex::Abort(
    "sgjsolve_simplified not implemented, choose a different solver ");
}

#endif
