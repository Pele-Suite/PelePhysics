#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  O
1  H
2  C
3  N
*/

// Species
#define O2_ID 0
#define H2O_ID 1
#define CH4_ID 2
#define CO_ID 3
#define CO2_ID 4
#define N2_ID 5

#define NUM_GAS_ELEMENTS 4  // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 6   // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 3 // Reactions in the homogeneous phase

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
  mm = 4;
  kk = 6;
  ii = 3;
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
  int kd = 4;
  // Zero ncf
  for (int id = 0; id < kd * 6; ++id) {
    ncf[id] = 0;
  }

  // O2
  ncf[0 * kd + 0] = 2; // O

  // H2O
  ncf[1 * kd + 1] = 2; // H
  ncf[1 * kd + 0] = 1; // O

  // CH4
  ncf[2 * kd + 2] = 1; // C
  ncf[2 * kd + 1] = 4; // H

  // CO
  ncf[3 * kd + 2] = 1; // C
  ncf[3 * kd + 0] = 1; // O

  // CO2
  ncf[4 * kd + 2] = 1; // C
  ncf[4 * kd + 0] = 2; // O

  // N2
  ncf[5 * kd + 3] = 2; // N
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[6] = {
  3.1251953247077940e-02, // O2
  5.5509297807382736e-02, // H2O
  6.2332481456086768e-02, // CH4
  3.5701535166012141e-02, // CO
  2.2722624917630486e-02, // CO2
  3.5696437495537948e-02, // N2
};
#endif
const amrex::Real h_global_imw[6] = {
  3.1251953247077940e-02, // O2
  5.5509297807382736e-02, // H2O
  6.2332481456086768e-02, // CH4
  3.5701535166012141e-02, // CO
  2.2722624917630486e-02, // CO2
  3.5696437495537948e-02, // N2
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[6] = {
  3.1998000000000001e+01, // O2
  1.8015000000000001e+01, // H2O
  1.6042999999999999e+01, // CH4
  2.8009999999999998e+01, // CO
  4.4009000000000000e+01, // CO2
  2.8013999999999999e+01, // N2
};
#endif
const amrex::Real h_global_mw[6] = {
  3.1998000000000001e+01, // O2
  1.8015000000000001e+01, // H2O
  1.6042999999999999e+01, // CH4
  2.8009999999999998e+01, // CO
  4.4009000000000000e+01, // CO2
  2.8013999999999999e+01, // N2
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 3.1251953247077940e-02; // O2
  imw_new[1] = 5.5509297807382736e-02; // H2O
  imw_new[2] = 6.2332481456086768e-02; // CH4
  imw_new[3] = 3.5701535166012141e-02; // CO
  imw_new[4] = 2.2722624917630486e-02; // CO2
  imw_new[5] = 3.5696437495537948e-02; // N2
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
  mw_new[0] = 3.1998000000000001e+01; // O2
  mw_new[1] = 1.8015000000000001e+01; // H2O
  mw_new[2] = 1.6042999999999999e+01; // CH4
  mw_new[3] = 2.8009999999999998e+01; // CO
  mw_new[4] = 4.4009000000000000e+01; // CO2
  mw_new[5] = 2.8013999999999999e+01; // N2
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +2.782456360000000e+00 - 2.996734160000000e-03 * T +
                 9.847302010000000e-06 * T2 - 9.681295090000001e-09 * T3 +
                 3.243728370000000e-12 * T4;
    // species 1: H2O
    species[1] = +3.198640560000000e+00 - 2.036434100000000e-03 * T +
                 6.520402110000000e-06 * T2 - 5.487970620000000e-09 * T3 +
                 1.771978170000000e-12 * T4;
    // species 2: CH4
    species[2] = +4.149876130000000e+00 - 1.367097880000000e-02 * T +
                 4.918005990000000e-05 * T2 - 4.847430260000000e-08 * T3 +
                 1.666939560000000e-11 * T4;
    // species 3: CO
    species[3] = +2.579533470000000e+00 - 6.103536800000000e-04 * T +
                 1.016814330000000e-06 * T2 + 9.070058840000000e-10 * T3 -
                 9.044244990000000e-13 * T4;
    // species 4: CO2
    species[4] = +1.356773520000000e+00 + 8.984596770000000e-03 * T -
                 7.123562690000000e-06 * T2 + 2.459190220000000e-09 * T3 -
                 1.436995480000000e-13 * T4;
    // species 5: N2
    species[5] = +2.298677000000000e+00 + 1.408240400000000e-03 * T -
                 3.963222000000000e-06 * T2 + 5.641515000000000e-09 * T3 -
                 2.444854000000000e-12 * T4;
  } else {
    // species 0: O2
    species[0] = +2.282537840000000e+00 + 1.483087540000000e-03 * T -
                 7.579666690000000e-07 * T2 + 2.094705550000000e-10 * T3 -
                 2.167177940000000e-14 * T4;
    // species 1: H2O
    species[1] = +2.033992490000000e+00 + 2.176918040000000e-03 * T -
                 1.640725180000000e-07 * T2 - 9.704198700000000e-11 * T3 +
                 1.682009920000000e-14 * T4;
    // species 2: CH4
    species[2] = -9.251485050000000e-01 + 1.339094670000000e-02 * T -
                 5.732858090000000e-06 * T2 + 1.222925350000000e-09 * T3 -
                 1.018152300000000e-13 * T4;
    // species 3: CO
    species[3] = +1.715185610000000e+00 + 2.062527430000000e-03 * T -
                 9.988257710000001e-07 * T2 + 2.300530080000000e-10 * T3 -
                 2.036477160000000e-14 * T4;
    // species 4: CO2
    species[4] = +2.857460290000000e+00 + 4.414370260000000e-03 * T -
                 2.214814040000000e-06 * T2 + 5.234901880000001e-10 * T3 -
                 4.720841640000000e-14 * T4;
    // species 5: N2
    species[5] = +1.926640000000000e+00 + 1.487976800000000e-03 * T -
                 5.684760000000000e-07 * T2 + 1.009703800000000e-10 * T3 -
                 6.753351000000000e-15 * T4;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +3.782456360000000e+00 - 2.996734160000000e-03 * T +
                 9.847302010000000e-06 * T2 - 9.681295090000001e-09 * T3 +
                 3.243728370000000e-12 * T4;
    // species 1: H2O
    species[1] = +4.198640560000000e+00 - 2.036434100000000e-03 * T +
                 6.520402110000000e-06 * T2 - 5.487970620000000e-09 * T3 +
                 1.771978170000000e-12 * T4;
    // species 2: CH4
    species[2] = +5.149876130000000e+00 - 1.367097880000000e-02 * T +
                 4.918005990000000e-05 * T2 - 4.847430260000000e-08 * T3 +
                 1.666939560000000e-11 * T4;
    // species 3: CO
    species[3] = +3.579533470000000e+00 - 6.103536800000000e-04 * T +
                 1.016814330000000e-06 * T2 + 9.070058840000000e-10 * T3 -
                 9.044244990000000e-13 * T4;
    // species 4: CO2
    species[4] = +2.356773520000000e+00 + 8.984596770000000e-03 * T -
                 7.123562690000000e-06 * T2 + 2.459190220000000e-09 * T3 -
                 1.436995480000000e-13 * T4;
    // species 5: N2
    species[5] = +3.298677000000000e+00 + 1.408240400000000e-03 * T -
                 3.963222000000000e-06 * T2 + 5.641515000000000e-09 * T3 -
                 2.444854000000000e-12 * T4;
  } else {
    // species 0: O2
    species[0] = +3.282537840000000e+00 + 1.483087540000000e-03 * T -
                 7.579666690000000e-07 * T2 + 2.094705550000000e-10 * T3 -
                 2.167177940000000e-14 * T4;
    // species 1: H2O
    species[1] = +3.033992490000000e+00 + 2.176918040000000e-03 * T -
                 1.640725180000000e-07 * T2 - 9.704198700000000e-11 * T3 +
                 1.682009920000000e-14 * T4;
    // species 2: CH4
    species[2] = +7.485149500000000e-02 + 1.339094670000000e-02 * T -
                 5.732858090000000e-06 * T2 + 1.222925350000000e-09 * T3 -
                 1.018152300000000e-13 * T4;
    // species 3: CO
    species[3] = +2.715185610000000e+00 + 2.062527430000000e-03 * T -
                 9.988257710000001e-07 * T2 + 2.300530080000000e-10 * T3 -
                 2.036477160000000e-14 * T4;
    // species 4: CO2
    species[4] = +3.857460290000000e+00 + 4.414370260000000e-03 * T -
                 2.214814040000000e-06 * T2 + 5.234901880000001e-10 * T3 -
                 4.720841640000000e-14 * T4;
    // species 5: N2
    species[5] = +2.926640000000000e+00 + 1.487976800000000e-03 * T -
                 5.684760000000000e-07 * T2 + 1.009703800000000e-10 * T3 -
                 6.753351000000000e-15 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = -1.063943560000000e+03 * invT + 1.247806300000001e-01 -
                 3.782456360000000e+00 * logT + 1.498367080000000e-03 * T -
                 1.641217001666667e-06 * T2 + 8.067745908333334e-10 * T3 -
                 1.621864185000000e-13 * T4;
    // species 1: H2O
    species[1] = -3.029372670000000e+04 * invT + 5.047672768000000e+00 -
                 4.198640560000000e+00 * logT + 1.018217050000000e-03 * T -
                 1.086733685000000e-06 * T2 + 4.573308850000000e-10 * T3 -
                 8.859890850000000e-14 * T4;
    // species 2: CH4
    species[2] = -1.024664760000000e+04 * invT + 9.791179889999999e+00 -
                 5.149876130000000e+00 * logT + 6.835489400000000e-03 * T -
                 8.196676650000000e-06 * T2 + 4.039525216666667e-09 * T3 -
                 8.334697800000000e-13 * T4;
    // species 3: CO
    species[3] = -1.434408600000000e+04 * invT + 7.112418999999992e-02 -
                 3.579533470000000e+00 * logT + 3.051768400000000e-04 * T -
                 1.694690550000000e-07 * T2 - 7.558382366666667e-11 * T3 +
                 4.522122495000000e-14 * T4;
    // species 4: CO2
    species[4] = -4.837196970000000e+04 * invT - 7.544278700000000e+00 -
                 2.356773520000000e+00 * logT - 4.492298385000000e-03 * T +
                 1.187260448333333e-06 * T2 - 2.049325183333333e-10 * T3 +
                 7.184977399999999e-15 * T4;
    // species 5: N2
    species[5] = -1.020899900000000e+03 * invT - 6.516950000000001e-01 -
                 3.298677000000000e+00 * logT - 7.041202000000000e-04 * T +
                 6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                 1.222427000000000e-13 * T4;
  } else {
    // species 0: O2
    species[0] = -1.088457720000000e+03 * invT - 2.170693450000000e+00 -
                 3.282537840000000e+00 * logT - 7.415437700000000e-04 * T +
                 1.263277781666667e-07 * T2 - 1.745587958333333e-11 * T3 +
                 1.083588970000000e-15 * T4;
    // species 1: H2O
    species[1] = -3.000429710000000e+04 * invT - 1.932777610000000e+00 -
                 3.033992490000000e+00 * logT - 1.088459020000000e-03 * T +
                 2.734541966666666e-08 * T2 + 8.086832250000000e-12 * T3 -
                 8.410049600000000e-16 * T4;
    // species 2: CH4
    species[2] = -9.468344590000001e+03 * invT - 1.836246650500000e+01 -
                 7.485149500000000e-02 * logT - 6.695473350000000e-03 * T +
                 9.554763483333333e-07 * T2 - 1.019104458333333e-10 * T3 +
                 5.090761500000000e-15 * T4;
    // species 3: CO
    species[3] = -1.415187240000000e+04 * invT - 5.103502110000000e+00 -
                 2.715185610000000e+00 * logT - 1.031263715000000e-03 * T +
                 1.664709618333334e-07 * T2 - 1.917108400000000e-11 * T3 +
                 1.018238580000000e-15 * T4;
    // species 4: CO2
    species[4] = -4.875916600000000e+04 * invT + 1.585822230000000e+00 -
                 3.857460290000000e+00 * logT - 2.207185130000000e-03 * T +
                 3.691356733333334e-07 * T2 - 4.362418233333334e-11 * T3 +
                 2.360420820000000e-15 * T4;
    // species 5: N2
    species[5] = -9.227977000000000e+02 * invT - 3.053888000000000e+00 -
                 2.926640000000000e+00 * logT - 7.439884000000000e-04 * T +
                 9.474600000000001e-08 * T2 - 8.414198333333333e-12 * T3 +
                 3.376675500000000e-16 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = -1.063943560000000e+03 * invT - 8.752193699999999e-01 -
                 3.782456360000000e+00 * logT + 1.498367080000000e-03 * T -
                 1.641217001666667e-06 * T2 + 8.067745908333334e-10 * T3 -
                 1.621864185000000e-13 * T4;
    // species 1: H2O
    species[1] = -3.029372670000000e+04 * invT + 4.047672768000000e+00 -
                 4.198640560000000e+00 * logT + 1.018217050000000e-03 * T -
                 1.086733685000000e-06 * T2 + 4.573308850000000e-10 * T3 -
                 8.859890850000000e-14 * T4;
    // species 2: CH4
    species[2] = -1.024664760000000e+04 * invT + 8.791179889999999e+00 -
                 5.149876130000000e+00 * logT + 6.835489400000000e-03 * T -
                 8.196676650000000e-06 * T2 + 4.039525216666667e-09 * T3 -
                 8.334697800000000e-13 * T4;
    // species 3: CO
    species[3] = -1.434408600000000e+04 * invT - 9.288758100000001e-01 -
                 3.579533470000000e+00 * logT + 3.051768400000000e-04 * T -
                 1.694690550000000e-07 * T2 - 7.558382366666667e-11 * T3 +
                 4.522122495000000e-14 * T4;
    // species 4: CO2
    species[4] = -4.837196970000000e+04 * invT - 8.544278700000000e+00 -
                 2.356773520000000e+00 * logT - 4.492298385000000e-03 * T +
                 1.187260448333333e-06 * T2 - 2.049325183333333e-10 * T3 +
                 7.184977399999999e-15 * T4;
    // species 5: N2
    species[5] = -1.020899900000000e+03 * invT - 1.651695000000000e+00 -
                 3.298677000000000e+00 * logT - 7.041202000000000e-04 * T +
                 6.605369999999999e-07 * T2 - 4.701262500000001e-10 * T3 +
                 1.222427000000000e-13 * T4;
  } else {
    // species 0: O2
    species[0] = -1.088457720000000e+03 * invT - 3.170693450000000e+00 -
                 3.282537840000000e+00 * logT - 7.415437700000000e-04 * T +
                 1.263277781666667e-07 * T2 - 1.745587958333333e-11 * T3 +
                 1.083588970000000e-15 * T4;
    // species 1: H2O
    species[1] = -3.000429710000000e+04 * invT - 2.932777610000000e+00 -
                 3.033992490000000e+00 * logT - 1.088459020000000e-03 * T +
                 2.734541966666666e-08 * T2 + 8.086832250000000e-12 * T3 -
                 8.410049600000000e-16 * T4;
    // species 2: CH4
    species[2] = -9.468344590000001e+03 * invT - 1.936246650500000e+01 -
                 7.485149500000000e-02 * logT - 6.695473350000000e-03 * T +
                 9.554763483333333e-07 * T2 - 1.019104458333333e-10 * T3 +
                 5.090761500000000e-15 * T4;
    // species 3: CO
    species[3] = -1.415187240000000e+04 * invT - 6.103502110000000e+00 -
                 2.715185610000000e+00 * logT - 1.031263715000000e-03 * T +
                 1.664709618333334e-07 * T2 - 1.917108400000000e-11 * T3 +
                 1.018238580000000e-15 * T4;
    // species 4: CO2
    species[4] = -4.875916600000000e+04 * invT + 5.858222300000002e-01 -
                 3.857460290000000e+00 * logT - 2.207185130000000e-03 * T +
                 3.691356733333334e-07 * T2 - 4.362418233333334e-11 * T3 +
                 2.360420820000000e-15 * T4;
    // species 5: N2
    species[5] = -9.227977000000000e+02 * invT - 4.053888000000000e+00 -
                 2.926640000000000e+00 * logT - 7.439884000000000e-04 * T +
                 9.474600000000001e-08 * T2 - 8.414198333333333e-12 * T3 +
                 3.376675500000000e-16 * T4;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +2.782456360000000e+00 - 1.498367080000000e-03 * T +
                 3.282434003333333e-06 * T2 - 2.420323772500000e-09 * T3 +
                 6.487456740000000e-13 * T4 - 1.063943560000000e+03 * invT;
    // species 1: H2O
    species[1] = +3.198640560000000e+00 - 1.018217050000000e-03 * T +
                 2.173467370000000e-06 * T2 - 1.371992655000000e-09 * T3 +
                 3.543956340000000e-13 * T4 - 3.029372670000000e+04 * invT;
    // species 2: CH4
    species[2] = +4.149876130000000e+00 - 6.835489400000000e-03 * T +
                 1.639335330000000e-05 * T2 - 1.211857565000000e-08 * T3 +
                 3.333879120000000e-12 * T4 - 1.024664760000000e+04 * invT;
    // species 3: CO
    species[3] = +2.579533470000000e+00 - 3.051768400000000e-04 * T +
                 3.389381100000000e-07 * T2 + 2.267514710000000e-10 * T3 -
                 1.808848998000000e-13 * T4 - 1.434408600000000e+04 * invT;
    // species 4: CO2
    species[4] = +1.356773520000000e+00 + 4.492298385000000e-03 * T -
                 2.374520896666666e-06 * T2 + 6.147975550000000e-10 * T3 -
                 2.873990960000000e-14 * T4 - 4.837196970000000e+04 * invT;
    // species 5: N2
    species[5] = +2.298677000000000e+00 + 7.041202000000000e-04 * T -
                 1.321074000000000e-06 * T2 + 1.410378750000000e-09 * T3 -
                 4.889707999999999e-13 * T4 - 1.020899900000000e+03 * invT;
  } else {
    // species 0: O2
    species[0] = +2.282537840000000e+00 + 7.415437700000000e-04 * T -
                 2.526555563333333e-07 * T2 + 5.236763875000000e-11 * T3 -
                 4.334355880000000e-15 * T4 - 1.088457720000000e+03 * invT;
    // species 1: H2O
    species[1] = +2.033992490000000e+00 + 1.088459020000000e-03 * T -
                 5.469083933333333e-08 * T2 - 2.426049675000000e-11 * T3 +
                 3.364019840000000e-15 * T4 - 3.000429710000000e+04 * invT;
    // species 2: CH4
    species[2] = -9.251485050000000e-01 + 6.695473350000000e-03 * T -
                 1.910952696666667e-06 * T2 + 3.057313375000000e-10 * T3 -
                 2.036304600000000e-14 * T4 - 9.468344590000001e+03 * invT;
    // species 3: CO
    species[3] = +1.715185610000000e+00 + 1.031263715000000e-03 * T -
                 3.329419236666667e-07 * T2 + 5.751325200000000e-11 * T3 -
                 4.072954320000000e-15 * T4 - 1.415187240000000e+04 * invT;
    // species 4: CO2
    species[4] = +2.857460290000000e+00 + 2.207185130000000e-03 * T -
                 7.382713466666667e-07 * T2 + 1.308725470000000e-10 * T3 -
                 9.441683280000000e-15 * T4 - 4.875916600000000e+04 * invT;
    // species 5: N2
    species[5] = +1.926640000000000e+00 + 7.439884000000000e-04 * T -
                 1.894920000000000e-07 * T2 + 2.524259500000000e-11 * T3 -
                 1.350670200000000e-15 * T4 - 9.227977000000000e+02 * invT;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +3.782456360000000e+00 - 1.498367080000000e-03 * T +
                 3.282434003333333e-06 * T2 - 2.420323772500000e-09 * T3 +
                 6.487456740000000e-13 * T4 - 1.063943560000000e+03 * invT;
    // species 1: H2O
    species[1] = +4.198640560000000e+00 - 1.018217050000000e-03 * T +
                 2.173467370000000e-06 * T2 - 1.371992655000000e-09 * T3 +
                 3.543956340000000e-13 * T4 - 3.029372670000000e+04 * invT;
    // species 2: CH4
    species[2] = +5.149876130000000e+00 - 6.835489400000000e-03 * T +
                 1.639335330000000e-05 * T2 - 1.211857565000000e-08 * T3 +
                 3.333879120000000e-12 * T4 - 1.024664760000000e+04 * invT;
    // species 3: CO
    species[3] = +3.579533470000000e+00 - 3.051768400000000e-04 * T +
                 3.389381100000000e-07 * T2 + 2.267514710000000e-10 * T3 -
                 1.808848998000000e-13 * T4 - 1.434408600000000e+04 * invT;
    // species 4: CO2
    species[4] = +2.356773520000000e+00 + 4.492298385000000e-03 * T -
                 2.374520896666666e-06 * T2 + 6.147975550000000e-10 * T3 -
                 2.873990960000000e-14 * T4 - 4.837196970000000e+04 * invT;
    // species 5: N2
    species[5] = +3.298677000000000e+00 + 7.041202000000000e-04 * T -
                 1.321074000000000e-06 * T2 + 1.410378750000000e-09 * T3 -
                 4.889707999999999e-13 * T4 - 1.020899900000000e+03 * invT;
  } else {
    // species 0: O2
    species[0] = +3.282537840000000e+00 + 7.415437700000000e-04 * T -
                 2.526555563333333e-07 * T2 + 5.236763875000000e-11 * T3 -
                 4.334355880000000e-15 * T4 - 1.088457720000000e+03 * invT;
    // species 1: H2O
    species[1] = +3.033992490000000e+00 + 1.088459020000000e-03 * T -
                 5.469083933333333e-08 * T2 - 2.426049675000000e-11 * T3 +
                 3.364019840000000e-15 * T4 - 3.000429710000000e+04 * invT;
    // species 2: CH4
    species[2] = +7.485149500000000e-02 + 6.695473350000000e-03 * T -
                 1.910952696666667e-06 * T2 + 3.057313375000000e-10 * T3 -
                 2.036304600000000e-14 * T4 - 9.468344590000001e+03 * invT;
    // species 3: CO
    species[3] = +2.715185610000000e+00 + 1.031263715000000e-03 * T -
                 3.329419236666667e-07 * T2 + 5.751325200000000e-11 * T3 -
                 4.072954320000000e-15 * T4 - 1.415187240000000e+04 * invT;
    // species 4: CO2
    species[4] = +3.857460290000000e+00 + 2.207185130000000e-03 * T -
                 7.382713466666667e-07 * T2 + 1.308725470000000e-10 * T3 -
                 9.441683280000000e-15 * T4 - 4.875916600000000e+04 * invT;
    // species 5: N2
    species[5] = +2.926640000000000e+00 + 7.439884000000000e-04 * T -
                 1.894920000000000e-07 * T2 + 2.524259500000000e-11 * T3 -
                 1.350670200000000e-15 * T4 - 9.227977000000000e+02 * invT;
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = +3.782456360000000e+00 * logT - 2.996734160000000e-03 * T +
                 4.923651005000000e-06 * T2 - 3.227098363333333e-09 * T3 +
                 8.109320925000000e-13 * T4 + 3.657675730000000e+00;
    // species 1: H2O
    species[1] = +4.198640560000000e+00 * logT - 2.036434100000000e-03 * T +
                 3.260201055000000e-06 * T2 - 1.829323540000000e-09 * T3 +
                 4.429945425000000e-13 * T4 - 8.490322080000000e-01;
    // species 2: CH4
    species[2] = +5.149876130000000e+00 * logT - 1.367097880000000e-02 * T +
                 2.459002995000000e-05 * T2 - 1.615810086666667e-08 * T3 +
                 4.167348900000000e-12 * T4 - 4.641303760000000e+00;
    // species 3: CO
    species[3] = +3.579533470000000e+00 * logT - 6.103536800000000e-04 * T +
                 5.084071650000000e-07 * T2 + 3.023352946666667e-10 * T3 -
                 2.261061247500000e-13 * T4 + 3.508409280000000e+00;
    // species 4: CO2
    species[4] = +2.356773520000000e+00 * logT + 8.984596770000000e-03 * T -
                 3.561781345000000e-06 * T2 + 8.197300733333333e-10 * T3 -
                 3.592488700000000e-14 * T4 + 9.901052220000000e+00;
    // species 5: N2
    species[5] = +3.298677000000000e+00 * logT + 1.408240400000000e-03 * T -
                 1.981611000000000e-06 * T2 + 1.880505000000000e-09 * T3 -
                 6.112135000000000e-13 * T4 + 3.950372000000000e+00;
  } else {
    // species 0: O2
    species[0] = +3.282537840000000e+00 * logT + 1.483087540000000e-03 * T -
                 3.789833345000000e-07 * T2 + 6.982351833333333e-11 * T3 -
                 5.417944850000000e-15 * T4 + 5.453231290000000e+00;
    // species 1: H2O
    species[1] = +3.033992490000000e+00 * logT + 2.176918040000000e-03 * T -
                 8.203625899999999e-08 * T2 - 3.234732900000000e-11 * T3 +
                 4.205024800000000e-15 * T4 + 4.966770100000000e+00;
    // species 2: CH4
    species[2] = +7.485149500000000e-02 * logT + 1.339094670000000e-02 * T -
                 2.866429045000000e-06 * T2 + 4.076417833333333e-10 * T3 -
                 2.545380750000000e-14 * T4 + 1.843731800000000e+01;
    // species 3: CO
    species[3] = +2.715185610000000e+00 * logT + 2.062527430000000e-03 * T -
                 4.994128855000000e-07 * T2 + 7.668433600000000e-11 * T3 -
                 5.091192900000000e-15 * T4 + 7.818687720000000e+00;
    // species 4: CO2
    species[4] = +3.857460290000000e+00 * logT + 4.414370260000000e-03 * T -
                 1.107407020000000e-06 * T2 + 1.744967293333334e-10 * T3 -
                 1.180210410000000e-14 * T4 + 2.271638060000000e+00;
    // species 5: N2
    species[5] = +2.926640000000000e+00 * logT + 1.487976800000000e-03 * T -
                 2.842380000000000e-07 * T2 + 3.365679333333333e-11 * T3 -
                 1.688337750000000e-15 * T4 + 5.980528000000000e+00;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    species[0] = -2.996734160000000e-03 + 1.969460402000000e-05 * T -
                 2.904388527000000e-08 * T2 + 1.297491348000000e-11 * T3;
    // species 1: H2O
    species[1] = -2.036434100000000e-03 + 1.304080422000000e-05 * T -
                 1.646391186000000e-08 * T2 + 7.087912680000000e-12 * T3;
    // species 2: CH4
    species[2] = -1.367097880000000e-02 + 9.836011980000000e-05 * T -
                 1.454229078000000e-07 * T2 + 6.667758240000000e-11 * T3;
    // species 3: CO
    species[3] = -6.103536800000000e-04 + 2.033628660000000e-06 * T +
                 2.721017652000000e-09 * T2 - 3.617697996000000e-12 * T3;
    // species 4: CO2
    species[4] = +8.984596770000000e-03 - 1.424712538000000e-05 * T +
                 7.377570659999999e-09 * T2 - 5.747981920000000e-13 * T3;
    // species 5: N2
    species[5] = +1.408240400000000e-03 - 7.926444000000000e-06 * T +
                 1.692454500000000e-08 * T2 - 9.779416000000000e-12 * T3;
  } else {
    // species 0: O2
    species[0] = +1.483087540000000e-03 - 1.515933338000000e-06 * T +
                 6.284116650000000e-10 * T2 - 8.668711760000000e-14 * T3;
    // species 1: H2O
    species[1] = +2.176918040000000e-03 - 3.281450360000000e-07 * T -
                 2.911259610000000e-10 * T2 + 6.728039680000000e-14 * T3;
    // species 2: CH4
    species[2] = +1.339094670000000e-02 - 1.146571618000000e-05 * T +
                 3.668776050000000e-09 * T2 - 4.072609200000000e-13 * T3;
    // species 3: CO
    species[3] = +2.062527430000000e-03 - 1.997651542000000e-06 * T +
                 6.901590240000000e-10 * T2 - 8.145908640000000e-14 * T3;
    // species 4: CO2
    species[4] = +4.414370260000000e-03 - 4.429628080000000e-06 * T +
                 1.570470564000000e-09 * T2 - 1.888336656000000e-13 * T3;
    // species 5: N2
    species[5] = +1.487976800000000e-03 - 1.136952000000000e-06 * T +
                 3.029111400000000e-10 * T2 - 2.701340400000000e-14 * T3;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[6]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 6; ++id) {
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    result += y[0] *
              (+3.782456360000000e+00 - 2.996734160000000e-03 * T +
               9.847302010000000e-06 * T2 - 9.681295090000001e-09 * T3 +
               3.243728370000000e-12 * T4) *
              0.0312519532470779;
    // species 1: H2O
    result += y[1] *
              (+4.198640560000000e+00 - 2.036434100000000e-03 * T +
               6.520402110000000e-06 * T2 - 5.487970620000000e-09 * T3 +
               1.771978170000000e-12 * T4) *
              0.0555092978073827;
    // species 2: CH4
    result += y[2] *
              (+5.149876130000000e+00 - 1.367097880000000e-02 * T +
               4.918005990000000e-05 * T2 - 4.847430260000000e-08 * T3 +
               1.666939560000000e-11 * T4) *
              0.0623324814560868;
    // species 3: CO
    result += y[3] *
              (+3.579533470000000e+00 - 6.103536800000000e-04 * T +
               1.016814330000000e-06 * T2 + 9.070058840000000e-10 * T3 -
               9.044244990000000e-13 * T4) *
              0.0357015351660121;
    // species 4: CO2
    result += y[4] *
              (+2.356773520000000e+00 + 8.984596770000000e-03 * T -
               7.123562690000000e-06 * T2 + 2.459190220000000e-09 * T3 -
               1.436995480000000e-13 * T4) *
              0.0227226249176305;
    // species 5: N2
    result += y[5] *
              (+3.298677000000000e+00 + 1.408240400000000e-03 * T -
               3.963222000000000e-06 * T2 + 5.641515000000000e-09 * T3 -
               2.444854000000000e-12 * T4) *
              0.0356964374955379;
  } else {
    // species 0: O2
    result += y[0] *
              (+3.282537840000000e+00 + 1.483087540000000e-03 * T -
               7.579666690000000e-07 * T2 + 2.094705550000000e-10 * T3 -
               2.167177940000000e-14 * T4) *
              0.0312519532470779;
    // species 1: H2O
    result += y[1] *
              (+3.033992490000000e+00 + 2.176918040000000e-03 * T -
               1.640725180000000e-07 * T2 - 9.704198700000000e-11 * T3 +
               1.682009920000000e-14 * T4) *
              0.0555092978073827;
    // species 2: CH4
    result += y[2] *
              (+7.485149500000000e-02 + 1.339094670000000e-02 * T -
               5.732858090000000e-06 * T2 + 1.222925350000000e-09 * T3 -
               1.018152300000000e-13 * T4) *
              0.0623324814560868;
    // species 3: CO
    result += y[3] *
              (+2.715185610000000e+00 + 2.062527430000000e-03 * T -
               9.988257710000001e-07 * T2 + 2.300530080000000e-10 * T3 -
               2.036477160000000e-14 * T4) *
              0.0357015351660121;
    // species 4: CO2
    result += y[4] *
              (+3.857460290000000e+00 + 4.414370260000000e-03 * T -
               2.214814040000000e-06 * T2 + 5.234901880000001e-10 * T3 -
               4.720841640000000e-14 * T4) *
              0.0227226249176305;
    // species 5: N2
    result += y[5] *
              (+2.926640000000000e+00 + 1.487976800000000e-03 * T -
               5.684760000000000e-07 * T2 + 1.009703800000000e-10 * T3 -
               6.753351000000000e-15 * T4) *
              0.0356964374955379;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[6]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 6; ++id) {
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    result += y[0] *
              (+2.782456360000000e+00 - 2.996734160000000e-03 * T +
               9.847302010000000e-06 * T2 - 9.681295090000001e-09 * T3 +
               3.243728370000000e-12 * T4) *
              0.0312519532470779;
    // species 1: H2O
    result += y[1] *
              (+3.198640560000000e+00 - 2.036434100000000e-03 * T +
               6.520402110000000e-06 * T2 - 5.487970620000000e-09 * T3 +
               1.771978170000000e-12 * T4) *
              0.0555092978073827;
    // species 2: CH4
    result += y[2] *
              (+4.149876130000000e+00 - 1.367097880000000e-02 * T +
               4.918005990000000e-05 * T2 - 4.847430260000000e-08 * T3 +
               1.666939560000000e-11 * T4) *
              0.0623324814560868;
    // species 3: CO
    result += y[3] *
              (+2.579533470000000e+00 - 6.103536800000000e-04 * T +
               1.016814330000000e-06 * T2 + 9.070058840000000e-10 * T3 -
               9.044244990000000e-13 * T4) *
              0.0357015351660121;
    // species 4: CO2
    result += y[4] *
              (+1.356773520000000e+00 + 8.984596770000000e-03 * T -
               7.123562690000000e-06 * T2 + 2.459190220000000e-09 * T3 -
               1.436995480000000e-13 * T4) *
              0.0227226249176305;
    // species 5: N2
    result += y[5] *
              (+2.298677000000000e+00 + 1.408240400000000e-03 * T -
               3.963222000000000e-06 * T2 + 5.641515000000000e-09 * T3 -
               2.444854000000000e-12 * T4) *
              0.0356964374955379;
  } else {
    // species 0: O2
    result += y[0] *
              (+2.282537840000000e+00 + 1.483087540000000e-03 * T -
               7.579666690000000e-07 * T2 + 2.094705550000000e-10 * T3 -
               2.167177940000000e-14 * T4) *
              0.0312519532470779;
    // species 1: H2O
    result += y[1] *
              (+2.033992490000000e+00 + 2.176918040000000e-03 * T -
               1.640725180000000e-07 * T2 - 9.704198700000000e-11 * T3 +
               1.682009920000000e-14 * T4) *
              0.0555092978073827;
    // species 2: CH4
    result += y[2] *
              (-9.251485050000000e-01 + 1.339094670000000e-02 * T -
               5.732858090000000e-06 * T2 + 1.222925350000000e-09 * T3 -
               1.018152300000000e-13 * T4) *
              0.0623324814560868;
    // species 3: CO
    result += y[3] *
              (+1.715185610000000e+00 + 2.062527430000000e-03 * T -
               9.988257710000001e-07 * T2 + 2.300530080000000e-10 * T3 -
               2.036477160000000e-14 * T4) *
              0.0357015351660121;
    // species 4: CO2
    result += y[4] *
              (+2.857460290000000e+00 + 4.414370260000000e-03 * T -
               2.214814040000000e-06 * T2 + 5.234901880000001e-10 * T3 -
               4.720841640000000e-14 * T4) *
              0.0227226249176305;
    // species 5: N2
    result += y[5] *
              (+1.926640000000000e+00 + 1.487976800000000e-03 * T -
               5.684760000000000e-07 * T2 + 1.009703800000000e-10 * T3 -
               6.753351000000000e-15 * T4) *
              0.0356964374955379;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[6];                        //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 6; ++id) {
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    result += y[0] *
              (+3.782456360000000e+00 - 1.498367080000000e-03 * T +
               3.282434003333333e-06 * T2 - 2.420323772500000e-09 * T3 +
               6.487456740000000e-13 * T4 - 1.063943560000000e+03 * invT) *
              0.0312519532470779;
    // species 1: H2O
    result += y[1] *
              (+4.198640560000000e+00 - 1.018217050000000e-03 * T +
               2.173467370000000e-06 * T2 - 1.371992655000000e-09 * T3 +
               3.543956340000000e-13 * T4 - 3.029372670000000e+04 * invT) *
              0.0555092978073827;
    // species 2: CH4
    result += y[2] *
              (+5.149876130000000e+00 - 6.835489400000000e-03 * T +
               1.639335330000000e-05 * T2 - 1.211857565000000e-08 * T3 +
               3.333879120000000e-12 * T4 - 1.024664760000000e+04 * invT) *
              0.0623324814560868;
    // species 3: CO
    result += y[3] *
              (+3.579533470000000e+00 - 3.051768400000000e-04 * T +
               3.389381100000000e-07 * T2 + 2.267514710000000e-10 * T3 -
               1.808848998000000e-13 * T4 - 1.434408600000000e+04 * invT) *
              0.0357015351660121;
    // species 4: CO2
    result += y[4] *
              (+2.356773520000000e+00 + 4.492298385000000e-03 * T -
               2.374520896666666e-06 * T2 + 6.147975550000000e-10 * T3 -
               2.873990960000000e-14 * T4 - 4.837196970000000e+04 * invT) *
              0.0227226249176305;
    // species 5: N2
    result += y[5] *
              (+3.298677000000000e+00 + 7.041202000000000e-04 * T -
               1.321074000000000e-06 * T2 + 1.410378750000000e-09 * T3 -
               4.889707999999999e-13 * T4 - 1.020899900000000e+03 * invT) *
              0.0356964374955379;
  } else {
    // species 0: O2
    result += y[0] *
              (+3.282537840000000e+00 + 7.415437700000000e-04 * T -
               2.526555563333333e-07 * T2 + 5.236763875000000e-11 * T3 -
               4.334355880000000e-15 * T4 - 1.088457720000000e+03 * invT) *
              0.0312519532470779;
    // species 1: H2O
    result += y[1] *
              (+3.033992490000000e+00 + 1.088459020000000e-03 * T -
               5.469083933333333e-08 * T2 - 2.426049675000000e-11 * T3 +
               3.364019840000000e-15 * T4 - 3.000429710000000e+04 * invT) *
              0.0555092978073827;
    // species 2: CH4
    result += y[2] *
              (+7.485149500000000e-02 + 6.695473350000000e-03 * T -
               1.910952696666667e-06 * T2 + 3.057313375000000e-10 * T3 -
               2.036304600000000e-14 * T4 - 9.468344590000001e+03 * invT) *
              0.0623324814560868;
    // species 3: CO
    result += y[3] *
              (+2.715185610000000e+00 + 1.031263715000000e-03 * T -
               3.329419236666667e-07 * T2 + 5.751325200000000e-11 * T3 -
               4.072954320000000e-15 * T4 - 1.415187240000000e+04 * invT) *
              0.0357015351660121;
    // species 4: CO2
    result += y[4] *
              (+3.857460290000000e+00 + 2.207185130000000e-03 * T -
               7.382713466666667e-07 * T2 + 1.308725470000000e-10 * T3 -
               9.441683280000000e-15 * T4 - 4.875916600000000e+04 * invT) *
              0.0227226249176305;
    // species 5: N2
    result += y[5] *
              (+2.926640000000000e+00 + 7.439884000000000e-04 * T -
               1.894920000000000e-07 * T2 + 2.524259500000000e-11 * T3 -
               1.350670200000000e-15 * T4 - 9.227977000000000e+02 * invT) *
              0.0356964374955379;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[6];                        //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 6; ++id) {
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

  // species with midpoint at T=1000 kelvin
  if (T < 1000) {
    // species 0: O2
    result += y[0] *
              (+2.782456360000000e+00 - 1.498367080000000e-03 * T +
               3.282434003333333e-06 * T2 - 2.420323772500000e-09 * T3 +
               6.487456740000000e-13 * T4 - 1.063943560000000e+03 * invT) *
              0.0312519532470779;
    // species 1: H2O
    result += y[1] *
              (+3.198640560000000e+00 - 1.018217050000000e-03 * T +
               2.173467370000000e-06 * T2 - 1.371992655000000e-09 * T3 +
               3.543956340000000e-13 * T4 - 3.029372670000000e+04 * invT) *
              0.0555092978073827;
    // species 2: CH4
    result += y[2] *
              (+4.149876130000000e+00 - 6.835489400000000e-03 * T +
               1.639335330000000e-05 * T2 - 1.211857565000000e-08 * T3 +
               3.333879120000000e-12 * T4 - 1.024664760000000e+04 * invT) *
              0.0623324814560868;
    // species 3: CO
    result += y[3] *
              (+2.579533470000000e+00 - 3.051768400000000e-04 * T +
               3.389381100000000e-07 * T2 + 2.267514710000000e-10 * T3 -
               1.808848998000000e-13 * T4 - 1.434408600000000e+04 * invT) *
              0.0357015351660121;
    // species 4: CO2
    result += y[4] *
              (+1.356773520000000e+00 + 4.492298385000000e-03 * T -
               2.374520896666666e-06 * T2 + 6.147975550000000e-10 * T3 -
               2.873990960000000e-14 * T4 - 4.837196970000000e+04 * invT) *
              0.0227226249176305;
    // species 5: N2
    result += y[5] *
              (+2.298677000000000e+00 + 7.041202000000000e-04 * T -
               1.321074000000000e-06 * T2 + 1.410378750000000e-09 * T3 -
               4.889707999999999e-13 * T4 - 1.020899900000000e+03 * invT) *
              0.0356964374955379;
  } else {
    // species 0: O2
    result += y[0] *
              (+2.282537840000000e+00 + 7.415437700000000e-04 * T -
               2.526555563333333e-07 * T2 + 5.236763875000000e-11 * T3 -
               4.334355880000000e-15 * T4 - 1.088457720000000e+03 * invT) *
              0.0312519532470779;
    // species 1: H2O
    result += y[1] *
              (+2.033992490000000e+00 + 1.088459020000000e-03 * T -
               5.469083933333333e-08 * T2 - 2.426049675000000e-11 * T3 +
               3.364019840000000e-15 * T4 - 3.000429710000000e+04 * invT) *
              0.0555092978073827;
    // species 2: CH4
    result += y[2] *
              (-9.251485050000000e-01 + 6.695473350000000e-03 * T -
               1.910952696666667e-06 * T2 + 3.057313375000000e-10 * T3 -
               2.036304600000000e-14 * T4 - 9.468344590000001e+03 * invT) *
              0.0623324814560868;
    // species 3: CO
    result += y[3] *
              (+1.715185610000000e+00 + 1.031263715000000e-03 * T -
               3.329419236666667e-07 * T2 + 5.751325200000000e-11 * T3 -
               4.072954320000000e-15 * T4 - 1.415187240000000e+04 * invT) *
              0.0357015351660121;
    // species 4: CO2
    result += y[4] *
              (+2.857460290000000e+00 + 2.207185130000000e-03 * T -
               7.382713466666667e-07 * T2 + 1.308725470000000e-10 * T3 -
               9.441683280000000e-15 * T4 - 4.875916600000000e+04 * invT) *
              0.0227226249176305;
    // species 5: N2
    result += y[5] *
              (+1.926640000000000e+00 + 7.439884000000000e-04 * T -
               1.894920000000000e-07 * T2 + 2.524259500000000e-11 * T3 -
               1.350670200000000e-15 * T4 - 9.227977000000000e+02 * invT) *
              0.0356964374955379;
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
  amrex::Real sor[6]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 6; ++id) {
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
  amrex::Real sor[6];  //  temporary storage
  amrex::Real x[6];    //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 6; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (3.1998000000000001e+01 * YOW);
  x[1] = y[1] / (1.8015000000000001e+01 * YOW);
  x[2] = y[2] / (1.6042999999999999e+01 * YOW);
  x[3] = y[3] / (2.8009999999999998e+01 * YOW);
  x[4] = y[4] / (4.4009000000000000e+01 * YOW);
  x[5] = y[5] / (2.8013999999999999e+01 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 6; i++) {
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
  XW += x[0] * 3.1998000000000001e+01;     // O2
  XW += x[1] * 1.8015000000000001e+01;     // H2O
  XW += x[2] * 1.6042999999999999e+01;     // CH4
  XW += x[3] * 2.8009999999999998e+01;     // CO
  XW += x[4] * 4.4009000000000000e+01;     // CO2
  XW += x[5] * 2.8013999999999999e+01;     // N2
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

  for (int i = 0; i < 6; i++) {
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
  W += c[0] * 3.1998000000000001e+01; // O2
  W += c[1] * 1.8015000000000001e+01; // H2O
  W += c[2] * 1.6042999999999999e+01; // CH4
  W += c[3] * 2.8009999999999998e+01; // CO
  W += c[4] * 4.4009000000000000e+01; // CO2
  W += c[5] * 2.8013999999999999e+01; // N2

  for (int id = 0; id < 6; ++id) {
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
  XW += x[0] * 3.1998000000000001e+01;       // O2
  XW += x[1] * 1.8015000000000001e+01;       // H2O
  XW += x[2] * 1.6042999999999999e+01;       // CH4
  XW += x[3] * 2.8009999999999998e+01;       // CO
  XW += x[4] * 4.4009000000000000e+01;       // CO2
  XW += x[5] * 2.8013999999999999e+01;       // N2
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

  for (int i = 0; i < 6; i++) {
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
  W += c[0] * 3.1998000000000001e+01; // O2
  W += c[1] * 1.8015000000000001e+01; // H2O
  W += c[2] * 1.6042999999999999e+01; // CH4
  W += c[3] * 2.8009999999999998e+01; // CO
  W += c[4] * 4.4009000000000000e+01; // CO2
  W += c[5] * 2.8013999999999999e+01; // N2

  for (int id = 0; id < 6; ++id) {
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

  for (int i = 0; i < 6; i++) {
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
  XW += x[0] * 3.1998000000000001e+01; // O2
  XW += x[1] * 1.8015000000000001e+01; // H2O
  XW += x[2] * 1.6042999999999999e+01; // CH4
  XW += x[3] * 2.8009999999999998e+01; // CO
  XW += x[4] * 4.4009000000000000e+01; // CO2
  XW += x[5] * 2.8013999999999999e+01; // N2
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
  W += c[0] * 3.1998000000000001e+01; // O2
  W += c[1] * 1.8015000000000001e+01; // H2O
  W += c[2] * 1.6042999999999999e+01; // CH4
  W += c[3] * 2.8009999999999998e+01; // CO
  W += c[4] * 4.4009000000000000e+01; // CO2
  W += c[5] * 2.8013999999999999e+01; // N2

  for (int id = 0; id < 6; ++id) {
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

  for (int i = 0; i < 6; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 6; i++) {
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
  for (int i = 0; i < 6; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 6; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 6; i++) {
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

  for (int i = 0; i < 6; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 3.1998000000000001e+01; // O2
  XW += x[1] * 1.8015000000000001e+01; // H2O
  XW += x[2] * 1.6042999999999999e+01; // CH4
  XW += x[3] * 2.8009999999999998e+01; // CO
  XW += x[4] * 4.4009000000000000e+01; // CO2
  XW += x[5] * 2.8013999999999999e+01; // N2
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 3.1998000000000001e+01 * XWinv;
  y[1] = x[1] * 1.8015000000000001e+01 * XWinv;
  y[2] = x[2] * 1.6042999999999999e+01 * XWinv;
  y[3] = x[3] * 2.8009999999999998e+01 * XWinv;
  y[4] = x[4] * 4.4009000000000000e+01 * XWinv;
  y[5] = x[5] * 2.8013999999999999e+01 * XWinv;
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
  for (int id = 0; id < 6; ++id) {
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
  XW += x[0] * 3.1998000000000001e+01; // O2
  XW += x[1] * 1.8015000000000001e+01; // H2O
  XW += x[2] * 1.6042999999999999e+01; // CH4
  XW += x[3] * 2.8009999999999998e+01; // CO
  XW += x[4] * 4.4009000000000000e+01; // CO2
  XW += x[5] * 2.8013999999999999e+01; // N2
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 6; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 6; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 6; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 3.1998000000000001e+01; // O2
  CW += c[1] * 1.8015000000000001e+01; // H2O
  CW += c[2] * 1.6042999999999999e+01; // CH4
  CW += c[3] * 2.8009999999999998e+01; // CO
  CW += c[4] * 4.4009000000000000e+01; // CO2
  CW += c[5] * 2.8013999999999999e+01; // N2
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 3.1998000000000001e+01 * CWinv;
  y[1] = c[1] * 1.8015000000000001e+01 * CWinv;
  y[2] = c[2] * 1.6042999999999999e+01 * CWinv;
  y[3] = c[3] * 2.8009999999999998e+01 * CWinv;
  y[4] = c[4] * 4.4009000000000000e+01 * CWinv;
  y[5] = c[5] * 2.8013999999999999e+01 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 6; ++id) {
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
  for (int id = 0; id < 6; ++id) {
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
  for (int id = 0; id < 6; ++id) {
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
  for (int id = 0; id < 6; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 6; ++id) {
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
  cvms[0] *= 2.598431970171023e+06; // O2
  cvms[1] *= 4.615299815794193e+06; // H2O
  cvms[2] *= 5.182610869633635e+06; // CH4
  cvms[3] *= 2.968390795484913e+06; // CO
  cvms[4] *= 1.889264154639560e+06; // CO2
  cvms[5] *= 2.967966951578939e+06; // N2
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 2.598431970171023e+06; // O2
  cpms[1] *= 4.615299815794193e+06; // H2O
  cpms[2] *= 5.182610869633635e+06; // CH4
  cpms[3] *= 2.968390795484913e+06; // CO
  cpms[4] *= 1.889264154639560e+06; // CO2
  cpms[5] *= 2.967966951578939e+06; // N2
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 6; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 6; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 2.598431970171023e+06; // O2
  sms[1] *= 4.615299815794193e+06; // H2O
  sms[2] *= 5.182610869633635e+06; // CH4
  sms[3] *= 2.968390795484913e+06; // CO
  sms[4] *= 1.889264154639560e+06; // CO2
  sms[5] *= 2.967966951578939e+06; // N2
}

// GPU version of productionRate: no more use of thermo namespace vectors
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
comp_qfqr(
  amrex::Real* qf,
  amrex::Real* qr,
  const amrex::Real* sc,
  const amrex::Real* /*sc_qss*/,
  const amrex::Real T,
  const amrex::Real invT,
  const amrex::Real logT)
{

  // reaction 0: 2 CH4 + 3 O2 => 2 CO + 4 H2O
  qf[0] = sc[0] * sc[2];
  qr[0] = 0.0;

  // reaction 1: 2 CO + O2 => 2 CO2
  qf[1] = pow(std::max(sc[0], 0.0), 0.250000) *
          std::sqrt(std::max(sc[1], 0.0)) * sc[3];
  qr[1] = 0.0;

  // reaction 2: 2 CO2 => 2 CO + O2
  qf[2] = sc[4];
  qr[2] = 0.0;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 6; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[6];
  gibbs(g_RT, T);
  // Kc from shared exp(-g/RT) factors where no factor or partial product can
  // overflow (checked by CEPTR on this T range); exp of the sum otherwise and
  // in single precision
  const bool kc_fast =
    sizeof(amrex::Real) == sizeof(double) && T >= 200.0 && T <= 10000.0;
  amrex::Real ge[6] = {}, inv_ge[6] = {};
  if (kc_fast) {
    for (int i = 0; i < 6; ++i) {
      ge[i] = exp(-g_RT[i]);
      inv_ge[i] = 1.0 / ge[i];
    }
  }

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;

  // reaction 0:  2 CH4 + 3 O2 => 2 CO + 4 H2O
  k_f = 154500000 * exp((0.5) * logT - (20075.8288822793) * invT);
  qf[0] *= k_f;
  qr[0] *= k_f *
           exp(
             -(3.000000 * g_RT[0] - 4.000000 * g_RT[1] + 2.000000 * g_RT[2] -
               2.000000 * g_RT[3])) *
           (refCinv);
  // reaction 1:  2 CO + O2 => 2 CO2
  k_f = 6296094821.39524 * exp(-(20128.6666321888) * invT);
  qf[1] *= k_f;
  qr[1] *=
    k_f *
    (kc_fast ? (ge[0] * ge[3] * inv_ge[4] * ge[3] * inv_ge[4])
             : exp(-(g_RT[0] + 2.000000 * g_RT[3] - 2.000000 * g_RT[4]))) *
    (refC);
  // reaction 2:  2 CO2 => 2 CO + O2
  k_f = 250000000 * exp(-(20128.6666321888) * invT);
  qf[2] *= k_f;
  qr[2] *=
    k_f *
    (kc_fast ? (inv_ge[0] * inv_ge[3] * ge[4] * inv_ge[3] * ge[4])
             : exp(-(-g_RT[0] - 2.000000 * g_RT[3] + 2.000000 * g_RT[4]))) *
    (refCinv);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 6; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 6; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[6];
  gibbs(g_RT, T);
  // Kc from shared exp(-g/RT) factors where no factor or partial product can
  // overflow (checked by CEPTR on this T range); exp of the sum otherwise and
  // in single precision
  const bool kc_fast =
    sizeof(amrex::Real) == sizeof(double) && T >= 200.0 && T <= 10000.0;
  amrex::Real ge[6] = {}, inv_ge[6] = {};
  if (kc_fast) {
    for (int i = 0; i < 6; ++i) {
      ge[i] = exp(-g_RT[i]);
      inv_ge[i] = 1.0 / ge[i];
    }
  }

  {
    // reaction 0:  2 CH4 + 3 O2 => 2 CO + 4 H2O
    const amrex::Real k_f =
      154500000 * exp((0.5) * logT - (20075.8288822793) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[2]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[0] -= 3.000000 * qdot;
    wdot[1] += 4.000000 * qdot;
    wdot[2] -= 2.000000 * qdot;
    wdot[3] += 2.000000 * qdot;
  }

  {
    // reaction 1:  2 CO + O2 => 2 CO2
    const amrex::Real k_f = 6296094821.39524 * exp(-(20128.6666321888) * invT);
    const amrex::Real qf = k_f * (pow(std::max(sc[0], 0.0), 0.250000) *
                                  std::sqrt(std::max(sc[1], 0.0)) * sc[3]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[3] -= 2.000000 * qdot;
    wdot[4] += 2.000000 * qdot;
  }

  {
    // reaction 2:  2 CO2 => 2 CO + O2
    const amrex::Real k_f = 250000000 * exp(-(20128.6666321888) * invT);
    const amrex::Real qf = k_f * (sc[4]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[0] += qdot;
    wdot[3] += 2.000000 * qdot;
    wdot[4] -= 2.000000 * qdot;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 6; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 6; ++id) {
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
  amrex::Real c[6]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 6; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 6; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 6; ++id) {
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
  amrex::Real c[6]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 6; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 6; ++id) {
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
  amrex::Real c[6]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 6; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 6; ++id) {
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
  amrex::Real c[6];   // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 3.1998000000000001e+01; // O2
  XW += x[1] * 1.8015000000000001e+01; // H2O
  XW += x[2] * 1.6042999999999999e+01; // CH4
  XW += x[3] * 2.8009999999999998e+01; // CO
  XW += x[4] * 4.4009000000000000e+01; // CO2
  XW += x[5] * 2.8013999999999999e+01; // N2
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 6; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 6; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0; // O2
  kcharge[1] = 0; // H2O
  kcharge[2] = 0; // CH4
  kcharge[3] = 0; // CO
  kcharge[4] = 0; // CO2
  kcharge[5] = 0; // N2
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[6];
  CKCHRG(kchrg);

  for (int id = 0; id < 6; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 49; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[6];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 6; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[6];
  gibbs(g_RT, T);
  // Kc from shared exp(-g/RT) factors where no factor or partial product can
  // overflow (checked by CEPTR on this T range); exp of the sum otherwise and
  // in single precision
  const bool kc_fast =
    sizeof(amrex::Real) == sizeof(double) && T >= 200.0 && T <= 10000.0;
  amrex::Real ge[6] = {}, inv_ge[6] = {};
  if (kc_fast) {
    for (int i = 0; i < 6; ++i) {
      ge[i] = exp(-g_RT[i]);
      inv_ge[i] = 1.0 / ge[i];
    }
  }

  // compute the species enthalpy
  amrex::Real h_RT[6];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[6];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: 2 CH4 + 3 O2 => 2 CO + 4 H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[2];
  k_f = 154500000 * exp(0.5 * logT - (20075.8288822793) * invT);
  dlnkfdT = 0.5 * invT + (20075.8288822793) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[0] -= 3 * q; // O2
  wdot[1] += 4 * q; // H2O
  wdot[2] -= 2 * q; // CH4
  wdot[3] += 2 * q; // CO
  // d()/d[O2]
  dqdci = +k_f * sc[2];
  J[0] += -3 * dqdci; // dwdot[O2]/d[O2]
  J[1] += 4 * dqdci;  // dwdot[H2O]/d[O2]
  J[2] += -2 * dqdci; // dwdot[CH4]/d[O2]
  J[3] += 2 * dqdci;  // dwdot[CO]/d[O2]
  // d()/d[CH4]
  dqdci = +k_f * sc[0];
  J[14] += -3 * dqdci; // dwdot[O2]/d[CH4]
  J[15] += 4 * dqdci;  // dwdot[H2O]/d[CH4]
  J[16] += -2 * dqdci; // dwdot[CH4]/d[CH4]
  J[17] += 2 * dqdci;  // dwdot[CO]/d[CH4]
  // d()/dT
  J[42] += -3 * dqdT; // dwdot[O2]/dT
  J[43] += 4 * dqdT;  // dwdot[H2O]/dT
  J[44] += -2 * dqdT; // dwdot[CH4]/dT
  J[45] += 2 * dqdT;  // dwdot[CO]/dT

  // reaction 1: 2 CO + O2 => 2 CO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = pow(std::max(sc[0], 0.0), 0.250000) *
          std::sqrt(std::max(sc[1], 0.0)) * sc[3];
  k_f = 6296094821.39524 * exp(-(20128.6666321888) * invT);
  dlnkfdT = (20128.6666321888) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[0] -= q;     // O2
  wdot[3] -= 2 * q; // CO
  wdot[4] += 2 * q; // CO2
  // d()/d[O2]
  dqdci = +k_f * 0.250000 * pow(std::max(sc[0], 1e-16), -0.750000) *
          std::sqrt(std::max(sc[1], 0.0)) * sc[3];
  J[0] -= dqdci;      // dwdot[O2]/d[O2]
  J[3] += -2 * dqdci; // dwdot[CO]/d[O2]
  J[4] += 2 * dqdci;  // dwdot[CO2]/d[O2]
  // d()/d[H2O]
  dqdci = +k_f * pow(std::max(sc[0], 0.0), 0.250000) * 0.500000 *
          pow(std::max(sc[1], 1e-16), -0.500000) * sc[3];
  J[7] -= dqdci;       // dwdot[O2]/d[H2O]
  J[10] += -2 * dqdci; // dwdot[CO]/d[H2O]
  J[11] += 2 * dqdci;  // dwdot[CO2]/d[H2O]
  // d()/d[CO]
  dqdci = +k_f * pow(std::max(sc[0], 0.0), 0.250000) *
          std::sqrt(std::max(sc[1], 0.0));
  J[21] -= dqdci;      // dwdot[O2]/d[CO]
  J[24] += -2 * dqdci; // dwdot[CO]/d[CO]
  J[25] += 2 * dqdci;  // dwdot[CO2]/d[CO]
  // d()/dT
  J[42] -= dqdT;      // dwdot[O2]/dT
  J[45] += -2 * dqdT; // dwdot[CO]/dT
  J[46] += 2 * dqdT;  // dwdot[CO2]/dT

  // reaction 2: 2 CO2 => 2 CO + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4];
  k_f = 250000000 * exp(-(20128.6666321888) * invT);
  dlnkfdT = (20128.6666321888) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[0] += q;     // O2
  wdot[3] += 2 * q; // CO
  wdot[4] -= 2 * q; // CO2
  // d()/d[CO2]
  dqdci = +k_f;
  J[28] += dqdci;      // dwdot[O2]/d[CO2]
  J[31] += 2 * dqdci;  // dwdot[CO]/d[CO2]
  J[32] += -2 * dqdci; // dwdot[CO2]/d[CO2]
  // d()/dT
  J[42] += dqdT;      // dwdot[O2]/dT
  J[45] += 2 * dqdT;  // dwdot[CO]/dT
  J[46] += -2 * dqdT; // dwdot[CO2]/dT

  amrex::Real c_R[6], dcRdT[6], e_RT[6];
  amrex::Real* eh_RT;
  if (HP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 6; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[42 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 6; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 6; ++m) {
      dehmixdc += eh_RT[m] * J[k * 7 + m];
    }
    J[k * 7 + 6] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[48] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[6];

  for (int k = 0; k < 6; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 6; k++) {
    J[42 + k] *= 1.e-6;
    J[k * 7 + 6] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 49; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[6];
  for (auto& val : wdot) {
    val = 0.0;
  }

  const amrex::Real invT = 1.0 / T;
  const amrex::Real invT2 = invT * invT;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 / T;
  amrex::Real refCinv = 1.0 / refC;

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int k = 0; k < 6; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[6];
  gibbs(g_RT, T);
  // Kc from shared exp(-g/RT) factors where no factor or partial product can
  // overflow (checked by CEPTR on this T range); exp of the sum otherwise and
  // in single precision
  const bool kc_fast =
    sizeof(amrex::Real) == sizeof(double) && T >= 200.0 && T <= 10000.0;
  amrex::Real ge[6] = {}, inv_ge[6] = {};
  if (kc_fast) {
    for (int i = 0; i < 6; ++i) {
      ge[i] = exp(-g_RT[i]);
      inv_ge[i] = 1.0 / ge[i];
    }
  }

  // compute the species enthalpy
  amrex::Real h_RT[6];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[6];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 0: 2 CH4 + 3 O2 => 2 CO + 4 H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[2];
  k_f = 154500000 * exp(0.5 * logT - (20075.8288822793) * invT);
  dlnkfdT = 0.5 * invT + (20075.8288822793) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[0] -= 3 * q; // O2
  wdot[1] += 4 * q; // H2O
  wdot[2] -= 2 * q; // CH4
  wdot[3] += 2 * q; // CO
  // d()/d[O2]
  dqdci = +k_f * sc[2];
  J[0] += -3 * dqdci; // dwdot[O2]/d[O2]
  J[1] += 4 * dqdci;  // dwdot[H2O]/d[O2]
  J[2] += -2 * dqdci; // dwdot[CH4]/d[O2]
  J[3] += 2 * dqdci;  // dwdot[CO]/d[O2]
  // d()/d[CH4]
  dqdci = +k_f * sc[0];
  J[14] += -3 * dqdci; // dwdot[O2]/d[CH4]
  J[15] += 4 * dqdci;  // dwdot[H2O]/d[CH4]
  J[16] += -2 * dqdci; // dwdot[CH4]/d[CH4]
  J[17] += 2 * dqdci;  // dwdot[CO]/d[CH4]
  // d()/dT
  J[42] += -3 * dqdT; // dwdot[O2]/dT
  J[43] += 4 * dqdT;  // dwdot[H2O]/dT
  J[44] += -2 * dqdT; // dwdot[CH4]/dT
  J[45] += 2 * dqdT;  // dwdot[CO]/dT

  // reaction 1: 2 CO + O2 => 2 CO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = pow(std::max(sc[0], 0.0), 0.250000) *
          std::sqrt(std::max(sc[1], 0.0)) * sc[3];
  k_f = 6296094821.39524 * exp(-(20128.6666321888) * invT);
  dlnkfdT = (20128.6666321888) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[0] -= q;     // O2
  wdot[3] -= 2 * q; // CO
  wdot[4] += 2 * q; // CO2
  // d()/d[O2]
  dqdci = +k_f * 0.250000 * pow(std::max(sc[0], 1e-16), -0.750000) *
          std::sqrt(std::max(sc[1], 0.0)) * sc[3];
  J[0] -= dqdci;      // dwdot[O2]/d[O2]
  J[3] += -2 * dqdci; // dwdot[CO]/d[O2]
  J[4] += 2 * dqdci;  // dwdot[CO2]/d[O2]
  // d()/d[H2O]
  dqdci = +k_f * pow(std::max(sc[0], 0.0), 0.250000) * 0.500000 *
          pow(std::max(sc[1], 1e-16), -0.500000) * sc[3];
  J[7] -= dqdci;       // dwdot[O2]/d[H2O]
  J[10] += -2 * dqdci; // dwdot[CO]/d[H2O]
  J[11] += 2 * dqdci;  // dwdot[CO2]/d[H2O]
  // d()/d[CO]
  dqdci = +k_f * pow(std::max(sc[0], 0.0), 0.250000) *
          std::sqrt(std::max(sc[1], 0.0));
  J[21] -= dqdci;      // dwdot[O2]/d[CO]
  J[24] += -2 * dqdci; // dwdot[CO]/d[CO]
  J[25] += 2 * dqdci;  // dwdot[CO2]/d[CO]
  // d()/dT
  J[42] -= dqdT;      // dwdot[O2]/dT
  J[45] += -2 * dqdT; // dwdot[CO]/dT
  J[46] += 2 * dqdT;  // dwdot[CO2]/dT

  // reaction 2: 2 CO2 => 2 CO + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4];
  k_f = 250000000 * exp(-(20128.6666321888) * invT);
  dlnkfdT = (20128.6666321888) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[0] += q;     // O2
  wdot[3] += 2 * q; // CO
  wdot[4] -= 2 * q; // CO2
  // d()/d[CO2]
  dqdci = +k_f;
  J[28] += dqdci;      // dwdot[O2]/d[CO2]
  J[31] += 2 * dqdci;  // dwdot[CO]/d[CO2]
  J[32] += -2 * dqdci; // dwdot[CO2]/d[CO2]
  // d()/dT
  J[42] += dqdT;      // dwdot[O2]/dT
  J[45] += 2 * dqdT;  // dwdot[CO]/dT
  J[46] += -2 * dqdT; // dwdot[CO2]/dT

  amrex::Real c_R[6], dcRdT[6], e_RT[6];
  amrex::Real* eh_RT;
  if (consP == 1) {
    cp_R(c_R, T);
    dcvpRdT(dcRdT, T);
    eh_RT = &h_RT[0];
  } else {
    cv_R(c_R, T);
    dcvpRdT(dcRdT, T);
    speciesInternalEnergy(e_RT, T);
    eh_RT = &e_RT[0];
  }

  amrex::Real cmix = 0.0, ehmix = 0.0, dcmixdT = 0.0, dehmixdT = 0.0;
  for (int k = 0; k < 6; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[42 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 6; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 6; ++m) {
      dehmixdc += eh_RT[m] * J[k * 7 + m];
    }
    J[k * 7 + 6] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[48] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
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
  amrex::Real c[6];

  for (int k = 0; k < 6; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 6; k++) {
    J[42 + k] *= 1.e-6;
    J[k * 7 + 6] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 24;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 846;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 6;
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
  WT[0] = 3.19980000E+01;
  WT[1] = 1.80150000E+01;
  WT[2] = 1.60430000E+01;
  WT[3] = 2.80100000E+01;
  WT[4] = 4.40090000E+01;
  WT[5] = 2.80140000E+01;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 1.07400000E+02;
  EPS[1] = 5.72400000E+02;
  EPS[2] = 1.41400000E+02;
  EPS[3] = 9.81000000E+01;
  EPS[4] = 2.44000000E+02;
  EPS[5] = 9.75300000E+01;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 3.45800000E+00;
  SIG[1] = 2.60500000E+00;
  SIG[2] = 3.74600000E+00;
  SIG[3] = 3.65000000E+00;
  SIG[4] = 3.76300000E+00;
  SIG[5] = 3.62100000E+00;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 1.84400000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 0.00000000E+00;
  DIP[4] = 0.00000000E+00;
  DIP[5] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 1.60000000E+00;
  POL[1] = 0.00000000E+00;
  POL[2] = 2.60000000E+00;
  POL[3] = 1.95000000E+00;
  POL[4] = 2.65000000E+00;
  POL[5] = 1.76000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 3.80000000E+00;
  ZROT[1] = 4.00000000E+00;
  ZROT[2] = 1.30000000E+01;
  ZROT[3] = 1.80000000E+00;
  ZROT[4] = 2.10000000E+00;
  ZROT[5] = 4.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 1;
  NLIN[1] = 2;
  NLIN[2] = 2;
  NLIN[3] = 1;
  NLIN[4] = 1;
  NLIN[5] = 1;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.68118998E+01;
  COFETA[1] = 2.52362554E+00;
  COFETA[2] = -2.49309128E-01;
  COFETA[3] = 1.10211025E-02;
  COFETA[4] = -1.17770995E+01;
  COFETA[5] = -8.26744826E-01;
  COFETA[6] = 3.39009392E-01;
  COFETA[7] = -2.00674472E-02;
  COFETA[8] = -1.95453436E+01;
  COFETA[9] = 3.36385478E+00;
  COFETA[10] = -3.56948469E-01;
  COFETA[11] = 1.56210922E-02;
  COFETA[12] = -1.63031343E+01;
  COFETA[13] = 2.26143219E+00;
  COFETA[14] = -2.15114671E-01;
  COFETA[15] = 9.53461976E-03;
  COFETA[16] = -2.36749638E+01;
  COFETA[17] = 4.99775518E+00;
  COFETA[18] = -5.52687718E-01;
  COFETA[19] = 2.34353338E-02;
  COFETA[20] = -1.62526676E+01;
  COFETA[21] = 2.24839597E+00;
  COFETA[22] = -2.13428438E-01;
  COFETA[23] = 9.46192413E-03;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = -3.01283106E+00;
  COFLAM[1] = 3.37554996E+00;
  COFLAM[2] = -3.43353130E-01;
  COFLAM[3] = 1.51043449E-02;
  COFLAM[4] = 2.28195672E+01;
  COFLAM[5] = -8.72278727E+00;
  COFLAM[6] = 1.49300458E+00;
  COFLAM[7] = -7.41523911E-02;
  COFLAM[8] = 8.43904098E+00;
  COFLAM[9] = -2.78020359E+00;
  COFLAM[10] = 7.09313628E-01;
  COFLAM[11] = -4.04300952E-02;
  COFLAM[12] = 9.92460061E+00;
  COFLAM[13] = -2.28317864E+00;
  COFLAM[14] = 4.73113319E-01;
  COFLAM[15] = -2.40056449E-02;
  COFLAM[16] = -1.24047472E+01;
  COFLAM[17] = 6.34783072E+00;
  COFLAM[18] = -6.37857831E-01;
  COFLAM[19] = 2.37613806E-02;
  COFLAM[20] = 1.15506870E+01;
  COFLAM[21] = -2.91452034E+00;
  COFLAM[22] = 5.55043078E-01;
  COFLAM[23] = -2.75172223E-02;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.53110597E+01;
  COFD[1] = 3.37317428E+00;
  COFD[2] = -2.24900439E-01;
  COFD[3] = 9.81228151E-03;
  COFD[4] = -2.10287057E+01;
  COFD[5] = 5.50980703E+00;
  COFD[6] = -4.78335492E-01;
  COFD[7] = 1.98515434E-02;
  COFD[8] = -1.31492509E+01;
  COFD[9] = 1.48004045E+00;
  COFD[10] = 1.60499956E-01;
  COFD[11] = -1.19765871E-02;
  COFD[12] = -1.59878735E+01;
  COFD[13] = 3.66478157E+00;
  COFD[14] = -2.61506432E-01;
  COFD[15] = 1.13466231E-02;
  COFD[16] = -2.12561676E+01;
  COFD[17] = 5.52385262E+00;
  COFD[18] = -4.69683817E-01;
  COFD[19] = 1.90677254E-02;
  COFD[20] = -1.68532873E+01;
  COFD[21] = 4.00572564E+00;
  COFD[22] = -3.04255586E-01;
  COFD[23] = 1.31384083E-02;
  COFD[24] = -1.50371688E+01;
  COFD[25] = 3.26249588E+00;
  COFD[26] = -2.10658287E-01;
  COFD[27] = 9.20032462E-03;
  COFD[28] = -2.08561919E+01;
  COFD[29] = 5.44718673E+00;
  COFD[30] = -4.72082974E-01;
  COFD[31] = 1.96531328E-02;
  COFD[32] = -1.56953785E+01;
  COFD[33] = 3.54443207E+00;
  COFD[34] = -2.46133003E-01;
  COFD[35] = 1.06905018E-02;
  COFD[36] = -1.48061406E+01;
  COFD[37] = 3.16912473E+00;
  COFD[38] = -1.98792456E-01;
  COFD[39] = 8.69726395E-03;
  COFD[40] = -1.81197250E+01;
  COFD[41] = 4.33684042E+00;
  COFD[42] = -3.44981265E-01;
  COFD[43] = 1.48142449E-02;
  COFD[44] = -2.11719364E+01;
  COFD[45] = 5.20775027E+00;
  COFD[46] = -4.07348285E-01;
  COFD[47] = 1.55473262E-02;
  COFD[48] = -1.89594762E+01;
  COFD[49] = 4.68735220E+00;
  COFD[50] = -3.87449514E-01;
  COFD[51] = 1.65352261E-02;
  COFD[52] = -1.77672912E+01;
  COFD[53] = 4.20234040E+00;
  COFD[54] = -3.28057658E-01;
  COFD[55] = 1.41006192E-02;
  COFD[56] = -2.10907633E+01;
  COFD[57] = 5.29211327E+00;
  COFD[58] = -4.56068366E-01;
  COFD[59] = 1.91195062E-02;
  COFD[60] = -1.50096252E+01;
  COFD[61] = 3.25515933E+00;
  COFD[62] = -2.09710110E-01;
  COFD[63] = 9.15941830E-03;
  COFD[64] = -2.07768625E+01;
  COFD[65] = 5.42470175E+00;
  COFD[66] = -4.69700438E-01;
  COFD[67] = 1.95706912E-02;
  COFD[68] = -1.56756124E+01;
  COFD[69] = 3.54035770E+00;
  COFD[70] = -2.45653442E-01;
  COFD[71] = 1.06717969E-02;
  COFD[72] = -1.47850505E+01;
  COFD[73] = 3.16433919E+00;
  COFD[74] = -1.98191564E-01;
  COFD[75] = 8.67209742E-03;
  COFD[76] = -1.77350630E+01;
  COFD[77] = 4.19328271E+00;
  COFD[78] = -3.26911461E-01;
  COFD[79] = 1.40520357E-02;
  COFD[80] = -1.47639411E+01;
  COFD[81] = 3.15955654E+00;
  COFD[82] = -1.97590757E-01;
  COFD[83] = 8.64692156E-03;
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

  amrex::Real EPS[6];
  amrex::Real SIG[6];
  amrex::Real wt[6];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: O2
  // Imported from NIST
  Tci[0] = 154.581000;
  ai[0] = 1e6 * 0.42748 * Rcst * Rcst * Tci[0] * Tci[0] /
          (31.998800 * 31.998800 * 50.430466);
  bi[0] = 0.08664 * Rcst * Tci[0] / (31.998800 * 50.430466);
  acentric_i[0] = 0.022200;

  // species 1: H2O
  // Imported from NIST
  Tci[1] = 647.096000;
  ai[1] = 1e6 * 0.42748 * Rcst * Rcst * Tci[1] * Tci[1] /
          (18.015340 * 18.015340 * 220.640000);
  bi[1] = 0.08664 * Rcst * Tci[1] / (18.015340 * 220.640000);
  acentric_i[1] = 0.344300;

  // species 2: CH4
  // Imported from NIST
  Tci[2] = 190.560000;
  ai[2] = 1e6 * 0.42748 * Rcst * Rcst * Tci[2] * Tci[2] /
          (16.043030 * 16.043030 * 45.990000);
  bi[2] = 0.08664 * Rcst * Tci[2] / (16.043030 * 45.990000);
  acentric_i[2] = 0.011000;

  // species 3: CO
  // Imported from NIST
  Tci[3] = 132.850000;
  ai[3] = 1e6 * 0.42748 * Rcst * Rcst * Tci[3] * Tci[3] /
          (28.010000 * 28.010000 * 34.940000);
  bi[3] = 0.08664 * Rcst * Tci[3] / (28.010000 * 34.940000);
  acentric_i[3] = 0.045000;

  // species 4: CO2
  // Imported from NIST
  Tci[4] = 304.120000;
  ai[4] = 1e6 * 0.42748 * Rcst * Rcst * Tci[4] * Tci[4] /
          (44.009950 * 44.009950 * 73.740000);
  bi[4] = 0.08664 * Rcst * Tci[4] / (44.009950 * 73.740000);
  acentric_i[4] = 0.225000;

  // species 5: N2
  // Imported from NIST
  Tci[5] = 126.192000;
  ai[5] = 1e6 * 0.42748 * Rcst * Rcst * Tci[5] * Tci[5] /
          (28.013400 * 28.013400 * 33.958000);
  bi[5] = 0.08664 * Rcst * Tci[5] / (28.013400 * 33.958000);
  acentric_i[5] = 0.037200;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: O2
  // Imported from NIST
  sqrtOneOverTc[0] = 8.0430717653241e-02;
  sqrtAsti[0] = 3.6980081924740e+04;
  Bi[0] = 6.9005216444999e-01;
  Fomega[0] = 5.1945301904908e-01;

  // species 1: H2O
  // Imported from NIST
  sqrtOneOverTc[1] = 3.9311140369715e-02;
  sqrtAsti[1] = 1.3145468419592e+05;
  Bi[1] = 1.1727204136223e+00;
  Fomega[1] = 1.0013577274636e+00;

  // species 2: CH4
  // Imported from NIST
  sqrtOneOverTc[2] = 7.2440948474968e-02;
  sqrtAsti[2] = 9.5215046085708e+04;
  Bi[2] = 1.8605203320660e+00;
  Fomega[2] = 5.0213035482700e-01;

  // species 3: CO
  // Imported from NIST
  sqrtOneOverTc[3] = 8.6759935530451e-02;
  sqrtAsti[3] = 4.3619324712923e+04;
  Bi[3] = 9.7786303823350e-01;
  Fomega[3] = 5.5459948367500e-01;

  // species 4: CO2
  // Imported from NIST
  sqrtOneOverTc[4] = 5.7342616962522e-02;
  sqrtAsti[4] = 4.3745610142919e+04;
  Bi[4] = 6.7506167841745e-01;
  Fomega[4] = 8.2653709187500e-01;

  // species 5: N2
  // Imported from NIST
  sqrtOneOverTc[5] = 8.9019282240563e-02;
  sqrtAsti[5] = 4.2022983929562e+04;
  Bi[5] = 9.5560052221563e-01;
  Fomega[5] = 5.4259343186608e-01;
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
