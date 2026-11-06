#ifndef MECHANISM_H
#define MECHANISM_H

#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>

/* Elements
0  N
1  H
2  O
3  C
4  E
*/

// Species
#define N2_ID 0
#define H2_ID 1
#define H_ID 2
#define O2_ID 3
#define O_ID 4
#define H2O_ID 5
#define OH_ID 6
#define H2O2_ID 7
#define HO2_ID 8
#define CO_ID 9
#define CO2_ID 10
#define CH4_ID 11
#define CH3_ID 12
#define CH2_ID 13
#define CH2_S_ID 14
#define C_ID 15
#define CH_ID 16
#define CH3OH_ID 17
#define CH3O_ID 18
#define CH2OH_ID 19
#define CH2O_ID 20
#define HCO_ID 21
#define CHOp_ID 22
#define H3Op_ID 23
#define O2n_ID 24
#define E_ID 25

#define NUM_GAS_ELEMENTS 5    // Elements in the homogeneous phase
#define NUM_GAS_SPECIES 26    // Species in the homogeneous phase
#define NUM_GAS_REACTIONS 134 // Reactions in the homogeneous phase

#define SITE_DENSITY 0.000000E+00 // mol/cm^2

#define NUM_SURFACE_ELEMENTS 0  // Additional elements in heterogeneous phase
#define NUM_SURFACE_SPECIES 0   // Species in the heterogeneous phase
#define NUM_SURFACE_REACTIONS 0 // Reactions in the heterogeneous phase

#define NUM_ELEMENTS (NUM_GAS_ELEMENTS + NUM_SURFACE_ELEMENTS)
#define NUM_SPECIES (NUM_GAS_SPECIES + NUM_SURFACE_SPECIES)
#define NUM_REACTIONS (NUM_GAS_REACTIONS + NUM_SURFACE_REACTIONS)

#define NUM_IONS 4

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
  kk = 26;
  ii = 134;
  nfit = -1; // Why do you need this anyway ?
}

// save atomic weights into array
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
atomicWeight(amrex::Real* awt)
{
  awt[0] = 14.007000; // N
  awt[1] = 1.008000;  // H
  awt[2] = 15.999000; // O
  awt[3] = 12.011000; // C
  awt[4] = 0.000549;  // E
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
  for (int id = 0; id < kd * 26; ++id) {
    ncf[id] = 0;
  }

  // N2
  ncf[0 * kd + 0] = 2; // N

  // H2
  ncf[1 * kd + 1] = 2; // H

  // H
  ncf[2 * kd + 1] = 1; // H

  // O2
  ncf[3 * kd + 2] = 2; // O

  // O
  ncf[4 * kd + 2] = 1; // O

  // H2O
  ncf[5 * kd + 1] = 2; // H
  ncf[5 * kd + 2] = 1; // O

  // OH
  ncf[6 * kd + 1] = 1; // H
  ncf[6 * kd + 2] = 1; // O

  // H2O2
  ncf[7 * kd + 1] = 2; // H
  ncf[7 * kd + 2] = 2; // O

  // HO2
  ncf[8 * kd + 1] = 1; // H
  ncf[8 * kd + 2] = 2; // O

  // CO
  ncf[9 * kd + 3] = 1; // C
  ncf[9 * kd + 2] = 1; // O

  // CO2
  ncf[10 * kd + 3] = 1; // C
  ncf[10 * kd + 2] = 2; // O

  // CH4
  ncf[11 * kd + 3] = 1; // C
  ncf[11 * kd + 1] = 4; // H

  // CH3
  ncf[12 * kd + 3] = 1; // C
  ncf[12 * kd + 1] = 3; // H

  // CH2
  ncf[13 * kd + 3] = 1; // C
  ncf[13 * kd + 1] = 2; // H

  // CH2(S)
  ncf[14 * kd + 3] = 1; // C
  ncf[14 * kd + 1] = 2; // H

  // C
  ncf[15 * kd + 3] = 1; // C

  // CH
  ncf[16 * kd + 3] = 1; // C
  ncf[16 * kd + 1] = 1; // H

  // CH3OH
  ncf[17 * kd + 3] = 1; // C
  ncf[17 * kd + 1] = 4; // H
  ncf[17 * kd + 2] = 1; // O

  // CH3O
  ncf[18 * kd + 3] = 1; // C
  ncf[18 * kd + 1] = 3; // H
  ncf[18 * kd + 2] = 1; // O

  // CH2OH
  ncf[19 * kd + 3] = 1; // C
  ncf[19 * kd + 1] = 3; // H
  ncf[19 * kd + 2] = 1; // O

  // CH2O
  ncf[20 * kd + 3] = 1; // C
  ncf[20 * kd + 1] = 2; // H
  ncf[20 * kd + 2] = 1; // O

  // HCO
  ncf[21 * kd + 3] = 1; // C
  ncf[21 * kd + 1] = 1; // H
  ncf[21 * kd + 2] = 1; // O

  // CHO+
  ncf[22 * kd + 3] = 1;  // C
  ncf[22 * kd + 4] = -1; // E
  ncf[22 * kd + 1] = 1;  // H
  ncf[22 * kd + 2] = 1;  // O

  // H3O+
  ncf[23 * kd + 4] = -1; // E
  ncf[23 * kd + 1] = 3;  // H
  ncf[23 * kd + 2] = 1;  // O

  // O2-
  ncf[24 * kd + 4] = 1; // E
  ncf[24 * kd + 2] = 2; // O

  // E
  ncf[25 * kd + 4] = 1; // E
}

//  inverse molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_imw[26] = {
  3.5696437495537948e-02, // N2
  4.9603174603174605e-01, // H2
  9.9206349206349209e-01, // H
  3.1251953247077940e-02, // O2
  6.2503906494155881e-02, // O
  5.5509297807382736e-02, // H2O
  5.8799317927912029e-02, // OH
  2.9399658963956014e-02, // H2O2
  3.0297521662727988e-02, // HO2
  3.5701535166012141e-02, // CO
  2.2722624917630486e-02, // CO2
  6.2332481456086768e-02, // CH4
  6.6511473229132020e-02, // CH3
  7.1291081485706148e-02, // CH2
  7.1291081485706148e-02, // CH2(S)
  8.3257014403463503e-02, // C
  7.6810814962746762e-02, // CH
  3.1209038137444602e-02, // CH3OH
  3.2222723464587225e-02, // CH3O
  3.2222723464587225e-02, // CH2OH
  3.3304469459801506e-02, // CH2O
  3.4461368805568958e-02, // HCO
  3.4462020303672125e-02, // CHO+
  5.2569460050970102e-02, // H3O+
  3.1251417466724607e-02, // O2-
  1.8228884868472639e+03, // E
};
#endif
const amrex::Real h_global_imw[26] = {
  3.5696437495537948e-02, // N2
  4.9603174603174605e-01, // H2
  9.9206349206349209e-01, // H
  3.1251953247077940e-02, // O2
  6.2503906494155881e-02, // O
  5.5509297807382736e-02, // H2O
  5.8799317927912029e-02, // OH
  2.9399658963956014e-02, // H2O2
  3.0297521662727988e-02, // HO2
  3.5701535166012141e-02, // CO
  2.2722624917630486e-02, // CO2
  6.2332481456086768e-02, // CH4
  6.6511473229132020e-02, // CH3
  7.1291081485706148e-02, // CH2
  7.1291081485706148e-02, // CH2(S)
  8.3257014403463503e-02, // C
  7.6810814962746762e-02, // CH
  3.1209038137444602e-02, // CH3OH
  3.2222723464587225e-02, // CH3O
  3.2222723464587225e-02, // CH2OH
  3.3304469459801506e-02, // CH2O
  3.4461368805568958e-02, // HCO
  3.4462020303672125e-02, // CHO+
  5.2569460050970102e-02, // H3O+
  3.1251417466724607e-02, // O2-
  1.8228884868472639e+03, // E
};

//  molecular weights
#ifdef AMREX_USE_GPU
AMREX_GPU_CONSTANT const amrex::Real global_mw[26] = {
  2.8013999999999999e+01, // N2
  2.0160000000000000e+00, // H2
  1.0080000000000000e+00, // H
  3.1998000000000001e+01, // O2
  1.5999000000000001e+01, // O
  1.8015000000000001e+01, // H2O
  1.7007000000000001e+01, // OH
  3.4014000000000003e+01, // H2O2
  3.3006000000000000e+01, // HO2
  2.8009999999999998e+01, // CO
  4.4009000000000000e+01, // CO2
  1.6042999999999999e+01, // CH4
  1.5035000000000000e+01, // CH3
  1.4026999999999999e+01, // CH2
  1.4026999999999999e+01, // CH2(S)
  1.2010999999999999e+01, // C
  1.3018999999999998e+01, // CH
  3.2042000000000002e+01, // CH3OH
  3.1033999999999999e+01, // CH3O
  3.1033999999999999e+01, // CH2OH
  3.0026000000000000e+01, // CH2O
  2.9018000000000001e+01, // HCO
  2.9017451420091128e+01, // CHO+
  1.9022451420091127e+01, // H3O+
  3.1998548579908874e+01, // O2-
  5.4857990887282834e-04, // E
};
#endif
const amrex::Real h_global_mw[26] = {
  2.8013999999999999e+01, // N2
  2.0160000000000000e+00, // H2
  1.0080000000000000e+00, // H
  3.1998000000000001e+01, // O2
  1.5999000000000001e+01, // O
  1.8015000000000001e+01, // H2O
  1.7007000000000001e+01, // OH
  3.4014000000000003e+01, // H2O2
  3.3006000000000000e+01, // HO2
  2.8009999999999998e+01, // CO
  4.4009000000000000e+01, // CO2
  1.6042999999999999e+01, // CH4
  1.5035000000000000e+01, // CH3
  1.4026999999999999e+01, // CH2
  1.4026999999999999e+01, // CH2(S)
  1.2010999999999999e+01, // C
  1.3018999999999998e+01, // CH
  3.2042000000000002e+01, // CH3OH
  3.1033999999999999e+01, // CH3O
  3.1033999999999999e+01, // CH2OH
  3.0026000000000000e+01, // CH2O
  2.9018000000000001e+01, // HCO
  2.9017451420091128e+01, // CHO+
  1.9022451420091127e+01, // H3O+
  3.1998548579908874e+01, // O2-
  5.4857990887282834e-04, // E
};

//  inverse molecular weights
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
get_imw(amrex::Real* imw_new)
{
  imw_new[0] = 3.5696437495537948e-02;  // N2
  imw_new[1] = 4.9603174603174605e-01;  // H2
  imw_new[2] = 9.9206349206349209e-01;  // H
  imw_new[3] = 3.1251953247077940e-02;  // O2
  imw_new[4] = 6.2503906494155881e-02;  // O
  imw_new[5] = 5.5509297807382736e-02;  // H2O
  imw_new[6] = 5.8799317927912029e-02;  // OH
  imw_new[7] = 2.9399658963956014e-02;  // H2O2
  imw_new[8] = 3.0297521662727988e-02;  // HO2
  imw_new[9] = 3.5701535166012141e-02;  // CO
  imw_new[10] = 2.2722624917630486e-02; // CO2
  imw_new[11] = 6.2332481456086768e-02; // CH4
  imw_new[12] = 6.6511473229132020e-02; // CH3
  imw_new[13] = 7.1291081485706148e-02; // CH2
  imw_new[14] = 7.1291081485706148e-02; // CH2(S)
  imw_new[15] = 8.3257014403463503e-02; // C
  imw_new[16] = 7.6810814962746762e-02; // CH
  imw_new[17] = 3.1209038137444602e-02; // CH3OH
  imw_new[18] = 3.2222723464587225e-02; // CH3O
  imw_new[19] = 3.2222723464587225e-02; // CH2OH
  imw_new[20] = 3.3304469459801506e-02; // CH2O
  imw_new[21] = 3.4461368805568958e-02; // HCO
  imw_new[22] = 3.4462020303672125e-02; // CHO+
  imw_new[23] = 5.2569460050970102e-02; // H3O+
  imw_new[24] = 3.1251417466724607e-02; // O2-
  imw_new[25] = 1.8228884868472639e+03; // E
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
  mw_new[0] = 2.8013999999999999e+01;  // N2
  mw_new[1] = 2.0160000000000000e+00;  // H2
  mw_new[2] = 1.0080000000000000e+00;  // H
  mw_new[3] = 3.1998000000000001e+01;  // O2
  mw_new[4] = 1.5999000000000001e+01;  // O
  mw_new[5] = 1.8015000000000001e+01;  // H2O
  mw_new[6] = 1.7007000000000001e+01;  // OH
  mw_new[7] = 3.4014000000000003e+01;  // H2O2
  mw_new[8] = 3.3006000000000000e+01;  // HO2
  mw_new[9] = 2.8009999999999998e+01;  // CO
  mw_new[10] = 4.4009000000000000e+01; // CO2
  mw_new[11] = 1.6042999999999999e+01; // CH4
  mw_new[12] = 1.5035000000000000e+01; // CH3
  mw_new[13] = 1.4026999999999999e+01; // CH2
  mw_new[14] = 1.4026999999999999e+01; // CH2(S)
  mw_new[15] = 1.2010999999999999e+01; // C
  mw_new[16] = 1.3018999999999998e+01; // CH
  mw_new[17] = 3.2042000000000002e+01; // CH3OH
  mw_new[18] = 3.1033999999999999e+01; // CH3O
  mw_new[19] = 3.1033999999999999e+01; // CH2OH
  mw_new[20] = 3.0026000000000000e+01; // CH2O
  mw_new[21] = 2.9018000000000001e+01; // HCO
  mw_new[22] = 2.9017451420091128e+01; // CHO+
  mw_new[23] = 1.9022451420091127e+01; // H3O+
  mw_new[24] = 3.1998548579908874e+01; // O2-
  mw_new[25] = 5.4857990887282834e-04; // E
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    species[1] = +1.642044380000000e+00 + 5.495292740000000e-03 * T -
                 1.271636340000000e-05 * T2 + 1.287491730000000e-08 * T3 -
                 4.700277490000000e-12 * T4;
    // species 3: O2
    species[3] = +2.744039210000000e+00 - 2.797401470000000e-03 * T +
                 9.801225580000000e-06 * T2 - 1.032596430000000e-08 * T3 +
                 3.799312470000000e-12 * T4;
    // species 8: HO2
    species[8] = +2.619942990000000e+00 + 1.058057050000000e-03 * T +
                 5.066789410000000e-06 * T2 - 6.338007620000000e-09 * T3 +
                 2.415972810000000e-12 * T4;
    // species 11: CH4
    species[11] = +4.239673350000000e+00 - 1.468351230000000e-02 * T +
                  5.297327110000000e-05 * T2 - 5.416688220000000e-08 * T3 +
                  1.963185660000000e-11 * T4;
    // species 15: C
    species[15] = +1.544951920000000e+00 - 2.477252810000000e-04 * T +
                  5.480182770000000e-07 * T2 - 5.485512500000000e-10 * T3 +
                  2.041173310000000e-13 * T4;
    // species 20: CH2O
    species[20] = +3.326212960000000e+00 - 7.011518530000000e-03 * T +
                  3.151769610000000e-05 * T2 - 3.364786380000000e-08 * T3 +
                  1.234540230000000e-11 * T4;
  } else {
    // species 1: H2
    species[1] = +2.781998810000000e+00 - 1.018732590000000e-03 * T +
                 1.242262330000000e-06 * T2 - 4.190118980000000e-10 * T3 +
                 4.755437930000000e-14 * T4;
    // species 3: O2
    species[3] = +1.820124080000000e+00 + 2.482113570000000e-03 * T -
                 1.512020940000000e-06 * T2 + 4.485562010000000e-10 * T3 -
                 4.873056680000000e-14 * T4;
    // species 8: HO2
    species[8] = +2.023918890000000e+00 + 4.463909070000000e-03 * T -
                 2.231464920000000e-06 * T2 + 6.127107990000001e-10 * T3 -
                 6.642662370000000e-14 * T4;
    // species 11: CH4
    species[11] = -4.946535949999999e-01 + 1.236978450000000e-02 * T -
                  4.998079220000000e-06 * T2 + 1.043927650000000e-09 * T3 -
                  8.628974160000000e-14 * T4;
    // species 15: C
    species[15] = +1.494725310000000e+00 + 3.928394760000000e-05 * T -
                  6.700149800000000e-08 * T2 + 3.718186940000000e-11 * T3 -
                  5.073068850000000e-15 * T4;
    // species 20: CH2O
    species[20] = +3.333565199999999e-01 + 1.009051830000000e-02 * T -
                  5.129525620000000e-06 * T2 + 1.254252070000000e-09 * T3 -
                  1.196391090000000e-13 * T4;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    species[4] = +2.147992010000000e+00 - 3.111740650000000e-03 * T +
                 6.181378970000000e-06 * T2 - 5.638087980000000e-09 * T3 +
                 1.948660160000000e-12 * T4;
  } else {
    // species 4: O
    species[4] = +1.625491430000000e+00 - 2.089596440000000e-04 * T +
                 1.339185460000000e-07 * T2 - 3.858758960000000e-11 * T3 +
                 4.389186890000000e-15 * T4;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    species[21] = +3.034839790000000e+00 - 2.158368640000000e-03 * T +
                  1.182338750000000e-05 * T2 - 1.184594060000000e-08 * T3 +
                  4.005939540000000e-12 * T4;
  } else {
    // species 21: HCO
    species[21] = +1.600493180000000e+00 + 5.292782580000000e-03 * T -
                  2.691842110000000e-06 * T2 + 7.213577980000000e-10 * T3 -
                  7.435214090000000e-14 * T4;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    species[2] = +1.499505440000000e+00 + 2.991640460000000e-06 * T -
                 5.927597590000000e-09 * T2 + 4.878101650000000e-12 * T3 -
                 1.455393200000000e-15 * T4;
  } else {
    // species 2: H
    species[2] = +1.500314930000000e+00 - 7.734068280000000e-07 * T +
                 6.393453460000000e-10 * T2 - 2.125517910000000e-13 * T3 +
                 2.444791910000000e-17 * T4;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    species[9] = +2.757238910000000e+00 - 2.144652410000000e-03 * T +
                 5.420790050000000e-06 * T2 - 4.170259630000000e-09 * T3 +
                 1.119011270000000e-12 * T4;
  } else {
    // species 9: CO
    species[9] = +1.792553810000000e+00 + 1.874868860000000e-03 * T -
                 8.597119260000000e-07 * T2 + 1.912000700000000e-10 * T3 -
                 1.678552860000000e-14 * T4;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    species[14] = +3.181854340000000e+00 - 2.211343140000000e-03 * T +
                  7.715275410000000e-06 * T2 - 5.959503810000000e-09 * T3 +
                  1.583146280000000e-12 * T4;
  } else {
    // species 14: CH2(S)
    species[14] = +1.759342990000000e+00 + 3.654683070000000e-03 * T -
                  1.355899140000000e-06 * T2 + 2.749804110000000e-10 * T3 -
                  2.367954720000000e-14 * T4;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    species[0] = +2.731006820000000e+00 - 1.831597300000000e-03 * T +
                 4.323246620000000e-06 * T2 - 3.043781510000000e-09 * T3 +
                 7.460715620000000e-13 * T4;
  } else {
    // species 0: N2
    species[0] = +1.811660730000000e+00 + 1.670673530000000e-03 * T -
                 6.799974280000000e-07 * T2 + 1.328813790000000e-10 * T3 -
                 1.027674420000000e-14 * T4;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    species[12] = +2.478293100000000e+00 + 3.547647730000000e-03 * T +
                  1.474084400000000e-06 * T2 - 1.943759550000000e-09 * T3 +
                  5.219212320000000e-13 * T4;
  } else {
    // species 12: CH3
    species[12] = +1.788051040000000e+00 + 6.152334770000000e-03 * T -
                  2.211793490000000e-06 * T2 + 3.744026480000000e-10 * T3 -
                  2.481513490000000e-14 * T4;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    species[19] = +1.348215790000000e+00 + 1.396001680000000e-02 * T -
                  1.086322060000000e-05 * T2 + 4.624984160000000e-09 * T3 -
                  8.084991620000000e-13 * T4;
  } else {
    // species 19: CH2OH
    species[19] = +4.045349500000000e+00 + 6.027270590000000e-03 * T -
                  2.113868210000000e-06 * T2 + 3.360859050000000e-10 * T3 -
                  2.009874820000000e-14 * T4;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    species[5] = +3.060611720000000e+00 - 8.658071890000000e-04 * T +
                 3.244095280000000e-06 * T2 - 1.802430790000000e-09 * T3 +
                 3.324832930000000e-13 * T4;
  } else {
    // species 5: H2O
    species[5] = +1.667770750000000e+00 + 3.057688490000000e-03 * T -
                 9.004424110000000e-07 * T2 + 1.433615520000000e-10 * T3 -
                 1.008578170000000e-14 * T4;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    species[10] = +1.316843470000000e+00 + 9.227550360000000e-03 * T -
                  7.756540929999999e-06 * T2 + 3.282253600000000e-09 * T3 -
                  5.487224820000000e-13 * T4;
  } else {
    // species 10: CO2
    species[10] = +3.708764680000000e+00 + 2.629147040000000e-03 * T -
                  9.306064620000000e-07 * T2 + 1.438929200000000e-10 * T3 -
                  7.625814139999999e-15 * T4;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    species[16] = +2.772643320000000e+00 - 1.585473500000000e-03 * T +
                  2.835122380000000e-06 * T2 - 1.361460580000000e-09 * T3 +
                  2.239953320000000e-13 * T4;
  } else {
    // species 16: CH
    species[16] = +1.279901280000000e+00 + 2.169852380000000e-03 * T -
                  7.076378850000000e-07 * T2 + 1.239734940000000e-10 * T3 -
                  9.563485150000000e-15 * T4;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    species[6] = +2.913546310000000e+00 - 1.662759260000000e-03 * T +
                 2.309200290000000e-06 * T2 - 1.023595080000000e-09 * T3 +
                 1.588296290000000e-13 * T4;
  } else {
    // species 6: OH
    species[6] = +1.498673690000000e+00 + 1.666352790000000e-03 * T -
                 6.282515160000000e-07 * T2 + 1.283468060000000e-10 * T3 -
                 1.057358940000000e-14 * T4;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    species[18] = -1.103390150000000e-01 + 1.701197670000000e-02 * T -
                  1.138073510000000e-05 * T2 + 3.882809280000000e-09 * T3 -
                  5.328414790000000e-13 * T4;
  } else {
    // species 18: CH3O
    species[18] = +4.722380620000000e+00 + 5.902276370000000e-03 * T -
                  1.803407200000000e-06 * T2 + 2.133350090000000e-10 * T3 -
                  5.618164090000000e-15 * T4;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    species[7] = +2.347742240000000e+00 + 7.050054370000000e-03 * T -
                 3.845220060000000e-06 * T2 + 1.167206610000000e-09 * T3 -
                 1.476181050000000e-13 * T4;
    // species 13: CH2
    species[13] = +2.764894600000000e+00 + 1.438391910000000e-03 * T +
                  4.755830770000000e-07 * T2 - 4.317885910000000e-10 * T3 +
                  7.582928740000000e-14 * T4;
    // species 17: CH3OH
    species[17] = -1.526695210000000e-01 + 1.630869040000000e-02 * T -
                  8.483449609999999e-06 * T2 + 2.293043410000000e-09 * T3 -
                  2.599520130000000e-13 * T4;
  } else {
    // species 7: H2O2
    species[7] = +3.768696390000000e+00 + 3.892378480000000e-03 * T -
                 1.213823490000000e-06 * T2 + 1.926152850000000e-10 * T3 -
                 1.225819900000000e-14 * T4;
    // species 13: CH2
    species[13] = +1.812729720000000e+00 + 3.554313880000000e-03 * T -
                  1.287685230000000e-06 * T2 + 2.212737440000000e-10 * T3 -
                  1.487381470000000e-14 * T4;
    // species 17: CH3OH
    species[17] = +1.717015300000000e+00 + 1.215383530000000e-02 * T -
                  5.021070310000000e-06 * T2 + 1.010680700000000e-09 * T3 -
                  8.184608230000000e-14 * T4;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    species[24] = +2.102171800000000e+00 + 2.798087500000000e-03 * T -
                  2.265112600000000e-06 * T2 + 8.691651700000000e-10 * T3 -
                  1.272188400000000e-13 * T4;
  } else {
    // species 24: O2-
    species[24] = +3.259286700000000e+00 + 2.246807200000000e-04 * T -
                  5.139795500000000e-08 * T2 + 7.354597800000001e-12 * T3 -
                  3.855865200000000e-16 * T4;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    species[22] = +1.876276800000000e+00 + 4.948602000000000e-03 * T -
                  2.308530600000000e-06 * T2 + 5.112175200000000e-10 * T3 -
                  4.352063100000000e-14 * T4;
  } else {
    // species 22: CHO+
    species[22] = +5.915726500000000e+00 + 1.663110700000000e-04 * T -
                  1.847634300000000e-08 * T2 + 9.085642900000000e-13 * T3 -
                  1.645444100000000e-17 * T4;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    species[23] = +1.688523200000000e+00 + 5.445365400000000e-03 * T -
                  1.705556600000000e-06 * T2 + 2.483329300000000e-10 * T3 -
                  1.376944000000000e-14 * T4;
  } else {
    // species 23: H3O+
    species[23] = +7.950880300000000e+00 + 2.528166800000000e-04 * T -
                  2.459439700000000e-08 * T2 + 1.087831800000000e-12 * T3 -
                  1.809083700000000e-17 * T4;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    species[25] = +1.500000000000000e+00 - 8.150260800000000e-18 * T +
                  1.302986700000000e-21 * T2 - 7.264833300000000e-26 * T3 +
                  1.306624700000000e-30 * T4;
  } else {
    // species 25: E
    species[25] = +1.500000000000000e+00;
  }
}

// compute Cp/R at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
cp_R(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;
  const amrex::Real T4 = T * T3;

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    species[1] = +2.642044380000000e+00 + 5.495292740000000e-03 * T -
                 1.271636340000000e-05 * T2 + 1.287491730000000e-08 * T3 -
                 4.700277490000000e-12 * T4;
    // species 3: O2
    species[3] = +3.744039210000000e+00 - 2.797401470000000e-03 * T +
                 9.801225580000000e-06 * T2 - 1.032596430000000e-08 * T3 +
                 3.799312470000000e-12 * T4;
    // species 8: HO2
    species[8] = +3.619942990000000e+00 + 1.058057050000000e-03 * T +
                 5.066789410000000e-06 * T2 - 6.338007620000000e-09 * T3 +
                 2.415972810000000e-12 * T4;
    // species 11: CH4
    species[11] = +5.239673350000000e+00 - 1.468351230000000e-02 * T +
                  5.297327110000000e-05 * T2 - 5.416688220000000e-08 * T3 +
                  1.963185660000000e-11 * T4;
    // species 15: C
    species[15] = +2.544951920000000e+00 - 2.477252810000000e-04 * T +
                  5.480182770000000e-07 * T2 - 5.485512500000000e-10 * T3 +
                  2.041173310000000e-13 * T4;
    // species 20: CH2O
    species[20] = +4.326212960000000e+00 - 7.011518530000000e-03 * T +
                  3.151769610000000e-05 * T2 - 3.364786380000000e-08 * T3 +
                  1.234540230000000e-11 * T4;
  } else {
    // species 1: H2
    species[1] = +3.781998810000000e+00 - 1.018732590000000e-03 * T +
                 1.242262330000000e-06 * T2 - 4.190118980000000e-10 * T3 +
                 4.755437930000000e-14 * T4;
    // species 3: O2
    species[3] = +2.820124080000000e+00 + 2.482113570000000e-03 * T -
                 1.512020940000000e-06 * T2 + 4.485562010000000e-10 * T3 -
                 4.873056680000000e-14 * T4;
    // species 8: HO2
    species[8] = +3.023918890000000e+00 + 4.463909070000000e-03 * T -
                 2.231464920000000e-06 * T2 + 6.127107990000001e-10 * T3 -
                 6.642662370000000e-14 * T4;
    // species 11: CH4
    species[11] = +5.053464050000001e-01 + 1.236978450000000e-02 * T -
                  4.998079220000000e-06 * T2 + 1.043927650000000e-09 * T3 -
                  8.628974160000000e-14 * T4;
    // species 15: C
    species[15] = +2.494725310000000e+00 + 3.928394760000000e-05 * T -
                  6.700149800000000e-08 * T2 + 3.718186940000000e-11 * T3 -
                  5.073068850000000e-15 * T4;
    // species 20: CH2O
    species[20] = +1.333356520000000e+00 + 1.009051830000000e-02 * T -
                  5.129525620000000e-06 * T2 + 1.254252070000000e-09 * T3 -
                  1.196391090000000e-13 * T4;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    species[4] = +3.147992010000000e+00 - 3.111740650000000e-03 * T +
                 6.181378970000000e-06 * T2 - 5.638087980000000e-09 * T3 +
                 1.948660160000000e-12 * T4;
  } else {
    // species 4: O
    species[4] = +2.625491430000000e+00 - 2.089596440000000e-04 * T +
                 1.339185460000000e-07 * T2 - 3.858758960000000e-11 * T3 +
                 4.389186890000000e-15 * T4;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    species[21] = +4.034839790000000e+00 - 2.158368640000000e-03 * T +
                  1.182338750000000e-05 * T2 - 1.184594060000000e-08 * T3 +
                  4.005939540000000e-12 * T4;
  } else {
    // species 21: HCO
    species[21] = +2.600493180000000e+00 + 5.292782580000000e-03 * T -
                  2.691842110000000e-06 * T2 + 7.213577980000000e-10 * T3 -
                  7.435214090000000e-14 * T4;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    species[2] = +2.499505440000000e+00 + 2.991640460000000e-06 * T -
                 5.927597590000000e-09 * T2 + 4.878101650000000e-12 * T3 -
                 1.455393200000000e-15 * T4;
  } else {
    // species 2: H
    species[2] = +2.500314930000000e+00 - 7.734068280000000e-07 * T +
                 6.393453460000000e-10 * T2 - 2.125517910000000e-13 * T3 +
                 2.444791910000000e-17 * T4;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    species[9] = +3.757238910000000e+00 - 2.144652410000000e-03 * T +
                 5.420790050000000e-06 * T2 - 4.170259630000000e-09 * T3 +
                 1.119011270000000e-12 * T4;
  } else {
    // species 9: CO
    species[9] = +2.792553810000000e+00 + 1.874868860000000e-03 * T -
                 8.597119260000000e-07 * T2 + 1.912000700000000e-10 * T3 -
                 1.678552860000000e-14 * T4;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    species[14] = +4.181854340000000e+00 - 2.211343140000000e-03 * T +
                  7.715275410000000e-06 * T2 - 5.959503810000000e-09 * T3 +
                  1.583146280000000e-12 * T4;
  } else {
    // species 14: CH2(S)
    species[14] = +2.759342990000000e+00 + 3.654683070000000e-03 * T -
                  1.355899140000000e-06 * T2 + 2.749804110000000e-10 * T3 -
                  2.367954720000000e-14 * T4;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    species[0] = +3.731006820000000e+00 - 1.831597300000000e-03 * T +
                 4.323246620000000e-06 * T2 - 3.043781510000000e-09 * T3 +
                 7.460715620000000e-13 * T4;
  } else {
    // species 0: N2
    species[0] = +2.811660730000000e+00 + 1.670673530000000e-03 * T -
                 6.799974280000000e-07 * T2 + 1.328813790000000e-10 * T3 -
                 1.027674420000000e-14 * T4;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    species[12] = +3.478293100000000e+00 + 3.547647730000000e-03 * T +
                  1.474084400000000e-06 * T2 - 1.943759550000000e-09 * T3 +
                  5.219212320000000e-13 * T4;
  } else {
    // species 12: CH3
    species[12] = +2.788051040000000e+00 + 6.152334770000000e-03 * T -
                  2.211793490000000e-06 * T2 + 3.744026480000000e-10 * T3 -
                  2.481513490000000e-14 * T4;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    species[19] = +2.348215790000000e+00 + 1.396001680000000e-02 * T -
                  1.086322060000000e-05 * T2 + 4.624984160000000e-09 * T3 -
                  8.084991620000000e-13 * T4;
  } else {
    // species 19: CH2OH
    species[19] = +5.045349500000000e+00 + 6.027270590000000e-03 * T -
                  2.113868210000000e-06 * T2 + 3.360859050000000e-10 * T3 -
                  2.009874820000000e-14 * T4;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    species[5] = +4.060611720000000e+00 - 8.658071890000000e-04 * T +
                 3.244095280000000e-06 * T2 - 1.802430790000000e-09 * T3 +
                 3.324832930000000e-13 * T4;
  } else {
    // species 5: H2O
    species[5] = +2.667770750000000e+00 + 3.057688490000000e-03 * T -
                 9.004424110000000e-07 * T2 + 1.433615520000000e-10 * T3 -
                 1.008578170000000e-14 * T4;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    species[10] = +2.316843470000000e+00 + 9.227550360000000e-03 * T -
                  7.756540929999999e-06 * T2 + 3.282253600000000e-09 * T3 -
                  5.487224820000000e-13 * T4;
  } else {
    // species 10: CO2
    species[10] = +4.708764680000000e+00 + 2.629147040000000e-03 * T -
                  9.306064620000000e-07 * T2 + 1.438929200000000e-10 * T3 -
                  7.625814139999999e-15 * T4;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    species[16] = +3.772643320000000e+00 - 1.585473500000000e-03 * T +
                  2.835122380000000e-06 * T2 - 1.361460580000000e-09 * T3 +
                  2.239953320000000e-13 * T4;
  } else {
    // species 16: CH
    species[16] = +2.279901280000000e+00 + 2.169852380000000e-03 * T -
                  7.076378850000000e-07 * T2 + 1.239734940000000e-10 * T3 -
                  9.563485150000000e-15 * T4;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    species[6] = +3.913546310000000e+00 - 1.662759260000000e-03 * T +
                 2.309200290000000e-06 * T2 - 1.023595080000000e-09 * T3 +
                 1.588296290000000e-13 * T4;
  } else {
    // species 6: OH
    species[6] = +2.498673690000000e+00 + 1.666352790000000e-03 * T -
                 6.282515160000000e-07 * T2 + 1.283468060000000e-10 * T3 -
                 1.057358940000000e-14 * T4;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    species[18] = +8.896609850000000e-01 + 1.701197670000000e-02 * T -
                  1.138073510000000e-05 * T2 + 3.882809280000000e-09 * T3 -
                  5.328414790000000e-13 * T4;
  } else {
    // species 18: CH3O
    species[18] = +5.722380620000000e+00 + 5.902276370000000e-03 * T -
                  1.803407200000000e-06 * T2 + 2.133350090000000e-10 * T3 -
                  5.618164090000000e-15 * T4;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    species[7] = +3.347742240000000e+00 + 7.050054370000000e-03 * T -
                 3.845220060000000e-06 * T2 + 1.167206610000000e-09 * T3 -
                 1.476181050000000e-13 * T4;
    // species 13: CH2
    species[13] = +3.764894600000000e+00 + 1.438391910000000e-03 * T +
                  4.755830770000000e-07 * T2 - 4.317885910000000e-10 * T3 +
                  7.582928740000000e-14 * T4;
    // species 17: CH3OH
    species[17] = +8.473304790000000e-01 + 1.630869040000000e-02 * T -
                  8.483449609999999e-06 * T2 + 2.293043410000000e-09 * T3 -
                  2.599520130000000e-13 * T4;
  } else {
    // species 7: H2O2
    species[7] = +4.768696390000000e+00 + 3.892378480000000e-03 * T -
                 1.213823490000000e-06 * T2 + 1.926152850000000e-10 * T3 -
                 1.225819900000000e-14 * T4;
    // species 13: CH2
    species[13] = +2.812729720000000e+00 + 3.554313880000000e-03 * T -
                  1.287685230000000e-06 * T2 + 2.212737440000000e-10 * T3 -
                  1.487381470000000e-14 * T4;
    // species 17: CH3OH
    species[17] = +2.717015300000000e+00 + 1.215383530000000e-02 * T -
                  5.021070310000000e-06 * T2 + 1.010680700000000e-09 * T3 -
                  8.184608230000000e-14 * T4;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    species[24] = +3.102171800000000e+00 + 2.798087500000000e-03 * T -
                  2.265112600000000e-06 * T2 + 8.691651700000000e-10 * T3 -
                  1.272188400000000e-13 * T4;
  } else {
    // species 24: O2-
    species[24] = +4.259286700000000e+00 + 2.246807200000000e-04 * T -
                  5.139795500000000e-08 * T2 + 7.354597800000001e-12 * T3 -
                  3.855865200000000e-16 * T4;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    species[22] = +2.876276800000000e+00 + 4.948602000000000e-03 * T -
                  2.308530600000000e-06 * T2 + 5.112175200000000e-10 * T3 -
                  4.352063100000000e-14 * T4;
  } else {
    // species 22: CHO+
    species[22] = +6.915726500000000e+00 + 1.663110700000000e-04 * T -
                  1.847634300000000e-08 * T2 + 9.085642900000000e-13 * T3 -
                  1.645444100000000e-17 * T4;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    species[23] = +2.688523200000000e+00 + 5.445365400000000e-03 * T -
                  1.705556600000000e-06 * T2 + 2.483329300000000e-10 * T3 -
                  1.376944000000000e-14 * T4;
  } else {
    // species 23: H3O+
    species[23] = +8.950880300000000e+00 + 2.528166800000000e-04 * T -
                  2.459439700000000e-08 * T2 + 1.087831800000000e-12 * T3 -
                  1.809083700000000e-17 * T4;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    species[25] = +2.500000000000000e+00 - 8.150260800000000e-18 * T +
                  1.302986700000000e-21 * T2 - 7.264833300000000e-26 * T3 +
                  1.306624700000000e-30 * T4;
  } else {
    // species 25: E
    species[25] = +2.500000000000000e+00;
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    species[1] = -9.432366140000000e+02 * invT + 3.154275482000000e+00 -
                 2.642044380000000e+00 * logT - 2.747646370000000e-03 * T +
                 2.119393900000000e-06 * T2 - 1.072909775000000e-09 * T3 +
                 2.350138745000000e-13 * T4;
    // species 3: O2
    species[3] = -1.060698270000000e+03 * invT - 7.728724999999992e-02 -
                 3.744039210000000e+00 * logT + 1.398700735000000e-03 * T -
                 1.633537596666667e-06 * T2 + 8.604970250000000e-10 * T3 -
                 1.899656235000000e-13 * T4;
    // species 8: HO2
    species[8] = +3.158982340000000e+02 * invT - 2.824171830000000e+00 -
                 3.619942990000000e+00 * logT - 5.290285250000000e-04 * T -
                 8.444649016666667e-07 * T2 + 5.281673016666667e-10 * T3 -
                 1.207986405000000e-13 * T4;
    // species 11: CH4
    species[11] = -1.025263080000000e+04 * invT + 1.021617083000000e+01 -
                  5.239673350000000e+00 * logT + 7.341756150000000e-03 * T -
                  8.828878516666666e-06 * T2 + 4.513906850000000e-09 * T3 -
                  9.815928299999999e-13 * T4;
    // species 15: C
    species[15] = +8.544341050000000e+04 * invT - 2.023790810000000e+00 -
                  2.544951920000000e+00 * logT + 1.238626405000000e-04 * T -
                  9.133637950000000e-08 * T2 + 4.571260416666667e-11 * T3 -
                  1.020586655000000e-14 * T4;
    // species 20: CH2O
    species[20] = -1.432701690000000e+04 * invT + 1.705923940000000e+00 -
                  4.326212960000000e+00 * logT + 3.505759265000000e-03 * T -
                  5.252949350000000e-06 * T2 + 2.803988650000000e-09 * T3 -
                  6.172701150000000e-13 * T4;
  } else {
    // species 1: H2
    species[1] = -1.102830230000000e+03 * invT + 9.387257909999999e+00 -
                 3.781998810000000e+00 * logT + 5.093662950000000e-04 * T -
                 2.070437216666667e-07 * T2 + 3.491765816666666e-11 * T3 -
                 2.377718965000000e-15 * T4;
    // species 3: O2
    species[3] = -9.313501480000000e+02 * invT - 5.129021440000001e+00 -
                 2.820124080000000e+00 * logT - 1.241056785000000e-03 * T +
                 2.520034900000000e-07 * T2 - 3.737968341666667e-11 * T3 +
                 2.436528340000000e-15 * T4;
    // species 8: HO2
    species[8] = +3.993416090000000e+02 * invT - 6.083080840000000e+00 -
                 3.023918890000000e+00 * logT - 2.231954535000000e-03 * T +
                 3.719108200000000e-07 * T2 - 5.105923325000000e-11 * T3 +
                 3.321331185000000e-15 * T4;
    // species 11: CH4
    species[11] = -9.589825010000000e+03 * invT - 1.566993109500000e+01 -
                  5.053464050000001e-01 * logT - 6.184892250000000e-03 * T +
                  8.330132033333333e-07 * T2 - 8.699397083333333e-11 * T3 +
                  4.314487080000000e-15 * T4;
    // species 15: C
    species[15] = +8.545044220000000e+04 * invT - 2.298417230000000e+00 -
                  2.494725310000000e+00 * logT - 1.964197380000000e-05 * T +
                  1.116691633333333e-08 * T2 - 3.098489116666667e-12 * T3 +
                  2.536534425000000e-16 * T4;
    // species 20: CH2O
    species[20] = -1.390801700000000e+04 * invT - 1.465825788000000e+01 -
                  1.333356520000000e+00 * logT - 5.045259150000000e-03 * T +
                  8.549209366666667e-07 * T2 - 1.045210058333333e-10 * T3 +
                  5.981955450000001e-15 * T4;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    species[4] = +2.913091180000000e+04 * invT + 1.013526520000000e+00 -
                 3.147992010000000e+00 * logT + 1.555870325000000e-03 * T -
                 1.030229828333333e-06 * T2 + 4.698406650000000e-10 * T3 -
                 9.743300800000001e-14 * T4;
  } else {
    // species 4: O
    species[4] = +2.920615190000000e+04 * invT - 1.858093760000000e+00 -
                 2.625491430000000e+00 * logT + 1.044798220000000e-04 * T -
                 2.231975766666666e-08 * T2 + 3.215632466666667e-12 * T3 -
                 2.194593445000000e-16 * T4;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    species[21] = +3.836363920000000e+03 * invT - 1.652479099999997e-01 -
                  4.034839790000000e+00 * logT + 1.079184320000000e-03 * T -
                  1.970564583333333e-06 * T2 + 9.871617166666666e-10 * T3 -
                  2.002969770000000e-13 * T4;
  } else {
    // species 21: HCO
    species[21] = +4.057253300000000e+03 * invT - 8.144600120000000e+00 -
                  2.600493180000000e+00 * logT - 2.646391290000000e-03 * T +
                  4.486403516666666e-07 * T2 - 6.011314983333334e-11 * T3 +
                  3.717607045000000e-15 * T4;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    species[2] = +2.547378660000000e+04 * invT + 2.944079458000000e+00 -
                 2.499505440000000e+00 * logT - 1.495820230000000e-06 * T +
                 9.879329316666667e-10 * T2 - 4.065084708333333e-13 * T3 +
                 7.276966000000000e-17 * T4;
  } else {
    // species 2: H
    species[2] = +2.547364740000000e+04 * invT + 2.948672158000000e+00 -
                 2.500314930000000e+00 * logT + 3.867034140000000e-07 * T -
                 1.065575576666667e-10 * T2 + 1.771264925000000e-14 * T3 -
                 1.222395955000000e-18 * T4;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    species[9] = -1.435755300000000e+04 * invT + 9.574709200000000e-01 -
                 3.757238910000000e+00 * logT + 1.072326205000000e-03 * T -
                 9.034650083333333e-07 * T2 + 3.475216358333333e-10 * T3 -
                 5.595056350000000e-14 * T4;
  } else {
    // species 9: CO
    species[9] = -1.417233350000000e+04 * invT - 4.621881790000000e+00 -
                 2.792553810000000e+00 * logT - 9.374344300000000e-04 * T +
                 1.432853210000000e-07 * T2 - 1.593333916666667e-11 * T3 +
                 8.392764300000001e-16 * T4;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    species[14] = +5.036694070000000e+04 * invT + 4.884856942000000e+00 -
                  4.181854340000000e+00 * logT + 1.105671570000000e-03 * T -
                  1.285879235000000e-06 * T2 + 4.966253174999999e-10 * T3 -
                  7.915731400000000e-14 * T4;
  } else {
    // species 14: CH2(S)
    species[14] = +5.064290790000000e+04 * invT - 3.357120840000000e+00 -
                  2.759342990000000e+00 * logT - 1.827341535000000e-03 * T +
                  2.259831900000000e-07 * T2 - 2.291503425000000e-11 * T3 +
                  1.183977360000000e-15 * T4;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    species[0] = -1.062874260000000e+03 * invT + 1.562794840000000e+00 -
                 3.731006820000000e+00 * logT + 9.157986500000000e-04 * T -
                 7.205411033333332e-07 * T2 + 2.536484591666667e-10 * T3 -
                 3.730357810000000e-14 * T4;
  } else {
    // species 0: N2
    species[0] = -8.698115790000001e+02 * invT - 3.836719770000000e+00 -
                 2.811660730000000e+00 * logT - 8.353367650000000e-04 * T +
                 1.133329046666667e-07 * T2 - 1.107344825000000e-11 * T3 +
                 5.138372100000000e-16 * T4;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    species[12] = +1.643995160000000e+04 * invT + 1.069533540000000e+00 -
                  3.478293100000000e+00 * logT - 1.773823865000000e-03 * T -
                  2.456807333333333e-07 * T2 + 1.619799625000000e-10 * T3 -
                  2.609606160000000e-14 * T4;
  } else {
    // species 12: CH3
    species[12] = +1.658628290000000e+04 * invT - 2.990947140000000e+00 -
                  2.788051040000000e+00 * logT - 3.076167385000000e-03 * T +
                  3.686322483333333e-07 * T2 - 3.120022066666666e-11 * T3 +
                  1.240756745000000e-15 * T4;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    species[19] = -3.302221060000000e+03 * invT - 9.917981910000000e+00 -
                  2.348215790000000e+00 * logT - 6.980008400000000e-03 * T +
                  1.810536766666667e-06 * T2 - 3.854153466666667e-10 * T3 +
                  4.042495810000000e-14 * T4;
  } else {
    // species 19: CH2OH
    species[19] = -4.035841430000000e+03 * invT + 6.620590230000000e+00 -
                  5.045349500000000e+00 * logT - 3.013635295000000e-03 * T +
                  3.523113683333333e-07 * T2 - 2.800715875000000e-11 * T3 +
                  1.004937410000000e-15 * T4;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    species[5] = -3.028313140000000e+04 * invT + 4.356762201000000e+00 -
                 4.060611720000000e+00 * logT + 4.329035945000000e-04 * T -
                 5.406825466666667e-07 * T2 + 1.502025658333333e-10 * T3 -
                 1.662416465000000e-14 * T4;
  } else {
    // species 5: H2O
    species[5] = -2.988756450000000e+04 * invT - 4.244140560000000e+00 -
                 2.667770750000000e+00 * logT - 1.528844245000000e-03 * T +
                 1.500737351666667e-07 * T2 - 1.194679600000000e-11 * T3 +
                 5.042890850000000e-16 * T4;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    species[10] = -4.836260670000000e+04 * invT - 7.761779929999999e+00 -
                  2.316843470000000e+00 * logT - 4.613775180000000e-03 * T +
                  1.292756821666666e-06 * T2 - 2.735211333333333e-10 * T3 +
                  2.743612410000000e-14 * T4;
  } else {
    // species 10: CO2
    species[10] = -4.905626390000000e+04 * invT + 7.058529200000000e+00 -
                  4.708764680000000e+00 * logT - 1.314573520000000e-03 * T +
                  1.551010770000000e-07 * T2 - 1.199107666666667e-11 * T3 +
                  3.812907069999999e-16 * T4;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    species[16] = +7.063124920000001e+04 * invT + 2.893235416000000e+00 -
                  3.772643320000000e+00 * logT + 7.927367500000000e-04 * T -
                  4.725203966666667e-07 * T2 + 1.134550483333333e-10 * T3 -
                  1.119976660000000e-14 * T4;
  } else {
    // species 16: CH
    species[16] = +7.110594120000000e+04 * invT - 6.493359330000000e+00 -
                  2.279901280000000e+00 * logT - 1.084926190000000e-03 * T +
                  1.179396475000000e-07 * T2 - 1.033112450000000e-11 * T3 +
                  4.781742575000000e-16 * T4;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    species[6] = +3.400050470000000e+03 * invT + 3.708071591000000e+00 -
                 3.913546310000000e+00 * logT + 8.313796300000000e-04 * T -
                 3.848667150000000e-07 * T2 + 8.529959000000000e-11 * T3 -
                 7.941481450000000e-15 * T4;
  } else {
    // species 6: OH
    species[6] = +3.881107160000000e+03 * invT - 5.283514930000001e+00 -
                 2.498673690000000e+00 * logT - 8.331763950000000e-04 * T +
                 1.047085860000000e-07 * T2 - 1.069556716666667e-11 * T3 +
                 5.286794700000000e-16 * T4;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    species[18] = +1.603161210000000e+03 * invT - 1.761045241500000e+01 -
                  8.896609850000000e-01 * logT - 8.505988350000000e-03 * T +
                  1.896789183333333e-06 * T2 - 3.235674400000000e-10 * T3 +
                  2.664207395000000e-14 * T4;
  } else {
    // species 18: CH3O
    species[18] = -7.862522250000001e+01 * invT + 1.321411739000000e+01 -
                  5.722380620000000e+00 * logT - 2.951138185000000e-03 * T +
                  3.005678666666667e-07 * T2 - 1.777791741666667e-11 * T3 +
                  2.809082045000000e-16 * T4;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    species[7] = -1.757847850000000e+04 * invT - 3.830946270000000e+00 -
                 3.347742240000000e+00 * logT - 3.525027185000000e-03 * T +
                 6.408700100000000e-07 * T2 - 9.726721749999999e-11 * T3 +
                 7.380905249999999e-15 * T4;
    // species 13: CH2
    species[13] = +4.586456990000000e+04 * invT + 2.275363070000000e+00 -
                  3.764894600000000e+00 * logT - 7.191959550000000e-04 * T -
                  7.926384616666667e-08 * T2 + 3.598238258333333e-11 * T3 -
                  3.791464370000000e-15 * T4;
    // species 17: CH3OH
    species[17] = -2.509625440000000e+04 * invT - 1.874599922100000e+01 -
                  8.473304790000000e-01 * logT - 8.154345200000000e-03 * T +
                  1.413908268333333e-06 * T2 - 1.910869508333333e-10 * T3 +
                  1.299760065000000e-14 * T4;
  } else {
    // species 7: H2O2
    species[7] = -1.809002200000000e+04 * invT + 5.280508167000000e+00 -
                 4.768696390000000e+00 * logT - 1.946189240000000e-03 * T +
                 2.023039150000000e-07 * T2 - 1.605127375000000e-11 * T3 +
                 6.129099500000000e-16 * T4;
    // species 13: CH2
    species[13] = +4.620734920000000e+04 * invT - 3.830116800000000e+00 -
                  2.812729720000000e+00 * logT - 1.777156940000000e-03 * T +
                  2.146142050000000e-07 * T2 - 1.843947866666667e-11 * T3 +
                  7.436907349999999e-16 * T4;
    // species 17: CH3OH
    species[17] = -2.576934090000000e+04 * invT - 6.757190100000001e+00 -
                  2.717015300000000e+00 * logT - 6.076917650000000e-03 * T +
                  8.368450516666667e-07 * T2 - 8.422339166666667e-11 * T3 +
                  4.092304115000000e-15 * T4;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    species[24] = -6.807479300000000e+03 * invT - 3.658730200000000e+00 -
                  3.102171800000000e+00 * logT - 1.399043750000000e-03 * T +
                  3.775187666666667e-07 * T2 - 7.243043083333333e-11 * T3 +
                  6.360942000000000e-15 * T4;
  } else {
    // species 24: O2-
    species[24] = -7.242625200000000e+03 * invT + 3.783289729999999e+00 -
                  4.259286700000000e+00 * logT - 1.123403600000000e-04 * T +
                  8.566325833333332e-09 * T2 - 6.128831500000000e-13 * T3 +
                  1.927932600000000e-17 * T4;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    species[22] = +9.913002200000000e+04 * invT - 3.625749800000000e+00 -
                  2.876276800000000e+00 * logT - 2.474301000000000e-03 * T +
                  3.847551000000000e-07 * T2 - 4.260146000000000e-11 * T3 +
                  2.176031550000000e-15 * T4;
  } else {
    // species 22: CHO+
    species[22] = +9.612933300000000e+04 * invT + 2.500523650000000e+01 -
                  6.915726500000000e+00 * logT - 8.315553500000000e-05 * T +
                  3.079390500000000e-09 * T2 - 7.571369083333333e-14 * T3 +
                  8.227220500000000e-19 * T4;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    species[23] = +7.089026800000000e+04 * invT - 3.672298500000000e+00 -
                  2.688523200000000e+00 * logT - 2.722682700000000e-03 * T +
                  2.842594333333333e-07 * T2 - 2.069441083333333e-11 * T3 +
                  6.884720000000001e-16 * T4;
  } else {
    // species 23: H3O+
    species[23] = +6.448074900000000e+04 * invT + 4.282419930000000e+01 -
                  8.950880300000000e+00 * logT - 1.264083400000000e-04 * T +
                  4.099066166666667e-09 * T2 - 9.065265000000000e-14 * T3 +
                  9.045418500000001e-19 * T4;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    species[25] = -7.453750000000000e+02 * invT + 1.423397500000000e+01 -
                  2.500000000000000e+00 * logT + 4.075130400000000e-18 * T -
                  2.171644500000000e-22 * T2 + 6.054027750000001e-27 * T3 -
                  6.533123499999999e-32 * T4;
  } else {
    // species 25: E
    species[25] = -7.453750000000000e+02 * invT + 1.423397500000000e+01 -
                  2.500000000000000e+00 * logT;
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    species[1] = -9.432366140000000e+02 * invT + 2.154275482000000e+00 -
                 2.642044380000000e+00 * logT - 2.747646370000000e-03 * T +
                 2.119393900000000e-06 * T2 - 1.072909775000000e-09 * T3 +
                 2.350138745000000e-13 * T4;
    // species 3: O2
    species[3] = -1.060698270000000e+03 * invT - 1.077287250000000e+00 -
                 3.744039210000000e+00 * logT + 1.398700735000000e-03 * T -
                 1.633537596666667e-06 * T2 + 8.604970250000000e-10 * T3 -
                 1.899656235000000e-13 * T4;
    // species 8: HO2
    species[8] = +3.158982340000000e+02 * invT - 3.824171830000000e+00 -
                 3.619942990000000e+00 * logT - 5.290285250000000e-04 * T -
                 8.444649016666667e-07 * T2 + 5.281673016666667e-10 * T3 -
                 1.207986405000000e-13 * T4;
    // species 11: CH4
    species[11] = -1.025263080000000e+04 * invT + 9.216170829999999e+00 -
                  5.239673350000000e+00 * logT + 7.341756150000000e-03 * T -
                  8.828878516666666e-06 * T2 + 4.513906850000000e-09 * T3 -
                  9.815928299999999e-13 * T4;
    // species 15: C
    species[15] = +8.544341050000000e+04 * invT - 3.023790810000000e+00 -
                  2.544951920000000e+00 * logT + 1.238626405000000e-04 * T -
                  9.133637950000000e-08 * T2 + 4.571260416666667e-11 * T3 -
                  1.020586655000000e-14 * T4;
    // species 20: CH2O
    species[20] = -1.432701690000000e+04 * invT + 7.059239400000004e-01 -
                  4.326212960000000e+00 * logT + 3.505759265000000e-03 * T -
                  5.252949350000000e-06 * T2 + 2.803988650000000e-09 * T3 -
                  6.172701150000000e-13 * T4;
  } else {
    // species 1: H2
    species[1] = -1.102830230000000e+03 * invT + 8.387257909999999e+00 -
                 3.781998810000000e+00 * logT + 5.093662950000000e-04 * T -
                 2.070437216666667e-07 * T2 + 3.491765816666666e-11 * T3 -
                 2.377718965000000e-15 * T4;
    // species 3: O2
    species[3] = -9.313501480000000e+02 * invT - 6.129021440000001e+00 -
                 2.820124080000000e+00 * logT - 1.241056785000000e-03 * T +
                 2.520034900000000e-07 * T2 - 3.737968341666667e-11 * T3 +
                 2.436528340000000e-15 * T4;
    // species 8: HO2
    species[8] = +3.993416090000000e+02 * invT - 7.083080840000000e+00 -
                 3.023918890000000e+00 * logT - 2.231954535000000e-03 * T +
                 3.719108200000000e-07 * T2 - 5.105923325000000e-11 * T3 +
                 3.321331185000000e-15 * T4;
    // species 11: CH4
    species[11] = -9.589825010000000e+03 * invT - 1.666993109500000e+01 -
                  5.053464050000001e-01 * logT - 6.184892250000000e-03 * T +
                  8.330132033333333e-07 * T2 - 8.699397083333333e-11 * T3 +
                  4.314487080000000e-15 * T4;
    // species 15: C
    species[15] = +8.545044220000000e+04 * invT - 3.298417230000000e+00 -
                  2.494725310000000e+00 * logT - 1.964197380000000e-05 * T +
                  1.116691633333333e-08 * T2 - 3.098489116666667e-12 * T3 +
                  2.536534425000000e-16 * T4;
    // species 20: CH2O
    species[20] = -1.390801700000000e+04 * invT - 1.565825788000000e+01 -
                  1.333356520000000e+00 * logT - 5.045259150000000e-03 * T +
                  8.549209366666667e-07 * T2 - 1.045210058333333e-10 * T3 +
                  5.981955450000001e-15 * T4;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    species[4] = +2.913091180000000e+04 * invT + 1.352651999999965e-02 -
                 3.147992010000000e+00 * logT + 1.555870325000000e-03 * T -
                 1.030229828333333e-06 * T2 + 4.698406650000000e-10 * T3 -
                 9.743300800000001e-14 * T4;
  } else {
    // species 4: O
    species[4] = +2.920615190000000e+04 * invT - 2.858093760000000e+00 -
                 2.625491430000000e+00 * logT + 1.044798220000000e-04 * T -
                 2.231975766666666e-08 * T2 + 3.215632466666667e-12 * T3 -
                 2.194593445000000e-16 * T4;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    species[21] = +3.836363920000000e+03 * invT - 1.165247910000000e+00 -
                  4.034839790000000e+00 * logT + 1.079184320000000e-03 * T -
                  1.970564583333333e-06 * T2 + 9.871617166666666e-10 * T3 -
                  2.002969770000000e-13 * T4;
  } else {
    // species 21: HCO
    species[21] = +4.057253300000000e+03 * invT - 9.144600120000000e+00 -
                  2.600493180000000e+00 * logT - 2.646391290000000e-03 * T +
                  4.486403516666666e-07 * T2 - 6.011314983333334e-11 * T3 +
                  3.717607045000000e-15 * T4;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    species[2] = +2.547378660000000e+04 * invT + 1.944079458000000e+00 -
                 2.499505440000000e+00 * logT - 1.495820230000000e-06 * T +
                 9.879329316666667e-10 * T2 - 4.065084708333333e-13 * T3 +
                 7.276966000000000e-17 * T4;
  } else {
    // species 2: H
    species[2] = +2.547364740000000e+04 * invT + 1.948672158000000e+00 -
                 2.500314930000000e+00 * logT + 3.867034140000000e-07 * T -
                 1.065575576666667e-10 * T2 + 1.771264925000000e-14 * T3 -
                 1.222395955000000e-18 * T4;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    species[9] = -1.435755300000000e+04 * invT - 4.252908000000000e-02 -
                 3.757238910000000e+00 * logT + 1.072326205000000e-03 * T -
                 9.034650083333333e-07 * T2 + 3.475216358333333e-10 * T3 -
                 5.595056350000000e-14 * T4;
  } else {
    // species 9: CO
    species[9] = -1.417233350000000e+04 * invT - 5.621881790000000e+00 -
                 2.792553810000000e+00 * logT - 9.374344300000000e-04 * T +
                 1.432853210000000e-07 * T2 - 1.593333916666667e-11 * T3 +
                 8.392764300000001e-16 * T4;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    species[14] = +5.036694070000000e+04 * invT + 3.884856942000000e+00 -
                  4.181854340000000e+00 * logT + 1.105671570000000e-03 * T -
                  1.285879235000000e-06 * T2 + 4.966253174999999e-10 * T3 -
                  7.915731400000000e-14 * T4;
  } else {
    // species 14: CH2(S)
    species[14] = +5.064290790000000e+04 * invT - 4.357120840000000e+00 -
                  2.759342990000000e+00 * logT - 1.827341535000000e-03 * T +
                  2.259831900000000e-07 * T2 - 2.291503425000000e-11 * T3 +
                  1.183977360000000e-15 * T4;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    species[0] = -1.062874260000000e+03 * invT + 5.627948400000000e-01 -
                 3.731006820000000e+00 * logT + 9.157986500000000e-04 * T -
                 7.205411033333332e-07 * T2 + 2.536484591666667e-10 * T3 -
                 3.730357810000000e-14 * T4;
  } else {
    // species 0: N2
    species[0] = -8.698115790000001e+02 * invT - 4.836719770000000e+00 -
                 2.811660730000000e+00 * logT - 8.353367650000000e-04 * T +
                 1.133329046666667e-07 * T2 - 1.107344825000000e-11 * T3 +
                 5.138372100000000e-16 * T4;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    species[12] = +1.643995160000000e+04 * invT + 6.953354000000012e-02 -
                  3.478293100000000e+00 * logT - 1.773823865000000e-03 * T -
                  2.456807333333333e-07 * T2 + 1.619799625000000e-10 * T3 -
                  2.609606160000000e-14 * T4;
  } else {
    // species 12: CH3
    species[12] = +1.658628290000000e+04 * invT - 3.990947140000000e+00 -
                  2.788051040000000e+00 * logT - 3.076167385000000e-03 * T +
                  3.686322483333333e-07 * T2 - 3.120022066666666e-11 * T3 +
                  1.240756745000000e-15 * T4;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    species[19] = -3.302221060000000e+03 * invT - 1.091798191000000e+01 -
                  2.348215790000000e+00 * logT - 6.980008400000000e-03 * T +
                  1.810536766666667e-06 * T2 - 3.854153466666667e-10 * T3 +
                  4.042495810000000e-14 * T4;
  } else {
    // species 19: CH2OH
    species[19] = -4.035841430000000e+03 * invT + 5.620590230000000e+00 -
                  5.045349500000000e+00 * logT - 3.013635295000000e-03 * T +
                  3.523113683333333e-07 * T2 - 2.800715875000000e-11 * T3 +
                  1.004937410000000e-15 * T4;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    species[5] = -3.028313140000000e+04 * invT + 3.356762201000000e+00 -
                 4.060611720000000e+00 * logT + 4.329035945000000e-04 * T -
                 5.406825466666667e-07 * T2 + 1.502025658333333e-10 * T3 -
                 1.662416465000000e-14 * T4;
  } else {
    // species 5: H2O
    species[5] = -2.988756450000000e+04 * invT - 5.244140560000000e+00 -
                 2.667770750000000e+00 * logT - 1.528844245000000e-03 * T +
                 1.500737351666667e-07 * T2 - 1.194679600000000e-11 * T3 +
                 5.042890850000000e-16 * T4;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    species[10] = -4.836260670000000e+04 * invT - 8.761779929999999e+00 -
                  2.316843470000000e+00 * logT - 4.613775180000000e-03 * T +
                  1.292756821666666e-06 * T2 - 2.735211333333333e-10 * T3 +
                  2.743612410000000e-14 * T4;
  } else {
    // species 10: CO2
    species[10] = -4.905626390000000e+04 * invT + 6.058529200000000e+00 -
                  4.708764680000000e+00 * logT - 1.314573520000000e-03 * T +
                  1.551010770000000e-07 * T2 - 1.199107666666667e-11 * T3 +
                  3.812907069999999e-16 * T4;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    species[16] = +7.063124920000001e+04 * invT + 1.893235416000000e+00 -
                  3.772643320000000e+00 * logT + 7.927367500000000e-04 * T -
                  4.725203966666667e-07 * T2 + 1.134550483333333e-10 * T3 -
                  1.119976660000000e-14 * T4;
  } else {
    // species 16: CH
    species[16] = +7.110594120000000e+04 * invT - 7.493359330000000e+00 -
                  2.279901280000000e+00 * logT - 1.084926190000000e-03 * T +
                  1.179396475000000e-07 * T2 - 1.033112450000000e-11 * T3 +
                  4.781742575000000e-16 * T4;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    species[6] = +3.400050470000000e+03 * invT + 2.708071591000000e+00 -
                 3.913546310000000e+00 * logT + 8.313796300000000e-04 * T -
                 3.848667150000000e-07 * T2 + 8.529959000000000e-11 * T3 -
                 7.941481450000000e-15 * T4;
  } else {
    // species 6: OH
    species[6] = +3.881107160000000e+03 * invT - 6.283514930000001e+00 -
                 2.498673690000000e+00 * logT - 8.331763950000000e-04 * T +
                 1.047085860000000e-07 * T2 - 1.069556716666667e-11 * T3 +
                 5.286794700000000e-16 * T4;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    species[18] = +1.603161210000000e+03 * invT - 1.861045241500000e+01 -
                  8.896609850000000e-01 * logT - 8.505988350000000e-03 * T +
                  1.896789183333333e-06 * T2 - 3.235674400000000e-10 * T3 +
                  2.664207395000000e-14 * T4;
  } else {
    // species 18: CH3O
    species[18] = -7.862522250000001e+01 * invT + 1.221411739000000e+01 -
                  5.722380620000000e+00 * logT - 2.951138185000000e-03 * T +
                  3.005678666666667e-07 * T2 - 1.777791741666667e-11 * T3 +
                  2.809082045000000e-16 * T4;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    species[7] = -1.757847850000000e+04 * invT - 4.830946270000000e+00 -
                 3.347742240000000e+00 * logT - 3.525027185000000e-03 * T +
                 6.408700100000000e-07 * T2 - 9.726721749999999e-11 * T3 +
                 7.380905249999999e-15 * T4;
    // species 13: CH2
    species[13] = +4.586456990000000e+04 * invT + 1.275363070000000e+00 -
                  3.764894600000000e+00 * logT - 7.191959550000000e-04 * T -
                  7.926384616666667e-08 * T2 + 3.598238258333333e-11 * T3 -
                  3.791464370000000e-15 * T4;
    // species 17: CH3OH
    species[17] = -2.509625440000000e+04 * invT - 1.974599922100000e+01 -
                  8.473304790000000e-01 * logT - 8.154345200000000e-03 * T +
                  1.413908268333333e-06 * T2 - 1.910869508333333e-10 * T3 +
                  1.299760065000000e-14 * T4;
  } else {
    // species 7: H2O2
    species[7] = -1.809002200000000e+04 * invT + 4.280508167000000e+00 -
                 4.768696390000000e+00 * logT - 1.946189240000000e-03 * T +
                 2.023039150000000e-07 * T2 - 1.605127375000000e-11 * T3 +
                 6.129099500000000e-16 * T4;
    // species 13: CH2
    species[13] = +4.620734920000000e+04 * invT - 4.830116800000000e+00 -
                  2.812729720000000e+00 * logT - 1.777156940000000e-03 * T +
                  2.146142050000000e-07 * T2 - 1.843947866666667e-11 * T3 +
                  7.436907349999999e-16 * T4;
    // species 17: CH3OH
    species[17] = -2.576934090000000e+04 * invT - 7.757190100000001e+00 -
                  2.717015300000000e+00 * logT - 6.076917650000000e-03 * T +
                  8.368450516666667e-07 * T2 - 8.422339166666667e-11 * T3 +
                  4.092304115000000e-15 * T4;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    species[24] = -6.807479300000000e+03 * invT - 4.658730200000000e+00 -
                  3.102171800000000e+00 * logT - 1.399043750000000e-03 * T +
                  3.775187666666667e-07 * T2 - 7.243043083333333e-11 * T3 +
                  6.360942000000000e-15 * T4;
  } else {
    // species 24: O2-
    species[24] = -7.242625200000000e+03 * invT + 2.783289729999999e+00 -
                  4.259286700000000e+00 * logT - 1.123403600000000e-04 * T +
                  8.566325833333332e-09 * T2 - 6.128831500000000e-13 * T3 +
                  1.927932600000000e-17 * T4;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    species[22] = +9.913002200000000e+04 * invT - 4.625749799999999e+00 -
                  2.876276800000000e+00 * logT - 2.474301000000000e-03 * T +
                  3.847551000000000e-07 * T2 - 4.260146000000000e-11 * T3 +
                  2.176031550000000e-15 * T4;
  } else {
    // species 22: CHO+
    species[22] = +9.612933300000000e+04 * invT + 2.400523650000000e+01 -
                  6.915726500000000e+00 * logT - 8.315553500000000e-05 * T +
                  3.079390500000000e-09 * T2 - 7.571369083333333e-14 * T3 +
                  8.227220500000000e-19 * T4;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    species[23] = +7.089026800000000e+04 * invT - 4.672298500000000e+00 -
                  2.688523200000000e+00 * logT - 2.722682700000000e-03 * T +
                  2.842594333333333e-07 * T2 - 2.069441083333333e-11 * T3 +
                  6.884720000000001e-16 * T4;
  } else {
    // species 23: H3O+
    species[23] = +6.448074900000000e+04 * invT + 4.182419930000000e+01 -
                  8.950880300000000e+00 * logT - 1.264083400000000e-04 * T +
                  4.099066166666667e-09 * T2 - 9.065265000000000e-14 * T3 +
                  9.045418500000001e-19 * T4;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    species[25] = -7.453750000000000e+02 * invT + 1.323397500000000e+01 -
                  2.500000000000000e+00 * logT + 4.075130400000000e-18 * T -
                  2.171644500000000e-22 * T2 + 6.054027750000001e-27 * T3 -
                  6.533123499999999e-32 * T4;
  } else {
    // species 25: E
    species[25] = -7.453750000000000e+02 * invT + 1.323397500000000e+01 -
                  2.500000000000000e+00 * logT;
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    species[1] = +1.642044380000000e+00 + 2.747646370000000e-03 * T -
                 4.238787800000000e-06 * T2 + 3.218729325000000e-09 * T3 -
                 9.400554980000000e-13 * T4 - 9.432366140000000e+02 * invT;
    // species 3: O2
    species[3] = +2.744039210000000e+00 - 1.398700735000000e-03 * T +
                 3.267075193333333e-06 * T2 - 2.581491075000000e-09 * T3 +
                 7.598624940000000e-13 * T4 - 1.060698270000000e+03 * invT;
    // species 8: HO2
    species[8] = +2.619942990000000e+00 + 5.290285250000000e-04 * T +
                 1.688929803333333e-06 * T2 - 1.584501905000000e-09 * T3 +
                 4.831945620000000e-13 * T4 + 3.158982340000000e+02 * invT;
    // species 11: CH4
    species[11] = +4.239673350000000e+00 - 7.341756150000000e-03 * T +
                  1.765775703333333e-05 * T2 - 1.354172055000000e-08 * T3 +
                  3.926371320000000e-12 * T4 - 1.025263080000000e+04 * invT;
    // species 15: C
    species[15] = +1.544951920000000e+00 - 1.238626405000000e-04 * T +
                  1.826727590000000e-07 * T2 - 1.371378125000000e-10 * T3 +
                  4.082346620000000e-14 * T4 + 8.544341050000000e+04 * invT;
    // species 20: CH2O
    species[20] = +3.326212960000000e+00 - 3.505759265000000e-03 * T +
                  1.050589870000000e-05 * T2 - 8.411965949999999e-09 * T3 +
                  2.469080460000000e-12 * T4 - 1.432701690000000e+04 * invT;
  } else {
    // species 1: H2
    species[1] = +2.781998810000000e+00 - 5.093662950000000e-04 * T +
                 4.140874433333334e-07 * T2 - 1.047529745000000e-10 * T3 +
                 9.510875860000000e-15 * T4 - 1.102830230000000e+03 * invT;
    // species 3: O2
    species[3] = +1.820124080000000e+00 + 1.241056785000000e-03 * T -
                 5.040069800000000e-07 * T2 + 1.121390502500000e-10 * T3 -
                 9.746113360000000e-15 * T4 - 9.313501480000000e+02 * invT;
    // species 8: HO2
    species[8] = +2.023918890000000e+00 + 2.231954535000000e-03 * T -
                 7.438216400000000e-07 * T2 + 1.531776997500000e-10 * T3 -
                 1.328532474000000e-14 * T4 + 3.993416090000000e+02 * invT;
    // species 11: CH4
    species[11] = -4.946535949999999e-01 + 6.184892250000000e-03 * T -
                  1.666026406666667e-06 * T2 + 2.609819125000000e-10 * T3 -
                  1.725794832000000e-14 * T4 - 9.589825010000000e+03 * invT;
    // species 15: C
    species[15] = +1.494725310000000e+00 + 1.964197380000000e-05 * T -
                  2.233383266666667e-08 * T2 + 9.295467350000000e-12 * T3 -
                  1.014613770000000e-15 * T4 + 8.545044220000000e+04 * invT;
    // species 20: CH2O
    species[20] = +3.333565199999999e-01 + 5.045259150000000e-03 * T -
                  1.709841873333333e-06 * T2 + 3.135630175000000e-10 * T3 -
                  2.392782180000000e-14 * T4 - 1.390801700000000e+04 * invT;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    species[4] = +2.147992010000000e+00 - 1.555870325000000e-03 * T +
                 2.060459656666667e-06 * T2 - 1.409521995000000e-09 * T3 +
                 3.897320320000000e-13 * T4 + 2.913091180000000e+04 * invT;
  } else {
    // species 4: O
    species[4] = +1.625491430000000e+00 - 1.044798220000000e-04 * T +
                 4.463951533333333e-08 * T2 - 9.646897399999999e-12 * T3 +
                 8.778373779999999e-16 * T4 + 2.920615190000000e+04 * invT;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    species[21] = +3.034839790000000e+00 - 1.079184320000000e-03 * T +
                  3.941129166666667e-06 * T2 - 2.961485150000000e-09 * T3 +
                  8.011879080000000e-13 * T4 + 3.836363920000000e+03 * invT;
  } else {
    // species 21: HCO
    species[21] = +1.600493180000000e+00 + 2.646391290000000e-03 * T -
                  8.972807033333333e-07 * T2 + 1.803394495000000e-10 * T3 -
                  1.487042818000000e-14 * T4 + 4.057253300000000e+03 * invT;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    species[2] = +1.499505440000000e+00 + 1.495820230000000e-06 * T -
                 1.975865863333333e-09 * T2 + 1.219525412500000e-12 * T3 -
                 2.910786400000000e-16 * T4 + 2.547378660000000e+04 * invT;
  } else {
    // species 2: H
    species[2] = +1.500314930000000e+00 - 3.867034140000000e-07 * T +
                 2.131151153333333e-10 * T2 - 5.313794775000000e-14 * T3 +
                 4.889583820000000e-18 * T4 + 2.547364740000000e+04 * invT;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    species[9] = +2.757238910000000e+00 - 1.072326205000000e-03 * T +
                 1.806930016666667e-06 * T2 - 1.042564907500000e-09 * T3 +
                 2.238022540000000e-13 * T4 - 1.435755300000000e+04 * invT;
  } else {
    // species 9: CO
    species[9] = +1.792553810000000e+00 + 9.374344300000000e-04 * T -
                 2.865706420000000e-07 * T2 + 4.780001750000000e-11 * T3 -
                 3.357105720000000e-15 * T4 - 1.417233350000000e+04 * invT;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    species[14] = +3.181854340000000e+00 - 1.105671570000000e-03 * T +
                  2.571758470000000e-06 * T2 - 1.489875952500000e-09 * T3 +
                  3.166292560000000e-13 * T4 + 5.036694070000000e+04 * invT;
  } else {
    // species 14: CH2(S)
    species[14] = +1.759342990000000e+00 + 1.827341535000000e-03 * T -
                  4.519663800000000e-07 * T2 + 6.874510274999999e-11 * T3 -
                  4.735909440000000e-15 * T4 + 5.064290790000000e+04 * invT;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    species[0] = +2.731006820000000e+00 - 9.157986500000000e-04 * T +
                 1.441082206666666e-06 * T2 - 7.609453775000000e-10 * T3 +
                 1.492143124000000e-13 * T4 - 1.062874260000000e+03 * invT;
  } else {
    // species 0: N2
    species[0] = +1.811660730000000e+00 + 8.353367650000000e-04 * T -
                 2.266658093333333e-07 * T2 + 3.322034475000000e-11 * T3 -
                 2.055348840000000e-15 * T4 - 8.698115790000001e+02 * invT;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    species[12] = +2.478293100000000e+00 + 1.773823865000000e-03 * T +
                  4.913614666666666e-07 * T2 - 4.859398875000000e-10 * T3 +
                  1.043842464000000e-13 * T4 + 1.643995160000000e+04 * invT;
  } else {
    // species 12: CH3
    species[12] = +1.788051040000000e+00 + 3.076167385000000e-03 * T -
                  7.372644966666667e-07 * T2 + 9.360066200000000e-11 * T3 -
                  4.963026980000000e-15 * T4 + 1.658628290000000e+04 * invT;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    species[19] = +1.348215790000000e+00 + 6.980008400000000e-03 * T -
                  3.621073533333334e-06 * T2 + 1.156246040000000e-09 * T3 -
                  1.616998324000000e-13 * T4 - 3.302221060000000e+03 * invT;
  } else {
    // species 19: CH2OH
    species[19] = +4.045349500000000e+00 + 3.013635295000000e-03 * T -
                  7.046227366666666e-07 * T2 + 8.402147625000000e-11 * T3 -
                  4.019749640000000e-15 * T4 - 4.035841430000000e+03 * invT;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    species[5] = +3.060611720000000e+00 - 4.329035945000000e-04 * T +
                 1.081365093333333e-06 * T2 - 4.506076975000000e-10 * T3 +
                 6.649665860000001e-14 * T4 - 3.028313140000000e+04 * invT;
  } else {
    // species 5: H2O
    species[5] = +1.667770750000000e+00 + 1.528844245000000e-03 * T -
                 3.001474703333333e-07 * T2 + 3.584038800000000e-11 * T3 -
                 2.017156340000000e-15 * T4 - 2.988756450000000e+04 * invT;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    species[10] = +1.316843470000000e+00 + 4.613775180000000e-03 * T -
                  2.585513643333333e-06 * T2 + 8.205634000000000e-10 * T3 -
                  1.097444964000000e-13 * T4 - 4.836260670000000e+04 * invT;
  } else {
    // species 10: CO2
    species[10] = +3.708764680000000e+00 + 1.314573520000000e-03 * T -
                  3.102021540000000e-07 * T2 + 3.597323000000000e-11 * T3 -
                  1.525162828000000e-15 * T4 - 4.905626390000000e+04 * invT;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    species[16] = +2.772643320000000e+00 - 7.927367500000000e-04 * T +
                  9.450407933333333e-07 * T2 - 3.403651450000000e-10 * T3 +
                  4.479906640000000e-14 * T4 + 7.063124920000001e+04 * invT;
  } else {
    // species 16: CH
    species[16] = +1.279901280000000e+00 + 1.084926190000000e-03 * T -
                  2.358792950000000e-07 * T2 + 3.099337350000000e-11 * T3 -
                  1.912697030000000e-15 * T4 + 7.110594120000000e+04 * invT;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    species[6] = +2.913546310000000e+00 - 8.313796300000000e-04 * T +
                 7.697334299999999e-07 * T2 - 2.558987700000000e-10 * T3 +
                 3.176592580000000e-14 * T4 + 3.400050470000000e+03 * invT;
  } else {
    // species 6: OH
    species[6] = +1.498673690000000e+00 + 8.331763950000000e-04 * T -
                 2.094171720000000e-07 * T2 + 3.208670150000000e-11 * T3 -
                 2.114717880000000e-15 * T4 + 3.881107160000000e+03 * invT;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    species[18] = -1.103390150000000e-01 + 8.505988350000000e-03 * T -
                  3.793578366666667e-06 * T2 + 9.707023200000000e-10 * T3 -
                  1.065682958000000e-13 * T4 + 1.603161210000000e+03 * invT;
  } else {
    // species 18: CH3O
    species[18] = +4.722380620000000e+00 + 2.951138185000000e-03 * T -
                  6.011357333333334e-07 * T2 + 5.333375225000000e-11 * T3 -
                  1.123632818000000e-15 * T4 - 7.862522250000001e+01 * invT;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    species[7] = +2.347742240000000e+00 + 3.525027185000000e-03 * T -
                 1.281740020000000e-06 * T2 + 2.918016525000000e-10 * T3 -
                 2.952362099999999e-14 * T4 - 1.757847850000000e+04 * invT;
    // species 13: CH2
    species[13] = +2.764894600000000e+00 + 7.191959550000000e-04 * T +
                  1.585276923333333e-07 * T2 - 1.079471477500000e-10 * T3 +
                  1.516585748000000e-14 * T4 + 4.586456990000000e+04 * invT;
    // species 17: CH3OH
    species[17] = -1.526695210000000e-01 + 8.154345200000000e-03 * T -
                  2.827816536666666e-06 * T2 + 5.732608525000000e-10 * T3 -
                  5.199040260000000e-14 * T4 - 2.509625440000000e+04 * invT;
  } else {
    // species 7: H2O2
    species[7] = +3.768696390000000e+00 + 1.946189240000000e-03 * T -
                 4.046078300000000e-07 * T2 + 4.815382125000000e-11 * T3 -
                 2.451639800000000e-15 * T4 - 1.809002200000000e+04 * invT;
    // species 13: CH2
    species[13] = +1.812729720000000e+00 + 1.777156940000000e-03 * T -
                  4.292284100000000e-07 * T2 + 5.531843600000000e-11 * T3 -
                  2.974762940000000e-15 * T4 + 4.620734920000000e+04 * invT;
    // species 17: CH3OH
    species[17] = +1.717015300000000e+00 + 6.076917650000000e-03 * T -
                  1.673690103333333e-06 * T2 + 2.526701750000000e-10 * T3 -
                  1.636921646000000e-14 * T4 - 2.576934090000000e+04 * invT;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    species[24] = +2.102171800000000e+00 + 1.399043750000000e-03 * T -
                  7.550375333333334e-07 * T2 + 2.172912925000000e-10 * T3 -
                  2.544376800000000e-14 * T4 - 6.807479300000000e+03 * invT;
  } else {
    // species 24: O2-
    species[24] = +3.259286700000000e+00 + 1.123403600000000e-04 * T -
                  1.713265166666666e-08 * T2 + 1.838649450000000e-12 * T3 -
                  7.711730400000001e-17 * T4 - 7.242625200000000e+03 * invT;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    species[22] = +1.876276800000000e+00 + 2.474301000000000e-03 * T -
                  7.695102000000001e-07 * T2 + 1.278043800000000e-10 * T3 -
                  8.704126200000000e-15 * T4 + 9.913002200000000e+04 * invT;
  } else {
    // species 22: CHO+
    species[22] = +5.915726500000000e+00 + 8.315553500000000e-05 * T -
                  6.158781000000000e-09 * T2 + 2.271410725000000e-13 * T3 -
                  3.290888200000000e-18 * T4 + 9.612933300000000e+04 * invT;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    species[23] = +1.688523200000000e+00 + 2.722682700000000e-03 * T -
                  5.685188666666667e-07 * T2 + 6.208323250000000e-11 * T3 -
                  2.753888000000000e-15 * T4 + 7.089026800000000e+04 * invT;
  } else {
    // species 23: H3O+
    species[23] = +7.950880300000000e+00 + 1.264083400000000e-04 * T -
                  8.198132333333333e-09 * T2 + 2.719579500000000e-13 * T3 -
                  3.618167400000001e-18 * T4 + 6.448074900000000e+04 * invT;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    species[25] = +1.500000000000000e+00 - 4.075130400000000e-18 * T +
                  4.343289000000000e-22 * T2 - 1.816208325000000e-26 * T3 +
                  2.613249400000000e-31 * T4 - 7.453750000000000e+02 * invT;
  } else {
    // species 25: E
    species[25] = +1.500000000000000e+00 - 7.453750000000000e+02 * invT;
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    species[1] = +2.642044380000000e+00 + 2.747646370000000e-03 * T -
                 4.238787800000000e-06 * T2 + 3.218729325000000e-09 * T3 -
                 9.400554980000000e-13 * T4 - 9.432366140000000e+02 * invT;
    // species 3: O2
    species[3] = +3.744039210000000e+00 - 1.398700735000000e-03 * T +
                 3.267075193333333e-06 * T2 - 2.581491075000000e-09 * T3 +
                 7.598624940000000e-13 * T4 - 1.060698270000000e+03 * invT;
    // species 8: HO2
    species[8] = +3.619942990000000e+00 + 5.290285250000000e-04 * T +
                 1.688929803333333e-06 * T2 - 1.584501905000000e-09 * T3 +
                 4.831945620000000e-13 * T4 + 3.158982340000000e+02 * invT;
    // species 11: CH4
    species[11] = +5.239673350000000e+00 - 7.341756150000000e-03 * T +
                  1.765775703333333e-05 * T2 - 1.354172055000000e-08 * T3 +
                  3.926371320000000e-12 * T4 - 1.025263080000000e+04 * invT;
    // species 15: C
    species[15] = +2.544951920000000e+00 - 1.238626405000000e-04 * T +
                  1.826727590000000e-07 * T2 - 1.371378125000000e-10 * T3 +
                  4.082346620000000e-14 * T4 + 8.544341050000000e+04 * invT;
    // species 20: CH2O
    species[20] = +4.326212960000000e+00 - 3.505759265000000e-03 * T +
                  1.050589870000000e-05 * T2 - 8.411965949999999e-09 * T3 +
                  2.469080460000000e-12 * T4 - 1.432701690000000e+04 * invT;
  } else {
    // species 1: H2
    species[1] = +3.781998810000000e+00 - 5.093662950000000e-04 * T +
                 4.140874433333334e-07 * T2 - 1.047529745000000e-10 * T3 +
                 9.510875860000000e-15 * T4 - 1.102830230000000e+03 * invT;
    // species 3: O2
    species[3] = +2.820124080000000e+00 + 1.241056785000000e-03 * T -
                 5.040069800000000e-07 * T2 + 1.121390502500000e-10 * T3 -
                 9.746113360000000e-15 * T4 - 9.313501480000000e+02 * invT;
    // species 8: HO2
    species[8] = +3.023918890000000e+00 + 2.231954535000000e-03 * T -
                 7.438216400000000e-07 * T2 + 1.531776997500000e-10 * T3 -
                 1.328532474000000e-14 * T4 + 3.993416090000000e+02 * invT;
    // species 11: CH4
    species[11] = +5.053464050000001e-01 + 6.184892250000000e-03 * T -
                  1.666026406666667e-06 * T2 + 2.609819125000000e-10 * T3 -
                  1.725794832000000e-14 * T4 - 9.589825010000000e+03 * invT;
    // species 15: C
    species[15] = +2.494725310000000e+00 + 1.964197380000000e-05 * T -
                  2.233383266666667e-08 * T2 + 9.295467350000000e-12 * T3 -
                  1.014613770000000e-15 * T4 + 8.545044220000000e+04 * invT;
    // species 20: CH2O
    species[20] = +1.333356520000000e+00 + 5.045259150000000e-03 * T -
                  1.709841873333333e-06 * T2 + 3.135630175000000e-10 * T3 -
                  2.392782180000000e-14 * T4 - 1.390801700000000e+04 * invT;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    species[4] = +3.147992010000000e+00 - 1.555870325000000e-03 * T +
                 2.060459656666667e-06 * T2 - 1.409521995000000e-09 * T3 +
                 3.897320320000000e-13 * T4 + 2.913091180000000e+04 * invT;
  } else {
    // species 4: O
    species[4] = +2.625491430000000e+00 - 1.044798220000000e-04 * T +
                 4.463951533333333e-08 * T2 - 9.646897399999999e-12 * T3 +
                 8.778373779999999e-16 * T4 + 2.920615190000000e+04 * invT;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    species[21] = +4.034839790000000e+00 - 1.079184320000000e-03 * T +
                  3.941129166666667e-06 * T2 - 2.961485150000000e-09 * T3 +
                  8.011879080000000e-13 * T4 + 3.836363920000000e+03 * invT;
  } else {
    // species 21: HCO
    species[21] = +2.600493180000000e+00 + 2.646391290000000e-03 * T -
                  8.972807033333333e-07 * T2 + 1.803394495000000e-10 * T3 -
                  1.487042818000000e-14 * T4 + 4.057253300000000e+03 * invT;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    species[2] = +2.499505440000000e+00 + 1.495820230000000e-06 * T -
                 1.975865863333333e-09 * T2 + 1.219525412500000e-12 * T3 -
                 2.910786400000000e-16 * T4 + 2.547378660000000e+04 * invT;
  } else {
    // species 2: H
    species[2] = +2.500314930000000e+00 - 3.867034140000000e-07 * T +
                 2.131151153333333e-10 * T2 - 5.313794775000000e-14 * T3 +
                 4.889583820000000e-18 * T4 + 2.547364740000000e+04 * invT;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    species[9] = +3.757238910000000e+00 - 1.072326205000000e-03 * T +
                 1.806930016666667e-06 * T2 - 1.042564907500000e-09 * T3 +
                 2.238022540000000e-13 * T4 - 1.435755300000000e+04 * invT;
  } else {
    // species 9: CO
    species[9] = +2.792553810000000e+00 + 9.374344300000000e-04 * T -
                 2.865706420000000e-07 * T2 + 4.780001750000000e-11 * T3 -
                 3.357105720000000e-15 * T4 - 1.417233350000000e+04 * invT;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    species[14] = +4.181854340000000e+00 - 1.105671570000000e-03 * T +
                  2.571758470000000e-06 * T2 - 1.489875952500000e-09 * T3 +
                  3.166292560000000e-13 * T4 + 5.036694070000000e+04 * invT;
  } else {
    // species 14: CH2(S)
    species[14] = +2.759342990000000e+00 + 1.827341535000000e-03 * T -
                  4.519663800000000e-07 * T2 + 6.874510274999999e-11 * T3 -
                  4.735909440000000e-15 * T4 + 5.064290790000000e+04 * invT;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    species[0] = +3.731006820000000e+00 - 9.157986500000000e-04 * T +
                 1.441082206666666e-06 * T2 - 7.609453775000000e-10 * T3 +
                 1.492143124000000e-13 * T4 - 1.062874260000000e+03 * invT;
  } else {
    // species 0: N2
    species[0] = +2.811660730000000e+00 + 8.353367650000000e-04 * T -
                 2.266658093333333e-07 * T2 + 3.322034475000000e-11 * T3 -
                 2.055348840000000e-15 * T4 - 8.698115790000001e+02 * invT;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    species[12] = +3.478293100000000e+00 + 1.773823865000000e-03 * T +
                  4.913614666666666e-07 * T2 - 4.859398875000000e-10 * T3 +
                  1.043842464000000e-13 * T4 + 1.643995160000000e+04 * invT;
  } else {
    // species 12: CH3
    species[12] = +2.788051040000000e+00 + 3.076167385000000e-03 * T -
                  7.372644966666667e-07 * T2 + 9.360066200000000e-11 * T3 -
                  4.963026980000000e-15 * T4 + 1.658628290000000e+04 * invT;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    species[19] = +2.348215790000000e+00 + 6.980008400000000e-03 * T -
                  3.621073533333334e-06 * T2 + 1.156246040000000e-09 * T3 -
                  1.616998324000000e-13 * T4 - 3.302221060000000e+03 * invT;
  } else {
    // species 19: CH2OH
    species[19] = +5.045349500000000e+00 + 3.013635295000000e-03 * T -
                  7.046227366666666e-07 * T2 + 8.402147625000000e-11 * T3 -
                  4.019749640000000e-15 * T4 - 4.035841430000000e+03 * invT;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    species[5] = +4.060611720000000e+00 - 4.329035945000000e-04 * T +
                 1.081365093333333e-06 * T2 - 4.506076975000000e-10 * T3 +
                 6.649665860000001e-14 * T4 - 3.028313140000000e+04 * invT;
  } else {
    // species 5: H2O
    species[5] = +2.667770750000000e+00 + 1.528844245000000e-03 * T -
                 3.001474703333333e-07 * T2 + 3.584038800000000e-11 * T3 -
                 2.017156340000000e-15 * T4 - 2.988756450000000e+04 * invT;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    species[10] = +2.316843470000000e+00 + 4.613775180000000e-03 * T -
                  2.585513643333333e-06 * T2 + 8.205634000000000e-10 * T3 -
                  1.097444964000000e-13 * T4 - 4.836260670000000e+04 * invT;
  } else {
    // species 10: CO2
    species[10] = +4.708764680000000e+00 + 1.314573520000000e-03 * T -
                  3.102021540000000e-07 * T2 + 3.597323000000000e-11 * T3 -
                  1.525162828000000e-15 * T4 - 4.905626390000000e+04 * invT;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    species[16] = +3.772643320000000e+00 - 7.927367500000000e-04 * T +
                  9.450407933333333e-07 * T2 - 3.403651450000000e-10 * T3 +
                  4.479906640000000e-14 * T4 + 7.063124920000001e+04 * invT;
  } else {
    // species 16: CH
    species[16] = +2.279901280000000e+00 + 1.084926190000000e-03 * T -
                  2.358792950000000e-07 * T2 + 3.099337350000000e-11 * T3 -
                  1.912697030000000e-15 * T4 + 7.110594120000000e+04 * invT;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    species[6] = +3.913546310000000e+00 - 8.313796300000000e-04 * T +
                 7.697334299999999e-07 * T2 - 2.558987700000000e-10 * T3 +
                 3.176592580000000e-14 * T4 + 3.400050470000000e+03 * invT;
  } else {
    // species 6: OH
    species[6] = +2.498673690000000e+00 + 8.331763950000000e-04 * T -
                 2.094171720000000e-07 * T2 + 3.208670150000000e-11 * T3 -
                 2.114717880000000e-15 * T4 + 3.881107160000000e+03 * invT;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    species[18] = +8.896609850000000e-01 + 8.505988350000000e-03 * T -
                  3.793578366666667e-06 * T2 + 9.707023200000000e-10 * T3 -
                  1.065682958000000e-13 * T4 + 1.603161210000000e+03 * invT;
  } else {
    // species 18: CH3O
    species[18] = +5.722380620000000e+00 + 2.951138185000000e-03 * T -
                  6.011357333333334e-07 * T2 + 5.333375225000000e-11 * T3 -
                  1.123632818000000e-15 * T4 - 7.862522250000001e+01 * invT;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    species[7] = +3.347742240000000e+00 + 3.525027185000000e-03 * T -
                 1.281740020000000e-06 * T2 + 2.918016525000000e-10 * T3 -
                 2.952362099999999e-14 * T4 - 1.757847850000000e+04 * invT;
    // species 13: CH2
    species[13] = +3.764894600000000e+00 + 7.191959550000000e-04 * T +
                  1.585276923333333e-07 * T2 - 1.079471477500000e-10 * T3 +
                  1.516585748000000e-14 * T4 + 4.586456990000000e+04 * invT;
    // species 17: CH3OH
    species[17] = +8.473304790000000e-01 + 8.154345200000000e-03 * T -
                  2.827816536666666e-06 * T2 + 5.732608525000000e-10 * T3 -
                  5.199040260000000e-14 * T4 - 2.509625440000000e+04 * invT;
  } else {
    // species 7: H2O2
    species[7] = +4.768696390000000e+00 + 1.946189240000000e-03 * T -
                 4.046078300000000e-07 * T2 + 4.815382125000000e-11 * T3 -
                 2.451639800000000e-15 * T4 - 1.809002200000000e+04 * invT;
    // species 13: CH2
    species[13] = +2.812729720000000e+00 + 1.777156940000000e-03 * T -
                  4.292284100000000e-07 * T2 + 5.531843600000000e-11 * T3 -
                  2.974762940000000e-15 * T4 + 4.620734920000000e+04 * invT;
    // species 17: CH3OH
    species[17] = +2.717015300000000e+00 + 6.076917650000000e-03 * T -
                  1.673690103333333e-06 * T2 + 2.526701750000000e-10 * T3 -
                  1.636921646000000e-14 * T4 - 2.576934090000000e+04 * invT;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    species[24] = +3.102171800000000e+00 + 1.399043750000000e-03 * T -
                  7.550375333333334e-07 * T2 + 2.172912925000000e-10 * T3 -
                  2.544376800000000e-14 * T4 - 6.807479300000000e+03 * invT;
  } else {
    // species 24: O2-
    species[24] = +4.259286700000000e+00 + 1.123403600000000e-04 * T -
                  1.713265166666666e-08 * T2 + 1.838649450000000e-12 * T3 -
                  7.711730400000001e-17 * T4 - 7.242625200000000e+03 * invT;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    species[22] = +2.876276800000000e+00 + 2.474301000000000e-03 * T -
                  7.695102000000001e-07 * T2 + 1.278043800000000e-10 * T3 -
                  8.704126200000000e-15 * T4 + 9.913002200000000e+04 * invT;
  } else {
    // species 22: CHO+
    species[22] = +6.915726500000000e+00 + 8.315553500000000e-05 * T -
                  6.158781000000000e-09 * T2 + 2.271410725000000e-13 * T3 -
                  3.290888200000000e-18 * T4 + 9.612933300000000e+04 * invT;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    species[23] = +2.688523200000000e+00 + 2.722682700000000e-03 * T -
                  5.685188666666667e-07 * T2 + 6.208323250000000e-11 * T3 -
                  2.753888000000000e-15 * T4 + 7.089026800000000e+04 * invT;
  } else {
    // species 23: H3O+
    species[23] = +8.950880300000000e+00 + 1.264083400000000e-04 * T -
                  8.198132333333333e-09 * T2 + 2.719579500000000e-13 * T3 -
                  3.618167400000001e-18 * T4 + 6.448074900000000e+04 * invT;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    species[25] = +2.500000000000000e+00 - 4.075130400000000e-18 * T +
                  4.343289000000000e-22 * T2 - 1.816208325000000e-26 * T3 +
                  2.613249400000000e-31 * T4 - 7.453750000000000e+02 * invT;
  } else {
    // species 25: E
    species[25] = +2.500000000000000e+00 - 7.453750000000000e+02 * invT;
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    species[1] = +2.642044380000000e+00 * logT + 5.495292740000000e-03 * T -
                 6.358181700000000e-06 * T2 + 4.291639100000000e-09 * T3 -
                 1.175069372500000e-12 * T4 - 5.122311020000000e-01;
    // species 3: O2
    species[3] = +3.744039210000000e+00 * logT - 2.797401470000000e-03 * T +
                 4.900612790000000e-06 * T2 - 3.441988100000000e-09 * T3 +
                 9.498281175000000e-13 * T4 + 3.821326460000000e+00;
    // species 8: HO2
    species[8] = +3.619942990000000e+00 * logT + 1.058057050000000e-03 * T +
                 2.533394705000000e-06 * T2 - 2.112669206666667e-09 * T3 +
                 6.039932025000000e-13 * T4 + 6.444114820000000e+00;
    // species 11: CH4
    species[11] = +5.239673350000000e+00 * logT - 1.468351230000000e-02 * T +
                  2.648663555000000e-05 * T2 - 1.805562740000000e-08 * T3 +
                  4.907964150000000e-12 * T4 - 4.976497480000000e+00;
    // species 15: C
    species[15] = +2.544951920000000e+00 * logT - 2.477252810000000e-04 * T +
                  2.740091385000000e-07 * T2 - 1.828504166666667e-10 * T3 +
                  5.102933275000000e-14 * T4 + 4.568742730000000e+00;
    // species 20: CH2O
    species[20] = +4.326212960000000e+00 * logT - 7.011518530000000e-03 * T +
                  1.575884805000000e-05 * T2 - 1.121595460000000e-08 * T3 +
                  3.086350575000000e-12 * T4 + 2.620289020000000e+00;
  } else {
    // species 1: H2
    species[1] = +3.781998810000000e+00 * logT - 1.018732590000000e-03 * T +
                 6.211311650000000e-07 * T2 - 1.396706326666667e-10 * T3 +
                 1.188859482500000e-14 * T4 - 5.605259100000000e+00;
    // species 3: O2
    species[3] = +2.820124080000000e+00 * logT + 2.482113570000000e-03 * T -
                 7.560104700000000e-07 * T2 + 1.495187336666667e-10 * T3 -
                 1.218264170000000e-14 * T4 + 7.949145520000000e+00;
    // species 8: HO2
    species[8] = +3.023918890000000e+00 * logT + 4.463909070000000e-03 * T -
                 1.115732460000000e-06 * T2 + 2.042369330000000e-10 * T3 -
                 1.660665592500000e-14 * T4 + 9.106999730000000e+00;
    // species 11: CH4
    species[11] = +5.053464050000001e-01 * logT + 1.236978450000000e-02 * T -
                  2.499039610000000e-06 * T2 + 3.479758833333333e-10 * T3 -
                  2.157243540000000e-14 * T4 + 1.617527750000000e+01;
    // species 15: C
    species[15] = +2.494725310000000e+00 * logT + 3.928394760000000e-05 * T -
                  3.350074900000000e-08 * T2 + 1.239395646666667e-11 * T3 -
                  1.268267212500000e-15 * T4 + 4.793142540000000e+00;
    // species 20: CH2O
    species[20] = +1.333356520000000e+00 * logT + 1.009051830000000e-02 * T -
                  2.564762810000000e-06 * T2 + 4.180840233333333e-10 * T3 -
                  2.990977725000000e-14 * T4 + 1.599161440000000e+01;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    species[4] = +3.147992010000000e+00 * logT - 3.111740650000000e-03 * T +
                 3.090689485000000e-06 * T2 - 1.879362660000000e-09 * T3 +
                 4.871650400000000e-13 * T4 + 2.134465490000000e+00;
  } else {
    // species 4: O
    species[4] = +2.625491430000000e+00 * logT - 2.089596440000000e-04 * T +
                 6.695927300000000e-08 * T2 - 1.286252986666667e-11 * T3 +
                 1.097296722500000e-15 * T4 + 4.483585190000000e+00;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    species[21] = +4.034839790000000e+00 * logT - 2.158368640000000e-03 * T +
                  5.911693750000000e-06 * T2 - 3.948646866666666e-09 * T3 +
                  1.001484885000000e-12 * T4 + 4.200087700000000e+00;
  } else {
    // species 21: HCO
    species[21] = +2.600493180000000e+00 * logT + 5.292782580000000e-03 * T -
                  1.345921055000000e-06 * T2 + 2.404525993333334e-10 * T3 -
                  1.858803522500000e-14 * T4 + 1.074509330000000e+01;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    species[2] = +2.499505440000000e+00 * logT + 2.991640460000000e-06 * T -
                 2.963798795000000e-09 * T2 + 1.626033883333333e-12 * T3 -
                 3.638483000000000e-16 * T4 - 4.445740180000000e-01;
  } else {
    // species 2: H
    species[2] = +2.500314930000000e+00 * logT - 7.734068280000000e-07 * T +
                 3.196726730000000e-10 * T2 - 7.085059700000000e-14 * T3 +
                 6.111979775000000e-18 * T4 - 4.483572280000000e-01;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    species[9] = +3.757238910000000e+00 * logT - 2.144652410000000e-03 * T +
                 2.710395025000000e-06 * T2 - 1.390086543333333e-09 * T3 +
                 2.797528175000000e-13 * T4 + 2.799767990000000e+00;
  } else {
    // species 9: CO
    species[9] = +2.792553810000000e+00 * logT + 1.874868860000000e-03 * T -
                 4.298559630000000e-07 * T2 + 6.373335666666666e-11 * T3 -
                 4.196382150000000e-15 * T4 + 7.414435600000000e+00;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    species[14] = +4.181854340000000e+00 * logT - 2.211343140000000e-03 * T +
                  3.857637705000000e-06 * T2 - 1.986501270000000e-09 * T3 +
                  3.957865700000000e-13 * T4 - 7.030026020000000e-01;
  } else {
    // species 14: CH2(S)
    species[14] = +2.759342990000000e+00 * logT + 3.654683070000000e-03 * T -
                  6.779495700000000e-07 * T2 + 9.166013699999999e-11 * T3 -
                  5.919886800000000e-15 * T4 + 6.116463830000000e+00;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    species[0] = +3.731006820000000e+00 * logT - 1.831597300000000e-03 * T +
                 2.161623310000000e-06 * T2 - 1.014593836666667e-09 * T3 +
                 1.865178905000000e-13 * T4 + 2.168211980000000e+00;
  } else {
    // species 0: N2
    species[0] = +2.811660730000000e+00 * logT + 1.670673530000000e-03 * T -
                 3.399987140000000e-07 * T2 + 4.429379300000000e-11 * T3 -
                 2.569186050000000e-15 * T4 + 6.648380500000000e+00;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    species[12] = +3.478293100000000e+00 * logT + 3.547647730000000e-03 * T +
                  7.370422000000000e-07 * T2 - 6.479198500000000e-10 * T3 +
                  1.304803080000000e-13 * T4 + 2.408759560000000e+00;
  } else {
    // species 12: CH3
    species[12] = +2.788051040000000e+00 * logT + 6.152334770000000e-03 * T -
                  1.105896745000000e-06 * T2 + 1.248008826666667e-10 * T3 -
                  6.203783725000000e-15 * T4 + 5.778998180000000e+00;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    species[19] = +2.348215790000000e+00 * logT + 1.396001680000000e-02 * T -
                  5.431610300000000e-06 * T2 + 1.541661386666667e-09 * T3 -
                  2.021247905000000e-13 * T4 + 1.226619770000000e+01;
  } else {
    // species 19: CH2OH
    species[19] = +5.045349500000000e+00 * logT + 6.027270590000000e-03 * T -
                  1.056934105000000e-06 * T2 + 1.120286350000000e-10 * T3 -
                  5.024687050000000e-15 * T4 - 1.575240730000000e+00;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    species[5] = +4.060611720000000e+00 * logT - 8.658071890000000e-04 * T +
                 1.622047640000000e-06 * T2 - 6.008102633333333e-10 * T3 +
                 8.312082325000000e-14 * T4 - 2.961504810000000e-01;
  } else {
    // species 5: H2O
    species[5] = +2.667770750000000e+00 * logT + 3.057688490000000e-03 * T -
                 4.502212055000000e-07 * T2 + 4.778718400000000e-11 * T3 -
                 2.521445425000000e-15 * T4 + 6.911911310000000e+00;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    species[10] = +2.316843470000000e+00 * logT + 9.227550360000000e-03 * T -
                  3.878270465000000e-06 * T2 + 1.094084533333333e-09 * T3 -
                  1.371806205000000e-13 * T4 + 1.007862340000000e+01;
  } else {
    // species 10: CO2
    species[10] = +4.708764680000000e+00 * logT + 2.629147040000000e-03 * T -
                  4.653032310000000e-07 * T2 + 4.796430666666667e-11 * T3 -
                  1.906453535000000e-15 * T4 - 2.349764520000000e+00;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    species[16] = +3.772643320000000e+00 * logT - 1.585473500000000e-03 * T +
                  1.417561190000000e-06 * T2 - 4.538201933333334e-10 * T3 +
                  5.599883300000000e-14 * T4 + 8.794079040000000e-01;
  } else {
    // species 16: CH
    species[16] = +2.279901280000000e+00 * logT + 2.169852380000000e-03 * T -
                  3.538189425000000e-07 * T2 + 4.132449800000000e-11 * T3 -
                  2.390871287500000e-15 * T4 + 8.773260609999999e+00;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    species[6] = +3.913546310000000e+00 * logT - 1.662759260000000e-03 * T +
                 1.154600145000000e-06 * T2 - 3.411983600000000e-10 * T3 +
                 3.970740725000000e-14 * T4 + 2.054747190000000e-01;
  } else {
    // species 6: OH
    species[6] = +2.498673690000000e+00 * logT + 1.666352790000000e-03 * T -
                 3.141257580000000e-07 * T2 + 4.278226866666667e-11 * T3 -
                 2.643397350000000e-15 * T4 + 7.782188620000000e+00;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    species[18] = +8.896609850000000e-01 * logT + 1.701197670000000e-02 * T -
                  5.690367550000000e-06 * T2 + 1.294269760000000e-09 * T3 -
                  1.332103697500000e-13 * T4 + 1.850011340000000e+01;
  } else {
    // species 18: CH3O
    species[18] = +5.722380620000000e+00 * logT + 5.902276370000000e-03 * T -
                  9.017036000000000e-07 * T2 + 7.111166966666667e-11 * T3 -
                  1.404541022500000e-15 * T4 - 7.491736770000000e+00;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    species[7] = +3.347742240000000e+00 * logT + 7.050054370000000e-03 * T -
                 1.922610030000000e-06 * T2 + 3.890688700000000e-10 * T3 -
                 3.690452625000000e-14 * T4 + 7.178688510000000e+00;
    // species 13: CH2
    species[13] = +3.764894600000000e+00 * logT + 1.438391910000000e-03 * T +
                  2.377915385000000e-07 * T2 - 1.439295303333333e-10 * T3 +
                  1.895732185000000e-14 * T4 + 1.489531530000000e+00;
    // species 17: CH3OH
    species[17] = +8.473304790000000e-01 * logT + 1.630869040000000e-02 * T -
                  4.241724805000000e-06 * T2 + 7.643478033333333e-10 * T3 -
                  6.498800325000000e-14 * T4 + 1.959332970000000e+01;
  } else {
    // species 7: H2O2
    species[7] = +4.768696390000000e+00 * logT + 3.892378480000000e-03 * T -
                 6.069117450000000e-07 * T2 + 6.420509500000000e-11 * T3 -
                 3.064549750000000e-15 * T4 - 5.118117770000000e-01;
    // species 13: CH2
    species[13] = +2.812729720000000e+00 * logT + 3.554313880000000e-03 * T -
                  6.438426150000000e-07 * T2 + 7.375791466666666e-11 * T3 -
                  3.718453675000000e-15 * T4 + 6.642846520000000e+00;
    // species 17: CH3OH
    species[17] = +2.717015300000000e+00 * logT + 1.215383530000000e-02 * T -
                  2.510535155000000e-06 * T2 + 3.368935666666667e-10 * T3 -
                  2.046152057500000e-14 * T4 + 9.474205400000001e+00;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    species[24] = +3.102171800000000e+00 * logT + 2.798087500000000e-03 * T -
                  1.132556300000000e-06 * T2 + 2.897217233333333e-10 * T3 -
                  3.180471000000000e-14 * T4 + 6.760902000000000e+00;
  } else {
    // species 24: O2-
    species[24] = +4.259286700000000e+00 * logT + 2.246807200000000e-04 * T -
                  2.569897750000000e-08 * T2 + 2.451532600000000e-12 * T3 -
                  9.639663000000000e-17 * T4 + 4.759969700000000e-01;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    species[22] = +2.876276800000000e+00 * logT + 4.948602000000000e-03 * T -
                  1.154265300000000e-06 * T2 + 1.704058400000000e-10 * T3 -
                  1.088015775000000e-14 * T4 + 6.502026600000000e+00;
  } else {
    // species 22: CHO+
    species[22] = +6.915726500000000e+00 * logT + 1.663110700000000e-04 * T -
                  9.238171500000000e-09 * T2 + 3.028547633333333e-13 * T3 -
                  4.113610250000000e-18 * T4 - 1.808951000000000e+01;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    species[23] = +2.688523200000000e+00 * logT + 5.445365400000000e-03 * T -
                  8.527783000000000e-07 * T2 + 8.277764333333333e-11 * T3 -
                  3.442360000000000e-15 * T4 + 6.360821700000000e+00;
  } else {
    // species 23: H3O+
    species[23] = +8.950880300000000e+00 * logT + 2.528166800000000e-04 * T -
                  1.229719850000000e-08 * T2 + 3.626106000000000e-13 * T3 -
                  4.522709250000000e-18 * T4 - 3.387331900000000e+01;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    species[25] = +2.500000000000000e+00 * logT - 8.150260800000000e-18 * T +
                  6.514933500000000e-22 * T2 - 2.421611100000000e-26 * T3 +
                  3.266561750000000e-31 * T4 - 1.173397500000000e+01;
  } else {
    // species 25: E
    species[25] = +2.500000000000000e+00 * logT - 1.173397500000000e+01;
  }
}

// compute d(Cp/R)/dT and d(Cv/R)/dT at the given temperature
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
dcvpRdT(amrex::Real* species, const amrex::Real T)
{
  const amrex::Real T2 = T * T;
  const amrex::Real T3 = T * T2;

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    species[1] = +5.495292740000000e-03 - 2.543272680000000e-05 * T +
                 3.862475190000000e-08 * T2 - 1.880110996000000e-11 * T3;
    // species 3: O2
    species[3] = -2.797401470000000e-03 + 1.960245116000000e-05 * T -
                 3.097789290000000e-08 * T2 + 1.519724988000000e-11 * T3;
    // species 8: HO2
    species[8] = +1.058057050000000e-03 + 1.013357882000000e-05 * T -
                 1.901402286000000e-08 * T2 + 9.663891240000000e-12 * T3;
    // species 11: CH4
    species[11] = -1.468351230000000e-02 + 1.059465422000000e-04 * T -
                  1.625006466000000e-07 * T2 + 7.852742640000000e-11 * T3;
    // species 15: C
    species[15] = -2.477252810000000e-04 + 1.096036554000000e-06 * T -
                  1.645653750000000e-09 * T2 + 8.164693240000000e-13 * T3;
    // species 20: CH2O
    species[20] = -7.011518530000000e-03 + 6.303539220000000e-05 * T -
                  1.009435914000000e-07 * T2 + 4.938160920000000e-11 * T3;
  } else {
    // species 1: H2
    species[1] = -1.018732590000000e-03 + 2.484524660000000e-06 * T -
                 1.257035694000000e-09 * T2 + 1.902175172000000e-13 * T3;
    // species 3: O2
    species[3] = +2.482113570000000e-03 - 3.024041880000000e-06 * T +
                 1.345668603000000e-09 * T2 - 1.949222672000000e-13 * T3;
    // species 8: HO2
    species[8] = +4.463909070000000e-03 - 4.462929840000000e-06 * T +
                 1.838132397000000e-09 * T2 - 2.657064948000000e-13 * T3;
    // species 11: CH4
    species[11] = +1.236978450000000e-02 - 9.996158440000000e-06 * T +
                  3.131782950000000e-09 * T2 - 3.451589664000000e-13 * T3;
    // species 15: C
    species[15] = +3.928394760000000e-05 - 1.340029960000000e-07 * T +
                  1.115456082000000e-10 * T2 - 2.029227540000000e-14 * T3;
    // species 20: CH2O
    species[20] = +1.009051830000000e-02 - 1.025905124000000e-05 * T +
                  3.762756210000000e-09 * T2 - 4.785564360000000e-13 * T3;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    species[4] = -3.111740650000000e-03 + 1.236275794000000e-05 * T -
                 1.691426394000000e-08 * T2 + 7.794640640000000e-12 * T3;
  } else {
    // species 4: O
    species[4] = -2.089596440000000e-04 + 2.678370920000000e-07 * T -
                 1.157627688000000e-10 * T2 + 1.755674756000000e-14 * T3;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    species[21] = -2.158368640000000e-03 + 2.364677500000000e-05 * T -
                  3.553782180000000e-08 * T2 + 1.602375816000000e-11 * T3;
  } else {
    // species 21: HCO
    species[21] = +5.292782580000000e-03 - 5.383684220000000e-06 * T +
                  2.164073394000000e-09 * T2 - 2.974085636000000e-13 * T3;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    species[2] = +2.991640460000000e-06 - 1.185519518000000e-08 * T +
                 1.463430495000000e-11 * T2 - 5.821572800000000e-15 * T3;
  } else {
    // species 2: H
    species[2] = -7.734068280000000e-07 + 1.278690692000000e-09 * T -
                 6.376553729999999e-13 * T2 + 9.779167640000000e-17 * T3;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    species[9] = -2.144652410000000e-03 + 1.084158010000000e-05 * T -
                 1.251077889000000e-08 * T2 + 4.476045080000000e-12 * T3;
  } else {
    // species 9: CO
    species[9] = +1.874868860000000e-03 - 1.719423852000000e-06 * T +
                 5.736002100000000e-10 * T2 - 6.714211440000000e-14 * T3;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    species[14] = -2.211343140000000e-03 + 1.543055082000000e-05 * T -
                  1.787851143000000e-08 * T2 + 6.332585120000000e-12 * T3;
  } else {
    // species 14: CH2(S)
    species[14] = +3.654683070000000e-03 - 2.711798280000000e-06 * T +
                  8.249412330000000e-10 * T2 - 9.471818880000000e-14 * T3;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    species[0] = -1.831597300000000e-03 + 8.646493239999999e-06 * T -
                 9.131344529999999e-09 * T2 + 2.984286248000000e-12 * T3;
  } else {
    // species 0: N2
    species[0] = +1.670673530000000e-03 - 1.359994856000000e-06 * T +
                 3.986441370000000e-10 * T2 - 4.110697680000000e-14 * T3;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    species[12] = +3.547647730000000e-03 + 2.948168800000000e-06 * T -
                  5.831278650000001e-09 * T2 + 2.087684928000000e-12 * T3;
  } else {
    // species 12: CH3
    species[12] = +6.152334770000000e-03 - 4.423586980000000e-06 * T +
                  1.123207944000000e-09 * T2 - 9.926053960000000e-14 * T3;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    species[19] = +1.396001680000000e-02 - 2.172644120000000e-05 * T +
                  1.387495248000000e-08 * T2 - 3.233996648000000e-12 * T3;
  } else {
    // species 19: CH2OH
    species[19] = +6.027270590000000e-03 - 4.227736420000000e-06 * T +
                  1.008257715000000e-09 * T2 - 8.039499280000000e-14 * T3;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    species[5] = -8.658071890000000e-04 + 6.488190560000000e-06 * T -
                 5.407292369999999e-09 * T2 + 1.329933172000000e-12 * T3;
  } else {
    // species 5: H2O
    species[5] = +3.057688490000000e-03 - 1.800884822000000e-06 * T +
                 4.300846560000000e-10 * T2 - 4.034312680000000e-14 * T3;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    species[10] = +9.227550360000000e-03 - 1.551308186000000e-05 * T +
                  9.846760799999999e-09 * T2 - 2.194889928000000e-12 * T3;
  } else {
    // species 10: CO2
    species[10] = +2.629147040000000e-03 - 1.861212924000000e-06 * T +
                  4.316787600000001e-10 * T2 - 3.050325656000000e-14 * T3;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    species[16] = -1.585473500000000e-03 + 5.670244760000000e-06 * T -
                  4.084381740000001e-09 * T2 + 8.959813280000000e-13 * T3;
  } else {
    // species 16: CH
    species[16] = +2.169852380000000e-03 - 1.415275770000000e-06 * T +
                  3.719204820000000e-10 * T2 - 3.825394060000000e-14 * T3;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    species[6] = -1.662759260000000e-03 + 4.618400580000000e-06 * T -
                 3.070785240000000e-09 * T2 + 6.353185160000000e-13 * T3;
  } else {
    // species 6: OH
    species[6] = +1.666352790000000e-03 - 1.256503032000000e-06 * T +
                 3.850404180000000e-10 * T2 - 4.229435760000000e-14 * T3;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    species[18] = +1.701197670000000e-02 - 2.276147020000000e-05 * T +
                  1.164842784000000e-08 * T2 - 2.131365916000000e-12 * T3;
  } else {
    // species 18: CH3O
    species[18] = +5.902276370000000e-03 - 3.606814400000000e-06 * T +
                  6.400050270000000e-10 * T2 - 2.247265636000000e-14 * T3;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    species[7] = +7.050054370000000e-03 - 7.690440120000001e-06 * T +
                 3.501619830000000e-09 * T2 - 5.904724200000000e-13 * T3;
    // species 13: CH2
    species[13] = +1.438391910000000e-03 + 9.511661540000000e-07 * T -
                  1.295365773000000e-09 * T2 + 3.033171496000000e-13 * T3;
    // species 17: CH3OH
    species[17] = +1.630869040000000e-02 - 1.696689922000000e-05 * T +
                  6.879130230000000e-09 * T2 - 1.039808052000000e-12 * T3;
  } else {
    // species 7: H2O2
    species[7] = +3.892378480000000e-03 - 2.427646980000000e-06 * T +
                 5.778458550000000e-10 * T2 - 4.903279600000000e-14 * T3;
    // species 13: CH2
    species[13] = +3.554313880000000e-03 - 2.575370460000000e-06 * T +
                  6.638212320000000e-10 * T2 - 5.949525879999999e-14 * T3;
    // species 17: CH3OH
    species[17] = +1.215383530000000e-02 - 1.004214062000000e-05 * T +
                  3.032042100000000e-09 * T2 - 3.273843292000000e-13 * T3;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    species[24] = +2.798087500000000e-03 - 4.530225200000000e-06 * T +
                  2.607495510000000e-09 * T2 - 5.088753600000000e-13 * T3;
  } else {
    // species 24: O2-
    species[24] = +2.246807200000000e-04 - 1.027959100000000e-07 * T +
                  2.206379340000000e-11 * T2 - 1.542346080000000e-15 * T3;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    species[22] = +4.948602000000000e-03 - 4.617061200000000e-06 * T +
                  1.533652560000000e-09 * T2 - 1.740825240000000e-13 * T3;
  } else {
    // species 22: CHO+
    species[22] = +1.663110700000000e-04 - 3.695268600000000e-08 * T +
                  2.725692870000000e-12 * T2 - 6.581776400000000e-17 * T3;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    species[23] = +5.445365400000000e-03 - 3.411113200000000e-06 * T +
                  7.449987900000000e-10 * T2 - 5.507776000000000e-14 * T3;
  } else {
    // species 23: H3O+
    species[23] = +2.528166800000000e-04 - 4.918879400000000e-08 * T +
                  3.263495400000000e-12 * T2 - 7.236334800000001e-17 * T3;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    species[25] = -8.150260800000000e-18 + 2.605973400000000e-21 * T -
                  2.179449990000000e-25 * T2 + 5.226498800000000e-30 * T3;
  } else {
    // species 25: E
    species[25] = 0.0;
  }
}

// Returns the mean specific heat at CP (Eq. 33)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cpbl)
{
  amrex::Real result = 0;
  amrex::Real cpor[26]; //  temporary storage
  cp_R(cpor, T);

  // perform dot product
  for (int id = 0; id < 26; ++id) {
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    result += y[1] *
              (+2.642044380000000e+00 + 5.495292740000000e-03 * T -
               1.271636340000000e-05 * T2 + 1.287491730000000e-08 * T3 -
               4.700277490000000e-12 * T4) *
              0.4960317460317460;
    // species 3: O2
    result += y[3] *
              (+3.744039210000000e+00 - 2.797401470000000e-03 * T +
               9.801225580000000e-06 * T2 - 1.032596430000000e-08 * T3 +
               3.799312470000000e-12 * T4) *
              0.0312519532470779;
    // species 8: HO2
    result += y[8] *
              (+3.619942990000000e+00 + 1.058057050000000e-03 * T +
               5.066789410000000e-06 * T2 - 6.338007620000000e-09 * T3 +
               2.415972810000000e-12 * T4) *
              0.0302975216627280;
    // species 11: CH4
    result += y[11] *
              (+5.239673350000000e+00 - 1.468351230000000e-02 * T +
               5.297327110000000e-05 * T2 - 5.416688220000000e-08 * T3 +
               1.963185660000000e-11 * T4) *
              0.0623324814560868;
    // species 15: C
    result += y[15] *
              (+2.544951920000000e+00 - 2.477252810000000e-04 * T +
               5.480182770000000e-07 * T2 - 5.485512500000000e-10 * T3 +
               2.041173310000000e-13 * T4) *
              0.0832570144034635;
    // species 20: CH2O
    result += y[20] *
              (+4.326212960000000e+00 - 7.011518530000000e-03 * T +
               3.151769610000000e-05 * T2 - 3.364786380000000e-08 * T3 +
               1.234540230000000e-11 * T4) *
              0.0333044694598015;
  } else {
    // species 1: H2
    result += y[1] *
              (+3.781998810000000e+00 - 1.018732590000000e-03 * T +
               1.242262330000000e-06 * T2 - 4.190118980000000e-10 * T3 +
               4.755437930000000e-14 * T4) *
              0.4960317460317460;
    // species 3: O2
    result += y[3] *
              (+2.820124080000000e+00 + 2.482113570000000e-03 * T -
               1.512020940000000e-06 * T2 + 4.485562010000000e-10 * T3 -
               4.873056680000000e-14 * T4) *
              0.0312519532470779;
    // species 8: HO2
    result += y[8] *
              (+3.023918890000000e+00 + 4.463909070000000e-03 * T -
               2.231464920000000e-06 * T2 + 6.127107990000001e-10 * T3 -
               6.642662370000000e-14 * T4) *
              0.0302975216627280;
    // species 11: CH4
    result += y[11] *
              (+5.053464050000001e-01 + 1.236978450000000e-02 * T -
               4.998079220000000e-06 * T2 + 1.043927650000000e-09 * T3 -
               8.628974160000000e-14 * T4) *
              0.0623324814560868;
    // species 15: C
    result += y[15] *
              (+2.494725310000000e+00 + 3.928394760000000e-05 * T -
               6.700149800000000e-08 * T2 + 3.718186940000000e-11 * T3 -
               5.073068850000000e-15 * T4) *
              0.0832570144034635;
    // species 20: CH2O
    result += y[20] *
              (+1.333356520000000e+00 + 1.009051830000000e-02 * T -
               5.129525620000000e-06 * T2 + 1.254252070000000e-09 * T3 -
               1.196391090000000e-13 * T4) *
              0.0333044694598015;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    result += y[4] *
              (+3.147992010000000e+00 - 3.111740650000000e-03 * T +
               6.181378970000000e-06 * T2 - 5.638087980000000e-09 * T3 +
               1.948660160000000e-12 * T4) *
              0.0625039064941559;
  } else {
    // species 4: O
    result += y[4] *
              (+2.625491430000000e+00 - 2.089596440000000e-04 * T +
               1.339185460000000e-07 * T2 - 3.858758960000000e-11 * T3 +
               4.389186890000000e-15 * T4) *
              0.0625039064941559;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    result += y[21] *
              (+4.034839790000000e+00 - 2.158368640000000e-03 * T +
               1.182338750000000e-05 * T2 - 1.184594060000000e-08 * T3 +
               4.005939540000000e-12 * T4) *
              0.0344613688055690;
  } else {
    // species 21: HCO
    result += y[21] *
              (+2.600493180000000e+00 + 5.292782580000000e-03 * T -
               2.691842110000000e-06 * T2 + 7.213577980000000e-10 * T3 -
               7.435214090000000e-14 * T4) *
              0.0344613688055690;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    result += y[2] *
              (+2.499505440000000e+00 + 2.991640460000000e-06 * T -
               5.927597590000000e-09 * T2 + 4.878101650000000e-12 * T3 -
               1.455393200000000e-15 * T4) *
              0.9920634920634921;
  } else {
    // species 2: H
    result += y[2] *
              (+2.500314930000000e+00 - 7.734068280000000e-07 * T +
               6.393453460000000e-10 * T2 - 2.125517910000000e-13 * T3 +
               2.444791910000000e-17 * T4) *
              0.9920634920634921;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    result += y[9] *
              (+3.757238910000000e+00 - 2.144652410000000e-03 * T +
               5.420790050000000e-06 * T2 - 4.170259630000000e-09 * T3 +
               1.119011270000000e-12 * T4) *
              0.0357015351660121;
  } else {
    // species 9: CO
    result += y[9] *
              (+2.792553810000000e+00 + 1.874868860000000e-03 * T -
               8.597119260000000e-07 * T2 + 1.912000700000000e-10 * T3 -
               1.678552860000000e-14 * T4) *
              0.0357015351660121;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    result += y[14] *
              (+4.181854340000000e+00 - 2.211343140000000e-03 * T +
               7.715275410000000e-06 * T2 - 5.959503810000000e-09 * T3 +
               1.583146280000000e-12 * T4) *
              0.0712910814857061;
  } else {
    // species 14: CH2(S)
    result += y[14] *
              (+2.759342990000000e+00 + 3.654683070000000e-03 * T -
               1.355899140000000e-06 * T2 + 2.749804110000000e-10 * T3 -
               2.367954720000000e-14 * T4) *
              0.0712910814857061;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    result += y[0] *
              (+3.731006820000000e+00 - 1.831597300000000e-03 * T +
               4.323246620000000e-06 * T2 - 3.043781510000000e-09 * T3 +
               7.460715620000000e-13 * T4) *
              0.0356964374955379;
  } else {
    // species 0: N2
    result += y[0] *
              (+2.811660730000000e+00 + 1.670673530000000e-03 * T -
               6.799974280000000e-07 * T2 + 1.328813790000000e-10 * T3 -
               1.027674420000000e-14 * T4) *
              0.0356964374955379;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    result += y[12] *
              (+3.478293100000000e+00 + 3.547647730000000e-03 * T +
               1.474084400000000e-06 * T2 - 1.943759550000000e-09 * T3 +
               5.219212320000000e-13 * T4) *
              0.0665114732291320;
  } else {
    // species 12: CH3
    result += y[12] *
              (+2.788051040000000e+00 + 6.152334770000000e-03 * T -
               2.211793490000000e-06 * T2 + 3.744026480000000e-10 * T3 -
               2.481513490000000e-14 * T4) *
              0.0665114732291320;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    result += y[19] *
              (+2.348215790000000e+00 + 1.396001680000000e-02 * T -
               1.086322060000000e-05 * T2 + 4.624984160000000e-09 * T3 -
               8.084991620000000e-13 * T4) *
              0.0322227234645872;
  } else {
    // species 19: CH2OH
    result += y[19] *
              (+5.045349500000000e+00 + 6.027270590000000e-03 * T -
               2.113868210000000e-06 * T2 + 3.360859050000000e-10 * T3 -
               2.009874820000000e-14 * T4) *
              0.0322227234645872;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    result += y[5] *
              (+4.060611720000000e+00 - 8.658071890000000e-04 * T +
               3.244095280000000e-06 * T2 - 1.802430790000000e-09 * T3 +
               3.324832930000000e-13 * T4) *
              0.0555092978073827;
  } else {
    // species 5: H2O
    result += y[5] *
              (+2.667770750000000e+00 + 3.057688490000000e-03 * T -
               9.004424110000000e-07 * T2 + 1.433615520000000e-10 * T3 -
               1.008578170000000e-14 * T4) *
              0.0555092978073827;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    result += y[10] *
              (+2.316843470000000e+00 + 9.227550360000000e-03 * T -
               7.756540929999999e-06 * T2 + 3.282253600000000e-09 * T3 -
               5.487224820000000e-13 * T4) *
              0.0227226249176305;
  } else {
    // species 10: CO2
    result += y[10] *
              (+4.708764680000000e+00 + 2.629147040000000e-03 * T -
               9.306064620000000e-07 * T2 + 1.438929200000000e-10 * T3 -
               7.625814139999999e-15 * T4) *
              0.0227226249176305;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    result += y[16] *
              (+3.772643320000000e+00 - 1.585473500000000e-03 * T +
               2.835122380000000e-06 * T2 - 1.361460580000000e-09 * T3 +
               2.239953320000000e-13 * T4) *
              0.0768108149627468;
  } else {
    // species 16: CH
    result += y[16] *
              (+2.279901280000000e+00 + 2.169852380000000e-03 * T -
               7.076378850000000e-07 * T2 + 1.239734940000000e-10 * T3 -
               9.563485150000000e-15 * T4) *
              0.0768108149627468;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    result += y[6] *
              (+3.913546310000000e+00 - 1.662759260000000e-03 * T +
               2.309200290000000e-06 * T2 - 1.023595080000000e-09 * T3 +
               1.588296290000000e-13 * T4) *
              0.0587993179279120;
  } else {
    // species 6: OH
    result += y[6] *
              (+2.498673690000000e+00 + 1.666352790000000e-03 * T -
               6.282515160000000e-07 * T2 + 1.283468060000000e-10 * T3 -
               1.057358940000000e-14 * T4) *
              0.0587993179279120;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    result += y[18] *
              (+8.896609850000000e-01 + 1.701197670000000e-02 * T -
               1.138073510000000e-05 * T2 + 3.882809280000000e-09 * T3 -
               5.328414790000000e-13 * T4) *
              0.0322227234645872;
  } else {
    // species 18: CH3O
    result += y[18] *
              (+5.722380620000000e+00 + 5.902276370000000e-03 * T -
               1.803407200000000e-06 * T2 + 2.133350090000000e-10 * T3 -
               5.618164090000000e-15 * T4) *
              0.0322227234645872;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    result += y[7] *
              (+3.347742240000000e+00 + 7.050054370000000e-03 * T -
               3.845220060000000e-06 * T2 + 1.167206610000000e-09 * T3 -
               1.476181050000000e-13 * T4) *
              0.0293996589639560;
    // species 13: CH2
    result += y[13] *
              (+3.764894600000000e+00 + 1.438391910000000e-03 * T +
               4.755830770000000e-07 * T2 - 4.317885910000000e-10 * T3 +
               7.582928740000000e-14 * T4) *
              0.0712910814857061;
    // species 17: CH3OH
    result += y[17] *
              (+8.473304790000000e-01 + 1.630869040000000e-02 * T -
               8.483449609999999e-06 * T2 + 2.293043410000000e-09 * T3 -
               2.599520130000000e-13 * T4) *
              0.0312090381374446;
  } else {
    // species 7: H2O2
    result += y[7] *
              (+4.768696390000000e+00 + 3.892378480000000e-03 * T -
               1.213823490000000e-06 * T2 + 1.926152850000000e-10 * T3 -
               1.225819900000000e-14 * T4) *
              0.0293996589639560;
    // species 13: CH2
    result += y[13] *
              (+2.812729720000000e+00 + 3.554313880000000e-03 * T -
               1.287685230000000e-06 * T2 + 2.212737440000000e-10 * T3 -
               1.487381470000000e-14 * T4) *
              0.0712910814857061;
    // species 17: CH3OH
    result += y[17] *
              (+2.717015300000000e+00 + 1.215383530000000e-02 * T -
               5.021070310000000e-06 * T2 + 1.010680700000000e-09 * T3 -
               8.184608230000000e-14 * T4) *
              0.0312090381374446;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    result += y[24] *
              (+3.102171800000000e+00 + 2.798087500000000e-03 * T -
               2.265112600000000e-06 * T2 + 8.691651700000000e-10 * T3 -
               1.272188400000000e-13 * T4) *
              0.0312514174667246;
  } else {
    // species 24: O2-
    result += y[24] *
              (+4.259286700000000e+00 + 2.246807200000000e-04 * T -
               5.139795500000000e-08 * T2 + 7.354597800000001e-12 * T3 -
               3.855865200000000e-16 * T4) *
              0.0312514174667246;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    result += y[22] *
              (+2.876276800000000e+00 + 4.948602000000000e-03 * T -
               2.308530600000000e-06 * T2 + 5.112175200000000e-10 * T3 -
               4.352063100000000e-14 * T4) *
              0.0344620203036721;
  } else {
    // species 22: CHO+
    result += y[22] *
              (+6.915726500000000e+00 + 1.663110700000000e-04 * T -
               1.847634300000000e-08 * T2 + 9.085642900000000e-13 * T3 -
               1.645444100000000e-17 * T4) *
              0.0344620203036721;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    result += y[23] *
              (+2.688523200000000e+00 + 5.445365400000000e-03 * T -
               1.705556600000000e-06 * T2 + 2.483329300000000e-10 * T3 -
               1.376944000000000e-14 * T4) *
              0.0525694600509701;
  } else {
    // species 23: H3O+
    result += y[23] *
              (+8.950880300000000e+00 + 2.528166800000000e-04 * T -
               2.459439700000000e-08 * T2 + 1.087831800000000e-12 * T3 -
               1.809083700000000e-17 * T4) *
              0.0525694600509701;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    result += y[25] *
              (+2.500000000000000e+00 - 8.150260800000000e-18 * T +
               1.302986700000000e-21 * T2 - 7.264833300000000e-26 * T3 +
               1.306624700000000e-30 * T4) *
              1822.8884868472639482;
  } else {
    // species 25: E
    result += y[25] * (+2.500000000000000e+00) * 1822.8884868472639482;
  }

  cpbs = result * 8.31446261815324e+07;
}

// Returns the mean specific heat at CV (Eq. 35)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVBL(const amrex::Real T, const amrex::Real x[], amrex::Real& cvbl)
{
  amrex::Real result = 0;
  amrex::Real cvor[26]; //  temporary storage
  cv_R(cvor, T);

  // perform dot product
  for (int id = 0; id < 26; ++id) {
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    result += y[1] *
              (+1.642044380000000e+00 + 5.495292740000000e-03 * T -
               1.271636340000000e-05 * T2 + 1.287491730000000e-08 * T3 -
               4.700277490000000e-12 * T4) *
              0.4960317460317460;
    // species 3: O2
    result += y[3] *
              (+2.744039210000000e+00 - 2.797401470000000e-03 * T +
               9.801225580000000e-06 * T2 - 1.032596430000000e-08 * T3 +
               3.799312470000000e-12 * T4) *
              0.0312519532470779;
    // species 8: HO2
    result += y[8] *
              (+2.619942990000000e+00 + 1.058057050000000e-03 * T +
               5.066789410000000e-06 * T2 - 6.338007620000000e-09 * T3 +
               2.415972810000000e-12 * T4) *
              0.0302975216627280;
    // species 11: CH4
    result += y[11] *
              (+4.239673350000000e+00 - 1.468351230000000e-02 * T +
               5.297327110000000e-05 * T2 - 5.416688220000000e-08 * T3 +
               1.963185660000000e-11 * T4) *
              0.0623324814560868;
    // species 15: C
    result += y[15] *
              (+1.544951920000000e+00 - 2.477252810000000e-04 * T +
               5.480182770000000e-07 * T2 - 5.485512500000000e-10 * T3 +
               2.041173310000000e-13 * T4) *
              0.0832570144034635;
    // species 20: CH2O
    result += y[20] *
              (+3.326212960000000e+00 - 7.011518530000000e-03 * T +
               3.151769610000000e-05 * T2 - 3.364786380000000e-08 * T3 +
               1.234540230000000e-11 * T4) *
              0.0333044694598015;
  } else {
    // species 1: H2
    result += y[1] *
              (+2.781998810000000e+00 - 1.018732590000000e-03 * T +
               1.242262330000000e-06 * T2 - 4.190118980000000e-10 * T3 +
               4.755437930000000e-14 * T4) *
              0.4960317460317460;
    // species 3: O2
    result += y[3] *
              (+1.820124080000000e+00 + 2.482113570000000e-03 * T -
               1.512020940000000e-06 * T2 + 4.485562010000000e-10 * T3 -
               4.873056680000000e-14 * T4) *
              0.0312519532470779;
    // species 8: HO2
    result += y[8] *
              (+2.023918890000000e+00 + 4.463909070000000e-03 * T -
               2.231464920000000e-06 * T2 + 6.127107990000001e-10 * T3 -
               6.642662370000000e-14 * T4) *
              0.0302975216627280;
    // species 11: CH4
    result += y[11] *
              (-4.946535949999999e-01 + 1.236978450000000e-02 * T -
               4.998079220000000e-06 * T2 + 1.043927650000000e-09 * T3 -
               8.628974160000000e-14 * T4) *
              0.0623324814560868;
    // species 15: C
    result += y[15] *
              (+1.494725310000000e+00 + 3.928394760000000e-05 * T -
               6.700149800000000e-08 * T2 + 3.718186940000000e-11 * T3 -
               5.073068850000000e-15 * T4) *
              0.0832570144034635;
    // species 20: CH2O
    result += y[20] *
              (+3.333565199999999e-01 + 1.009051830000000e-02 * T -
               5.129525620000000e-06 * T2 + 1.254252070000000e-09 * T3 -
               1.196391090000000e-13 * T4) *
              0.0333044694598015;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    result += y[4] *
              (+2.147992010000000e+00 - 3.111740650000000e-03 * T +
               6.181378970000000e-06 * T2 - 5.638087980000000e-09 * T3 +
               1.948660160000000e-12 * T4) *
              0.0625039064941559;
  } else {
    // species 4: O
    result += y[4] *
              (+1.625491430000000e+00 - 2.089596440000000e-04 * T +
               1.339185460000000e-07 * T2 - 3.858758960000000e-11 * T3 +
               4.389186890000000e-15 * T4) *
              0.0625039064941559;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    result += y[21] *
              (+3.034839790000000e+00 - 2.158368640000000e-03 * T +
               1.182338750000000e-05 * T2 - 1.184594060000000e-08 * T3 +
               4.005939540000000e-12 * T4) *
              0.0344613688055690;
  } else {
    // species 21: HCO
    result += y[21] *
              (+1.600493180000000e+00 + 5.292782580000000e-03 * T -
               2.691842110000000e-06 * T2 + 7.213577980000000e-10 * T3 -
               7.435214090000000e-14 * T4) *
              0.0344613688055690;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    result += y[2] *
              (+1.499505440000000e+00 + 2.991640460000000e-06 * T -
               5.927597590000000e-09 * T2 + 4.878101650000000e-12 * T3 -
               1.455393200000000e-15 * T4) *
              0.9920634920634921;
  } else {
    // species 2: H
    result += y[2] *
              (+1.500314930000000e+00 - 7.734068280000000e-07 * T +
               6.393453460000000e-10 * T2 - 2.125517910000000e-13 * T3 +
               2.444791910000000e-17 * T4) *
              0.9920634920634921;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    result += y[9] *
              (+2.757238910000000e+00 - 2.144652410000000e-03 * T +
               5.420790050000000e-06 * T2 - 4.170259630000000e-09 * T3 +
               1.119011270000000e-12 * T4) *
              0.0357015351660121;
  } else {
    // species 9: CO
    result += y[9] *
              (+1.792553810000000e+00 + 1.874868860000000e-03 * T -
               8.597119260000000e-07 * T2 + 1.912000700000000e-10 * T3 -
               1.678552860000000e-14 * T4) *
              0.0357015351660121;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    result += y[14] *
              (+3.181854340000000e+00 - 2.211343140000000e-03 * T +
               7.715275410000000e-06 * T2 - 5.959503810000000e-09 * T3 +
               1.583146280000000e-12 * T4) *
              0.0712910814857061;
  } else {
    // species 14: CH2(S)
    result += y[14] *
              (+1.759342990000000e+00 + 3.654683070000000e-03 * T -
               1.355899140000000e-06 * T2 + 2.749804110000000e-10 * T3 -
               2.367954720000000e-14 * T4) *
              0.0712910814857061;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    result += y[0] *
              (+2.731006820000000e+00 - 1.831597300000000e-03 * T +
               4.323246620000000e-06 * T2 - 3.043781510000000e-09 * T3 +
               7.460715620000000e-13 * T4) *
              0.0356964374955379;
  } else {
    // species 0: N2
    result += y[0] *
              (+1.811660730000000e+00 + 1.670673530000000e-03 * T -
               6.799974280000000e-07 * T2 + 1.328813790000000e-10 * T3 -
               1.027674420000000e-14 * T4) *
              0.0356964374955379;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    result += y[12] *
              (+2.478293100000000e+00 + 3.547647730000000e-03 * T +
               1.474084400000000e-06 * T2 - 1.943759550000000e-09 * T3 +
               5.219212320000000e-13 * T4) *
              0.0665114732291320;
  } else {
    // species 12: CH3
    result += y[12] *
              (+1.788051040000000e+00 + 6.152334770000000e-03 * T -
               2.211793490000000e-06 * T2 + 3.744026480000000e-10 * T3 -
               2.481513490000000e-14 * T4) *
              0.0665114732291320;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    result += y[19] *
              (+1.348215790000000e+00 + 1.396001680000000e-02 * T -
               1.086322060000000e-05 * T2 + 4.624984160000000e-09 * T3 -
               8.084991620000000e-13 * T4) *
              0.0322227234645872;
  } else {
    // species 19: CH2OH
    result += y[19] *
              (+4.045349500000000e+00 + 6.027270590000000e-03 * T -
               2.113868210000000e-06 * T2 + 3.360859050000000e-10 * T3 -
               2.009874820000000e-14 * T4) *
              0.0322227234645872;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    result += y[5] *
              (+3.060611720000000e+00 - 8.658071890000000e-04 * T +
               3.244095280000000e-06 * T2 - 1.802430790000000e-09 * T3 +
               3.324832930000000e-13 * T4) *
              0.0555092978073827;
  } else {
    // species 5: H2O
    result += y[5] *
              (+1.667770750000000e+00 + 3.057688490000000e-03 * T -
               9.004424110000000e-07 * T2 + 1.433615520000000e-10 * T3 -
               1.008578170000000e-14 * T4) *
              0.0555092978073827;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    result += y[10] *
              (+1.316843470000000e+00 + 9.227550360000000e-03 * T -
               7.756540929999999e-06 * T2 + 3.282253600000000e-09 * T3 -
               5.487224820000000e-13 * T4) *
              0.0227226249176305;
  } else {
    // species 10: CO2
    result += y[10] *
              (+3.708764680000000e+00 + 2.629147040000000e-03 * T -
               9.306064620000000e-07 * T2 + 1.438929200000000e-10 * T3 -
               7.625814139999999e-15 * T4) *
              0.0227226249176305;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    result += y[16] *
              (+2.772643320000000e+00 - 1.585473500000000e-03 * T +
               2.835122380000000e-06 * T2 - 1.361460580000000e-09 * T3 +
               2.239953320000000e-13 * T4) *
              0.0768108149627468;
  } else {
    // species 16: CH
    result += y[16] *
              (+1.279901280000000e+00 + 2.169852380000000e-03 * T -
               7.076378850000000e-07 * T2 + 1.239734940000000e-10 * T3 -
               9.563485150000000e-15 * T4) *
              0.0768108149627468;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    result += y[6] *
              (+2.913546310000000e+00 - 1.662759260000000e-03 * T +
               2.309200290000000e-06 * T2 - 1.023595080000000e-09 * T3 +
               1.588296290000000e-13 * T4) *
              0.0587993179279120;
  } else {
    // species 6: OH
    result += y[6] *
              (+1.498673690000000e+00 + 1.666352790000000e-03 * T -
               6.282515160000000e-07 * T2 + 1.283468060000000e-10 * T3 -
               1.057358940000000e-14 * T4) *
              0.0587993179279120;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    result += y[18] *
              (-1.103390150000000e-01 + 1.701197670000000e-02 * T -
               1.138073510000000e-05 * T2 + 3.882809280000000e-09 * T3 -
               5.328414790000000e-13 * T4) *
              0.0322227234645872;
  } else {
    // species 18: CH3O
    result += y[18] *
              (+4.722380620000000e+00 + 5.902276370000000e-03 * T -
               1.803407200000000e-06 * T2 + 2.133350090000000e-10 * T3 -
               5.618164090000000e-15 * T4) *
              0.0322227234645872;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    result += y[7] *
              (+2.347742240000000e+00 + 7.050054370000000e-03 * T -
               3.845220060000000e-06 * T2 + 1.167206610000000e-09 * T3 -
               1.476181050000000e-13 * T4) *
              0.0293996589639560;
    // species 13: CH2
    result += y[13] *
              (+2.764894600000000e+00 + 1.438391910000000e-03 * T +
               4.755830770000000e-07 * T2 - 4.317885910000000e-10 * T3 +
               7.582928740000000e-14 * T4) *
              0.0712910814857061;
    // species 17: CH3OH
    result += y[17] *
              (-1.526695210000000e-01 + 1.630869040000000e-02 * T -
               8.483449609999999e-06 * T2 + 2.293043410000000e-09 * T3 -
               2.599520130000000e-13 * T4) *
              0.0312090381374446;
  } else {
    // species 7: H2O2
    result += y[7] *
              (+3.768696390000000e+00 + 3.892378480000000e-03 * T -
               1.213823490000000e-06 * T2 + 1.926152850000000e-10 * T3 -
               1.225819900000000e-14 * T4) *
              0.0293996589639560;
    // species 13: CH2
    result += y[13] *
              (+1.812729720000000e+00 + 3.554313880000000e-03 * T -
               1.287685230000000e-06 * T2 + 2.212737440000000e-10 * T3 -
               1.487381470000000e-14 * T4) *
              0.0712910814857061;
    // species 17: CH3OH
    result += y[17] *
              (+1.717015300000000e+00 + 1.215383530000000e-02 * T -
               5.021070310000000e-06 * T2 + 1.010680700000000e-09 * T3 -
               8.184608230000000e-14 * T4) *
              0.0312090381374446;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    result += y[24] *
              (+2.102171800000000e+00 + 2.798087500000000e-03 * T -
               2.265112600000000e-06 * T2 + 8.691651700000000e-10 * T3 -
               1.272188400000000e-13 * T4) *
              0.0312514174667246;
  } else {
    // species 24: O2-
    result += y[24] *
              (+3.259286700000000e+00 + 2.246807200000000e-04 * T -
               5.139795500000000e-08 * T2 + 7.354597800000001e-12 * T3 -
               3.855865200000000e-16 * T4) *
              0.0312514174667246;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    result += y[22] *
              (+1.876276800000000e+00 + 4.948602000000000e-03 * T -
               2.308530600000000e-06 * T2 + 5.112175200000000e-10 * T3 -
               4.352063100000000e-14 * T4) *
              0.0344620203036721;
  } else {
    // species 22: CHO+
    result += y[22] *
              (+5.915726500000000e+00 + 1.663110700000000e-04 * T -
               1.847634300000000e-08 * T2 + 9.085642900000000e-13 * T3 -
               1.645444100000000e-17 * T4) *
              0.0344620203036721;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    result += y[23] *
              (+1.688523200000000e+00 + 5.445365400000000e-03 * T -
               1.705556600000000e-06 * T2 + 2.483329300000000e-10 * T3 -
               1.376944000000000e-14 * T4) *
              0.0525694600509701;
  } else {
    // species 23: H3O+
    result += y[23] *
              (+7.950880300000000e+00 + 2.528166800000000e-04 * T -
               2.459439700000000e-08 * T2 + 1.087831800000000e-12 * T3 -
               1.809083700000000e-17 * T4) *
              0.0525694600509701;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    result += y[25] *
              (+1.500000000000000e+00 - 8.150260800000000e-18 * T +
               1.302986700000000e-21 * T2 - 7.264833300000000e-26 * T3 +
               1.306624700000000e-30 * T4) *
              1822.8884868472639482;
  } else {
    // species 25: E
    result += y[25] * (+1.500000000000000e+00) * 1822.8884868472639482;
  }

  cvbs = result * 8.31446261815324e+07;
}

// Returns the mean enthalpy of the mixture in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHBML(const amrex::Real T, const amrex::Real x[], amrex::Real& hbml)
{
  amrex::Real result = 0;
  amrex::Real hml[26];                       //  temporary storage
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesEnthalpy(hml, T);

  // perform dot product
  for (int id = 0; id < 26; ++id) {
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    result += y[1] *
              (+2.642044380000000e+00 + 2.747646370000000e-03 * T -
               4.238787800000000e-06 * T2 + 3.218729325000000e-09 * T3 -
               9.400554980000000e-13 * T4 - 9.432366140000000e+02 * invT) *
              0.4960317460317460;
    // species 3: O2
    result += y[3] *
              (+3.744039210000000e+00 - 1.398700735000000e-03 * T +
               3.267075193333333e-06 * T2 - 2.581491075000000e-09 * T3 +
               7.598624940000000e-13 * T4 - 1.060698270000000e+03 * invT) *
              0.0312519532470779;
    // species 8: HO2
    result += y[8] *
              (+3.619942990000000e+00 + 5.290285250000000e-04 * T +
               1.688929803333333e-06 * T2 - 1.584501905000000e-09 * T3 +
               4.831945620000000e-13 * T4 + 3.158982340000000e+02 * invT) *
              0.0302975216627280;
    // species 11: CH4
    result += y[11] *
              (+5.239673350000000e+00 - 7.341756150000000e-03 * T +
               1.765775703333333e-05 * T2 - 1.354172055000000e-08 * T3 +
               3.926371320000000e-12 * T4 - 1.025263080000000e+04 * invT) *
              0.0623324814560868;
    // species 15: C
    result += y[15] *
              (+2.544951920000000e+00 - 1.238626405000000e-04 * T +
               1.826727590000000e-07 * T2 - 1.371378125000000e-10 * T3 +
               4.082346620000000e-14 * T4 + 8.544341050000000e+04 * invT) *
              0.0832570144034635;
    // species 20: CH2O
    result += y[20] *
              (+4.326212960000000e+00 - 3.505759265000000e-03 * T +
               1.050589870000000e-05 * T2 - 8.411965949999999e-09 * T3 +
               2.469080460000000e-12 * T4 - 1.432701690000000e+04 * invT) *
              0.0333044694598015;
  } else {
    // species 1: H2
    result += y[1] *
              (+3.781998810000000e+00 - 5.093662950000000e-04 * T +
               4.140874433333334e-07 * T2 - 1.047529745000000e-10 * T3 +
               9.510875860000000e-15 * T4 - 1.102830230000000e+03 * invT) *
              0.4960317460317460;
    // species 3: O2
    result += y[3] *
              (+2.820124080000000e+00 + 1.241056785000000e-03 * T -
               5.040069800000000e-07 * T2 + 1.121390502500000e-10 * T3 -
               9.746113360000000e-15 * T4 - 9.313501480000000e+02 * invT) *
              0.0312519532470779;
    // species 8: HO2
    result += y[8] *
              (+3.023918890000000e+00 + 2.231954535000000e-03 * T -
               7.438216400000000e-07 * T2 + 1.531776997500000e-10 * T3 -
               1.328532474000000e-14 * T4 + 3.993416090000000e+02 * invT) *
              0.0302975216627280;
    // species 11: CH4
    result += y[11] *
              (+5.053464050000001e-01 + 6.184892250000000e-03 * T -
               1.666026406666667e-06 * T2 + 2.609819125000000e-10 * T3 -
               1.725794832000000e-14 * T4 - 9.589825010000000e+03 * invT) *
              0.0623324814560868;
    // species 15: C
    result += y[15] *
              (+2.494725310000000e+00 + 1.964197380000000e-05 * T -
               2.233383266666667e-08 * T2 + 9.295467350000000e-12 * T3 -
               1.014613770000000e-15 * T4 + 8.545044220000000e+04 * invT) *
              0.0832570144034635;
    // species 20: CH2O
    result += y[20] *
              (+1.333356520000000e+00 + 5.045259150000000e-03 * T -
               1.709841873333333e-06 * T2 + 3.135630175000000e-10 * T3 -
               2.392782180000000e-14 * T4 - 1.390801700000000e+04 * invT) *
              0.0333044694598015;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    result += y[4] *
              (+3.147992010000000e+00 - 1.555870325000000e-03 * T +
               2.060459656666667e-06 * T2 - 1.409521995000000e-09 * T3 +
               3.897320320000000e-13 * T4 + 2.913091180000000e+04 * invT) *
              0.0625039064941559;
  } else {
    // species 4: O
    result += y[4] *
              (+2.625491430000000e+00 - 1.044798220000000e-04 * T +
               4.463951533333333e-08 * T2 - 9.646897399999999e-12 * T3 +
               8.778373779999999e-16 * T4 + 2.920615190000000e+04 * invT) *
              0.0625039064941559;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    result += y[21] *
              (+4.034839790000000e+00 - 1.079184320000000e-03 * T +
               3.941129166666667e-06 * T2 - 2.961485150000000e-09 * T3 +
               8.011879080000000e-13 * T4 + 3.836363920000000e+03 * invT) *
              0.0344613688055690;
  } else {
    // species 21: HCO
    result += y[21] *
              (+2.600493180000000e+00 + 2.646391290000000e-03 * T -
               8.972807033333333e-07 * T2 + 1.803394495000000e-10 * T3 -
               1.487042818000000e-14 * T4 + 4.057253300000000e+03 * invT) *
              0.0344613688055690;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    result += y[2] *
              (+2.499505440000000e+00 + 1.495820230000000e-06 * T -
               1.975865863333333e-09 * T2 + 1.219525412500000e-12 * T3 -
               2.910786400000000e-16 * T4 + 2.547378660000000e+04 * invT) *
              0.9920634920634921;
  } else {
    // species 2: H
    result += y[2] *
              (+2.500314930000000e+00 - 3.867034140000000e-07 * T +
               2.131151153333333e-10 * T2 - 5.313794775000000e-14 * T3 +
               4.889583820000000e-18 * T4 + 2.547364740000000e+04 * invT) *
              0.9920634920634921;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    result += y[9] *
              (+3.757238910000000e+00 - 1.072326205000000e-03 * T +
               1.806930016666667e-06 * T2 - 1.042564907500000e-09 * T3 +
               2.238022540000000e-13 * T4 - 1.435755300000000e+04 * invT) *
              0.0357015351660121;
  } else {
    // species 9: CO
    result += y[9] *
              (+2.792553810000000e+00 + 9.374344300000000e-04 * T -
               2.865706420000000e-07 * T2 + 4.780001750000000e-11 * T3 -
               3.357105720000000e-15 * T4 - 1.417233350000000e+04 * invT) *
              0.0357015351660121;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    result += y[14] *
              (+4.181854340000000e+00 - 1.105671570000000e-03 * T +
               2.571758470000000e-06 * T2 - 1.489875952500000e-09 * T3 +
               3.166292560000000e-13 * T4 + 5.036694070000000e+04 * invT) *
              0.0712910814857061;
  } else {
    // species 14: CH2(S)
    result += y[14] *
              (+2.759342990000000e+00 + 1.827341535000000e-03 * T -
               4.519663800000000e-07 * T2 + 6.874510274999999e-11 * T3 -
               4.735909440000000e-15 * T4 + 5.064290790000000e+04 * invT) *
              0.0712910814857061;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    result += y[0] *
              (+3.731006820000000e+00 - 9.157986500000000e-04 * T +
               1.441082206666666e-06 * T2 - 7.609453775000000e-10 * T3 +
               1.492143124000000e-13 * T4 - 1.062874260000000e+03 * invT) *
              0.0356964374955379;
  } else {
    // species 0: N2
    result += y[0] *
              (+2.811660730000000e+00 + 8.353367650000000e-04 * T -
               2.266658093333333e-07 * T2 + 3.322034475000000e-11 * T3 -
               2.055348840000000e-15 * T4 - 8.698115790000001e+02 * invT) *
              0.0356964374955379;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    result += y[12] *
              (+3.478293100000000e+00 + 1.773823865000000e-03 * T +
               4.913614666666666e-07 * T2 - 4.859398875000000e-10 * T3 +
               1.043842464000000e-13 * T4 + 1.643995160000000e+04 * invT) *
              0.0665114732291320;
  } else {
    // species 12: CH3
    result += y[12] *
              (+2.788051040000000e+00 + 3.076167385000000e-03 * T -
               7.372644966666667e-07 * T2 + 9.360066200000000e-11 * T3 -
               4.963026980000000e-15 * T4 + 1.658628290000000e+04 * invT) *
              0.0665114732291320;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    result += y[19] *
              (+2.348215790000000e+00 + 6.980008400000000e-03 * T -
               3.621073533333334e-06 * T2 + 1.156246040000000e-09 * T3 -
               1.616998324000000e-13 * T4 - 3.302221060000000e+03 * invT) *
              0.0322227234645872;
  } else {
    // species 19: CH2OH
    result += y[19] *
              (+5.045349500000000e+00 + 3.013635295000000e-03 * T -
               7.046227366666666e-07 * T2 + 8.402147625000000e-11 * T3 -
               4.019749640000000e-15 * T4 - 4.035841430000000e+03 * invT) *
              0.0322227234645872;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    result += y[5] *
              (+4.060611720000000e+00 - 4.329035945000000e-04 * T +
               1.081365093333333e-06 * T2 - 4.506076975000000e-10 * T3 +
               6.649665860000001e-14 * T4 - 3.028313140000000e+04 * invT) *
              0.0555092978073827;
  } else {
    // species 5: H2O
    result += y[5] *
              (+2.667770750000000e+00 + 1.528844245000000e-03 * T -
               3.001474703333333e-07 * T2 + 3.584038800000000e-11 * T3 -
               2.017156340000000e-15 * T4 - 2.988756450000000e+04 * invT) *
              0.0555092978073827;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    result += y[10] *
              (+2.316843470000000e+00 + 4.613775180000000e-03 * T -
               2.585513643333333e-06 * T2 + 8.205634000000000e-10 * T3 -
               1.097444964000000e-13 * T4 - 4.836260670000000e+04 * invT) *
              0.0227226249176305;
  } else {
    // species 10: CO2
    result += y[10] *
              (+4.708764680000000e+00 + 1.314573520000000e-03 * T -
               3.102021540000000e-07 * T2 + 3.597323000000000e-11 * T3 -
               1.525162828000000e-15 * T4 - 4.905626390000000e+04 * invT) *
              0.0227226249176305;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    result += y[16] *
              (+3.772643320000000e+00 - 7.927367500000000e-04 * T +
               9.450407933333333e-07 * T2 - 3.403651450000000e-10 * T3 +
               4.479906640000000e-14 * T4 + 7.063124920000001e+04 * invT) *
              0.0768108149627468;
  } else {
    // species 16: CH
    result += y[16] *
              (+2.279901280000000e+00 + 1.084926190000000e-03 * T -
               2.358792950000000e-07 * T2 + 3.099337350000000e-11 * T3 -
               1.912697030000000e-15 * T4 + 7.110594120000000e+04 * invT) *
              0.0768108149627468;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    result += y[6] *
              (+3.913546310000000e+00 - 8.313796300000000e-04 * T +
               7.697334299999999e-07 * T2 - 2.558987700000000e-10 * T3 +
               3.176592580000000e-14 * T4 + 3.400050470000000e+03 * invT) *
              0.0587993179279120;
  } else {
    // species 6: OH
    result += y[6] *
              (+2.498673690000000e+00 + 8.331763950000000e-04 * T -
               2.094171720000000e-07 * T2 + 3.208670150000000e-11 * T3 -
               2.114717880000000e-15 * T4 + 3.881107160000000e+03 * invT) *
              0.0587993179279120;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    result += y[18] *
              (+8.896609850000000e-01 + 8.505988350000000e-03 * T -
               3.793578366666667e-06 * T2 + 9.707023200000000e-10 * T3 -
               1.065682958000000e-13 * T4 + 1.603161210000000e+03 * invT) *
              0.0322227234645872;
  } else {
    // species 18: CH3O
    result += y[18] *
              (+5.722380620000000e+00 + 2.951138185000000e-03 * T -
               6.011357333333334e-07 * T2 + 5.333375225000000e-11 * T3 -
               1.123632818000000e-15 * T4 - 7.862522250000001e+01 * invT) *
              0.0322227234645872;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    result += y[7] *
              (+3.347742240000000e+00 + 3.525027185000000e-03 * T -
               1.281740020000000e-06 * T2 + 2.918016525000000e-10 * T3 -
               2.952362099999999e-14 * T4 - 1.757847850000000e+04 * invT) *
              0.0293996589639560;
    // species 13: CH2
    result += y[13] *
              (+3.764894600000000e+00 + 7.191959550000000e-04 * T +
               1.585276923333333e-07 * T2 - 1.079471477500000e-10 * T3 +
               1.516585748000000e-14 * T4 + 4.586456990000000e+04 * invT) *
              0.0712910814857061;
    // species 17: CH3OH
    result += y[17] *
              (+8.473304790000000e-01 + 8.154345200000000e-03 * T -
               2.827816536666666e-06 * T2 + 5.732608525000000e-10 * T3 -
               5.199040260000000e-14 * T4 - 2.509625440000000e+04 * invT) *
              0.0312090381374446;
  } else {
    // species 7: H2O2
    result += y[7] *
              (+4.768696390000000e+00 + 1.946189240000000e-03 * T -
               4.046078300000000e-07 * T2 + 4.815382125000000e-11 * T3 -
               2.451639800000000e-15 * T4 - 1.809002200000000e+04 * invT) *
              0.0293996589639560;
    // species 13: CH2
    result += y[13] *
              (+2.812729720000000e+00 + 1.777156940000000e-03 * T -
               4.292284100000000e-07 * T2 + 5.531843600000000e-11 * T3 -
               2.974762940000000e-15 * T4 + 4.620734920000000e+04 * invT) *
              0.0712910814857061;
    // species 17: CH3OH
    result += y[17] *
              (+2.717015300000000e+00 + 6.076917650000000e-03 * T -
               1.673690103333333e-06 * T2 + 2.526701750000000e-10 * T3 -
               1.636921646000000e-14 * T4 - 2.576934090000000e+04 * invT) *
              0.0312090381374446;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    result += y[24] *
              (+3.102171800000000e+00 + 1.399043750000000e-03 * T -
               7.550375333333334e-07 * T2 + 2.172912925000000e-10 * T3 -
               2.544376800000000e-14 * T4 - 6.807479300000000e+03 * invT) *
              0.0312514174667246;
  } else {
    // species 24: O2-
    result += y[24] *
              (+4.259286700000000e+00 + 1.123403600000000e-04 * T -
               1.713265166666666e-08 * T2 + 1.838649450000000e-12 * T3 -
               7.711730400000001e-17 * T4 - 7.242625200000000e+03 * invT) *
              0.0312514174667246;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    result += y[22] *
              (+2.876276800000000e+00 + 2.474301000000000e-03 * T -
               7.695102000000001e-07 * T2 + 1.278043800000000e-10 * T3 -
               8.704126200000000e-15 * T4 + 9.913002200000000e+04 * invT) *
              0.0344620203036721;
  } else {
    // species 22: CHO+
    result += y[22] *
              (+6.915726500000000e+00 + 8.315553500000000e-05 * T -
               6.158781000000000e-09 * T2 + 2.271410725000000e-13 * T3 -
               3.290888200000000e-18 * T4 + 9.612933300000000e+04 * invT) *
              0.0344620203036721;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    result += y[23] *
              (+2.688523200000000e+00 + 2.722682700000000e-03 * T -
               5.685188666666667e-07 * T2 + 6.208323250000000e-11 * T3 -
               2.753888000000000e-15 * T4 + 7.089026800000000e+04 * invT) *
              0.0525694600509701;
  } else {
    // species 23: H3O+
    result += y[23] *
              (+8.950880300000000e+00 + 1.264083400000000e-04 * T -
               8.198132333333333e-09 * T2 + 2.719579500000000e-13 * T3 -
               3.618167400000001e-18 * T4 + 6.448074900000000e+04 * invT) *
              0.0525694600509701;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    result += y[25] *
              (+2.500000000000000e+00 - 4.075130400000000e-18 * T +
               4.343289000000000e-22 * T2 - 1.816208325000000e-26 * T3 +
               2.613249400000000e-31 * T4 - 7.453750000000000e+02 * invT) *
              1822.8884868472639482;
  } else {
    // species 25: E
    result += y[25] * (+2.500000000000000e+00 - 7.453750000000000e+02 * invT) *
              1822.8884868472639482;
  }

  const amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  hbms = result * RT;
}

// get mean internal energy in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUBML(const amrex::Real T, const amrex::Real x[], amrex::Real& ubml)
{
  amrex::Real result = 0;
  amrex::Real uml[26];                       //  temporary energy array
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T
  speciesInternalEnergy(uml, T);

  // perform dot product
  for (int id = 0; id < 26; ++id) {
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

  // species with midpoint at T=700 kelvin
  if (T < 700) {
    // species 1: H2
    result += y[1] *
              (+1.642044380000000e+00 + 2.747646370000000e-03 * T -
               4.238787800000000e-06 * T2 + 3.218729325000000e-09 * T3 -
               9.400554980000000e-13 * T4 - 9.432366140000000e+02 * invT) *
              0.4960317460317460;
    // species 3: O2
    result += y[3] *
              (+2.744039210000000e+00 - 1.398700735000000e-03 * T +
               3.267075193333333e-06 * T2 - 2.581491075000000e-09 * T3 +
               7.598624940000000e-13 * T4 - 1.060698270000000e+03 * invT) *
              0.0312519532470779;
    // species 8: HO2
    result += y[8] *
              (+2.619942990000000e+00 + 5.290285250000000e-04 * T +
               1.688929803333333e-06 * T2 - 1.584501905000000e-09 * T3 +
               4.831945620000000e-13 * T4 + 3.158982340000000e+02 * invT) *
              0.0302975216627280;
    // species 11: CH4
    result += y[11] *
              (+4.239673350000000e+00 - 7.341756150000000e-03 * T +
               1.765775703333333e-05 * T2 - 1.354172055000000e-08 * T3 +
               3.926371320000000e-12 * T4 - 1.025263080000000e+04 * invT) *
              0.0623324814560868;
    // species 15: C
    result += y[15] *
              (+1.544951920000000e+00 - 1.238626405000000e-04 * T +
               1.826727590000000e-07 * T2 - 1.371378125000000e-10 * T3 +
               4.082346620000000e-14 * T4 + 8.544341050000000e+04 * invT) *
              0.0832570144034635;
    // species 20: CH2O
    result += y[20] *
              (+3.326212960000000e+00 - 3.505759265000000e-03 * T +
               1.050589870000000e-05 * T2 - 8.411965949999999e-09 * T3 +
               2.469080460000000e-12 * T4 - 1.432701690000000e+04 * invT) *
              0.0333044694598015;
  } else {
    // species 1: H2
    result += y[1] *
              (+2.781998810000000e+00 - 5.093662950000000e-04 * T +
               4.140874433333334e-07 * T2 - 1.047529745000000e-10 * T3 +
               9.510875860000000e-15 * T4 - 1.102830230000000e+03 * invT) *
              0.4960317460317460;
    // species 3: O2
    result += y[3] *
              (+1.820124080000000e+00 + 1.241056785000000e-03 * T -
               5.040069800000000e-07 * T2 + 1.121390502500000e-10 * T3 -
               9.746113360000000e-15 * T4 - 9.313501480000000e+02 * invT) *
              0.0312519532470779;
    // species 8: HO2
    result += y[8] *
              (+2.023918890000000e+00 + 2.231954535000000e-03 * T -
               7.438216400000000e-07 * T2 + 1.531776997500000e-10 * T3 -
               1.328532474000000e-14 * T4 + 3.993416090000000e+02 * invT) *
              0.0302975216627280;
    // species 11: CH4
    result += y[11] *
              (-4.946535949999999e-01 + 6.184892250000000e-03 * T -
               1.666026406666667e-06 * T2 + 2.609819125000000e-10 * T3 -
               1.725794832000000e-14 * T4 - 9.589825010000000e+03 * invT) *
              0.0623324814560868;
    // species 15: C
    result += y[15] *
              (+1.494725310000000e+00 + 1.964197380000000e-05 * T -
               2.233383266666667e-08 * T2 + 9.295467350000000e-12 * T3 -
               1.014613770000000e-15 * T4 + 8.545044220000000e+04 * invT) *
              0.0832570144034635;
    // species 20: CH2O
    result += y[20] *
              (+3.333565199999999e-01 + 5.045259150000000e-03 * T -
               1.709841873333333e-06 * T2 + 3.135630175000000e-10 * T3 -
               2.392782180000000e-14 * T4 - 1.390801700000000e+04 * invT) *
              0.0333044694598015;
  }

  // species with midpoint at T=720 kelvin
  if (T < 720) {
    // species 4: O
    result += y[4] *
              (+2.147992010000000e+00 - 1.555870325000000e-03 * T +
               2.060459656666667e-06 * T2 - 1.409521995000000e-09 * T3 +
               3.897320320000000e-13 * T4 + 2.913091180000000e+04 * invT) *
              0.0625039064941559;
  } else {
    // species 4: O
    result += y[4] *
              (+1.625491430000000e+00 - 1.044798220000000e-04 * T +
               4.463951533333333e-08 * T2 - 9.646897399999999e-12 * T3 +
               8.778373779999999e-16 * T4 + 2.920615190000000e+04 * invT) *
              0.0625039064941559;
  }

  // species with midpoint at T=770 kelvin
  if (T < 770) {
    // species 21: HCO
    result += y[21] *
              (+3.034839790000000e+00 - 1.079184320000000e-03 * T +
               3.941129166666667e-06 * T2 - 2.961485150000000e-09 * T3 +
               8.011879080000000e-13 * T4 + 3.836363920000000e+03 * invT) *
              0.0344613688055690;
  } else {
    // species 21: HCO
    result += y[21] *
              (+1.600493180000000e+00 + 2.646391290000000e-03 * T -
               8.972807033333333e-07 * T2 + 1.803394495000000e-10 * T3 -
               1.487042818000000e-14 * T4 + 4.057253300000000e+03 * invT) *
              0.0344613688055690;
  }

  // species with midpoint at T=860 kelvin
  if (T < 860) {
    // species 2: H
    result += y[2] *
              (+1.499505440000000e+00 + 1.495820230000000e-06 * T -
               1.975865863333333e-09 * T2 + 1.219525412500000e-12 * T3 -
               2.910786400000000e-16 * T4 + 2.547378660000000e+04 * invT) *
              0.9920634920634921;
  } else {
    // species 2: H
    result += y[2] *
              (+1.500314930000000e+00 - 3.867034140000000e-07 * T +
               2.131151153333333e-10 * T2 - 5.313794775000000e-14 * T3 +
               4.889583820000000e-18 * T4 + 2.547364740000000e+04 * invT) *
              0.9920634920634921;
  }

  // species with midpoint at T=960 kelvin
  if (T < 960) {
    // species 9: CO
    result += y[9] *
              (+2.757238910000000e+00 - 1.072326205000000e-03 * T +
               1.806930016666667e-06 * T2 - 1.042564907500000e-09 * T3 +
               2.238022540000000e-13 * T4 - 1.435755300000000e+04 * invT) *
              0.0357015351660121;
  } else {
    // species 9: CO
    result += y[9] *
              (+1.792553810000000e+00 + 9.374344300000000e-04 * T -
               2.865706420000000e-07 * T2 + 4.780001750000000e-11 * T3 -
               3.357105720000000e-15 * T4 - 1.417233350000000e+04 * invT) *
              0.0357015351660121;
  }

  // species with midpoint at T=970 kelvin
  if (T < 970) {
    // species 14: CH2(S)
    result += y[14] *
              (+3.181854340000000e+00 - 1.105671570000000e-03 * T +
               2.571758470000000e-06 * T2 - 1.489875952500000e-09 * T3 +
               3.166292560000000e-13 * T4 + 5.036694070000000e+04 * invT) *
              0.0712910814857061;
  } else {
    // species 14: CH2(S)
    result += y[14] *
              (+1.759342990000000e+00 + 1.827341535000000e-03 * T -
               4.519663800000000e-07 * T2 + 6.874510274999999e-11 * T3 -
               4.735909440000000e-15 * T4 + 5.064290790000000e+04 * invT) *
              0.0712910814857061;
  }

  // species with midpoint at T=1050 kelvin
  if (T < 1050) {
    // species 0: N2
    result += y[0] *
              (+2.731006820000000e+00 - 9.157986500000000e-04 * T +
               1.441082206666666e-06 * T2 - 7.609453775000000e-10 * T3 +
               1.492143124000000e-13 * T4 - 1.062874260000000e+03 * invT) *
              0.0356964374955379;
  } else {
    // species 0: N2
    result += y[0] *
              (+1.811660730000000e+00 + 8.353367650000000e-04 * T -
               2.266658093333333e-07 * T2 + 3.322034475000000e-11 * T3 -
               2.055348840000000e-15 * T4 - 8.698115790000001e+02 * invT) *
              0.0356964374955379;
  }

  // species with midpoint at T=1060 kelvin
  if (T < 1060) {
    // species 12: CH3
    result += y[12] *
              (+2.478293100000000e+00 + 1.773823865000000e-03 * T +
               4.913614666666666e-07 * T2 - 4.859398875000000e-10 * T3 +
               1.043842464000000e-13 * T4 + 1.643995160000000e+04 * invT) *
              0.0665114732291320;
  } else {
    // species 12: CH3
    result += y[12] *
              (+1.788051040000000e+00 + 3.076167385000000e-03 * T -
               7.372644966666667e-07 * T2 + 9.360066200000000e-11 * T3 -
               4.963026980000000e-15 * T4 + 1.658628290000000e+04 * invT) *
              0.0665114732291320;
  }

  // species with midpoint at T=1360 kelvin
  if (T < 1360) {
    // species 19: CH2OH
    result += y[19] *
              (+1.348215790000000e+00 + 6.980008400000000e-03 * T -
               3.621073533333334e-06 * T2 + 1.156246040000000e-09 * T3 -
               1.616998324000000e-13 * T4 - 3.302221060000000e+03 * invT) *
              0.0322227234645872;
  } else {
    // species 19: CH2OH
    result += y[19] *
              (+4.045349500000000e+00 + 3.013635295000000e-03 * T -
               7.046227366666666e-07 * T2 + 8.402147625000000e-11 * T3 -
               4.019749640000000e-15 * T4 - 4.035841430000000e+03 * invT) *
              0.0322227234645872;
  }

  // species with midpoint at T=1420 kelvin
  if (T < 1420) {
    // species 5: H2O
    result += y[5] *
              (+3.060611720000000e+00 - 4.329035945000000e-04 * T +
               1.081365093333333e-06 * T2 - 4.506076975000000e-10 * T3 +
               6.649665860000001e-14 * T4 - 3.028313140000000e+04 * invT) *
              0.0555092978073827;
  } else {
    // species 5: H2O
    result += y[5] *
              (+1.667770750000000e+00 + 1.528844245000000e-03 * T -
               3.001474703333333e-07 * T2 + 3.584038800000000e-11 * T3 -
               2.017156340000000e-15 * T4 - 2.988756450000000e+04 * invT) *
              0.0555092978073827;
  }

  // species with midpoint at T=1450 kelvin
  if (T < 1450) {
    // species 10: CO2
    result += y[10] *
              (+1.316843470000000e+00 + 4.613775180000000e-03 * T -
               2.585513643333333e-06 * T2 + 8.205634000000000e-10 * T3 -
               1.097444964000000e-13 * T4 - 4.836260670000000e+04 * invT) *
              0.0227226249176305;
  } else {
    // species 10: CO2
    result += y[10] *
              (+3.708764680000000e+00 + 1.314573520000000e-03 * T -
               3.102021540000000e-07 * T2 + 3.597323000000000e-11 * T3 -
               1.525162828000000e-15 * T4 - 4.905626390000000e+04 * invT) *
              0.0227226249176305;
  }

  // species with midpoint at T=1590 kelvin
  if (T < 1590) {
    // species 16: CH
    result += y[16] *
              (+2.772643320000000e+00 - 7.927367500000000e-04 * T +
               9.450407933333333e-07 * T2 - 3.403651450000000e-10 * T3 +
               4.479906640000000e-14 * T4 + 7.063124920000001e+04 * invT) *
              0.0768108149627468;
  } else {
    // species 16: CH
    result += y[16] *
              (+1.279901280000000e+00 + 1.084926190000000e-03 * T -
               2.358792950000000e-07 * T2 + 3.099337350000000e-11 * T3 -
               1.912697030000000e-15 * T4 + 7.110594120000000e+04 * invT) *
              0.0768108149627468;
  }

  // species with midpoint at T=1700 kelvin
  if (T < 1700) {
    // species 6: OH
    result += y[6] *
              (+2.913546310000000e+00 - 8.313796300000000e-04 * T +
               7.697334299999999e-07 * T2 - 2.558987700000000e-10 * T3 +
               3.176592580000000e-14 * T4 + 3.400050470000000e+03 * invT) *
              0.0587993179279120;
  } else {
    // species 6: OH
    result += y[6] *
              (+1.498673690000000e+00 + 8.331763950000000e-04 * T -
               2.094171720000000e-07 * T2 + 3.208670150000000e-11 * T3 -
               2.114717880000000e-15 * T4 + 3.881107160000000e+03 * invT) *
              0.0587993179279120;
  }

  // species with midpoint at T=1740 kelvin
  if (T < 1740) {
    // species 18: CH3O
    result += y[18] *
              (-1.103390150000000e-01 + 8.505988350000000e-03 * T -
               3.793578366666667e-06 * T2 + 9.707023200000000e-10 * T3 -
               1.065682958000000e-13 * T4 + 1.603161210000000e+03 * invT) *
              0.0322227234645872;
  } else {
    // species 18: CH3O
    result += y[18] *
              (+4.722380620000000e+00 + 2.951138185000000e-03 * T -
               6.011357333333334e-07 * T2 + 5.333375225000000e-11 * T3 -
               1.123632818000000e-15 * T4 - 7.862522250000001e+01 * invT) *
              0.0322227234645872;
  }

  // species with midpoint at T=1800 kelvin
  if (T < 1800) {
    // species 7: H2O2
    result += y[7] *
              (+2.347742240000000e+00 + 3.525027185000000e-03 * T -
               1.281740020000000e-06 * T2 + 2.918016525000000e-10 * T3 -
               2.952362099999999e-14 * T4 - 1.757847850000000e+04 * invT) *
              0.0293996589639560;
    // species 13: CH2
    result += y[13] *
              (+2.764894600000000e+00 + 7.191959550000000e-04 * T +
               1.585276923333333e-07 * T2 - 1.079471477500000e-10 * T3 +
               1.516585748000000e-14 * T4 + 4.586456990000000e+04 * invT) *
              0.0712910814857061;
    // species 17: CH3OH
    result += y[17] *
              (-1.526695210000000e-01 + 8.154345200000000e-03 * T -
               2.827816536666666e-06 * T2 + 5.732608525000000e-10 * T3 -
               5.199040260000000e-14 * T4 - 2.509625440000000e+04 * invT) *
              0.0312090381374446;
  } else {
    // species 7: H2O2
    result += y[7] *
              (+3.768696390000000e+00 + 1.946189240000000e-03 * T -
               4.046078300000000e-07 * T2 + 4.815382125000000e-11 * T3 -
               2.451639800000000e-15 * T4 - 1.809002200000000e+04 * invT) *
              0.0293996589639560;
    // species 13: CH2
    result += y[13] *
              (+1.812729720000000e+00 + 1.777156940000000e-03 * T -
               4.292284100000000e-07 * T2 + 5.531843600000000e-11 * T3 -
               2.974762940000000e-15 * T4 + 4.620734920000000e+04 * invT) *
              0.0712910814857061;
    // species 17: CH3OH
    result += y[17] *
              (+1.717015300000000e+00 + 6.076917650000000e-03 * T -
               1.673690103333333e-06 * T2 + 2.526701750000000e-10 * T3 -
               1.636921646000000e-14 * T4 - 2.576934090000000e+04 * invT) *
              0.0312090381374446;
  }

  // species with midpoint at T=2008.71 kelvin
  if (T < 2008.71) {
    // species 24: O2-
    result += y[24] *
              (+2.102171800000000e+00 + 1.399043750000000e-03 * T -
               7.550375333333334e-07 * T2 + 2.172912925000000e-10 * T3 -
               2.544376800000000e-14 * T4 - 6.807479300000000e+03 * invT) *
              0.0312514174667246;
  } else {
    // species 24: O2-
    result += y[24] *
              (+3.259286700000000e+00 + 1.123403600000000e-04 * T -
               1.713265166666666e-08 * T2 + 1.838649450000000e-12 * T3 -
               7.711730400000001e-17 * T4 - 7.242625200000000e+03 * invT) *
              0.0312514174667246;
  }

  // species with midpoint at T=3654.18 kelvin
  if (T < 3654.18) {
    // species 22: CHO+
    result += y[22] *
              (+1.876276800000000e+00 + 2.474301000000000e-03 * T -
               7.695102000000001e-07 * T2 + 1.278043800000000e-10 * T3 -
               8.704126200000000e-15 * T4 + 9.913002200000000e+04 * invT) *
              0.0344620203036721;
  } else {
    // species 22: CHO+
    result += y[22] *
              (+5.915726500000000e+00 + 8.315553500000000e-05 * T -
               6.158781000000000e-09 * T2 + 2.271410725000000e-13 * T3 -
               3.290888200000000e-18 * T4 + 9.612933300000000e+04 * invT) *
              0.0344620203036721;
  }

  // species with midpoint at T=5430.9 kelvin
  if (T < 5430.9) {
    // species 23: H3O+
    result += y[23] *
              (+1.688523200000000e+00 + 2.722682700000000e-03 * T -
               5.685188666666667e-07 * T2 + 6.208323250000000e-11 * T3 -
               2.753888000000000e-15 * T4 + 7.089026800000000e+04 * invT) *
              0.0525694600509701;
  } else {
    // species 23: H3O+
    result += y[23] *
              (+7.950880300000000e+00 + 1.264083400000000e-04 * T -
               8.198132333333333e-09 * T2 + 2.719579500000000e-13 * T3 -
               3.618167400000001e-18 * T4 + 6.448074900000000e+04 * invT) *
              0.0525694600509701;
  }

  // species with midpoint at T=27300 kelvin
  if (T < 27300) {
    // species 25: E
    result += y[25] *
              (+1.500000000000000e+00 - 4.075130400000000e-18 * T +
               4.343289000000000e-22 * T2 - 1.816208325000000e-26 * T3 +
               2.613249400000000e-31 * T4 - 7.453750000000000e+02 * invT) *
              1822.8884868472639482;
  } else {
    // species 25: E
    result += y[25] * (+1.500000000000000e+00 - 7.453750000000000e+02 * invT) *
              1822.8884868472639482;
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
  amrex::Real sor[26]; //  temporary storage
  speciesEntropy(sor, T);

  // Compute Eq 42
  for (int id = 0; id < 26; ++id) {
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
  amrex::Real sor[26]; //  temporary storage
  amrex::Real x[26];   //  need a ytx conversion
  amrex::Real YOW = 0; // See Eq 4, 6 in CK Manual

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 26; i++) {
    YOW += y[i] * imw(i);
  }
  // Now compute y to x conversion
  x[0] = y[0] / (2.8013999999999999e+01 * YOW);
  x[1] = y[1] / (2.0160000000000000e+00 * YOW);
  x[2] = y[2] / (1.0080000000000000e+00 * YOW);
  x[3] = y[3] / (3.1998000000000001e+01 * YOW);
  x[4] = y[4] / (1.5999000000000001e+01 * YOW);
  x[5] = y[5] / (1.8015000000000001e+01 * YOW);
  x[6] = y[6] / (1.7007000000000001e+01 * YOW);
  x[7] = y[7] / (3.4014000000000003e+01 * YOW);
  x[8] = y[8] / (3.3006000000000000e+01 * YOW);
  x[9] = y[9] / (2.8009999999999998e+01 * YOW);
  x[10] = y[10] / (4.4009000000000000e+01 * YOW);
  x[11] = y[11] / (1.6042999999999999e+01 * YOW);
  x[12] = y[12] / (1.5035000000000000e+01 * YOW);
  x[13] = y[13] / (1.4026999999999999e+01 * YOW);
  x[14] = y[14] / (1.4026999999999999e+01 * YOW);
  x[15] = y[15] / (1.2010999999999999e+01 * YOW);
  x[16] = y[16] / (1.3018999999999998e+01 * YOW);
  x[17] = y[17] / (3.2042000000000002e+01 * YOW);
  x[18] = y[18] / (3.1033999999999999e+01 * YOW);
  x[19] = y[19] / (3.1033999999999999e+01 * YOW);
  x[20] = y[20] / (3.0026000000000000e+01 * YOW);
  x[21] = y[21] / (2.9018000000000001e+01 * YOW);
  x[22] = y[22] / (2.9017451420091128e+01 * YOW);
  x[23] = y[23] / (1.9022451420091127e+01 * YOW);
  x[24] = y[24] / (3.1998548579908874e+01 * YOW);
  x[25] = y[25] / (5.4857990887282834e-04 * YOW);
  speciesEntropy(sor, T);
  // Perform computation in Eq 42 and 43
  for (int i = 0; i < 26; i++) {
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
  XW += x[0] * 2.8013999999999999e+01;     // N2
  XW += x[1] * 2.0160000000000000e+00;     // H2
  XW += x[2] * 1.0080000000000000e+00;     // H
  XW += x[3] * 3.1998000000000001e+01;     // O2
  XW += x[4] * 1.5999000000000001e+01;     // O
  XW += x[5] * 1.8015000000000001e+01;     // H2O
  XW += x[6] * 1.7007000000000001e+01;     // OH
  XW += x[7] * 3.4014000000000003e+01;     // H2O2
  XW += x[8] * 3.3006000000000000e+01;     // HO2
  XW += x[9] * 2.8009999999999998e+01;     // CO
  XW += x[10] * 4.4009000000000000e+01;    // CO2
  XW += x[11] * 1.6042999999999999e+01;    // CH4
  XW += x[12] * 1.5035000000000000e+01;    // CH3
  XW += x[13] * 1.4026999999999999e+01;    // CH2
  XW += x[14] * 1.4026999999999999e+01;    // CH2(S)
  XW += x[15] * 1.2010999999999999e+01;    // C
  XW += x[16] * 1.3018999999999998e+01;    // CH
  XW += x[17] * 3.2042000000000002e+01;    // CH3OH
  XW += x[18] * 3.1033999999999999e+01;    // CH3O
  XW += x[19] * 3.1033999999999999e+01;    // CH2OH
  XW += x[20] * 3.0026000000000000e+01;    // CH2O
  XW += x[21] * 2.9018000000000001e+01;    // HCO
  XW += x[22] * 2.9017451420091128e+01;    // CHO+
  XW += x[23] * 1.9022451420091127e+01;    // H3O+
  XW += x[24] * 3.1998548579908874e+01;    // O2-
  XW += x[25] * 5.4857990887282834e-04;    // E
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

  for (int i = 0; i < 26; i++) {
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
  W += c[0] * 2.8013999999999999e+01;  // N2
  W += c[1] * 2.0160000000000000e+00;  // H2
  W += c[2] * 1.0080000000000000e+00;  // H
  W += c[3] * 3.1998000000000001e+01;  // O2
  W += c[4] * 1.5999000000000001e+01;  // O
  W += c[5] * 1.8015000000000001e+01;  // H2O
  W += c[6] * 1.7007000000000001e+01;  // OH
  W += c[7] * 3.4014000000000003e+01;  // H2O2
  W += c[8] * 3.3006000000000000e+01;  // HO2
  W += c[9] * 2.8009999999999998e+01;  // CO
  W += c[10] * 4.4009000000000000e+01; // CO2
  W += c[11] * 1.6042999999999999e+01; // CH4
  W += c[12] * 1.5035000000000000e+01; // CH3
  W += c[13] * 1.4026999999999999e+01; // CH2
  W += c[14] * 1.4026999999999999e+01; // CH2(S)
  W += c[15] * 1.2010999999999999e+01; // C
  W += c[16] * 1.3018999999999998e+01; // CH
  W += c[17] * 3.2042000000000002e+01; // CH3OH
  W += c[18] * 3.1033999999999999e+01; // CH3O
  W += c[19] * 3.1033999999999999e+01; // CH2OH
  W += c[20] * 3.0026000000000000e+01; // CH2O
  W += c[21] * 2.9018000000000001e+01; // HCO
  W += c[22] * 2.9017451420091128e+01; // CHO+
  W += c[23] * 1.9022451420091127e+01; // H3O+
  W += c[24] * 3.1998548579908874e+01; // O2-
  W += c[25] * 5.4857990887282834e-04; // E

  for (int id = 0; id < 26; ++id) {
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
  XW += x[0] * 2.8013999999999999e+01;       // N2
  XW += x[1] * 2.0160000000000000e+00;       // H2
  XW += x[2] * 1.0080000000000000e+00;       // H
  XW += x[3] * 3.1998000000000001e+01;       // O2
  XW += x[4] * 1.5999000000000001e+01;       // O
  XW += x[5] * 1.8015000000000001e+01;       // H2O
  XW += x[6] * 1.7007000000000001e+01;       // OH
  XW += x[7] * 3.4014000000000003e+01;       // H2O2
  XW += x[8] * 3.3006000000000000e+01;       // HO2
  XW += x[9] * 2.8009999999999998e+01;       // CO
  XW += x[10] * 4.4009000000000000e+01;      // CO2
  XW += x[11] * 1.6042999999999999e+01;      // CH4
  XW += x[12] * 1.5035000000000000e+01;      // CH3
  XW += x[13] * 1.4026999999999999e+01;      // CH2
  XW += x[14] * 1.4026999999999999e+01;      // CH2(S)
  XW += x[15] * 1.2010999999999999e+01;      // C
  XW += x[16] * 1.3018999999999998e+01;      // CH
  XW += x[17] * 3.2042000000000002e+01;      // CH3OH
  XW += x[18] * 3.1033999999999999e+01;      // CH3O
  XW += x[19] * 3.1033999999999999e+01;      // CH2OH
  XW += x[20] * 3.0026000000000000e+01;      // CH2O
  XW += x[21] * 2.9018000000000001e+01;      // HCO
  XW += x[22] * 2.9017451420091128e+01;      // CHO+
  XW += x[23] * 1.9022451420091127e+01;      // H3O+
  XW += x[24] * 3.1998548579908874e+01;      // O2-
  XW += x[25] * 5.4857990887282834e-04;      // E
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

  for (int i = 0; i < 26; i++) {
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
  W += c[0] * 2.8013999999999999e+01;  // N2
  W += c[1] * 2.0160000000000000e+00;  // H2
  W += c[2] * 1.0080000000000000e+00;  // H
  W += c[3] * 3.1998000000000001e+01;  // O2
  W += c[4] * 1.5999000000000001e+01;  // O
  W += c[5] * 1.8015000000000001e+01;  // H2O
  W += c[6] * 1.7007000000000001e+01;  // OH
  W += c[7] * 3.4014000000000003e+01;  // H2O2
  W += c[8] * 3.3006000000000000e+01;  // HO2
  W += c[9] * 2.8009999999999998e+01;  // CO
  W += c[10] * 4.4009000000000000e+01; // CO2
  W += c[11] * 1.6042999999999999e+01; // CH4
  W += c[12] * 1.5035000000000000e+01; // CH3
  W += c[13] * 1.4026999999999999e+01; // CH2
  W += c[14] * 1.4026999999999999e+01; // CH2(S)
  W += c[15] * 1.2010999999999999e+01; // C
  W += c[16] * 1.3018999999999998e+01; // CH
  W += c[17] * 3.2042000000000002e+01; // CH3OH
  W += c[18] * 3.1033999999999999e+01; // CH3O
  W += c[19] * 3.1033999999999999e+01; // CH2OH
  W += c[20] * 3.0026000000000000e+01; // CH2O
  W += c[21] * 2.9018000000000001e+01; // HCO
  W += c[22] * 2.9017451420091128e+01; // CHO+
  W += c[23] * 1.9022451420091127e+01; // H3O+
  W += c[24] * 3.1998548579908874e+01; // O2-
  W += c[25] * 5.4857990887282834e-04; // E

  for (int id = 0; id < 26; ++id) {
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

  for (int i = 0; i < 26; i++) {
    YOW += y[i] * imw(i);
  }

  wtm = 1.0 / YOW;
}

// given x[species]: mole fractions
// returns mean molecular weight (gm/mole)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKMMWX(const amrex::Real x[], amrex::Real& wtm)
{
  amrex::Real XW = 0;                   //  see Eq 4 in CK Manual
  XW += x[0] * 2.8013999999999999e+01;  // N2
  XW += x[1] * 2.0160000000000000e+00;  // H2
  XW += x[2] * 1.0080000000000000e+00;  // H
  XW += x[3] * 3.1998000000000001e+01;  // O2
  XW += x[4] * 1.5999000000000001e+01;  // O
  XW += x[5] * 1.8015000000000001e+01;  // H2O
  XW += x[6] * 1.7007000000000001e+01;  // OH
  XW += x[7] * 3.4014000000000003e+01;  // H2O2
  XW += x[8] * 3.3006000000000000e+01;  // HO2
  XW += x[9] * 2.8009999999999998e+01;  // CO
  XW += x[10] * 4.4009000000000000e+01; // CO2
  XW += x[11] * 1.6042999999999999e+01; // CH4
  XW += x[12] * 1.5035000000000000e+01; // CH3
  XW += x[13] * 1.4026999999999999e+01; // CH2
  XW += x[14] * 1.4026999999999999e+01; // CH2(S)
  XW += x[15] * 1.2010999999999999e+01; // C
  XW += x[16] * 1.3018999999999998e+01; // CH
  XW += x[17] * 3.2042000000000002e+01; // CH3OH
  XW += x[18] * 3.1033999999999999e+01; // CH3O
  XW += x[19] * 3.1033999999999999e+01; // CH2OH
  XW += x[20] * 3.0026000000000000e+01; // CH2O
  XW += x[21] * 2.9018000000000001e+01; // HCO
  XW += x[22] * 2.9017451420091128e+01; // CHO+
  XW += x[23] * 1.9022451420091127e+01; // H3O+
  XW += x[24] * 3.1998548579908874e+01; // O2-
  XW += x[25] * 5.4857990887282834e-04; // E
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
  W += c[0] * 2.8013999999999999e+01;  // N2
  W += c[1] * 2.0160000000000000e+00;  // H2
  W += c[2] * 1.0080000000000000e+00;  // H
  W += c[3] * 3.1998000000000001e+01;  // O2
  W += c[4] * 1.5999000000000001e+01;  // O
  W += c[5] * 1.8015000000000001e+01;  // H2O
  W += c[6] * 1.7007000000000001e+01;  // OH
  W += c[7] * 3.4014000000000003e+01;  // H2O2
  W += c[8] * 3.3006000000000000e+01;  // HO2
  W += c[9] * 2.8009999999999998e+01;  // CO
  W += c[10] * 4.4009000000000000e+01; // CO2
  W += c[11] * 1.6042999999999999e+01; // CH4
  W += c[12] * 1.5035000000000000e+01; // CH3
  W += c[13] * 1.4026999999999999e+01; // CH2
  W += c[14] * 1.4026999999999999e+01; // CH2(S)
  W += c[15] * 1.2010999999999999e+01; // C
  W += c[16] * 1.3018999999999998e+01; // CH
  W += c[17] * 3.2042000000000002e+01; // CH3OH
  W += c[18] * 3.1033999999999999e+01; // CH3O
  W += c[19] * 3.1033999999999999e+01; // CH2OH
  W += c[20] * 3.0026000000000000e+01; // CH2O
  W += c[21] * 2.9018000000000001e+01; // HCO
  W += c[22] * 2.9017451420091128e+01; // CHO+
  W += c[23] * 1.9022451420091127e+01; // H3O+
  W += c[24] * 3.1998548579908874e+01; // O2-
  W += c[25] * 5.4857990887282834e-04; // E

  for (int id = 0; id < 26; ++id) {
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

  for (int i = 0; i < 26; i++) {
    YOW += y[i] * imw(i);
  }

  amrex::Real YOWINV = 1.0 / YOW;

  for (int i = 0; i < 26; i++) {
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
  for (int i = 0; i < 26; i++) {
    c[i] = y[i] * imw(i);
  }
  for (int i = 0; i < 26; i++) {
    YOW += c[i];
  }

  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // Now compute conversion

  for (int i = 0; i < 26; i++) {
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

  for (int i = 0; i < 26; i++) {
    c[i] = rho * y[i] * imw(i);
  }
}

// convert x[species] (mole fracs) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKXTY(const amrex::Real x[], amrex::Real y[])
{
  amrex::Real XW = 0; // See Eq 4, 9 in CK Manual
  // Compute mean molecular wt first
  XW += x[0] * 2.8013999999999999e+01;  // N2
  XW += x[1] * 2.0160000000000000e+00;  // H2
  XW += x[2] * 1.0080000000000000e+00;  // H
  XW += x[3] * 3.1998000000000001e+01;  // O2
  XW += x[4] * 1.5999000000000001e+01;  // O
  XW += x[5] * 1.8015000000000001e+01;  // H2O
  XW += x[6] * 1.7007000000000001e+01;  // OH
  XW += x[7] * 3.4014000000000003e+01;  // H2O2
  XW += x[8] * 3.3006000000000000e+01;  // HO2
  XW += x[9] * 2.8009999999999998e+01;  // CO
  XW += x[10] * 4.4009000000000000e+01; // CO2
  XW += x[11] * 1.6042999999999999e+01; // CH4
  XW += x[12] * 1.5035000000000000e+01; // CH3
  XW += x[13] * 1.4026999999999999e+01; // CH2
  XW += x[14] * 1.4026999999999999e+01; // CH2(S)
  XW += x[15] * 1.2010999999999999e+01; // C
  XW += x[16] * 1.3018999999999998e+01; // CH
  XW += x[17] * 3.2042000000000002e+01; // CH3OH
  XW += x[18] * 3.1033999999999999e+01; // CH3O
  XW += x[19] * 3.1033999999999999e+01; // CH2OH
  XW += x[20] * 3.0026000000000000e+01; // CH2O
  XW += x[21] * 2.9018000000000001e+01; // HCO
  XW += x[22] * 2.9017451420091128e+01; // CHO+
  XW += x[23] * 1.9022451420091127e+01; // H3O+
  XW += x[24] * 3.1998548579908874e+01; // O2-
  XW += x[25] * 5.4857990887282834e-04; // E
  // Now compute conversion
  amrex::Real XWinv = 1.0 / XW;
  y[0] = x[0] * 2.8013999999999999e+01 * XWinv;
  y[1] = x[1] * 2.0160000000000000e+00 * XWinv;
  y[2] = x[2] * 1.0080000000000000e+00 * XWinv;
  y[3] = x[3] * 3.1998000000000001e+01 * XWinv;
  y[4] = x[4] * 1.5999000000000001e+01 * XWinv;
  y[5] = x[5] * 1.8015000000000001e+01 * XWinv;
  y[6] = x[6] * 1.7007000000000001e+01 * XWinv;
  y[7] = x[7] * 3.4014000000000003e+01 * XWinv;
  y[8] = x[8] * 3.3006000000000000e+01 * XWinv;
  y[9] = x[9] * 2.8009999999999998e+01 * XWinv;
  y[10] = x[10] * 4.4009000000000000e+01 * XWinv;
  y[11] = x[11] * 1.6042999999999999e+01 * XWinv;
  y[12] = x[12] * 1.5035000000000000e+01 * XWinv;
  y[13] = x[13] * 1.4026999999999999e+01 * XWinv;
  y[14] = x[14] * 1.4026999999999999e+01 * XWinv;
  y[15] = x[15] * 1.2010999999999999e+01 * XWinv;
  y[16] = x[16] * 1.3018999999999998e+01 * XWinv;
  y[17] = x[17] * 3.2042000000000002e+01 * XWinv;
  y[18] = x[18] * 3.1033999999999999e+01 * XWinv;
  y[19] = x[19] * 3.1033999999999999e+01 * XWinv;
  y[20] = x[20] * 3.0026000000000000e+01 * XWinv;
  y[21] = x[21] * 2.9018000000000001e+01 * XWinv;
  y[22] = x[22] * 2.9017451420091128e+01 * XWinv;
  y[23] = x[23] * 1.9022451420091127e+01 * XWinv;
  y[24] = x[24] * 3.1998548579908874e+01 * XWinv;
  y[25] = x[25] * 5.4857990887282834e-04 * XWinv;
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
  for (int id = 0; id < 26; ++id) {
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
  XW += x[0] * 2.8013999999999999e+01;  // N2
  XW += x[1] * 2.0160000000000000e+00;  // H2
  XW += x[2] * 1.0080000000000000e+00;  // H
  XW += x[3] * 3.1998000000000001e+01;  // O2
  XW += x[4] * 1.5999000000000001e+01;  // O
  XW += x[5] * 1.8015000000000001e+01;  // H2O
  XW += x[6] * 1.7007000000000001e+01;  // OH
  XW += x[7] * 3.4014000000000003e+01;  // H2O2
  XW += x[8] * 3.3006000000000000e+01;  // HO2
  XW += x[9] * 2.8009999999999998e+01;  // CO
  XW += x[10] * 4.4009000000000000e+01; // CO2
  XW += x[11] * 1.6042999999999999e+01; // CH4
  XW += x[12] * 1.5035000000000000e+01; // CH3
  XW += x[13] * 1.4026999999999999e+01; // CH2
  XW += x[14] * 1.4026999999999999e+01; // CH2(S)
  XW += x[15] * 1.2010999999999999e+01; // C
  XW += x[16] * 1.3018999999999998e+01; // CH
  XW += x[17] * 3.2042000000000002e+01; // CH3OH
  XW += x[18] * 3.1033999999999999e+01; // CH3O
  XW += x[19] * 3.1033999999999999e+01; // CH2OH
  XW += x[20] * 3.0026000000000000e+01; // CH2O
  XW += x[21] * 2.9018000000000001e+01; // HCO
  XW += x[22] * 2.9017451420091128e+01; // CHO+
  XW += x[23] * 1.9022451420091127e+01; // H3O+
  XW += x[24] * 3.1998548579908874e+01; // O2-
  XW += x[25] * 5.4857990887282834e-04; // E
  ROW = rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 26; ++id) {
    c[id] = x[id] * ROW;
  }
}

// convert c[species] (molar conc) to x[species] (mole fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTX(const amrex::Real c[], amrex::Real x[])
{
  amrex::Real sumC = 0;

  // compute sum of c
  for (int id = 0; id < 26; ++id) {
    sumC += c[id];
  }

  //  See Eq 13
  amrex::Real sumCinv = 1.0 / sumC;
  for (int id = 0; id < 26; ++id) {
    x[id] = c[id] * sumCinv;
  }
}

// convert c[species] (molar conc) to y[species] (mass fracs)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCTY(const amrex::Real c[], amrex::Real y[])
{
  amrex::Real CW = 0; // See Eq 12 in CK Manual
  // compute denominator in eq 12 first
  CW += c[0] * 2.8013999999999999e+01;  // N2
  CW += c[1] * 2.0160000000000000e+00;  // H2
  CW += c[2] * 1.0080000000000000e+00;  // H
  CW += c[3] * 3.1998000000000001e+01;  // O2
  CW += c[4] * 1.5999000000000001e+01;  // O
  CW += c[5] * 1.8015000000000001e+01;  // H2O
  CW += c[6] * 1.7007000000000001e+01;  // OH
  CW += c[7] * 3.4014000000000003e+01;  // H2O2
  CW += c[8] * 3.3006000000000000e+01;  // HO2
  CW += c[9] * 2.8009999999999998e+01;  // CO
  CW += c[10] * 4.4009000000000000e+01; // CO2
  CW += c[11] * 1.6042999999999999e+01; // CH4
  CW += c[12] * 1.5035000000000000e+01; // CH3
  CW += c[13] * 1.4026999999999999e+01; // CH2
  CW += c[14] * 1.4026999999999999e+01; // CH2(S)
  CW += c[15] * 1.2010999999999999e+01; // C
  CW += c[16] * 1.3018999999999998e+01; // CH
  CW += c[17] * 3.2042000000000002e+01; // CH3OH
  CW += c[18] * 3.1033999999999999e+01; // CH3O
  CW += c[19] * 3.1033999999999999e+01; // CH2OH
  CW += c[20] * 3.0026000000000000e+01; // CH2O
  CW += c[21] * 2.9018000000000001e+01; // HCO
  CW += c[22] * 2.9017451420091128e+01; // CHO+
  CW += c[23] * 1.9022451420091127e+01; // H3O+
  CW += c[24] * 3.1998548579908874e+01; // O2-
  CW += c[25] * 5.4857990887282834e-04; // E
  // Now compute conversion
  amrex::Real CWinv = 1.0 / CW;
  y[0] = c[0] * 2.8013999999999999e+01 * CWinv;
  y[1] = c[1] * 2.0160000000000000e+00 * CWinv;
  y[2] = c[2] * 1.0080000000000000e+00 * CWinv;
  y[3] = c[3] * 3.1998000000000001e+01 * CWinv;
  y[4] = c[4] * 1.5999000000000001e+01 * CWinv;
  y[5] = c[5] * 1.8015000000000001e+01 * CWinv;
  y[6] = c[6] * 1.7007000000000001e+01 * CWinv;
  y[7] = c[7] * 3.4014000000000003e+01 * CWinv;
  y[8] = c[8] * 3.3006000000000000e+01 * CWinv;
  y[9] = c[9] * 2.8009999999999998e+01 * CWinv;
  y[10] = c[10] * 4.4009000000000000e+01 * CWinv;
  y[11] = c[11] * 1.6042999999999999e+01 * CWinv;
  y[12] = c[12] * 1.5035000000000000e+01 * CWinv;
  y[13] = c[13] * 1.4026999999999999e+01 * CWinv;
  y[14] = c[14] * 1.4026999999999999e+01 * CWinv;
  y[15] = c[15] * 1.2010999999999999e+01 * CWinv;
  y[16] = c[16] * 1.3018999999999998e+01 * CWinv;
  y[17] = c[17] * 3.2042000000000002e+01 * CWinv;
  y[18] = c[18] * 3.1033999999999999e+01 * CWinv;
  y[19] = c[19] * 3.1033999999999999e+01 * CWinv;
  y[20] = c[20] * 3.0026000000000000e+01 * CWinv;
  y[21] = c[21] * 2.9018000000000001e+01 * CWinv;
  y[22] = c[22] * 2.9017451420091128e+01 * CWinv;
  y[23] = c[23] * 1.9022451420091127e+01 * CWinv;
  y[24] = c[24] * 3.1998548579908874e+01 * CWinv;
  y[25] = c[25] * 5.4857990887282834e-04 * CWinv;
}

// get specific heat at constant volume as a function
// of T for all species (molar units)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCVML(const amrex::Real T, amrex::Real cvml[])
{
  cv_R(cvml, T);

  // convert to chemkin units
  for (int id = 0; id < 26; ++id) {
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
  for (int id = 0; id < 26; ++id) {
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
  for (int id = 0; id < 26; ++id) {
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
  for (int id = 0; id < 26; ++id) {
    hml[id] *= RT;
  }
}

// Returns the standard-state entropies in molar units
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSML(const amrex::Real T, amrex::Real sml[])
{
  speciesEntropy(sml, T);

  // convert to chemkin units
  for (int id = 0; id < 26; ++id) {
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
  cvms[0] *= 2.967966951578939e+06;  // N2
  cvms[1] *= 4.124237409798234e+07;  // H2
  cvms[2] *= 8.248474819596468e+07;  // H
  cvms[3] *= 2.598431970171023e+06;  // O2
  cvms[4] *= 5.196863940342046e+06;  // O
  cvms[5] *= 4.615299815794193e+06;  // H2O
  cvms[6] *= 4.888847308845322e+06;  // OH
  cvms[7] *= 2.444423654422661e+06;  // H2O2
  cvms[8] *= 2.519076112874398e+06;  // HO2
  cvms[9] *= 2.968390795484913e+06;  // CO
  cvms[10] *= 1.889264154639560e+06; // CO2
  cvms[11] *= 5.182610869633635e+06; // CH4
  cvms[12] *= 5.530071578419182e+06; // CH3
  cvms[13] *= 5.927470320206203e+06; // CH2
  cvms[14] *= 5.927470320206203e+06; // CH2(S)
  cvms[15] *= 6.922373339566431e+06; // C
  cvms[16] *= 6.386406496776436e+06; // CH
  cvms[17] *= 2.594863809423020e+06; // CH3OH
  cvms[18] *= 2.679146297013998e+06; // CH3O
  cvms[19] *= 2.679146297013998e+06; // CH2OH
  cvms[20] *= 2.769087663409458e+06; // CH2O
  cvms[21] *= 2.865277627042952e+06; // HCO
  cvms[22] *= 2.865331795609199e+06; // CHO+
  cvms[23] *= 4.370868104502911e+06; // H3O+
  cvms[24] *= 2.598387422913829e+06; // O2-
  cvms[25] *= 1.515633818095350e+11; // E
}

// Returns the specific heats at constant pressure
// in mass units (Eq. 26)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCPMS(const amrex::Real T, amrex::Real cpms[])
{
  cp_R(cpms, T);
  // multiply by R/molecularweight
  cpms[0] *= 2.967966951578939e+06;  // N2
  cpms[1] *= 4.124237409798234e+07;  // H2
  cpms[2] *= 8.248474819596468e+07;  // H
  cpms[3] *= 2.598431970171023e+06;  // O2
  cpms[4] *= 5.196863940342046e+06;  // O
  cpms[5] *= 4.615299815794193e+06;  // H2O
  cpms[6] *= 4.888847308845322e+06;  // OH
  cpms[7] *= 2.444423654422661e+06;  // H2O2
  cpms[8] *= 2.519076112874398e+06;  // HO2
  cpms[9] *= 2.968390795484913e+06;  // CO
  cpms[10] *= 1.889264154639560e+06; // CO2
  cpms[11] *= 5.182610869633635e+06; // CH4
  cpms[12] *= 5.530071578419182e+06; // CH3
  cpms[13] *= 5.927470320206203e+06; // CH2
  cpms[14] *= 5.927470320206203e+06; // CH2(S)
  cpms[15] *= 6.922373339566431e+06; // C
  cpms[16] *= 6.386406496776436e+06; // CH
  cpms[17] *= 2.594863809423020e+06; // CH3OH
  cpms[18] *= 2.679146297013998e+06; // CH3O
  cpms[19] *= 2.679146297013998e+06; // CH2OH
  cpms[20] *= 2.769087663409458e+06; // CH2O
  cpms[21] *= 2.865277627042952e+06; // HCO
  cpms[22] *= 2.865331795609199e+06; // CHO+
  cpms[23] *= 4.370868104502911e+06; // H3O+
  cpms[24] *= 2.598387422913829e+06; // O2-
  cpms[25] *= 1.515633818095350e+11; // E
}

// Returns internal energy in mass units (Eq 30.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKUMS(const amrex::Real T, amrex::Real ums[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesInternalEnergy(ums, T);

  for (int i = 0; i < 26; i++) {
    ums[i] *= RT * imw(i);
  }
}

// Returns enthalpy in mass units (Eq 27.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKHMS(const amrex::Real T, amrex::Real hms[])
{
  amrex::Real RT = 8.31446261815324e+07 * T; // R*T

  speciesEnthalpy(hms, T);

  for (int i = 0; i < 26; i++) {
    hms[i] *= RT * imw(i);
  }
}

// Returns the entropies in mass units (Eq 28.)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKSMS(const amrex::Real T, amrex::Real sms[])
{
  speciesEntropy(sms, T);
  // multiply by R/molecularweight
  sms[0] *= 2.967966951578939e+06;  // N2
  sms[1] *= 4.124237409798234e+07;  // H2
  sms[2] *= 8.248474819596468e+07;  // H
  sms[3] *= 2.598431970171023e+06;  // O2
  sms[4] *= 5.196863940342046e+06;  // O
  sms[5] *= 4.615299815794193e+06;  // H2O
  sms[6] *= 4.888847308845322e+06;  // OH
  sms[7] *= 2.444423654422661e+06;  // H2O2
  sms[8] *= 2.519076112874398e+06;  // HO2
  sms[9] *= 2.968390795484913e+06;  // CO
  sms[10] *= 1.889264154639560e+06; // CO2
  sms[11] *= 5.182610869633635e+06; // CH4
  sms[12] *= 5.530071578419182e+06; // CH3
  sms[13] *= 5.927470320206203e+06; // CH2
  sms[14] *= 5.927470320206203e+06; // CH2(S)
  sms[15] *= 6.922373339566431e+06; // C
  sms[16] *= 6.386406496776436e+06; // CH
  sms[17] *= 2.594863809423020e+06; // CH3OH
  sms[18] *= 2.679146297013998e+06; // CH3O
  sms[19] *= 2.679146297013998e+06; // CH2OH
  sms[20] *= 2.769087663409458e+06; // CH2O
  sms[21] *= 2.865277627042952e+06; // HCO
  sms[22] *= 2.865331795609199e+06; // CHO+
  sms[23] *= 4.370868104502911e+06; // H3O+
  sms[24] *= 2.598387422913829e+06; // O2-
  sms[25] *= 1.515633818095350e+11; // E
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

  // reaction 6: H2O2 (+M) <=> 2 OH (+M)
  qf[0] = sc[7];
  qr[0] = (sc[6] * sc[6]);

  // reaction 19: H + O2 (+M) <=> HO2 (+M)
  qf[1] = sc[2] * sc[3];
  qr[1] = sc[8];

  // reaction 24: CH3 + H (+M) <=> CH4 (+M)
  qf[2] = sc[2] * sc[12];
  qr[2] = sc[11];

  // reaction 43: CH2 + H (+M) <=> CH3 (+M)
  qf[3] = sc[2] * sc[13];
  qr[3] = sc[12];

  // reaction 66: CH3OH (+M) <=> CH3 + OH (+M)
  qf[4] = sc[17];
  qr[4] = sc[6] * sc[12];

  // reaction 67: CH3OH (+M) <=> CH2(S) + H2O (+M)
  qf[5] = sc[17];
  qr[5] = sc[5] * sc[14];

  // reaction 106: CH2O + H (+M) <=> CH2OH (+M)
  qf[6] = sc[2] * sc[20];
  qr[6] = sc[19];

  // reaction 107: CH3O (+M) <=> CH2O + H (+M)
  qf[7] = sc[18];
  qr[7] = sc[2] * sc[20];

  // reaction 20: CO + O (+M) <=> CO2 (+M)
  qf[8] = sc[4] * sc[9];
  qr[8] = sc[10];

  // reaction 3: H + OH + M <=> H2O + M
  qf[9] = sc[2] * sc[6];
  qr[9] = sc[5];

  // reaction 5: H + O + M <=> OH + M
  qf[10] = sc[2] * sc[4];
  qr[10] = sc[6];

  // reaction 97: HCO + M <=> CO + H + M
  qf[11] = sc[21];
  qr[11] = sc[2] * sc[9];

  // reaction 108: O + OH + M <=> HO2 + M
  qf[12] = sc[4] * sc[6];
  qr[12] = sc[8];

  // reaction 120: CH3 + O + M => CH3O + M
  qf[13] = sc[4] * sc[12];
  qr[13] = 0.0;

  // reaction 131: E + O2 + O2 <=> O2- + O2
  qf[14] = (sc[3] * sc[3]) * sc[25];
  qr[14] = sc[3] * sc[24];

  // reaction 132: E + O2 + H2O <=> O2- + H2O
  qf[15] = sc[3] * sc[5] * sc[25];
  qr[15] = sc[5] * sc[24];

  // reaction 133: E + O2 + N2 <=> O2- + N2
  qf[16] = sc[0] * sc[3] * sc[25];
  qr[16] = sc[0] * sc[24];

  // reaction 0: H2 + O <=> H + OH
  qf[17] = sc[1] * sc[4];
  qr[17] = sc[2] * sc[6];

  // reaction 1: H2 + OH <=> H + H2O
  qf[18] = sc[1] * sc[6];
  qr[18] = sc[2] * sc[5];

  // reaction 2: H + O2 <=> O + OH
  qf[19] = sc[2] * sc[3];
  qr[19] = sc[4] * sc[6];

  // reaction 4: H2O + O <=> 2 OH
  qf[20] = sc[4] * sc[5];
  qr[20] = (sc[6] * sc[6]);

  // reaction 7: H + H2O2 <=> H2O + OH
  qf[21] = sc[2] * sc[7];
  qr[21] = sc[5] * sc[6];

  // reaction 8: H + H2O2 <=> H2 + HO2
  qf[22] = sc[2] * sc[7];
  qr[22] = sc[1] * sc[8];

  // reaction 9: H2O2 + O <=> HO2 + OH
  qf[23] = sc[4] * sc[7];
  qr[23] = sc[6] * sc[8];

  // reaction 10: H2O2 + OH <=> H2O + HO2
  qf[24] = sc[6] * sc[7];
  qr[24] = sc[5] * sc[8];

  // reaction 11: H2O2 + OH <=> H2O + HO2
  qf[25] = sc[6] * sc[7];
  qr[25] = sc[5] * sc[8];

  // reaction 12: H + HO2 <=> 2 OH
  qf[26] = sc[2] * sc[8];
  qr[26] = (sc[6] * sc[6]);

  // reaction 13: H + HO2 <=> H2 + O2
  qf[27] = sc[2] * sc[8];
  qr[27] = sc[1] * sc[3];

  // reaction 14: HO2 + O <=> O2 + OH
  qf[28] = sc[4] * sc[8];
  qr[28] = sc[3] * sc[6];

  // reaction 15: HO2 + OH <=> H2O + O2
  qf[29] = sc[6] * sc[8];
  qr[29] = sc[3] * sc[5];

  // reaction 16: HO2 + OH <=> H2O + O2
  qf[30] = sc[6] * sc[8];
  qr[30] = sc[3] * sc[5];

  // reaction 17: 2 HO2 <=> H2O2 + O2
  qf[31] = (sc[8] * sc[8]);
  qr[31] = sc[3] * sc[7];

  // reaction 18: 2 HO2 <=> H2O2 + O2
  qf[32] = (sc[8] * sc[8]);
  qr[32] = sc[3] * sc[7];

  // reaction 21: CO + OH <=> CO2 + H
  qf[33] = sc[6] * sc[9];
  qr[33] = sc[2] * sc[10];

  // reaction 22: CO + OH <=> CO2 + H
  qf[34] = sc[6] * sc[9];
  qr[34] = sc[2] * sc[10];

  // reaction 23: CO + HO2 <=> CO2 + OH
  qf[35] = sc[8] * sc[9];
  qr[35] = sc[6] * sc[10];

  // reaction 25: CH4 + H <=> CH3 + H2
  qf[36] = sc[2] * sc[11];
  qr[36] = sc[1] * sc[12];

  // reaction 26: CH4 + O <=> CH3 + OH
  qf[37] = sc[4] * sc[11];
  qr[37] = sc[6] * sc[12];

  // reaction 27: CH4 + OH <=> CH3 + H2O
  qf[38] = sc[6] * sc[11];
  qr[38] = sc[5] * sc[12];

  // reaction 28: CH4 + HO2 <=> CH3 + H2O2
  qf[39] = sc[8] * sc[11];
  qr[39] = sc[7] * sc[12];

  // reaction 29: CH3 + HO2 <=> CH4 + O2
  qf[40] = sc[8] * sc[12];
  qr[40] = sc[3] * sc[11];

  // reaction 30: CH2 + CH4 <=> 2 CH3
  qf[41] = sc[11] * sc[13];
  qr[41] = (sc[12] * sc[12]);

  // reaction 31: CH2(S) + N2 <=> CH2 + N2
  qf[42] = sc[0] * sc[14];
  qr[42] = sc[0] * sc[13];

  // reaction 32: CH2(S) + H2O <=> CH2 + H2O
  qf[43] = sc[5] * sc[14];
  qr[43] = sc[5] * sc[13];

  // reaction 33: CH2(S) + CO <=> CH2 + CO
  qf[44] = sc[9] * sc[14];
  qr[44] = sc[9] * sc[13];

  // reaction 34: CH2(S) + CO2 <=> CH2 + CO2
  qf[45] = sc[10] * sc[14];
  qr[45] = sc[10] * sc[13];

  // reaction 35: CH2(S) + O2 => CO + H + OH
  qf[46] = sc[3] * sc[14];
  qr[46] = 0.0;

  // reaction 36: CH2(S) + O2 <=> CO + H2O
  qf[47] = sc[3] * sc[14];
  qr[47] = sc[5] * sc[9];

  // reaction 37: CH2(S) + O <=> CO + H2
  qf[48] = sc[4] * sc[14];
  qr[48] = sc[1] * sc[9];

  // reaction 38: CH2(S) + O <=> H + HCO
  qf[49] = sc[4] * sc[14];
  qr[49] = sc[2] * sc[21];

  // reaction 39: CH2(S) + H2 <=> CH3 + H
  qf[50] = sc[1] * sc[14];
  qr[50] = sc[2] * sc[12];

  // reaction 40: CH2(S) + H <=> CH + H2
  qf[51] = sc[2] * sc[14];
  qr[51] = sc[1] * sc[16];

  // reaction 41: CH2(S) + OH <=> CH2O + H
  qf[52] = sc[6] * sc[14];
  qr[52] = sc[2] * sc[20];

  // reaction 42: CH2(S) + CO2 <=> CH2O + CO
  qf[53] = sc[10] * sc[14];
  qr[53] = sc[9] * sc[20];

  // reaction 44: CH2 + O2 <=> HCO + OH
  qf[54] = sc[3] * sc[13];
  qr[54] = sc[6] * sc[21];

  // reaction 45: CH2 + O2 => CO2 + 2 H
  qf[55] = sc[3] * sc[13];
  qr[55] = 0.0;

  // reaction 46: CH2 + O => CO + 2 H
  qf[56] = sc[4] * sc[13];
  qr[56] = 0.0;

  // reaction 47: CH2 + H <=> CH + H2
  qf[57] = sc[2] * sc[13];
  qr[57] = sc[1] * sc[16];

  // reaction 48: CH2 + OH <=> CH + H2O
  qf[58] = sc[6] * sc[13];
  qr[58] = sc[5] * sc[16];

  // reaction 49: CH + O2 <=> HCO + O
  qf[59] = sc[3] * sc[16];
  qr[59] = sc[4] * sc[21];

  // reaction 50: CH + O <=> CO + H
  qf[60] = sc[4] * sc[16];
  qr[60] = sc[2] * sc[9];

  // reaction 51: CH + H <=> C + H2
  qf[61] = sc[2] * sc[16];
  qr[61] = sc[1] * sc[15];

  // reaction 52: CH + OH <=> H + HCO
  qf[62] = sc[6] * sc[16];
  qr[62] = sc[2] * sc[21];

  // reaction 53: CH + H2O <=> CH2O + H
  qf[63] = sc[5] * sc[16];
  qr[63] = sc[2] * sc[20];

  // reaction 54: CH + CO2 <=> CO + HCO
  qf[64] = sc[10] * sc[16];
  qr[64] = sc[9] * sc[21];

  // reaction 55: C + OH <=> CO + H
  qf[65] = sc[6] * sc[15];
  qr[65] = sc[2] * sc[9];

  // reaction 56: C + O2 <=> CO + O
  qf[66] = sc[3] * sc[15];
  qr[66] = sc[4] * sc[9];

  // reaction 57: CH3 + O2 <=> CH3O + O
  qf[67] = sc[3] * sc[12];
  qr[67] = sc[4] * sc[18];

  // reaction 58: CH3 + O2 <=> CH2O + OH
  qf[68] = sc[3] * sc[12];
  qr[68] = sc[6] * sc[20];

  // reaction 59: CH3 + O <=> CH2O + H
  qf[69] = sc[4] * sc[12];
  qr[69] = sc[2] * sc[20];

  // reaction 60: CH3 + OH <=> CH2(S) + H2O
  qf[70] = sc[6] * sc[12];
  qr[70] = sc[5] * sc[14];

  // reaction 61: CH3 + OH <=> CH2O + H2
  qf[71] = sc[6] * sc[12];
  qr[71] = sc[1] * sc[20];

  // reaction 62: CH3 + OH <=> CH2OH + H
  qf[72] = sc[6] * sc[12];
  qr[72] = sc[2] * sc[19];

  // reaction 63: CH3 + OH <=> CH3O + H
  qf[73] = sc[6] * sc[12];
  qr[73] = sc[2] * sc[18];

  // reaction 64: CH3 + OH <=> CH2 + H2O
  qf[74] = sc[6] * sc[12];
  qr[74] = sc[5] * sc[13];

  // reaction 65: CH3 + HO2 <=> CH3O + OH
  qf[75] = sc[8] * sc[12];
  qr[75] = sc[6] * sc[18];

  // reaction 68: CH3OH + H <=> CH3O + H2
  qf[76] = sc[2] * sc[17];
  qr[76] = sc[1] * sc[18];

  // reaction 69: CH3OH + H <=> CH2OH + H2
  qf[77] = sc[2] * sc[17];
  qr[77] = sc[1] * sc[19];

  // reaction 70: CH3OH + O <=> CH3O + OH
  qf[78] = sc[4] * sc[17];
  qr[78] = sc[6] * sc[18];

  // reaction 71: CH3OH + O <=> CH2OH + OH
  qf[79] = sc[4] * sc[17];
  qr[79] = sc[6] * sc[19];

  // reaction 72: CH3OH + OH <=> CH3O + H2O
  qf[80] = sc[6] * sc[17];
  qr[80] = sc[5] * sc[18];

  // reaction 73: CH3OH + OH <=> CH2OH + H2O
  qf[81] = sc[6] * sc[17];
  qr[81] = sc[5] * sc[19];

  // reaction 74: CH3OH + O2 <=> CH3O + HO2
  qf[82] = sc[3] * sc[17];
  qr[82] = sc[8] * sc[18];

  // reaction 75: CH3OH + HO2 <=> CH3O + H2O2
  qf[83] = sc[8] * sc[17];
  qr[83] = sc[7] * sc[18];

  // reaction 76: CH3OH + HO2 <=> CH2OH + H2O2
  qf[84] = sc[8] * sc[17];
  qr[84] = sc[7] * sc[19];

  // reaction 77: CH3 + CH3OH <=> CH2OH + CH4
  qf[85] = sc[12] * sc[17];
  qr[85] = sc[11] * sc[19];

  // reaction 78: CH3 + CH3OH <=> CH3O + CH4
  qf[86] = sc[12] * sc[17];
  qr[86] = sc[11] * sc[18];

  // reaction 79: CH2OH + O2 <=> CH2O + HO2
  qf[87] = sc[3] * sc[19];
  qr[87] = sc[8] * sc[20];

  // reaction 80: CH2OH + O2 <=> CH2O + HO2
  qf[88] = sc[3] * sc[19];
  qr[88] = sc[8] * sc[20];

  // reaction 81: CH2OH + H <=> CH2O + H2
  qf[89] = sc[2] * sc[19];
  qr[89] = sc[1] * sc[20];

  // reaction 82: CH2OH + HO2 <=> CH2O + H2O2
  qf[90] = sc[8] * sc[19];
  qr[90] = sc[7] * sc[20];

  // reaction 83: CH2OH + HCO <=> 2 CH2O
  qf[91] = sc[19] * sc[21];
  qr[91] = (sc[20] * sc[20]);

  // reaction 84: CH2OH + OH <=> CH2O + H2O
  qf[92] = sc[6] * sc[19];
  qr[92] = sc[5] * sc[20];

  // reaction 85: CH2OH + O <=> CH2O + OH
  qf[93] = sc[4] * sc[19];
  qr[93] = sc[6] * sc[20];

  // reaction 86: CH3O + O2 <=> CH2O + HO2
  qf[94] = sc[3] * sc[18];
  qr[94] = sc[8] * sc[20];

  // reaction 87: CH3O + H <=> CH2O + H2
  qf[95] = sc[2] * sc[18];
  qr[95] = sc[1] * sc[20];

  // reaction 88: CH3O + HO2 <=> CH2O + H2O2
  qf[96] = sc[8] * sc[18];
  qr[96] = sc[7] * sc[20];

  // reaction 89: CH3 + CH3O <=> CH2O + CH4
  qf[97] = sc[12] * sc[18];
  qr[97] = sc[11] * sc[20];

  // reaction 90: 2 CH3O <=> CH2O + CH3OH
  qf[98] = (sc[18] * sc[18]);
  qr[98] = sc[17] * sc[20];

  // reaction 91: CH2O + O <=> HCO + OH
  qf[99] = sc[4] * sc[20];
  qr[99] = sc[6] * sc[21];

  // reaction 92: CH2O + H <=> H2 + HCO
  qf[100] = sc[2] * sc[20];
  qr[100] = sc[1] * sc[21];

  // reaction 93: CH2O + OH <=> H2O + HCO
  qf[101] = sc[6] * sc[20];
  qr[101] = sc[5] * sc[21];

  // reaction 94: CH2O + HO2 <=> H2O2 + HCO
  qf[102] = sc[8] * sc[20];
  qr[102] = sc[7] * sc[21];

  // reaction 95: CH2O + CH3 <=> CH4 + HCO
  qf[103] = sc[12] * sc[20];
  qr[103] = sc[11] * sc[21];

  // reaction 96: CH2O + CH3O <=> CH3OH + HCO
  qf[104] = sc[18] * sc[20];
  qr[104] = sc[17] * sc[21];

  // reaction 98: HCO + O2 <=> CO + HO2
  qf[105] = sc[3] * sc[21];
  qr[105] = sc[8] * sc[9];

  // reaction 99: HCO + O <=> CO + OH
  qf[106] = sc[4] * sc[21];
  qr[106] = sc[6] * sc[9];

  // reaction 100: H + HCO <=> CO + H2
  qf[107] = sc[2] * sc[21];
  qr[107] = sc[1] * sc[9];

  // reaction 101: HCO + OH <=> CO + H2O
  qf[108] = sc[6] * sc[21];
  qr[108] = sc[5] * sc[9];

  // reaction 102: CH3 + HCO <=> CH4 + CO
  qf[109] = sc[12] * sc[21];
  qr[109] = sc[9] * sc[11];

  // reaction 103: 2 HCO <=> CH2O + CO
  qf[110] = (sc[21] * sc[21]);
  qr[110] = sc[9] * sc[20];

  // reaction 104: HCO + O <=> CO2 + H
  qf[111] = sc[4] * sc[21];
  qr[111] = sc[2] * sc[10];

  // reaction 105: HCO + HO2 => CO2 + H + OH
  qf[112] = sc[8] * sc[21];
  qr[112] = 0.0;

  // reaction 109: CH + OH <=> C + H2O
  qf[113] = sc[6] * sc[16];
  qr[113] = sc[5] * sc[15];

  // reaction 110: CH2 + HCO <=> CH3 + CO
  qf[114] = sc[13] * sc[21];
  qr[114] = sc[9] * sc[12];

  // reaction 111: CH2 + O <=> CO + H2
  qf[115] = sc[4] * sc[13];
  qr[115] = sc[1] * sc[9];

  // reaction 112: CH2 + OH <=> CH2O + H
  qf[116] = sc[6] * sc[13];
  qr[116] = sc[2] * sc[20];

  // reaction 113: CH2 + CO2 <=> CH2O + CO
  qf[117] = sc[10] * sc[13];
  qr[117] = sc[9] * sc[20];

  // reaction 114: CH2(S) + O <=> CO + 2 H
  qf[118] = sc[4] * sc[14];
  qr[118] = (sc[2] * sc[2]) * sc[9];

  // reaction 115: CH2(S) + CH4 <=> 2 CH3
  qf[119] = sc[11] * sc[14];
  qr[119] = (sc[12] * sc[12]);

  // reaction 116: CH3OH + H => CH3 + H2O
  qf[120] = sc[2] * sc[17];
  qr[120] = 0.0;

  // reaction 117: CH2O + O => CO2 + 2 H
  qf[121] = sc[4] * sc[20];
  qr[121] = 0.0;

  // reaction 118: CH2O + OH => CO2 + H + H2
  qf[122] = sc[6] * sc[20];
  qr[122] = 0.0;

  // reaction 119: CH3O + CO <=> CH3 + CO2
  qf[123] = sc[9] * sc[18];
  qr[123] = sc[10] * sc[12];

  // reaction 121: HCO + HO2 <=> CO + H2O2
  qf[124] = sc[8] * sc[21];
  qr[124] = sc[7] * sc[9];

  // reaction 122: CH3O + OH => CH2O + H2O
  qf[125] = sc[6] * sc[18];
  qr[125] = 0.0;

  // reaction 123: CH2OH + CH3 => CH2O + CH4
  qf[126] = sc[12] * sc[19];
  qr[126] = 0.0;

  // reaction 124: CH + O <=> CHO+ + E
  qf[127] = sc[4] * sc[16];
  qr[127] = sc[22] * sc[25];

  // reaction 125: CHO+ + H2O <=> CO + H3O+
  qf[128] = sc[5] * sc[22];
  qr[128] = sc[9] * sc[23];

  // reaction 126: E + H3O+ <=> H + H2O
  qf[129] = sc[23] * sc[25];
  qr[129] = sc[2] * sc[5];

  // reaction 127: E + H3O+ <=> 2 H + OH
  qf[130] = sc[23] * sc[25];
  qr[130] = (sc[2] * sc[2]) * sc[6];

  // reaction 128: E + H3O+ <=> H2 + OH
  qf[131] = sc[23] * sc[25];
  qr[131] = sc[1] * sc[6];

  // reaction 129: H2 + O2- <=> E + H2O2
  qf[132] = sc[1] * sc[24];
  qr[132] = sc[7] * sc[25];

  // reaction 130: H + O2- <=> E + HO2
  qf[133] = sc[2] * sc[24];
  qr[133] = sc[8] * sc[25];

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 26; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[26];
  gibbs(g_RT, T);
  // Kc from shared exp(-g/RT) factors where no factor or partial product can
  // overflow (checked by CEPTR on this T range); exp of the sum otherwise and
  // in single precision
  const bool kc_fast =
    sizeof(amrex::Real) == sizeof(double) && T >= 200.0 && T <= 10000.0;
  amrex::Real ge[26] = {}, inv_ge[26] = {};
  if (kc_fast) {
    for (int i = 0; i < 26; ++i) {
      ge[i] = exp(-g_RT[i]);
      inv_ge[i] = 1.0 / ge[i];
    }
  }

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  amrex::Real refC = 101325 / 8.31446 * invT;
  amrex::Real refCinv = 1 / refC;

  // Evaluate the kfs
  amrex::Real k_f, Corr;
  amrex::Real redP, F, logPred, logFcent, troe_c, troe_n, troe, F_troe;

  // reaction 0:  H2 + O <=> H + OH
  k_f = 0.0508 * exp((2.67) * logT - (3166.2392612433) * invT);
  qf[17] *= k_f;
  qr[17] *= k_f * (kc_fast ? (ge[1] * inv_ge[6] * ge[4] * inv_ge[2])
                           : exp(-(g_RT[1] - g_RT[2] + g_RT[4] - g_RT[6])));
  // reaction 1:  H2 + OH <=> H + H2O
  k_f = 43800000 * exp(-(3517.48449397499) * invT);
  qf[18] *= k_f;
  qr[18] *= k_f * (kc_fast ? (ge[1] * ge[6] * inv_ge[2] * inv_ge[5])
                           : exp(-(g_RT[1] - g_RT[2] - g_RT[5] + g_RT[6])));
  // reaction 2:  H + O2 <=> O + OH
  k_f = 114000000 * exp(-(7692.16995349094) * invT);
  qf[19] *= k_f;
  qr[19] *= k_f * (kc_fast ? (inv_ge[6] * ge[3] * ge[2] * inv_ge[4])
                           : exp(-(g_RT[2] + g_RT[3] - g_RT[4] - g_RT[6])));
  // reaction 3:  H + OH + M <=> H2O + M
  k_f = 35000000000 * exp((-2) * logT);
  Corr = mixture + (-0.27) * sc[1] + (2.65) * sc[5] + sc[11];
  qf[9] *= Corr * k_f;
  qr[9] *= Corr * k_f *
           (kc_fast ? (ge[6] * ge[2] * inv_ge[5])
                    : exp(-(g_RT[2] - g_RT[5] + g_RT[6]))) *
           (refC);
  // reaction 4:  H2O + O <=> 2 OH
  k_f = 67 * exp((1.704) * logT - (7541.60752708217) * invT);
  qf[20] *= k_f;
  qr[20] *= k_f * (kc_fast ? (inv_ge[6] * inv_ge[6] * ge[4] * ge[5])
                           : exp(-(g_RT[4] + g_RT[5] - 2.000000 * g_RT[6])));
  // reaction 5:  H + O + M <=> OH + M
  k_f = 4714000 * exp((-1) * logT);
  Corr =
    mixture + (1.5) * sc[1] + (11) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
  qf[10] *= Corr * k_f;
  qr[10] *= Corr * k_f *
            (kc_fast ? (inv_ge[6] * ge[2] * ge[4])
                     : exp(-(g_RT[2] + g_RT[4] - g_RT[6]))) *
            (refC);
  // reaction 6:  H2O2 (+M) <=> 2 OH (+M)
  k_f = 2000000000000 * exp((0.9) * logT - (24531.3092413143) * invT);
  Corr = mixture + (0.5) * sc[0] + (2.7) * sc[1] + (0.2) * sc[3] +
         (6.65) * sc[5] + (6.7) * sc[7] + (1.8) * sc[9] + (0.6) * sc[10];
  redP = Corr / k_f * 2.49e+18 * exp(-2.3 * logT - (24531.3092413143) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (0.57) * exp(-T * 1e+30) + 0.43 * exp(-T * 1e-30) + exp(-1e-16 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[0] *= Corr * k_f;
  qr[0] *= Corr * k_f *
           (kc_fast ? (inv_ge[6] * inv_ge[6] * ge[7])
                    : exp(-(-2.000000 * g_RT[6] + g_RT[7]))) *
           (refCinv);
  // reaction 7:  H + H2O2 <=> H2O + OH
  k_f = 24100000 * exp(-(1997.77016324474) * invT);
  qf[21] *= k_f;
  qr[21] *= k_f * (kc_fast ? (inv_ge[6] * ge[7] * ge[2] * inv_ge[5])
                           : exp(-(g_RT[2] - g_RT[5] - g_RT[6] + g_RT[7])));
  // reaction 8:  H + H2O2 <=> H2 + HO2
  k_f = 21500 * exp((1) * logT - (3019.29999482832) * invT);
  qf[22] *= k_f;
  qr[22] *= k_f * (kc_fast ? (inv_ge[1] * inv_ge[8] * ge[7] * ge[2])
                           : exp(-(-g_RT[1] + g_RT[2] + g_RT[7] - g_RT[8])));
  // reaction 9:  H2O2 + O <=> HO2 + OH
  k_f = 9.55 * exp((2) * logT - (1997.77016324474) * invT);
  qf[23] *= k_f;
  qr[23] *= k_f * (kc_fast ? (inv_ge[6] * inv_ge[8] * ge[7] * ge[4])
                           : exp(-(g_RT[4] - g_RT[6] + g_RT[7] - g_RT[8])));
  // reaction 10:  H2O2 + OH <=> H2O + HO2
  k_f = 1740000 * exp(-(160.022899725901) * invT);
  qf[24] *= k_f;
  qr[24] *= k_f * (kc_fast ? (ge[6] * inv_ge[8] * ge[7] * inv_ge[5])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[7] - g_RT[8])));
  // reaction 11:  H2O2 + OH <=> H2O + HO2
  k_f = 75900000 * exp(-(3657.88194373451) * invT);
  qf[25] *= k_f;
  qr[25] *= k_f * (kc_fast ? (ge[6] * inv_ge[8] * ge[7] * inv_ge[5])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[7] - g_RT[8])));
  // reaction 12:  H + HO2 <=> 2 OH
  k_f = 70790000 * exp(-(148.448916412392) * invT);
  qf[26] *= k_f;
  qr[26] *= k_f * (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[6] * ge[2])
                           : exp(-(g_RT[2] - 2.000000 * g_RT[6] + g_RT[8])));
  // reaction 13:  H + HO2 <=> H2 + O2
  k_f = 11402 * exp((1.083) * logT - (278.671325189338) * invT);
  qf[27] *= k_f;
  qr[27] *= k_f * (kc_fast ? (inv_ge[1] * ge[8] * inv_ge[3] * ge[2])
                           : exp(-(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[8])));
  // reaction 14:  HO2 + O <=> O2 + OH
  k_f = 32500000;
  qf[28] *= k_f;
  qr[28] *= k_f * (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[3] * ge[4])
                           : exp(-(-g_RT[3] + g_RT[4] - g_RT[6] + g_RT[8])));
  // reaction 15:  HO2 + OH <=> H2O + O2
  k_f = 7000000 * exp(-(-549.995687057926) * invT);
  qf[29] *= k_f;
  qr[29] *= k_f * (kc_fast ? (ge[6] * inv_ge[3] * ge[8] * inv_ge[5])
                           : exp(-(-g_RT[3] - g_RT[5] + g_RT[6] + g_RT[8])));
  // reaction 16:  HO2 + OH <=> H2O + O2
  k_f = 450000000 * exp(-(5499.95687057926) * invT);
  qf[30] *= k_f;
  qr[30] *= k_f * (kc_fast ? (ge[6] * inv_ge[3] * ge[8] * inv_ge[5])
                           : exp(-(-g_RT[3] - g_RT[5] + g_RT[6] + g_RT[8])));
  // reaction 17:  2 HO2 <=> H2O2 + O2
  k_f = 100000000 * exp(-(5555.95482115001) * invT);
  qf[31] *= k_f;
  qr[31] *= k_f * (kc_fast ? (inv_ge[3] * ge[8] * ge[8] * inv_ge[7])
                           : exp(-(-g_RT[3] - g_RT[7] + 2.000000 * g_RT[8])));
  // reaction 18:  2 HO2 <=> H2O2 + O2
  k_f = 190000 * exp(-(-708.992024785586) * invT);
  qf[32] *= k_f;
  qr[32] *= k_f * (kc_fast ? (inv_ge[3] * ge[8] * ge[8] * inv_ge[7])
                           : exp(-(-g_RT[3] - g_RT[7] + 2.000000 * g_RT[8])));
  // reaction 19:  H + O2 (+M) <=> HO2 (+M)
  k_f = 4650000 * exp((0.44) * logT);
  Corr = mixture + (0.3) * sc[1] + (9) * sc[5] + (0.9) * sc[9] +
         (2.8) * sc[10] + sc[11];
  redP = Corr / k_f * 17400000 * exp(-1.23 * logT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (0.33) * exp(-T * 1e+30) + 0.67 * exp(-T * 1e-30) + exp(-1e+30 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[1] *= Corr * k_f;
  qr[1] *= Corr * k_f *
           (kc_fast ? (ge[3] * inv_ge[8] * ge[2])
                    : exp(-(g_RT[2] + g_RT[3] - g_RT[8]))) *
           (refC);
  // reaction 20:  CO + O (+M) <=> CO2 (+M)
  k_f = 13620 * exp(-(1199.66853127845) * invT);
  Corr = mixture + sc[1] + (11) * sc[5] + (0.75) * sc[9] + (2.6) * sc[10];
  redP =
    Corr / k_f * 1170000000000 * exp(-2.79 * logT - (2108.98104638758) * invT);
  Corr = redP / (1. + redP);
  qf[8] *= Corr * k_f;
  qr[8] *= Corr * k_f *
           (kc_fast ? (ge[9] * ge[4] * inv_ge[10])
                    : exp(-(g_RT[4] + g_RT[9] - g_RT[10]))) *
           (refC);
  // reaction 21:  CO + OH <=> CO2 + H
  k_f = 0.07015 * exp((2.053) * logT - (-178.994168026739) * invT);
  qf[33] *= k_f;
  qr[33] *= k_f * (kc_fast ? (ge[6] * ge[9] * inv_ge[10] * inv_ge[2])
                           : exp(-(-g_RT[2] + g_RT[6] + g_RT[9] - g_RT[10])));
  // reaction 22:  CO + OH <=> CO2 + H
  k_f = 5757000 * exp((-0.664) * logT - (166.967289714006) * invT);
  qf[34] *= k_f;
  qr[34] *= k_f * (kc_fast ? (ge[6] * ge[9] * inv_ge[10] * inv_ge[2])
                           : exp(-(-g_RT[2] + g_RT[6] + g_RT[9] - g_RT[10])));
  // reaction 23:  CO + HO2 <=> CO2 + OH
  k_f = 0.157 * exp((2.18) * logT - (9027.70698453667) * invT);
  qf[35] *= k_f;
  qr[35] *= k_f * (kc_fast ? (inv_ge[6] * ge[8] * ge[9] * inv_ge[10])
                           : exp(-(-g_RT[6] + g_RT[8] + g_RT[9] - g_RT[10])));
  // reaction 24:  CH3 + H (+M) <=> CH4 (+M)
  k_f = 12700000000 * exp((-0.63) * logT - (192.731983003208) * invT);
  Corr = mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
  redP = Corr / k_f * 2.48e+21 * exp(-4.76 * logT - (1227.84866456352) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (0.217) * exp(-T * 0.0135135135135135) +
    0.783 * exp(-T * 0.000340020401224073) + exp(-6964 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[2] *= Corr * k_f;
  qr[2] *= Corr * k_f *
           (kc_fast ? (ge[12] * inv_ge[11] * ge[2])
                    : exp(-(g_RT[2] - g_RT[11] + g_RT[12]))) *
           (refC);
  // reaction 25:  CH4 + H <=> CH3 + H2
  k_f = 0.614 * exp((2.5) * logT - (4824.33817506985) * invT);
  qf[36] *= k_f;
  qr[36] *= k_f * (kc_fast ? (inv_ge[1] * ge[11] * ge[2] * inv_ge[12])
                           : exp(-(-g_RT[1] + g_RT[2] + g_RT[11] - g_RT[12])));
  // reaction 26:  CH4 + O <=> CH3 + OH
  k_f = 1020 * exp((1.5) * logT - (4327.66332592059) * invT);
  qf[37] *= k_f;
  qr[37] *= k_f * (kc_fast ? (inv_ge[6] * ge[11] * ge[4] * inv_ge[12])
                           : exp(-(g_RT[4] - g_RT[6] + g_RT[11] - g_RT[12])));
  // reaction 27:  CH4 + OH <=> CH3 + H2O
  k_f = 0.0583 * exp((2.6) * logT - (1102.04449811234) * invT);
  qf[38] *= k_f;
  qr[38] *= k_f * (kc_fast ? (ge[6] * inv_ge[12] * inv_ge[5] * ge[11])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[11] - g_RT[12])));
  // reaction 28:  CH4 + HO2 <=> CH3 + H2O2
  k_f = 1.13e-05 * exp((3.74) * logT - (10572.5821485572) * invT);
  qf[39] *= k_f;
  qr[39] *= k_f * (kc_fast ? (ge[8] * inv_ge[12] * inv_ge[7] * ge[11])
                           : exp(-(-g_RT[7] + g_RT[8] + g_RT[11] - g_RT[12])));
  // reaction 29:  CH3 + HO2 <=> CH4 + O2
  k_f = 0.116 * exp((2.23) * logT - (-1520.72076406186) * invT);
  qf[40] *= k_f;
  qr[40] *= k_f * (kc_fast ? (inv_ge[3] * ge[8] * ge[12] * inv_ge[11])
                           : exp(-(-g_RT[3] + g_RT[8] - g_RT[11] + g_RT[12])));
  // reaction 30:  CH2 + CH4 <=> 2 CH3
  k_f = 2.46 * exp((2) * logT - (4161.60182620503) * invT);
  qf[41] *= k_f;
  qr[41] *= k_f * (kc_fast ? (inv_ge[12] * inv_ge[12] * ge[13] * ge[11])
                           : exp(-(g_RT[11] - 2.000000 * g_RT[12] + g_RT[13])));
  // reaction 31:  CH2(S) + N2 <=> CH2 + N2
  k_f = 15000000 * exp(-(301.929999482832) * invT);
  qf[42] *= k_f;
  qr[42] *= k_f * (kc_fast ? (inv_ge[13] * ge[14])
                           : exp(-(g_RT[0] - g_RT[0] - g_RT[13] + g_RT[14])));
  // reaction 32:  CH2(S) + H2O <=> CH2 + H2O
  k_f = 30000000;
  qf[43] *= k_f;
  qr[43] *= k_f * (kc_fast ? (inv_ge[13] * ge[14])
                           : exp(-(g_RT[5] - g_RT[5] - g_RT[13] + g_RT[14])));
  // reaction 33:  CH2(S) + CO <=> CH2 + CO
  k_f = 9000000;
  qf[44] *= k_f;
  qr[44] *= k_f * (kc_fast ? (inv_ge[13] * ge[14])
                           : exp(-(g_RT[9] - g_RT[9] - g_RT[13] + g_RT[14])));
  // reaction 34:  CH2(S) + CO2 <=> CH2 + CO2
  k_f = 7000000;
  qf[45] *= k_f;
  qr[45] *= k_f * (kc_fast ? (inv_ge[13] * ge[14])
                           : exp(-(g_RT[10] - g_RT[10] - g_RT[13] + g_RT[14])));
  // reaction 35:  CH2(S) + O2 => CO + H + OH
  k_f = 28000000;
  qf[46] *= k_f;
  qr[46] *=
    k_f *
    (kc_fast ? (inv_ge[6] * ge[3] * inv_ge[9] * inv_ge[2] * ge[14])
             : exp(-(-g_RT[2] + g_RT[3] - g_RT[6] - g_RT[9] + g_RT[14]))) *
    (refCinv);
  // reaction 36:  CH2(S) + O2 <=> CO + H2O
  k_f = 12000000;
  qf[47] *= k_f;
  qr[47] *= k_f * (kc_fast ? (ge[3] * inv_ge[9] * inv_ge[5] * ge[14])
                           : exp(-(g_RT[3] - g_RT[5] - g_RT[9] + g_RT[14])));
  // reaction 37:  CH2(S) + O <=> CO + H2
  k_f = 15000000;
  qf[48] *= k_f;
  qr[48] *= k_f * (kc_fast ? (inv_ge[1] * inv_ge[9] * ge[4] * ge[14])
                           : exp(-(-g_RT[1] + g_RT[4] - g_RT[9] + g_RT[14])));
  // reaction 38:  CH2(S) + O <=> H + HCO
  k_f = 15000000;
  qf[49] *= k_f;
  qr[49] *= k_f * (kc_fast ? (inv_ge[21] * inv_ge[2] * ge[4] * ge[14])
                           : exp(-(-g_RT[2] + g_RT[4] + g_RT[14] - g_RT[21])));
  // reaction 39:  CH2(S) + H2 <=> CH3 + H
  k_f = 70000000;
  qf[50] *= k_f;
  qr[50] *= k_f * (kc_fast ? (ge[1] * inv_ge[12] * ge[14] * inv_ge[2])
                           : exp(-(g_RT[1] - g_RT[2] - g_RT[12] + g_RT[14])));
  // reaction 40:  CH2(S) + H <=> CH + H2
  k_f = 30000000;
  qf[51] *= k_f;
  qr[51] *= k_f * (kc_fast ? (inv_ge[1] * ge[2] * inv_ge[16] * ge[14])
                           : exp(-(-g_RT[1] + g_RT[2] + g_RT[14] - g_RT[16])));
  // reaction 41:  CH2(S) + OH <=> CH2O + H
  k_f = 30000000;
  qf[52] *= k_f;
  qr[52] *= k_f * (kc_fast ? (ge[6] * inv_ge[20] * inv_ge[2] * ge[14])
                           : exp(-(-g_RT[2] + g_RT[6] + g_RT[14] - g_RT[20])));
  // reaction 42:  CH2(S) + CO2 <=> CH2O + CO
  k_f = 14000000;
  qf[53] *= k_f;
  qr[53] *= k_f * (kc_fast ? (inv_ge[9] * ge[10] * ge[14] * inv_ge[20])
                           : exp(-(-g_RT[9] + g_RT[10] + g_RT[14] - g_RT[20])));
  // reaction 43:  CH2 + H (+M) <=> CH3 (+M)
  k_f = 25000000000 * exp((-0.8) * logT);
  Corr = mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
  redP = Corr / k_f * 3.2e+15 * exp(-3.14 * logT - (618.956498939805) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (0.32) * exp(-T * 0.0128205128205128) +
    0.68 * exp(-T * 0.00050125313283208) + exp(-5590 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[3] *= Corr * k_f;
  qr[3] *= Corr * k_f *
           (kc_fast ? (inv_ge[12] * ge[2] * ge[13])
                    : exp(-(g_RT[2] - g_RT[12] + g_RT[13]))) *
           (refC);
  // reaction 44:  CH2 + O2 <=> HCO + OH
  k_f = 10600000 * exp(-(754.82499870708) * invT);
  qf[54] *= k_f;
  qr[54] *= k_f * (kc_fast ? (inv_ge[6] * ge[3] * inv_ge[21] * ge[13])
                           : exp(-(g_RT[3] - g_RT[6] + g_RT[13] - g_RT[21])));
  // reaction 45:  CH2 + O2 => CO2 + 2 H
  k_f = 2640000 * exp(-(754.82499870708) * invT);
  qf[55] *= k_f;
  qr[55] *=
    k_f *
    (kc_fast ? (ge[3] * inv_ge[2] * ge[13] * inv_ge[2] * inv_ge[10])
             : exp(-(-2.000000 * g_RT[2] + g_RT[3] - g_RT[10] + g_RT[13]))) *
    (refCinv);
  // reaction 46:  CH2 + O => CO + 2 H
  k_f = 50000000;
  qf[56] *= k_f;
  qr[56] *=
    k_f *
    (kc_fast ? (inv_ge[9] * inv_ge[2] * ge[4] * inv_ge[2] * ge[13])
             : exp(-(-2.000000 * g_RT[2] + g_RT[4] - g_RT[9] + g_RT[13]))) *
    (refCinv);
  // reaction 47:  CH2 + H <=> CH + H2
  k_f = 283000000 * exp(-(369.864249366469) * invT);
  qf[57] *= k_f;
  qr[57] *= k_f * (kc_fast ? (inv_ge[1] * ge[2] * inv_ge[16] * ge[13])
                           : exp(-(-g_RT[1] + g_RT[2] + g_RT[13] - g_RT[16])));
  // reaction 48:  CH2 + OH <=> CH + H2O
  k_f = 11.3 * exp((2) * logT - (1509.64999741416) * invT);
  qf[58] *= k_f;
  qr[58] *= k_f * (kc_fast ? (ge[6] * inv_ge[5] * inv_ge[16] * ge[13])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[13] - g_RT[16])));
  // reaction 49:  CH + O2 <=> HCO + O
  k_f = 33000000;
  qf[59] *= k_f;
  qr[59] *= k_f * (kc_fast ? (ge[3] * inv_ge[21] * inv_ge[4] * ge[16])
                           : exp(-(g_RT[3] - g_RT[4] + g_RT[16] - g_RT[21])));
  // reaction 50:  CH + O <=> CO + H
  k_f = 57000000;
  qf[60] *= k_f;
  qr[60] *= k_f * (kc_fast ? (inv_ge[9] * inv_ge[2] * ge[4] * ge[16])
                           : exp(-(-g_RT[2] + g_RT[4] - g_RT[9] + g_RT[16])));
  // reaction 51:  CH + H <=> C + H2
  k_f = 110000000;
  qf[61] *= k_f;
  qr[61] *= k_f * (kc_fast ? (inv_ge[1] * ge[2] * inv_ge[15] * ge[16])
                           : exp(-(-g_RT[1] + g_RT[2] - g_RT[15] + g_RT[16])));
  // reaction 52:  CH + OH <=> H + HCO
  k_f = 30000000;
  qf[62] *= k_f;
  qr[62] *= k_f * (kc_fast ? (ge[6] * inv_ge[21] * inv_ge[2] * ge[16])
                           : exp(-(-g_RT[2] + g_RT[6] + g_RT[16] - g_RT[21])));
  // reaction 53:  CH + H2O <=> CH2O + H
  k_f = 17130000 * exp(-(-379.928582682563) * invT);
  qf[63] *= k_f;
  qr[63] *= k_f * (kc_fast ? (inv_ge[20] * inv_ge[2] * ge[5] * ge[16])
                           : exp(-(-g_RT[2] + g_RT[5] + g_RT[16] - g_RT[20])));
  // reaction 54:  CH + CO2 <=> CO + HCO
  k_f = 1700000 * exp(-(344.703416076233) * invT);
  qf[64] *= k_f;
  qr[64] *= k_f * (kc_fast ? (inv_ge[21] * inv_ge[9] * ge[10] * ge[16])
                           : exp(-(-g_RT[9] + g_RT[10] + g_RT[16] - g_RT[21])));
  // reaction 55:  C + OH <=> CO + H
  k_f = 50000000;
  qf[65] *= k_f;
  qr[65] *= k_f * (kc_fast ? (ge[6] * inv_ge[9] * inv_ge[2] * ge[15])
                           : exp(-(-g_RT[2] + g_RT[6] - g_RT[9] + g_RT[15])));
  // reaction 56:  C + O2 <=> CO + O
  k_f = 50000000;
  qf[66] *= k_f;
  qr[66] *= k_f * (kc_fast ? (ge[3] * inv_ge[9] * inv_ge[4] * ge[15])
                           : exp(-(g_RT[3] - g_RT[4] - g_RT[9] + g_RT[15])));
  // reaction 57:  CH3 + O2 <=> CH3O + O
  k_f = 7546000 * exp(-(14251.0959755897) * invT);
  qf[67] *= k_f;
  qr[67] *= k_f * (kc_fast ? (ge[3] * inv_ge[18] * ge[12] * inv_ge[4])
                           : exp(-(g_RT[3] - g_RT[4] + g_RT[12] - g_RT[18])));
  // reaction 58:  CH3 + O2 <=> CH2O + OH
  k_f = 2.641e-06 * exp((3.283) * logT - (4078.57107634725) * invT);
  qf[68] *= k_f;
  qr[68] *= k_f * (kc_fast ? (inv_ge[6] * ge[3] * ge[12] * inv_ge[20])
                           : exp(-(g_RT[3] - g_RT[6] + g_RT[12] - g_RT[20])));
  // reaction 59:  CH3 + O <=> CH2O + H
  k_f = 55400000 * exp((0.05) * logT - (-68.4374665494419) * invT);
  qf[69] *= k_f;
  qr[69] *= k_f * (kc_fast ? (ge[12] * inv_ge[2] * inv_ge[20] * ge[4])
                           : exp(-(-g_RT[2] + g_RT[4] + g_RT[12] - g_RT[20])));
  // reaction 60:  CH3 + OH <=> CH2(S) + H2O
  k_f = 4.788e+17 * exp((-3.155) * logT - (3524.02631063045) * invT);
  qf[70] *= k_f;
  qr[70] *= k_f * (kc_fast ? (ge[6] * ge[12] * inv_ge[14] * inv_ge[5])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[12] - g_RT[14])));
  // reaction 61:  CH3 + OH <=> CH2O + H2
  k_f = 5374 * exp((0.287) * logT - (140.900666425322) * invT);
  qf[71] *= k_f;
  qr[71] *= k_f * (kc_fast ? (inv_ge[1] * ge[6] * ge[12] * inv_ge[20])
                           : exp(-(-g_RT[1] + g_RT[6] + g_RT[12] - g_RT[20])));
  // reaction 62:  CH3 + OH <=> CH2OH + H
  k_f = 15250000 * exp((0.134) * logT - (2838.64521180442) * invT);
  qf[72] *= k_f;
  qr[72] *= k_f * (kc_fast ? (ge[6] * inv_ge[19] * inv_ge[2] * ge[12])
                           : exp(-(-g_RT[2] + g_RT[6] + g_RT[12] - g_RT[19])));
  // reaction 63:  CH3 + OH <=> CH3O + H
  k_f = 1798 * exp((0.965) * logT - (6068.79298960492) * invT);
  qf[73] *= k_f;
  qr[73] *= k_f * (kc_fast ? (ge[6] * inv_ge[18] * ge[12] * inv_ge[2])
                           : exp(-(-g_RT[2] + g_RT[6] + g_RT[12] - g_RT[18])));
  // reaction 64:  CH3 + OH <=> CH2 + H2O
  k_f = 0.04293 * exp((2.568) * logT - (2011.75958655411) * invT);
  qf[74] *= k_f;
  qr[74] *= k_f * (kc_fast ? (ge[6] * ge[12] * inv_ge[13] * inv_ge[5])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[12] - g_RT[13])));
  // reaction 65:  CH3 + HO2 <=> CH3O + OH
  k_f = 1000000 * exp((0.269) * logT - (-345.961457740745) * invT);
  qf[75] *= k_f;
  qr[75] *= k_f * (kc_fast ? (inv_ge[6] * ge[8] * ge[12] * inv_ge[18])
                           : exp(-(-g_RT[6] + g_RT[8] + g_RT[12] - g_RT[18])));
  // reaction 66:  CH3OH (+M) <=> CH3 + OH (+M)
  k_f = 2.084e+18 * exp((-0.615) * logT - (46567.9721835682) * invT);
  Corr = mixture;
  redP = Corr / k_f * 1.5e+37 * exp(-6.995 * logT - (49311.3081588693) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (1.4748) * exp(-T * 2.81056773468241e-05) +
    -0.4748 * exp(-T * 0.000896057347670251) + exp(-9023 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[4] *= Corr * k_f;
  qr[4] *= Corr * k_f *
           (kc_fast ? (inv_ge[6] * inv_ge[12] * ge[17])
                    : exp(-(-g_RT[6] - g_RT[12] + g_RT[17]))) *
           (refCinv);
  // reaction 67:  CH3OH (+M) <=> CH2(S) + H2O (+M)
  k_f = 3.121e+18 * exp((-1.017) * logT - (46151.0068542825) * invT);
  Corr = mixture;
  redP = Corr / k_f * 1.43e+41 * exp(-8.227 * logT - (50028.3415859744) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (-1.545) * exp(-T * 0.000303951367781155) +
    2.545 * exp(-T * 2.11327134404057e-05) + exp(-47110 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[5] *= Corr * k_f;
  qr[5] *= Corr * k_f *
           (kc_fast ? (ge[17] * inv_ge[5] * inv_ge[14])
                    : exp(-(-g_RT[5] - g_RT[14] + g_RT[17]))) *
           (refCinv);
  // reaction 68:  CH3OH + H <=> CH3O + H2
  k_f = 0.199 * exp((2.56) * logT - (5183.13165778861) * invT);
  qf[76] *= k_f;
  qr[76] *= k_f * (kc_fast ? (inv_ge[1] * inv_ge[18] * ge[17] * ge[2])
                           : exp(-(-g_RT[1] + g_RT[2] + g_RT[17] - g_RT[18])));
  // reaction 69:  CH3OH + H <=> CH2OH + H2
  k_f = 0.307 * exp((2.55) * logT - (2737.49866197768) * invT);
  qf[77] *= k_f;
  qr[77] *= k_f * (kc_fast ? (inv_ge[1] * inv_ge[19] * ge[17] * ge[2])
                           : exp(-(-g_RT[1] + g_RT[2] + g_RT[17] - g_RT[19])));
  // reaction 70:  CH3OH + O <=> CH3O + OH
  k_f = 0.0388 * exp((2.5) * logT - (1549.90733067854) * invT);
  qf[78] *= k_f;
  qr[78] *= k_f * (kc_fast ? (inv_ge[6] * inv_ge[18] * ge[17] * ge[4])
                           : exp(-(g_RT[4] - g_RT[6] + g_RT[17] - g_RT[18])));
  // reaction 71:  CH3OH + O <=> CH2OH + OH
  k_f = 0.388 * exp((2.5) * logT - (1549.90733067854) * invT);
  qf[79] *= k_f;
  qr[79] *= k_f * (kc_fast ? (inv_ge[6] * inv_ge[19] * ge[17] * ge[4])
                           : exp(-(g_RT[4] - g_RT[6] + g_RT[17] - g_RT[19])));
  // reaction 72:  CH3OH + OH <=> CH3O + H2O
  k_f = 0.00015 * exp((3.03) * logT - (-383.954316009001) * invT);
  qf[80] *= k_f;
  qr[80] *= k_f * (kc_fast ? (ge[6] * inv_ge[18] * ge[17] * inv_ge[5])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[17] - g_RT[18])));
  // reaction 73:  CH3OH + OH <=> CH2OH + H2O
  k_f = 0.0308 * exp((2.65) * logT - (-405.944884304667) * invT);
  qf[81] *= k_f;
  qr[81] *= k_f * (kc_fast ? (ge[6] * inv_ge[19] * ge[17] * inv_ge[5])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[17] - g_RT[19])));
  // reaction 74:  CH3OH + O2 <=> CH3O + HO2
  k_f = 0.0358 * exp((2.27) * logT - (21519.8091048059) * invT);
  qf[82] *= k_f;
  qr[82] *= k_f * (kc_fast ? (ge[3] * inv_ge[8] * inv_ge[18] * ge[17])
                           : exp(-(g_RT[3] - g_RT[8] + g_RT[17] - g_RT[18])));
  // reaction 75:  CH3OH + HO2 <=> CH3O + H2O2
  k_f = 1220000 * exp(-(10099.9107343668) * invT);
  qf[83] *= k_f;
  qr[83] *= k_f * (kc_fast ? (ge[8] * inv_ge[18] * inv_ge[7] * ge[17])
                           : exp(-(-g_RT[7] + g_RT[8] + g_RT[17] - g_RT[18])));
  // reaction 76:  CH3OH + HO2 <=> CH2OH + H2O2
  k_f = 32600000 * exp(-(9451.5160604774) * invT);
  qf[84] *= k_f;
  qr[84] *= k_f * (kc_fast ? (ge[8] * inv_ge[19] * inv_ge[7] * ge[17])
                           : exp(-(-g_RT[7] + g_RT[8] + g_RT[17] - g_RT[19])));
  // reaction 77:  CH3 + CH3OH <=> CH2OH + CH4
  k_f = 2.13e-07 * exp((3.953) * logT - (3550.24389891888) * invT);
  qf[85] *= k_f;
  qr[85] *=
    k_f * (kc_fast ? (inv_ge[19] * ge[12] * ge[17] * inv_ge[11])
                   : exp(-(-g_RT[11] + g_RT[12] + g_RT[17] - g_RT[19])));
  // reaction 78:  CH3 + CH3OH <=> CH3O + CH4
  k_f = 0.00322 * exp((2.425) * logT - (4317.34738427159) * invT);
  qf[86] *= k_f;
  qr[86] *=
    k_f * (kc_fast ? (inv_ge[18] * ge[12] * ge[17] * inv_ge[11])
                   : exp(-(-g_RT[11] + g_RT[12] + g_RT[17] - g_RT[18])));
  // reaction 79:  CH2OH + O2 <=> CH2O + HO2
  k_f = 1510000000 * exp((-1) * logT);
  qf[87] *= k_f;
  qr[87] *= k_f * (kc_fast ? (ge[3] * inv_ge[8] * ge[19] * inv_ge[20])
                           : exp(-(g_RT[3] - g_RT[8] + g_RT[19] - g_RT[20])));
  // reaction 80:  CH2OH + O2 <=> CH2O + HO2
  k_f = 241000000 * exp(-(2524.63801234228) * invT);
  qf[88] *= k_f;
  qr[88] *= k_f * (kc_fast ? (ge[3] * inv_ge[8] * ge[19] * inv_ge[20])
                           : exp(-(g_RT[3] - g_RT[8] + g_RT[19] - g_RT[20])));
  // reaction 81:  CH2OH + H <=> CH2O + H2
  k_f = 6000000;
  qf[89] *= k_f;
  qr[89] *= k_f * (kc_fast ? (inv_ge[1] * ge[19] * inv_ge[20] * ge[2])
                           : exp(-(-g_RT[1] + g_RT[2] + g_RT[19] - g_RT[20])));
  // reaction 82:  CH2OH + HO2 <=> CH2O + H2O2
  k_f = 12000000;
  qf[90] *= k_f;
  qr[90] *= k_f * (kc_fast ? (ge[8] * inv_ge[20] * ge[19] * inv_ge[7])
                           : exp(-(-g_RT[7] + g_RT[8] + g_RT[19] - g_RT[20])));
  // reaction 83:  CH2OH + HCO <=> 2 CH2O
  k_f = 180000000;
  qf[91] *= k_f;
  qr[91] *= k_f * (kc_fast ? (ge[21] * inv_ge[20] * ge[19] * inv_ge[20])
                           : exp(-(g_RT[19] - 2.000000 * g_RT[20] + g_RT[21])));
  // reaction 84:  CH2OH + OH <=> CH2O + H2O
  k_f = 24000000;
  qf[92] *= k_f;
  qr[92] *= k_f * (kc_fast ? (ge[6] * ge[19] * inv_ge[20] * inv_ge[5])
                           : exp(-(-g_RT[5] + g_RT[6] + g_RT[19] - g_RT[20])));
  // reaction 85:  CH2OH + O <=> CH2O + OH
  k_f = 42000000;
  qf[93] *= k_f;
  qr[93] *= k_f * (kc_fast ? (inv_ge[6] * ge[19] * inv_ge[20] * ge[4])
                           : exp(-(g_RT[4] - g_RT[6] + g_RT[19] - g_RT[20])));
  // reaction 86:  CH3O + O2 <=> CH2O + HO2
  k_f = 4.38e-25 * exp((9.5) * logT - (-2768.19487859176) * invT);
  qf[94] *= k_f;
  qr[94] *= k_f * (kc_fast ? (ge[3] * inv_ge[8] * ge[18] * inv_ge[20])
                           : exp(-(g_RT[3] - g_RT[8] + g_RT[18] - g_RT[20])));
  // reaction 87:  CH3O + H <=> CH2O + H2
  k_f = 20000000;
  qf[95] *= k_f;
  qr[95] *= k_f * (kc_fast ? (inv_ge[1] * ge[18] * inv_ge[20] * ge[2])
                           : exp(-(-g_RT[1] + g_RT[2] + g_RT[18] - g_RT[20])));
  // reaction 88:  CH3O + HO2 <=> CH2O + H2O2
  k_f = 301000;
  qf[96] *= k_f;
  qr[96] *= k_f * (kc_fast ? (ge[8] * inv_ge[20] * ge[18] * inv_ge[7])
                           : exp(-(-g_RT[7] + g_RT[8] + g_RT[18] - g_RT[20])));
  // reaction 89:  CH3 + CH3O <=> CH2O + CH4
  k_f = 12000000;
  qf[97] *= k_f;
  qr[97] *=
    k_f * (kc_fast ? (ge[18] * inv_ge[11] * ge[12] * inv_ge[20])
                   : exp(-(-g_RT[11] + g_RT[12] + g_RT[18] - g_RT[20])));
  // reaction 90:  2 CH3O <=> CH2O + CH3OH
  k_f = 60300000;
  qf[98] *= k_f;
  qr[98] *=
    k_f * (kc_fast ? (ge[18] * inv_ge[20] * ge[18] * inv_ge[17])
                   : exp(-(-g_RT[17] + 2.000000 * g_RT[18] - g_RT[20])));
  // reaction 91:  CH2O + O <=> HCO + OH
  k_f = 6260 * exp((1.15) * logT - (1137.26966471867) * invT);
  qf[99] *= k_f;
  qr[99] *= k_f * (kc_fast ? (inv_ge[6] * inv_ge[21] * ge[20] * ge[4])
                           : exp(-(g_RT[4] - g_RT[6] + g_RT[20] - g_RT[21])));
  // reaction 92:  CH2O + H <=> H2 + HCO
  k_f = 57.4 * exp((1.9) * logT - (1378.81366430493) * invT);
  qf[100] *= k_f;
  qr[100] *= k_f * (kc_fast ? (inv_ge[1] * inv_ge[21] * ge[20] * ge[2])
                            : exp(-(-g_RT[1] + g_RT[2] + g_RT[20] - g_RT[21])));
  // reaction 93:  CH2O + OH <=> H2O + HCO
  k_f = 78.2 * exp((1.63) * logT - (-530.893582423979) * invT);
  qf[101] *= k_f;
  qr[101] *= k_f * (kc_fast ? (ge[6] * inv_ge[21] * ge[20] * inv_ge[5])
                            : exp(-(-g_RT[5] + g_RT[6] + g_RT[20] - g_RT[21])));
  // reaction 94:  CH2O + HO2 <=> H2O2 + HCO
  k_f = 0.0188 * exp((2.7) * logT - (5797.05599007037) * invT);
  qf[102] *= k_f;
  qr[102] *= k_f * (kc_fast ? (inv_ge[21] * ge[8] * inv_ge[7] * ge[20])
                            : exp(-(-g_RT[7] + g_RT[8] + g_RT[20] - g_RT[21])));
  // reaction 95:  CH2O + CH3 <=> CH4 + HCO
  k_f = 3.83e-05 * exp((3.36) * logT - (2169.87026294995) * invT);
  qf[103] *= k_f;
  qr[103] *=
    k_f * (kc_fast ? (inv_ge[21] * ge[12] * ge[20] * inv_ge[11])
                   : exp(-(-g_RT[11] + g_RT[12] + g_RT[20] - g_RT[21])));
  // reaction 96:  CH2O + CH3O <=> CH3OH + HCO
  k_f = 662000 * exp(-(1154.37903135603) * invT);
  qf[104] *= k_f;
  qr[104] *=
    k_f * (kc_fast ? (inv_ge[21] * ge[18] * ge[20] * inv_ge[17])
                   : exp(-(-g_RT[17] + g_RT[18] + g_RT[20] - g_RT[21])));
  // reaction 97:  HCO + M <=> CO + H + M
  k_f = 570000 * exp((0.66) * logT - (7482.83182051618) * invT);
  Corr = mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
  qf[11] *= Corr * k_f;
  qr[11] *= Corr * k_f *
            (kc_fast ? (ge[21] * inv_ge[9] * inv_ge[2])
                     : exp(-(-g_RT[2] - g_RT[9] + g_RT[21]))) *
            (refCinv);
  // reaction 98:  HCO + O2 <=> CO + HO2
  k_f = 7580000 * exp(-(206.318832979935) * invT);
  qf[105] *= k_f;
  qr[105] *= k_f * (kc_fast ? (ge[3] * inv_ge[8] * ge[21] * inv_ge[9])
                            : exp(-(g_RT[3] - g_RT[8] - g_RT[9] + g_RT[21])));
  // reaction 99:  HCO + O <=> CO + OH
  k_f = 30200000;
  qf[106] *= k_f;
  qr[106] *= k_f * (kc_fast ? (inv_ge[6] * ge[21] * inv_ge[9] * ge[4])
                            : exp(-(g_RT[4] - g_RT[6] - g_RT[9] + g_RT[21])));
  // reaction 100:  H + HCO <=> CO + H2
  k_f = 73400000;
  qf[107] *= k_f;
  qr[107] *= k_f * (kc_fast ? (inv_ge[1] * ge[21] * inv_ge[9] * ge[2])
                            : exp(-(-g_RT[1] + g_RT[2] - g_RT[9] + g_RT[21])));
  // reaction 101:  HCO + OH <=> CO + H2O
  k_f = 30110000;
  qf[108] *= k_f;
  qr[108] *= k_f * (kc_fast ? (ge[6] * ge[21] * inv_ge[9] * inv_ge[5])
                            : exp(-(-g_RT[5] + g_RT[6] - g_RT[9] + g_RT[21])));
  // reaction 102:  CH3 + HCO <=> CH4 + CO
  k_f = 26500000;
  qf[109] *= k_f;
  qr[109] *=
    k_f * (kc_fast ? (ge[21] * inv_ge[11] * ge[12] * inv_ge[9])
                   : exp(-(-g_RT[9] - g_RT[11] + g_RT[12] + g_RT[21])));
  // reaction 103:  2 HCO <=> CH2O + CO
  k_f = 18000000;
  qf[110] *= k_f;
  qr[110] *=
    k_f * (kc_fast ? (ge[21] * ge[21] * inv_ge[9] * inv_ge[20])
                   : exp(-(-g_RT[9] - g_RT[20] + 2.000000 * g_RT[21])));
  // reaction 104:  HCO + O <=> CO2 + H
  k_f = 30000000;
  qf[111] *= k_f;
  qr[111] *= k_f * (kc_fast ? (ge[21] * inv_ge[2] * ge[4] * inv_ge[10])
                            : exp(-(-g_RT[2] + g_RT[4] - g_RT[10] + g_RT[21])));
  // reaction 105:  HCO + HO2 => CO2 + H + OH
  k_f = 30000000;
  qf[112] *= k_f;
  qr[112] *=
    k_f *
    (kc_fast ? (inv_ge[6] * ge[21] * ge[8] * inv_ge[2] * inv_ge[10])
             : exp(-(-g_RT[2] - g_RT[6] + g_RT[8] - g_RT[10] + g_RT[21]))) *
    (refCinv);
  // reaction 106:  CH2O + H (+M) <=> CH2OH (+M)
  k_f = 540000 * exp((0.454) * logT - (1811.57999689699) * invT);
  Corr = mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
  redP = Corr / k_f * 1.27e+20 * exp(-4.82 * logT - (3286.00482770482) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (0.2813) * exp(-T * 0.00970873786407767) +
    0.7187 * exp(-T * 0.000774593338497289) + exp(-4160 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[6] *= Corr * k_f;
  qr[6] *= Corr * k_f *
           (kc_fast ? (inv_ge[19] * ge[20] * ge[2])
                    : exp(-(g_RT[2] - g_RT[19] + g_RT[20]))) *
           (refC);
  // reaction 107:  CH3O (+M) <=> CH2O + H (+M)
  k_f = 68000000000000 * exp(-(13169.1801441095) * invT);
  Corr = mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
  redP = Corr / k_f * 1.87e+19 * exp(-3 * logT - (12231.6874957153) * invT);
  F = redP / (1.0 + redP);
  logPred = log10(redP);
  logFcent = log10(
    (0.1) * exp(-T * 0.0004) + 0.9 * exp(-T * 0.000769230769230769) +
    exp(-1e+99 * invT));
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
  Corr = F * F_troe;
  qf[7] *= Corr * k_f;
  qr[7] *= Corr * k_f *
           (kc_fast ? (ge[18] * inv_ge[20] * inv_ge[2])
                    : exp(-(-g_RT[2] + g_RT[18] - g_RT[20]))) *
           (refCinv);
  // reaction 108:  O + OH + M <=> HO2 + M
  k_f = 10000;
  Corr = mixture;
  qf[12] *= Corr * k_f;
  qr[12] *= Corr * k_f *
            (kc_fast ? (ge[6] * inv_ge[8] * ge[4])
                     : exp(-(g_RT[4] + g_RT[6] - g_RT[8]))) *
            (refC);
  // reaction 109:  CH + OH <=> C + H2O
  k_f = 40 * exp((2) * logT - (1509.64999741416) * invT);
  qf[113] *= k_f;
  qr[113] *= k_f * (kc_fast ? (ge[6] * inv_ge[5] * inv_ge[15] * ge[16])
                            : exp(-(-g_RT[5] + g_RT[6] - g_RT[15] + g_RT[16])));
  // reaction 110:  CH2 + HCO <=> CH3 + CO
  k_f = 20000000;
  qf[114] *= k_f;
  qr[114] *=
    k_f * (kc_fast ? (ge[21] * inv_ge[12] * inv_ge[9] * ge[13])
                   : exp(-(-g_RT[9] - g_RT[12] + g_RT[13] + g_RT[21])));
  // reaction 111:  CH2 + O <=> CO + H2
  k_f = 50000000;
  qf[115] *= k_f;
  qr[115] *= k_f * (kc_fast ? (inv_ge[1] * inv_ge[9] * ge[4] * ge[13])
                            : exp(-(-g_RT[1] + g_RT[4] - g_RT[9] + g_RT[13])));
  // reaction 112:  CH2 + OH <=> CH2O + H
  k_f = 30000000;
  qf[116] *= k_f;
  qr[116] *= k_f * (kc_fast ? (ge[6] * inv_ge[20] * inv_ge[2] * ge[13])
                            : exp(-(-g_RT[2] + g_RT[6] + g_RT[13] - g_RT[20])));
  // reaction 113:  CH2 + CO2 <=> CH2O + CO
  k_f = 110000 * exp(-(503.21666580472) * invT);
  qf[117] *= k_f;
  qr[117] *=
    k_f * (kc_fast ? (inv_ge[9] * ge[10] * ge[13] * inv_ge[20])
                   : exp(-(-g_RT[9] + g_RT[10] + g_RT[13] - g_RT[20])));
  // reaction 114:  CH2(S) + O <=> CO + 2 H
  k_f = 30000000;
  qf[118] *= k_f;
  qr[118] *=
    k_f *
    (kc_fast ? (inv_ge[9] * inv_ge[2] * ge[4] * inv_ge[2] * ge[14])
             : exp(-(-2.000000 * g_RT[2] + g_RT[4] - g_RT[9] + g_RT[14]))) *
    (refCinv);
  // reaction 115:  CH2(S) + CH4 <=> 2 CH3
  k_f = 43000000;
  qf[119] *= k_f;
  qr[119] *=
    k_f * (kc_fast ? (inv_ge[12] * inv_ge[12] * ge[14] * ge[11])
                   : exp(-(g_RT[11] - 2.000000 * g_RT[12] + g_RT[14])));
  // reaction 116:  CH3OH + H => CH3 + H2O
  k_f = 650000 * exp(-(2667.04832876501) * invT);
  qf[120] *= k_f;
  qr[120] *= k_f * (kc_fast ? (inv_ge[12] * ge[2] * ge[17] * inv_ge[5])
                            : exp(-(g_RT[2] - g_RT[5] - g_RT[12] + g_RT[17])));
  // reaction 117:  CH2O + O => CO2 + 2 H
  k_f = 20000000 * exp(-(2516.0833290236) * invT);
  qf[121] *= k_f;
  qr[121] *=
    k_f *
    (kc_fast ? (ge[20] * ge[4] * inv_ge[2] * inv_ge[10] * inv_ge[2])
             : exp(-(-2.000000 * g_RT[2] + g_RT[4] - g_RT[10] + g_RT[20]))) *
    (refCinv);
  // reaction 118:  CH2O + OH => CO2 + H + H2
  k_f = 100000;
  qf[122] *= k_f;
  qr[122] *=
    k_f *
    (kc_fast ? (inv_ge[1] * ge[6] * ge[20] * inv_ge[10] * inv_ge[2])
             : exp(-(-g_RT[1] - g_RT[2] + g_RT[6] - g_RT[10] + g_RT[20]))) *
    (refCinv);
  // reaction 119:  CH3O + CO <=> CH3 + CO2
  k_f = 500000 * exp(-(3270.90832773068) * invT);
  qf[123] *= k_f;
  qr[123] *= k_f * (kc_fast ? (ge[18] * inv_ge[12] * ge[9] * inv_ge[10])
                            : exp(-(g_RT[9] - g_RT[10] - g_RT[12] + g_RT[18])));
  // reaction 120:  CH3 + O + M => CH3O + M
  k_f = 50000;
  Corr = mixture;
  qf[13] *= Corr * k_f;
  qr[13] *= Corr * k_f *
            (kc_fast ? (inv_ge[18] * ge[12] * ge[4])
                     : exp(-(g_RT[4] + g_RT[12] - g_RT[18]))) *
            (refC);
  // reaction 121:  HCO + HO2 <=> CO + H2O2
  k_f = 400000;
  qf[124] *= k_f;
  qr[124] *= k_f * (kc_fast ? (ge[21] * ge[8] * inv_ge[9] * inv_ge[7])
                            : exp(-(-g_RT[7] + g_RT[8] - g_RT[9] + g_RT[21])));
  // reaction 122:  CH3O + OH => CH2O + H2O
  k_f = 15000000;
  qf[125] *= k_f;
  qr[125] *= k_f * (kc_fast ? (ge[6] * ge[18] * inv_ge[20] * inv_ge[5])
                            : exp(-(-g_RT[5] + g_RT[6] + g_RT[18] - g_RT[20])));
  // reaction 123:  CH2OH + CH3 => CH2O + CH4
  k_f = 15000000;
  qf[126] *= k_f;
  qr[126] *=
    k_f * (kc_fast ? (ge[19] * inv_ge[11] * ge[12] * inv_ge[20])
                   : exp(-(-g_RT[11] + g_RT[12] + g_RT[19] - g_RT[20])));
  // reaction 124:  CH + O <=> CHO+ + E
  k_f = 1746000000000 * exp((-2.19) * logT - (164.999209334044) * invT);
  qf[127] *= k_f;
  qr[127] *= k_f * (kc_fast ? (inv_ge[25] * ge[4] * inv_ge[22] * ge[16])
                            : exp(-(g_RT[4] + g_RT[16] - g_RT[22] - g_RT[25])));
  // reaction 125:  CHO+ + H2O <=> CO + H3O+
  k_f = 26080000000 * exp((-0.5) * logT);
  qf[128] *= k_f;
  qr[128] *= k_f * (kc_fast ? (inv_ge[9] * ge[5] * ge[22] * inv_ge[23])
                            : exp(-(g_RT[5] - g_RT[9] + g_RT[22] - g_RT[23])));
  // reaction 126:  E + H3O+ <=> H + H2O
  k_f = 739500000000 * exp((-0.5) * logT);
  qf[129] *= k_f;
  qr[129] *= k_f * (kc_fast ? (ge[25] * inv_ge[2] * inv_ge[5] * ge[23])
                            : exp(-(-g_RT[2] - g_RT[5] + g_RT[23] + g_RT[25])));
  // reaction 127:  E + H3O+ <=> 2 H + OH
  k_f = 3181000000000 * exp((-0.5) * logT);
  qf[130] *= k_f;
  qr[130] *=
    k_f *
    (kc_fast ? (ge[25] * inv_ge[6] * inv_ge[2] * ge[23] * inv_ge[2])
             : exp(-(-2.000000 * g_RT[2] - g_RT[6] + g_RT[23] + g_RT[25]))) *
    (refCinv);
  // reaction 128:  E + H3O+ <=> H2 + OH
  k_f = 560100000000 * exp((-0.5) * logT);
  qf[131] *= k_f;
  qr[131] *= k_f * (kc_fast ? (ge[25] * inv_ge[1] * inv_ge[6] * ge[23])
                            : exp(-(-g_RT[1] - g_RT[6] + g_RT[23] + g_RT[25])));
  // reaction 129:  H2 + O2- <=> E + H2O2
  k_f = 602000000;
  qf[132] *= k_f;
  qr[132] *= k_f * (kc_fast ? (inv_ge[25] * ge[1] * ge[24] * inv_ge[7])
                            : exp(-(g_RT[1] - g_RT[7] + g_RT[24] - g_RT[25])));
  // reaction 130:  H + O2- <=> E + HO2
  k_f = 723000000;
  qf[133] *= k_f;
  qr[133] *= k_f * (kc_fast ? (inv_ge[25] * inv_ge[8] * ge[24] * ge[2])
                            : exp(-(g_RT[2] - g_RT[8] + g_RT[24] - g_RT[25])));
  // reaction 131:  E + O2 + O2 <=> O2- + O2
  k_f = 1520000000 * exp((-1) * logT - (600.156324305341) * invT);
  qf[14] *= k_f;
  qr[14] *= k_f *
            (kc_fast ? (ge[25] * ge[3] * inv_ge[24])
                     : exp(-(g_RT[3] - g_RT[24] + g_RT[25]))) *
            (refC);
  // reaction 132:  E + O2 + H2O <=> O2- + H2O
  k_f = 5080000;
  qf[15] *= k_f;
  qr[15] *= k_f *
            (kc_fast ? (ge[25] * ge[3] * inv_ge[24])
                     : exp(-(g_RT[3] - g_RT[24] + g_RT[25]))) *
            (refC);
  // reaction 133:  E + O2 + N2 <=> O2- + N2
  k_f = 3590000000 * exp((-2) * logT - (69.7558942138502) * invT);
  qf[16] *= k_f;
  qr[16] *= k_f *
            (kc_fast ? (ge[25] * ge[3] * inv_ge[24])
                     : exp(-(g_RT[3] - g_RT[24] + g_RT[25]))) *
            (refC);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
productionRate(amrex::Real* wdot, const amrex::Real* sc, const amrex::Real T)
{
  const amrex::Real invT = 1.0 / T;
  const amrex::Real logT = log(T);

  // reference concentration: P_atm / (RT) in inverse mol/m^3
  const amrex::Real refC = 101325 / 8.31446 * invT;
  const amrex::Real refCinv = 1 / refC;

  for (int i = 0; i < 26; ++i) {
    wdot[i] = 0.0;
  }

  // compute the mixture concentration
  amrex::Real mixture = 0.0;
  for (int i = 0; i < 26; ++i) {
    mixture += sc[i];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[26];
  gibbs(g_RT, T);
  // Kc from shared exp(-g/RT) factors where no factor or partial product can
  // overflow (checked by CEPTR on this T range); exp of the sum otherwise and
  // in single precision
  const bool kc_fast =
    sizeof(amrex::Real) == sizeof(double) && T >= 200.0 && T <= 10000.0;
  amrex::Real ge[26] = {}, inv_ge[26] = {};
  if (kc_fast) {
    for (int i = 0; i < 26; ++i) {
      ge[i] = exp(-g_RT[i]);
      inv_ge[i] = 1.0 / ge[i];
    }
  }

  {
    // reaction 6:  H2O2 (+M) <=> 2 OH (+M)
    const amrex::Real k_f =
      2000000000000 * exp((0.9) * logT - (24531.3092413143) * invT);
    amrex::Real Corr = mixture + (0.5) * sc[0] + (2.7) * sc[1] + (0.2) * sc[3] +
                       (6.65) * sc[5] + (6.7) * sc[7] + (1.8) * sc[9] +
                       (0.6) * sc[10];
    const amrex::Real redP =
      Corr / k_f * 2.49e+18 * exp(-2.3 * logT - 24531.3092413143 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.57 * exp(-T * 1e+30) + 0.43 * exp(-T * 1e-30) + exp(-1e-16 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[7]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (inv_ge[6] * inv_ge[6] * ge[7])
                                    : exp(-(-2.000000 * g_RT[6] + g_RT[7]))) *
                           (refCinv) * ((sc[6] * sc[6]));
    const amrex::Real qdot = qf - qr;
    wdot[6] += 2.000000 * qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 19:  H + O2 (+M) <=> HO2 (+M)
    const amrex::Real k_f = 4650000 * exp((0.44) * logT);
    amrex::Real Corr = mixture + (0.3) * sc[1] + (9) * sc[5] + (0.9) * sc[9] +
                       (2.8) * sc[10] + sc[11];
    const amrex::Real redP = Corr / k_f * 17400000 * exp(-1.23 * logT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.33 * exp(-T * 1e+30) + 0.67 * exp(-T * 1e-30) + exp(-1e+30 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[2] * sc[3]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (ge[3] * inv_ge[8] * ge[2])
                                    : exp(-(g_RT[2] + g_RT[3] - g_RT[8]))) *
                           (refC) * (sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] -= qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 24:  CH3 + H (+M) <=> CH4 (+M)
    const amrex::Real k_f =
      12700000000 * exp((-0.63) * logT - (192.731983003208) * invT);
    amrex::Real Corr =
      mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
    const amrex::Real redP =
      Corr / k_f * 2.48e+21 * exp(-4.76 * logT - 1227.84866456352 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.217 * exp(-T * 0.0135135135135135) +
      0.783 * exp(-T * 0.000340020401224073) + exp(-6964 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[2] * sc[12]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (ge[12] * inv_ge[11] * ge[2])
                                    : exp(-(g_RT[2] - g_RT[11] + g_RT[12]))) *
                           (refC) * (sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;
  }

  {
    // reaction 43:  CH2 + H (+M) <=> CH3 (+M)
    const amrex::Real k_f = 25000000000 * exp((-0.8) * logT);
    amrex::Real Corr =
      mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
    const amrex::Real redP =
      Corr / k_f * 3.2e+15 * exp(-3.14 * logT - 618.956498939805 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.32 * exp(-T * 0.0128205128205128) +
      0.68 * exp(-T * 0.00050125313283208) + exp(-5590 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[2] * sc[13]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (inv_ge[12] * ge[2] * ge[13])
                                    : exp(-(g_RT[2] - g_RT[12] + g_RT[13]))) *
                           (refC) * (sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;
  }

  {
    // reaction 66:  CH3OH (+M) <=> CH3 + OH (+M)
    const amrex::Real k_f =
      2.084e+18 * exp((-0.615) * logT - (46567.9721835682) * invT);
    amrex::Real Corr = mixture;
    const amrex::Real redP =
      Corr / k_f * 1.5e+37 * exp(-6.995 * logT - 49311.3081588693 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      1.4748 * exp(-T * 2.81056773468241e-05) +
      -0.4748 * exp(-T * 0.000896057347670251) + exp(-9023 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[17]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (inv_ge[6] * inv_ge[12] * ge[17])
                                    : exp(-(-g_RT[6] - g_RT[12] + g_RT[17]))) *
                           (refCinv) * (sc[6] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[6] += qdot;
    wdot[12] += qdot;
    wdot[17] -= qdot;
  }

  {
    // reaction 67:  CH3OH (+M) <=> CH2(S) + H2O (+M)
    const amrex::Real k_f =
      3.121e+18 * exp((-1.017) * logT - (46151.0068542825) * invT);
    amrex::Real Corr = mixture;
    const amrex::Real redP =
      Corr / k_f * 1.43e+41 * exp(-8.227 * logT - 50028.3415859744 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      -1.545 * exp(-T * 0.000303951367781155) +
      2.545 * exp(-T * 2.11327134404057e-05) + exp(-47110 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[17]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (ge[17] * inv_ge[5] * inv_ge[14])
                                    : exp(-(-g_RT[5] - g_RT[14] + g_RT[17]))) *
                           (refCinv) * (sc[5] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[14] += qdot;
    wdot[17] -= qdot;
  }

  {
    // reaction 106:  CH2O + H (+M) <=> CH2OH (+M)
    const amrex::Real k_f =
      540000 * exp((0.454) * logT - (1811.57999689699) * invT);
    amrex::Real Corr =
      mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
    const amrex::Real redP =
      Corr / k_f * 1.27e+20 * exp(-4.82 * logT - 3286.00482770482 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.2813 * exp(-T * 0.00970873786407767) +
      0.7187 * exp(-T * 0.000774593338497289) + exp(-4160 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[2] * sc[20]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (inv_ge[19] * ge[20] * ge[2])
                                    : exp(-(g_RT[2] - g_RT[19] + g_RT[20]))) *
                           (refC) * (sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[19] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 107:  CH3O (+M) <=> CH2O + H (+M)
    const amrex::Real k_f = 68000000000000 * exp(-(13169.1801441095) * invT);
    amrex::Real Corr =
      mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
    const amrex::Real redP =
      Corr / k_f * 1.87e+19 * exp(-3 * logT - 12231.6874957153 * invT);
    const amrex::Real F = redP / (1.0 + redP);
    const amrex::Real logPred = log10(redP);
    const amrex::Real logFcent = log10(
      0.1 * exp(-T * 0.0004) + 0.9 * exp(-T * 0.000769230769230769) +
      exp(-1e+99 * invT));
    const amrex::Real troe_c = -0.4 - 0.67 * logFcent;
    const amrex::Real troe_n = 0.75 - 1.27 * logFcent;
    const amrex::Real troe =
      (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
    const amrex::Real F_troe = exp(M_LN10 * logFcent / (1.0 + troe * troe));
    Corr = F * F_troe;
    const amrex::Real qf = Corr * k_f * (sc[18]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (ge[18] * inv_ge[20] * inv_ge[2])
                                    : exp(-(-g_RT[2] + g_RT[18] - g_RT[20]))) *
                           (refCinv) * (sc[2] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[18] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 20:  CO + O (+M) <=> CO2 (+M)
    const amrex::Real k_f = 13620 * exp(-(1199.66853127845) * invT);
    amrex::Real Corr =
      mixture + sc[1] + (11) * sc[5] + (0.75) * sc[9] + (2.6) * sc[10];
    const amrex::Real redP =
      Corr / k_f * 1170000000000 * exp(-2.79 * logT - 2108.98104638758 * invT);
    Corr = redP / (1.0 + redP);
    const amrex::Real qf = Corr * k_f * (sc[4] * sc[9]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (ge[9] * ge[4] * inv_ge[10])
                                    : exp(-(g_RT[4] + g_RT[9] - g_RT[10]))) *
                           (refC) * (sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[9] -= qdot;
    wdot[10] += qdot;
  }

  {
    // reaction 3:  H + OH + M <=> H2O + M
    const amrex::Real k_f = 35000000000 * exp((-2) * logT);
    const amrex::Real Corr =
      mixture + (-0.27) * sc[1] + (2.65) * sc[5] + sc[11];
    const amrex::Real qf = Corr * k_f * (sc[2] * sc[6]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (ge[6] * ge[2] * inv_ge[5])
                                    : exp(-(g_RT[2] - g_RT[5] + g_RT[6]))) *
                           (refC) * (sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[5] += qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 5:  H + O + M <=> OH + M
    const amrex::Real k_f = 4714000 * exp((-1) * logT);
    const amrex::Real Corr =
      mixture + (1.5) * sc[1] + (11) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
    const amrex::Real qf = Corr * k_f * (sc[2] * sc[4]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (inv_ge[6] * ge[2] * ge[4])
                                    : exp(-(g_RT[2] + g_RT[4] - g_RT[6]))) *
                           (refC) * (sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 97:  HCO + M <=> CO + H + M
    const amrex::Real k_f =
      570000 * exp((0.66) * logT - (7482.83182051618) * invT);
    const amrex::Real Corr =
      mixture + sc[1] + (5) * sc[5] + (0.5) * sc[9] + sc[10] + sc[11];
    const amrex::Real qf = Corr * k_f * (sc[21]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (ge[21] * inv_ge[9] * inv_ge[2])
                                    : exp(-(-g_RT[2] - g_RT[9] + g_RT[21]))) *
                           (refCinv) * (sc[2] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[9] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 108:  O + OH + M <=> HO2 + M
    const amrex::Real k_f = 10000;
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[4] * sc[6]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (ge[6] * inv_ge[8] * ge[4])
                                    : exp(-(g_RT[4] + g_RT[6] - g_RT[8]))) *
                           (refC) * (sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] -= qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 120:  CH3 + O + M => CH3O + M
    const amrex::Real k_f = 50000;
    const amrex::Real Corr = mixture;
    const amrex::Real qf = Corr * k_f * (sc[4] * sc[12]);
    const amrex::Real qr = Corr * k_f *
                           (kc_fast ? (inv_ge[18] * ge[12] * ge[4])
                                    : exp(-(g_RT[4] + g_RT[12] - g_RT[18]))) *
                           (refC) * (0.0);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[12] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 131:  E + O2 + O2 <=> O2- + O2
    const amrex::Real k_f =
      1520000000 * exp((-1) * logT - (600.156324305341) * invT);
    const amrex::Real qf = k_f * ((sc[3] * sc[3]) * sc[25]);
    const amrex::Real qr = k_f *
                           (kc_fast ? (ge[25] * ge[3] * inv_ge[24])
                                    : exp(-(g_RT[3] - g_RT[24] + g_RT[25]))) *
                           (refC) * (sc[3] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 132:  E + O2 + H2O <=> O2- + H2O
    const amrex::Real k_f = 5080000;
    const amrex::Real qf = k_f * (sc[3] * sc[5] * sc[25]);
    const amrex::Real qr = k_f *
                           (kc_fast ? (ge[25] * ge[3] * inv_ge[24])
                                    : exp(-(g_RT[3] - g_RT[24] + g_RT[25]))) *
                           (refC) * (sc[5] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 133:  E + O2 + N2 <=> O2- + N2
    const amrex::Real k_f =
      3590000000 * exp((-2) * logT - (69.7558942138502) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[3] * sc[25]);
    const amrex::Real qr = k_f *
                           (kc_fast ? (ge[25] * ge[3] * inv_ge[24])
                                    : exp(-(g_RT[3] - g_RT[24] + g_RT[25]))) *
                           (refC) * (sc[0] * sc[24]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[24] += qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 0:  H2 + O <=> H + OH
    const amrex::Real k_f =
      0.0508 * exp((2.67) * logT - (3166.2392612433) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[4]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[1] * inv_ge[6] * ge[4] * inv_ge[2])
               : exp(-(g_RT[1] - g_RT[2] + g_RT[4] - g_RT[6]))) *
      (sc[2] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 1:  H2 + OH <=> H + H2O
    const amrex::Real k_f = 43800000 * exp(-(3517.48449397499) * invT);
    const amrex::Real qf = k_f * (sc[1] * sc[6]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[1] * ge[6] * inv_ge[2] * inv_ge[5])
               : exp(-(g_RT[1] - g_RT[2] - g_RT[5] + g_RT[6]))) *
      (sc[2] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[5] += qdot;
    wdot[6] -= qdot;
  }

  {
    // reaction 2:  H + O2 <=> O + OH
    const amrex::Real k_f = 114000000 * exp(-(7692.16995349094) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[3]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[3] * ge[2] * inv_ge[4])
               : exp(-(g_RT[2] + g_RT[3] - g_RT[4] - g_RT[6]))) *
      (sc[4] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[6] += qdot;
  }

  {
    // reaction 4:  H2O + O <=> 2 OH
    const amrex::Real k_f =
      67 * exp((1.704) * logT - (7541.60752708217) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[5]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * inv_ge[6] * ge[4] * ge[5])
               : exp(-(g_RT[4] + g_RT[5] - 2.000000 * g_RT[6]))) *
      ((sc[6] * sc[6]));
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[5] -= qdot;
    wdot[6] += 2.000000 * qdot;
  }

  {
    // reaction 7:  H + H2O2 <=> H2O + OH
    const amrex::Real k_f = 24100000 * exp(-(1997.77016324474) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[7]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[7] * ge[2] * inv_ge[5])
               : exp(-(g_RT[2] - g_RT[5] - g_RT[6] + g_RT[7]))) *
      (sc[5] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[5] += qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
  }

  {
    // reaction 8:  H + H2O2 <=> H2 + HO2
    const amrex::Real k_f = 21500 * exp((1) * logT - (3019.29999482832) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[7]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * inv_ge[8] * ge[7] * ge[2])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[7] - g_RT[8]))) *
      (sc[1] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[7] -= qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 9:  H2O2 + O <=> HO2 + OH
    const amrex::Real k_f = 9.55 * exp((2) * logT - (1997.77016324474) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[7]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * inv_ge[8] * ge[7] * ge[4])
               : exp(-(g_RT[4] - g_RT[6] + g_RT[7] - g_RT[8]))) *
      (sc[6] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[7] -= qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 10:  H2O2 + OH <=> H2O + HO2
    const amrex::Real k_f = 1740000 * exp(-(160.022899725901) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[7]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[8] * ge[7] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[7] - g_RT[8]))) *
      (sc[5] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[7] -= qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 11:  H2O2 + OH <=> H2O + HO2
    const amrex::Real k_f = 75900000 * exp(-(3657.88194373451) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[7]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[8] * ge[7] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[7] - g_RT[8]))) *
      (sc[5] * sc[8]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[7] -= qdot;
    wdot[8] += qdot;
  }

  {
    // reaction 12:  H + HO2 <=> 2 OH
    const amrex::Real k_f = 70790000 * exp(-(148.448916412392) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[8]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[6] * ge[2])
               : exp(-(g_RT[2] - 2.000000 * g_RT[6] + g_RT[8]))) *
      ((sc[6] * sc[6]));
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[6] += 2.000000 * qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 13:  H + HO2 <=> H2 + O2
    const amrex::Real k_f =
      11402 * exp((1.083) * logT - (278.671325189338) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[8]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[8] * inv_ge[3] * ge[2])
               : exp(-(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[8]))) *
      (sc[1] * sc[3]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[3] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 14:  HO2 + O <=> O2 + OH
    const amrex::Real k_f = 32500000;
    const amrex::Real qf = k_f * (sc[4] * sc[8]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[3] * ge[4])
               : exp(-(-g_RT[3] + g_RT[4] - g_RT[6] + g_RT[8]))) *
      (sc[3] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 15:  HO2 + OH <=> H2O + O2
    const amrex::Real k_f = 7000000 * exp(-(-549.995687057926) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[8]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[3] * ge[8] * inv_ge[5])
               : exp(-(-g_RT[3] - g_RT[5] + g_RT[6] + g_RT[8]))) *
      (sc[3] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 16:  HO2 + OH <=> H2O + O2
    const amrex::Real k_f = 450000000 * exp(-(5499.95687057926) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[8]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[3] * ge[8] * inv_ge[5])
               : exp(-(-g_RT[3] - g_RT[5] + g_RT[6] + g_RT[8]))) *
      (sc[3] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[8] -= qdot;
  }

  {
    // reaction 17:  2 HO2 <=> H2O2 + O2
    const amrex::Real k_f = 100000000 * exp(-(5555.95482115001) * invT);
    const amrex::Real qf = k_f * ((sc[8] * sc[8]));
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[3] * ge[8] * ge[8] * inv_ge[7])
               : exp(-(-g_RT[3] - g_RT[7] + 2.000000 * g_RT[8]))) *
      (sc[3] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[7] += qdot;
    wdot[8] -= 2.000000 * qdot;
  }

  {
    // reaction 18:  2 HO2 <=> H2O2 + O2
    const amrex::Real k_f = 190000 * exp(-(-708.992024785586) * invT);
    const amrex::Real qf = k_f * ((sc[8] * sc[8]));
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[3] * ge[8] * ge[8] * inv_ge[7])
               : exp(-(-g_RT[3] - g_RT[7] + 2.000000 * g_RT[8]))) *
      (sc[3] * sc[7]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[7] += qdot;
    wdot[8] -= 2.000000 * qdot;
  }

  {
    // reaction 21:  CO + OH <=> CO2 + H
    const amrex::Real k_f =
      0.07015 * exp((2.053) * logT - (-178.994168026739) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[9]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * ge[9] * inv_ge[10] * inv_ge[2])
               : exp(-(-g_RT[2] + g_RT[6] + g_RT[9] - g_RT[10]))) *
      (sc[2] * sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[9] -= qdot;
    wdot[10] += qdot;
  }

  {
    // reaction 22:  CO + OH <=> CO2 + H
    const amrex::Real k_f =
      5757000 * exp((-0.664) * logT - (166.967289714006) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[9]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * ge[9] * inv_ge[10] * inv_ge[2])
               : exp(-(-g_RT[2] + g_RT[6] + g_RT[9] - g_RT[10]))) *
      (sc[2] * sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[9] -= qdot;
    wdot[10] += qdot;
  }

  {
    // reaction 23:  CO + HO2 <=> CO2 + OH
    const amrex::Real k_f =
      0.157 * exp((2.18) * logT - (9027.70698453667) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[9]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[8] * ge[9] * inv_ge[10])
               : exp(-(-g_RT[6] + g_RT[8] + g_RT[9] - g_RT[10]))) *
      (sc[6] * sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[6] += qdot;
    wdot[8] -= qdot;
    wdot[9] -= qdot;
    wdot[10] += qdot;
  }

  {
    // reaction 25:  CH4 + H <=> CH3 + H2
    const amrex::Real k_f =
      0.614 * exp((2.5) * logT - (4824.33817506985) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[11]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[11] * ge[2] * inv_ge[12])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[11] - g_RT[12]))) *
      (sc[1] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
  }

  {
    // reaction 26:  CH4 + O <=> CH3 + OH
    const amrex::Real k_f =
      1020 * exp((1.5) * logT - (4327.66332592059) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[11]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[11] * ge[4] * inv_ge[12])
               : exp(-(g_RT[4] - g_RT[6] + g_RT[11] - g_RT[12]))) *
      (sc[6] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
  }

  {
    // reaction 27:  CH4 + OH <=> CH3 + H2O
    const amrex::Real k_f =
      0.0583 * exp((2.6) * logT - (1102.04449811234) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[11]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[12] * inv_ge[5] * ge[11])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[11] - g_RT[12]))) *
      (sc[5] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
  }

  {
    // reaction 28:  CH4 + HO2 <=> CH3 + H2O2
    const amrex::Real k_f =
      1.13e-05 * exp((3.74) * logT - (10572.5821485572) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[11]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[8] * inv_ge[12] * inv_ge[7] * ge[11])
               : exp(-(-g_RT[7] + g_RT[8] + g_RT[11] - g_RT[12]))) *
      (sc[7] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[7] += qdot;
    wdot[8] -= qdot;
    wdot[11] -= qdot;
    wdot[12] += qdot;
  }

  {
    // reaction 29:  CH3 + HO2 <=> CH4 + O2
    const amrex::Real k_f =
      0.116 * exp((2.23) * logT - (-1520.72076406186) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[3] * ge[8] * ge[12] * inv_ge[11])
               : exp(-(-g_RT[3] + g_RT[8] - g_RT[11] + g_RT[12]))) *
      (sc[3] * sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[3] += qdot;
    wdot[8] -= qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;
  }

  {
    // reaction 30:  CH2 + CH4 <=> 2 CH3
    const amrex::Real k_f = 2.46 * exp((2) * logT - (4161.60182620503) * invT);
    const amrex::Real qf = k_f * (sc[11] * sc[13]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[12] * inv_ge[12] * ge[13] * ge[11])
               : exp(-(g_RT[11] - 2.000000 * g_RT[12] + g_RT[13]))) *
      ((sc[12] * sc[12]));
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += 2.000000 * qdot;
    wdot[13] -= qdot;
  }

  {
    // reaction 31:  CH2(S) + N2 <=> CH2 + N2
    const amrex::Real k_f = 15000000 * exp(-(301.929999482832) * invT);
    const amrex::Real qf = k_f * (sc[0] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[13] * ge[14])
               : exp(-(g_RT[0] - g_RT[0] - g_RT[13] + g_RT[14]))) *
      (sc[0] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[0] -= qdot;
    wdot[0] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 32:  CH2(S) + H2O <=> CH2 + H2O
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[5] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[13] * ge[14])
               : exp(-(g_RT[5] - g_RT[5] - g_RT[13] + g_RT[14]))) *
      (sc[5] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[5] -= qdot;
    wdot[5] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 33:  CH2(S) + CO <=> CH2 + CO
    const amrex::Real k_f = 9000000;
    const amrex::Real qf = k_f * (sc[9] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[13] * ge[14])
               : exp(-(g_RT[9] - g_RT[9] - g_RT[13] + g_RT[14]))) *
      (sc[9] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[9] -= qdot;
    wdot[9] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 34:  CH2(S) + CO2 <=> CH2 + CO2
    const amrex::Real k_f = 7000000;
    const amrex::Real qf = k_f * (sc[10] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[13] * ge[14])
               : exp(-(g_RT[10] - g_RT[10] - g_RT[13] + g_RT[14]))) *
      (sc[10] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[10] -= qdot;
    wdot[10] += qdot;
    wdot[13] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 35:  CH2(S) + O2 => CO + H + OH
    const amrex::Real k_f = 28000000;
    const amrex::Real qf = k_f * (sc[3] * sc[14]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[9] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 36:  CH2(S) + O2 <=> CO + H2O
    const amrex::Real k_f = 12000000;
    const amrex::Real qf = k_f * (sc[3] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[9] * inv_ge[5] * ge[14])
               : exp(-(g_RT[3] - g_RT[5] - g_RT[9] + g_RT[14]))) *
      (sc[5] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[5] += qdot;
    wdot[9] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 37:  CH2(S) + O <=> CO + H2
    const amrex::Real k_f = 15000000;
    const amrex::Real qf = k_f * (sc[4] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * inv_ge[9] * ge[4] * ge[14])
               : exp(-(-g_RT[1] + g_RT[4] - g_RT[9] + g_RT[14]))) *
      (sc[1] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 38:  CH2(S) + O <=> H + HCO
    const amrex::Real k_f = 15000000;
    const amrex::Real qf = k_f * (sc[4] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[21] * inv_ge[2] * ge[4] * ge[14])
               : exp(-(-g_RT[2] + g_RT[4] + g_RT[14] - g_RT[21]))) *
      (sc[2] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[14] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 39:  CH2(S) + H2 <=> CH3 + H
    const amrex::Real k_f = 70000000;
    const amrex::Real qf = k_f * (sc[1] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[1] * inv_ge[12] * ge[14] * inv_ge[2])
               : exp(-(g_RT[1] - g_RT[2] - g_RT[12] + g_RT[14]))) *
      (sc[2] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[2] += qdot;
    wdot[12] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 40:  CH2(S) + H <=> CH + H2
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[2] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[2] * inv_ge[16] * ge[14])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[14] - g_RT[16]))) *
      (sc[1] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[14] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 41:  CH2(S) + OH <=> CH2O + H
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[6] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[20] * inv_ge[2] * ge[14])
               : exp(-(-g_RT[2] + g_RT[6] + g_RT[14] - g_RT[20]))) *
      (sc[2] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[14] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 42:  CH2(S) + CO2 <=> CH2O + CO
    const amrex::Real k_f = 14000000;
    const amrex::Real qf = k_f * (sc[10] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[9] * ge[10] * ge[14] * inv_ge[20])
               : exp(-(-g_RT[9] + g_RT[10] + g_RT[14] - g_RT[20]))) *
      (sc[9] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[14] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 44:  CH2 + O2 <=> HCO + OH
    const amrex::Real k_f = 10600000 * exp(-(754.82499870708) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[13]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[3] * inv_ge[21] * ge[13])
               : exp(-(g_RT[3] - g_RT[6] + g_RT[13] - g_RT[21]))) *
      (sc[6] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[13] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 45:  CH2 + O2 => CO2 + 2 H
    const amrex::Real k_f = 2640000 * exp(-(754.82499870708) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[13]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] += 2.000000 * qdot;
    wdot[3] -= qdot;
    wdot[10] += qdot;
    wdot[13] -= qdot;
  }

  {
    // reaction 46:  CH2 + O => CO + 2 H
    const amrex::Real k_f = 50000000;
    const amrex::Real qf = k_f * (sc[4] * sc[13]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] += 2.000000 * qdot;
    wdot[4] -= qdot;
    wdot[9] += qdot;
    wdot[13] -= qdot;
  }

  {
    // reaction 47:  CH2 + H <=> CH + H2
    const amrex::Real k_f = 283000000 * exp(-(369.864249366469) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[13]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[2] * inv_ge[16] * ge[13])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[13] - g_RT[16]))) *
      (sc[1] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 48:  CH2 + OH <=> CH + H2O
    const amrex::Real k_f = 11.3 * exp((2) * logT - (1509.64999741416) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[13]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[5] * inv_ge[16] * ge[13])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[13] - g_RT[16]))) *
      (sc[5] * sc[16]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[13] -= qdot;
    wdot[16] += qdot;
  }

  {
    // reaction 49:  CH + O2 <=> HCO + O
    const amrex::Real k_f = 33000000;
    const amrex::Real qf = k_f * (sc[3] * sc[16]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[21] * inv_ge[4] * ge[16])
               : exp(-(g_RT[3] - g_RT[4] + g_RT[16] - g_RT[21]))) *
      (sc[4] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[16] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 50:  CH + O <=> CO + H
    const amrex::Real k_f = 57000000;
    const amrex::Real qf = k_f * (sc[4] * sc[16]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[9] * inv_ge[2] * ge[4] * ge[16])
               : exp(-(-g_RT[2] + g_RT[4] - g_RT[9] + g_RT[16]))) *
      (sc[2] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[9] += qdot;
    wdot[16] -= qdot;
  }

  {
    // reaction 51:  CH + H <=> C + H2
    const amrex::Real k_f = 110000000;
    const amrex::Real qf = k_f * (sc[2] * sc[16]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[2] * inv_ge[15] * ge[16])
               : exp(-(-g_RT[1] + g_RT[2] - g_RT[15] + g_RT[16]))) *
      (sc[1] * sc[15]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[15] += qdot;
    wdot[16] -= qdot;
  }

  {
    // reaction 52:  CH + OH <=> H + HCO
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[6] * sc[16]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[21] * inv_ge[2] * ge[16])
               : exp(-(-g_RT[2] + g_RT[6] + g_RT[16] - g_RT[21]))) *
      (sc[2] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[16] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 53:  CH + H2O <=> CH2O + H
    const amrex::Real k_f = 17130000 * exp(-(-379.928582682563) * invT);
    const amrex::Real qf = k_f * (sc[5] * sc[16]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[20] * inv_ge[2] * ge[5] * ge[16])
               : exp(-(-g_RT[2] + g_RT[5] + g_RT[16] - g_RT[20]))) *
      (sc[2] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[5] -= qdot;
    wdot[16] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 54:  CH + CO2 <=> CO + HCO
    const amrex::Real k_f = 1700000 * exp(-(344.703416076233) * invT);
    const amrex::Real qf = k_f * (sc[10] * sc[16]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[21] * inv_ge[9] * ge[10] * ge[16])
               : exp(-(-g_RT[9] + g_RT[10] + g_RT[16] - g_RT[21]))) *
      (sc[9] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[16] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 55:  C + OH <=> CO + H
    const amrex::Real k_f = 50000000;
    const amrex::Real qf = k_f * (sc[6] * sc[15]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[9] * inv_ge[2] * ge[15])
               : exp(-(-g_RT[2] + g_RT[6] - g_RT[9] + g_RT[15]))) *
      (sc[2] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[9] += qdot;
    wdot[15] -= qdot;
  }

  {
    // reaction 56:  C + O2 <=> CO + O
    const amrex::Real k_f = 50000000;
    const amrex::Real qf = k_f * (sc[3] * sc[15]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[9] * inv_ge[4] * ge[15])
               : exp(-(g_RT[3] - g_RT[4] - g_RT[9] + g_RT[15]))) *
      (sc[4] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[9] += qdot;
    wdot[15] -= qdot;
  }

  {
    // reaction 57:  CH3 + O2 <=> CH3O + O
    const amrex::Real k_f = 7546000 * exp(-(14251.0959755897) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[18] * ge[12] * inv_ge[4])
               : exp(-(g_RT[3] - g_RT[4] + g_RT[12] - g_RT[18]))) *
      (sc[4] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[4] += qdot;
    wdot[12] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 58:  CH3 + O2 <=> CH2O + OH
    const amrex::Real k_f =
      2.641e-06 * exp((3.283) * logT - (4078.57107634725) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[3] * ge[12] * inv_ge[20])
               : exp(-(g_RT[3] - g_RT[6] + g_RT[12] - g_RT[20]))) *
      (sc[6] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[6] += qdot;
    wdot[12] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 59:  CH3 + O <=> CH2O + H
    const amrex::Real k_f =
      55400000 * exp((0.05) * logT - (-68.4374665494419) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[12] * inv_ge[2] * inv_ge[20] * ge[4])
               : exp(-(-g_RT[2] + g_RT[4] + g_RT[12] - g_RT[20]))) *
      (sc[2] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[12] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 60:  CH3 + OH <=> CH2(S) + H2O
    const amrex::Real k_f =
      4.788e+17 * exp((-3.155) * logT - (3524.02631063045) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * ge[12] * inv_ge[14] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[12] - g_RT[14]))) *
      (sc[5] * sc[14]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[12] -= qdot;
    wdot[14] += qdot;
  }

  {
    // reaction 61:  CH3 + OH <=> CH2O + H2
    const amrex::Real k_f =
      5374 * exp((0.287) * logT - (140.900666425322) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[6] * ge[12] * inv_ge[20])
               : exp(-(-g_RT[1] + g_RT[6] + g_RT[12] - g_RT[20]))) *
      (sc[1] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[6] -= qdot;
    wdot[12] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 62:  CH3 + OH <=> CH2OH + H
    const amrex::Real k_f =
      15250000 * exp((0.134) * logT - (2838.64521180442) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[19] * inv_ge[2] * ge[12])
               : exp(-(-g_RT[2] + g_RT[6] + g_RT[12] - g_RT[19]))) *
      (sc[2] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[12] -= qdot;
    wdot[19] += qdot;
  }

  {
    // reaction 63:  CH3 + OH <=> CH3O + H
    const amrex::Real k_f =
      1798 * exp((0.965) * logT - (6068.79298960492) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[18] * ge[12] * inv_ge[2])
               : exp(-(-g_RT[2] + g_RT[6] + g_RT[12] - g_RT[18]))) *
      (sc[2] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[12] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 64:  CH3 + OH <=> CH2 + H2O
    const amrex::Real k_f =
      0.04293 * exp((2.568) * logT - (2011.75958655411) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * ge[12] * inv_ge[13] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[12] - g_RT[13]))) *
      (sc[5] * sc[13]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[12] -= qdot;
    wdot[13] += qdot;
  }

  {
    // reaction 65:  CH3 + HO2 <=> CH3O + OH
    const amrex::Real k_f =
      1000000 * exp((0.269) * logT - (-345.961457740745) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[12]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[8] * ge[12] * inv_ge[18])
               : exp(-(-g_RT[6] + g_RT[8] + g_RT[12] - g_RT[18]))) *
      (sc[6] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[6] += qdot;
    wdot[8] -= qdot;
    wdot[12] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 68:  CH3OH + H <=> CH3O + H2
    const amrex::Real k_f =
      0.199 * exp((2.56) * logT - (5183.13165778861) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * inv_ge[18] * ge[17] * ge[2])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[17] - g_RT[18]))) *
      (sc[1] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[17] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 69:  CH3OH + H <=> CH2OH + H2
    const amrex::Real k_f =
      0.307 * exp((2.55) * logT - (2737.49866197768) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * inv_ge[19] * ge[17] * ge[2])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[17] - g_RT[19]))) *
      (sc[1] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;
  }

  {
    // reaction 70:  CH3OH + O <=> CH3O + OH
    const amrex::Real k_f =
      0.0388 * exp((2.5) * logT - (1549.90733067854) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * inv_ge[18] * ge[17] * ge[4])
               : exp(-(g_RT[4] - g_RT[6] + g_RT[17] - g_RT[18]))) *
      (sc[6] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[17] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 71:  CH3OH + O <=> CH2OH + OH
    const amrex::Real k_f =
      0.388 * exp((2.5) * logT - (1549.90733067854) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * inv_ge[19] * ge[17] * ge[4])
               : exp(-(g_RT[4] - g_RT[6] + g_RT[17] - g_RT[19]))) *
      (sc[6] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;
  }

  {
    // reaction 72:  CH3OH + OH <=> CH3O + H2O
    const amrex::Real k_f =
      0.00015 * exp((3.03) * logT - (-383.954316009001) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[18] * ge[17] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[17] - g_RT[18]))) *
      (sc[5] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[17] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 73:  CH3OH + OH <=> CH2OH + H2O
    const amrex::Real k_f =
      0.0308 * exp((2.65) * logT - (-405.944884304667) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[19] * ge[17] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[17] - g_RT[19]))) *
      (sc[5] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;
  }

  {
    // reaction 74:  CH3OH + O2 <=> CH3O + HO2
    const amrex::Real k_f =
      0.0358 * exp((2.27) * logT - (21519.8091048059) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[8] * inv_ge[18] * ge[17])
               : exp(-(g_RT[3] - g_RT[8] + g_RT[17] - g_RT[18]))) *
      (sc[8] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[8] += qdot;
    wdot[17] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 75:  CH3OH + HO2 <=> CH3O + H2O2
    const amrex::Real k_f = 1220000 * exp(-(10099.9107343668) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[8] * inv_ge[18] * inv_ge[7] * ge[17])
               : exp(-(-g_RT[7] + g_RT[8] + g_RT[17] - g_RT[18]))) *
      (sc[7] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[7] += qdot;
    wdot[8] -= qdot;
    wdot[17] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 76:  CH3OH + HO2 <=> CH2OH + H2O2
    const amrex::Real k_f = 32600000 * exp(-(9451.5160604774) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[8] * inv_ge[19] * inv_ge[7] * ge[17])
               : exp(-(-g_RT[7] + g_RT[8] + g_RT[17] - g_RT[19]))) *
      (sc[7] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[7] += qdot;
    wdot[8] -= qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;
  }

  {
    // reaction 77:  CH3 + CH3OH <=> CH2OH + CH4
    const amrex::Real k_f =
      2.13e-07 * exp((3.953) * logT - (3550.24389891888) * invT);
    const amrex::Real qf = k_f * (sc[12] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[19] * ge[12] * ge[17] * inv_ge[11])
               : exp(-(-g_RT[11] + g_RT[12] + g_RT[17] - g_RT[19]))) *
      (sc[11] * sc[19]);
    const amrex::Real qdot = qf - qr;
    wdot[11] += qdot;
    wdot[12] -= qdot;
    wdot[17] -= qdot;
    wdot[19] += qdot;
  }

  {
    // reaction 78:  CH3 + CH3OH <=> CH3O + CH4
    const amrex::Real k_f =
      0.00322 * exp((2.425) * logT - (4317.34738427159) * invT);
    const amrex::Real qf = k_f * (sc[12] * sc[17]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[18] * ge[12] * ge[17] * inv_ge[11])
               : exp(-(-g_RT[11] + g_RT[12] + g_RT[17] - g_RT[18]))) *
      (sc[11] * sc[18]);
    const amrex::Real qdot = qf - qr;
    wdot[11] += qdot;
    wdot[12] -= qdot;
    wdot[17] -= qdot;
    wdot[18] += qdot;
  }

  {
    // reaction 79:  CH2OH + O2 <=> CH2O + HO2
    const amrex::Real k_f = 1510000000 * exp((-1) * logT);
    const amrex::Real qf = k_f * (sc[3] * sc[19]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[8] * ge[19] * inv_ge[20])
               : exp(-(g_RT[3] - g_RT[8] + g_RT[19] - g_RT[20]))) *
      (sc[8] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[8] += qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 80:  CH2OH + O2 <=> CH2O + HO2
    const amrex::Real k_f = 241000000 * exp(-(2524.63801234228) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[19]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[8] * ge[19] * inv_ge[20])
               : exp(-(g_RT[3] - g_RT[8] + g_RT[19] - g_RT[20]))) *
      (sc[8] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[8] += qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 81:  CH2OH + H <=> CH2O + H2
    const amrex::Real k_f = 6000000;
    const amrex::Real qf = k_f * (sc[2] * sc[19]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[19] * inv_ge[20] * ge[2])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[19] - g_RT[20]))) *
      (sc[1] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 82:  CH2OH + HO2 <=> CH2O + H2O2
    const amrex::Real k_f = 12000000;
    const amrex::Real qf = k_f * (sc[8] * sc[19]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[8] * inv_ge[20] * ge[19] * inv_ge[7])
               : exp(-(-g_RT[7] + g_RT[8] + g_RT[19] - g_RT[20]))) *
      (sc[7] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[7] += qdot;
    wdot[8] -= qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 83:  CH2OH + HCO <=> 2 CH2O
    const amrex::Real k_f = 180000000;
    const amrex::Real qf = k_f * (sc[19] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[21] * inv_ge[20] * ge[19] * inv_ge[20])
               : exp(-(g_RT[19] - 2.000000 * g_RT[20] + g_RT[21]))) *
      ((sc[20] * sc[20]));
    const amrex::Real qdot = qf - qr;
    wdot[19] -= qdot;
    wdot[20] += 2.000000 * qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 84:  CH2OH + OH <=> CH2O + H2O
    const amrex::Real k_f = 24000000;
    const amrex::Real qf = k_f * (sc[6] * sc[19]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * ge[19] * inv_ge[20] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[19] - g_RT[20]))) *
      (sc[5] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 85:  CH2OH + O <=> CH2O + OH
    const amrex::Real k_f = 42000000;
    const amrex::Real qf = k_f * (sc[4] * sc[19]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[19] * inv_ge[20] * ge[4])
               : exp(-(g_RT[4] - g_RT[6] + g_RT[19] - g_RT[20]))) *
      (sc[6] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 86:  CH3O + O2 <=> CH2O + HO2
    const amrex::Real k_f =
      4.38e-25 * exp((9.5) * logT - (-2768.19487859176) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[18]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[8] * ge[18] * inv_ge[20])
               : exp(-(g_RT[3] - g_RT[8] + g_RT[18] - g_RT[20]))) *
      (sc[8] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[8] += qdot;
    wdot[18] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 87:  CH3O + H <=> CH2O + H2
    const amrex::Real k_f = 20000000;
    const amrex::Real qf = k_f * (sc[2] * sc[18]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[18] * inv_ge[20] * ge[2])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[18] - g_RT[20]))) *
      (sc[1] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[18] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 88:  CH3O + HO2 <=> CH2O + H2O2
    const amrex::Real k_f = 301000;
    const amrex::Real qf = k_f * (sc[8] * sc[18]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[8] * inv_ge[20] * ge[18] * inv_ge[7])
               : exp(-(-g_RT[7] + g_RT[8] + g_RT[18] - g_RT[20]))) *
      (sc[7] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[7] += qdot;
    wdot[8] -= qdot;
    wdot[18] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 89:  CH3 + CH3O <=> CH2O + CH4
    const amrex::Real k_f = 12000000;
    const amrex::Real qf = k_f * (sc[12] * sc[18]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[18] * inv_ge[11] * ge[12] * inv_ge[20])
               : exp(-(-g_RT[11] + g_RT[12] + g_RT[18] - g_RT[20]))) *
      (sc[11] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[11] += qdot;
    wdot[12] -= qdot;
    wdot[18] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 90:  2 CH3O <=> CH2O + CH3OH
    const amrex::Real k_f = 60300000;
    const amrex::Real qf = k_f * ((sc[18] * sc[18]));
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[18] * inv_ge[20] * ge[18] * inv_ge[17])
               : exp(-(-g_RT[17] + 2.000000 * g_RT[18] - g_RT[20]))) *
      (sc[17] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[17] += qdot;
    wdot[18] -= 2.000000 * qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 91:  CH2O + O <=> HCO + OH
    const amrex::Real k_f =
      6260 * exp((1.15) * logT - (1137.26966471867) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[20]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * inv_ge[21] * ge[20] * ge[4])
               : exp(-(g_RT[4] - g_RT[6] + g_RT[20] - g_RT[21]))) *
      (sc[6] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 92:  CH2O + H <=> H2 + HCO
    const amrex::Real k_f =
      57.4 * exp((1.9) * logT - (1378.81366430493) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[20]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * inv_ge[21] * ge[20] * ge[2])
               : exp(-(-g_RT[1] + g_RT[2] + g_RT[20] - g_RT[21]))) *
      (sc[1] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 93:  CH2O + OH <=> H2O + HCO
    const amrex::Real k_f =
      78.2 * exp((1.63) * logT - (-530.893582423979) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[20]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[21] * ge[20] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] + g_RT[20] - g_RT[21]))) *
      (sc[5] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 94:  CH2O + HO2 <=> H2O2 + HCO
    const amrex::Real k_f =
      0.0188 * exp((2.7) * logT - (5797.05599007037) * invT);
    const amrex::Real qf = k_f * (sc[8] * sc[20]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[21] * ge[8] * inv_ge[7] * ge[20])
               : exp(-(-g_RT[7] + g_RT[8] + g_RT[20] - g_RT[21]))) *
      (sc[7] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[7] += qdot;
    wdot[8] -= qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 95:  CH2O + CH3 <=> CH4 + HCO
    const amrex::Real k_f =
      3.83e-05 * exp((3.36) * logT - (2169.87026294995) * invT);
    const amrex::Real qf = k_f * (sc[12] * sc[20]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[21] * ge[12] * ge[20] * inv_ge[11])
               : exp(-(-g_RT[11] + g_RT[12] + g_RT[20] - g_RT[21]))) *
      (sc[11] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[11] += qdot;
    wdot[12] -= qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 96:  CH2O + CH3O <=> CH3OH + HCO
    const amrex::Real k_f = 662000 * exp(-(1154.37903135603) * invT);
    const amrex::Real qf = k_f * (sc[18] * sc[20]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[21] * ge[18] * ge[20] * inv_ge[17])
               : exp(-(-g_RT[17] + g_RT[18] + g_RT[20] - g_RT[21]))) *
      (sc[17] * sc[21]);
    const amrex::Real qdot = qf - qr;
    wdot[17] += qdot;
    wdot[18] -= qdot;
    wdot[20] -= qdot;
    wdot[21] += qdot;
  }

  {
    // reaction 98:  HCO + O2 <=> CO + HO2
    const amrex::Real k_f = 7580000 * exp(-(206.318832979935) * invT);
    const amrex::Real qf = k_f * (sc[3] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[3] * inv_ge[8] * ge[21] * inv_ge[9])
               : exp(-(g_RT[3] - g_RT[8] - g_RT[9] + g_RT[21]))) *
      (sc[8] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[3] -= qdot;
    wdot[8] += qdot;
    wdot[9] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 99:  HCO + O <=> CO + OH
    const amrex::Real k_f = 30200000;
    const amrex::Real qf = k_f * (sc[4] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[6] * ge[21] * inv_ge[9] * ge[4])
               : exp(-(g_RT[4] - g_RT[6] - g_RT[9] + g_RT[21]))) *
      (sc[6] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[6] += qdot;
    wdot[9] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 100:  H + HCO <=> CO + H2
    const amrex::Real k_f = 73400000;
    const amrex::Real qf = k_f * (sc[2] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * ge[21] * inv_ge[9] * ge[2])
               : exp(-(-g_RT[1] + g_RT[2] - g_RT[9] + g_RT[21]))) *
      (sc[1] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] -= qdot;
    wdot[9] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 101:  HCO + OH <=> CO + H2O
    const amrex::Real k_f = 30110000;
    const amrex::Real qf = k_f * (sc[6] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * ge[21] * inv_ge[9] * inv_ge[5])
               : exp(-(-g_RT[5] + g_RT[6] - g_RT[9] + g_RT[21]))) *
      (sc[5] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[9] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 102:  CH3 + HCO <=> CH4 + CO
    const amrex::Real k_f = 26500000;
    const amrex::Real qf = k_f * (sc[12] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[21] * inv_ge[11] * ge[12] * inv_ge[9])
               : exp(-(-g_RT[9] - g_RT[11] + g_RT[12] + g_RT[21]))) *
      (sc[9] * sc[11]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[11] += qdot;
    wdot[12] -= qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 103:  2 HCO <=> CH2O + CO
    const amrex::Real k_f = 18000000;
    const amrex::Real qf = k_f * ((sc[21] * sc[21]));
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[21] * ge[21] * inv_ge[9] * inv_ge[20])
               : exp(-(-g_RT[9] - g_RT[20] + 2.000000 * g_RT[21]))) *
      (sc[9] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[20] += qdot;
    wdot[21] -= 2.000000 * qdot;
  }

  {
    // reaction 104:  HCO + O <=> CO2 + H
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[4] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[21] * inv_ge[2] * ge[4] * inv_ge[10])
               : exp(-(-g_RT[2] + g_RT[4] - g_RT[10] + g_RT[21]))) *
      (sc[2] * sc[10]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[4] -= qdot;
    wdot[10] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 105:  HCO + HO2 => CO2 + H + OH
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[8] * sc[21]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] += qdot;
    wdot[8] -= qdot;
    wdot[10] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 109:  CH + OH <=> C + H2O
    const amrex::Real k_f = 40 * exp((2) * logT - (1509.64999741416) * invT);
    const amrex::Real qf = k_f * (sc[6] * sc[16]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[5] * inv_ge[15] * ge[16])
               : exp(-(-g_RT[5] + g_RT[6] - g_RT[15] + g_RT[16]))) *
      (sc[5] * sc[15]);
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[15] += qdot;
    wdot[16] -= qdot;
  }

  {
    // reaction 110:  CH2 + HCO <=> CH3 + CO
    const amrex::Real k_f = 20000000;
    const amrex::Real qf = k_f * (sc[13] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[21] * inv_ge[12] * inv_ge[9] * ge[13])
               : exp(-(-g_RT[9] - g_RT[12] + g_RT[13] + g_RT[21]))) *
      (sc[9] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[12] += qdot;
    wdot[13] -= qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 111:  CH2 + O <=> CO + H2
    const amrex::Real k_f = 50000000;
    const amrex::Real qf = k_f * (sc[4] * sc[13]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[1] * inv_ge[9] * ge[4] * ge[13])
               : exp(-(-g_RT[1] + g_RT[4] - g_RT[9] + g_RT[13]))) *
      (sc[1] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[4] -= qdot;
    wdot[9] += qdot;
    wdot[13] -= qdot;
  }

  {
    // reaction 112:  CH2 + OH <=> CH2O + H
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[6] * sc[13]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[6] * inv_ge[20] * inv_ge[2] * ge[13])
               : exp(-(-g_RT[2] + g_RT[6] + g_RT[13] - g_RT[20]))) *
      (sc[2] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[13] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 113:  CH2 + CO2 <=> CH2O + CO
    const amrex::Real k_f = 110000 * exp(-(503.21666580472) * invT);
    const amrex::Real qf = k_f * (sc[10] * sc[13]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[9] * ge[10] * ge[13] * inv_ge[20])
               : exp(-(-g_RT[9] + g_RT[10] + g_RT[13] - g_RT[20]))) *
      (sc[9] * sc[20]);
    const amrex::Real qdot = qf - qr;
    wdot[9] += qdot;
    wdot[10] -= qdot;
    wdot[13] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 114:  CH2(S) + O <=> CO + 2 H
    const amrex::Real k_f = 30000000;
    const amrex::Real qf = k_f * (sc[4] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[9] * inv_ge[2] * ge[4] * inv_ge[2] * ge[14])
               : exp(-(-2.000000 * g_RT[2] + g_RT[4] - g_RT[9] + g_RT[14]))) *
      (refCinv) * ((sc[2] * sc[2]) * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += 2.000000 * qdot;
    wdot[4] -= qdot;
    wdot[9] += qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 115:  CH2(S) + CH4 <=> 2 CH3
    const amrex::Real k_f = 43000000;
    const amrex::Real qf = k_f * (sc[11] * sc[14]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[12] * inv_ge[12] * ge[14] * ge[11])
               : exp(-(g_RT[11] - 2.000000 * g_RT[12] + g_RT[14]))) *
      ((sc[12] * sc[12]));
    const amrex::Real qdot = qf - qr;
    wdot[11] -= qdot;
    wdot[12] += 2.000000 * qdot;
    wdot[14] -= qdot;
  }

  {
    // reaction 116:  CH3OH + H => CH3 + H2O
    const amrex::Real k_f = 650000 * exp(-(2667.04832876501) * invT);
    const amrex::Real qf = k_f * (sc[2] * sc[17]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[5] += qdot;
    wdot[12] += qdot;
    wdot[17] -= qdot;
  }

  {
    // reaction 117:  CH2O + O => CO2 + 2 H
    const amrex::Real k_f = 20000000 * exp(-(2516.0833290236) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[20]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[2] += 2.000000 * qdot;
    wdot[4] -= qdot;
    wdot[10] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 118:  CH2O + OH => CO2 + H + H2
    const amrex::Real k_f = 100000;
    const amrex::Real qf = k_f * (sc[6] * sc[20]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[2] += qdot;
    wdot[6] -= qdot;
    wdot[10] += qdot;
    wdot[20] -= qdot;
  }

  {
    // reaction 119:  CH3O + CO <=> CH3 + CO2
    const amrex::Real k_f = 500000 * exp(-(3270.90832773068) * invT);
    const amrex::Real qf = k_f * (sc[9] * sc[18]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[18] * inv_ge[12] * ge[9] * inv_ge[10])
               : exp(-(g_RT[9] - g_RT[10] - g_RT[12] + g_RT[18]))) *
      (sc[10] * sc[12]);
    const amrex::Real qdot = qf - qr;
    wdot[9] -= qdot;
    wdot[10] += qdot;
    wdot[12] += qdot;
    wdot[18] -= qdot;
  }

  {
    // reaction 121:  HCO + HO2 <=> CO + H2O2
    const amrex::Real k_f = 400000;
    const amrex::Real qf = k_f * (sc[8] * sc[21]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[21] * ge[8] * inv_ge[9] * inv_ge[7])
               : exp(-(-g_RT[7] + g_RT[8] - g_RT[9] + g_RT[21]))) *
      (sc[7] * sc[9]);
    const amrex::Real qdot = qf - qr;
    wdot[7] += qdot;
    wdot[8] -= qdot;
    wdot[9] += qdot;
    wdot[21] -= qdot;
  }

  {
    // reaction 122:  CH3O + OH => CH2O + H2O
    const amrex::Real k_f = 15000000;
    const amrex::Real qf = k_f * (sc[6] * sc[18]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[5] += qdot;
    wdot[6] -= qdot;
    wdot[18] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 123:  CH2OH + CH3 => CH2O + CH4
    const amrex::Real k_f = 15000000;
    const amrex::Real qf = k_f * (sc[12] * sc[19]);
    const amrex::Real qr = 0.0;
    const amrex::Real qdot = qf - qr;
    wdot[11] += qdot;
    wdot[12] -= qdot;
    wdot[19] -= qdot;
    wdot[20] += qdot;
  }

  {
    // reaction 124:  CH + O <=> CHO+ + E
    const amrex::Real k_f =
      1746000000000 * exp((-2.19) * logT - (164.999209334044) * invT);
    const amrex::Real qf = k_f * (sc[4] * sc[16]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[25] * ge[4] * inv_ge[22] * ge[16])
               : exp(-(g_RT[4] + g_RT[16] - g_RT[22] - g_RT[25]))) *
      (sc[22] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[4] -= qdot;
    wdot[16] -= qdot;
    wdot[22] += qdot;
    wdot[25] += qdot;
  }

  {
    // reaction 125:  CHO+ + H2O <=> CO + H3O+
    const amrex::Real k_f = 26080000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * (sc[5] * sc[22]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[9] * ge[5] * ge[22] * inv_ge[23])
               : exp(-(g_RT[5] - g_RT[9] + g_RT[22] - g_RT[23]))) *
      (sc[9] * sc[23]);
    const amrex::Real qdot = qf - qr;
    wdot[5] -= qdot;
    wdot[9] += qdot;
    wdot[22] -= qdot;
    wdot[23] += qdot;
  }

  {
    // reaction 126:  E + H3O+ <=> H + H2O
    const amrex::Real k_f = 739500000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * (sc[23] * sc[25]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[25] * inv_ge[2] * inv_ge[5] * ge[23])
               : exp(-(-g_RT[2] - g_RT[5] + g_RT[23] + g_RT[25]))) *
      (sc[2] * sc[5]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += qdot;
    wdot[5] += qdot;
    wdot[23] -= qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 127:  E + H3O+ <=> 2 H + OH
    const amrex::Real k_f = 3181000000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * (sc[23] * sc[25]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[25] * inv_ge[6] * inv_ge[2] * ge[23] * inv_ge[2])
               : exp(-(-2.000000 * g_RT[2] - g_RT[6] + g_RT[23] + g_RT[25]))) *
      (refCinv) * ((sc[2] * sc[2]) * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[2] += 2.000000 * qdot;
    wdot[6] += qdot;
    wdot[23] -= qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 128:  E + H3O+ <=> H2 + OH
    const amrex::Real k_f = 560100000000 * exp((-0.5) * logT);
    const amrex::Real qf = k_f * (sc[23] * sc[25]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (ge[25] * inv_ge[1] * inv_ge[6] * ge[23])
               : exp(-(-g_RT[1] - g_RT[6] + g_RT[23] + g_RT[25]))) *
      (sc[1] * sc[6]);
    const amrex::Real qdot = qf - qr;
    wdot[1] += qdot;
    wdot[6] += qdot;
    wdot[23] -= qdot;
    wdot[25] -= qdot;
  }

  {
    // reaction 129:  H2 + O2- <=> E + H2O2
    const amrex::Real k_f = 602000000;
    const amrex::Real qf = k_f * (sc[1] * sc[24]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[25] * ge[1] * ge[24] * inv_ge[7])
               : exp(-(g_RT[1] - g_RT[7] + g_RT[24] - g_RT[25]))) *
      (sc[7] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[1] -= qdot;
    wdot[7] += qdot;
    wdot[24] -= qdot;
    wdot[25] += qdot;
  }

  {
    // reaction 130:  H + O2- <=> E + HO2
    const amrex::Real k_f = 723000000;
    const amrex::Real qf = k_f * (sc[2] * sc[24]);
    const amrex::Real qr =
      k_f *
      (kc_fast ? (inv_ge[25] * inv_ge[8] * ge[24] * ge[2])
               : exp(-(g_RT[2] - g_RT[8] + g_RT[24] - g_RT[25]))) *
      (sc[8] * sc[25]);
    const amrex::Real qdot = qf - qr;
    wdot[2] -= qdot;
    wdot[8] += qdot;
    wdot[24] -= qdot;
    wdot[25] += qdot;
  }
}

// compute the production rate for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKWC(const amrex::Real T, amrex::Real C[], amrex::Real wdot[])
{

  // convert to SI
  for (int id = 0; id < 26; ++id) {
    C[id] *= 1.0e6;
  }

  // convert to chemkin units
  productionRate(wdot, C, T);

  // convert to chemkin units
  for (int id = 0; id < 26; ++id) {
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
  amrex::Real c[26]; // temporary storage
  amrex::Real YOW = 0;
  amrex::Real PWORT;

  // Compute inverse of mean molecular wt first
  for (int i = 0; i < 26; i++) {
    YOW += y[i] * imw(i);
  }
  // PW/RT (see Eq. 7)
  PWORT = P / (YOW * 8.31446261815324e+07 * T);
  // multiply by 1e6 so c goes to SI
  PWORT *= 1e6;
  // Now compute conversion (and go to SI)
  for (int i = 0; i < 26; i++) {
    c[i] = PWORT * y[i] * imw(i);
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 26; ++id) {
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
  amrex::Real c[26]; // temporary storage
  amrex::Real PORT =
    1e6 * P / (8.31446261815324e+07 * T); // 1e6 * P/RT so c goes to SI units

  // Compute conversion, see Eq 10
  for (int id = 0; id < 26; ++id) {
    c[id] = x[id] * PORT;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 26; ++id) {
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
  amrex::Real c[26]; // temporary storage

  // See Eq 8 with an extra 1e6 so c goes to SI
  for (int i = 0; i < 26; i++) {
    c[i] = 1e6 * rho * y[i] * imw(i);
  }

  // call productionRate
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 26; ++id) {
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
  amrex::Real c[26];  // temporary storage
  amrex::Real XW = 0; // See Eq 4, 11 in CK Manual
  amrex::Real ROW;
  // Compute mean molecular wt first
  XW += x[0] * 2.8013999999999999e+01;  // N2
  XW += x[1] * 2.0160000000000000e+00;  // H2
  XW += x[2] * 1.0080000000000000e+00;  // H
  XW += x[3] * 3.1998000000000001e+01;  // O2
  XW += x[4] * 1.5999000000000001e+01;  // O
  XW += x[5] * 1.8015000000000001e+01;  // H2O
  XW += x[6] * 1.7007000000000001e+01;  // OH
  XW += x[7] * 3.4014000000000003e+01;  // H2O2
  XW += x[8] * 3.3006000000000000e+01;  // HO2
  XW += x[9] * 2.8009999999999998e+01;  // CO
  XW += x[10] * 4.4009000000000000e+01; // CO2
  XW += x[11] * 1.6042999999999999e+01; // CH4
  XW += x[12] * 1.5035000000000000e+01; // CH3
  XW += x[13] * 1.4026999999999999e+01; // CH2
  XW += x[14] * 1.4026999999999999e+01; // CH2(S)
  XW += x[15] * 1.2010999999999999e+01; // C
  XW += x[16] * 1.3018999999999998e+01; // CH
  XW += x[17] * 3.2042000000000002e+01; // CH3OH
  XW += x[18] * 3.1033999999999999e+01; // CH3O
  XW += x[19] * 3.1033999999999999e+01; // CH2OH
  XW += x[20] * 3.0026000000000000e+01; // CH2O
  XW += x[21] * 2.9018000000000001e+01; // HCO
  XW += x[22] * 2.9017451420091128e+01; // CHO+
  XW += x[23] * 1.9022451420091127e+01; // H3O+
  XW += x[24] * 3.1998548579908874e+01; // O2-
  XW += x[25] * 5.4857990887282834e-04; // E
  // Extra 1e6 factor to take c to SI
  ROW = 1e6 * rho / XW;

  // Compute conversion, see Eq 11
  for (int id = 0; id < 26; ++id) {
    c[id] = x[id] * ROW;
  }

  // convert to chemkin units
  productionRate(wdot, c, T);

  // convert to chemkin units
  for (int id = 0; id < 26; ++id) {
    wdot[id] *= 1.0e-6;
  }
}

//  species unit charge number
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRG(int kcharge[])
{
  kcharge[0] = 0;   // N2
  kcharge[1] = 0;   // H2
  kcharge[2] = 0;   // H
  kcharge[3] = 0;   // O2
  kcharge[4] = 0;   // O
  kcharge[5] = 0;   // H2O
  kcharge[6] = 0;   // OH
  kcharge[7] = 0;   // H2O2
  kcharge[8] = 0;   // HO2
  kcharge[9] = 0;   // CO
  kcharge[10] = 0;  // CO2
  kcharge[11] = 0;  // CH4
  kcharge[12] = 0;  // CH3
  kcharge[13] = 0;  // CH2
  kcharge[14] = 0;  // CH2(S)
  kcharge[15] = 0;  // C
  kcharge[16] = 0;  // CH
  kcharge[17] = 0;  // CH3OH
  kcharge[18] = 0;  // CH3O
  kcharge[19] = 0;  // CH2OH
  kcharge[20] = 0;  // CH2O
  kcharge[21] = 0;  // HCO
  kcharge[22] = 1;  // CHO+
  kcharge[23] = 1;  // H3O+
  kcharge[24] = -1; // O2-
  kcharge[25] = -1; // E
}

//  species charge per unit mass
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
CKCHRGMASS(amrex::Real zk[])
{

  int kchrg[26];
  CKCHRG(kchrg);

  for (int id = 0; id < 26; ++id) {
    zk[id] = 6.02214076e+23 * 1.60217663e-19 * kchrg[id] * imw(id);
  }
}

// compute an approx to the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian_precond(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int HP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 729; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[26];
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
  for (int k = 0; k < 26; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[26];
  gibbs(g_RT, T);
  // Kc from shared exp(-g/RT) factors where no factor or partial product can
  // overflow (checked by CEPTR on this T range); exp of the sum otherwise and
  // in single precision
  const bool kc_fast =
    sizeof(amrex::Real) == sizeof(double) && T >= 200.0 && T <= 10000.0;
  amrex::Real ge[26] = {}, inv_ge[26] = {};
  if (kc_fast) {
    for (int i = 0; i < 26; ++i) {
      ge[i] = exp(-g_RT[i]);
      inv_ge[i] = 1.0 / ge[i];
    }
  }

  // compute the species enthalpy
  amrex::Real h_RT[26];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[26];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 6: H2O2 (+M) <=> 2 OH (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (1.5 - 1) * sc[0] + (3.7 - 1) * sc[1] + (1.2 - 1) * sc[3] +
          (7.65 - 1) * sc[5] + (7.7 - 1) * sc[7] + (2.8 - 1) * sc[9] +
          (1.6 - 1) * sc[10];
  // forward
  phi_f = sc[7];
  k_f = 2000000000000 * exp(0.9 * logT - (24531.3092413143) * invT);
  dlnkfdT = 0.9 * invT + (24531.3092413143) * invT2;
  // pressure-fall-off
  k_0 = 2.49e+24 * exp(-2.3 * logT - (24531.3092413143) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -2.3 * invT + (24531.3092413143) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.43) * exp(-T / 1e-30);
  Fcent2 = 0.43 * exp(-T / 1e+30);
  Fcent3 = exp(-1e-16 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT = log10e / Fcent *
                (-Fcent1 / 1e-30 - Fcent2 / 1e+30 + Fcent3 * 1e-16 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = (sc[6] * sc[6]);
  Kc = refC * (kc_fast ? (ge[6] * ge[6] * inv_ge[7])
                       : exp(-2.000000 * g_RT[6] + g_RT[7]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[7]) + (2.000000 * h_RT[6]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[6] += 2 * q; // OH
  wdot[7] -= q;     // H2O2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = 1.5 * dcdc_fac;
  dqdc[1] = 3.7 * dcdc_fac;
  dqdc[2] = dcdc_fac;
  dqdc[3] = 1.2 * dcdc_fac;
  dqdc[4] = dcdc_fac;
  dqdc[5] = 7.65 * dcdc_fac;
  dqdc[6] = dcdc_fac - k_r * 2.000000 * sc[6];
  dqdc[7] = 7.7 * dcdc_fac + k_f;
  dqdc[8] = dcdc_fac;
  dqdc[9] = 2.8 * dcdc_fac;
  dqdc[10] = 1.6 * dcdc_fac;
  dqdc[11] = dcdc_fac;
  dqdc[12] = dcdc_fac;
  dqdc[13] = dcdc_fac;
  dqdc[14] = dcdc_fac;
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac;
  dqdc[18] = dcdc_fac;
  dqdc[19] = dcdc_fac;
  dqdc[20] = dcdc_fac;
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 6] += 2 * dqdc[k];
    J[27 * k + 7] -= dqdc[k];
  }
  J[708] += 2 * dqdT; // dwdot[OH]/dT
  J[709] -= dqdT;     // dwdot[H2O2]/dT

  // reaction 19: H + O2 (+M) <=> HO2 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (1.3 - 1) * sc[1] + (10 - 1) * sc[5] + (1.9 - 1) * sc[9] +
          (3.8 - 1) * sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[3];
  k_f = 4650000 * exp(0.44 * logT);
  dlnkfdT = 0.44 * invT;
  // pressure-fall-off
  k_0 = 1.74e+19 * exp(-1.23 * logT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -1.23 * invT;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.67) * exp(-T / 1e-30);
  Fcent2 = 0.67 * exp(-T / 1e+30);
  Fcent3 = exp(-1e+30 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT = log10e / Fcent *
                (-Fcent1 / 1e-30 - Fcent2 / 1e+30 + Fcent3 * 1e+30 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[8];
  Kc = refCinv * (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[2])
                          : exp(g_RT[2] + g_RT[3] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[8]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] -= q; // H
  wdot[3] -= q; // O2
  wdot[8] += q; // HO2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = 1.3 * dcdc_fac;
  dqdc[2] = dcdc_fac + k_f * sc[3];
  dqdc[3] = dcdc_fac + k_f * sc[2];
  dqdc[4] = dcdc_fac;
  dqdc[5] = 10 * dcdc_fac;
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac - k_r;
  dqdc[9] = 1.9 * dcdc_fac;
  dqdc[10] = 3.8 * dcdc_fac;
  dqdc[11] = 2 * dcdc_fac;
  dqdc[12] = dcdc_fac;
  dqdc[13] = dcdc_fac;
  dqdc[14] = dcdc_fac;
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac;
  dqdc[18] = dcdc_fac;
  dqdc[19] = dcdc_fac;
  dqdc[20] = dcdc_fac;
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 2] -= dqdc[k];
    J[27 * k + 3] -= dqdc[k];
    J[27 * k + 8] += dqdc[k];
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 24: CH3 + H (+M) <=> CH4 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[12];
  k_f = 12700000000 * exp(-0.63 * logT - (192.731983003208) * invT);
  dlnkfdT = -0.63 * invT + (192.731983003208) * invT2;
  // pressure-fall-off
  k_0 = 2.48e+33 * exp(-4.76 * logT - (1227.84866456352) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -4.76 * invT + (1227.84866456352) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.783) * exp(-T / 74);
  Fcent2 = 0.783 * exp(-T / 2941);
  Fcent3 = exp(-6964 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 74 - Fcent2 / 2941 + Fcent3 * 6964 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[11];
  Kc = refCinv * (kc_fast ? (inv_ge[12] * ge[11] * inv_ge[2])
                          : exp(g_RT[2] - g_RT[11] + g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[12]) + (h_RT[11]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] -= q;  // H
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = 2 * dcdc_fac;
  dqdc[2] = dcdc_fac + k_f * sc[12];
  dqdc[3] = dcdc_fac;
  dqdc[4] = dcdc_fac;
  dqdc[5] = 6 * dcdc_fac;
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac;
  dqdc[9] = 1.5 * dcdc_fac;
  dqdc[10] = 2 * dcdc_fac;
  dqdc[11] = 2 * dcdc_fac - k_r;
  dqdc[12] = dcdc_fac + k_f * sc[2];
  dqdc[13] = dcdc_fac;
  dqdc[14] = dcdc_fac;
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac;
  dqdc[18] = dcdc_fac;
  dqdc[19] = dcdc_fac;
  dqdc[20] = dcdc_fac;
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 2] -= dqdc[k];
    J[27 * k + 11] += dqdc[k];
    J[27 * k + 12] -= dqdc[k];
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT

  // reaction 43: CH2 + H (+M) <=> CH3 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[13];
  k_f = 25000000000 * exp(-0.8 * logT);
  dlnkfdT = -0.8 * invT;
  // pressure-fall-off
  k_0 = 3.2e+27 * exp(-3.14 * logT - (618.956498939805) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -3.14 * invT + (618.956498939805) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.68) * exp(-T / 78);
  Fcent2 = 0.68 * exp(-T / 1995);
  Fcent3 = exp(-5590 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 78 - Fcent2 / 1995 + Fcent3 * 5590 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[12];
  Kc = refCinv * (kc_fast ? (ge[12] * inv_ge[2] * inv_ge[13])
                          : exp(g_RT[2] - g_RT[12] + g_RT[13]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[13]) + (h_RT[12]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] -= q;  // H
  wdot[12] += q; // CH3
  wdot[13] -= q; // CH2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = 2 * dcdc_fac;
  dqdc[2] = dcdc_fac + k_f * sc[13];
  dqdc[3] = dcdc_fac;
  dqdc[4] = dcdc_fac;
  dqdc[5] = 6 * dcdc_fac;
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac;
  dqdc[9] = 1.5 * dcdc_fac;
  dqdc[10] = 2 * dcdc_fac;
  dqdc[11] = 2 * dcdc_fac;
  dqdc[12] = dcdc_fac - k_r;
  dqdc[13] = dcdc_fac + k_f * sc[2];
  dqdc[14] = dcdc_fac;
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac;
  dqdc[18] = dcdc_fac;
  dqdc[19] = dcdc_fac;
  dqdc[20] = dcdc_fac;
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 2] -= dqdc[k];
    J[27 * k + 12] += dqdc[k];
    J[27 * k + 13] -= dqdc[k];
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT

  // reaction 66: CH3OH (+M) <=> CH3 + OH (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[17];
  k_f = 2.084e+18 * exp(-0.615 * logT - (46567.9721835682) * invT);
  dlnkfdT = -0.615 * invT + (46567.9721835682) * invT2;
  // pressure-fall-off
  k_0 = 1.5e+43 * exp(-6.995 * logT - (49311.3081588693) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -6.995 * invT + (49311.3081588693) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. + 0.4748) * exp(-T / 35580);
  Fcent2 = -0.4748 * exp(-T / 1116);
  Fcent3 = exp(-9023 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 35580 - Fcent2 / 1116 + Fcent3 * 9023 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[6] * sc[12];
  Kc = refC * (kc_fast ? (ge[6] * ge[12] * inv_ge[17])
                       : exp(-g_RT[6] - g_RT[12] + g_RT[17]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[17]) + (h_RT[6] + h_RT[12]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[6] += q;  // OH
  wdot[12] += q; // CH3
  wdot[17] -= q; // CH3OH
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = dcdc_fac;
  dqdc[2] = dcdc_fac;
  dqdc[3] = dcdc_fac;
  dqdc[4] = dcdc_fac;
  dqdc[5] = dcdc_fac;
  dqdc[6] = dcdc_fac - k_r * sc[12];
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac;
  dqdc[9] = dcdc_fac;
  dqdc[10] = dcdc_fac;
  dqdc[11] = dcdc_fac;
  dqdc[12] = dcdc_fac - k_r * sc[6];
  dqdc[13] = dcdc_fac;
  dqdc[14] = dcdc_fac;
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac + k_f;
  dqdc[18] = dcdc_fac;
  dqdc[19] = dcdc_fac;
  dqdc[20] = dcdc_fac;
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 6] += dqdc[k];
    J[27 * k + 12] += dqdc[k];
    J[27 * k + 17] -= dqdc[k];
  }
  J[708] += dqdT; // dwdot[OH]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT

  // reaction 67: CH3OH (+M) <=> CH2(S) + H2O (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[17];
  k_f = 3.121e+18 * exp(-1.017 * logT - (46151.0068542825) * invT);
  dlnkfdT = -1.017 * invT + (46151.0068542825) * invT2;
  // pressure-fall-off
  k_0 = 1.43e+47 * exp(-8.227 * logT - (50028.3415859744) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -8.227 * invT + (50028.3415859744) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 2.545) * exp(-T / 3290);
  Fcent2 = 2.545 * exp(-T / 47320);
  Fcent3 = exp(-47110 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 3290 - Fcent2 / 47320 + Fcent3 * 47110 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[5] * sc[14];
  Kc = refC * (kc_fast ? (inv_ge[17] * ge[5] * ge[14])
                       : exp(-g_RT[5] - g_RT[14] + g_RT[17]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[17]) + (h_RT[5] + h_RT[14]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[5] += q;  // H2O
  wdot[14] += q; // CH2(S)
  wdot[17] -= q; // CH3OH
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = dcdc_fac;
  dqdc[2] = dcdc_fac;
  dqdc[3] = dcdc_fac;
  dqdc[4] = dcdc_fac;
  dqdc[5] = dcdc_fac - k_r * sc[14];
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac;
  dqdc[9] = dcdc_fac;
  dqdc[10] = dcdc_fac;
  dqdc[11] = dcdc_fac;
  dqdc[12] = dcdc_fac;
  dqdc[13] = dcdc_fac;
  dqdc[14] = dcdc_fac - k_r * sc[5];
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac + k_f;
  dqdc[18] = dcdc_fac;
  dqdc[19] = dcdc_fac;
  dqdc[20] = dcdc_fac;
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 5] += dqdc[k];
    J[27 * k + 14] += dqdc[k];
    J[27 * k + 17] -= dqdc[k];
  }
  J[707] += dqdT; // dwdot[H2O]/dT
  J[716] += dqdT; // dwdot[CH2(S)]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT

  // reaction 106: CH2O + H (+M) <=> CH2OH (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[20];
  k_f = 540000 * exp(0.454 * logT - (1811.57999689699) * invT);
  dlnkfdT = 0.454 * invT + (1811.57999689699) * invT2;
  // pressure-fall-off
  k_0 = 1.27e+32 * exp(-4.82 * logT - (3286.00482770482) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -4.82 * invT + (3286.00482770482) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.7187) * exp(-T / 103);
  Fcent2 = 0.7187 * exp(-T / 1291);
  Fcent3 = exp(-4160 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 103 - Fcent2 / 1291 + Fcent3 * 4160 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[19];
  Kc = refCinv * (kc_fast ? (ge[19] * inv_ge[20] * inv_ge[2])
                          : exp(g_RT[2] - g_RT[19] + g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[20]) + (h_RT[19]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] -= q;  // H
  wdot[19] += q; // CH2OH
  wdot[20] -= q; // CH2O
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = 2 * dcdc_fac;
  dqdc[2] = dcdc_fac + k_f * sc[20];
  dqdc[3] = dcdc_fac;
  dqdc[4] = dcdc_fac;
  dqdc[5] = 6 * dcdc_fac;
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac;
  dqdc[9] = 1.5 * dcdc_fac;
  dqdc[10] = 2 * dcdc_fac;
  dqdc[11] = 2 * dcdc_fac;
  dqdc[12] = dcdc_fac;
  dqdc[13] = dcdc_fac;
  dqdc[14] = dcdc_fac;
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac;
  dqdc[18] = dcdc_fac;
  dqdc[19] = dcdc_fac - k_r;
  dqdc[20] = dcdc_fac + k_f * sc[2];
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 2] -= dqdc[k];
    J[27 * k + 19] += dqdc[k];
    J[27 * k + 20] -= dqdc[k];
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT

  // reaction 107: CH3O (+M) <=> CH2O + H (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[18];
  k_f = 68000000000000 * exp(-(13169.1801441095) * invT);
  dlnkfdT = (13169.1801441095) * invT2;
  // pressure-fall-off
  k_0 = 1.87e+25 * exp(-3 * logT - (12231.6874957153) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -3 * invT + (12231.6874957153) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.9) * exp(-T / 2500);
  Fcent2 = 0.9 * exp(-T / 1300);
  Fcent3 = exp(-1e+99 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 2500 - Fcent2 / 1300 + Fcent3 * 1e+99 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc = refC * (kc_fast ? (inv_ge[18] * ge[20] * ge[2])
                       : exp(-g_RT[2] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[18]) + (h_RT[2] + h_RT[20]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] += q;  // H
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = 2 * dcdc_fac;
  dqdc[2] = dcdc_fac - k_r * sc[20];
  dqdc[3] = dcdc_fac;
  dqdc[4] = dcdc_fac;
  dqdc[5] = 6 * dcdc_fac;
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac;
  dqdc[9] = 1.5 * dcdc_fac;
  dqdc[10] = 2 * dcdc_fac;
  dqdc[11] = 2 * dcdc_fac;
  dqdc[12] = dcdc_fac;
  dqdc[13] = dcdc_fac;
  dqdc[14] = dcdc_fac;
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac;
  dqdc[18] = dcdc_fac + k_f;
  dqdc[19] = dcdc_fac;
  dqdc[20] = dcdc_fac - k_r * sc[2];
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 2] += dqdc[k];
    J[27 * k + 18] -= dqdc[k];
    J[27 * k + 20] += dqdc[k];
  }
  J[704] += dqdT; // dwdot[H]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 20: CO + O (+M) <=> CO2 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + sc[1] + (12 - 1) * sc[5] + (1.75 - 1) * sc[9] +
          (3.6 - 1) * sc[10];
  // forward
  phi_f = sc[4] * sc[9];
  k_f = 13620 * exp(-(1199.66853127845) * invT);
  dlnkfdT = (1199.66853127845) * invT2;
  // pressure-fall-off
  k_0 = 1.17e+24 * exp(-2.79 * logT - (2108.98104638758) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -2.79 * invT + (2108.98104638758) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Lindemann form
  F = 1.0;
  // dlogFdlogPr is 0.0 and unused
  dlogFdT = 0.0;
  // reverse
  phi_r = sc[10];
  Kc = refCinv * (kc_fast ? (inv_ge[9] * inv_ge[4] * ge[10])
                          : exp(g_RT[4] + g_RT[9] - g_RT[10]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[9]) + (h_RT[10]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[4] -= q;  // O
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = 0.0;
  dqdc[0] = dcdc_fac;
  dqdc[1] = 2 * dcdc_fac;
  dqdc[2] = dcdc_fac;
  dqdc[3] = dcdc_fac;
  dqdc[4] = dcdc_fac + k_f * sc[9];
  dqdc[5] = 12 * dcdc_fac;
  dqdc[6] = dcdc_fac;
  dqdc[7] = dcdc_fac;
  dqdc[8] = dcdc_fac;
  dqdc[9] = 1.75 * dcdc_fac + k_f * sc[4];
  dqdc[10] = 3.6 * dcdc_fac - k_r;
  dqdc[11] = dcdc_fac;
  dqdc[12] = dcdc_fac;
  dqdc[13] = dcdc_fac;
  dqdc[14] = dcdc_fac;
  dqdc[15] = dcdc_fac;
  dqdc[16] = dcdc_fac;
  dqdc[17] = dcdc_fac;
  dqdc[18] = dcdc_fac;
  dqdc[19] = dcdc_fac;
  dqdc[20] = dcdc_fac;
  dqdc[21] = dcdc_fac;
  dqdc[22] = dcdc_fac;
  dqdc[23] = dcdc_fac;
  dqdc[24] = dcdc_fac;
  dqdc[25] = dcdc_fac;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 4] -= dqdc[k];
    J[27 * k + 9] -= dqdc[k];
    J[27 * k + 10] += dqdc[k];
  }
  J[706] -= dqdT; // dwdot[O]/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT

  // reaction 3: H + OH + M <=> H2O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (0.73 - 1) * sc[1] + (3.65 - 1) * sc[5] + sc[11];
  // forward
  phi_f = sc[2] * sc[6];
  k_f = 35000000000 * exp(-2 * logT);
  dlnkfdT = -2 * invT;
  // reverse
  phi_r = sc[5];
  Kc = refCinv * (kc_fast ? (inv_ge[6] * inv_ge[2] * ge[5])
                          : exp(g_RT[2] - g_RT[5] + g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (h_RT[5]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = 0.73 * q_nocor;
  dqdc[2] = q_nocor + k_f * sc[6];
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = 3.65 * q_nocor - k_r;
  dqdc[6] = q_nocor + k_f * sc[2];
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = q_nocor;
  dqdc[10] = q_nocor;
  dqdc[11] = 2 * q_nocor;
  dqdc[12] = q_nocor;
  dqdc[13] = q_nocor;
  dqdc[14] = q_nocor;
  dqdc[15] = q_nocor;
  dqdc[16] = q_nocor;
  dqdc[17] = q_nocor;
  dqdc[18] = q_nocor;
  dqdc[19] = q_nocor;
  dqdc[20] = q_nocor;
  dqdc[21] = q_nocor;
  dqdc[22] = q_nocor;
  dqdc[23] = q_nocor;
  dqdc[24] = q_nocor;
  dqdc[25] = q_nocor;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 2] -= dqdc[k];
    J[27 * k + 5] += dqdc[k];
    J[27 * k + 6] -= dqdc[k];
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT

  // reaction 5: H + O + M <=> OH + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5] + (1.5 - 1) * sc[9] +
          sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 4714000 * exp(-1 * logT);
  dlnkfdT = -1 * invT;
  // reverse
  phi_r = sc[6];
  Kc = refCinv * (kc_fast ? (ge[6] * inv_ge[2] * inv_ge[4])
                          : exp(g_RT[2] + g_RT[4] - g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[4]) + (h_RT[6]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[4] -= q; // O
  wdot[6] += q; // OH
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = 2.5 * q_nocor;
  dqdc[2] = q_nocor + k_f * sc[4];
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor + k_f * sc[2];
  dqdc[5] = 12 * q_nocor;
  dqdc[6] = q_nocor - k_r;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = 1.5 * q_nocor;
  dqdc[10] = 2 * q_nocor;
  dqdc[11] = 2 * q_nocor;
  dqdc[12] = q_nocor;
  dqdc[13] = q_nocor;
  dqdc[14] = q_nocor;
  dqdc[15] = q_nocor;
  dqdc[16] = q_nocor;
  dqdc[17] = q_nocor;
  dqdc[18] = q_nocor;
  dqdc[19] = q_nocor;
  dqdc[20] = q_nocor;
  dqdc[21] = q_nocor;
  dqdc[22] = q_nocor;
  dqdc[23] = q_nocor;
  dqdc[24] = q_nocor;
  dqdc[25] = q_nocor;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 2] -= dqdc[k];
    J[27 * k + 4] -= dqdc[k];
    J[27 * k + 6] += dqdc[k];
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT

  // reaction 97: HCO + M <=> CO + H + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[21];
  k_f = 570000 * exp(0.66 * logT - (7482.83182051618) * invT);
  dlnkfdT = 0.66 * invT + (7482.83182051618) * invT2;
  // reverse
  phi_r = sc[2] * sc[9];
  Kc = refC * (kc_fast ? (inv_ge[21] * ge[9] * ge[2])
                       : exp(-g_RT[2] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[21]) + (h_RT[2] + h_RT[9]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = 2 * q_nocor;
  dqdc[2] = q_nocor - k_r * sc[9];
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor;
  dqdc[5] = 6 * q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = 1.5 * q_nocor - k_r * sc[2];
  dqdc[10] = 2 * q_nocor;
  dqdc[11] = 2 * q_nocor;
  dqdc[12] = q_nocor;
  dqdc[13] = q_nocor;
  dqdc[14] = q_nocor;
  dqdc[15] = q_nocor;
  dqdc[16] = q_nocor;
  dqdc[17] = q_nocor;
  dqdc[18] = q_nocor;
  dqdc[19] = q_nocor;
  dqdc[20] = q_nocor;
  dqdc[21] = q_nocor + k_f;
  dqdc[22] = q_nocor;
  dqdc[23] = q_nocor;
  dqdc[24] = q_nocor;
  dqdc[25] = q_nocor;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 2] += dqdc[k];
    J[27 * k + 9] += dqdc[k];
    J[27 * k + 21] -= dqdc[k];
  }
  J[704] += dqdT; // dwdot[H]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 108: O + OH + M <=> HO2 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[4] * sc[6];
  k_f = 10000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[8];
  Kc = refCinv * (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[4])
                          : exp(g_RT[4] + g_RT[6] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[6]) + (h_RT[8]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q; // O
  wdot[6] -= q; // OH
  wdot[8] += q; // HO2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor + k_f * sc[6];
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor + k_f * sc[4];
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor - k_r;
  dqdc[9] = q_nocor;
  dqdc[10] = q_nocor;
  dqdc[11] = q_nocor;
  dqdc[12] = q_nocor;
  dqdc[13] = q_nocor;
  dqdc[14] = q_nocor;
  dqdc[15] = q_nocor;
  dqdc[16] = q_nocor;
  dqdc[17] = q_nocor;
  dqdc[18] = q_nocor;
  dqdc[19] = q_nocor;
  dqdc[20] = q_nocor;
  dqdc[21] = q_nocor;
  dqdc[22] = q_nocor;
  dqdc[23] = q_nocor;
  dqdc[24] = q_nocor;
  dqdc[25] = q_nocor;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 4] -= dqdc[k];
    J[27 * k + 6] -= dqdc[k];
    J[27 * k + 8] += dqdc[k];
  }
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 120: CH3 + O + M => CH3O + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[4] * sc[12];
  k_f = 50000;
  dlnkfdT = 0.0;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] -= q;  // O
  wdot[12] -= q; // CH3
  wdot[18] += q; // CH3O
  // for convenience
  k_f *= alpha;
  dqdc[0] = q_nocor;
  dqdc[1] = q_nocor;
  dqdc[2] = q_nocor;
  dqdc[3] = q_nocor;
  dqdc[4] = q_nocor + k_f * sc[12];
  dqdc[5] = q_nocor;
  dqdc[6] = q_nocor;
  dqdc[7] = q_nocor;
  dqdc[8] = q_nocor;
  dqdc[9] = q_nocor;
  dqdc[10] = q_nocor;
  dqdc[11] = q_nocor;
  dqdc[12] = q_nocor + k_f * sc[4];
  dqdc[13] = q_nocor;
  dqdc[14] = q_nocor;
  dqdc[15] = q_nocor;
  dqdc[16] = q_nocor;
  dqdc[17] = q_nocor;
  dqdc[18] = q_nocor;
  dqdc[19] = q_nocor;
  dqdc[20] = q_nocor;
  dqdc[21] = q_nocor;
  dqdc[22] = q_nocor;
  dqdc[23] = q_nocor;
  dqdc[24] = q_nocor;
  dqdc[25] = q_nocor;
  for (int k = 0; k < 26; k++) {
    J[27 * k + 4] -= dqdc[k];
    J[27 * k + 12] -= dqdc[k];
    J[27 * k + 18] += dqdc[k];
  }
  J[706] -= dqdT; // dwdot[O]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 131: E + O2 + O2 <=> O2- + O2
  // a third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[3] * sc[3]) * sc[25];
  k_f = 1520000000 * exp(-1 * logT - (600.156324305341) * invT);
  dlnkfdT = -1 * invT + (600.156324305341) * invT2;
  // reverse
  phi_r = sc[3] * sc[24];
  Kc = refCinv * (kc_fast ? (inv_ge[25] * inv_ge[3] * ge[24])
                          : exp(g_RT[3] - g_RT[24] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(2.000000 * h_RT[3] + h_RT[25]) + (h_RT[3] + h_RT[24]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[24] += q; // O2-
  wdot[25] -= q; // E
  // d()/d[O2]
  dqdci = +k_f * 2.000000 * sc[3] * sc[25] - k_r * sc[24];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[105] += dqdci; // dwdot[O2-]/d[O2]
  J[106] -= dqdci; // dwdot[E]/d[O2]
  // d()/d[O2-]
  dqdci = -k_r * sc[3];
  J[651] -= dqdci; // dwdot[O2]/d[O2-]
  J[672] += dqdci; // dwdot[O2-]/d[O2-]
  J[673] -= dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = +k_f * sc[3] * sc[3];
  J[678] -= dqdci; // dwdot[O2]/d[E]
  J[699] += dqdci; // dwdot[O2-]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[726] += dqdT; // dwdot[O2-]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 132: E + O2 + H2O <=> O2- + H2O
  // a third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[5] * sc[25];
  k_f = 5080000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[24];
  Kc = refCinv * (kc_fast ? (inv_ge[25] * inv_ge[3] * ge[24])
                          : exp(g_RT[3] - g_RT[24] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[3] + h_RT[5] + h_RT[25]) + (h_RT[5] + h_RT[24]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[24] += q; // O2-
  wdot[25] -= q; // E
  // d()/d[O2]
  dqdci = +k_f * sc[5] * sc[25];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[105] += dqdci; // dwdot[O2-]/d[O2]
  J[106] -= dqdci; // dwdot[E]/d[O2]
  // d()/d[H2O]
  dqdci = +k_f * sc[3] * sc[25] - k_r * sc[24];
  J[138] -= dqdci; // dwdot[O2]/d[H2O]
  J[159] += dqdci; // dwdot[O2-]/d[H2O]
  J[160] -= dqdci; // dwdot[E]/d[H2O]
  // d()/d[O2-]
  dqdci = -k_r * sc[5];
  J[651] -= dqdci; // dwdot[O2]/d[O2-]
  J[672] += dqdci; // dwdot[O2-]/d[O2-]
  J[673] -= dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = +k_f * sc[3] * sc[5];
  J[678] -= dqdci; // dwdot[O2]/d[E]
  J[699] += dqdci; // dwdot[O2-]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[726] += dqdT; // dwdot[O2-]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 133: E + O2 + N2 <=> O2- + N2
  // a third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[3] * sc[25];
  k_f = 3590000000 * exp(-2 * logT - (69.7558942138502) * invT);
  dlnkfdT = -2 * invT + (69.7558942138502) * invT2;
  // reverse
  phi_r = sc[0] * sc[24];
  Kc = refCinv * (kc_fast ? (inv_ge[25] * inv_ge[3] * ge[24])
                          : exp(g_RT[3] - g_RT[24] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[0] + h_RT[3] + h_RT[25]) + (h_RT[0] + h_RT[24]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[24] += q; // O2-
  wdot[25] -= q; // E
  // d()/d[N2]
  dqdci = +k_f * sc[3] * sc[25] - k_r * sc[24];
  J[3] -= dqdci;  // dwdot[O2]/d[N2]
  J[24] += dqdci; // dwdot[O2-]/d[N2]
  J[25] -= dqdci; // dwdot[E]/d[N2]
  // d()/d[O2]
  dqdci = +k_f * sc[0] * sc[25];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[105] += dqdci; // dwdot[O2-]/d[O2]
  J[106] -= dqdci; // dwdot[E]/d[O2]
  // d()/d[O2-]
  dqdci = -k_r * sc[0];
  J[651] -= dqdci; // dwdot[O2]/d[O2-]
  J[672] += dqdci; // dwdot[O2-]/d[O2-]
  J[673] -= dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = +k_f * sc[0] * sc[3];
  J[678] -= dqdci; // dwdot[O2]/d[E]
  J[699] += dqdci; // dwdot[O2-]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[726] += dqdT; // dwdot[O2-]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 0: H2 + O <=> H + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[4];
  k_f = 0.0508 * exp(2.67 * logT - (3166.2392612433) * invT);
  dlnkfdT = 2.67 * invT + (3166.2392612433) * invT2;
  // reverse
  phi_r = sc[2] * sc[6];
  Kc =
    (kc_fast ? (inv_ge[1] * ge[6] * inv_ge[4] * ge[2])
             : exp(g_RT[1] - g_RT[2] + g_RT[4] - g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[2] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H2
  wdot[2] += q; // H
  wdot[4] -= q; // O
  wdot[6] += q; // OH
  // d()/d[H2]
  dqdci = +k_f * sc[4];
  J[28] -= dqdci; // dwdot[H2]/d[H2]
  J[29] += dqdci; // dwdot[H]/d[H2]
  J[31] -= dqdci; // dwdot[O]/d[H2]
  J[33] += dqdci; // dwdot[OH]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[6];
  J[55] -= dqdci; // dwdot[H2]/d[H]
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[60] += dqdci; // dwdot[OH]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[1];
  J[109] -= dqdci; // dwdot[H2]/d[O]
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[2];
  J[163] -= dqdci; // dwdot[H2]/d[OH]
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  // d()/dT
  J[703] -= dqdT; // dwdot[H2]/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT

  // reaction 1: H2 + OH <=> H + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[6];
  k_f = 43800000 * exp(-(3517.48449397499) * invT);
  dlnkfdT = (3517.48449397499) * invT2;
  // reverse
  phi_r = sc[2] * sc[5];
  Kc =
    (kc_fast ? (inv_ge[1] * inv_ge[6] * ge[2] * ge[5])
             : exp(g_RT[1] - g_RT[2] - g_RT[5] + g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[6]) + (h_RT[2] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H2
  wdot[2] += q; // H
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  // d()/d[H2]
  dqdci = +k_f * sc[6];
  J[28] -= dqdci; // dwdot[H2]/d[H2]
  J[29] += dqdci; // dwdot[H]/d[H2]
  J[32] += dqdci; // dwdot[H2O]/d[H2]
  J[33] -= dqdci; // dwdot[OH]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[5];
  J[55] -= dqdci; // dwdot[H2]/d[H]
  J[56] += dqdci; // dwdot[H]/d[H]
  J[59] += dqdci; // dwdot[H2O]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  // d()/d[H2O]
  dqdci = -k_r * sc[2];
  J[136] -= dqdci; // dwdot[H2]/d[H2O]
  J[137] += dqdci; // dwdot[H]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[1];
  J[163] -= dqdci; // dwdot[H2]/d[OH]
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  // d()/dT
  J[703] -= dqdT; // dwdot[H2]/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT

  // reaction 2: H + O2 <=> O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[3];
  k_f = 114000000 * exp(-(7692.16995349094) * invT);
  dlnkfdT = (7692.16995349094) * invT2;
  // reverse
  phi_r = sc[4] * sc[6];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[3] * inv_ge[2] * ge[4])
             : exp(g_RT[2] + g_RT[3] - g_RT[4] - g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[4] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[3] -= q; // O2
  wdot[4] += q; // O
  wdot[6] += q; // OH
  // d()/d[H]
  dqdci = +k_f * sc[3];
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[57] -= dqdci; // dwdot[O2]/d[H]
  J[58] += dqdci; // dwdot[O]/d[H]
  J[60] += dqdci; // dwdot[OH]/d[H]
  // d()/d[O2]
  dqdci = +k_f * sc[2];
  J[83] -= dqdci; // dwdot[H]/d[O2]
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[85] += dqdci; // dwdot[O]/d[O2]
  J[87] += dqdci; // dwdot[OH]/d[O2]
  // d()/d[O]
  dqdci = -k_r * sc[6];
  J[110] -= dqdci; // dwdot[H]/d[O]
  J[111] -= dqdci; // dwdot[O2]/d[O]
  J[112] += dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[4];
  J[164] -= dqdci; // dwdot[H]/d[OH]
  J[165] -= dqdci; // dwdot[O2]/d[OH]
  J[166] += dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  // d()/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[706] += dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT

  // reaction 4: H2O + O <=> 2 OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[5];
  k_f = 67 * exp(1.704 * logT - (7541.60752708217) * invT);
  dlnkfdT = 1.704 * invT + (7541.60752708217) * invT2;
  // reverse
  phi_r = (sc[6] * sc[6]);
  Kc =
    (kc_fast ? (ge[6] * ge[6] * inv_ge[4] * inv_ge[5])
             : exp(g_RT[4] + g_RT[5] - 2.000000 * g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[5]) + (2.000000 * h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;     // O
  wdot[5] -= q;     // H2O
  wdot[6] += 2 * q; // OH
  // d()/d[O]
  dqdci = +k_f * sc[5];
  J[112] -= dqdci;     // dwdot[O]/d[O]
  J[113] -= dqdci;     // dwdot[H2O]/d[O]
  J[114] += 2 * dqdci; // dwdot[OH]/d[O]
  // d()/d[H2O]
  dqdci = +k_f * sc[4];
  J[139] -= dqdci;     // dwdot[O]/d[H2O]
  J[140] -= dqdci;     // dwdot[H2O]/d[H2O]
  J[141] += 2 * dqdci; // dwdot[OH]/d[H2O]
  // d()/d[OH]
  dqdci = -k_r * 2.000000 * sc[6];
  J[166] -= dqdci;     // dwdot[O]/d[OH]
  J[167] -= dqdci;     // dwdot[H2O]/d[OH]
  J[168] += 2 * dqdci; // dwdot[OH]/d[OH]
  // d()/dT
  J[706] -= dqdT;     // dwdot[O]/dT
  J[707] -= dqdT;     // dwdot[H2O]/dT
  J[708] += 2 * dqdT; // dwdot[OH]/dT

  // reaction 7: H + H2O2 <=> H2O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[7];
  k_f = 24100000 * exp(-(1997.77016324474) * invT);
  dlnkfdT = (1997.77016324474) * invT2;
  // reverse
  phi_r = sc[5] * sc[6];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[7] * inv_ge[2] * ge[5])
             : exp(g_RT[2] - g_RT[5] - g_RT[6] + g_RT[7]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[5] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[5] += q; // H2O
  wdot[6] += q; // OH
  wdot[7] -= q; // H2O2
  // d()/d[H]
  dqdci = +k_f * sc[7];
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[59] += dqdci; // dwdot[H2O]/d[H]
  J[60] += dqdci; // dwdot[OH]/d[H]
  J[61] -= dqdci; // dwdot[H2O2]/d[H]
  // d()/d[H2O]
  dqdci = -k_r * sc[6];
  J[137] -= dqdci; // dwdot[H]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] += dqdci; // dwdot[OH]/d[H2O]
  J[142] -= dqdci; // dwdot[H2O2]/d[H2O]
  // d()/d[OH]
  dqdci = -k_r * sc[5];
  J[164] -= dqdci; // dwdot[H]/d[OH]
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[169] -= dqdci; // dwdot[H2O2]/d[OH]
  // d()/d[H2O2]
  dqdci = +k_f * sc[2];
  J[191] -= dqdci; // dwdot[H]/d[H2O2]
  J[194] += dqdci; // dwdot[H2O]/d[H2O2]
  J[195] += dqdci; // dwdot[OH]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  // d()/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT

  // reaction 8: H + H2O2 <=> H2 + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[7];
  k_f = 21500 * exp(1 * logT - (3019.29999482832) * invT);
  dlnkfdT = 1 * invT + (3019.29999482832) * invT2;
  // reverse
  phi_r = sc[1] * sc[8];
  Kc =
    (kc_fast ? (ge[1] * ge[8] * inv_ge[7] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[7] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[1] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[7] -= q; // H2O2
  wdot[8] += q; // HO2
  // d()/d[H2]
  dqdci = -k_r * sc[8];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[34] -= dqdci; // dwdot[H2O2]/d[H2]
  J[35] += dqdci; // dwdot[HO2]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[7];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[61] -= dqdci; // dwdot[H2O2]/d[H]
  J[62] += dqdci; // dwdot[HO2]/d[H]
  // d()/d[H2O2]
  dqdci = +k_f * sc[2];
  J[190] += dqdci; // dwdot[H2]/d[H2O2]
  J[191] -= dqdci; // dwdot[H]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] += dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[1];
  J[217] += dqdci; // dwdot[H2]/d[HO2]
  J[218] -= dqdci; // dwdot[H]/d[HO2]
  J[223] -= dqdci; // dwdot[H2O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 9: H2O2 + O <=> HO2 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[7];
  k_f = 9.55 * exp(2 * logT - (1997.77016324474) * invT);
  dlnkfdT = 2 * invT + (1997.77016324474) * invT2;
  // reverse
  phi_r = sc[6] * sc[8];
  Kc =
    (kc_fast ? (ge[6] * ge[8] * inv_ge[7] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[7] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[7]) + (h_RT[6] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q; // O
  wdot[6] += q; // OH
  wdot[7] -= q; // H2O2
  wdot[8] += q; // HO2
  // d()/d[O]
  dqdci = +k_f * sc[7];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[115] -= dqdci; // dwdot[H2O2]/d[O]
  J[116] += dqdci; // dwdot[HO2]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[8];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[169] -= dqdci; // dwdot[H2O2]/d[OH]
  J[170] += dqdci; // dwdot[HO2]/d[OH]
  // d()/d[H2O2]
  dqdci = +k_f * sc[4];
  J[193] -= dqdci; // dwdot[O]/d[H2O2]
  J[195] += dqdci; // dwdot[OH]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] += dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[6];
  J[220] -= dqdci; // dwdot[O]/d[HO2]
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[223] -= dqdci; // dwdot[H2O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 10: H2O2 + OH <=> H2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[7];
  k_f = 1740000 * exp(-(160.022899725901) * invT);
  dlnkfdT = (160.022899725901) * invT2;
  // reverse
  phi_r = sc[5] * sc[8];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[7] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[7] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[7]) + (h_RT[5] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  wdot[7] -= q; // H2O2
  wdot[8] += q; // HO2
  // d()/d[H2O]
  dqdci = -k_r * sc[8];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[142] -= dqdci; // dwdot[H2O2]/d[H2O]
  J[143] += dqdci; // dwdot[HO2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[7];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[169] -= dqdci; // dwdot[H2O2]/d[OH]
  J[170] += dqdci; // dwdot[HO2]/d[OH]
  // d()/d[H2O2]
  dqdci = +k_f * sc[6];
  J[194] += dqdci; // dwdot[H2O]/d[H2O2]
  J[195] -= dqdci; // dwdot[OH]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] += dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[5];
  J[221] += dqdci; // dwdot[H2O]/d[HO2]
  J[222] -= dqdci; // dwdot[OH]/d[HO2]
  J[223] -= dqdci; // dwdot[H2O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 11: H2O2 + OH <=> H2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[7];
  k_f = 75900000 * exp(-(3657.88194373451) * invT);
  dlnkfdT = (3657.88194373451) * invT2;
  // reverse
  phi_r = sc[5] * sc[8];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[7] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[7] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[7]) + (h_RT[5] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  wdot[7] -= q; // H2O2
  wdot[8] += q; // HO2
  // d()/d[H2O]
  dqdci = -k_r * sc[8];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[142] -= dqdci; // dwdot[H2O2]/d[H2O]
  J[143] += dqdci; // dwdot[HO2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[7];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[169] -= dqdci; // dwdot[H2O2]/d[OH]
  J[170] += dqdci; // dwdot[HO2]/d[OH]
  // d()/d[H2O2]
  dqdci = +k_f * sc[6];
  J[194] += dqdci; // dwdot[H2O]/d[H2O2]
  J[195] -= dqdci; // dwdot[OH]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] += dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[5];
  J[221] += dqdci; // dwdot[H2O]/d[HO2]
  J[222] -= dqdci; // dwdot[OH]/d[HO2]
  J[223] -= dqdci; // dwdot[H2O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 12: H + HO2 <=> 2 OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 70790000 * exp(-(148.448916412392) * invT);
  dlnkfdT = (148.448916412392) * invT2;
  // reverse
  phi_r = (sc[6] * sc[6]);
  Kc =
    (kc_fast ? (ge[6] * inv_ge[8] * ge[6] * inv_ge[2])
             : exp(g_RT[2] - 2.000000 * g_RT[6] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (2.000000 * h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q;     // H
  wdot[6] += 2 * q; // OH
  wdot[8] -= q;     // HO2
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[56] -= dqdci;     // dwdot[H]/d[H]
  J[60] += 2 * dqdci; // dwdot[OH]/d[H]
  J[62] -= dqdci;     // dwdot[HO2]/d[H]
  // d()/d[OH]
  dqdci = -k_r * 2.000000 * sc[6];
  J[164] -= dqdci;     // dwdot[H]/d[OH]
  J[168] += 2 * dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci;     // dwdot[HO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[2];
  J[218] -= dqdci;     // dwdot[H]/d[HO2]
  J[222] += 2 * dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci;     // dwdot[HO2]/d[HO2]
  // d()/dT
  J[704] -= dqdT;     // dwdot[H]/dT
  J[708] += 2 * dqdT; // dwdot[OH]/dT
  J[710] -= dqdT;     // dwdot[HO2]/dT

  // reaction 13: H + HO2 <=> H2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 11402 * exp(1.083 * logT - (278.671325189338) * invT);
  dlnkfdT = 1.083 * invT + (278.671325189338) * invT2;
  // reverse
  phi_r = sc[1] * sc[3];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[8] * ge[3] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (h_RT[1] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[3] += q; // O2
  wdot[8] -= q; // HO2
  // d()/d[H2]
  dqdci = -k_r * sc[3];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[30] += dqdci; // dwdot[O2]/d[H2]
  J[35] -= dqdci; // dwdot[HO2]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[57] += dqdci; // dwdot[O2]/d[H]
  J[62] -= dqdci; // dwdot[HO2]/d[H]
  // d()/d[O2]
  dqdci = -k_r * sc[1];
  J[82] += dqdci; // dwdot[H2]/d[O2]
  J[83] -= dqdci; // dwdot[H]/d[O2]
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[2];
  J[217] += dqdci; // dwdot[H2]/d[HO2]
  J[218] -= dqdci; // dwdot[H]/d[HO2]
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT

  // reaction 14: HO2 + O <=> O2 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[8];
  k_f = 32500000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[3] * sc[6];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[8] * ge[3] * inv_ge[4])
             : exp(-g_RT[3] + g_RT[4] - g_RT[6] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[8]) + (h_RT[3] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q; // O2
  wdot[4] -= q; // O
  wdot[6] += q; // OH
  wdot[8] -= q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[6];
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[85] -= dqdci; // dwdot[O]/d[O2]
  J[87] += dqdci; // dwdot[OH]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  // d()/d[O]
  dqdci = +k_f * sc[8];
  J[111] += dqdci; // dwdot[O2]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[116] -= dqdci; // dwdot[HO2]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[3];
  J[165] += dqdci; // dwdot[O2]/d[OH]
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[4];
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[220] -= dqdci; // dwdot[O]/d[HO2]
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT

  // reaction 15: HO2 + OH <=> H2O + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[8];
  k_f = 7000000 * exp(-(-549.995687057926) * invT);
  dlnkfdT = (-549.995687057926) * invT2;
  // reverse
  phi_r = sc[3] * sc[5];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[3] * inv_ge[8] * ge[5])
             : exp(-g_RT[3] - g_RT[5] + g_RT[6] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[8]) + (h_RT[3] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q; // O2
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  wdot[8] -= q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[5];
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[86] += dqdci; // dwdot[H2O]/d[O2]
  J[87] -= dqdci; // dwdot[OH]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  // d()/d[H2O]
  dqdci = -k_r * sc[3];
  J[138] += dqdci; // dwdot[O2]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[143] -= dqdci; // dwdot[HO2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[8];
  J[165] += dqdci; // dwdot[O2]/d[OH]
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[6];
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[221] += dqdci; // dwdot[H2O]/d[HO2]
  J[222] -= dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT

  // reaction 16: HO2 + OH <=> H2O + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[8];
  k_f = 450000000 * exp(-(5499.95687057926) * invT);
  dlnkfdT = (5499.95687057926) * invT2;
  // reverse
  phi_r = sc[3] * sc[5];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[3] * inv_ge[8] * ge[5])
             : exp(-g_RT[3] - g_RT[5] + g_RT[6] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[8]) + (h_RT[3] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q; // O2
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  wdot[8] -= q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[5];
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[86] += dqdci; // dwdot[H2O]/d[O2]
  J[87] -= dqdci; // dwdot[OH]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  // d()/d[H2O]
  dqdci = -k_r * sc[3];
  J[138] += dqdci; // dwdot[O2]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[143] -= dqdci; // dwdot[HO2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[8];
  J[165] += dqdci; // dwdot[O2]/d[OH]
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[6];
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[221] += dqdci; // dwdot[H2O]/d[HO2]
  J[222] -= dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT

  // reaction 17: 2 HO2 <=> H2O2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[8] * sc[8]);
  k_f = 100000000 * exp(-(5555.95482115001) * invT);
  dlnkfdT = (5555.95482115001) * invT2;
  // reverse
  phi_r = sc[3] * sc[7];
  Kc =
    (kc_fast ? (ge[3] * inv_ge[8] * inv_ge[8] * ge[7])
             : exp(-g_RT[3] - g_RT[7] + 2.000000 * g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[8]) + (h_RT[3] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;     // O2
  wdot[7] += q;     // H2O2
  wdot[8] -= 2 * q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[7];
  J[84] += dqdci;      // dwdot[O2]/d[O2]
  J[88] += dqdci;      // dwdot[H2O2]/d[O2]
  J[89] += -2 * dqdci; // dwdot[HO2]/d[O2]
  // d()/d[H2O2]
  dqdci = -k_r * sc[3];
  J[192] += dqdci;      // dwdot[O2]/d[H2O2]
  J[196] += dqdci;      // dwdot[H2O2]/d[H2O2]
  J[197] += -2 * dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * 2.000000 * sc[8];
  J[219] += dqdci;      // dwdot[O2]/d[HO2]
  J[223] += dqdci;      // dwdot[H2O2]/d[HO2]
  J[224] += -2 * dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT;      // dwdot[O2]/dT
  J[709] += dqdT;      // dwdot[H2O2]/dT
  J[710] += -2 * dqdT; // dwdot[HO2]/dT

  // reaction 18: 2 HO2 <=> H2O2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[8] * sc[8]);
  k_f = 190000 * exp(-(-708.992024785586) * invT);
  dlnkfdT = (-708.992024785586) * invT2;
  // reverse
  phi_r = sc[3] * sc[7];
  Kc =
    (kc_fast ? (ge[3] * inv_ge[8] * inv_ge[8] * ge[7])
             : exp(-g_RT[3] - g_RT[7] + 2.000000 * g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[8]) + (h_RT[3] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;     // O2
  wdot[7] += q;     // H2O2
  wdot[8] -= 2 * q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[7];
  J[84] += dqdci;      // dwdot[O2]/d[O2]
  J[88] += dqdci;      // dwdot[H2O2]/d[O2]
  J[89] += -2 * dqdci; // dwdot[HO2]/d[O2]
  // d()/d[H2O2]
  dqdci = -k_r * sc[3];
  J[192] += dqdci;      // dwdot[O2]/d[H2O2]
  J[196] += dqdci;      // dwdot[H2O2]/d[H2O2]
  J[197] += -2 * dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * 2.000000 * sc[8];
  J[219] += dqdci;      // dwdot[O2]/d[HO2]
  J[223] += dqdci;      // dwdot[H2O2]/d[HO2]
  J[224] += -2 * dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT;      // dwdot[O2]/dT
  J[709] += dqdT;      // dwdot[H2O2]/dT
  J[710] += -2 * dqdT; // dwdot[HO2]/dT

  // reaction 21: CO + OH <=> CO2 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[9];
  k_f = 0.07015 * exp(2.053 * logT - (-178.994168026739) * invT);
  dlnkfdT = 2.053 * invT + (-178.994168026739) * invT2;
  // reverse
  phi_r = sc[2] * sc[10];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[9] * ge[10] * ge[2])
             : exp(-g_RT[2] + g_RT[6] + g_RT[9] - g_RT[10]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[9]) + (h_RT[2] + h_RT[10]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  // d()/d[H]
  dqdci = -k_r * sc[10];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[63] -= dqdci; // dwdot[CO]/d[H]
  J[64] += dqdci; // dwdot[CO2]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[9];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[171] -= dqdci; // dwdot[CO]/d[OH]
  J[172] += dqdci; // dwdot[CO2]/d[OH]
  // d()/d[CO]
  dqdci = +k_f * sc[6];
  J[245] += dqdci; // dwdot[H]/d[CO]
  J[249] -= dqdci; // dwdot[OH]/d[CO]
  J[252] -= dqdci; // dwdot[CO]/d[CO]
  J[253] += dqdci; // dwdot[CO2]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * sc[2];
  J[272] += dqdci; // dwdot[H]/d[CO2]
  J[276] -= dqdci; // dwdot[OH]/d[CO2]
  J[279] -= dqdci; // dwdot[CO]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT

  // reaction 22: CO + OH <=> CO2 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[9];
  k_f = 5757000 * exp(-0.664 * logT - (166.967289714006) * invT);
  dlnkfdT = -0.664 * invT + (166.967289714006) * invT2;
  // reverse
  phi_r = sc[2] * sc[10];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[9] * ge[10] * ge[2])
             : exp(-g_RT[2] + g_RT[6] + g_RT[9] - g_RT[10]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[9]) + (h_RT[2] + h_RT[10]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  // d()/d[H]
  dqdci = -k_r * sc[10];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[63] -= dqdci; // dwdot[CO]/d[H]
  J[64] += dqdci; // dwdot[CO2]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[9];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[171] -= dqdci; // dwdot[CO]/d[OH]
  J[172] += dqdci; // dwdot[CO2]/d[OH]
  // d()/d[CO]
  dqdci = +k_f * sc[6];
  J[245] += dqdci; // dwdot[H]/d[CO]
  J[249] -= dqdci; // dwdot[OH]/d[CO]
  J[252] -= dqdci; // dwdot[CO]/d[CO]
  J[253] += dqdci; // dwdot[CO2]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * sc[2];
  J[272] += dqdci; // dwdot[H]/d[CO2]
  J[276] -= dqdci; // dwdot[OH]/d[CO2]
  J[279] -= dqdci; // dwdot[CO]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT

  // reaction 23: CO + HO2 <=> CO2 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[9];
  k_f = 0.157 * exp(2.18 * logT - (9027.70698453667) * invT);
  dlnkfdT = 2.18 * invT + (9027.70698453667) * invT2;
  // reverse
  phi_r = sc[6] * sc[10];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[8] * inv_ge[9] * ge[10])
             : exp(-g_RT[6] + g_RT[8] + g_RT[9] - g_RT[10]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[9]) + (h_RT[6] + h_RT[10]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[6] += q;  // OH
  wdot[8] -= q;  // HO2
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  // d()/d[OH]
  dqdci = -k_r * sc[10];
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  J[171] -= dqdci; // dwdot[CO]/d[OH]
  J[172] += dqdci; // dwdot[CO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[9];
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[225] -= dqdci; // dwdot[CO]/d[HO2]
  J[226] += dqdci; // dwdot[CO2]/d[HO2]
  // d()/d[CO]
  dqdci = +k_f * sc[8];
  J[249] += dqdci; // dwdot[OH]/d[CO]
  J[251] -= dqdci; // dwdot[HO2]/d[CO]
  J[252] -= dqdci; // dwdot[CO]/d[CO]
  J[253] += dqdci; // dwdot[CO2]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * sc[6];
  J[276] += dqdci; // dwdot[OH]/d[CO2]
  J[278] -= dqdci; // dwdot[HO2]/d[CO2]
  J[279] -= dqdci; // dwdot[CO]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  // d()/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT

  // reaction 25: CH4 + H <=> CH3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[11];
  k_f = 0.614 * exp(2.5 * logT - (4824.33817506985) * invT);
  dlnkfdT = 2.5 * invT + (4824.33817506985) * invT2;
  // reverse
  phi_r = sc[1] * sc[12];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[11] * inv_ge[2] * ge[12])
             : exp(-g_RT[1] + g_RT[2] + g_RT[11] - g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[11]) + (h_RT[1] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[11] -= q; // CH4
  wdot[12] += q; // CH3
  // d()/d[H2]
  dqdci = -k_r * sc[12];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[38] -= dqdci; // dwdot[CH4]/d[H2]
  J[39] += dqdci; // dwdot[CH3]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[11];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[65] -= dqdci; // dwdot[CH4]/d[H]
  J[66] += dqdci; // dwdot[CH3]/d[H]
  // d()/d[CH4]
  dqdci = +k_f * sc[2];
  J[298] += dqdci; // dwdot[H2]/d[CH4]
  J[299] -= dqdci; // dwdot[H]/d[CH4]
  J[308] -= dqdci; // dwdot[CH4]/d[CH4]
  J[309] += dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * sc[1];
  J[325] += dqdci; // dwdot[H2]/d[CH3]
  J[326] -= dqdci; // dwdot[H]/d[CH3]
  J[335] -= dqdci; // dwdot[CH4]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[713] -= dqdT; // dwdot[CH4]/dT
  J[714] += dqdT; // dwdot[CH3]/dT

  // reaction 26: CH4 + O <=> CH3 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[11];
  k_f = 1020 * exp(1.5 * logT - (4327.66332592059) * invT);
  dlnkfdT = 1.5 * invT + (4327.66332592059) * invT2;
  // reverse
  phi_r = sc[6] * sc[12];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[11] * inv_ge[4] * ge[12])
             : exp(g_RT[4] - g_RT[6] + g_RT[11] - g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[11]) + (h_RT[6] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[11] -= q; // CH4
  wdot[12] += q; // CH3
  // d()/d[O]
  dqdci = +k_f * sc[11];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[119] -= dqdci; // dwdot[CH4]/d[O]
  J[120] += dqdci; // dwdot[CH3]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[12];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[173] -= dqdci; // dwdot[CH4]/d[OH]
  J[174] += dqdci; // dwdot[CH3]/d[OH]
  // d()/d[CH4]
  dqdci = +k_f * sc[4];
  J[301] -= dqdci; // dwdot[O]/d[CH4]
  J[303] += dqdci; // dwdot[OH]/d[CH4]
  J[308] -= dqdci; // dwdot[CH4]/d[CH4]
  J[309] += dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * sc[6];
  J[328] -= dqdci; // dwdot[O]/d[CH3]
  J[330] += dqdci; // dwdot[OH]/d[CH3]
  J[335] -= dqdci; // dwdot[CH4]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[713] -= dqdT; // dwdot[CH4]/dT
  J[714] += dqdT; // dwdot[CH3]/dT

  // reaction 27: CH4 + OH <=> CH3 + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[11];
  k_f = 0.0583 * exp(2.6 * logT - (1102.04449811234) * invT);
  dlnkfdT = 2.6 * invT + (1102.04449811234) * invT2;
  // reverse
  phi_r = sc[5] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[12] * ge[5] * inv_ge[11])
             : exp(-g_RT[5] + g_RT[6] + g_RT[11] - g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[11]) + (h_RT[5] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[11] -= q; // CH4
  wdot[12] += q; // CH3
  // d()/d[H2O]
  dqdci = -k_r * sc[12];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[146] -= dqdci; // dwdot[CH4]/d[H2O]
  J[147] += dqdci; // dwdot[CH3]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[11];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[173] -= dqdci; // dwdot[CH4]/d[OH]
  J[174] += dqdci; // dwdot[CH3]/d[OH]
  // d()/d[CH4]
  dqdci = +k_f * sc[6];
  J[302] += dqdci; // dwdot[H2O]/d[CH4]
  J[303] -= dqdci; // dwdot[OH]/d[CH4]
  J[308] -= dqdci; // dwdot[CH4]/d[CH4]
  J[309] += dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * sc[5];
  J[329] += dqdci; // dwdot[H2O]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[335] -= dqdci; // dwdot[CH4]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[713] -= dqdT; // dwdot[CH4]/dT
  J[714] += dqdT; // dwdot[CH3]/dT

  // reaction 28: CH4 + HO2 <=> CH3 + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[11];
  k_f = 1.13e-05 * exp(3.74 * logT - (10572.5821485572) * invT);
  dlnkfdT = 3.74 * invT + (10572.5821485572) * invT2;
  // reverse
  phi_r = sc[7] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[12] * ge[7] * inv_ge[11])
             : exp(-g_RT[7] + g_RT[8] + g_RT[11] - g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[11]) + (h_RT[7] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[11] -= q; // CH4
  wdot[12] += q; // CH3
  // d()/d[H2O2]
  dqdci = -k_r * sc[12];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[200] -= dqdci; // dwdot[CH4]/d[H2O2]
  J[201] += dqdci; // dwdot[CH3]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[11];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[227] -= dqdci; // dwdot[CH4]/d[HO2]
  J[228] += dqdci; // dwdot[CH3]/d[HO2]
  // d()/d[CH4]
  dqdci = +k_f * sc[8];
  J[304] += dqdci; // dwdot[H2O2]/d[CH4]
  J[305] -= dqdci; // dwdot[HO2]/d[CH4]
  J[308] -= dqdci; // dwdot[CH4]/d[CH4]
  J[309] += dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * sc[7];
  J[331] += dqdci; // dwdot[H2O2]/d[CH3]
  J[332] -= dqdci; // dwdot[HO2]/d[CH3]
  J[335] -= dqdci; // dwdot[CH4]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[713] -= dqdT; // dwdot[CH4]/dT
  J[714] += dqdT; // dwdot[CH3]/dT

  // reaction 29: CH3 + HO2 <=> CH4 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[12];
  k_f = 0.116 * exp(2.23 * logT - (-1520.72076406186) * invT);
  dlnkfdT = 2.23 * invT + (-1520.72076406186) * invT2;
  // reverse
  phi_r = sc[3] * sc[11];
  Kc =
    (kc_fast ? (ge[3] * inv_ge[8] * inv_ge[12] * ge[11])
             : exp(-g_RT[3] + g_RT[8] - g_RT[11] + g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[12]) + (h_RT[3] + h_RT[11]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;  // O2
  wdot[8] -= q;  // HO2
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  // d()/d[O2]
  dqdci = -k_r * sc[11];
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  J[92] += dqdci; // dwdot[CH4]/d[O2]
  J[93] -= dqdci; // dwdot[CH3]/d[O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[12];
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[227] += dqdci; // dwdot[CH4]/d[HO2]
  J[228] -= dqdci; // dwdot[CH3]/d[HO2]
  // d()/d[CH4]
  dqdci = -k_r * sc[3];
  J[300] += dqdci; // dwdot[O2]/d[CH4]
  J[305] -= dqdci; // dwdot[HO2]/d[CH4]
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[8];
  J[327] += dqdci; // dwdot[O2]/d[CH3]
  J[332] -= dqdci; // dwdot[HO2]/d[CH3]
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT

  // reaction 30: CH2 + CH4 <=> 2 CH3
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[11] * sc[13];
  k_f = 2.46 * exp(2 * logT - (4161.60182620503) * invT);
  dlnkfdT = 2 * invT + (4161.60182620503) * invT2;
  // reverse
  phi_r = (sc[12] * sc[12]);
  Kc =
    (kc_fast ? (ge[12] * ge[12] * inv_ge[13] * inv_ge[11])
             : exp(g_RT[11] - 2.000000 * g_RT[12] + g_RT[13]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[11] + h_RT[13]) + (2.000000 * h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] -= q;     // CH4
  wdot[12] += 2 * q; // CH3
  wdot[13] -= q;     // CH2
  // d()/d[CH4]
  dqdci = +k_f * sc[13];
  J[308] -= dqdci;     // dwdot[CH4]/d[CH4]
  J[309] += 2 * dqdci; // dwdot[CH3]/d[CH4]
  J[310] -= dqdci;     // dwdot[CH2]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * 2.000000 * sc[12];
  J[335] -= dqdci;     // dwdot[CH4]/d[CH3]
  J[336] += 2 * dqdci; // dwdot[CH3]/d[CH3]
  J[337] -= dqdci;     // dwdot[CH2]/d[CH3]
  // d()/d[CH2]
  dqdci = +k_f * sc[11];
  J[362] -= dqdci;     // dwdot[CH4]/d[CH2]
  J[363] += 2 * dqdci; // dwdot[CH3]/d[CH2]
  J[364] -= dqdci;     // dwdot[CH2]/d[CH2]
  // d()/dT
  J[713] -= dqdT;     // dwdot[CH4]/dT
  J[714] += 2 * dqdT; // dwdot[CH3]/dT
  J[715] -= dqdT;     // dwdot[CH2]/dT

  // reaction 31: CH2(S) + N2 <=> CH2 + N2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[14];
  k_f = 15000000 * exp(-(301.929999482832) * invT);
  dlnkfdT = (301.929999482832) * invT2;
  // reverse
  phi_r = sc[0] * sc[13];
  Kc =
    (kc_fast ? (ge[13] * inv_ge[14])
             : exp(g_RT[0] - g_RT[0] - g_RT[13] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[14]) + (h_RT[0] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[13] += q; // CH2
  wdot[14] -= q; // CH2(S)
  // d()/d[N2]
  dqdci = +k_f * sc[14] - k_r * sc[13];
  J[13] += dqdci; // dwdot[CH2]/d[N2]
  J[14] -= dqdci; // dwdot[CH2(S)]/d[N2]
  // d()/d[CH2]
  dqdci = -k_r * sc[0];
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  J[365] -= dqdci; // dwdot[CH2(S)]/d[CH2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[0];
  J[391] += dqdci; // dwdot[CH2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[715] += dqdT; // dwdot[CH2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 32: CH2(S) + H2O <=> CH2 + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[5] * sc[14];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[13];
  Kc =
    (kc_fast ? (ge[13] * inv_ge[14])
             : exp(g_RT[5] - g_RT[5] - g_RT[13] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[5] + h_RT[14]) + (h_RT[5] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[13] += q; // CH2
  wdot[14] -= q; // CH2(S)
  // d()/d[H2O]
  dqdci = +k_f * sc[14] - k_r * sc[13];
  J[148] += dqdci; // dwdot[CH2]/d[H2O]
  J[149] -= dqdci; // dwdot[CH2(S)]/d[H2O]
  // d()/d[CH2]
  dqdci = -k_r * sc[5];
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  J[365] -= dqdci; // dwdot[CH2(S)]/d[CH2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[5];
  J[391] += dqdci; // dwdot[CH2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[715] += dqdT; // dwdot[CH2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 33: CH2(S) + CO <=> CH2 + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[9] * sc[14];
  k_f = 9000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[13];
  Kc =
    (kc_fast ? (ge[13] * inv_ge[14])
             : exp(g_RT[9] - g_RT[9] - g_RT[13] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[9] + h_RT[14]) + (h_RT[9] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[13] += q; // CH2
  wdot[14] -= q; // CH2(S)
  // d()/d[CO]
  dqdci = +k_f * sc[14] - k_r * sc[13];
  J[256] += dqdci; // dwdot[CH2]/d[CO]
  J[257] -= dqdci; // dwdot[CH2(S)]/d[CO]
  // d()/d[CH2]
  dqdci = -k_r * sc[9];
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  J[365] -= dqdci; // dwdot[CH2(S)]/d[CH2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[9];
  J[391] += dqdci; // dwdot[CH2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[715] += dqdT; // dwdot[CH2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 34: CH2(S) + CO2 <=> CH2 + CO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[10] * sc[14];
  k_f = 7000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[10] * sc[13];
  Kc =
    (kc_fast ? (ge[13] * inv_ge[14])
             : exp(g_RT[10] - g_RT[10] - g_RT[13] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[10] + h_RT[14]) + (h_RT[10] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[13] += q; // CH2
  wdot[14] -= q; // CH2(S)
  // d()/d[CO2]
  dqdci = +k_f * sc[14] - k_r * sc[13];
  J[283] += dqdci; // dwdot[CH2]/d[CO2]
  J[284] -= dqdci; // dwdot[CH2(S)]/d[CO2]
  // d()/d[CH2]
  dqdci = -k_r * sc[10];
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  J[365] -= dqdci; // dwdot[CH2(S)]/d[CH2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[10];
  J[391] += dqdci; // dwdot[CH2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[715] += dqdT; // dwdot[CH2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 35: CH2(S) + O2 => CO + H + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[14];
  k_f = 28000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += q;  // H
  wdot[3] -= q;  // O2
  wdot[6] += q;  // OH
  wdot[9] += q;  // CO
  wdot[14] -= q; // CH2(S)
  // d()/d[O2]
  dqdci = +k_f * sc[14];
  J[83] += dqdci; // dwdot[H]/d[O2]
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[87] += dqdci; // dwdot[OH]/d[O2]
  J[90] += dqdci; // dwdot[CO]/d[O2]
  J[95] -= dqdci; // dwdot[CH2(S)]/d[O2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[3];
  J[380] += dqdci; // dwdot[H]/d[CH2(S)]
  J[381] -= dqdci; // dwdot[O2]/d[CH2(S)]
  J[384] += dqdci; // dwdot[OH]/d[CH2(S)]
  J[387] += dqdci; // dwdot[CO]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 36: CH2(S) + O2 <=> CO + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[14];
  k_f = 12000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[9] * ge[5] * inv_ge[14])
             : exp(g_RT[3] - g_RT[5] - g_RT[9] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[14]) + (h_RT[5] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[5] += q;  // H2O
  wdot[9] += q;  // CO
  wdot[14] -= q; // CH2(S)
  // d()/d[O2]
  dqdci = +k_f * sc[14];
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[86] += dqdci; // dwdot[H2O]/d[O2]
  J[90] += dqdci; // dwdot[CO]/d[O2]
  J[95] -= dqdci; // dwdot[CH2(S)]/d[O2]
  // d()/d[H2O]
  dqdci = -k_r * sc[9];
  J[138] -= dqdci; // dwdot[O2]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[144] += dqdci; // dwdot[CO]/d[H2O]
  J[149] -= dqdci; // dwdot[CH2(S)]/d[H2O]
  // d()/d[CO]
  dqdci = -k_r * sc[5];
  J[246] -= dqdci; // dwdot[O2]/d[CO]
  J[248] += dqdci; // dwdot[H2O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[257] -= dqdci; // dwdot[CH2(S)]/d[CO]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[3];
  J[381] -= dqdci; // dwdot[O2]/d[CH2(S)]
  J[383] += dqdci; // dwdot[H2O]/d[CH2(S)]
  J[387] += dqdci; // dwdot[CO]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 37: CH2(S) + O <=> CO + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[14];
  k_f = 15000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc =
    (kc_fast ? (ge[1] * ge[9] * inv_ge[4] * inv_ge[14])
             : exp(-g_RT[1] + g_RT[4] - g_RT[9] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[14]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[4] -= q;  // O
  wdot[9] += q;  // CO
  wdot[14] -= q; // CH2(S)
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[31] -= dqdci; // dwdot[O]/d[H2]
  J[36] += dqdci; // dwdot[CO]/d[H2]
  J[41] -= dqdci; // dwdot[CH2(S)]/d[H2]
  // d()/d[O]
  dqdci = +k_f * sc[14];
  J[109] += dqdci; // dwdot[H2]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[122] -= dqdci; // dwdot[CH2(S)]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[1];
  J[244] += dqdci; // dwdot[H2]/d[CO]
  J[247] -= dqdci; // dwdot[O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[257] -= dqdci; // dwdot[CH2(S)]/d[CO]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[4];
  J[379] += dqdci; // dwdot[H2]/d[CH2(S)]
  J[382] -= dqdci; // dwdot[O]/d[CH2(S)]
  J[387] += dqdci; // dwdot[CO]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 38: CH2(S) + O <=> H + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[14];
  k_f = 15000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * ge[2] * inv_ge[4] * inv_ge[14])
             : exp(-g_RT[2] + g_RT[4] + g_RT[14] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[14]) + (h_RT[2] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[4] -= q;  // O
  wdot[14] -= q; // CH2(S)
  wdot[21] += q; // HCO
  // d()/d[H]
  dqdci = -k_r * sc[21];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[68] -= dqdci; // dwdot[CH2(S)]/d[H]
  J[75] += dqdci; // dwdot[HCO]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[14];
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[122] -= dqdci; // dwdot[CH2(S)]/d[O]
  J[129] += dqdci; // dwdot[HCO]/d[O]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[4];
  J[380] += dqdci; // dwdot[H]/d[CH2(S)]
  J[382] -= dqdci; // dwdot[O]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  J[399] += dqdci; // dwdot[HCO]/d[CH2(S)]
  // d()/d[HCO]
  dqdci = -k_r * sc[2];
  J[569] += dqdci; // dwdot[H]/d[HCO]
  J[571] -= dqdci; // dwdot[O]/d[HCO]
  J[581] -= dqdci; // dwdot[CH2(S)]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 39: CH2(S) + H2 <=> CH3 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[14];
  k_f = 70000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[1] * ge[12] * inv_ge[14] * ge[2])
             : exp(g_RT[1] - g_RT[2] - g_RT[12] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[14]) + (h_RT[2] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q;  // H2
  wdot[2] += q;  // H
  wdot[12] += q; // CH3
  wdot[14] -= q; // CH2(S)
  // d()/d[H2]
  dqdci = +k_f * sc[14];
  J[28] -= dqdci; // dwdot[H2]/d[H2]
  J[29] += dqdci; // dwdot[H]/d[H2]
  J[39] += dqdci; // dwdot[CH3]/d[H2]
  J[41] -= dqdci; // dwdot[CH2(S)]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[12];
  J[55] -= dqdci; // dwdot[H2]/d[H]
  J[56] += dqdci; // dwdot[H]/d[H]
  J[66] += dqdci; // dwdot[CH3]/d[H]
  J[68] -= dqdci; // dwdot[CH2(S)]/d[H]
  // d()/d[CH3]
  dqdci = -k_r * sc[2];
  J[325] -= dqdci; // dwdot[H2]/d[CH3]
  J[326] += dqdci; // dwdot[H]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  J[338] -= dqdci; // dwdot[CH2(S)]/d[CH3]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[1];
  J[379] -= dqdci; // dwdot[H2]/d[CH2(S)]
  J[380] += dqdci; // dwdot[H]/d[CH2(S)]
  J[390] += dqdci; // dwdot[CH3]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[703] -= dqdT; // dwdot[H2]/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 40: CH2(S) + H <=> CH + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[14];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[16];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[2] * ge[16] * inv_ge[14])
             : exp(-g_RT[1] + g_RT[2] + g_RT[14] - g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[14]) + (h_RT[1] + h_RT[16]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[14] -= q; // CH2(S)
  wdot[16] += q; // CH
  // d()/d[H2]
  dqdci = -k_r * sc[16];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[41] -= dqdci; // dwdot[CH2(S)]/d[H2]
  J[43] += dqdci; // dwdot[CH]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[14];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[68] -= dqdci; // dwdot[CH2(S)]/d[H]
  J[70] += dqdci; // dwdot[CH]/d[H]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[2];
  J[379] += dqdci; // dwdot[H2]/d[CH2(S)]
  J[380] -= dqdci; // dwdot[H]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  J[394] += dqdci; // dwdot[CH]/d[CH2(S)]
  // d()/d[CH]
  dqdci = -k_r * sc[1];
  J[433] += dqdci; // dwdot[H2]/d[CH]
  J[434] -= dqdci; // dwdot[H]/d[CH]
  J[446] -= dqdci; // dwdot[CH2(S)]/d[CH]
  J[448] += dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT
  J[718] += dqdT; // dwdot[CH]/dT

  // reaction 41: CH2(S) + OH <=> CH2O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[14];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[20] * ge[2] * inv_ge[14])
             : exp(-g_RT[2] + g_RT[6] + g_RT[14] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[14]) + (h_RT[2] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[14] -= q; // CH2(S)
  wdot[20] += q; // CH2O
  // d()/d[H]
  dqdci = -k_r * sc[20];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[68] -= dqdci; // dwdot[CH2(S)]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[14];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[176] -= dqdci; // dwdot[CH2(S)]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[6];
  J[380] += dqdci; // dwdot[H]/d[CH2(S)]
  J[384] -= dqdci; // dwdot[OH]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  J[398] += dqdci; // dwdot[CH2O]/d[CH2(S)]
  // d()/d[CH2O]
  dqdci = -k_r * sc[2];
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[554] -= dqdci; // dwdot[CH2(S)]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 42: CH2(S) + CO2 <=> CH2O + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[10] * sc[14];
  k_f = 14000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[20];
  Kc =
    (kc_fast ? (ge[9] * inv_ge[10] * inv_ge[14] * ge[20])
             : exp(-g_RT[9] + g_RT[10] + g_RT[14] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[10] + h_RT[14]) + (h_RT[9] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[10] -= q; // CO2
  wdot[14] -= q; // CH2(S)
  wdot[20] += q; // CH2O
  // d()/d[CO]
  dqdci = -k_r * sc[20];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[253] -= dqdci; // dwdot[CO2]/d[CO]
  J[257] -= dqdci; // dwdot[CH2(S)]/d[CO]
  J[263] += dqdci; // dwdot[CH2O]/d[CO]
  // d()/d[CO2]
  dqdci = +k_f * sc[14];
  J[279] += dqdci; // dwdot[CO]/d[CO2]
  J[280] -= dqdci; // dwdot[CO2]/d[CO2]
  J[284] -= dqdci; // dwdot[CH2(S)]/d[CO2]
  J[290] += dqdci; // dwdot[CH2O]/d[CO2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[10];
  J[387] += dqdci; // dwdot[CO]/d[CH2(S)]
  J[388] -= dqdci; // dwdot[CO2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  J[398] += dqdci; // dwdot[CH2O]/d[CH2(S)]
  // d()/d[CH2O]
  dqdci = -k_r * sc[9];
  J[549] += dqdci; // dwdot[CO]/d[CH2O]
  J[550] -= dqdci; // dwdot[CO2]/d[CH2O]
  J[554] -= dqdci; // dwdot[CH2(S)]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[712] -= dqdT; // dwdot[CO2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 44: CH2 + O2 <=> HCO + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[13];
  k_f = 10600000 * exp(-(754.82499870708) * invT);
  dlnkfdT = (754.82499870708) * invT2;
  // reverse
  phi_r = sc[6] * sc[21];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[3] * ge[21] * inv_ge[13])
             : exp(g_RT[3] - g_RT[6] + g_RT[13] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[13]) + (h_RT[6] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[6] += q;  // OH
  wdot[13] -= q; // CH2
  wdot[21] += q; // HCO
  // d()/d[O2]
  dqdci = +k_f * sc[13];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[87] += dqdci;  // dwdot[OH]/d[O2]
  J[94] -= dqdci;  // dwdot[CH2]/d[O2]
  J[102] += dqdci; // dwdot[HCO]/d[O2]
  // d()/d[OH]
  dqdci = -k_r * sc[21];
  J[165] -= dqdci; // dwdot[O2]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[175] -= dqdci; // dwdot[CH2]/d[OH]
  J[183] += dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CH2]
  dqdci = +k_f * sc[3];
  J[354] -= dqdci; // dwdot[O2]/d[CH2]
  J[357] += dqdci; // dwdot[OH]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[372] += dqdci; // dwdot[HCO]/d[CH2]
  // d()/d[HCO]
  dqdci = -k_r * sc[6];
  J[570] -= dqdci; // dwdot[O2]/d[HCO]
  J[573] += dqdci; // dwdot[OH]/d[HCO]
  J[580] -= dqdci; // dwdot[CH2]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 45: CH2 + O2 => CO2 + 2 H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[13];
  k_f = 2640000 * exp(-(754.82499870708) * invT);
  dlnkfdT = (754.82499870708) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[3] -= q;     // O2
  wdot[10] += q;    // CO2
  wdot[13] -= q;    // CH2
  // d()/d[O2]
  dqdci = +k_f * sc[13];
  J[83] += 2 * dqdci; // dwdot[H]/d[O2]
  J[84] -= dqdci;     // dwdot[O2]/d[O2]
  J[91] += dqdci;     // dwdot[CO2]/d[O2]
  J[94] -= dqdci;     // dwdot[CH2]/d[O2]
  // d()/d[CH2]
  dqdci = +k_f * sc[3];
  J[353] += 2 * dqdci; // dwdot[H]/d[CH2]
  J[354] -= dqdci;     // dwdot[O2]/d[CH2]
  J[361] += dqdci;     // dwdot[CO2]/d[CH2]
  J[364] -= dqdci;     // dwdot[CH2]/d[CH2]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[705] -= dqdT;     // dwdot[O2]/dT
  J[712] += dqdT;     // dwdot[CO2]/dT
  J[715] -= dqdT;     // dwdot[CH2]/dT

  // reaction 46: CH2 + O => CO + 2 H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[13];
  k_f = 50000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[4] -= q;     // O
  wdot[9] += q;     // CO
  wdot[13] -= q;    // CH2
  // d()/d[O]
  dqdci = +k_f * sc[13];
  J[110] += 2 * dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci;     // dwdot[O]/d[O]
  J[117] += dqdci;     // dwdot[CO]/d[O]
  J[121] -= dqdci;     // dwdot[CH2]/d[O]
  // d()/d[CH2]
  dqdci = +k_f * sc[4];
  J[353] += 2 * dqdci; // dwdot[H]/d[CH2]
  J[355] -= dqdci;     // dwdot[O]/d[CH2]
  J[360] += dqdci;     // dwdot[CO]/d[CH2]
  J[364] -= dqdci;     // dwdot[CH2]/d[CH2]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[706] -= dqdT;     // dwdot[O]/dT
  J[711] += dqdT;     // dwdot[CO]/dT
  J[715] -= dqdT;     // dwdot[CH2]/dT

  // reaction 47: CH2 + H <=> CH + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[13];
  k_f = 283000000 * exp(-(369.864249366469) * invT);
  dlnkfdT = (369.864249366469) * invT2;
  // reverse
  phi_r = sc[1] * sc[16];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[2] * ge[16] * inv_ge[13])
             : exp(-g_RT[1] + g_RT[2] + g_RT[13] - g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[13]) + (h_RT[1] + h_RT[16]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[13] -= q; // CH2
  wdot[16] += q; // CH
  // d()/d[H2]
  dqdci = -k_r * sc[16];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[40] -= dqdci; // dwdot[CH2]/d[H2]
  J[43] += dqdci; // dwdot[CH]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[13];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[67] -= dqdci; // dwdot[CH2]/d[H]
  J[70] += dqdci; // dwdot[CH]/d[H]
  // d()/d[CH2]
  dqdci = +k_f * sc[2];
  J[352] += dqdci; // dwdot[H2]/d[CH2]
  J[353] -= dqdci; // dwdot[H]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[367] += dqdci; // dwdot[CH]/d[CH2]
  // d()/d[CH]
  dqdci = -k_r * sc[1];
  J[433] += dqdci; // dwdot[H2]/d[CH]
  J[434] -= dqdci; // dwdot[H]/d[CH]
  J[445] -= dqdci; // dwdot[CH2]/d[CH]
  J[448] += dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[718] += dqdT; // dwdot[CH]/dT

  // reaction 48: CH2 + OH <=> CH + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[13];
  k_f = 11.3 * exp(2 * logT - (1509.64999741416) * invT);
  dlnkfdT = 2 * invT + (1509.64999741416) * invT2;
  // reverse
  phi_r = sc[5] * sc[16];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[5] * ge[16] * inv_ge[13])
             : exp(-g_RT[5] + g_RT[6] + g_RT[13] - g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[13]) + (h_RT[5] + h_RT[16]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[13] -= q; // CH2
  wdot[16] += q; // CH
  // d()/d[H2O]
  dqdci = -k_r * sc[16];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[148] -= dqdci; // dwdot[CH2]/d[H2O]
  J[151] += dqdci; // dwdot[CH]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[13];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[175] -= dqdci; // dwdot[CH2]/d[OH]
  J[178] += dqdci; // dwdot[CH]/d[OH]
  // d()/d[CH2]
  dqdci = +k_f * sc[6];
  J[356] += dqdci; // dwdot[H2O]/d[CH2]
  J[357] -= dqdci; // dwdot[OH]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[367] += dqdci; // dwdot[CH]/d[CH2]
  // d()/d[CH]
  dqdci = -k_r * sc[5];
  J[437] += dqdci; // dwdot[H2O]/d[CH]
  J[438] -= dqdci; // dwdot[OH]/d[CH]
  J[445] -= dqdci; // dwdot[CH2]/d[CH]
  J[448] += dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[718] += dqdT; // dwdot[CH]/dT

  // reaction 49: CH + O2 <=> HCO + O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[16];
  k_f = 33000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[4] * sc[21];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[21] * ge[4] * inv_ge[16])
             : exp(g_RT[3] - g_RT[4] + g_RT[16] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[16]) + (h_RT[4] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[4] += q;  // O
  wdot[16] -= q; // CH
  wdot[21] += q; // HCO
  // d()/d[O2]
  dqdci = +k_f * sc[16];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[85] += dqdci;  // dwdot[O]/d[O2]
  J[97] -= dqdci;  // dwdot[CH]/d[O2]
  J[102] += dqdci; // dwdot[HCO]/d[O2]
  // d()/d[O]
  dqdci = -k_r * sc[21];
  J[111] -= dqdci; // dwdot[O2]/d[O]
  J[112] += dqdci; // dwdot[O]/d[O]
  J[124] -= dqdci; // dwdot[CH]/d[O]
  J[129] += dqdci; // dwdot[HCO]/d[O]
  // d()/d[CH]
  dqdci = +k_f * sc[3];
  J[435] -= dqdci; // dwdot[O2]/d[CH]
  J[436] += dqdci; // dwdot[O]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[453] += dqdci; // dwdot[HCO]/d[CH]
  // d()/d[HCO]
  dqdci = -k_r * sc[4];
  J[570] -= dqdci; // dwdot[O2]/d[HCO]
  J[571] += dqdci; // dwdot[O]/d[HCO]
  J[583] -= dqdci; // dwdot[CH]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[706] += dqdT; // dwdot[O]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 50: CH + O <=> CO + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[16];
  k_f = 57000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[9];
  Kc =
    (kc_fast ? (ge[9] * ge[2] * inv_ge[4] * inv_ge[16])
             : exp(-g_RT[2] + g_RT[4] - g_RT[9] + g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[16]) + (h_RT[2] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[4] -= q;  // O
  wdot[9] += q;  // CO
  wdot[16] -= q; // CH
  // d()/d[H]
  dqdci = -k_r * sc[9];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[63] += dqdci; // dwdot[CO]/d[H]
  J[70] -= dqdci; // dwdot[CH]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[16];
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[124] -= dqdci; // dwdot[CH]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[2];
  J[245] += dqdci; // dwdot[H]/d[CO]
  J[247] -= dqdci; // dwdot[O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[259] -= dqdci; // dwdot[CH]/d[CO]
  // d()/d[CH]
  dqdci = +k_f * sc[4];
  J[434] += dqdci; // dwdot[H]/d[CH]
  J[436] -= dqdci; // dwdot[O]/d[CH]
  J[441] += dqdci; // dwdot[CO]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[718] -= dqdT; // dwdot[CH]/dT

  // reaction 51: CH + H <=> C + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[16];
  k_f = 110000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[15];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[2] * ge[15] * inv_ge[16])
             : exp(-g_RT[1] + g_RT[2] - g_RT[15] + g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[16]) + (h_RT[1] + h_RT[15]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[15] += q; // C
  wdot[16] -= q; // CH
  // d()/d[H2]
  dqdci = -k_r * sc[15];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[42] += dqdci; // dwdot[C]/d[H2]
  J[43] -= dqdci; // dwdot[CH]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[16];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[69] += dqdci; // dwdot[C]/d[H]
  J[70] -= dqdci; // dwdot[CH]/d[H]
  // d()/d[C]
  dqdci = -k_r * sc[1];
  J[406] += dqdci; // dwdot[H2]/d[C]
  J[407] -= dqdci; // dwdot[H]/d[C]
  J[420] += dqdci; // dwdot[C]/d[C]
  J[421] -= dqdci; // dwdot[CH]/d[C]
  // d()/d[CH]
  dqdci = +k_f * sc[2];
  J[433] += dqdci; // dwdot[H2]/d[CH]
  J[434] -= dqdci; // dwdot[H]/d[CH]
  J[447] += dqdci; // dwdot[C]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[717] += dqdT; // dwdot[C]/dT
  J[718] -= dqdT; // dwdot[CH]/dT

  // reaction 52: CH + OH <=> H + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[16];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[21];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[21] * ge[2] * inv_ge[16])
             : exp(-g_RT[2] + g_RT[6] + g_RT[16] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[16]) + (h_RT[2] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[16] -= q; // CH
  wdot[21] += q; // HCO
  // d()/d[H]
  dqdci = -k_r * sc[21];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[70] -= dqdci; // dwdot[CH]/d[H]
  J[75] += dqdci; // dwdot[HCO]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[16];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[178] -= dqdci; // dwdot[CH]/d[OH]
  J[183] += dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CH]
  dqdci = +k_f * sc[6];
  J[434] += dqdci; // dwdot[H]/d[CH]
  J[438] -= dqdci; // dwdot[OH]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[453] += dqdci; // dwdot[HCO]/d[CH]
  // d()/d[HCO]
  dqdci = -k_r * sc[2];
  J[569] += dqdci; // dwdot[H]/d[HCO]
  J[573] -= dqdci; // dwdot[OH]/d[HCO]
  J[583] -= dqdci; // dwdot[CH]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 53: CH + H2O <=> CH2O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[5] * sc[16];
  k_f = 17130000 * exp(-(-379.928582682563) * invT);
  dlnkfdT = (-379.928582682563) * invT2;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc =
    (kc_fast ? (ge[20] * ge[2] * inv_ge[5] * inv_ge[16])
             : exp(-g_RT[2] + g_RT[5] + g_RT[16] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[5] + h_RT[16]) + (h_RT[2] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[5] -= q;  // H2O
  wdot[16] -= q; // CH
  wdot[20] += q; // CH2O
  // d()/d[H]
  dqdci = -k_r * sc[20];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[59] -= dqdci; // dwdot[H2O]/d[H]
  J[70] -= dqdci; // dwdot[CH]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[H2O]
  dqdci = +k_f * sc[16];
  J[137] += dqdci; // dwdot[H]/d[H2O]
  J[140] -= dqdci; // dwdot[H2O]/d[H2O]
  J[151] -= dqdci; // dwdot[CH]/d[H2O]
  J[155] += dqdci; // dwdot[CH2O]/d[H2O]
  // d()/d[CH]
  dqdci = +k_f * sc[5];
  J[434] += dqdci; // dwdot[H]/d[CH]
  J[437] -= dqdci; // dwdot[H2O]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[452] += dqdci; // dwdot[CH2O]/d[CH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[2];
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[545] -= dqdci; // dwdot[H2O]/d[CH2O]
  J[556] -= dqdci; // dwdot[CH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[707] -= dqdT; // dwdot[H2O]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 54: CH + CO2 <=> CO + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[10] * sc[16];
  k_f = 1700000 * exp(-(344.703416076233) * invT);
  dlnkfdT = (344.703416076233) * invT2;
  // reverse
  phi_r = sc[9] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * ge[9] * inv_ge[10] * inv_ge[16])
             : exp(-g_RT[9] + g_RT[10] + g_RT[16] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[10] + h_RT[16]) + (h_RT[9] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[10] -= q; // CO2
  wdot[16] -= q; // CH
  wdot[21] += q; // HCO
  // d()/d[CO]
  dqdci = -k_r * sc[21];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[253] -= dqdci; // dwdot[CO2]/d[CO]
  J[259] -= dqdci; // dwdot[CH]/d[CO]
  J[264] += dqdci; // dwdot[HCO]/d[CO]
  // d()/d[CO2]
  dqdci = +k_f * sc[16];
  J[279] += dqdci; // dwdot[CO]/d[CO2]
  J[280] -= dqdci; // dwdot[CO2]/d[CO2]
  J[286] -= dqdci; // dwdot[CH]/d[CO2]
  J[291] += dqdci; // dwdot[HCO]/d[CO2]
  // d()/d[CH]
  dqdci = +k_f * sc[10];
  J[441] += dqdci; // dwdot[CO]/d[CH]
  J[442] -= dqdci; // dwdot[CO2]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[453] += dqdci; // dwdot[HCO]/d[CH]
  // d()/d[HCO]
  dqdci = -k_r * sc[9];
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[577] -= dqdci; // dwdot[CO2]/d[HCO]
  J[583] -= dqdci; // dwdot[CH]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[712] -= dqdT; // dwdot[CO2]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 55: C + OH <=> CO + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[15];
  k_f = 50000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[9] * ge[2] * inv_ge[15])
             : exp(-g_RT[2] + g_RT[6] - g_RT[9] + g_RT[15]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[15]) + (h_RT[2] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[9] += q;  // CO
  wdot[15] -= q; // C
  // d()/d[H]
  dqdci = -k_r * sc[9];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[63] += dqdci; // dwdot[CO]/d[H]
  J[69] -= dqdci; // dwdot[C]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[15];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[171] += dqdci; // dwdot[CO]/d[OH]
  J[177] -= dqdci; // dwdot[C]/d[OH]
  // d()/d[CO]
  dqdci = -k_r * sc[2];
  J[245] += dqdci; // dwdot[H]/d[CO]
  J[249] -= dqdci; // dwdot[OH]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[258] -= dqdci; // dwdot[C]/d[CO]
  // d()/d[C]
  dqdci = +k_f * sc[6];
  J[407] += dqdci; // dwdot[H]/d[C]
  J[411] -= dqdci; // dwdot[OH]/d[C]
  J[414] += dqdci; // dwdot[CO]/d[C]
  J[420] -= dqdci; // dwdot[C]/d[C]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[717] -= dqdT; // dwdot[C]/dT

  // reaction 56: C + O2 <=> CO + O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[15];
  k_f = 50000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[4] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[9] * ge[4] * inv_ge[15])
             : exp(g_RT[3] - g_RT[4] - g_RT[9] + g_RT[15]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[15]) + (h_RT[4] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[4] += q;  // O
  wdot[9] += q;  // CO
  wdot[15] -= q; // C
  // d()/d[O2]
  dqdci = +k_f * sc[15];
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[85] += dqdci; // dwdot[O]/d[O2]
  J[90] += dqdci; // dwdot[CO]/d[O2]
  J[96] -= dqdci; // dwdot[C]/d[O2]
  // d()/d[O]
  dqdci = -k_r * sc[9];
  J[111] -= dqdci; // dwdot[O2]/d[O]
  J[112] += dqdci; // dwdot[O]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[123] -= dqdci; // dwdot[C]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[4];
  J[246] -= dqdci; // dwdot[O2]/d[CO]
  J[247] += dqdci; // dwdot[O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[258] -= dqdci; // dwdot[C]/d[CO]
  // d()/d[C]
  dqdci = +k_f * sc[3];
  J[408] -= dqdci; // dwdot[O2]/d[C]
  J[409] += dqdci; // dwdot[O]/d[C]
  J[414] += dqdci; // dwdot[CO]/d[C]
  J[420] -= dqdci; // dwdot[C]/d[C]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[706] += dqdT; // dwdot[O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[717] -= dqdT; // dwdot[C]/dT

  // reaction 57: CH3 + O2 <=> CH3O + O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[12];
  k_f = 7546000 * exp(-(14251.0959755897) * invT);
  dlnkfdT = (14251.0959755897) * invT2;
  // reverse
  phi_r = sc[4] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[18] * inv_ge[12] * ge[4])
             : exp(g_RT[3] - g_RT[4] + g_RT[12] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[12]) + (h_RT[4] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[4] += q;  // O
  wdot[12] -= q; // CH3
  wdot[18] += q; // CH3O
  // d()/d[O2]
  dqdci = +k_f * sc[12];
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[85] += dqdci; // dwdot[O]/d[O2]
  J[93] -= dqdci; // dwdot[CH3]/d[O2]
  J[99] += dqdci; // dwdot[CH3O]/d[O2]
  // d()/d[O]
  dqdci = -k_r * sc[18];
  J[111] -= dqdci; // dwdot[O2]/d[O]
  J[112] += dqdci; // dwdot[O]/d[O]
  J[120] -= dqdci; // dwdot[CH3]/d[O]
  J[126] += dqdci; // dwdot[CH3O]/d[O]
  // d()/d[CH3]
  dqdci = +k_f * sc[3];
  J[327] -= dqdci; // dwdot[O2]/d[CH3]
  J[328] += dqdci; // dwdot[O]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3O]
  dqdci = -k_r * sc[4];
  J[489] -= dqdci; // dwdot[O2]/d[CH3O]
  J[490] += dqdci; // dwdot[O]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[706] += dqdT; // dwdot[O]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 58: CH3 + O2 <=> CH2O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[12];
  k_f = 2.641e-06 * exp(3.283 * logT - (4078.57107634725) * invT);
  dlnkfdT = 3.283 * invT + (4078.57107634725) * invT2;
  // reverse
  phi_r = sc[6] * sc[20];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[3] * inv_ge[12] * ge[20])
             : exp(g_RT[3] - g_RT[6] + g_RT[12] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[12]) + (h_RT[6] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[6] += q;  // OH
  wdot[12] -= q; // CH3
  wdot[20] += q; // CH2O
  // d()/d[O2]
  dqdci = +k_f * sc[12];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[87] += dqdci;  // dwdot[OH]/d[O2]
  J[93] -= dqdci;  // dwdot[CH3]/d[O2]
  J[101] += dqdci; // dwdot[CH2O]/d[O2]
  // d()/d[OH]
  dqdci = -k_r * sc[20];
  J[165] -= dqdci; // dwdot[O2]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[3];
  J[327] -= dqdci; // dwdot[O2]/d[CH3]
  J[330] += dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH2O]
  dqdci = -k_r * sc[6];
  J[543] -= dqdci; // dwdot[O2]/d[CH2O]
  J[546] += dqdci; // dwdot[OH]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 59: CH3 + O <=> CH2O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[12];
  k_f = 55400000 * exp(0.05 * logT - (-68.4374665494419) * invT);
  dlnkfdT = 0.05 * invT + (-68.4374665494419) * invT2;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[12] * ge[2] * ge[20] * inv_ge[4])
             : exp(-g_RT[2] + g_RT[4] + g_RT[12] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[12]) + (h_RT[2] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[4] -= q;  // O
  wdot[12] -= q; // CH3
  wdot[20] += q; // CH2O
  // d()/d[H]
  dqdci = -k_r * sc[20];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[66] -= dqdci; // dwdot[CH3]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[12];
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[120] -= dqdci; // dwdot[CH3]/d[O]
  J[128] += dqdci; // dwdot[CH2O]/d[O]
  // d()/d[CH3]
  dqdci = +k_f * sc[4];
  J[326] += dqdci; // dwdot[H]/d[CH3]
  J[328] -= dqdci; // dwdot[O]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH2O]
  dqdci = -k_r * sc[2];
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[544] -= dqdci; // dwdot[O]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 60: CH3 + OH <=> CH2(S) + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 4.788e+17 * exp(-3.155 * logT - (3524.02631063045) * invT);
  dlnkfdT = -3.155 * invT + (3524.02631063045) * invT2;
  // reverse
  phi_r = sc[5] * sc[14];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[12] * ge[14] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[12] - g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[5] + h_RT[14]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[14] += q; // CH2(S)
  // d()/d[H2O]
  dqdci = -k_r * sc[14];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[147] -= dqdci; // dwdot[CH3]/d[H2O]
  J[149] += dqdci; // dwdot[CH2(S)]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[176] += dqdci; // dwdot[CH2(S)]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[329] += dqdci; // dwdot[H2O]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[338] += dqdci; // dwdot[CH2(S)]/d[CH3]
  // d()/d[CH2(S)]
  dqdci = -k_r * sc[5];
  J[383] += dqdci; // dwdot[H2O]/d[CH2(S)]
  J[384] -= dqdci; // dwdot[OH]/d[CH2(S)]
  J[390] -= dqdci; // dwdot[CH3]/d[CH2(S)]
  J[392] += dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[716] += dqdT; // dwdot[CH2(S)]/dT

  // reaction 61: CH3 + OH <=> CH2O + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 5374 * exp(0.287 * logT - (140.900666425322) * invT);
  dlnkfdT = 0.287 * invT + (140.900666425322) * invT2;
  // reverse
  phi_r = sc[1] * sc[20];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[6] * inv_ge[12] * ge[20])
             : exp(-g_RT[1] + g_RT[6] + g_RT[12] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[1] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[20] += q; // CH2O
  // d()/d[H2]
  dqdci = -k_r * sc[20];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[33] -= dqdci; // dwdot[OH]/d[H2]
  J[39] -= dqdci; // dwdot[CH3]/d[H2]
  J[47] += dqdci; // dwdot[CH2O]/d[H2]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[163] += dqdci; // dwdot[H2]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[325] += dqdci; // dwdot[H2]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH2O]
  dqdci = -k_r * sc[1];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 62: CH3 + OH <=> CH2OH + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 15250000 * exp(0.134 * logT - (2838.64521180442) * invT);
  dlnkfdT = 0.134 * invT + (2838.64521180442) * invT2;
  // reverse
  phi_r = sc[2] * sc[19];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[19] * ge[2] * inv_ge[12])
             : exp(-g_RT[2] + g_RT[6] + g_RT[12] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[2] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[19] += q; // CH2OH
  // d()/d[H]
  dqdci = -k_r * sc[19];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[66] -= dqdci; // dwdot[CH3]/d[H]
  J[73] += dqdci; // dwdot[CH2OH]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[181] += dqdci; // dwdot[CH2OH]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[326] += dqdci; // dwdot[H]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[343] += dqdci; // dwdot[CH2OH]/d[CH3]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[2];
  J[515] += dqdci; // dwdot[H]/d[CH2OH]
  J[519] -= dqdci; // dwdot[OH]/d[CH2OH]
  J[525] -= dqdci; // dwdot[CH3]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 63: CH3 + OH <=> CH3O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 1798 * exp(0.965 * logT - (6068.79298960492) * invT);
  dlnkfdT = 0.965 * invT + (6068.79298960492) * invT2;
  // reverse
  phi_r = sc[2] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[18] * inv_ge[12] * ge[2])
             : exp(-g_RT[2] + g_RT[6] + g_RT[12] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[2] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[18] += q; // CH3O
  // d()/d[H]
  dqdci = -k_r * sc[18];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[66] -= dqdci; // dwdot[CH3]/d[H]
  J[72] += dqdci; // dwdot[CH3O]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[180] += dqdci; // dwdot[CH3O]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[326] += dqdci; // dwdot[H]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3O]
  dqdci = -k_r * sc[2];
  J[488] += dqdci; // dwdot[H]/d[CH3O]
  J[492] -= dqdci; // dwdot[OH]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 64: CH3 + OH <=> CH2 + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 0.04293 * exp(2.568 * logT - (2011.75958655411) * invT);
  dlnkfdT = 2.568 * invT + (2011.75958655411) * invT2;
  // reverse
  phi_r = sc[5] * sc[13];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[12] * ge[13] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[12] - g_RT[13]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[5] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[13] += q; // CH2
  // d()/d[H2O]
  dqdci = -k_r * sc[13];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[147] -= dqdci; // dwdot[CH3]/d[H2O]
  J[148] += dqdci; // dwdot[CH2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[175] += dqdci; // dwdot[CH2]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[329] += dqdci; // dwdot[H2O]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[337] += dqdci; // dwdot[CH2]/d[CH3]
  // d()/d[CH2]
  dqdci = -k_r * sc[5];
  J[356] += dqdci; // dwdot[H2O]/d[CH2]
  J[357] -= dqdci; // dwdot[OH]/d[CH2]
  J[363] -= dqdci; // dwdot[CH3]/d[CH2]
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[715] += dqdT; // dwdot[CH2]/dT

  // reaction 65: CH3 + HO2 <=> CH3O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[12];
  k_f = 1000000 * exp(0.269 * logT - (-345.961457740745) * invT);
  dlnkfdT = 0.269 * invT + (-345.961457740745) * invT2;
  // reverse
  phi_r = sc[6] * sc[18];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[8] * inv_ge[12] * ge[18])
             : exp(-g_RT[6] + g_RT[8] + g_RT[12] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[12]) + (h_RT[6] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[6] += q;  // OH
  wdot[8] -= q;  // HO2
  wdot[12] -= q; // CH3
  wdot[18] += q; // CH3O
  // d()/d[OH]
  dqdci = -k_r * sc[18];
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[180] += dqdci; // dwdot[CH3O]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[12];
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[228] -= dqdci; // dwdot[CH3]/d[HO2]
  J[234] += dqdci; // dwdot[CH3O]/d[HO2]
  // d()/d[CH3]
  dqdci = +k_f * sc[8];
  J[330] += dqdci; // dwdot[OH]/d[CH3]
  J[332] -= dqdci; // dwdot[HO2]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3O]
  dqdci = -k_r * sc[6];
  J[492] += dqdci; // dwdot[OH]/d[CH3O]
  J[494] -= dqdci; // dwdot[HO2]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 68: CH3OH + H <=> CH3O + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[17];
  k_f = 0.199 * exp(2.56 * logT - (5183.13165778861) * invT);
  dlnkfdT = 2.56 * invT + (5183.13165778861) * invT2;
  // reverse
  phi_r = sc[1] * sc[18];
  Kc =
    (kc_fast ? (ge[1] * ge[18] * inv_ge[17] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[17]) + (h_RT[1] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[H2]
  dqdci = -k_r * sc[18];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[44] -= dqdci; // dwdot[CH3OH]/d[H2]
  J[45] += dqdci; // dwdot[CH3O]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[17];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[71] -= dqdci; // dwdot[CH3OH]/d[H]
  J[72] += dqdci; // dwdot[CH3O]/d[H]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[2];
  J[460] += dqdci; // dwdot[H2]/d[CH3OH]
  J[461] -= dqdci; // dwdot[H]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[1];
  J[487] += dqdci; // dwdot[H2]/d[CH3O]
  J[488] -= dqdci; // dwdot[H]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 69: CH3OH + H <=> CH2OH + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[17];
  k_f = 0.307 * exp(2.55 * logT - (2737.49866197768) * invT);
  dlnkfdT = 2.55 * invT + (2737.49866197768) * invT2;
  // reverse
  phi_r = sc[1] * sc[19];
  Kc =
    (kc_fast ? (ge[1] * ge[19] * inv_ge[17] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[17]) + (h_RT[1] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[H2]
  dqdci = -k_r * sc[19];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[44] -= dqdci; // dwdot[CH3OH]/d[H2]
  J[46] += dqdci; // dwdot[CH2OH]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[17];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[71] -= dqdci; // dwdot[CH3OH]/d[H]
  J[73] += dqdci; // dwdot[CH2OH]/d[H]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[2];
  J[460] += dqdci; // dwdot[H2]/d[CH3OH]
  J[461] -= dqdci; // dwdot[H]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[1];
  J[514] += dqdci; // dwdot[H2]/d[CH2OH]
  J[515] -= dqdci; // dwdot[H]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 70: CH3OH + O <=> CH3O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[17];
  k_f = 0.0388 * exp(2.5 * logT - (1549.90733067854) * invT);
  dlnkfdT = 2.5 * invT + (1549.90733067854) * invT2;
  // reverse
  phi_r = sc[6] * sc[18];
  Kc =
    (kc_fast ? (ge[6] * ge[18] * inv_ge[17] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[17]) + (h_RT[6] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[O]
  dqdci = +k_f * sc[17];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[125] -= dqdci; // dwdot[CH3OH]/d[O]
  J[126] += dqdci; // dwdot[CH3O]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[18];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
  J[180] += dqdci; // dwdot[CH3O]/d[OH]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[4];
  J[463] -= dqdci; // dwdot[O]/d[CH3OH]
  J[465] += dqdci; // dwdot[OH]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[6];
  J[490] -= dqdci; // dwdot[O]/d[CH3O]
  J[492] += dqdci; // dwdot[OH]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 71: CH3OH + O <=> CH2OH + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[17];
  k_f = 0.388 * exp(2.5 * logT - (1549.90733067854) * invT);
  dlnkfdT = 2.5 * invT + (1549.90733067854) * invT2;
  // reverse
  phi_r = sc[6] * sc[19];
  Kc =
    (kc_fast ? (ge[6] * ge[19] * inv_ge[17] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[17]) + (h_RT[6] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[O]
  dqdci = +k_f * sc[17];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[125] -= dqdci; // dwdot[CH3OH]/d[O]
  J[127] += dqdci; // dwdot[CH2OH]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[19];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
  J[181] += dqdci; // dwdot[CH2OH]/d[OH]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[4];
  J[463] -= dqdci; // dwdot[O]/d[CH3OH]
  J[465] += dqdci; // dwdot[OH]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[6];
  J[517] -= dqdci; // dwdot[O]/d[CH2OH]
  J[519] += dqdci; // dwdot[OH]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 72: CH3OH + OH <=> CH3O + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[17];
  k_f = 0.00015 * exp(3.03 * logT - (-383.954316009001) * invT);
  dlnkfdT = 3.03 * invT + (-383.954316009001) * invT2;
  // reverse
  phi_r = sc[5] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[18] * inv_ge[17] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[17]) + (h_RT[5] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[H2O]
  dqdci = -k_r * sc[18];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[152] -= dqdci; // dwdot[CH3OH]/d[H2O]
  J[153] += dqdci; // dwdot[CH3O]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[17];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
  J[180] += dqdci; // dwdot[CH3O]/d[OH]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[6];
  J[464] += dqdci; // dwdot[H2O]/d[CH3OH]
  J[465] -= dqdci; // dwdot[OH]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[5];
  J[491] += dqdci; // dwdot[H2O]/d[CH3O]
  J[492] -= dqdci; // dwdot[OH]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 73: CH3OH + OH <=> CH2OH + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[17];
  k_f = 0.0308 * exp(2.65 * logT - (-405.944884304667) * invT);
  dlnkfdT = 2.65 * invT + (-405.944884304667) * invT2;
  // reverse
  phi_r = sc[5] * sc[19];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[19] * inv_ge[17] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[17]) + (h_RT[5] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[H2O]
  dqdci = -k_r * sc[19];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[152] -= dqdci; // dwdot[CH3OH]/d[H2O]
  J[154] += dqdci; // dwdot[CH2OH]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[17];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
  J[181] += dqdci; // dwdot[CH2OH]/d[OH]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[6];
  J[464] += dqdci; // dwdot[H2O]/d[CH3OH]
  J[465] -= dqdci; // dwdot[OH]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[5];
  J[518] += dqdci; // dwdot[H2O]/d[CH2OH]
  J[519] -= dqdci; // dwdot[OH]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 74: CH3OH + O2 <=> CH3O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[17];
  k_f = 0.0358 * exp(2.27 * logT - (21519.8091048059) * invT);
  dlnkfdT = 2.27 * invT + (21519.8091048059) * invT2;
  // reverse
  phi_r = sc[8] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * ge[18] * inv_ge[17])
             : exp(g_RT[3] - g_RT[8] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[17]) + (h_RT[8] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[O2]
  dqdci = +k_f * sc[17];
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[89] += dqdci; // dwdot[HO2]/d[O2]
  J[98] -= dqdci; // dwdot[CH3OH]/d[O2]
  J[99] += dqdci; // dwdot[CH3O]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[18];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[233] -= dqdci; // dwdot[CH3OH]/d[HO2]
  J[234] += dqdci; // dwdot[CH3O]/d[HO2]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[3];
  J[462] -= dqdci; // dwdot[O2]/d[CH3OH]
  J[467] += dqdci; // dwdot[HO2]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[8];
  J[489] -= dqdci; // dwdot[O2]/d[CH3O]
  J[494] += dqdci; // dwdot[HO2]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 75: CH3OH + HO2 <=> CH3O + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[17];
  k_f = 1220000 * exp(-(10099.9107343668) * invT);
  dlnkfdT = (10099.9107343668) * invT2;
  // reverse
  phi_r = sc[7] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[18] * ge[7] * inv_ge[17])
             : exp(-g_RT[7] + g_RT[8] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[17]) + (h_RT[7] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[H2O2]
  dqdci = -k_r * sc[18];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[206] -= dqdci; // dwdot[CH3OH]/d[H2O2]
  J[207] += dqdci; // dwdot[CH3O]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[17];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[233] -= dqdci; // dwdot[CH3OH]/d[HO2]
  J[234] += dqdci; // dwdot[CH3O]/d[HO2]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[8];
  J[466] += dqdci; // dwdot[H2O2]/d[CH3OH]
  J[467] -= dqdci; // dwdot[HO2]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[7];
  J[493] += dqdci; // dwdot[H2O2]/d[CH3O]
  J[494] -= dqdci; // dwdot[HO2]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 76: CH3OH + HO2 <=> CH2OH + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[17];
  k_f = 32600000 * exp(-(9451.5160604774) * invT);
  dlnkfdT = (9451.5160604774) * invT2;
  // reverse
  phi_r = sc[7] * sc[19];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[19] * ge[7] * inv_ge[17])
             : exp(-g_RT[7] + g_RT[8] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[17]) + (h_RT[7] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[H2O2]
  dqdci = -k_r * sc[19];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[206] -= dqdci; // dwdot[CH3OH]/d[H2O2]
  J[208] += dqdci; // dwdot[CH2OH]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[17];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[233] -= dqdci; // dwdot[CH3OH]/d[HO2]
  J[235] += dqdci; // dwdot[CH2OH]/d[HO2]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[8];
  J[466] += dqdci; // dwdot[H2O2]/d[CH3OH]
  J[467] -= dqdci; // dwdot[HO2]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[7];
  J[520] += dqdci; // dwdot[H2O2]/d[CH2OH]
  J[521] -= dqdci; // dwdot[HO2]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 77: CH3 + CH3OH <=> CH2OH + CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[17];
  k_f = 2.13e-07 * exp(3.953 * logT - (3550.24389891888) * invT);
  dlnkfdT = 3.953 * invT + (3550.24389891888) * invT2;
  // reverse
  phi_r = sc[11] * sc[19];
  Kc =
    (kc_fast ? (ge[19] * inv_ge[12] * inv_ge[17] * ge[11])
             : exp(-g_RT[11] + g_RT[12] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[17]) + (h_RT[11] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[CH4]
  dqdci = -k_r * sc[19];
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[314] -= dqdci; // dwdot[CH3OH]/d[CH4]
  J[316] += dqdci; // dwdot[CH2OH]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[17];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[341] -= dqdci; // dwdot[CH3OH]/d[CH3]
  J[343] += dqdci; // dwdot[CH2OH]/d[CH3]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[12];
  J[470] += dqdci; // dwdot[CH4]/d[CH3OH]
  J[471] -= dqdci; // dwdot[CH3]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[11];
  J[524] += dqdci; // dwdot[CH4]/d[CH2OH]
  J[525] -= dqdci; // dwdot[CH3]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 78: CH3 + CH3OH <=> CH3O + CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[17];
  k_f = 0.00322 * exp(2.425 * logT - (4317.34738427159) * invT);
  dlnkfdT = 2.425 * invT + (4317.34738427159) * invT2;
  // reverse
  phi_r = sc[11] * sc[18];
  Kc =
    (kc_fast ? (ge[18] * inv_ge[12] * inv_ge[17] * ge[11])
             : exp(-g_RT[11] + g_RT[12] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[17]) + (h_RT[11] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[CH4]
  dqdci = -k_r * sc[18];
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[314] -= dqdci; // dwdot[CH3OH]/d[CH4]
  J[315] += dqdci; // dwdot[CH3O]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[17];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[341] -= dqdci; // dwdot[CH3OH]/d[CH3]
  J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[12];
  J[470] += dqdci; // dwdot[CH4]/d[CH3OH]
  J[471] -= dqdci; // dwdot[CH3]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[11];
  J[497] += dqdci; // dwdot[CH4]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 79: CH2OH + O2 <=> CH2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[19];
  k_f = 1510000000 * exp(-1 * logT);
  dlnkfdT = -1 * invT;
  // reverse
  phi_r = sc[8] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[19] * ge[20])
             : exp(g_RT[3] - g_RT[8] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[19]) + (h_RT[8] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[O2]
  dqdci = +k_f * sc[19];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[89] += dqdci;  // dwdot[HO2]/d[O2]
  J[100] -= dqdci; // dwdot[CH2OH]/d[O2]
  J[101] += dqdci; // dwdot[CH2O]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[20];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[235] -= dqdci; // dwdot[CH2OH]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[3];
  J[516] -= dqdci; // dwdot[O2]/d[CH2OH]
  J[521] += dqdci; // dwdot[HO2]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[8];
  J[543] -= dqdci; // dwdot[O2]/d[CH2O]
  J[548] += dqdci; // dwdot[HO2]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 80: CH2OH + O2 <=> CH2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[19];
  k_f = 241000000 * exp(-(2524.63801234228) * invT);
  dlnkfdT = (2524.63801234228) * invT2;
  // reverse
  phi_r = sc[8] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[19] * ge[20])
             : exp(g_RT[3] - g_RT[8] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[19]) + (h_RT[8] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[O2]
  dqdci = +k_f * sc[19];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[89] += dqdci;  // dwdot[HO2]/d[O2]
  J[100] -= dqdci; // dwdot[CH2OH]/d[O2]
  J[101] += dqdci; // dwdot[CH2O]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[20];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[235] -= dqdci; // dwdot[CH2OH]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[3];
  J[516] -= dqdci; // dwdot[O2]/d[CH2OH]
  J[521] += dqdci; // dwdot[HO2]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[8];
  J[543] -= dqdci; // dwdot[O2]/d[CH2O]
  J[548] += dqdci; // dwdot[HO2]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 81: CH2OH + H <=> CH2O + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[19];
  k_f = 6000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[20];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[19] * ge[20] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[19]) + (h_RT[1] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[H2]
  dqdci = -k_r * sc[20];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[46] -= dqdci; // dwdot[CH2OH]/d[H2]
  J[47] += dqdci; // dwdot[CH2O]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[19];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[73] -= dqdci; // dwdot[CH2OH]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[2];
  J[514] += dqdci; // dwdot[H2]/d[CH2OH]
  J[515] -= dqdci; // dwdot[H]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[1];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[542] -= dqdci; // dwdot[H]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 82: CH2OH + HO2 <=> CH2O + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[19];
  k_f = 12000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[7] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[20] * inv_ge[19] * ge[7])
             : exp(-g_RT[7] + g_RT[8] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[19]) + (h_RT[7] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[H2O2]
  dqdci = -k_r * sc[20];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[208] -= dqdci; // dwdot[CH2OH]/d[H2O2]
  J[209] += dqdci; // dwdot[CH2O]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[19];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[235] -= dqdci; // dwdot[CH2OH]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[8];
  J[520] += dqdci; // dwdot[H2O2]/d[CH2OH]
  J[521] -= dqdci; // dwdot[HO2]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[7];
  J[547] += dqdci; // dwdot[H2O2]/d[CH2O]
  J[548] -= dqdci; // dwdot[HO2]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 83: CH2OH + HCO <=> 2 CH2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[19] * sc[21];
  k_f = 180000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = (sc[20] * sc[20]);
  Kc =
    (kc_fast ? (inv_ge[21] * ge[20] * inv_ge[19] * ge[20])
             : exp(g_RT[19] - 2.000000 * g_RT[20] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[19] + h_RT[21]) + (2.000000 * h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[19] -= q;     // CH2OH
  wdot[20] += 2 * q; // CH2O
  wdot[21] -= q;     // HCO
  // d()/d[CH2OH]
  dqdci = +k_f * sc[21];
  J[532] -= dqdci;     // dwdot[CH2OH]/d[CH2OH]
  J[533] += 2 * dqdci; // dwdot[CH2O]/d[CH2OH]
  J[534] -= dqdci;     // dwdot[HCO]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * 2.000000 * sc[20];
  J[559] -= dqdci;     // dwdot[CH2OH]/d[CH2O]
  J[560] += 2 * dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] -= dqdci;     // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = +k_f * sc[19];
  J[586] -= dqdci;     // dwdot[CH2OH]/d[HCO]
  J[587] += 2 * dqdci; // dwdot[CH2O]/d[HCO]
  J[588] -= dqdci;     // dwdot[HCO]/d[HCO]
  // d()/dT
  J[721] -= dqdT;     // dwdot[CH2OH]/dT
  J[722] += 2 * dqdT; // dwdot[CH2O]/dT
  J[723] -= dqdT;     // dwdot[HCO]/dT

  // reaction 84: CH2OH + OH <=> CH2O + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[19];
  k_f = 24000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[19] * ge[20] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[19]) + (h_RT[5] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[H2O]
  dqdci = -k_r * sc[20];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[154] -= dqdci; // dwdot[CH2OH]/d[H2O]
  J[155] += dqdci; // dwdot[CH2O]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[19];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[181] -= dqdci; // dwdot[CH2OH]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[6];
  J[518] += dqdci; // dwdot[H2O]/d[CH2OH]
  J[519] -= dqdci; // dwdot[OH]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[5];
  J[545] += dqdci; // dwdot[H2O]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 85: CH2OH + O <=> CH2O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[19];
  k_f = 42000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[6] * sc[20];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[19] * ge[20] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[19]) + (h_RT[6] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[O]
  dqdci = +k_f * sc[19];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[127] -= dqdci; // dwdot[CH2OH]/d[O]
  J[128] += dqdci; // dwdot[CH2O]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[20];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[181] -= dqdci; // dwdot[CH2OH]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[4];
  J[517] -= dqdci; // dwdot[O]/d[CH2OH]
  J[519] += dqdci; // dwdot[OH]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[6];
  J[544] -= dqdci; // dwdot[O]/d[CH2O]
  J[546] += dqdci; // dwdot[OH]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 86: CH3O + O2 <=> CH2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[18];
  k_f = 4.38e-25 * exp(9.5 * logT - (-2768.19487859176) * invT);
  dlnkfdT = 9.5 * invT + (-2768.19487859176) * invT2;
  // reverse
  phi_r = sc[8] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[18] * ge[20])
             : exp(g_RT[3] - g_RT[8] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[18]) + (h_RT[8] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[O2]
  dqdci = +k_f * sc[18];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[89] += dqdci;  // dwdot[HO2]/d[O2]
  J[99] -= dqdci;  // dwdot[CH3O]/d[O2]
  J[101] += dqdci; // dwdot[CH2O]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[20];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[234] -= dqdci; // dwdot[CH3O]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH3O]
  dqdci = +k_f * sc[3];
  J[489] -= dqdci; // dwdot[O2]/d[CH3O]
  J[494] += dqdci; // dwdot[HO2]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[8];
  J[543] -= dqdci; // dwdot[O2]/d[CH2O]
  J[548] += dqdci; // dwdot[HO2]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 87: CH3O + H <=> CH2O + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[18];
  k_f = 20000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[20];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[18] * ge[20] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[18]) + (h_RT[1] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[H2]
  dqdci = -k_r * sc[20];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[45] -= dqdci; // dwdot[CH3O]/d[H2]
  J[47] += dqdci; // dwdot[CH2O]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[18];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[72] -= dqdci; // dwdot[CH3O]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[CH3O]
  dqdci = +k_f * sc[2];
  J[487] += dqdci; // dwdot[H2]/d[CH3O]
  J[488] -= dqdci; // dwdot[H]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[1];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[542] -= dqdci; // dwdot[H]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 88: CH3O + HO2 <=> CH2O + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[18];
  k_f = 301000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[7] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[20] * inv_ge[18] * ge[7])
             : exp(-g_RT[7] + g_RT[8] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[18]) + (h_RT[7] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[H2O2]
  dqdci = -k_r * sc[20];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[207] -= dqdci; // dwdot[CH3O]/d[H2O2]
  J[209] += dqdci; // dwdot[CH2O]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[18];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[234] -= dqdci; // dwdot[CH3O]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH3O]
  dqdci = +k_f * sc[8];
  J[493] += dqdci; // dwdot[H2O2]/d[CH3O]
  J[494] -= dqdci; // dwdot[HO2]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[7];
  J[547] += dqdci; // dwdot[H2O2]/d[CH2O]
  J[548] -= dqdci; // dwdot[HO2]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 89: CH3 + CH3O <=> CH2O + CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[18];
  k_f = 12000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[11] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[18] * ge[11] * inv_ge[12] * ge[20])
             : exp(-g_RT[11] + g_RT[12] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[18]) + (h_RT[11] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[CH4]
  dqdci = -k_r * sc[20];
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[315] -= dqdci; // dwdot[CH3O]/d[CH4]
  J[317] += dqdci; // dwdot[CH2O]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[18];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[342] -= dqdci; // dwdot[CH3O]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH3O]
  dqdci = +k_f * sc[12];
  J[497] += dqdci; // dwdot[CH4]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[11];
  J[551] += dqdci; // dwdot[CH4]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 90: 2 CH3O <=> CH2O + CH3OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[18] * sc[18]);
  k_f = 60300000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[17] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[18] * ge[20] * inv_ge[18] * ge[17])
             : exp(-g_RT[17] + 2.000000 * g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[18]) + (h_RT[17] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[17] += q;     // CH3OH
  wdot[18] -= 2 * q; // CH3O
  wdot[20] += q;     // CH2O
  // d()/d[CH3OH]
  dqdci = -k_r * sc[20];
  J[476] += dqdci;      // dwdot[CH3OH]/d[CH3OH]
  J[477] += -2 * dqdci; // dwdot[CH3O]/d[CH3OH]
  J[479] += dqdci;      // dwdot[CH2O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = +k_f * 2.000000 * sc[18];
  J[503] += dqdci;      // dwdot[CH3OH]/d[CH3O]
  J[504] += -2 * dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci;      // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[17];
  J[557] += dqdci;      // dwdot[CH3OH]/d[CH2O]
  J[558] += -2 * dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci;      // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[719] += dqdT;      // dwdot[CH3OH]/dT
  J[720] += -2 * dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT;      // dwdot[CH2O]/dT

  // reaction 91: CH2O + O <=> HCO + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[20];
  k_f = 6260 * exp(1.15 * logT - (1137.26966471867) * invT);
  dlnkfdT = 1.15 * invT + (1137.26966471867) * invT2;
  // reverse
  phi_r = sc[6] * sc[21];
  Kc =
    (kc_fast ? (ge[6] * ge[21] * inv_ge[20] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[20]) + (h_RT[6] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[O]
  dqdci = +k_f * sc[20];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[128] -= dqdci; // dwdot[CH2O]/d[O]
  J[129] += dqdci; // dwdot[HCO]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[21];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[182] -= dqdci; // dwdot[CH2O]/d[OH]
  J[183] += dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CH2O]
  dqdci = +k_f * sc[4];
  J[544] -= dqdci; // dwdot[O]/d[CH2O]
  J[546] += dqdci; // dwdot[OH]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[6];
  J[571] -= dqdci; // dwdot[O]/d[HCO]
  J[573] += dqdci; // dwdot[OH]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 92: CH2O + H <=> H2 + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[20];
  k_f = 57.4 * exp(1.9 * logT - (1378.81366430493) * invT);
  dlnkfdT = 1.9 * invT + (1378.81366430493) * invT2;
  // reverse
  phi_r = sc[1] * sc[21];
  Kc =
    (kc_fast ? (ge[1] * ge[21] * inv_ge[20] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[20]) + (h_RT[1] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[H2]
  dqdci = -k_r * sc[21];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[47] -= dqdci; // dwdot[CH2O]/d[H2]
  J[48] += dqdci; // dwdot[HCO]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[20];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[74] -= dqdci; // dwdot[CH2O]/d[H]
  J[75] += dqdci; // dwdot[HCO]/d[H]
  // d()/d[CH2O]
  dqdci = +k_f * sc[2];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[542] -= dqdci; // dwdot[H]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[1];
  J[568] += dqdci; // dwdot[H2]/d[HCO]
  J[569] -= dqdci; // dwdot[H]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 93: CH2O + OH <=> H2O + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[20];
  k_f = 78.2 * exp(1.63 * logT - (-530.893582423979) * invT);
  dlnkfdT = 1.63 * invT + (-530.893582423979) * invT2;
  // reverse
  phi_r = sc[5] * sc[21];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[21] * inv_ge[20] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[20]) + (h_RT[5] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[H2O]
  dqdci = -k_r * sc[21];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[155] -= dqdci; // dwdot[CH2O]/d[H2O]
  J[156] += dqdci; // dwdot[HCO]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[20];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[182] -= dqdci; // dwdot[CH2O]/d[OH]
  J[183] += dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CH2O]
  dqdci = +k_f * sc[6];
  J[545] += dqdci; // dwdot[H2O]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[5];
  J[572] += dqdci; // dwdot[H2O]/d[HCO]
  J[573] -= dqdci; // dwdot[OH]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 94: CH2O + HO2 <=> H2O2 + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[20];
  k_f = 0.0188 * exp(2.7 * logT - (5797.05599007037) * invT);
  dlnkfdT = 2.7 * invT + (5797.05599007037) * invT2;
  // reverse
  phi_r = sc[7] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * inv_ge[8] * ge[7] * inv_ge[20])
             : exp(-g_RT[7] + g_RT[8] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[20]) + (h_RT[7] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[H2O2]
  dqdci = -k_r * sc[21];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[209] -= dqdci; // dwdot[CH2O]/d[H2O2]
  J[210] += dqdci; // dwdot[HCO]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[20];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[236] -= dqdci; // dwdot[CH2O]/d[HO2]
  J[237] += dqdci; // dwdot[HCO]/d[HO2]
  // d()/d[CH2O]
  dqdci = +k_f * sc[8];
  J[547] += dqdci; // dwdot[H2O2]/d[CH2O]
  J[548] -= dqdci; // dwdot[HO2]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[7];
  J[574] += dqdci; // dwdot[H2O2]/d[HCO]
  J[575] -= dqdci; // dwdot[HO2]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 95: CH2O + CH3 <=> CH4 + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[20];
  k_f = 3.83e-05 * exp(3.36 * logT - (2169.87026294995) * invT);
  dlnkfdT = 3.36 * invT + (2169.87026294995) * invT2;
  // reverse
  phi_r = sc[11] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * inv_ge[12] * inv_ge[20] * ge[11])
             : exp(-g_RT[11] + g_RT[12] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[20]) + (h_RT[11] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[CH4]
  dqdci = -k_r * sc[21];
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[317] -= dqdci; // dwdot[CH2O]/d[CH4]
  J[318] += dqdci; // dwdot[HCO]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[20];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[344] -= dqdci; // dwdot[CH2O]/d[CH3]
  J[345] += dqdci; // dwdot[HCO]/d[CH3]
  // d()/d[CH2O]
  dqdci = +k_f * sc[12];
  J[551] += dqdci; // dwdot[CH4]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[11];
  J[578] += dqdci; // dwdot[CH4]/d[HCO]
  J[579] -= dqdci; // dwdot[CH3]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 96: CH2O + CH3O <=> CH3OH + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[18] * sc[20];
  k_f = 662000 * exp(-(1154.37903135603) * invT);
  dlnkfdT = (1154.37903135603) * invT2;
  // reverse
  phi_r = sc[17] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * inv_ge[18] * inv_ge[20] * ge[17])
             : exp(-g_RT[17] + g_RT[18] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[18] + h_RT[20]) + (h_RT[17] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[17] += q; // CH3OH
  wdot[18] -= q; // CH3O
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[CH3OH]
  dqdci = -k_r * sc[21];
  J[476] += dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] -= dqdci; // dwdot[CH3O]/d[CH3OH]
  J[479] -= dqdci; // dwdot[CH2O]/d[CH3OH]
  J[480] += dqdci; // dwdot[HCO]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = +k_f * sc[20];
  J[503] += dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] -= dqdci; // dwdot[CH2O]/d[CH3O]
  J[507] += dqdci; // dwdot[HCO]/d[CH3O]
  // d()/d[CH2O]
  dqdci = +k_f * sc[18];
  J[557] += dqdci; // dwdot[CH3OH]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[17];
  J[584] += dqdci; // dwdot[CH3OH]/d[HCO]
  J[585] -= dqdci; // dwdot[CH3O]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[719] += dqdT; // dwdot[CH3OH]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 98: HCO + O2 <=> CO + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[21];
  k_f = 7580000 * exp(-(206.318832979935) * invT);
  dlnkfdT = (206.318832979935) * invT2;
  // reverse
  phi_r = sc[8] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[21] * ge[9])
             : exp(g_RT[3] - g_RT[8] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[21]) + (h_RT[8] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[O2]
  dqdci = +k_f * sc[21];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[89] += dqdci;  // dwdot[HO2]/d[O2]
  J[90] += dqdci;  // dwdot[CO]/d[O2]
  J[102] -= dqdci; // dwdot[HCO]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[9];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[225] += dqdci; // dwdot[CO]/d[HO2]
  J[237] -= dqdci; // dwdot[HCO]/d[HO2]
  // d()/d[CO]
  dqdci = -k_r * sc[8];
  J[246] -= dqdci; // dwdot[O2]/d[CO]
  J[251] += dqdci; // dwdot[HO2]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[3];
  J[570] -= dqdci; // dwdot[O2]/d[HCO]
  J[575] += dqdci; // dwdot[HO2]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 99: HCO + O <=> CO + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[21];
  k_f = 30200000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[6] * sc[9];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[21] * ge[9] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[21]) + (h_RT[6] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[O]
  dqdci = +k_f * sc[21];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[129] -= dqdci; // dwdot[HCO]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[9];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[171] += dqdci; // dwdot[CO]/d[OH]
  J[183] -= dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CO]
  dqdci = -k_r * sc[6];
  J[247] -= dqdci; // dwdot[O]/d[CO]
  J[249] += dqdci; // dwdot[OH]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[4];
  J[571] -= dqdci; // dwdot[O]/d[HCO]
  J[573] += dqdci; // dwdot[OH]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 100: H + HCO <=> CO + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[21];
  k_f = 73400000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[21] * ge[9] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[21]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[36] += dqdci; // dwdot[CO]/d[H2]
  J[48] -= dqdci; // dwdot[HCO]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[21];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[63] += dqdci; // dwdot[CO]/d[H]
  J[75] -= dqdci; // dwdot[HCO]/d[H]
  // d()/d[CO]
  dqdci = -k_r * sc[1];
  J[244] += dqdci; // dwdot[H2]/d[CO]
  J[245] -= dqdci; // dwdot[H]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[2];
  J[568] += dqdci; // dwdot[H2]/d[HCO]
  J[569] -= dqdci; // dwdot[H]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 101: HCO + OH <=> CO + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[21];
  k_f = 30110000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[21] * ge[9] * ge[5])
             : exp(-g_RT[5] + g_RT[6] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[21]) + (h_RT[5] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[H2O]
  dqdci = -k_r * sc[9];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[144] += dqdci; // dwdot[CO]/d[H2O]
  J[156] -= dqdci; // dwdot[HCO]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[21];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[171] += dqdci; // dwdot[CO]/d[OH]
  J[183] -= dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CO]
  dqdci = -k_r * sc[5];
  J[248] += dqdci; // dwdot[H2O]/d[CO]
  J[249] -= dqdci; // dwdot[OH]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[6];
  J[572] += dqdci; // dwdot[H2O]/d[HCO]
  J[573] -= dqdci; // dwdot[OH]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 102: CH3 + HCO <=> CH4 + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[21];
  k_f = 26500000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[11];
  Kc =
    (kc_fast ? (inv_ge[21] * ge[11] * inv_ge[12] * ge[9])
             : exp(-g_RT[9] - g_RT[11] + g_RT[12] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[21]) + (h_RT[9] + h_RT[11]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[21] -= q; // HCO
  // d()/d[CO]
  dqdci = -k_r * sc[11];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[254] += dqdci; // dwdot[CH4]/d[CO]
  J[255] -= dqdci; // dwdot[CH3]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[CH4]
  dqdci = -k_r * sc[9];
  J[306] += dqdci; // dwdot[CO]/d[CH4]
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[318] -= dqdci; // dwdot[HCO]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[21];
  J[333] += dqdci; // dwdot[CO]/d[CH3]
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[345] -= dqdci; // dwdot[HCO]/d[CH3]
  // d()/d[HCO]
  dqdci = +k_f * sc[12];
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[578] += dqdci; // dwdot[CH4]/d[HCO]
  J[579] -= dqdci; // dwdot[CH3]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 103: 2 HCO <=> CH2O + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[21] * sc[21]);
  k_f = 18000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[21] * inv_ge[21] * ge[9] * ge[20])
             : exp(-g_RT[9] - g_RT[20] + 2.000000 * g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[21]) + (h_RT[9] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;      // CO
  wdot[20] += q;     // CH2O
  wdot[21] -= 2 * q; // HCO
  // d()/d[CO]
  dqdci = -k_r * sc[20];
  J[252] += dqdci;      // dwdot[CO]/d[CO]
  J[263] += dqdci;      // dwdot[CH2O]/d[CO]
  J[264] += -2 * dqdci; // dwdot[HCO]/d[CO]
  // d()/d[CH2O]
  dqdci = -k_r * sc[9];
  J[549] += dqdci;      // dwdot[CO]/d[CH2O]
  J[560] += dqdci;      // dwdot[CH2O]/d[CH2O]
  J[561] += -2 * dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = +k_f * 2.000000 * sc[21];
  J[576] += dqdci;      // dwdot[CO]/d[HCO]
  J[587] += dqdci;      // dwdot[CH2O]/d[HCO]
  J[588] += -2 * dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[711] += dqdT;      // dwdot[CO]/dT
  J[722] += dqdT;      // dwdot[CH2O]/dT
  J[723] += -2 * dqdT; // dwdot[HCO]/dT

  // reaction 104: HCO + O <=> CO2 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[21];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[10];
  Kc =
    (kc_fast ? (inv_ge[21] * ge[2] * inv_ge[4] * ge[10])
             : exp(-g_RT[2] + g_RT[4] - g_RT[10] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[21]) + (h_RT[2] + h_RT[10]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[4] -= q;  // O
  wdot[10] += q; // CO2
  wdot[21] -= q; // HCO
  // d()/d[H]
  dqdci = -k_r * sc[10];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[64] += dqdci; // dwdot[CO2]/d[H]
  J[75] -= dqdci; // dwdot[HCO]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[21];
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[118] += dqdci; // dwdot[CO2]/d[O]
  J[129] -= dqdci; // dwdot[HCO]/d[O]
  // d()/d[CO2]
  dqdci = -k_r * sc[2];
  J[272] += dqdci; // dwdot[H]/d[CO2]
  J[274] -= dqdci; // dwdot[O]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  J[291] -= dqdci; // dwdot[HCO]/d[CO2]
  // d()/d[HCO]
  dqdci = +k_f * sc[4];
  J[569] += dqdci; // dwdot[H]/d[HCO]
  J[571] -= dqdci; // dwdot[O]/d[HCO]
  J[577] += dqdci; // dwdot[CO2]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[712] += dqdT; // dwdot[CO2]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 105: HCO + HO2 => CO2 + H + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[21];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += q;  // H
  wdot[6] += q;  // OH
  wdot[8] -= q;  // HO2
  wdot[10] += q; // CO2
  wdot[21] -= q; // HCO
  // d()/d[HO2]
  dqdci = +k_f * sc[21];
  J[218] += dqdci; // dwdot[H]/d[HO2]
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[226] += dqdci; // dwdot[CO2]/d[HO2]
  J[237] -= dqdci; // dwdot[HCO]/d[HO2]
  // d()/d[HCO]
  dqdci = +k_f * sc[8];
  J[569] += dqdci; // dwdot[H]/d[HCO]
  J[573] += dqdci; // dwdot[OH]/d[HCO]
  J[575] -= dqdci; // dwdot[HO2]/d[HCO]
  J[577] += dqdci; // dwdot[CO2]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[712] += dqdT; // dwdot[CO2]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 109: CH + OH <=> C + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[16];
  k_f = 40 * exp(2 * logT - (1509.64999741416) * invT);
  dlnkfdT = 2 * invT + (1509.64999741416) * invT2;
  // reverse
  phi_r = sc[5] * sc[15];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[5] * ge[15] * inv_ge[16])
             : exp(-g_RT[5] + g_RT[6] - g_RT[15] + g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[16]) + (h_RT[5] + h_RT[15]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[15] += q; // C
  wdot[16] -= q; // CH
  // d()/d[H2O]
  dqdci = -k_r * sc[15];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[150] += dqdci; // dwdot[C]/d[H2O]
  J[151] -= dqdci; // dwdot[CH]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[16];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[177] += dqdci; // dwdot[C]/d[OH]
  J[178] -= dqdci; // dwdot[CH]/d[OH]
  // d()/d[C]
  dqdci = -k_r * sc[5];
  J[410] += dqdci; // dwdot[H2O]/d[C]
  J[411] -= dqdci; // dwdot[OH]/d[C]
  J[420] += dqdci; // dwdot[C]/d[C]
  J[421] -= dqdci; // dwdot[CH]/d[C]
  // d()/d[CH]
  dqdci = +k_f * sc[6];
  J[437] += dqdci; // dwdot[H2O]/d[CH]
  J[438] -= dqdci; // dwdot[OH]/d[CH]
  J[447] += dqdci; // dwdot[C]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[717] += dqdT; // dwdot[C]/dT
  J[718] -= dqdT; // dwdot[CH]/dT

  // reaction 110: CH2 + HCO <=> CH3 + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[13] * sc[21];
  k_f = 20000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[21] * ge[12] * ge[9] * inv_ge[13])
             : exp(-g_RT[9] - g_RT[12] + g_RT[13] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[13] + h_RT[21]) + (h_RT[9] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[12] += q; // CH3
  wdot[13] -= q; // CH2
  wdot[21] -= q; // HCO
  // d()/d[CO]
  dqdci = -k_r * sc[12];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[255] += dqdci; // dwdot[CH3]/d[CO]
  J[256] -= dqdci; // dwdot[CH2]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[CH3]
  dqdci = -k_r * sc[9];
  J[333] += dqdci; // dwdot[CO]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  J[337] -= dqdci; // dwdot[CH2]/d[CH3]
  J[345] -= dqdci; // dwdot[HCO]/d[CH3]
  // d()/d[CH2]
  dqdci = +k_f * sc[21];
  J[360] += dqdci; // dwdot[CO]/d[CH2]
  J[363] += dqdci; // dwdot[CH3]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[372] -= dqdci; // dwdot[HCO]/d[CH2]
  // d()/d[HCO]
  dqdci = +k_f * sc[13];
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[579] += dqdci; // dwdot[CH3]/d[HCO]
  J[580] -= dqdci; // dwdot[CH2]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 111: CH2 + O <=> CO + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[13];
  k_f = 50000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc =
    (kc_fast ? (ge[1] * ge[9] * inv_ge[4] * inv_ge[13])
             : exp(-g_RT[1] + g_RT[4] - g_RT[9] + g_RT[13]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[13]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[4] -= q;  // O
  wdot[9] += q;  // CO
  wdot[13] -= q; // CH2
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[31] -= dqdci; // dwdot[O]/d[H2]
  J[36] += dqdci; // dwdot[CO]/d[H2]
  J[40] -= dqdci; // dwdot[CH2]/d[H2]
  // d()/d[O]
  dqdci = +k_f * sc[13];
  J[109] += dqdci; // dwdot[H2]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[121] -= dqdci; // dwdot[CH2]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[1];
  J[244] += dqdci; // dwdot[H2]/d[CO]
  J[247] -= dqdci; // dwdot[O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[256] -= dqdci; // dwdot[CH2]/d[CO]
  // d()/d[CH2]
  dqdci = +k_f * sc[4];
  J[352] += dqdci; // dwdot[H2]/d[CH2]
  J[355] -= dqdci; // dwdot[O]/d[CH2]
  J[360] += dqdci; // dwdot[CO]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT

  // reaction 112: CH2 + OH <=> CH2O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[13];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[20] * ge[2] * inv_ge[13])
             : exp(-g_RT[2] + g_RT[6] + g_RT[13] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[13]) + (h_RT[2] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[13] -= q; // CH2
  wdot[20] += q; // CH2O
  // d()/d[H]
  dqdci = -k_r * sc[20];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[67] -= dqdci; // dwdot[CH2]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[13];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[175] -= dqdci; // dwdot[CH2]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2]
  dqdci = +k_f * sc[6];
  J[353] += dqdci; // dwdot[H]/d[CH2]
  J[357] -= dqdci; // dwdot[OH]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[371] += dqdci; // dwdot[CH2O]/d[CH2]
  // d()/d[CH2O]
  dqdci = -k_r * sc[2];
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[553] -= dqdci; // dwdot[CH2]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 113: CH2 + CO2 <=> CH2O + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[10] * sc[13];
  k_f = 110000 * exp(-(503.21666580472) * invT);
  dlnkfdT = (503.21666580472) * invT2;
  // reverse
  phi_r = sc[9] * sc[20];
  Kc =
    (kc_fast ? (ge[9] * inv_ge[10] * inv_ge[13] * ge[20])
             : exp(-g_RT[9] + g_RT[10] + g_RT[13] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[10] + h_RT[13]) + (h_RT[9] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[10] -= q; // CO2
  wdot[13] -= q; // CH2
  wdot[20] += q; // CH2O
  // d()/d[CO]
  dqdci = -k_r * sc[20];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[253] -= dqdci; // dwdot[CO2]/d[CO]
  J[256] -= dqdci; // dwdot[CH2]/d[CO]
  J[263] += dqdci; // dwdot[CH2O]/d[CO]
  // d()/d[CO2]
  dqdci = +k_f * sc[13];
  J[279] += dqdci; // dwdot[CO]/d[CO2]
  J[280] -= dqdci; // dwdot[CO2]/d[CO2]
  J[283] -= dqdci; // dwdot[CH2]/d[CO2]
  J[290] += dqdci; // dwdot[CH2O]/d[CO2]
  // d()/d[CH2]
  dqdci = +k_f * sc[10];
  J[360] += dqdci; // dwdot[CO]/d[CH2]
  J[361] -= dqdci; // dwdot[CO2]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[371] += dqdci; // dwdot[CH2O]/d[CH2]
  // d()/d[CH2O]
  dqdci = -k_r * sc[9];
  J[549] += dqdci; // dwdot[CO]/d[CH2O]
  J[550] -= dqdci; // dwdot[CO2]/d[CH2O]
  J[553] -= dqdci; // dwdot[CH2]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[712] -= dqdT; // dwdot[CO2]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 114: CH2(S) + O <=> CO + 2 H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[14];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = (sc[2] * sc[2]) * sc[9];
  Kc =
    refC * (kc_fast ? (ge[9] * ge[2] * inv_ge[4] * ge[2] * inv_ge[14])
                    : exp(-2.000000 * g_RT[2] + g_RT[4] - g_RT[9] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[4] + h_RT[14]) + (2.000000 * h_RT[2] + h_RT[9]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[4] -= q;     // O
  wdot[9] += q;     // CO
  wdot[14] -= q;    // CH2(S)
  // d()/d[H]
  dqdci = -k_r * 2.000000 * sc[2] * sc[9];
  J[56] += 2 * dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci;     // dwdot[O]/d[H]
  J[63] += dqdci;     // dwdot[CO]/d[H]
  J[68] -= dqdci;     // dwdot[CH2(S)]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[14];
  J[110] += 2 * dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci;     // dwdot[O]/d[O]
  J[117] += dqdci;     // dwdot[CO]/d[O]
  J[122] -= dqdci;     // dwdot[CH2(S)]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[2] * sc[2];
  J[245] += 2 * dqdci; // dwdot[H]/d[CO]
  J[247] -= dqdci;     // dwdot[O]/d[CO]
  J[252] += dqdci;     // dwdot[CO]/d[CO]
  J[257] -= dqdci;     // dwdot[CH2(S)]/d[CO]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[4];
  J[380] += 2 * dqdci; // dwdot[H]/d[CH2(S)]
  J[382] -= dqdci;     // dwdot[O]/d[CH2(S)]
  J[387] += dqdci;     // dwdot[CO]/d[CH2(S)]
  J[392] -= dqdci;     // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[706] -= dqdT;     // dwdot[O]/dT
  J[711] += dqdT;     // dwdot[CO]/dT
  J[716] -= dqdT;     // dwdot[CH2(S)]/dT

  // reaction 115: CH2(S) + CH4 <=> 2 CH3
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[11] * sc[14];
  k_f = 43000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = (sc[12] * sc[12]);
  Kc =
    (kc_fast ? (ge[12] * ge[12] * inv_ge[14] * inv_ge[11])
             : exp(g_RT[11] - 2.000000 * g_RT[12] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[11] + h_RT[14]) + (2.000000 * h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] -= q;     // CH4
  wdot[12] += 2 * q; // CH3
  wdot[14] -= q;     // CH2(S)
  // d()/d[CH4]
  dqdci = +k_f * sc[14];
  J[308] -= dqdci;     // dwdot[CH4]/d[CH4]
  J[309] += 2 * dqdci; // dwdot[CH3]/d[CH4]
  J[311] -= dqdci;     // dwdot[CH2(S)]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * 2.000000 * sc[12];
  J[335] -= dqdci;     // dwdot[CH4]/d[CH3]
  J[336] += 2 * dqdci; // dwdot[CH3]/d[CH3]
  J[338] -= dqdci;     // dwdot[CH2(S)]/d[CH3]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[11];
  J[389] -= dqdci;     // dwdot[CH4]/d[CH2(S)]
  J[390] += 2 * dqdci; // dwdot[CH3]/d[CH2(S)]
  J[392] -= dqdci;     // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[713] -= dqdT;     // dwdot[CH4]/dT
  J[714] += 2 * dqdT; // dwdot[CH3]/dT
  J[716] -= dqdT;     // dwdot[CH2(S)]/dT

  // reaction 116: CH3OH + H => CH3 + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[17];
  k_f = 650000 * exp(-(2667.04832876501) * invT);
  dlnkfdT = (2667.04832876501) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] -= q;  // H
  wdot[5] += q;  // H2O
  wdot[12] += q; // CH3
  wdot[17] -= q; // CH3OH
  // d()/d[H]
  dqdci = +k_f * sc[17];
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[59] += dqdci; // dwdot[H2O]/d[H]
  J[66] += dqdci; // dwdot[CH3]/d[H]
  J[71] -= dqdci; // dwdot[CH3OH]/d[H]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[2];
  J[461] -= dqdci; // dwdot[H]/d[CH3OH]
  J[464] += dqdci; // dwdot[H2O]/d[CH3OH]
  J[471] += dqdci; // dwdot[CH3]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  // d()/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT

  // reaction 117: CH2O + O => CO2 + 2 H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[20];
  k_f = 20000000 * exp(-(2516.0833290236) * invT);
  dlnkfdT = (2516.0833290236) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[4] -= q;     // O
  wdot[10] += q;    // CO2
  wdot[20] -= q;    // CH2O
  // d()/d[O]
  dqdci = +k_f * sc[20];
  J[110] += 2 * dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci;     // dwdot[O]/d[O]
  J[118] += dqdci;     // dwdot[CO2]/d[O]
  J[128] -= dqdci;     // dwdot[CH2O]/d[O]
  // d()/d[CH2O]
  dqdci = +k_f * sc[4];
  J[542] += 2 * dqdci; // dwdot[H]/d[CH2O]
  J[544] -= dqdci;     // dwdot[O]/d[CH2O]
  J[550] += dqdci;     // dwdot[CO2]/d[CH2O]
  J[560] -= dqdci;     // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[706] -= dqdT;     // dwdot[O]/dT
  J[712] += dqdT;     // dwdot[CO2]/dT
  J[722] -= dqdT;     // dwdot[CH2O]/dT

  // reaction 118: CH2O + OH => CO2 + H + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[20];
  k_f = 100000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[10] += q; // CO2
  wdot[20] -= q; // CH2O
  // d()/d[OH]
  dqdci = +k_f * sc[20];
  J[163] += dqdci; // dwdot[H2]/d[OH]
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[172] += dqdci; // dwdot[CO2]/d[OH]
  J[182] -= dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2O]
  dqdci = +k_f * sc[6];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[550] += dqdci; // dwdot[CO2]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[712] += dqdT; // dwdot[CO2]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT

  // reaction 119: CH3O + CO <=> CH3 + CO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[9] * sc[18];
  k_f = 500000 * exp(-(3270.90832773068) * invT);
  dlnkfdT = (3270.90832773068) * invT2;
  // reverse
  phi_r = sc[10] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[18] * ge[12] * inv_ge[9] * ge[10])
             : exp(g_RT[9] - g_RT[10] - g_RT[12] + g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[9] + h_RT[18]) + (h_RT[10] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  wdot[12] += q; // CH3
  wdot[18] -= q; // CH3O
  // d()/d[CO]
  dqdci = +k_f * sc[18];
  J[252] -= dqdci; // dwdot[CO]/d[CO]
  J[253] += dqdci; // dwdot[CO2]/d[CO]
  J[255] += dqdci; // dwdot[CH3]/d[CO]
  J[261] -= dqdci; // dwdot[CH3O]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * sc[12];
  J[279] -= dqdci; // dwdot[CO]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  J[282] += dqdci; // dwdot[CH3]/d[CO2]
  J[288] -= dqdci; // dwdot[CH3O]/d[CO2]
  // d()/d[CH3]
  dqdci = -k_r * sc[10];
  J[333] -= dqdci; // dwdot[CO]/d[CH3]
  J[334] += dqdci; // dwdot[CO2]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  J[342] -= dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3O]
  dqdci = +k_f * sc[9];
  J[495] -= dqdci; // dwdot[CO]/d[CH3O]
  J[496] += dqdci; // dwdot[CO2]/d[CH3O]
  J[498] += dqdci; // dwdot[CH3]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT

  // reaction 121: HCO + HO2 <=> CO + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[21];
  k_f = 400000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[7] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[21] * inv_ge[8] * ge[9] * ge[7])
             : exp(-g_RT[7] + g_RT[8] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[21]) + (h_RT[7] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[H2O2]
  dqdci = -k_r * sc[9];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[198] += dqdci; // dwdot[CO]/d[H2O2]
  J[210] -= dqdci; // dwdot[HCO]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[21];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[225] += dqdci; // dwdot[CO]/d[HO2]
  J[237] -= dqdci; // dwdot[HCO]/d[HO2]
  // d()/d[CO]
  dqdci = -k_r * sc[7];
  J[250] += dqdci; // dwdot[H2O2]/d[CO]
  J[251] -= dqdci; // dwdot[HO2]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[8];
  J[574] += dqdci; // dwdot[H2O2]/d[HCO]
  J[575] -= dqdci; // dwdot[HO2]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 122: CH3O + OH => CH2O + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[18];
  k_f = 15000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[OH]
  dqdci = +k_f * sc[18];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[180] -= dqdci; // dwdot[CH3O]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH3O]
  dqdci = +k_f * sc[6];
  J[491] += dqdci; // dwdot[H2O]/d[CH3O]
  J[492] -= dqdci; // dwdot[OH]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 123: CH2OH + CH3 => CH2O + CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[19];
  k_f = 15000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[CH3]
  dqdci = +k_f * sc[19];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[343] -= dqdci; // dwdot[CH2OH]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[12];
  J[524] += dqdci; // dwdot[CH4]/d[CH2OH]
  J[525] -= dqdci; // dwdot[CH3]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 124: CH + O <=> CHO+ + E
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[16];
  k_f = 1746000000000 * exp(-2.19 * logT - (164.999209334044) * invT);
  dlnkfdT = -2.19 * invT + (164.999209334044) * invT2;
  // reverse
  phi_r = sc[22] * sc[25];
  Kc =
    (kc_fast ? (ge[25] * inv_ge[4] * ge[22] * inv_ge[16])
             : exp(g_RT[4] + g_RT[16] - g_RT[22] - g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[16]) + (h_RT[22] + h_RT[25]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[16] -= q; // CH
  wdot[22] += q; // CHO+
  wdot[25] += q; // E
  // d()/d[O]
  dqdci = +k_f * sc[16];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[124] -= dqdci; // dwdot[CH]/d[O]
  J[130] += dqdci; // dwdot[CHO+]/d[O]
  J[133] += dqdci; // dwdot[E]/d[O]
  // d()/d[CH]
  dqdci = +k_f * sc[4];
  J[436] -= dqdci; // dwdot[O]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[454] += dqdci; // dwdot[CHO+]/d[CH]
  J[457] += dqdci; // dwdot[E]/d[CH]
  // d()/d[CHO+]
  dqdci = -k_r * sc[25];
  J[598] -= dqdci; // dwdot[O]/d[CHO+]
  J[610] -= dqdci; // dwdot[CH]/d[CHO+]
  J[616] += dqdci; // dwdot[CHO+]/d[CHO+]
  J[619] += dqdci; // dwdot[E]/d[CHO+]
  // d()/d[E]
  dqdci = -k_r * sc[22];
  J[679] -= dqdci; // dwdot[O]/d[E]
  J[691] -= dqdci; // dwdot[CH]/d[E]
  J[697] += dqdci; // dwdot[CHO+]/d[E]
  J[700] += dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[724] += dqdT; // dwdot[CHO+]/dT
  J[727] += dqdT; // dwdot[E]/dT

  // reaction 125: CHO+ + H2O <=> CO + H3O+
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[5] * sc[22];
  k_f = 26080000000 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = sc[9] * sc[23];
  Kc =
    (kc_fast ? (ge[9] * inv_ge[5] * inv_ge[22] * ge[23])
             : exp(g_RT[5] - g_RT[9] + g_RT[22] - g_RT[23]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[5] + h_RT[22]) + (h_RT[9] + h_RT[23]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] -= q;  // H2O
  wdot[9] += q;  // CO
  wdot[22] -= q; // CHO+
  wdot[23] += q; // H3O+
  // d()/d[H2O]
  dqdci = +k_f * sc[22];
  J[140] -= dqdci; // dwdot[H2O]/d[H2O]
  J[144] += dqdci; // dwdot[CO]/d[H2O]
  J[157] -= dqdci; // dwdot[CHO+]/d[H2O]
  J[158] += dqdci; // dwdot[H3O+]/d[H2O]
  // d()/d[CO]
  dqdci = -k_r * sc[23];
  J[248] -= dqdci; // dwdot[H2O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[265] -= dqdci; // dwdot[CHO+]/d[CO]
  J[266] += dqdci; // dwdot[H3O+]/d[CO]
  // d()/d[CHO+]
  dqdci = +k_f * sc[5];
  J[599] -= dqdci; // dwdot[H2O]/d[CHO+]
  J[603] += dqdci; // dwdot[CO]/d[CHO+]
  J[616] -= dqdci; // dwdot[CHO+]/d[CHO+]
  J[617] += dqdci; // dwdot[H3O+]/d[CHO+]
  // d()/d[H3O+]
  dqdci = -k_r * sc[9];
  J[626] -= dqdci; // dwdot[H2O]/d[H3O+]
  J[630] += dqdci; // dwdot[CO]/d[H3O+]
  J[643] -= dqdci; // dwdot[CHO+]/d[H3O+]
  J[644] += dqdci; // dwdot[H3O+]/d[H3O+]
  // d()/dT
  J[707] -= dqdT; // dwdot[H2O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[724] -= dqdT; // dwdot[CHO+]/dT
  J[725] += dqdT; // dwdot[H3O+]/dT

  // reaction 126: E + H3O+ <=> H + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[23] * sc[25];
  k_f = 739500000000 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = sc[2] * sc[5];
  Kc =
    (kc_fast ? (inv_ge[25] * ge[2] * ge[5] * inv_ge[23])
             : exp(-g_RT[2] - g_RT[5] + g_RT[23] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[23] + h_RT[25]) + (h_RT[2] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[5] += q;  // H2O
  wdot[23] -= q; // H3O+
  wdot[25] -= q; // E
  // d()/d[H]
  dqdci = -k_r * sc[5];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[59] += dqdci; // dwdot[H2O]/d[H]
  J[77] -= dqdci; // dwdot[H3O+]/d[H]
  J[79] -= dqdci; // dwdot[E]/d[H]
  // d()/d[H2O]
  dqdci = -k_r * sc[2];
  J[137] += dqdci; // dwdot[H]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[158] -= dqdci; // dwdot[H3O+]/d[H2O]
  J[160] -= dqdci; // dwdot[E]/d[H2O]
  // d()/d[H3O+]
  dqdci = +k_f * sc[25];
  J[623] += dqdci; // dwdot[H]/d[H3O+]
  J[626] += dqdci; // dwdot[H2O]/d[H3O+]
  J[644] -= dqdci; // dwdot[H3O+]/d[H3O+]
  J[646] -= dqdci; // dwdot[E]/d[H3O+]
  // d()/d[E]
  dqdci = +k_f * sc[23];
  J[677] += dqdci; // dwdot[H]/d[E]
  J[680] += dqdci; // dwdot[H2O]/d[E]
  J[698] -= dqdci; // dwdot[H3O+]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[725] -= dqdT; // dwdot[H3O+]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 127: E + H3O+ <=> 2 H + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[23] * sc[25];
  k_f = 3181000000000 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = (sc[2] * sc[2]) * sc[6];
  Kc =
    refC * (kc_fast ? (inv_ge[25] * ge[6] * ge[2] * inv_ge[23] * ge[2])
                    : exp(-2.000000 * g_RT[2] - g_RT[6] + g_RT[23] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[23] + h_RT[25]) + (2.000000 * h_RT[2] + h_RT[6]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[6] += q;     // OH
  wdot[23] -= q;    // H3O+
  wdot[25] -= q;    // E
  // d()/d[H]
  dqdci = -k_r * 2.000000 * sc[2] * sc[6];
  J[56] += 2 * dqdci; // dwdot[H]/d[H]
  J[60] += dqdci;     // dwdot[OH]/d[H]
  J[77] -= dqdci;     // dwdot[H3O+]/d[H]
  J[79] -= dqdci;     // dwdot[E]/d[H]
  // d()/d[OH]
  dqdci = -k_r * sc[2] * sc[2];
  J[164] += 2 * dqdci; // dwdot[H]/d[OH]
  J[168] += dqdci;     // dwdot[OH]/d[OH]
  J[185] -= dqdci;     // dwdot[H3O+]/d[OH]
  J[187] -= dqdci;     // dwdot[E]/d[OH]
  // d()/d[H3O+]
  dqdci = +k_f * sc[25];
  J[623] += 2 * dqdci; // dwdot[H]/d[H3O+]
  J[627] += dqdci;     // dwdot[OH]/d[H3O+]
  J[644] -= dqdci;     // dwdot[H3O+]/d[H3O+]
  J[646] -= dqdci;     // dwdot[E]/d[H3O+]
  // d()/d[E]
  dqdci = +k_f * sc[23];
  J[677] += 2 * dqdci; // dwdot[H]/d[E]
  J[681] += dqdci;     // dwdot[OH]/d[E]
  J[698] -= dqdci;     // dwdot[H3O+]/d[E]
  J[700] -= dqdci;     // dwdot[E]/d[E]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[708] += dqdT;     // dwdot[OH]/dT
  J[725] -= dqdT;     // dwdot[H3O+]/dT
  J[727] -= dqdT;     // dwdot[E]/dT

  // reaction 128: E + H3O+ <=> H2 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[23] * sc[25];
  k_f = 560100000000 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = sc[1] * sc[6];
  Kc =
    (kc_fast ? (inv_ge[25] * ge[1] * ge[6] * inv_ge[23])
             : exp(-g_RT[1] - g_RT[6] + g_RT[23] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[23] + h_RT[25]) + (h_RT[1] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[6] += q;  // OH
  wdot[23] -= q; // H3O+
  wdot[25] -= q; // E
  // d()/d[H2]
  dqdci = -k_r * sc[6];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[33] += dqdci; // dwdot[OH]/d[H2]
  J[50] -= dqdci; // dwdot[H3O+]/d[H2]
  J[52] -= dqdci; // dwdot[E]/d[H2]
  // d()/d[OH]
  dqdci = -k_r * sc[1];
  J[163] += dqdci; // dwdot[H2]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[185] -= dqdci; // dwdot[H3O+]/d[OH]
  J[187] -= dqdci; // dwdot[E]/d[OH]
  // d()/d[H3O+]
  dqdci = +k_f * sc[25];
  J[622] += dqdci; // dwdot[H2]/d[H3O+]
  J[627] += dqdci; // dwdot[OH]/d[H3O+]
  J[644] -= dqdci; // dwdot[H3O+]/d[H3O+]
  J[646] -= dqdci; // dwdot[E]/d[H3O+]
  // d()/d[E]
  dqdci = +k_f * sc[23];
  J[676] += dqdci; // dwdot[H2]/d[E]
  J[681] += dqdci; // dwdot[OH]/d[E]
  J[698] -= dqdci; // dwdot[H3O+]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[725] -= dqdT; // dwdot[H3O+]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 129: H2 + O2- <=> E + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[24];
  k_f = 602000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[7] * sc[25];
  Kc =
    (kc_fast ? (ge[25] * inv_ge[1] * inv_ge[24] * ge[7])
             : exp(g_RT[1] - g_RT[7] + g_RT[24] - g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[24]) + (h_RT[7] + h_RT[25]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q;  // H2
  wdot[7] += q;  // H2O2
  wdot[24] -= q; // O2-
  wdot[25] += q; // E
  // d()/d[H2]
  dqdci = +k_f * sc[24];
  J[28] -= dqdci; // dwdot[H2]/d[H2]
  J[34] += dqdci; // dwdot[H2O2]/d[H2]
  J[51] -= dqdci; // dwdot[O2-]/d[H2]
  J[52] += dqdci; // dwdot[E]/d[H2]
  // d()/d[H2O2]
  dqdci = -k_r * sc[25];
  J[190] -= dqdci; // dwdot[H2]/d[H2O2]
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[213] -= dqdci; // dwdot[O2-]/d[H2O2]
  J[214] += dqdci; // dwdot[E]/d[H2O2]
  // d()/d[O2-]
  dqdci = +k_f * sc[1];
  J[649] -= dqdci; // dwdot[H2]/d[O2-]
  J[655] += dqdci; // dwdot[H2O2]/d[O2-]
  J[672] -= dqdci; // dwdot[O2-]/d[O2-]
  J[673] += dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = -k_r * sc[7];
  J[676] -= dqdci; // dwdot[H2]/d[E]
  J[682] += dqdci; // dwdot[H2O2]/d[E]
  J[699] -= dqdci; // dwdot[O2-]/d[E]
  J[700] += dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[703] -= dqdT; // dwdot[H2]/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[726] -= dqdT; // dwdot[O2-]/dT
  J[727] += dqdT; // dwdot[E]/dT

  // reaction 130: H + O2- <=> E + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[24];
  k_f = 723000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[8] * sc[25];
  Kc =
    (kc_fast ? (ge[25] * ge[8] * inv_ge[24] * inv_ge[2])
             : exp(g_RT[2] - g_RT[8] + g_RT[24] - g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[24]) + (h_RT[8] + h_RT[25]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q;  // H
  wdot[8] += q;  // HO2
  wdot[24] -= q; // O2-
  wdot[25] += q; // E
  // d()/d[H]
  dqdci = +k_f * sc[24];
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[62] += dqdci; // dwdot[HO2]/d[H]
  J[78] -= dqdci; // dwdot[O2-]/d[H]
  J[79] += dqdci; // dwdot[E]/d[H]
  // d()/d[HO2]
  dqdci = -k_r * sc[25];
  J[218] -= dqdci; // dwdot[H]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[240] -= dqdci; // dwdot[O2-]/d[HO2]
  J[241] += dqdci; // dwdot[E]/d[HO2]
  // d()/d[O2-]
  dqdci = +k_f * sc[2];
  J[650] -= dqdci; // dwdot[H]/d[O2-]
  J[656] += dqdci; // dwdot[HO2]/d[O2-]
  J[672] -= dqdci; // dwdot[O2-]/d[O2-]
  J[673] += dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = -k_r * sc[8];
  J[677] -= dqdci; // dwdot[H]/d[E]
  J[683] += dqdci; // dwdot[HO2]/d[E]
  J[699] -= dqdci; // dwdot[O2-]/d[E]
  J[700] += dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[726] -= dqdT; // dwdot[O2-]/dT
  J[727] += dqdT; // dwdot[E]/dT

  amrex::Real c_R[26], dcRdT[26], e_RT[26];
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
  for (int k = 0; k < 26; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[702 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 26; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 26; ++m) {
      dehmixdc += eh_RT[m] * J[k * 27 + m];
    }
    J[k * 27 + 26] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[728] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
#else
  amrex::Abort();
#endif
}

// compute an approx to the reaction Jacobian (for preconditioning)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
DWDOT_SIMPLIFIED(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real* Tp, const int* HP)
{
  amrex::Real c[26];

  for (int k = 0; k < 26; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian_precond(J, c, *Tp, *HP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 26; k++) {
    J[702 + k] *= 1.e-6;
    J[k * 27 + 26] *= 1.e6;
  }
}

// compute the reaction Jacobian
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
aJacobian(
  amrex::Real* J, const amrex::Real* sc, const amrex::Real T, const int consP)
{

#if defined(PELE_COMPILE_AJACOBIAN) || !defined(AMREX_USE_HIP)
  for (int i = 0; i < 729; i++) {
    J[i] = 0.0;
  }

  amrex::Real wdot[26];
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
  for (int k = 0; k < 26; ++k) {
    mixture += sc[k];
  }

  // compute the Gibbs free energy
  amrex::Real g_RT[26];
  gibbs(g_RT, T);
  // Kc from shared exp(-g/RT) factors where no factor or partial product can
  // overflow (checked by CEPTR on this T range); exp of the sum otherwise and
  // in single precision
  const bool kc_fast =
    sizeof(amrex::Real) == sizeof(double) && T >= 200.0 && T <= 10000.0;
  amrex::Real ge[26] = {}, inv_ge[26] = {};
  if (kc_fast) {
    for (int i = 0; i < 26; ++i) {
      ge[i] = exp(-g_RT[i]);
      inv_ge[i] = 1.0 / ge[i];
    }
  }

  // compute the species enthalpy
  amrex::Real h_RT[26];
  speciesEnthalpy(h_RT, T);

  amrex::Real phi_f, k_f, k_r, phi_r, Kc, q, q_nocor, Corr, alpha;
  amrex::Real dlnkfdT, dlnk0dT, dlnKcdT, dkrdT, dqdT;
  amrex::Real dqdci, dcdc_fac, dqdc[26];
  amrex::Real Pr, fPr, F, k_0, logPr;
  amrex::Real logFcent, troe_c, troe_n, troePr_den, troePr, troe;
  amrex::Real Fcent1, Fcent2, Fcent3, Fcent;
  amrex::Real dlogFdc, dlogFdn, dlogFdcn_fac;
  amrex::Real dlogPrdT, dlogfPrdT, dlogFdT, dlogFcentdT, dlogFdlogPr, dlnCorrdT;
  const amrex::Real ln10 = log(10.0);
  const amrex::Real log10e = 1.0 / log(10.0);
  // reaction 6: H2O2 (+M) <=> 2 OH (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (1.5 - 1) * sc[0] + (3.7 - 1) * sc[1] + (1.2 - 1) * sc[3] +
          (7.65 - 1) * sc[5] + (7.7 - 1) * sc[7] + (2.8 - 1) * sc[9] +
          (1.6 - 1) * sc[10];
  // forward
  phi_f = sc[7];
  k_f = 2000000000000 * exp(0.9 * logT - (24531.3092413143) * invT);
  dlnkfdT = 0.9 * invT + (24531.3092413143) * invT2;
  // pressure-fall-off
  k_0 = 2.49e+24 * exp(-2.3 * logT - (24531.3092413143) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -2.3 * invT + (24531.3092413143) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.43) * exp(-T / 1e-30);
  Fcent2 = 0.43 * exp(-T / 1e+30);
  Fcent3 = exp(-1e-16 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT = log10e / Fcent *
                (-Fcent1 / 1e-30 - Fcent2 / 1e+30 + Fcent3 * 1e-16 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = (sc[6] * sc[6]);
  Kc = refC * (kc_fast ? (ge[6] * ge[6] * inv_ge[7])
                       : exp(-2.000000 * g_RT[6] + g_RT[7]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[7]) + (2.000000 * h_RT[6]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[6] += 2 * q; // OH
  wdot[7] -= q;     // H2O2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[N2]
    dqdci = (1.5 - 1) * dcdc_fac;
    J[6] += 2 * dqdci; // dwdot[OH]/d[N2]
    J[7] -= dqdci;     // dwdot[H2O2]/d[N2]
    // d()/d[H2]
    dqdci = (3.7 - 1) * dcdc_fac;
    J[33] += 2 * dqdci; // dwdot[OH]/d[H2]
    J[34] -= dqdci;     // dwdot[H2O2]/d[H2]
    // d()/d[O2]
    dqdci = (1.2 - 1) * dcdc_fac;
    J[87] += 2 * dqdci; // dwdot[OH]/d[O2]
    J[88] -= dqdci;     // dwdot[H2O2]/d[O2]
    // d()/d[H2O]
    dqdci = (7.65 - 1) * dcdc_fac;
    J[141] += 2 * dqdci; // dwdot[OH]/d[H2O]
    J[142] -= dqdci;     // dwdot[H2O2]/d[H2O]
    // d()/d[OH]
    dqdci = -k_r * 2.000000 * sc[6];
    J[168] += 2 * dqdci; // dwdot[OH]/d[OH]
    J[169] -= dqdci;     // dwdot[H2O2]/d[OH]
    // d()/d[H2O2]
    dqdci = (7.7 - 1) * dcdc_fac + k_f;
    J[195] += 2 * dqdci; // dwdot[OH]/d[H2O2]
    J[196] -= dqdci;     // dwdot[H2O2]/d[H2O2]
    // d()/d[CO]
    dqdci = (2.8 - 1) * dcdc_fac;
    J[249] += 2 * dqdci; // dwdot[OH]/d[CO]
    J[250] -= dqdci;     // dwdot[H2O2]/d[CO]
    // d()/d[CO2]
    dqdci = (1.6 - 1) * dcdc_fac;
    J[276] += 2 * dqdci; // dwdot[OH]/d[CO2]
    J[277] -= dqdci;     // dwdot[H2O2]/d[CO2]
  } else {
    dqdc[0] = 1.5 * dcdc_fac;
    dqdc[1] = 3.7 * dcdc_fac;
    dqdc[2] = dcdc_fac;
    dqdc[3] = 1.2 * dcdc_fac;
    dqdc[4] = dcdc_fac;
    dqdc[5] = 7.65 * dcdc_fac;
    dqdc[6] = dcdc_fac - k_r * 2.000000 * sc[6];
    dqdc[7] = 7.7 * dcdc_fac + k_f;
    dqdc[8] = dcdc_fac;
    dqdc[9] = 2.8 * dcdc_fac;
    dqdc[10] = 1.6 * dcdc_fac;
    dqdc[11] = dcdc_fac;
    dqdc[12] = dcdc_fac;
    dqdc[13] = dcdc_fac;
    dqdc[14] = dcdc_fac;
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac;
    dqdc[18] = dcdc_fac;
    dqdc[19] = dcdc_fac;
    dqdc[20] = dcdc_fac;
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 6] += 2 * dqdc[k];
      J[27 * k + 7] -= dqdc[k];
    }
  }
  J[708] += 2 * dqdT; // dwdot[OH]/dT
  J[709] -= dqdT;     // dwdot[H2O2]/dT

  // reaction 19: H + O2 (+M) <=> HO2 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (1.3 - 1) * sc[1] + (10 - 1) * sc[5] + (1.9 - 1) * sc[9] +
          (3.8 - 1) * sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[3];
  k_f = 4650000 * exp(0.44 * logT);
  dlnkfdT = 0.44 * invT;
  // pressure-fall-off
  k_0 = 1.74e+19 * exp(-1.23 * logT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -1.23 * invT;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.67) * exp(-T / 1e-30);
  Fcent2 = 0.67 * exp(-T / 1e+30);
  Fcent3 = exp(-1e+30 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT = log10e / Fcent *
                (-Fcent1 / 1e-30 - Fcent2 / 1e+30 + Fcent3 * 1e+30 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[8];
  Kc = refCinv * (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[2])
                          : exp(g_RT[2] + g_RT[3] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[8]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] -= q; // H
  wdot[3] -= q; // O2
  wdot[8] += q; // HO2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (1.3 - 1) * dcdc_fac;
    J[29] -= dqdci; // dwdot[H]/d[H2]
    J[30] -= dqdci; // dwdot[O2]/d[H2]
    J[35] += dqdci; // dwdot[HO2]/d[H2]
    // d()/d[H]
    dqdci = +k_f * sc[3];
    J[56] -= dqdci; // dwdot[H]/d[H]
    J[57] -= dqdci; // dwdot[O2]/d[H]
    J[62] += dqdci; // dwdot[HO2]/d[H]
    // d()/d[O2]
    dqdci = +k_f * sc[2];
    J[83] -= dqdci; // dwdot[H]/d[O2]
    J[84] -= dqdci; // dwdot[O2]/d[O2]
    J[89] += dqdci; // dwdot[HO2]/d[O2]
    // d()/d[H2O]
    dqdci = (10 - 1) * dcdc_fac;
    J[137] -= dqdci; // dwdot[H]/d[H2O]
    J[138] -= dqdci; // dwdot[O2]/d[H2O]
    J[143] += dqdci; // dwdot[HO2]/d[H2O]
    // d()/d[HO2]
    dqdci = -k_r;
    J[218] -= dqdci; // dwdot[H]/d[HO2]
    J[219] -= dqdci; // dwdot[O2]/d[HO2]
    J[224] += dqdci; // dwdot[HO2]/d[HO2]
    // d()/d[CO]
    dqdci = (1.9 - 1) * dcdc_fac;
    J[245] -= dqdci; // dwdot[H]/d[CO]
    J[246] -= dqdci; // dwdot[O2]/d[CO]
    J[251] += dqdci; // dwdot[HO2]/d[CO]
    // d()/d[CO2]
    dqdci = (3.8 - 1) * dcdc_fac;
    J[272] -= dqdci; // dwdot[H]/d[CO2]
    J[273] -= dqdci; // dwdot[O2]/d[CO2]
    J[278] += dqdci; // dwdot[HO2]/d[CO2]
    // d()/d[CH4]
    dqdci = (2 - 1) * dcdc_fac;
    J[299] -= dqdci; // dwdot[H]/d[CH4]
    J[300] -= dqdci; // dwdot[O2]/d[CH4]
    J[305] += dqdci; // dwdot[HO2]/d[CH4]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = 1.3 * dcdc_fac;
    dqdc[2] = dcdc_fac + k_f * sc[3];
    dqdc[3] = dcdc_fac + k_f * sc[2];
    dqdc[4] = dcdc_fac;
    dqdc[5] = 10 * dcdc_fac;
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac - k_r;
    dqdc[9] = 1.9 * dcdc_fac;
    dqdc[10] = 3.8 * dcdc_fac;
    dqdc[11] = 2 * dcdc_fac;
    dqdc[12] = dcdc_fac;
    dqdc[13] = dcdc_fac;
    dqdc[14] = dcdc_fac;
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac;
    dqdc[18] = dcdc_fac;
    dqdc[19] = dcdc_fac;
    dqdc[20] = dcdc_fac;
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 2] -= dqdc[k];
      J[27 * k + 3] -= dqdc[k];
      J[27 * k + 8] += dqdc[k];
    }
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 24: CH3 + H (+M) <=> CH4 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[12];
  k_f = 12700000000 * exp(-0.63 * logT - (192.731983003208) * invT);
  dlnkfdT = -0.63 * invT + (192.731983003208) * invT2;
  // pressure-fall-off
  k_0 = 2.48e+33 * exp(-4.76 * logT - (1227.84866456352) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -4.76 * invT + (1227.84866456352) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.783) * exp(-T / 74);
  Fcent2 = 0.783 * exp(-T / 2941);
  Fcent3 = exp(-6964 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 74 - Fcent2 / 2941 + Fcent3 * 6964 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[11];
  Kc = refCinv * (kc_fast ? (inv_ge[12] * ge[11] * inv_ge[2])
                          : exp(g_RT[2] - g_RT[11] + g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[12]) + (h_RT[11]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] -= q;  // H
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (2 - 1) * dcdc_fac;
    J[29] -= dqdci; // dwdot[H]/d[H2]
    J[38] += dqdci; // dwdot[CH4]/d[H2]
    J[39] -= dqdci; // dwdot[CH3]/d[H2]
    // d()/d[H]
    dqdci = +k_f * sc[12];
    J[56] -= dqdci; // dwdot[H]/d[H]
    J[65] += dqdci; // dwdot[CH4]/d[H]
    J[66] -= dqdci; // dwdot[CH3]/d[H]
    // d()/d[H2O]
    dqdci = (6 - 1) * dcdc_fac;
    J[137] -= dqdci; // dwdot[H]/d[H2O]
    J[146] += dqdci; // dwdot[CH4]/d[H2O]
    J[147] -= dqdci; // dwdot[CH3]/d[H2O]
    // d()/d[CO]
    dqdci = (1.5 - 1) * dcdc_fac;
    J[245] -= dqdci; // dwdot[H]/d[CO]
    J[254] += dqdci; // dwdot[CH4]/d[CO]
    J[255] -= dqdci; // dwdot[CH3]/d[CO]
    // d()/d[CO2]
    dqdci = (2 - 1) * dcdc_fac;
    J[272] -= dqdci; // dwdot[H]/d[CO2]
    J[281] += dqdci; // dwdot[CH4]/d[CO2]
    J[282] -= dqdci; // dwdot[CH3]/d[CO2]
    // d()/d[CH4]
    dqdci = (2 - 1) * dcdc_fac - k_r;
    J[299] -= dqdci; // dwdot[H]/d[CH4]
    J[308] += dqdci; // dwdot[CH4]/d[CH4]
    J[309] -= dqdci; // dwdot[CH3]/d[CH4]
    // d()/d[CH3]
    dqdci = +k_f * sc[2];
    J[326] -= dqdci; // dwdot[H]/d[CH3]
    J[335] += dqdci; // dwdot[CH4]/d[CH3]
    J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = 2 * dcdc_fac;
    dqdc[2] = dcdc_fac + k_f * sc[12];
    dqdc[3] = dcdc_fac;
    dqdc[4] = dcdc_fac;
    dqdc[5] = 6 * dcdc_fac;
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac;
    dqdc[9] = 1.5 * dcdc_fac;
    dqdc[10] = 2 * dcdc_fac;
    dqdc[11] = 2 * dcdc_fac - k_r;
    dqdc[12] = dcdc_fac + k_f * sc[2];
    dqdc[13] = dcdc_fac;
    dqdc[14] = dcdc_fac;
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac;
    dqdc[18] = dcdc_fac;
    dqdc[19] = dcdc_fac;
    dqdc[20] = dcdc_fac;
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 2] -= dqdc[k];
      J[27 * k + 11] += dqdc[k];
      J[27 * k + 12] -= dqdc[k];
    }
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT

  // reaction 43: CH2 + H (+M) <=> CH3 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[13];
  k_f = 25000000000 * exp(-0.8 * logT);
  dlnkfdT = -0.8 * invT;
  // pressure-fall-off
  k_0 = 3.2e+27 * exp(-3.14 * logT - (618.956498939805) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -3.14 * invT + (618.956498939805) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.68) * exp(-T / 78);
  Fcent2 = 0.68 * exp(-T / 1995);
  Fcent3 = exp(-5590 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 78 - Fcent2 / 1995 + Fcent3 * 5590 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[12];
  Kc = refCinv * (kc_fast ? (ge[12] * inv_ge[2] * inv_ge[13])
                          : exp(g_RT[2] - g_RT[12] + g_RT[13]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[13]) + (h_RT[12]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] -= q;  // H
  wdot[12] += q; // CH3
  wdot[13] -= q; // CH2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (2 - 1) * dcdc_fac;
    J[29] -= dqdci; // dwdot[H]/d[H2]
    J[39] += dqdci; // dwdot[CH3]/d[H2]
    J[40] -= dqdci; // dwdot[CH2]/d[H2]
    // d()/d[H]
    dqdci = +k_f * sc[13];
    J[56] -= dqdci; // dwdot[H]/d[H]
    J[66] += dqdci; // dwdot[CH3]/d[H]
    J[67] -= dqdci; // dwdot[CH2]/d[H]
    // d()/d[H2O]
    dqdci = (6 - 1) * dcdc_fac;
    J[137] -= dqdci; // dwdot[H]/d[H2O]
    J[147] += dqdci; // dwdot[CH3]/d[H2O]
    J[148] -= dqdci; // dwdot[CH2]/d[H2O]
    // d()/d[CO]
    dqdci = (1.5 - 1) * dcdc_fac;
    J[245] -= dqdci; // dwdot[H]/d[CO]
    J[255] += dqdci; // dwdot[CH3]/d[CO]
    J[256] -= dqdci; // dwdot[CH2]/d[CO]
    // d()/d[CO2]
    dqdci = (2 - 1) * dcdc_fac;
    J[272] -= dqdci; // dwdot[H]/d[CO2]
    J[282] += dqdci; // dwdot[CH3]/d[CO2]
    J[283] -= dqdci; // dwdot[CH2]/d[CO2]
    // d()/d[CH4]
    dqdci = (2 - 1) * dcdc_fac;
    J[299] -= dqdci; // dwdot[H]/d[CH4]
    J[309] += dqdci; // dwdot[CH3]/d[CH4]
    J[310] -= dqdci; // dwdot[CH2]/d[CH4]
    // d()/d[CH3]
    dqdci = -k_r;
    J[326] -= dqdci; // dwdot[H]/d[CH3]
    J[336] += dqdci; // dwdot[CH3]/d[CH3]
    J[337] -= dqdci; // dwdot[CH2]/d[CH3]
    // d()/d[CH2]
    dqdci = +k_f * sc[2];
    J[353] -= dqdci; // dwdot[H]/d[CH2]
    J[363] += dqdci; // dwdot[CH3]/d[CH2]
    J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = 2 * dcdc_fac;
    dqdc[2] = dcdc_fac + k_f * sc[13];
    dqdc[3] = dcdc_fac;
    dqdc[4] = dcdc_fac;
    dqdc[5] = 6 * dcdc_fac;
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac;
    dqdc[9] = 1.5 * dcdc_fac;
    dqdc[10] = 2 * dcdc_fac;
    dqdc[11] = 2 * dcdc_fac;
    dqdc[12] = dcdc_fac - k_r;
    dqdc[13] = dcdc_fac + k_f * sc[2];
    dqdc[14] = dcdc_fac;
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac;
    dqdc[18] = dcdc_fac;
    dqdc[19] = dcdc_fac;
    dqdc[20] = dcdc_fac;
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 2] -= dqdc[k];
      J[27 * k + 12] += dqdc[k];
      J[27 * k + 13] -= dqdc[k];
    }
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT

  // reaction 66: CH3OH (+M) <=> CH3 + OH (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[17];
  k_f = 2.084e+18 * exp(-0.615 * logT - (46567.9721835682) * invT);
  dlnkfdT = -0.615 * invT + (46567.9721835682) * invT2;
  // pressure-fall-off
  k_0 = 1.5e+43 * exp(-6.995 * logT - (49311.3081588693) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -6.995 * invT + (49311.3081588693) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. + 0.4748) * exp(-T / 35580);
  Fcent2 = -0.4748 * exp(-T / 1116);
  Fcent3 = exp(-9023 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 35580 - Fcent2 / 1116 + Fcent3 * 9023 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[6] * sc[12];
  Kc = refC * (kc_fast ? (ge[6] * ge[12] * inv_ge[17])
                       : exp(-g_RT[6] - g_RT[12] + g_RT[17]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[17]) + (h_RT[6] + h_RT[12]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[6] += q;  // OH
  wdot[12] += q; // CH3
  wdot[17] -= q; // CH3OH
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[OH]
    dqdci = -k_r * sc[12];
    J[168] += dqdci; // dwdot[OH]/d[OH]
    J[174] += dqdci; // dwdot[CH3]/d[OH]
    J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
    // d()/d[CH3]
    dqdci = -k_r * sc[6];
    J[330] += dqdci; // dwdot[OH]/d[CH3]
    J[336] += dqdci; // dwdot[CH3]/d[CH3]
    J[341] -= dqdci; // dwdot[CH3OH]/d[CH3]
    // d()/d[CH3OH]
    dqdci = +k_f;
    J[465] += dqdci; // dwdot[OH]/d[CH3OH]
    J[471] += dqdci; // dwdot[CH3]/d[CH3OH]
    J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = dcdc_fac;
    dqdc[2] = dcdc_fac;
    dqdc[3] = dcdc_fac;
    dqdc[4] = dcdc_fac;
    dqdc[5] = dcdc_fac;
    dqdc[6] = dcdc_fac - k_r * sc[12];
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac;
    dqdc[9] = dcdc_fac;
    dqdc[10] = dcdc_fac;
    dqdc[11] = dcdc_fac;
    dqdc[12] = dcdc_fac - k_r * sc[6];
    dqdc[13] = dcdc_fac;
    dqdc[14] = dcdc_fac;
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac + k_f;
    dqdc[18] = dcdc_fac;
    dqdc[19] = dcdc_fac;
    dqdc[20] = dcdc_fac;
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 6] += dqdc[k];
      J[27 * k + 12] += dqdc[k];
      J[27 * k + 17] -= dqdc[k];
    }
  }
  J[708] += dqdT; // dwdot[OH]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT

  // reaction 67: CH3OH (+M) <=> CH2(S) + H2O (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[17];
  k_f = 3.121e+18 * exp(-1.017 * logT - (46151.0068542825) * invT);
  dlnkfdT = -1.017 * invT + (46151.0068542825) * invT2;
  // pressure-fall-off
  k_0 = 1.43e+47 * exp(-8.227 * logT - (50028.3415859744) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -8.227 * invT + (50028.3415859744) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 2.545) * exp(-T / 3290);
  Fcent2 = 2.545 * exp(-T / 47320);
  Fcent3 = exp(-47110 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 3290 - Fcent2 / 47320 + Fcent3 * 47110 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[5] * sc[14];
  Kc = refC * (kc_fast ? (inv_ge[17] * ge[5] * ge[14])
                       : exp(-g_RT[5] - g_RT[14] + g_RT[17]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[17]) + (h_RT[5] + h_RT[14]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[5] += q;  // H2O
  wdot[14] += q; // CH2(S)
  wdot[17] -= q; // CH3OH
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[H2O]
    dqdci = -k_r * sc[14];
    J[140] += dqdci; // dwdot[H2O]/d[H2O]
    J[149] += dqdci; // dwdot[CH2(S)]/d[H2O]
    J[152] -= dqdci; // dwdot[CH3OH]/d[H2O]
    // d()/d[CH2(S)]
    dqdci = -k_r * sc[5];
    J[383] += dqdci; // dwdot[H2O]/d[CH2(S)]
    J[392] += dqdci; // dwdot[CH2(S)]/d[CH2(S)]
    J[395] -= dqdci; // dwdot[CH3OH]/d[CH2(S)]
    // d()/d[CH3OH]
    dqdci = +k_f;
    J[464] += dqdci; // dwdot[H2O]/d[CH3OH]
    J[473] += dqdci; // dwdot[CH2(S)]/d[CH3OH]
    J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = dcdc_fac;
    dqdc[2] = dcdc_fac;
    dqdc[3] = dcdc_fac;
    dqdc[4] = dcdc_fac;
    dqdc[5] = dcdc_fac - k_r * sc[14];
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac;
    dqdc[9] = dcdc_fac;
    dqdc[10] = dcdc_fac;
    dqdc[11] = dcdc_fac;
    dqdc[12] = dcdc_fac;
    dqdc[13] = dcdc_fac;
    dqdc[14] = dcdc_fac - k_r * sc[5];
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac + k_f;
    dqdc[18] = dcdc_fac;
    dqdc[19] = dcdc_fac;
    dqdc[20] = dcdc_fac;
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 5] += dqdc[k];
      J[27 * k + 14] += dqdc[k];
      J[27 * k + 17] -= dqdc[k];
    }
  }
  J[707] += dqdT; // dwdot[H2O]/dT
  J[716] += dqdT; // dwdot[CH2(S)]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT

  // reaction 106: CH2O + H (+M) <=> CH2OH (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[20];
  k_f = 540000 * exp(0.454 * logT - (1811.57999689699) * invT);
  dlnkfdT = 0.454 * invT + (1811.57999689699) * invT2;
  // pressure-fall-off
  k_0 = 1.27e+32 * exp(-4.82 * logT - (3286.00482770482) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -4.82 * invT + (3286.00482770482) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.7187) * exp(-T / 103);
  Fcent2 = 0.7187 * exp(-T / 1291);
  Fcent3 = exp(-4160 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 103 - Fcent2 / 1291 + Fcent3 * 4160 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[19];
  Kc = refCinv * (kc_fast ? (ge[19] * inv_ge[20] * inv_ge[2])
                          : exp(g_RT[2] - g_RT[19] + g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[20]) + (h_RT[19]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] -= q;  // H
  wdot[19] += q; // CH2OH
  wdot[20] -= q; // CH2O
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (2 - 1) * dcdc_fac;
    J[29] -= dqdci; // dwdot[H]/d[H2]
    J[46] += dqdci; // dwdot[CH2OH]/d[H2]
    J[47] -= dqdci; // dwdot[CH2O]/d[H2]
    // d()/d[H]
    dqdci = +k_f * sc[20];
    J[56] -= dqdci; // dwdot[H]/d[H]
    J[73] += dqdci; // dwdot[CH2OH]/d[H]
    J[74] -= dqdci; // dwdot[CH2O]/d[H]
    // d()/d[H2O]
    dqdci = (6 - 1) * dcdc_fac;
    J[137] -= dqdci; // dwdot[H]/d[H2O]
    J[154] += dqdci; // dwdot[CH2OH]/d[H2O]
    J[155] -= dqdci; // dwdot[CH2O]/d[H2O]
    // d()/d[CO]
    dqdci = (1.5 - 1) * dcdc_fac;
    J[245] -= dqdci; // dwdot[H]/d[CO]
    J[262] += dqdci; // dwdot[CH2OH]/d[CO]
    J[263] -= dqdci; // dwdot[CH2O]/d[CO]
    // d()/d[CO2]
    dqdci = (2 - 1) * dcdc_fac;
    J[272] -= dqdci; // dwdot[H]/d[CO2]
    J[289] += dqdci; // dwdot[CH2OH]/d[CO2]
    J[290] -= dqdci; // dwdot[CH2O]/d[CO2]
    // d()/d[CH4]
    dqdci = (2 - 1) * dcdc_fac;
    J[299] -= dqdci; // dwdot[H]/d[CH4]
    J[316] += dqdci; // dwdot[CH2OH]/d[CH4]
    J[317] -= dqdci; // dwdot[CH2O]/d[CH4]
    // d()/d[CH2OH]
    dqdci = -k_r;
    J[515] -= dqdci; // dwdot[H]/d[CH2OH]
    J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
    J[533] -= dqdci; // dwdot[CH2O]/d[CH2OH]
    // d()/d[CH2O]
    dqdci = +k_f * sc[2];
    J[542] -= dqdci; // dwdot[H]/d[CH2O]
    J[559] += dqdci; // dwdot[CH2OH]/d[CH2O]
    J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = 2 * dcdc_fac;
    dqdc[2] = dcdc_fac + k_f * sc[20];
    dqdc[3] = dcdc_fac;
    dqdc[4] = dcdc_fac;
    dqdc[5] = 6 * dcdc_fac;
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac;
    dqdc[9] = 1.5 * dcdc_fac;
    dqdc[10] = 2 * dcdc_fac;
    dqdc[11] = 2 * dcdc_fac;
    dqdc[12] = dcdc_fac;
    dqdc[13] = dcdc_fac;
    dqdc[14] = dcdc_fac;
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac;
    dqdc[18] = dcdc_fac;
    dqdc[19] = dcdc_fac - k_r;
    dqdc[20] = dcdc_fac + k_f * sc[2];
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 2] -= dqdc[k];
      J[27 * k + 19] += dqdc[k];
      J[27 * k + 20] -= dqdc[k];
    }
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT

  // reaction 107: CH3O (+M) <=> CH2O + H (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[18];
  k_f = 68000000000000 * exp(-(13169.1801441095) * invT);
  dlnkfdT = (13169.1801441095) * invT2;
  // pressure-fall-off
  k_0 = 1.87e+25 * exp(-3 * logT - (12231.6874957153) * invT);
  Pr = 1e-6 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -3 * invT + (12231.6874957153) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Troe form
  logPr = log10(Pr);
  Fcent1 = (1. - 0.9) * exp(-T / 2500);
  Fcent2 = 0.9 * exp(-T / 1300);
  Fcent3 = exp(-1e+99 * invT);
  Fcent = Fcent1 + Fcent2 + Fcent3;
  logFcent = log10(Fcent);
  troe_c = -.4 - .67 * logFcent;
  troe_n = .75 - 1.27 * logFcent;
  troePr_den = 1.0 / (troe_n - .14 * (troe_c + logPr));
  troePr = (troe_c + logPr) * troePr_den;
  troe = 1.0 / (1.0 + troePr * troePr);
  F = exp(M_LN10 * logFcent * troe);
  dlogFcentdT =
    log10e / Fcent * (-Fcent1 / 2500 - Fcent2 / 1300 + Fcent3 * 1e+99 * invT2);
  dlogFdcn_fac = 2.0 * logFcent * troe * troe * troePr * troePr_den;
  dlogFdc = -troe_n * dlogFdcn_fac * troePr_den;
  dlogFdn = dlogFdcn_fac * troePr;
  dlogFdlogPr = dlogFdc;
  dlogFdT = dlogFcentdT * (troe - 0.67 * dlogFdc - 1.27 * dlogFdn) +
            dlogFdlogPr * dlogPrdT;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc = refC * (kc_fast ? (inv_ge[18] * ge[20] * ge[2])
                       : exp(-g_RT[2] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[18]) + (h_RT[2] + h_RT[20]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[2] += q;  // H
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (2 - 1) * dcdc_fac;
    J[29] += dqdci; // dwdot[H]/d[H2]
    J[45] -= dqdci; // dwdot[CH3O]/d[H2]
    J[47] += dqdci; // dwdot[CH2O]/d[H2]
    // d()/d[H]
    dqdci = -k_r * sc[20];
    J[56] += dqdci; // dwdot[H]/d[H]
    J[72] -= dqdci; // dwdot[CH3O]/d[H]
    J[74] += dqdci; // dwdot[CH2O]/d[H]
    // d()/d[H2O]
    dqdci = (6 - 1) * dcdc_fac;
    J[137] += dqdci; // dwdot[H]/d[H2O]
    J[153] -= dqdci; // dwdot[CH3O]/d[H2O]
    J[155] += dqdci; // dwdot[CH2O]/d[H2O]
    // d()/d[CO]
    dqdci = (1.5 - 1) * dcdc_fac;
    J[245] += dqdci; // dwdot[H]/d[CO]
    J[261] -= dqdci; // dwdot[CH3O]/d[CO]
    J[263] += dqdci; // dwdot[CH2O]/d[CO]
    // d()/d[CO2]
    dqdci = (2 - 1) * dcdc_fac;
    J[272] += dqdci; // dwdot[H]/d[CO2]
    J[288] -= dqdci; // dwdot[CH3O]/d[CO2]
    J[290] += dqdci; // dwdot[CH2O]/d[CO2]
    // d()/d[CH4]
    dqdci = (2 - 1) * dcdc_fac;
    J[299] += dqdci; // dwdot[H]/d[CH4]
    J[315] -= dqdci; // dwdot[CH3O]/d[CH4]
    J[317] += dqdci; // dwdot[CH2O]/d[CH4]
    // d()/d[CH3O]
    dqdci = +k_f;
    J[488] += dqdci; // dwdot[H]/d[CH3O]
    J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
    J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
    // d()/d[CH2O]
    dqdci = -k_r * sc[2];
    J[542] += dqdci; // dwdot[H]/d[CH2O]
    J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
    J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = 2 * dcdc_fac;
    dqdc[2] = dcdc_fac - k_r * sc[20];
    dqdc[3] = dcdc_fac;
    dqdc[4] = dcdc_fac;
    dqdc[5] = 6 * dcdc_fac;
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac;
    dqdc[9] = 1.5 * dcdc_fac;
    dqdc[10] = 2 * dcdc_fac;
    dqdc[11] = 2 * dcdc_fac;
    dqdc[12] = dcdc_fac;
    dqdc[13] = dcdc_fac;
    dqdc[14] = dcdc_fac;
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac;
    dqdc[18] = dcdc_fac + k_f;
    dqdc[19] = dcdc_fac;
    dqdc[20] = dcdc_fac - k_r * sc[2];
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 2] += dqdc[k];
      J[27 * k + 18] -= dqdc[k];
      J[27 * k + 20] += dqdc[k];
    }
  }
  J[704] += dqdT; // dwdot[H]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 20: CO + O (+M) <=> CO2 (+M)
  // a third-body and pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + sc[1] + (12 - 1) * sc[5] + (1.75 - 1) * sc[9] +
          (3.6 - 1) * sc[10];
  // forward
  phi_f = sc[4] * sc[9];
  k_f = 13620 * exp(-(1199.66853127845) * invT);
  dlnkfdT = (1199.66853127845) * invT2;
  // pressure-fall-off
  k_0 = 1.17e+24 * exp(-2.79 * logT - (2108.98104638758) * invT);
  Pr = 1e-12 * alpha / k_f * k_0;
  fPr = Pr / (1.0 + Pr);
  dlnk0dT = -2.79 * invT + (2108.98104638758) * invT2;
  dlogPrdT = log10e * (dlnk0dT - dlnkfdT);
  dlogfPrdT = dlogPrdT / (1.0 + Pr);
  // Lindemann form
  F = 1.0;
  dlogFdlogPr = 0.0;
  dlogFdT = 0.0;
  // reverse
  phi_r = sc[10];
  Kc = refCinv * (kc_fast ? (inv_ge[9] * inv_ge[4] * ge[10])
                          : exp(g_RT[4] + g_RT[9] - g_RT[10]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[9]) + (h_RT[10]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  Corr = fPr * F;
  q = Corr * q_nocor;
  dlnCorrdT = ln10 * (dlogfPrdT + dlogFdT);
  dqdT = Corr * (dlnkfdT * k_f * phi_f - dkrdT * phi_r) + dlnCorrdT * q;
  // update wdot
  wdot[4] -= q;  // O
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  // for convenience
  k_f *= Corr;
  k_r *= Corr;
  dcdc_fac = q / alpha * (1.0 / (Pr + 1.0) + dlogFdlogPr);
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (2 - 1) * dcdc_fac;
    J[31] -= dqdci; // dwdot[O]/d[H2]
    J[36] -= dqdci; // dwdot[CO]/d[H2]
    J[37] += dqdci; // dwdot[CO2]/d[H2]
    // d()/d[O]
    dqdci = +k_f * sc[9];
    J[112] -= dqdci; // dwdot[O]/d[O]
    J[117] -= dqdci; // dwdot[CO]/d[O]
    J[118] += dqdci; // dwdot[CO2]/d[O]
    // d()/d[H2O]
    dqdci = (12 - 1) * dcdc_fac;
    J[139] -= dqdci; // dwdot[O]/d[H2O]
    J[144] -= dqdci; // dwdot[CO]/d[H2O]
    J[145] += dqdci; // dwdot[CO2]/d[H2O]
    // d()/d[CO]
    dqdci = (1.75 - 1) * dcdc_fac + k_f * sc[4];
    J[247] -= dqdci; // dwdot[O]/d[CO]
    J[252] -= dqdci; // dwdot[CO]/d[CO]
    J[253] += dqdci; // dwdot[CO2]/d[CO]
    // d()/d[CO2]
    dqdci = (3.6 - 1) * dcdc_fac - k_r;
    J[274] -= dqdci; // dwdot[O]/d[CO2]
    J[279] -= dqdci; // dwdot[CO]/d[CO2]
    J[280] += dqdci; // dwdot[CO2]/d[CO2]
  } else {
    dqdc[0] = dcdc_fac;
    dqdc[1] = 2 * dcdc_fac;
    dqdc[2] = dcdc_fac;
    dqdc[3] = dcdc_fac;
    dqdc[4] = dcdc_fac + k_f * sc[9];
    dqdc[5] = 12 * dcdc_fac;
    dqdc[6] = dcdc_fac;
    dqdc[7] = dcdc_fac;
    dqdc[8] = dcdc_fac;
    dqdc[9] = 1.75 * dcdc_fac + k_f * sc[4];
    dqdc[10] = 3.6 * dcdc_fac - k_r;
    dqdc[11] = dcdc_fac;
    dqdc[12] = dcdc_fac;
    dqdc[13] = dcdc_fac;
    dqdc[14] = dcdc_fac;
    dqdc[15] = dcdc_fac;
    dqdc[16] = dcdc_fac;
    dqdc[17] = dcdc_fac;
    dqdc[18] = dcdc_fac;
    dqdc[19] = dcdc_fac;
    dqdc[20] = dcdc_fac;
    dqdc[21] = dcdc_fac;
    dqdc[22] = dcdc_fac;
    dqdc[23] = dcdc_fac;
    dqdc[24] = dcdc_fac;
    dqdc[25] = dcdc_fac;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 4] -= dqdc[k];
      J[27 * k + 9] -= dqdc[k];
      J[27 * k + 10] += dqdc[k];
    }
  }
  J[706] -= dqdT; // dwdot[O]/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT

  // reaction 3: H + OH + M <=> H2O + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (0.73 - 1) * sc[1] + (3.65 - 1) * sc[5] + sc[11];
  // forward
  phi_f = sc[2] * sc[6];
  k_f = 35000000000 * exp(-2 * logT);
  dlnkfdT = -2 * invT;
  // reverse
  phi_r = sc[5];
  Kc = refCinv * (kc_fast ? (inv_ge[6] * inv_ge[2] * ge[5])
                          : exp(g_RT[2] - g_RT[5] + g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[6]) + (h_RT[5]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (0.73 - 1) * q_nocor;
    J[29] -= dqdci; // dwdot[H]/d[H2]
    J[32] += dqdci; // dwdot[H2O]/d[H2]
    J[33] -= dqdci; // dwdot[OH]/d[H2]
    // d()/d[H]
    dqdci = +k_f * sc[6];
    J[56] -= dqdci; // dwdot[H]/d[H]
    J[59] += dqdci; // dwdot[H2O]/d[H]
    J[60] -= dqdci; // dwdot[OH]/d[H]
    // d()/d[H2O]
    dqdci = (3.65 - 1) * q_nocor - k_r;
    J[137] -= dqdci; // dwdot[H]/d[H2O]
    J[140] += dqdci; // dwdot[H2O]/d[H2O]
    J[141] -= dqdci; // dwdot[OH]/d[H2O]
    // d()/d[OH]
    dqdci = +k_f * sc[2];
    J[164] -= dqdci; // dwdot[H]/d[OH]
    J[167] += dqdci; // dwdot[H2O]/d[OH]
    J[168] -= dqdci; // dwdot[OH]/d[OH]
    // d()/d[CH4]
    dqdci = (2 - 1) * q_nocor;
    J[299] -= dqdci; // dwdot[H]/d[CH4]
    J[302] += dqdci; // dwdot[H2O]/d[CH4]
    J[303] -= dqdci; // dwdot[OH]/d[CH4]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = 0.73 * q_nocor;
    dqdc[2] = q_nocor + k_f * sc[6];
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = 3.65 * q_nocor - k_r;
    dqdc[6] = q_nocor + k_f * sc[2];
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = q_nocor;
    dqdc[10] = q_nocor;
    dqdc[11] = 2 * q_nocor;
    dqdc[12] = q_nocor;
    dqdc[13] = q_nocor;
    dqdc[14] = q_nocor;
    dqdc[15] = q_nocor;
    dqdc[16] = q_nocor;
    dqdc[17] = q_nocor;
    dqdc[18] = q_nocor;
    dqdc[19] = q_nocor;
    dqdc[20] = q_nocor;
    dqdc[21] = q_nocor;
    dqdc[22] = q_nocor;
    dqdc[23] = q_nocor;
    dqdc[24] = q_nocor;
    dqdc[25] = q_nocor;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 2] -= dqdc[k];
      J[27 * k + 5] += dqdc[k];
      J[27 * k + 6] -= dqdc[k];
    }
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT

  // reaction 5: H + O + M <=> OH + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture + (2.5 - 1) * sc[1] + (12 - 1) * sc[5] + (1.5 - 1) * sc[9] +
          sc[10] + sc[11];
  // forward
  phi_f = sc[2] * sc[4];
  k_f = 4714000 * exp(-1 * logT);
  dlnkfdT = -1 * invT;
  // reverse
  phi_r = sc[6];
  Kc = refCinv * (kc_fast ? (ge[6] * inv_ge[2] * inv_ge[4])
                          : exp(g_RT[2] + g_RT[4] - g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[4]) + (h_RT[6]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[4] -= q; // O
  wdot[6] += q; // OH
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (2.5 - 1) * q_nocor;
    J[29] -= dqdci; // dwdot[H]/d[H2]
    J[31] -= dqdci; // dwdot[O]/d[H2]
    J[33] += dqdci; // dwdot[OH]/d[H2]
    // d()/d[H]
    dqdci = +k_f * sc[4];
    J[56] -= dqdci; // dwdot[H]/d[H]
    J[58] -= dqdci; // dwdot[O]/d[H]
    J[60] += dqdci; // dwdot[OH]/d[H]
    // d()/d[O]
    dqdci = +k_f * sc[2];
    J[110] -= dqdci; // dwdot[H]/d[O]
    J[112] -= dqdci; // dwdot[O]/d[O]
    J[114] += dqdci; // dwdot[OH]/d[O]
    // d()/d[H2O]
    dqdci = (12 - 1) * q_nocor;
    J[137] -= dqdci; // dwdot[H]/d[H2O]
    J[139] -= dqdci; // dwdot[O]/d[H2O]
    J[141] += dqdci; // dwdot[OH]/d[H2O]
    // d()/d[OH]
    dqdci = -k_r;
    J[164] -= dqdci; // dwdot[H]/d[OH]
    J[166] -= dqdci; // dwdot[O]/d[OH]
    J[168] += dqdci; // dwdot[OH]/d[OH]
    // d()/d[CO]
    dqdci = (1.5 - 1) * q_nocor;
    J[245] -= dqdci; // dwdot[H]/d[CO]
    J[247] -= dqdci; // dwdot[O]/d[CO]
    J[249] += dqdci; // dwdot[OH]/d[CO]
    // d()/d[CO2]
    dqdci = (2 - 1) * q_nocor;
    J[272] -= dqdci; // dwdot[H]/d[CO2]
    J[274] -= dqdci; // dwdot[O]/d[CO2]
    J[276] += dqdci; // dwdot[OH]/d[CO2]
    // d()/d[CH4]
    dqdci = (2 - 1) * q_nocor;
    J[299] -= dqdci; // dwdot[H]/d[CH4]
    J[301] -= dqdci; // dwdot[O]/d[CH4]
    J[303] += dqdci; // dwdot[OH]/d[CH4]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = 2.5 * q_nocor;
    dqdc[2] = q_nocor + k_f * sc[4];
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor + k_f * sc[2];
    dqdc[5] = 12 * q_nocor;
    dqdc[6] = q_nocor - k_r;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = 1.5 * q_nocor;
    dqdc[10] = 2 * q_nocor;
    dqdc[11] = 2 * q_nocor;
    dqdc[12] = q_nocor;
    dqdc[13] = q_nocor;
    dqdc[14] = q_nocor;
    dqdc[15] = q_nocor;
    dqdc[16] = q_nocor;
    dqdc[17] = q_nocor;
    dqdc[18] = q_nocor;
    dqdc[19] = q_nocor;
    dqdc[20] = q_nocor;
    dqdc[21] = q_nocor;
    dqdc[22] = q_nocor;
    dqdc[23] = q_nocor;
    dqdc[24] = q_nocor;
    dqdc[25] = q_nocor;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 2] -= dqdc[k];
      J[27 * k + 4] -= dqdc[k];
      J[27 * k + 6] += dqdc[k];
    }
  }
  J[704] -= dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT

  // reaction 97: HCO + M <=> CO + H + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha =
    mixture + sc[1] + (6 - 1) * sc[5] + (1.5 - 1) * sc[9] + sc[10] + sc[11];
  // forward
  phi_f = sc[21];
  k_f = 570000 * exp(0.66 * logT - (7482.83182051618) * invT);
  dlnkfdT = 0.66 * invT + (7482.83182051618) * invT2;
  // reverse
  phi_r = sc[2] * sc[9];
  Kc = refC * (kc_fast ? (inv_ge[21] * ge[9] * ge[2])
                       : exp(-g_RT[2] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[21]) + (h_RT[2] + h_RT[9]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[H2]
    dqdci = (2 - 1) * q_nocor;
    J[29] += dqdci; // dwdot[H]/d[H2]
    J[36] += dqdci; // dwdot[CO]/d[H2]
    J[48] -= dqdci; // dwdot[HCO]/d[H2]
    // d()/d[H]
    dqdci = -k_r * sc[9];
    J[56] += dqdci; // dwdot[H]/d[H]
    J[63] += dqdci; // dwdot[CO]/d[H]
    J[75] -= dqdci; // dwdot[HCO]/d[H]
    // d()/d[H2O]
    dqdci = (6 - 1) * q_nocor;
    J[137] += dqdci; // dwdot[H]/d[H2O]
    J[144] += dqdci; // dwdot[CO]/d[H2O]
    J[156] -= dqdci; // dwdot[HCO]/d[H2O]
    // d()/d[CO]
    dqdci = (1.5 - 1) * q_nocor - k_r * sc[2];
    J[245] += dqdci; // dwdot[H]/d[CO]
    J[252] += dqdci; // dwdot[CO]/d[CO]
    J[264] -= dqdci; // dwdot[HCO]/d[CO]
    // d()/d[CO2]
    dqdci = (2 - 1) * q_nocor;
    J[272] += dqdci; // dwdot[H]/d[CO2]
    J[279] += dqdci; // dwdot[CO]/d[CO2]
    J[291] -= dqdci; // dwdot[HCO]/d[CO2]
    // d()/d[CH4]
    dqdci = (2 - 1) * q_nocor;
    J[299] += dqdci; // dwdot[H]/d[CH4]
    J[306] += dqdci; // dwdot[CO]/d[CH4]
    J[318] -= dqdci; // dwdot[HCO]/d[CH4]
    // d()/d[HCO]
    dqdci = +k_f;
    J[569] += dqdci; // dwdot[H]/d[HCO]
    J[576] += dqdci; // dwdot[CO]/d[HCO]
    J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = 2 * q_nocor;
    dqdc[2] = q_nocor - k_r * sc[9];
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor;
    dqdc[5] = 6 * q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = 1.5 * q_nocor - k_r * sc[2];
    dqdc[10] = 2 * q_nocor;
    dqdc[11] = 2 * q_nocor;
    dqdc[12] = q_nocor;
    dqdc[13] = q_nocor;
    dqdc[14] = q_nocor;
    dqdc[15] = q_nocor;
    dqdc[16] = q_nocor;
    dqdc[17] = q_nocor;
    dqdc[18] = q_nocor;
    dqdc[19] = q_nocor;
    dqdc[20] = q_nocor;
    dqdc[21] = q_nocor + k_f;
    dqdc[22] = q_nocor;
    dqdc[23] = q_nocor;
    dqdc[24] = q_nocor;
    dqdc[25] = q_nocor;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 2] += dqdc[k];
      J[27 * k + 9] += dqdc[k];
      J[27 * k + 21] -= dqdc[k];
    }
  }
  J[704] += dqdT; // dwdot[H]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 108: O + OH + M <=> HO2 + M
  // a third-body and non-pressure-fall-off reaction
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[4] * sc[6];
  k_f = 10000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[8];
  Kc = refCinv * (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[4])
                          : exp(g_RT[4] + g_RT[6] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[6]) + (h_RT[8]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q_nocor = k_f * phi_f - k_r * phi_r;
  q = alpha * q_nocor;
  dqdT = alpha * (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q; // O
  wdot[6] -= q; // OH
  wdot[8] += q; // HO2
  // for convenience
  k_f *= alpha;
  k_r *= alpha;
  if (consP == 1) {
    // d()/d[O]
    dqdci = +k_f * sc[6];
    J[112] -= dqdci; // dwdot[O]/d[O]
    J[114] -= dqdci; // dwdot[OH]/d[O]
    J[116] += dqdci; // dwdot[HO2]/d[O]
    // d()/d[OH]
    dqdci = +k_f * sc[4];
    J[166] -= dqdci; // dwdot[O]/d[OH]
    J[168] -= dqdci; // dwdot[OH]/d[OH]
    J[170] += dqdci; // dwdot[HO2]/d[OH]
    // d()/d[HO2]
    dqdci = -k_r;
    J[220] -= dqdci; // dwdot[O]/d[HO2]
    J[222] -= dqdci; // dwdot[OH]/d[HO2]
    J[224] += dqdci; // dwdot[HO2]/d[HO2]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor + k_f * sc[6];
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor + k_f * sc[4];
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor - k_r;
    dqdc[9] = q_nocor;
    dqdc[10] = q_nocor;
    dqdc[11] = q_nocor;
    dqdc[12] = q_nocor;
    dqdc[13] = q_nocor;
    dqdc[14] = q_nocor;
    dqdc[15] = q_nocor;
    dqdc[16] = q_nocor;
    dqdc[17] = q_nocor;
    dqdc[18] = q_nocor;
    dqdc[19] = q_nocor;
    dqdc[20] = q_nocor;
    dqdc[21] = q_nocor;
    dqdc[22] = q_nocor;
    dqdc[23] = q_nocor;
    dqdc[24] = q_nocor;
    dqdc[25] = q_nocor;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 4] -= dqdc[k];
      J[27 * k + 6] -= dqdc[k];
      J[27 * k + 8] += dqdc[k];
    }
  }
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 120: CH3 + O + M => CH3O + M
  // a third-body and non-pressure-fall-off reaction
  // Irreversible reaction in _ajac_reaction may not work
  // 3-body correction factor
  alpha = mixture;
  // forward
  phi_f = sc[4] * sc[12];
  k_f = 50000;
  dlnkfdT = 0.0;
  // rate of progress
  q_nocor = k_f * phi_f;
  q = alpha * q_nocor;
  dqdT = alpha * dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[4] -= q;  // O
  wdot[12] -= q; // CH3
  wdot[18] += q; // CH3O
  // for convenience
  k_f *= alpha;
  if (consP == 1) {
    // d()/d[O]
    dqdci = +k_f * sc[12];
    J[112] -= dqdci; // dwdot[O]/d[O]
    J[120] -= dqdci; // dwdot[CH3]/d[O]
    J[126] += dqdci; // dwdot[CH3O]/d[O]
    // d()/d[CH3]
    dqdci = +k_f * sc[4];
    J[328] -= dqdci; // dwdot[O]/d[CH3]
    J[336] -= dqdci; // dwdot[CH3]/d[CH3]
    J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  } else {
    dqdc[0] = q_nocor;
    dqdc[1] = q_nocor;
    dqdc[2] = q_nocor;
    dqdc[3] = q_nocor;
    dqdc[4] = q_nocor + k_f * sc[12];
    dqdc[5] = q_nocor;
    dqdc[6] = q_nocor;
    dqdc[7] = q_nocor;
    dqdc[8] = q_nocor;
    dqdc[9] = q_nocor;
    dqdc[10] = q_nocor;
    dqdc[11] = q_nocor;
    dqdc[12] = q_nocor + k_f * sc[4];
    dqdc[13] = q_nocor;
    dqdc[14] = q_nocor;
    dqdc[15] = q_nocor;
    dqdc[16] = q_nocor;
    dqdc[17] = q_nocor;
    dqdc[18] = q_nocor;
    dqdc[19] = q_nocor;
    dqdc[20] = q_nocor;
    dqdc[21] = q_nocor;
    dqdc[22] = q_nocor;
    dqdc[23] = q_nocor;
    dqdc[24] = q_nocor;
    dqdc[25] = q_nocor;
    for (int k = 0; k < 26; k++) {
      J[27 * k + 4] -= dqdc[k];
      J[27 * k + 12] -= dqdc[k];
      J[27 * k + 18] += dqdc[k];
    }
  }
  J[706] -= dqdT; // dwdot[O]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 131: E + O2 + O2 <=> O2- + O2
  // a third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[3] * sc[3]) * sc[25];
  k_f = 1520000000 * exp(-1 * logT - (600.156324305341) * invT);
  dlnkfdT = -1 * invT + (600.156324305341) * invT2;
  // reverse
  phi_r = sc[3] * sc[24];
  Kc = refCinv * (kc_fast ? (inv_ge[25] * inv_ge[3] * ge[24])
                          : exp(g_RT[3] - g_RT[24] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(2.000000 * h_RT[3] + h_RT[25]) + (h_RT[3] + h_RT[24]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[24] += q; // O2-
  wdot[25] -= q; // E
  // d()/d[O2]
  dqdci = +k_f * 2.000000 * sc[3] * sc[25] - k_r * sc[24];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[105] += dqdci; // dwdot[O2-]/d[O2]
  J[106] -= dqdci; // dwdot[E]/d[O2]
  // d()/d[O2-]
  dqdci = -k_r * sc[3];
  J[651] -= dqdci; // dwdot[O2]/d[O2-]
  J[672] += dqdci; // dwdot[O2-]/d[O2-]
  J[673] -= dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = +k_f * sc[3] * sc[3];
  J[678] -= dqdci; // dwdot[O2]/d[E]
  J[699] += dqdci; // dwdot[O2-]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[726] += dqdT; // dwdot[O2-]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 132: E + O2 + H2O <=> O2- + H2O
  // a third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[5] * sc[25];
  k_f = 5080000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[24];
  Kc = refCinv * (kc_fast ? (inv_ge[25] * inv_ge[3] * ge[24])
                          : exp(g_RT[3] - g_RT[24] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[3] + h_RT[5] + h_RT[25]) + (h_RT[5] + h_RT[24]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[24] += q; // O2-
  wdot[25] -= q; // E
  // d()/d[O2]
  dqdci = +k_f * sc[5] * sc[25];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[105] += dqdci; // dwdot[O2-]/d[O2]
  J[106] -= dqdci; // dwdot[E]/d[O2]
  // d()/d[H2O]
  dqdci = +k_f * sc[3] * sc[25] - k_r * sc[24];
  J[138] -= dqdci; // dwdot[O2]/d[H2O]
  J[159] += dqdci; // dwdot[O2-]/d[H2O]
  J[160] -= dqdci; // dwdot[E]/d[H2O]
  // d()/d[O2-]
  dqdci = -k_r * sc[5];
  J[651] -= dqdci; // dwdot[O2]/d[O2-]
  J[672] += dqdci; // dwdot[O2-]/d[O2-]
  J[673] -= dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = +k_f * sc[3] * sc[5];
  J[678] -= dqdci; // dwdot[O2]/d[E]
  J[699] += dqdci; // dwdot[O2-]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[726] += dqdT; // dwdot[O2-]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 133: E + O2 + N2 <=> O2- + N2
  // a third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[3] * sc[25];
  k_f = 3590000000 * exp(-2 * logT - (69.7558942138502) * invT);
  dlnkfdT = -2 * invT + (69.7558942138502) * invT2;
  // reverse
  phi_r = sc[0] * sc[24];
  Kc = refCinv * (kc_fast ? (inv_ge[25] * inv_ge[3] * ge[24])
                          : exp(g_RT[3] - g_RT[24] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[0] + h_RT[3] + h_RT[25]) + (h_RT[0] + h_RT[24]) + 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[24] += q; // O2-
  wdot[25] -= q; // E
  // d()/d[N2]
  dqdci = +k_f * sc[3] * sc[25] - k_r * sc[24];
  J[3] -= dqdci;  // dwdot[O2]/d[N2]
  J[24] += dqdci; // dwdot[O2-]/d[N2]
  J[25] -= dqdci; // dwdot[E]/d[N2]
  // d()/d[O2]
  dqdci = +k_f * sc[0] * sc[25];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[105] += dqdci; // dwdot[O2-]/d[O2]
  J[106] -= dqdci; // dwdot[E]/d[O2]
  // d()/d[O2-]
  dqdci = -k_r * sc[0];
  J[651] -= dqdci; // dwdot[O2]/d[O2-]
  J[672] += dqdci; // dwdot[O2-]/d[O2-]
  J[673] -= dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = +k_f * sc[0] * sc[3];
  J[678] -= dqdci; // dwdot[O2]/d[E]
  J[699] += dqdci; // dwdot[O2-]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[726] += dqdT; // dwdot[O2-]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 0: H2 + O <=> H + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[4];
  k_f = 0.0508 * exp(2.67 * logT - (3166.2392612433) * invT);
  dlnkfdT = 2.67 * invT + (3166.2392612433) * invT2;
  // reverse
  phi_r = sc[2] * sc[6];
  Kc =
    (kc_fast ? (inv_ge[1] * ge[6] * inv_ge[4] * ge[2])
             : exp(g_RT[1] - g_RT[2] + g_RT[4] - g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[4]) + (h_RT[2] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H2
  wdot[2] += q; // H
  wdot[4] -= q; // O
  wdot[6] += q; // OH
  // d()/d[H2]
  dqdci = +k_f * sc[4];
  J[28] -= dqdci; // dwdot[H2]/d[H2]
  J[29] += dqdci; // dwdot[H]/d[H2]
  J[31] -= dqdci; // dwdot[O]/d[H2]
  J[33] += dqdci; // dwdot[OH]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[6];
  J[55] -= dqdci; // dwdot[H2]/d[H]
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[60] += dqdci; // dwdot[OH]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[1];
  J[109] -= dqdci; // dwdot[H2]/d[O]
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[2];
  J[163] -= dqdci; // dwdot[H2]/d[OH]
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  // d()/dT
  J[703] -= dqdT; // dwdot[H2]/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT

  // reaction 1: H2 + OH <=> H + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[6];
  k_f = 43800000 * exp(-(3517.48449397499) * invT);
  dlnkfdT = (3517.48449397499) * invT2;
  // reverse
  phi_r = sc[2] * sc[5];
  Kc =
    (kc_fast ? (inv_ge[1] * inv_ge[6] * ge[2] * ge[5])
             : exp(g_RT[1] - g_RT[2] - g_RT[5] + g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[6]) + (h_RT[2] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q; // H2
  wdot[2] += q; // H
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  // d()/d[H2]
  dqdci = +k_f * sc[6];
  J[28] -= dqdci; // dwdot[H2]/d[H2]
  J[29] += dqdci; // dwdot[H]/d[H2]
  J[32] += dqdci; // dwdot[H2O]/d[H2]
  J[33] -= dqdci; // dwdot[OH]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[5];
  J[55] -= dqdci; // dwdot[H2]/d[H]
  J[56] += dqdci; // dwdot[H]/d[H]
  J[59] += dqdci; // dwdot[H2O]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  // d()/d[H2O]
  dqdci = -k_r * sc[2];
  J[136] -= dqdci; // dwdot[H2]/d[H2O]
  J[137] += dqdci; // dwdot[H]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[1];
  J[163] -= dqdci; // dwdot[H2]/d[OH]
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  // d()/dT
  J[703] -= dqdT; // dwdot[H2]/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT

  // reaction 2: H + O2 <=> O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[3];
  k_f = 114000000 * exp(-(7692.16995349094) * invT);
  dlnkfdT = (7692.16995349094) * invT2;
  // reverse
  phi_r = sc[4] * sc[6];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[3] * inv_ge[2] * ge[4])
             : exp(g_RT[2] + g_RT[3] - g_RT[4] - g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[3]) + (h_RT[4] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[3] -= q; // O2
  wdot[4] += q; // O
  wdot[6] += q; // OH
  // d()/d[H]
  dqdci = +k_f * sc[3];
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[57] -= dqdci; // dwdot[O2]/d[H]
  J[58] += dqdci; // dwdot[O]/d[H]
  J[60] += dqdci; // dwdot[OH]/d[H]
  // d()/d[O2]
  dqdci = +k_f * sc[2];
  J[83] -= dqdci; // dwdot[H]/d[O2]
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[85] += dqdci; // dwdot[O]/d[O2]
  J[87] += dqdci; // dwdot[OH]/d[O2]
  // d()/d[O]
  dqdci = -k_r * sc[6];
  J[110] -= dqdci; // dwdot[H]/d[O]
  J[111] -= dqdci; // dwdot[O2]/d[O]
  J[112] += dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[4];
  J[164] -= dqdci; // dwdot[H]/d[OH]
  J[165] -= dqdci; // dwdot[O2]/d[OH]
  J[166] += dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  // d()/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[706] += dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT

  // reaction 4: H2O + O <=> 2 OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[5];
  k_f = 67 * exp(1.704 * logT - (7541.60752708217) * invT);
  dlnkfdT = 1.704 * invT + (7541.60752708217) * invT2;
  // reverse
  phi_r = (sc[6] * sc[6]);
  Kc =
    (kc_fast ? (ge[6] * ge[6] * inv_ge[4] * inv_ge[5])
             : exp(g_RT[4] + g_RT[5] - 2.000000 * g_RT[6]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[5]) + (2.000000 * h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;     // O
  wdot[5] -= q;     // H2O
  wdot[6] += 2 * q; // OH
  // d()/d[O]
  dqdci = +k_f * sc[5];
  J[112] -= dqdci;     // dwdot[O]/d[O]
  J[113] -= dqdci;     // dwdot[H2O]/d[O]
  J[114] += 2 * dqdci; // dwdot[OH]/d[O]
  // d()/d[H2O]
  dqdci = +k_f * sc[4];
  J[139] -= dqdci;     // dwdot[O]/d[H2O]
  J[140] -= dqdci;     // dwdot[H2O]/d[H2O]
  J[141] += 2 * dqdci; // dwdot[OH]/d[H2O]
  // d()/d[OH]
  dqdci = -k_r * 2.000000 * sc[6];
  J[166] -= dqdci;     // dwdot[O]/d[OH]
  J[167] -= dqdci;     // dwdot[H2O]/d[OH]
  J[168] += 2 * dqdci; // dwdot[OH]/d[OH]
  // d()/dT
  J[706] -= dqdT;     // dwdot[O]/dT
  J[707] -= dqdT;     // dwdot[H2O]/dT
  J[708] += 2 * dqdT; // dwdot[OH]/dT

  // reaction 7: H + H2O2 <=> H2O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[7];
  k_f = 24100000 * exp(-(1997.77016324474) * invT);
  dlnkfdT = (1997.77016324474) * invT2;
  // reverse
  phi_r = sc[5] * sc[6];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[7] * inv_ge[2] * ge[5])
             : exp(g_RT[2] - g_RT[5] - g_RT[6] + g_RT[7]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[5] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q; // H
  wdot[5] += q; // H2O
  wdot[6] += q; // OH
  wdot[7] -= q; // H2O2
  // d()/d[H]
  dqdci = +k_f * sc[7];
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[59] += dqdci; // dwdot[H2O]/d[H]
  J[60] += dqdci; // dwdot[OH]/d[H]
  J[61] -= dqdci; // dwdot[H2O2]/d[H]
  // d()/d[H2O]
  dqdci = -k_r * sc[6];
  J[137] -= dqdci; // dwdot[H]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] += dqdci; // dwdot[OH]/d[H2O]
  J[142] -= dqdci; // dwdot[H2O2]/d[H2O]
  // d()/d[OH]
  dqdci = -k_r * sc[5];
  J[164] -= dqdci; // dwdot[H]/d[OH]
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[169] -= dqdci; // dwdot[H2O2]/d[OH]
  // d()/d[H2O2]
  dqdci = +k_f * sc[2];
  J[191] -= dqdci; // dwdot[H]/d[H2O2]
  J[194] += dqdci; // dwdot[H2O]/d[H2O2]
  J[195] += dqdci; // dwdot[OH]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  // d()/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT

  // reaction 8: H + H2O2 <=> H2 + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[7];
  k_f = 21500 * exp(1 * logT - (3019.29999482832) * invT);
  dlnkfdT = 1 * invT + (3019.29999482832) * invT2;
  // reverse
  phi_r = sc[1] * sc[8];
  Kc =
    (kc_fast ? (ge[1] * ge[8] * inv_ge[7] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[7] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[7]) + (h_RT[1] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[7] -= q; // H2O2
  wdot[8] += q; // HO2
  // d()/d[H2]
  dqdci = -k_r * sc[8];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[34] -= dqdci; // dwdot[H2O2]/d[H2]
  J[35] += dqdci; // dwdot[HO2]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[7];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[61] -= dqdci; // dwdot[H2O2]/d[H]
  J[62] += dqdci; // dwdot[HO2]/d[H]
  // d()/d[H2O2]
  dqdci = +k_f * sc[2];
  J[190] += dqdci; // dwdot[H2]/d[H2O2]
  J[191] -= dqdci; // dwdot[H]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] += dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[1];
  J[217] += dqdci; // dwdot[H2]/d[HO2]
  J[218] -= dqdci; // dwdot[H]/d[HO2]
  J[223] -= dqdci; // dwdot[H2O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 9: H2O2 + O <=> HO2 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[7];
  k_f = 9.55 * exp(2 * logT - (1997.77016324474) * invT);
  dlnkfdT = 2 * invT + (1997.77016324474) * invT2;
  // reverse
  phi_r = sc[6] * sc[8];
  Kc =
    (kc_fast ? (ge[6] * ge[8] * inv_ge[7] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[7] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[7]) + (h_RT[6] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q; // O
  wdot[6] += q; // OH
  wdot[7] -= q; // H2O2
  wdot[8] += q; // HO2
  // d()/d[O]
  dqdci = +k_f * sc[7];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[115] -= dqdci; // dwdot[H2O2]/d[O]
  J[116] += dqdci; // dwdot[HO2]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[8];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[169] -= dqdci; // dwdot[H2O2]/d[OH]
  J[170] += dqdci; // dwdot[HO2]/d[OH]
  // d()/d[H2O2]
  dqdci = +k_f * sc[4];
  J[193] -= dqdci; // dwdot[O]/d[H2O2]
  J[195] += dqdci; // dwdot[OH]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] += dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[6];
  J[220] -= dqdci; // dwdot[O]/d[HO2]
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[223] -= dqdci; // dwdot[H2O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 10: H2O2 + OH <=> H2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[7];
  k_f = 1740000 * exp(-(160.022899725901) * invT);
  dlnkfdT = (160.022899725901) * invT2;
  // reverse
  phi_r = sc[5] * sc[8];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[7] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[7] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[7]) + (h_RT[5] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  wdot[7] -= q; // H2O2
  wdot[8] += q; // HO2
  // d()/d[H2O]
  dqdci = -k_r * sc[8];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[142] -= dqdci; // dwdot[H2O2]/d[H2O]
  J[143] += dqdci; // dwdot[HO2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[7];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[169] -= dqdci; // dwdot[H2O2]/d[OH]
  J[170] += dqdci; // dwdot[HO2]/d[OH]
  // d()/d[H2O2]
  dqdci = +k_f * sc[6];
  J[194] += dqdci; // dwdot[H2O]/d[H2O2]
  J[195] -= dqdci; // dwdot[OH]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] += dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[5];
  J[221] += dqdci; // dwdot[H2O]/d[HO2]
  J[222] -= dqdci; // dwdot[OH]/d[HO2]
  J[223] -= dqdci; // dwdot[H2O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 11: H2O2 + OH <=> H2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[7];
  k_f = 75900000 * exp(-(3657.88194373451) * invT);
  dlnkfdT = (3657.88194373451) * invT2;
  // reverse
  phi_r = sc[5] * sc[8];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[8] * inv_ge[7] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[7] - g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[7]) + (h_RT[5] + h_RT[8]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  wdot[7] -= q; // H2O2
  wdot[8] += q; // HO2
  // d()/d[H2O]
  dqdci = -k_r * sc[8];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[142] -= dqdci; // dwdot[H2O2]/d[H2O]
  J[143] += dqdci; // dwdot[HO2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[7];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[169] -= dqdci; // dwdot[H2O2]/d[OH]
  J[170] += dqdci; // dwdot[HO2]/d[OH]
  // d()/d[H2O2]
  dqdci = +k_f * sc[6];
  J[194] += dqdci; // dwdot[H2O]/d[H2O2]
  J[195] -= dqdci; // dwdot[OH]/d[H2O2]
  J[196] -= dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] += dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[5];
  J[221] += dqdci; // dwdot[H2O]/d[HO2]
  J[222] -= dqdci; // dwdot[OH]/d[HO2]
  J[223] -= dqdci; // dwdot[H2O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[709] -= dqdT; // dwdot[H2O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT

  // reaction 12: H + HO2 <=> 2 OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 70790000 * exp(-(148.448916412392) * invT);
  dlnkfdT = (148.448916412392) * invT2;
  // reverse
  phi_r = (sc[6] * sc[6]);
  Kc =
    (kc_fast ? (ge[6] * inv_ge[8] * ge[6] * inv_ge[2])
             : exp(g_RT[2] - 2.000000 * g_RT[6] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (2.000000 * h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q;     // H
  wdot[6] += 2 * q; // OH
  wdot[8] -= q;     // HO2
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[56] -= dqdci;     // dwdot[H]/d[H]
  J[60] += 2 * dqdci; // dwdot[OH]/d[H]
  J[62] -= dqdci;     // dwdot[HO2]/d[H]
  // d()/d[OH]
  dqdci = -k_r * 2.000000 * sc[6];
  J[164] -= dqdci;     // dwdot[H]/d[OH]
  J[168] += 2 * dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci;     // dwdot[HO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[2];
  J[218] -= dqdci;     // dwdot[H]/d[HO2]
  J[222] += 2 * dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci;     // dwdot[HO2]/d[HO2]
  // d()/dT
  J[704] -= dqdT;     // dwdot[H]/dT
  J[708] += 2 * dqdT; // dwdot[OH]/dT
  J[710] -= dqdT;     // dwdot[HO2]/dT

  // reaction 13: H + HO2 <=> H2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[8];
  k_f = 11402 * exp(1.083 * logT - (278.671325189338) * invT);
  dlnkfdT = 1.083 * invT + (278.671325189338) * invT2;
  // reverse
  phi_r = sc[1] * sc[3];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[8] * ge[3] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] - g_RT[3] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[8]) + (h_RT[1] + h_RT[3]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q; // H2
  wdot[2] -= q; // H
  wdot[3] += q; // O2
  wdot[8] -= q; // HO2
  // d()/d[H2]
  dqdci = -k_r * sc[3];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[30] += dqdci; // dwdot[O2]/d[H2]
  J[35] -= dqdci; // dwdot[HO2]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[8];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[57] += dqdci; // dwdot[O2]/d[H]
  J[62] -= dqdci; // dwdot[HO2]/d[H]
  // d()/d[O2]
  dqdci = -k_r * sc[1];
  J[82] += dqdci; // dwdot[H2]/d[O2]
  J[83] -= dqdci; // dwdot[H]/d[O2]
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[2];
  J[217] += dqdci; // dwdot[H2]/d[HO2]
  J[218] -= dqdci; // dwdot[H]/d[HO2]
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT

  // reaction 14: HO2 + O <=> O2 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[8];
  k_f = 32500000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[3] * sc[6];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[8] * ge[3] * inv_ge[4])
             : exp(-g_RT[3] + g_RT[4] - g_RT[6] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[8]) + (h_RT[3] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q; // O2
  wdot[4] -= q; // O
  wdot[6] += q; // OH
  wdot[8] -= q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[6];
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[85] -= dqdci; // dwdot[O]/d[O2]
  J[87] += dqdci; // dwdot[OH]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  // d()/d[O]
  dqdci = +k_f * sc[8];
  J[111] += dqdci; // dwdot[O2]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[116] -= dqdci; // dwdot[HO2]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[3];
  J[165] += dqdci; // dwdot[O2]/d[OH]
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[4];
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[220] -= dqdci; // dwdot[O]/d[HO2]
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT

  // reaction 15: HO2 + OH <=> H2O + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[8];
  k_f = 7000000 * exp(-(-549.995687057926) * invT);
  dlnkfdT = (-549.995687057926) * invT2;
  // reverse
  phi_r = sc[3] * sc[5];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[3] * inv_ge[8] * ge[5])
             : exp(-g_RT[3] - g_RT[5] + g_RT[6] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[8]) + (h_RT[3] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q; // O2
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  wdot[8] -= q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[5];
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[86] += dqdci; // dwdot[H2O]/d[O2]
  J[87] -= dqdci; // dwdot[OH]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  // d()/d[H2O]
  dqdci = -k_r * sc[3];
  J[138] += dqdci; // dwdot[O2]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[143] -= dqdci; // dwdot[HO2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[8];
  J[165] += dqdci; // dwdot[O2]/d[OH]
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[6];
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[221] += dqdci; // dwdot[H2O]/d[HO2]
  J[222] -= dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT

  // reaction 16: HO2 + OH <=> H2O + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[8];
  k_f = 450000000 * exp(-(5499.95687057926) * invT);
  dlnkfdT = (5499.95687057926) * invT2;
  // reverse
  phi_r = sc[3] * sc[5];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[3] * inv_ge[8] * ge[5])
             : exp(-g_RT[3] - g_RT[5] + g_RT[6] + g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[8]) + (h_RT[3] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q; // O2
  wdot[5] += q; // H2O
  wdot[6] -= q; // OH
  wdot[8] -= q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[5];
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[86] += dqdci; // dwdot[H2O]/d[O2]
  J[87] -= dqdci; // dwdot[OH]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  // d()/d[H2O]
  dqdci = -k_r * sc[3];
  J[138] += dqdci; // dwdot[O2]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[143] -= dqdci; // dwdot[HO2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[8];
  J[165] += dqdci; // dwdot[O2]/d[OH]
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[6];
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[221] += dqdci; // dwdot[H2O]/d[HO2]
  J[222] -= dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT

  // reaction 17: 2 HO2 <=> H2O2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[8] * sc[8]);
  k_f = 100000000 * exp(-(5555.95482115001) * invT);
  dlnkfdT = (5555.95482115001) * invT2;
  // reverse
  phi_r = sc[3] * sc[7];
  Kc =
    (kc_fast ? (ge[3] * inv_ge[8] * inv_ge[8] * ge[7])
             : exp(-g_RT[3] - g_RT[7] + 2.000000 * g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[8]) + (h_RT[3] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;     // O2
  wdot[7] += q;     // H2O2
  wdot[8] -= 2 * q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[7];
  J[84] += dqdci;      // dwdot[O2]/d[O2]
  J[88] += dqdci;      // dwdot[H2O2]/d[O2]
  J[89] += -2 * dqdci; // dwdot[HO2]/d[O2]
  // d()/d[H2O2]
  dqdci = -k_r * sc[3];
  J[192] += dqdci;      // dwdot[O2]/d[H2O2]
  J[196] += dqdci;      // dwdot[H2O2]/d[H2O2]
  J[197] += -2 * dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * 2.000000 * sc[8];
  J[219] += dqdci;      // dwdot[O2]/d[HO2]
  J[223] += dqdci;      // dwdot[H2O2]/d[HO2]
  J[224] += -2 * dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT;      // dwdot[O2]/dT
  J[709] += dqdT;      // dwdot[H2O2]/dT
  J[710] += -2 * dqdT; // dwdot[HO2]/dT

  // reaction 18: 2 HO2 <=> H2O2 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[8] * sc[8]);
  k_f = 190000 * exp(-(-708.992024785586) * invT);
  dlnkfdT = (-708.992024785586) * invT2;
  // reverse
  phi_r = sc[3] * sc[7];
  Kc =
    (kc_fast ? (ge[3] * inv_ge[8] * inv_ge[8] * ge[7])
             : exp(-g_RT[3] - g_RT[7] + 2.000000 * g_RT[8]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[8]) + (h_RT[3] + h_RT[7]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;     // O2
  wdot[7] += q;     // H2O2
  wdot[8] -= 2 * q; // HO2
  // d()/d[O2]
  dqdci = -k_r * sc[7];
  J[84] += dqdci;      // dwdot[O2]/d[O2]
  J[88] += dqdci;      // dwdot[H2O2]/d[O2]
  J[89] += -2 * dqdci; // dwdot[HO2]/d[O2]
  // d()/d[H2O2]
  dqdci = -k_r * sc[3];
  J[192] += dqdci;      // dwdot[O2]/d[H2O2]
  J[196] += dqdci;      // dwdot[H2O2]/d[H2O2]
  J[197] += -2 * dqdci; // dwdot[HO2]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * 2.000000 * sc[8];
  J[219] += dqdci;      // dwdot[O2]/d[HO2]
  J[223] += dqdci;      // dwdot[H2O2]/d[HO2]
  J[224] += -2 * dqdci; // dwdot[HO2]/d[HO2]
  // d()/dT
  J[705] += dqdT;      // dwdot[O2]/dT
  J[709] += dqdT;      // dwdot[H2O2]/dT
  J[710] += -2 * dqdT; // dwdot[HO2]/dT

  // reaction 21: CO + OH <=> CO2 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[9];
  k_f = 0.07015 * exp(2.053 * logT - (-178.994168026739) * invT);
  dlnkfdT = 2.053 * invT + (-178.994168026739) * invT2;
  // reverse
  phi_r = sc[2] * sc[10];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[9] * ge[10] * ge[2])
             : exp(-g_RT[2] + g_RT[6] + g_RT[9] - g_RT[10]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[9]) + (h_RT[2] + h_RT[10]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  // d()/d[H]
  dqdci = -k_r * sc[10];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[63] -= dqdci; // dwdot[CO]/d[H]
  J[64] += dqdci; // dwdot[CO2]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[9];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[171] -= dqdci; // dwdot[CO]/d[OH]
  J[172] += dqdci; // dwdot[CO2]/d[OH]
  // d()/d[CO]
  dqdci = +k_f * sc[6];
  J[245] += dqdci; // dwdot[H]/d[CO]
  J[249] -= dqdci; // dwdot[OH]/d[CO]
  J[252] -= dqdci; // dwdot[CO]/d[CO]
  J[253] += dqdci; // dwdot[CO2]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * sc[2];
  J[272] += dqdci; // dwdot[H]/d[CO2]
  J[276] -= dqdci; // dwdot[OH]/d[CO2]
  J[279] -= dqdci; // dwdot[CO]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT

  // reaction 22: CO + OH <=> CO2 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[9];
  k_f = 5757000 * exp(-0.664 * logT - (166.967289714006) * invT);
  dlnkfdT = -0.664 * invT + (166.967289714006) * invT2;
  // reverse
  phi_r = sc[2] * sc[10];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[9] * ge[10] * ge[2])
             : exp(-g_RT[2] + g_RT[6] + g_RT[9] - g_RT[10]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[9]) + (h_RT[2] + h_RT[10]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  // d()/d[H]
  dqdci = -k_r * sc[10];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[63] -= dqdci; // dwdot[CO]/d[H]
  J[64] += dqdci; // dwdot[CO2]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[9];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[171] -= dqdci; // dwdot[CO]/d[OH]
  J[172] += dqdci; // dwdot[CO2]/d[OH]
  // d()/d[CO]
  dqdci = +k_f * sc[6];
  J[245] += dqdci; // dwdot[H]/d[CO]
  J[249] -= dqdci; // dwdot[OH]/d[CO]
  J[252] -= dqdci; // dwdot[CO]/d[CO]
  J[253] += dqdci; // dwdot[CO2]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * sc[2];
  J[272] += dqdci; // dwdot[H]/d[CO2]
  J[276] -= dqdci; // dwdot[OH]/d[CO2]
  J[279] -= dqdci; // dwdot[CO]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT

  // reaction 23: CO + HO2 <=> CO2 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[9];
  k_f = 0.157 * exp(2.18 * logT - (9027.70698453667) * invT);
  dlnkfdT = 2.18 * invT + (9027.70698453667) * invT2;
  // reverse
  phi_r = sc[6] * sc[10];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[8] * inv_ge[9] * ge[10])
             : exp(-g_RT[6] + g_RT[8] + g_RT[9] - g_RT[10]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[9]) + (h_RT[6] + h_RT[10]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[6] += q;  // OH
  wdot[8] -= q;  // HO2
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  // d()/d[OH]
  dqdci = -k_r * sc[10];
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  J[171] -= dqdci; // dwdot[CO]/d[OH]
  J[172] += dqdci; // dwdot[CO2]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[9];
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[225] -= dqdci; // dwdot[CO]/d[HO2]
  J[226] += dqdci; // dwdot[CO2]/d[HO2]
  // d()/d[CO]
  dqdci = +k_f * sc[8];
  J[249] += dqdci; // dwdot[OH]/d[CO]
  J[251] -= dqdci; // dwdot[HO2]/d[CO]
  J[252] -= dqdci; // dwdot[CO]/d[CO]
  J[253] += dqdci; // dwdot[CO2]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * sc[6];
  J[276] += dqdci; // dwdot[OH]/d[CO2]
  J[278] -= dqdci; // dwdot[HO2]/d[CO2]
  J[279] -= dqdci; // dwdot[CO]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  // d()/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT

  // reaction 25: CH4 + H <=> CH3 + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[11];
  k_f = 0.614 * exp(2.5 * logT - (4824.33817506985) * invT);
  dlnkfdT = 2.5 * invT + (4824.33817506985) * invT2;
  // reverse
  phi_r = sc[1] * sc[12];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[11] * inv_ge[2] * ge[12])
             : exp(-g_RT[1] + g_RT[2] + g_RT[11] - g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[11]) + (h_RT[1] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[11] -= q; // CH4
  wdot[12] += q; // CH3
  // d()/d[H2]
  dqdci = -k_r * sc[12];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[38] -= dqdci; // dwdot[CH4]/d[H2]
  J[39] += dqdci; // dwdot[CH3]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[11];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[65] -= dqdci; // dwdot[CH4]/d[H]
  J[66] += dqdci; // dwdot[CH3]/d[H]
  // d()/d[CH4]
  dqdci = +k_f * sc[2];
  J[298] += dqdci; // dwdot[H2]/d[CH4]
  J[299] -= dqdci; // dwdot[H]/d[CH4]
  J[308] -= dqdci; // dwdot[CH4]/d[CH4]
  J[309] += dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * sc[1];
  J[325] += dqdci; // dwdot[H2]/d[CH3]
  J[326] -= dqdci; // dwdot[H]/d[CH3]
  J[335] -= dqdci; // dwdot[CH4]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[713] -= dqdT; // dwdot[CH4]/dT
  J[714] += dqdT; // dwdot[CH3]/dT

  // reaction 26: CH4 + O <=> CH3 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[11];
  k_f = 1020 * exp(1.5 * logT - (4327.66332592059) * invT);
  dlnkfdT = 1.5 * invT + (4327.66332592059) * invT2;
  // reverse
  phi_r = sc[6] * sc[12];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[11] * inv_ge[4] * ge[12])
             : exp(g_RT[4] - g_RT[6] + g_RT[11] - g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[11]) + (h_RT[6] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[11] -= q; // CH4
  wdot[12] += q; // CH3
  // d()/d[O]
  dqdci = +k_f * sc[11];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[119] -= dqdci; // dwdot[CH4]/d[O]
  J[120] += dqdci; // dwdot[CH3]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[12];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[173] -= dqdci; // dwdot[CH4]/d[OH]
  J[174] += dqdci; // dwdot[CH3]/d[OH]
  // d()/d[CH4]
  dqdci = +k_f * sc[4];
  J[301] -= dqdci; // dwdot[O]/d[CH4]
  J[303] += dqdci; // dwdot[OH]/d[CH4]
  J[308] -= dqdci; // dwdot[CH4]/d[CH4]
  J[309] += dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * sc[6];
  J[328] -= dqdci; // dwdot[O]/d[CH3]
  J[330] += dqdci; // dwdot[OH]/d[CH3]
  J[335] -= dqdci; // dwdot[CH4]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[713] -= dqdT; // dwdot[CH4]/dT
  J[714] += dqdT; // dwdot[CH3]/dT

  // reaction 27: CH4 + OH <=> CH3 + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[11];
  k_f = 0.0583 * exp(2.6 * logT - (1102.04449811234) * invT);
  dlnkfdT = 2.6 * invT + (1102.04449811234) * invT2;
  // reverse
  phi_r = sc[5] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[12] * ge[5] * inv_ge[11])
             : exp(-g_RT[5] + g_RT[6] + g_RT[11] - g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[11]) + (h_RT[5] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[11] -= q; // CH4
  wdot[12] += q; // CH3
  // d()/d[H2O]
  dqdci = -k_r * sc[12];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[146] -= dqdci; // dwdot[CH4]/d[H2O]
  J[147] += dqdci; // dwdot[CH3]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[11];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[173] -= dqdci; // dwdot[CH4]/d[OH]
  J[174] += dqdci; // dwdot[CH3]/d[OH]
  // d()/d[CH4]
  dqdci = +k_f * sc[6];
  J[302] += dqdci; // dwdot[H2O]/d[CH4]
  J[303] -= dqdci; // dwdot[OH]/d[CH4]
  J[308] -= dqdci; // dwdot[CH4]/d[CH4]
  J[309] += dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * sc[5];
  J[329] += dqdci; // dwdot[H2O]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[335] -= dqdci; // dwdot[CH4]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[713] -= dqdT; // dwdot[CH4]/dT
  J[714] += dqdT; // dwdot[CH3]/dT

  // reaction 28: CH4 + HO2 <=> CH3 + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[11];
  k_f = 1.13e-05 * exp(3.74 * logT - (10572.5821485572) * invT);
  dlnkfdT = 3.74 * invT + (10572.5821485572) * invT2;
  // reverse
  phi_r = sc[7] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[12] * ge[7] * inv_ge[11])
             : exp(-g_RT[7] + g_RT[8] + g_RT[11] - g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[11]) + (h_RT[7] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[11] -= q; // CH4
  wdot[12] += q; // CH3
  // d()/d[H2O2]
  dqdci = -k_r * sc[12];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[200] -= dqdci; // dwdot[CH4]/d[H2O2]
  J[201] += dqdci; // dwdot[CH3]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[11];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[227] -= dqdci; // dwdot[CH4]/d[HO2]
  J[228] += dqdci; // dwdot[CH3]/d[HO2]
  // d()/d[CH4]
  dqdci = +k_f * sc[8];
  J[304] += dqdci; // dwdot[H2O2]/d[CH4]
  J[305] -= dqdci; // dwdot[HO2]/d[CH4]
  J[308] -= dqdci; // dwdot[CH4]/d[CH4]
  J[309] += dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * sc[7];
  J[331] += dqdci; // dwdot[H2O2]/d[CH3]
  J[332] -= dqdci; // dwdot[HO2]/d[CH3]
  J[335] -= dqdci; // dwdot[CH4]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[713] -= dqdT; // dwdot[CH4]/dT
  J[714] += dqdT; // dwdot[CH3]/dT

  // reaction 29: CH3 + HO2 <=> CH4 + O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[12];
  k_f = 0.116 * exp(2.23 * logT - (-1520.72076406186) * invT);
  dlnkfdT = 2.23 * invT + (-1520.72076406186) * invT2;
  // reverse
  phi_r = sc[3] * sc[11];
  Kc =
    (kc_fast ? (ge[3] * inv_ge[8] * inv_ge[12] * ge[11])
             : exp(-g_RT[3] + g_RT[8] - g_RT[11] + g_RT[12]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[12]) + (h_RT[3] + h_RT[11]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] += q;  // O2
  wdot[8] -= q;  // HO2
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  // d()/d[O2]
  dqdci = -k_r * sc[11];
  J[84] += dqdci; // dwdot[O2]/d[O2]
  J[89] -= dqdci; // dwdot[HO2]/d[O2]
  J[92] += dqdci; // dwdot[CH4]/d[O2]
  J[93] -= dqdci; // dwdot[CH3]/d[O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[12];
  J[219] += dqdci; // dwdot[O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[227] += dqdci; // dwdot[CH4]/d[HO2]
  J[228] -= dqdci; // dwdot[CH3]/d[HO2]
  // d()/d[CH4]
  dqdci = -k_r * sc[3];
  J[300] += dqdci; // dwdot[O2]/d[CH4]
  J[305] -= dqdci; // dwdot[HO2]/d[CH4]
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[8];
  J[327] += dqdci; // dwdot[O2]/d[CH3]
  J[332] -= dqdci; // dwdot[HO2]/d[CH3]
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  // d()/dT
  J[705] += dqdT; // dwdot[O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT

  // reaction 30: CH2 + CH4 <=> 2 CH3
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[11] * sc[13];
  k_f = 2.46 * exp(2 * logT - (4161.60182620503) * invT);
  dlnkfdT = 2 * invT + (4161.60182620503) * invT2;
  // reverse
  phi_r = (sc[12] * sc[12]);
  Kc =
    (kc_fast ? (ge[12] * ge[12] * inv_ge[13] * inv_ge[11])
             : exp(g_RT[11] - 2.000000 * g_RT[12] + g_RT[13]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[11] + h_RT[13]) + (2.000000 * h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] -= q;     // CH4
  wdot[12] += 2 * q; // CH3
  wdot[13] -= q;     // CH2
  // d()/d[CH4]
  dqdci = +k_f * sc[13];
  J[308] -= dqdci;     // dwdot[CH4]/d[CH4]
  J[309] += 2 * dqdci; // dwdot[CH3]/d[CH4]
  J[310] -= dqdci;     // dwdot[CH2]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * 2.000000 * sc[12];
  J[335] -= dqdci;     // dwdot[CH4]/d[CH3]
  J[336] += 2 * dqdci; // dwdot[CH3]/d[CH3]
  J[337] -= dqdci;     // dwdot[CH2]/d[CH3]
  // d()/d[CH2]
  dqdci = +k_f * sc[11];
  J[362] -= dqdci;     // dwdot[CH4]/d[CH2]
  J[363] += 2 * dqdci; // dwdot[CH3]/d[CH2]
  J[364] -= dqdci;     // dwdot[CH2]/d[CH2]
  // d()/dT
  J[713] -= dqdT;     // dwdot[CH4]/dT
  J[714] += 2 * dqdT; // dwdot[CH3]/dT
  J[715] -= dqdT;     // dwdot[CH2]/dT

  // reaction 31: CH2(S) + N2 <=> CH2 + N2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[0] * sc[14];
  k_f = 15000000 * exp(-(301.929999482832) * invT);
  dlnkfdT = (301.929999482832) * invT2;
  // reverse
  phi_r = sc[0] * sc[13];
  Kc =
    (kc_fast ? (ge[13] * inv_ge[14])
             : exp(g_RT[0] - g_RT[0] - g_RT[13] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[0] + h_RT[14]) + (h_RT[0] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[13] += q; // CH2
  wdot[14] -= q; // CH2(S)
  // d()/d[N2]
  dqdci = +k_f * sc[14] - k_r * sc[13];
  J[13] += dqdci; // dwdot[CH2]/d[N2]
  J[14] -= dqdci; // dwdot[CH2(S)]/d[N2]
  // d()/d[CH2]
  dqdci = -k_r * sc[0];
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  J[365] -= dqdci; // dwdot[CH2(S)]/d[CH2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[0];
  J[391] += dqdci; // dwdot[CH2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[715] += dqdT; // dwdot[CH2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 32: CH2(S) + H2O <=> CH2 + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[5] * sc[14];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[13];
  Kc =
    (kc_fast ? (ge[13] * inv_ge[14])
             : exp(g_RT[5] - g_RT[5] - g_RT[13] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[5] + h_RT[14]) + (h_RT[5] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[13] += q; // CH2
  wdot[14] -= q; // CH2(S)
  // d()/d[H2O]
  dqdci = +k_f * sc[14] - k_r * sc[13];
  J[148] += dqdci; // dwdot[CH2]/d[H2O]
  J[149] -= dqdci; // dwdot[CH2(S)]/d[H2O]
  // d()/d[CH2]
  dqdci = -k_r * sc[5];
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  J[365] -= dqdci; // dwdot[CH2(S)]/d[CH2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[5];
  J[391] += dqdci; // dwdot[CH2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[715] += dqdT; // dwdot[CH2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 33: CH2(S) + CO <=> CH2 + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[9] * sc[14];
  k_f = 9000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[13];
  Kc =
    (kc_fast ? (ge[13] * inv_ge[14])
             : exp(g_RT[9] - g_RT[9] - g_RT[13] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[9] + h_RT[14]) + (h_RT[9] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[13] += q; // CH2
  wdot[14] -= q; // CH2(S)
  // d()/d[CO]
  dqdci = +k_f * sc[14] - k_r * sc[13];
  J[256] += dqdci; // dwdot[CH2]/d[CO]
  J[257] -= dqdci; // dwdot[CH2(S)]/d[CO]
  // d()/d[CH2]
  dqdci = -k_r * sc[9];
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  J[365] -= dqdci; // dwdot[CH2(S)]/d[CH2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[9];
  J[391] += dqdci; // dwdot[CH2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[715] += dqdT; // dwdot[CH2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 34: CH2(S) + CO2 <=> CH2 + CO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[10] * sc[14];
  k_f = 7000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[10] * sc[13];
  Kc =
    (kc_fast ? (ge[13] * inv_ge[14])
             : exp(g_RT[10] - g_RT[10] - g_RT[13] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[10] + h_RT[14]) + (h_RT[10] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[13] += q; // CH2
  wdot[14] -= q; // CH2(S)
  // d()/d[CO2]
  dqdci = +k_f * sc[14] - k_r * sc[13];
  J[283] += dqdci; // dwdot[CH2]/d[CO2]
  J[284] -= dqdci; // dwdot[CH2(S)]/d[CO2]
  // d()/d[CH2]
  dqdci = -k_r * sc[10];
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  J[365] -= dqdci; // dwdot[CH2(S)]/d[CH2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[10];
  J[391] += dqdci; // dwdot[CH2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[715] += dqdT; // dwdot[CH2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 35: CH2(S) + O2 => CO + H + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[14];
  k_f = 28000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += q;  // H
  wdot[3] -= q;  // O2
  wdot[6] += q;  // OH
  wdot[9] += q;  // CO
  wdot[14] -= q; // CH2(S)
  // d()/d[O2]
  dqdci = +k_f * sc[14];
  J[83] += dqdci; // dwdot[H]/d[O2]
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[87] += dqdci; // dwdot[OH]/d[O2]
  J[90] += dqdci; // dwdot[CO]/d[O2]
  J[95] -= dqdci; // dwdot[CH2(S)]/d[O2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[3];
  J[380] += dqdci; // dwdot[H]/d[CH2(S)]
  J[381] -= dqdci; // dwdot[O2]/d[CH2(S)]
  J[384] += dqdci; // dwdot[OH]/d[CH2(S)]
  J[387] += dqdci; // dwdot[CO]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 36: CH2(S) + O2 <=> CO + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[14];
  k_f = 12000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[9] * ge[5] * inv_ge[14])
             : exp(g_RT[3] - g_RT[5] - g_RT[9] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[14]) + (h_RT[5] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[5] += q;  // H2O
  wdot[9] += q;  // CO
  wdot[14] -= q; // CH2(S)
  // d()/d[O2]
  dqdci = +k_f * sc[14];
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[86] += dqdci; // dwdot[H2O]/d[O2]
  J[90] += dqdci; // dwdot[CO]/d[O2]
  J[95] -= dqdci; // dwdot[CH2(S)]/d[O2]
  // d()/d[H2O]
  dqdci = -k_r * sc[9];
  J[138] -= dqdci; // dwdot[O2]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[144] += dqdci; // dwdot[CO]/d[H2O]
  J[149] -= dqdci; // dwdot[CH2(S)]/d[H2O]
  // d()/d[CO]
  dqdci = -k_r * sc[5];
  J[246] -= dqdci; // dwdot[O2]/d[CO]
  J[248] += dqdci; // dwdot[H2O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[257] -= dqdci; // dwdot[CH2(S)]/d[CO]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[3];
  J[381] -= dqdci; // dwdot[O2]/d[CH2(S)]
  J[383] += dqdci; // dwdot[H2O]/d[CH2(S)]
  J[387] += dqdci; // dwdot[CO]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 37: CH2(S) + O <=> CO + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[14];
  k_f = 15000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc =
    (kc_fast ? (ge[1] * ge[9] * inv_ge[4] * inv_ge[14])
             : exp(-g_RT[1] + g_RT[4] - g_RT[9] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[14]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[4] -= q;  // O
  wdot[9] += q;  // CO
  wdot[14] -= q; // CH2(S)
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[31] -= dqdci; // dwdot[O]/d[H2]
  J[36] += dqdci; // dwdot[CO]/d[H2]
  J[41] -= dqdci; // dwdot[CH2(S)]/d[H2]
  // d()/d[O]
  dqdci = +k_f * sc[14];
  J[109] += dqdci; // dwdot[H2]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[122] -= dqdci; // dwdot[CH2(S)]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[1];
  J[244] += dqdci; // dwdot[H2]/d[CO]
  J[247] -= dqdci; // dwdot[O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[257] -= dqdci; // dwdot[CH2(S)]/d[CO]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[4];
  J[379] += dqdci; // dwdot[H2]/d[CH2(S)]
  J[382] -= dqdci; // dwdot[O]/d[CH2(S)]
  J[387] += dqdci; // dwdot[CO]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 38: CH2(S) + O <=> H + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[14];
  k_f = 15000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * ge[2] * inv_ge[4] * inv_ge[14])
             : exp(-g_RT[2] + g_RT[4] + g_RT[14] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[14]) + (h_RT[2] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[4] -= q;  // O
  wdot[14] -= q; // CH2(S)
  wdot[21] += q; // HCO
  // d()/d[H]
  dqdci = -k_r * sc[21];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[68] -= dqdci; // dwdot[CH2(S)]/d[H]
  J[75] += dqdci; // dwdot[HCO]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[14];
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[122] -= dqdci; // dwdot[CH2(S)]/d[O]
  J[129] += dqdci; // dwdot[HCO]/d[O]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[4];
  J[380] += dqdci; // dwdot[H]/d[CH2(S)]
  J[382] -= dqdci; // dwdot[O]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  J[399] += dqdci; // dwdot[HCO]/d[CH2(S)]
  // d()/d[HCO]
  dqdci = -k_r * sc[2];
  J[569] += dqdci; // dwdot[H]/d[HCO]
  J[571] -= dqdci; // dwdot[O]/d[HCO]
  J[581] -= dqdci; // dwdot[CH2(S)]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 39: CH2(S) + H2 <=> CH3 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[14];
  k_f = 70000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[1] * ge[12] * inv_ge[14] * ge[2])
             : exp(g_RT[1] - g_RT[2] - g_RT[12] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[14]) + (h_RT[2] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q;  // H2
  wdot[2] += q;  // H
  wdot[12] += q; // CH3
  wdot[14] -= q; // CH2(S)
  // d()/d[H2]
  dqdci = +k_f * sc[14];
  J[28] -= dqdci; // dwdot[H2]/d[H2]
  J[29] += dqdci; // dwdot[H]/d[H2]
  J[39] += dqdci; // dwdot[CH3]/d[H2]
  J[41] -= dqdci; // dwdot[CH2(S)]/d[H2]
  // d()/d[H]
  dqdci = -k_r * sc[12];
  J[55] -= dqdci; // dwdot[H2]/d[H]
  J[56] += dqdci; // dwdot[H]/d[H]
  J[66] += dqdci; // dwdot[CH3]/d[H]
  J[68] -= dqdci; // dwdot[CH2(S)]/d[H]
  // d()/d[CH3]
  dqdci = -k_r * sc[2];
  J[325] -= dqdci; // dwdot[H2]/d[CH3]
  J[326] += dqdci; // dwdot[H]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  J[338] -= dqdci; // dwdot[CH2(S)]/d[CH3]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[1];
  J[379] -= dqdci; // dwdot[H2]/d[CH2(S)]
  J[380] += dqdci; // dwdot[H]/d[CH2(S)]
  J[390] += dqdci; // dwdot[CH3]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[703] -= dqdT; // dwdot[H2]/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT

  // reaction 40: CH2(S) + H <=> CH + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[14];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[16];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[2] * ge[16] * inv_ge[14])
             : exp(-g_RT[1] + g_RT[2] + g_RT[14] - g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[14]) + (h_RT[1] + h_RT[16]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[14] -= q; // CH2(S)
  wdot[16] += q; // CH
  // d()/d[H2]
  dqdci = -k_r * sc[16];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[41] -= dqdci; // dwdot[CH2(S)]/d[H2]
  J[43] += dqdci; // dwdot[CH]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[14];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[68] -= dqdci; // dwdot[CH2(S)]/d[H]
  J[70] += dqdci; // dwdot[CH]/d[H]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[2];
  J[379] += dqdci; // dwdot[H2]/d[CH2(S)]
  J[380] -= dqdci; // dwdot[H]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  J[394] += dqdci; // dwdot[CH]/d[CH2(S)]
  // d()/d[CH]
  dqdci = -k_r * sc[1];
  J[433] += dqdci; // dwdot[H2]/d[CH]
  J[434] -= dqdci; // dwdot[H]/d[CH]
  J[446] -= dqdci; // dwdot[CH2(S)]/d[CH]
  J[448] += dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT
  J[718] += dqdT; // dwdot[CH]/dT

  // reaction 41: CH2(S) + OH <=> CH2O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[14];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[20] * ge[2] * inv_ge[14])
             : exp(-g_RT[2] + g_RT[6] + g_RT[14] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[14]) + (h_RT[2] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[14] -= q; // CH2(S)
  wdot[20] += q; // CH2O
  // d()/d[H]
  dqdci = -k_r * sc[20];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[68] -= dqdci; // dwdot[CH2(S)]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[14];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[176] -= dqdci; // dwdot[CH2(S)]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[6];
  J[380] += dqdci; // dwdot[H]/d[CH2(S)]
  J[384] -= dqdci; // dwdot[OH]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  J[398] += dqdci; // dwdot[CH2O]/d[CH2(S)]
  // d()/d[CH2O]
  dqdci = -k_r * sc[2];
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[554] -= dqdci; // dwdot[CH2(S)]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 42: CH2(S) + CO2 <=> CH2O + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[10] * sc[14];
  k_f = 14000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[20];
  Kc =
    (kc_fast ? (ge[9] * inv_ge[10] * inv_ge[14] * ge[20])
             : exp(-g_RT[9] + g_RT[10] + g_RT[14] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[10] + h_RT[14]) + (h_RT[9] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[10] -= q; // CO2
  wdot[14] -= q; // CH2(S)
  wdot[20] += q; // CH2O
  // d()/d[CO]
  dqdci = -k_r * sc[20];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[253] -= dqdci; // dwdot[CO2]/d[CO]
  J[257] -= dqdci; // dwdot[CH2(S)]/d[CO]
  J[263] += dqdci; // dwdot[CH2O]/d[CO]
  // d()/d[CO2]
  dqdci = +k_f * sc[14];
  J[279] += dqdci; // dwdot[CO]/d[CO2]
  J[280] -= dqdci; // dwdot[CO2]/d[CO2]
  J[284] -= dqdci; // dwdot[CH2(S)]/d[CO2]
  J[290] += dqdci; // dwdot[CH2O]/d[CO2]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[10];
  J[387] += dqdci; // dwdot[CO]/d[CH2(S)]
  J[388] -= dqdci; // dwdot[CO2]/d[CH2(S)]
  J[392] -= dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  J[398] += dqdci; // dwdot[CH2O]/d[CH2(S)]
  // d()/d[CH2O]
  dqdci = -k_r * sc[9];
  J[549] += dqdci; // dwdot[CO]/d[CH2O]
  J[550] -= dqdci; // dwdot[CO2]/d[CH2O]
  J[554] -= dqdci; // dwdot[CH2(S)]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[712] -= dqdT; // dwdot[CO2]/dT
  J[716] -= dqdT; // dwdot[CH2(S)]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 44: CH2 + O2 <=> HCO + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[13];
  k_f = 10600000 * exp(-(754.82499870708) * invT);
  dlnkfdT = (754.82499870708) * invT2;
  // reverse
  phi_r = sc[6] * sc[21];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[3] * ge[21] * inv_ge[13])
             : exp(g_RT[3] - g_RT[6] + g_RT[13] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[13]) + (h_RT[6] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[6] += q;  // OH
  wdot[13] -= q; // CH2
  wdot[21] += q; // HCO
  // d()/d[O2]
  dqdci = +k_f * sc[13];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[87] += dqdci;  // dwdot[OH]/d[O2]
  J[94] -= dqdci;  // dwdot[CH2]/d[O2]
  J[102] += dqdci; // dwdot[HCO]/d[O2]
  // d()/d[OH]
  dqdci = -k_r * sc[21];
  J[165] -= dqdci; // dwdot[O2]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[175] -= dqdci; // dwdot[CH2]/d[OH]
  J[183] += dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CH2]
  dqdci = +k_f * sc[3];
  J[354] -= dqdci; // dwdot[O2]/d[CH2]
  J[357] += dqdci; // dwdot[OH]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[372] += dqdci; // dwdot[HCO]/d[CH2]
  // d()/d[HCO]
  dqdci = -k_r * sc[6];
  J[570] -= dqdci; // dwdot[O2]/d[HCO]
  J[573] += dqdci; // dwdot[OH]/d[HCO]
  J[580] -= dqdci; // dwdot[CH2]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 45: CH2 + O2 => CO2 + 2 H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[13];
  k_f = 2640000 * exp(-(754.82499870708) * invT);
  dlnkfdT = (754.82499870708) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[3] -= q;     // O2
  wdot[10] += q;    // CO2
  wdot[13] -= q;    // CH2
  // d()/d[O2]
  dqdci = +k_f * sc[13];
  J[83] += 2 * dqdci; // dwdot[H]/d[O2]
  J[84] -= dqdci;     // dwdot[O2]/d[O2]
  J[91] += dqdci;     // dwdot[CO2]/d[O2]
  J[94] -= dqdci;     // dwdot[CH2]/d[O2]
  // d()/d[CH2]
  dqdci = +k_f * sc[3];
  J[353] += 2 * dqdci; // dwdot[H]/d[CH2]
  J[354] -= dqdci;     // dwdot[O2]/d[CH2]
  J[361] += dqdci;     // dwdot[CO2]/d[CH2]
  J[364] -= dqdci;     // dwdot[CH2]/d[CH2]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[705] -= dqdT;     // dwdot[O2]/dT
  J[712] += dqdT;     // dwdot[CO2]/dT
  J[715] -= dqdT;     // dwdot[CH2]/dT

  // reaction 46: CH2 + O => CO + 2 H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[13];
  k_f = 50000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[4] -= q;     // O
  wdot[9] += q;     // CO
  wdot[13] -= q;    // CH2
  // d()/d[O]
  dqdci = +k_f * sc[13];
  J[110] += 2 * dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci;     // dwdot[O]/d[O]
  J[117] += dqdci;     // dwdot[CO]/d[O]
  J[121] -= dqdci;     // dwdot[CH2]/d[O]
  // d()/d[CH2]
  dqdci = +k_f * sc[4];
  J[353] += 2 * dqdci; // dwdot[H]/d[CH2]
  J[355] -= dqdci;     // dwdot[O]/d[CH2]
  J[360] += dqdci;     // dwdot[CO]/d[CH2]
  J[364] -= dqdci;     // dwdot[CH2]/d[CH2]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[706] -= dqdT;     // dwdot[O]/dT
  J[711] += dqdT;     // dwdot[CO]/dT
  J[715] -= dqdT;     // dwdot[CH2]/dT

  // reaction 47: CH2 + H <=> CH + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[13];
  k_f = 283000000 * exp(-(369.864249366469) * invT);
  dlnkfdT = (369.864249366469) * invT2;
  // reverse
  phi_r = sc[1] * sc[16];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[2] * ge[16] * inv_ge[13])
             : exp(-g_RT[1] + g_RT[2] + g_RT[13] - g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[13]) + (h_RT[1] + h_RT[16]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[13] -= q; // CH2
  wdot[16] += q; // CH
  // d()/d[H2]
  dqdci = -k_r * sc[16];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[40] -= dqdci; // dwdot[CH2]/d[H2]
  J[43] += dqdci; // dwdot[CH]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[13];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[67] -= dqdci; // dwdot[CH2]/d[H]
  J[70] += dqdci; // dwdot[CH]/d[H]
  // d()/d[CH2]
  dqdci = +k_f * sc[2];
  J[352] += dqdci; // dwdot[H2]/d[CH2]
  J[353] -= dqdci; // dwdot[H]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[367] += dqdci; // dwdot[CH]/d[CH2]
  // d()/d[CH]
  dqdci = -k_r * sc[1];
  J[433] += dqdci; // dwdot[H2]/d[CH]
  J[434] -= dqdci; // dwdot[H]/d[CH]
  J[445] -= dqdci; // dwdot[CH2]/d[CH]
  J[448] += dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[718] += dqdT; // dwdot[CH]/dT

  // reaction 48: CH2 + OH <=> CH + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[13];
  k_f = 11.3 * exp(2 * logT - (1509.64999741416) * invT);
  dlnkfdT = 2 * invT + (1509.64999741416) * invT2;
  // reverse
  phi_r = sc[5] * sc[16];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[5] * ge[16] * inv_ge[13])
             : exp(-g_RT[5] + g_RT[6] + g_RT[13] - g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[13]) + (h_RT[5] + h_RT[16]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[13] -= q; // CH2
  wdot[16] += q; // CH
  // d()/d[H2O]
  dqdci = -k_r * sc[16];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[148] -= dqdci; // dwdot[CH2]/d[H2O]
  J[151] += dqdci; // dwdot[CH]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[13];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[175] -= dqdci; // dwdot[CH2]/d[OH]
  J[178] += dqdci; // dwdot[CH]/d[OH]
  // d()/d[CH2]
  dqdci = +k_f * sc[6];
  J[356] += dqdci; // dwdot[H2O]/d[CH2]
  J[357] -= dqdci; // dwdot[OH]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[367] += dqdci; // dwdot[CH]/d[CH2]
  // d()/d[CH]
  dqdci = -k_r * sc[5];
  J[437] += dqdci; // dwdot[H2O]/d[CH]
  J[438] -= dqdci; // dwdot[OH]/d[CH]
  J[445] -= dqdci; // dwdot[CH2]/d[CH]
  J[448] += dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[718] += dqdT; // dwdot[CH]/dT

  // reaction 49: CH + O2 <=> HCO + O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[16];
  k_f = 33000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[4] * sc[21];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[21] * ge[4] * inv_ge[16])
             : exp(g_RT[3] - g_RT[4] + g_RT[16] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[16]) + (h_RT[4] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[4] += q;  // O
  wdot[16] -= q; // CH
  wdot[21] += q; // HCO
  // d()/d[O2]
  dqdci = +k_f * sc[16];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[85] += dqdci;  // dwdot[O]/d[O2]
  J[97] -= dqdci;  // dwdot[CH]/d[O2]
  J[102] += dqdci; // dwdot[HCO]/d[O2]
  // d()/d[O]
  dqdci = -k_r * sc[21];
  J[111] -= dqdci; // dwdot[O2]/d[O]
  J[112] += dqdci; // dwdot[O]/d[O]
  J[124] -= dqdci; // dwdot[CH]/d[O]
  J[129] += dqdci; // dwdot[HCO]/d[O]
  // d()/d[CH]
  dqdci = +k_f * sc[3];
  J[435] -= dqdci; // dwdot[O2]/d[CH]
  J[436] += dqdci; // dwdot[O]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[453] += dqdci; // dwdot[HCO]/d[CH]
  // d()/d[HCO]
  dqdci = -k_r * sc[4];
  J[570] -= dqdci; // dwdot[O2]/d[HCO]
  J[571] += dqdci; // dwdot[O]/d[HCO]
  J[583] -= dqdci; // dwdot[CH]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[706] += dqdT; // dwdot[O]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 50: CH + O <=> CO + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[16];
  k_f = 57000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[9];
  Kc =
    (kc_fast ? (ge[9] * ge[2] * inv_ge[4] * inv_ge[16])
             : exp(-g_RT[2] + g_RT[4] - g_RT[9] + g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[16]) + (h_RT[2] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[4] -= q;  // O
  wdot[9] += q;  // CO
  wdot[16] -= q; // CH
  // d()/d[H]
  dqdci = -k_r * sc[9];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[63] += dqdci; // dwdot[CO]/d[H]
  J[70] -= dqdci; // dwdot[CH]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[16];
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[124] -= dqdci; // dwdot[CH]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[2];
  J[245] += dqdci; // dwdot[H]/d[CO]
  J[247] -= dqdci; // dwdot[O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[259] -= dqdci; // dwdot[CH]/d[CO]
  // d()/d[CH]
  dqdci = +k_f * sc[4];
  J[434] += dqdci; // dwdot[H]/d[CH]
  J[436] -= dqdci; // dwdot[O]/d[CH]
  J[441] += dqdci; // dwdot[CO]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[718] -= dqdT; // dwdot[CH]/dT

  // reaction 51: CH + H <=> C + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[16];
  k_f = 110000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[15];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[2] * ge[15] * inv_ge[16])
             : exp(-g_RT[1] + g_RT[2] - g_RT[15] + g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[16]) + (h_RT[1] + h_RT[15]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[15] += q; // C
  wdot[16] -= q; // CH
  // d()/d[H2]
  dqdci = -k_r * sc[15];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[42] += dqdci; // dwdot[C]/d[H2]
  J[43] -= dqdci; // dwdot[CH]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[16];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[69] += dqdci; // dwdot[C]/d[H]
  J[70] -= dqdci; // dwdot[CH]/d[H]
  // d()/d[C]
  dqdci = -k_r * sc[1];
  J[406] += dqdci; // dwdot[H2]/d[C]
  J[407] -= dqdci; // dwdot[H]/d[C]
  J[420] += dqdci; // dwdot[C]/d[C]
  J[421] -= dqdci; // dwdot[CH]/d[C]
  // d()/d[CH]
  dqdci = +k_f * sc[2];
  J[433] += dqdci; // dwdot[H2]/d[CH]
  J[434] -= dqdci; // dwdot[H]/d[CH]
  J[447] += dqdci; // dwdot[C]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[717] += dqdT; // dwdot[C]/dT
  J[718] -= dqdT; // dwdot[CH]/dT

  // reaction 52: CH + OH <=> H + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[16];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[21];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[21] * ge[2] * inv_ge[16])
             : exp(-g_RT[2] + g_RT[6] + g_RT[16] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[16]) + (h_RT[2] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[16] -= q; // CH
  wdot[21] += q; // HCO
  // d()/d[H]
  dqdci = -k_r * sc[21];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[70] -= dqdci; // dwdot[CH]/d[H]
  J[75] += dqdci; // dwdot[HCO]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[16];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[178] -= dqdci; // dwdot[CH]/d[OH]
  J[183] += dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CH]
  dqdci = +k_f * sc[6];
  J[434] += dqdci; // dwdot[H]/d[CH]
  J[438] -= dqdci; // dwdot[OH]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[453] += dqdci; // dwdot[HCO]/d[CH]
  // d()/d[HCO]
  dqdci = -k_r * sc[2];
  J[569] += dqdci; // dwdot[H]/d[HCO]
  J[573] -= dqdci; // dwdot[OH]/d[HCO]
  J[583] -= dqdci; // dwdot[CH]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 53: CH + H2O <=> CH2O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[5] * sc[16];
  k_f = 17130000 * exp(-(-379.928582682563) * invT);
  dlnkfdT = (-379.928582682563) * invT2;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc =
    (kc_fast ? (ge[20] * ge[2] * inv_ge[5] * inv_ge[16])
             : exp(-g_RT[2] + g_RT[5] + g_RT[16] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[5] + h_RT[16]) + (h_RT[2] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[5] -= q;  // H2O
  wdot[16] -= q; // CH
  wdot[20] += q; // CH2O
  // d()/d[H]
  dqdci = -k_r * sc[20];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[59] -= dqdci; // dwdot[H2O]/d[H]
  J[70] -= dqdci; // dwdot[CH]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[H2O]
  dqdci = +k_f * sc[16];
  J[137] += dqdci; // dwdot[H]/d[H2O]
  J[140] -= dqdci; // dwdot[H2O]/d[H2O]
  J[151] -= dqdci; // dwdot[CH]/d[H2O]
  J[155] += dqdci; // dwdot[CH2O]/d[H2O]
  // d()/d[CH]
  dqdci = +k_f * sc[5];
  J[434] += dqdci; // dwdot[H]/d[CH]
  J[437] -= dqdci; // dwdot[H2O]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[452] += dqdci; // dwdot[CH2O]/d[CH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[2];
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[545] -= dqdci; // dwdot[H2O]/d[CH2O]
  J[556] -= dqdci; // dwdot[CH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[707] -= dqdT; // dwdot[H2O]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 54: CH + CO2 <=> CO + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[10] * sc[16];
  k_f = 1700000 * exp(-(344.703416076233) * invT);
  dlnkfdT = (344.703416076233) * invT2;
  // reverse
  phi_r = sc[9] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * ge[9] * inv_ge[10] * inv_ge[16])
             : exp(-g_RT[9] + g_RT[10] + g_RT[16] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[10] + h_RT[16]) + (h_RT[9] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[10] -= q; // CO2
  wdot[16] -= q; // CH
  wdot[21] += q; // HCO
  // d()/d[CO]
  dqdci = -k_r * sc[21];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[253] -= dqdci; // dwdot[CO2]/d[CO]
  J[259] -= dqdci; // dwdot[CH]/d[CO]
  J[264] += dqdci; // dwdot[HCO]/d[CO]
  // d()/d[CO2]
  dqdci = +k_f * sc[16];
  J[279] += dqdci; // dwdot[CO]/d[CO2]
  J[280] -= dqdci; // dwdot[CO2]/d[CO2]
  J[286] -= dqdci; // dwdot[CH]/d[CO2]
  J[291] += dqdci; // dwdot[HCO]/d[CO2]
  // d()/d[CH]
  dqdci = +k_f * sc[10];
  J[441] += dqdci; // dwdot[CO]/d[CH]
  J[442] -= dqdci; // dwdot[CO2]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[453] += dqdci; // dwdot[HCO]/d[CH]
  // d()/d[HCO]
  dqdci = -k_r * sc[9];
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[577] -= dqdci; // dwdot[CO2]/d[HCO]
  J[583] -= dqdci; // dwdot[CH]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[712] -= dqdT; // dwdot[CO2]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 55: C + OH <=> CO + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[15];
  k_f = 50000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[9] * ge[2] * inv_ge[15])
             : exp(-g_RT[2] + g_RT[6] - g_RT[9] + g_RT[15]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[15]) + (h_RT[2] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[9] += q;  // CO
  wdot[15] -= q; // C
  // d()/d[H]
  dqdci = -k_r * sc[9];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[63] += dqdci; // dwdot[CO]/d[H]
  J[69] -= dqdci; // dwdot[C]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[15];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[171] += dqdci; // dwdot[CO]/d[OH]
  J[177] -= dqdci; // dwdot[C]/d[OH]
  // d()/d[CO]
  dqdci = -k_r * sc[2];
  J[245] += dqdci; // dwdot[H]/d[CO]
  J[249] -= dqdci; // dwdot[OH]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[258] -= dqdci; // dwdot[C]/d[CO]
  // d()/d[C]
  dqdci = +k_f * sc[6];
  J[407] += dqdci; // dwdot[H]/d[C]
  J[411] -= dqdci; // dwdot[OH]/d[C]
  J[414] += dqdci; // dwdot[CO]/d[C]
  J[420] -= dqdci; // dwdot[C]/d[C]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[717] -= dqdT; // dwdot[C]/dT

  // reaction 56: C + O2 <=> CO + O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[15];
  k_f = 50000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[4] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[9] * ge[4] * inv_ge[15])
             : exp(g_RT[3] - g_RT[4] - g_RT[9] + g_RT[15]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[15]) + (h_RT[4] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[4] += q;  // O
  wdot[9] += q;  // CO
  wdot[15] -= q; // C
  // d()/d[O2]
  dqdci = +k_f * sc[15];
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[85] += dqdci; // dwdot[O]/d[O2]
  J[90] += dqdci; // dwdot[CO]/d[O2]
  J[96] -= dqdci; // dwdot[C]/d[O2]
  // d()/d[O]
  dqdci = -k_r * sc[9];
  J[111] -= dqdci; // dwdot[O2]/d[O]
  J[112] += dqdci; // dwdot[O]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[123] -= dqdci; // dwdot[C]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[4];
  J[246] -= dqdci; // dwdot[O2]/d[CO]
  J[247] += dqdci; // dwdot[O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[258] -= dqdci; // dwdot[C]/d[CO]
  // d()/d[C]
  dqdci = +k_f * sc[3];
  J[408] -= dqdci; // dwdot[O2]/d[C]
  J[409] += dqdci; // dwdot[O]/d[C]
  J[414] += dqdci; // dwdot[CO]/d[C]
  J[420] -= dqdci; // dwdot[C]/d[C]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[706] += dqdT; // dwdot[O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[717] -= dqdT; // dwdot[C]/dT

  // reaction 57: CH3 + O2 <=> CH3O + O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[12];
  k_f = 7546000 * exp(-(14251.0959755897) * invT);
  dlnkfdT = (14251.0959755897) * invT2;
  // reverse
  phi_r = sc[4] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[18] * inv_ge[12] * ge[4])
             : exp(g_RT[3] - g_RT[4] + g_RT[12] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[12]) + (h_RT[4] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[4] += q;  // O
  wdot[12] -= q; // CH3
  wdot[18] += q; // CH3O
  // d()/d[O2]
  dqdci = +k_f * sc[12];
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[85] += dqdci; // dwdot[O]/d[O2]
  J[93] -= dqdci; // dwdot[CH3]/d[O2]
  J[99] += dqdci; // dwdot[CH3O]/d[O2]
  // d()/d[O]
  dqdci = -k_r * sc[18];
  J[111] -= dqdci; // dwdot[O2]/d[O]
  J[112] += dqdci; // dwdot[O]/d[O]
  J[120] -= dqdci; // dwdot[CH3]/d[O]
  J[126] += dqdci; // dwdot[CH3O]/d[O]
  // d()/d[CH3]
  dqdci = +k_f * sc[3];
  J[327] -= dqdci; // dwdot[O2]/d[CH3]
  J[328] += dqdci; // dwdot[O]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3O]
  dqdci = -k_r * sc[4];
  J[489] -= dqdci; // dwdot[O2]/d[CH3O]
  J[490] += dqdci; // dwdot[O]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[706] += dqdT; // dwdot[O]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 58: CH3 + O2 <=> CH2O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[12];
  k_f = 2.641e-06 * exp(3.283 * logT - (4078.57107634725) * invT);
  dlnkfdT = 3.283 * invT + (4078.57107634725) * invT2;
  // reverse
  phi_r = sc[6] * sc[20];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[3] * inv_ge[12] * ge[20])
             : exp(g_RT[3] - g_RT[6] + g_RT[12] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[12]) + (h_RT[6] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[6] += q;  // OH
  wdot[12] -= q; // CH3
  wdot[20] += q; // CH2O
  // d()/d[O2]
  dqdci = +k_f * sc[12];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[87] += dqdci;  // dwdot[OH]/d[O2]
  J[93] -= dqdci;  // dwdot[CH3]/d[O2]
  J[101] += dqdci; // dwdot[CH2O]/d[O2]
  // d()/d[OH]
  dqdci = -k_r * sc[20];
  J[165] -= dqdci; // dwdot[O2]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[3];
  J[327] -= dqdci; // dwdot[O2]/d[CH3]
  J[330] += dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH2O]
  dqdci = -k_r * sc[6];
  J[543] -= dqdci; // dwdot[O2]/d[CH2O]
  J[546] += dqdci; // dwdot[OH]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 59: CH3 + O <=> CH2O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[12];
  k_f = 55400000 * exp(0.05 * logT - (-68.4374665494419) * invT);
  dlnkfdT = 0.05 * invT + (-68.4374665494419) * invT2;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[12] * ge[2] * ge[20] * inv_ge[4])
             : exp(-g_RT[2] + g_RT[4] + g_RT[12] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[12]) + (h_RT[2] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[4] -= q;  // O
  wdot[12] -= q; // CH3
  wdot[20] += q; // CH2O
  // d()/d[H]
  dqdci = -k_r * sc[20];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[66] -= dqdci; // dwdot[CH3]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[12];
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[120] -= dqdci; // dwdot[CH3]/d[O]
  J[128] += dqdci; // dwdot[CH2O]/d[O]
  // d()/d[CH3]
  dqdci = +k_f * sc[4];
  J[326] += dqdci; // dwdot[H]/d[CH3]
  J[328] -= dqdci; // dwdot[O]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH2O]
  dqdci = -k_r * sc[2];
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[544] -= dqdci; // dwdot[O]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 60: CH3 + OH <=> CH2(S) + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 4.788e+17 * exp(-3.155 * logT - (3524.02631063045) * invT);
  dlnkfdT = -3.155 * invT + (3524.02631063045) * invT2;
  // reverse
  phi_r = sc[5] * sc[14];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[12] * ge[14] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[12] - g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[5] + h_RT[14]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[14] += q; // CH2(S)
  // d()/d[H2O]
  dqdci = -k_r * sc[14];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[147] -= dqdci; // dwdot[CH3]/d[H2O]
  J[149] += dqdci; // dwdot[CH2(S)]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[176] += dqdci; // dwdot[CH2(S)]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[329] += dqdci; // dwdot[H2O]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[338] += dqdci; // dwdot[CH2(S)]/d[CH3]
  // d()/d[CH2(S)]
  dqdci = -k_r * sc[5];
  J[383] += dqdci; // dwdot[H2O]/d[CH2(S)]
  J[384] -= dqdci; // dwdot[OH]/d[CH2(S)]
  J[390] -= dqdci; // dwdot[CH3]/d[CH2(S)]
  J[392] += dqdci; // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[716] += dqdT; // dwdot[CH2(S)]/dT

  // reaction 61: CH3 + OH <=> CH2O + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 5374 * exp(0.287 * logT - (140.900666425322) * invT);
  dlnkfdT = 0.287 * invT + (140.900666425322) * invT2;
  // reverse
  phi_r = sc[1] * sc[20];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[6] * inv_ge[12] * ge[20])
             : exp(-g_RT[1] + g_RT[6] + g_RT[12] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[1] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[20] += q; // CH2O
  // d()/d[H2]
  dqdci = -k_r * sc[20];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[33] -= dqdci; // dwdot[OH]/d[H2]
  J[39] -= dqdci; // dwdot[CH3]/d[H2]
  J[47] += dqdci; // dwdot[CH2O]/d[H2]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[163] += dqdci; // dwdot[H2]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[325] += dqdci; // dwdot[H2]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH2O]
  dqdci = -k_r * sc[1];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 62: CH3 + OH <=> CH2OH + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 15250000 * exp(0.134 * logT - (2838.64521180442) * invT);
  dlnkfdT = 0.134 * invT + (2838.64521180442) * invT2;
  // reverse
  phi_r = sc[2] * sc[19];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[19] * ge[2] * inv_ge[12])
             : exp(-g_RT[2] + g_RT[6] + g_RT[12] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[2] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[19] += q; // CH2OH
  // d()/d[H]
  dqdci = -k_r * sc[19];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[66] -= dqdci; // dwdot[CH3]/d[H]
  J[73] += dqdci; // dwdot[CH2OH]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[181] += dqdci; // dwdot[CH2OH]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[326] += dqdci; // dwdot[H]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[343] += dqdci; // dwdot[CH2OH]/d[CH3]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[2];
  J[515] += dqdci; // dwdot[H]/d[CH2OH]
  J[519] -= dqdci; // dwdot[OH]/d[CH2OH]
  J[525] -= dqdci; // dwdot[CH3]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 63: CH3 + OH <=> CH3O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 1798 * exp(0.965 * logT - (6068.79298960492) * invT);
  dlnkfdT = 0.965 * invT + (6068.79298960492) * invT2;
  // reverse
  phi_r = sc[2] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[18] * inv_ge[12] * ge[2])
             : exp(-g_RT[2] + g_RT[6] + g_RT[12] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[2] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[18] += q; // CH3O
  // d()/d[H]
  dqdci = -k_r * sc[18];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[66] -= dqdci; // dwdot[CH3]/d[H]
  J[72] += dqdci; // dwdot[CH3O]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[180] += dqdci; // dwdot[CH3O]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[326] += dqdci; // dwdot[H]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3O]
  dqdci = -k_r * sc[2];
  J[488] += dqdci; // dwdot[H]/d[CH3O]
  J[492] -= dqdci; // dwdot[OH]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 64: CH3 + OH <=> CH2 + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[12];
  k_f = 0.04293 * exp(2.568 * logT - (2011.75958655411) * invT);
  dlnkfdT = 2.568 * invT + (2011.75958655411) * invT2;
  // reverse
  phi_r = sc[5] * sc[13];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[12] * ge[13] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[12] - g_RT[13]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[12]) + (h_RT[5] + h_RT[13]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[12] -= q; // CH3
  wdot[13] += q; // CH2
  // d()/d[H2O]
  dqdci = -k_r * sc[13];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[147] -= dqdci; // dwdot[CH3]/d[H2O]
  J[148] += dqdci; // dwdot[CH2]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[12];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[175] += dqdci; // dwdot[CH2]/d[OH]
  // d()/d[CH3]
  dqdci = +k_f * sc[6];
  J[329] += dqdci; // dwdot[H2O]/d[CH3]
  J[330] -= dqdci; // dwdot[OH]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[337] += dqdci; // dwdot[CH2]/d[CH3]
  // d()/d[CH2]
  dqdci = -k_r * sc[5];
  J[356] += dqdci; // dwdot[H2O]/d[CH2]
  J[357] -= dqdci; // dwdot[OH]/d[CH2]
  J[363] -= dqdci; // dwdot[CH3]/d[CH2]
  J[364] += dqdci; // dwdot[CH2]/d[CH2]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[715] += dqdT; // dwdot[CH2]/dT

  // reaction 65: CH3 + HO2 <=> CH3O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[12];
  k_f = 1000000 * exp(0.269 * logT - (-345.961457740745) * invT);
  dlnkfdT = 0.269 * invT + (-345.961457740745) * invT2;
  // reverse
  phi_r = sc[6] * sc[18];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[8] * inv_ge[12] * ge[18])
             : exp(-g_RT[6] + g_RT[8] + g_RT[12] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[12]) + (h_RT[6] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[6] += q;  // OH
  wdot[8] -= q;  // HO2
  wdot[12] -= q; // CH3
  wdot[18] += q; // CH3O
  // d()/d[OH]
  dqdci = -k_r * sc[18];
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[170] -= dqdci; // dwdot[HO2]/d[OH]
  J[174] -= dqdci; // dwdot[CH3]/d[OH]
  J[180] += dqdci; // dwdot[CH3O]/d[OH]
  // d()/d[HO2]
  dqdci = +k_f * sc[12];
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[228] -= dqdci; // dwdot[CH3]/d[HO2]
  J[234] += dqdci; // dwdot[CH3O]/d[HO2]
  // d()/d[CH3]
  dqdci = +k_f * sc[8];
  J[330] += dqdci; // dwdot[OH]/d[CH3]
  J[332] -= dqdci; // dwdot[HO2]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3O]
  dqdci = -k_r * sc[6];
  J[492] += dqdci; // dwdot[OH]/d[CH3O]
  J[494] -= dqdci; // dwdot[HO2]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 68: CH3OH + H <=> CH3O + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[17];
  k_f = 0.199 * exp(2.56 * logT - (5183.13165778861) * invT);
  dlnkfdT = 2.56 * invT + (5183.13165778861) * invT2;
  // reverse
  phi_r = sc[1] * sc[18];
  Kc =
    (kc_fast ? (ge[1] * ge[18] * inv_ge[17] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[17]) + (h_RT[1] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[H2]
  dqdci = -k_r * sc[18];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[44] -= dqdci; // dwdot[CH3OH]/d[H2]
  J[45] += dqdci; // dwdot[CH3O]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[17];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[71] -= dqdci; // dwdot[CH3OH]/d[H]
  J[72] += dqdci; // dwdot[CH3O]/d[H]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[2];
  J[460] += dqdci; // dwdot[H2]/d[CH3OH]
  J[461] -= dqdci; // dwdot[H]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[1];
  J[487] += dqdci; // dwdot[H2]/d[CH3O]
  J[488] -= dqdci; // dwdot[H]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 69: CH3OH + H <=> CH2OH + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[17];
  k_f = 0.307 * exp(2.55 * logT - (2737.49866197768) * invT);
  dlnkfdT = 2.55 * invT + (2737.49866197768) * invT2;
  // reverse
  phi_r = sc[1] * sc[19];
  Kc =
    (kc_fast ? (ge[1] * ge[19] * inv_ge[17] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[17]) + (h_RT[1] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[H2]
  dqdci = -k_r * sc[19];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[44] -= dqdci; // dwdot[CH3OH]/d[H2]
  J[46] += dqdci; // dwdot[CH2OH]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[17];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[71] -= dqdci; // dwdot[CH3OH]/d[H]
  J[73] += dqdci; // dwdot[CH2OH]/d[H]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[2];
  J[460] += dqdci; // dwdot[H2]/d[CH3OH]
  J[461] -= dqdci; // dwdot[H]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[1];
  J[514] += dqdci; // dwdot[H2]/d[CH2OH]
  J[515] -= dqdci; // dwdot[H]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 70: CH3OH + O <=> CH3O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[17];
  k_f = 0.0388 * exp(2.5 * logT - (1549.90733067854) * invT);
  dlnkfdT = 2.5 * invT + (1549.90733067854) * invT2;
  // reverse
  phi_r = sc[6] * sc[18];
  Kc =
    (kc_fast ? (ge[6] * ge[18] * inv_ge[17] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[17]) + (h_RT[6] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[O]
  dqdci = +k_f * sc[17];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[125] -= dqdci; // dwdot[CH3OH]/d[O]
  J[126] += dqdci; // dwdot[CH3O]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[18];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
  J[180] += dqdci; // dwdot[CH3O]/d[OH]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[4];
  J[463] -= dqdci; // dwdot[O]/d[CH3OH]
  J[465] += dqdci; // dwdot[OH]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[6];
  J[490] -= dqdci; // dwdot[O]/d[CH3O]
  J[492] += dqdci; // dwdot[OH]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 71: CH3OH + O <=> CH2OH + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[17];
  k_f = 0.388 * exp(2.5 * logT - (1549.90733067854) * invT);
  dlnkfdT = 2.5 * invT + (1549.90733067854) * invT2;
  // reverse
  phi_r = sc[6] * sc[19];
  Kc =
    (kc_fast ? (ge[6] * ge[19] * inv_ge[17] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[17]) + (h_RT[6] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[O]
  dqdci = +k_f * sc[17];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[125] -= dqdci; // dwdot[CH3OH]/d[O]
  J[127] += dqdci; // dwdot[CH2OH]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[19];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
  J[181] += dqdci; // dwdot[CH2OH]/d[OH]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[4];
  J[463] -= dqdci; // dwdot[O]/d[CH3OH]
  J[465] += dqdci; // dwdot[OH]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[6];
  J[517] -= dqdci; // dwdot[O]/d[CH2OH]
  J[519] += dqdci; // dwdot[OH]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 72: CH3OH + OH <=> CH3O + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[17];
  k_f = 0.00015 * exp(3.03 * logT - (-383.954316009001) * invT);
  dlnkfdT = 3.03 * invT + (-383.954316009001) * invT2;
  // reverse
  phi_r = sc[5] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[18] * inv_ge[17] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[17]) + (h_RT[5] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[H2O]
  dqdci = -k_r * sc[18];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[152] -= dqdci; // dwdot[CH3OH]/d[H2O]
  J[153] += dqdci; // dwdot[CH3O]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[17];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
  J[180] += dqdci; // dwdot[CH3O]/d[OH]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[6];
  J[464] += dqdci; // dwdot[H2O]/d[CH3OH]
  J[465] -= dqdci; // dwdot[OH]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[5];
  J[491] += dqdci; // dwdot[H2O]/d[CH3O]
  J[492] -= dqdci; // dwdot[OH]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 73: CH3OH + OH <=> CH2OH + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[17];
  k_f = 0.0308 * exp(2.65 * logT - (-405.944884304667) * invT);
  dlnkfdT = 2.65 * invT + (-405.944884304667) * invT2;
  // reverse
  phi_r = sc[5] * sc[19];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[19] * inv_ge[17] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[17]) + (h_RT[5] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[H2O]
  dqdci = -k_r * sc[19];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[152] -= dqdci; // dwdot[CH3OH]/d[H2O]
  J[154] += dqdci; // dwdot[CH2OH]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[17];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[179] -= dqdci; // dwdot[CH3OH]/d[OH]
  J[181] += dqdci; // dwdot[CH2OH]/d[OH]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[6];
  J[464] += dqdci; // dwdot[H2O]/d[CH3OH]
  J[465] -= dqdci; // dwdot[OH]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[5];
  J[518] += dqdci; // dwdot[H2O]/d[CH2OH]
  J[519] -= dqdci; // dwdot[OH]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 74: CH3OH + O2 <=> CH3O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[17];
  k_f = 0.0358 * exp(2.27 * logT - (21519.8091048059) * invT);
  dlnkfdT = 2.27 * invT + (21519.8091048059) * invT2;
  // reverse
  phi_r = sc[8] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * ge[18] * inv_ge[17])
             : exp(g_RT[3] - g_RT[8] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[17]) + (h_RT[8] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[O2]
  dqdci = +k_f * sc[17];
  J[84] -= dqdci; // dwdot[O2]/d[O2]
  J[89] += dqdci; // dwdot[HO2]/d[O2]
  J[98] -= dqdci; // dwdot[CH3OH]/d[O2]
  J[99] += dqdci; // dwdot[CH3O]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[18];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[233] -= dqdci; // dwdot[CH3OH]/d[HO2]
  J[234] += dqdci; // dwdot[CH3O]/d[HO2]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[3];
  J[462] -= dqdci; // dwdot[O2]/d[CH3OH]
  J[467] += dqdci; // dwdot[HO2]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[8];
  J[489] -= dqdci; // dwdot[O2]/d[CH3O]
  J[494] += dqdci; // dwdot[HO2]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 75: CH3OH + HO2 <=> CH3O + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[17];
  k_f = 1220000 * exp(-(10099.9107343668) * invT);
  dlnkfdT = (10099.9107343668) * invT2;
  // reverse
  phi_r = sc[7] * sc[18];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[18] * ge[7] * inv_ge[17])
             : exp(-g_RT[7] + g_RT[8] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[17]) + (h_RT[7] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[H2O2]
  dqdci = -k_r * sc[18];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[206] -= dqdci; // dwdot[CH3OH]/d[H2O2]
  J[207] += dqdci; // dwdot[CH3O]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[17];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[233] -= dqdci; // dwdot[CH3OH]/d[HO2]
  J[234] += dqdci; // dwdot[CH3O]/d[HO2]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[8];
  J[466] += dqdci; // dwdot[H2O2]/d[CH3OH]
  J[467] -= dqdci; // dwdot[HO2]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[7];
  J[493] += dqdci; // dwdot[H2O2]/d[CH3O]
  J[494] -= dqdci; // dwdot[HO2]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 76: CH3OH + HO2 <=> CH2OH + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[17];
  k_f = 32600000 * exp(-(9451.5160604774) * invT);
  dlnkfdT = (9451.5160604774) * invT2;
  // reverse
  phi_r = sc[7] * sc[19];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[19] * ge[7] * inv_ge[17])
             : exp(-g_RT[7] + g_RT[8] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[17]) + (h_RT[7] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[H2O2]
  dqdci = -k_r * sc[19];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[206] -= dqdci; // dwdot[CH3OH]/d[H2O2]
  J[208] += dqdci; // dwdot[CH2OH]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[17];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[233] -= dqdci; // dwdot[CH3OH]/d[HO2]
  J[235] += dqdci; // dwdot[CH2OH]/d[HO2]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[8];
  J[466] += dqdci; // dwdot[H2O2]/d[CH3OH]
  J[467] -= dqdci; // dwdot[HO2]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[7];
  J[520] += dqdci; // dwdot[H2O2]/d[CH2OH]
  J[521] -= dqdci; // dwdot[HO2]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 77: CH3 + CH3OH <=> CH2OH + CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[17];
  k_f = 2.13e-07 * exp(3.953 * logT - (3550.24389891888) * invT);
  dlnkfdT = 3.953 * invT + (3550.24389891888) * invT2;
  // reverse
  phi_r = sc[11] * sc[19];
  Kc =
    (kc_fast ? (ge[19] * inv_ge[12] * inv_ge[17] * ge[11])
             : exp(-g_RT[11] + g_RT[12] + g_RT[17] - g_RT[19]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[17]) + (h_RT[11] + h_RT[19]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[17] -= q; // CH3OH
  wdot[19] += q; // CH2OH
  // d()/d[CH4]
  dqdci = -k_r * sc[19];
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[314] -= dqdci; // dwdot[CH3OH]/d[CH4]
  J[316] += dqdci; // dwdot[CH2OH]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[17];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[341] -= dqdci; // dwdot[CH3OH]/d[CH3]
  J[343] += dqdci; // dwdot[CH2OH]/d[CH3]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[12];
  J[470] += dqdci; // dwdot[CH4]/d[CH3OH]
  J[471] -= dqdci; // dwdot[CH3]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[478] += dqdci; // dwdot[CH2OH]/d[CH3OH]
  // d()/d[CH2OH]
  dqdci = -k_r * sc[11];
  J[524] += dqdci; // dwdot[CH4]/d[CH2OH]
  J[525] -= dqdci; // dwdot[CH3]/d[CH2OH]
  J[530] -= dqdci; // dwdot[CH3OH]/d[CH2OH]
  J[532] += dqdci; // dwdot[CH2OH]/d[CH2OH]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[721] += dqdT; // dwdot[CH2OH]/dT

  // reaction 78: CH3 + CH3OH <=> CH3O + CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[17];
  k_f = 0.00322 * exp(2.425 * logT - (4317.34738427159) * invT);
  dlnkfdT = 2.425 * invT + (4317.34738427159) * invT2;
  // reverse
  phi_r = sc[11] * sc[18];
  Kc =
    (kc_fast ? (ge[18] * inv_ge[12] * inv_ge[17] * ge[11])
             : exp(-g_RT[11] + g_RT[12] + g_RT[17] - g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[17]) + (h_RT[11] + h_RT[18]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[17] -= q; // CH3OH
  wdot[18] += q; // CH3O
  // d()/d[CH4]
  dqdci = -k_r * sc[18];
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[314] -= dqdci; // dwdot[CH3OH]/d[CH4]
  J[315] += dqdci; // dwdot[CH3O]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[17];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[341] -= dqdci; // dwdot[CH3OH]/d[CH3]
  J[342] += dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[12];
  J[470] += dqdci; // dwdot[CH4]/d[CH3OH]
  J[471] -= dqdci; // dwdot[CH3]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] += dqdci; // dwdot[CH3O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = -k_r * sc[11];
  J[497] += dqdci; // dwdot[CH4]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[503] -= dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] += dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT
  J[720] += dqdT; // dwdot[CH3O]/dT

  // reaction 79: CH2OH + O2 <=> CH2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[19];
  k_f = 1510000000 * exp(-1 * logT);
  dlnkfdT = -1 * invT;
  // reverse
  phi_r = sc[8] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[19] * ge[20])
             : exp(g_RT[3] - g_RT[8] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[19]) + (h_RT[8] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[O2]
  dqdci = +k_f * sc[19];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[89] += dqdci;  // dwdot[HO2]/d[O2]
  J[100] -= dqdci; // dwdot[CH2OH]/d[O2]
  J[101] += dqdci; // dwdot[CH2O]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[20];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[235] -= dqdci; // dwdot[CH2OH]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[3];
  J[516] -= dqdci; // dwdot[O2]/d[CH2OH]
  J[521] += dqdci; // dwdot[HO2]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[8];
  J[543] -= dqdci; // dwdot[O2]/d[CH2O]
  J[548] += dqdci; // dwdot[HO2]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 80: CH2OH + O2 <=> CH2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[19];
  k_f = 241000000 * exp(-(2524.63801234228) * invT);
  dlnkfdT = (2524.63801234228) * invT2;
  // reverse
  phi_r = sc[8] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[19] * ge[20])
             : exp(g_RT[3] - g_RT[8] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[19]) + (h_RT[8] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[O2]
  dqdci = +k_f * sc[19];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[89] += dqdci;  // dwdot[HO2]/d[O2]
  J[100] -= dqdci; // dwdot[CH2OH]/d[O2]
  J[101] += dqdci; // dwdot[CH2O]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[20];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[235] -= dqdci; // dwdot[CH2OH]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[3];
  J[516] -= dqdci; // dwdot[O2]/d[CH2OH]
  J[521] += dqdci; // dwdot[HO2]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[8];
  J[543] -= dqdci; // dwdot[O2]/d[CH2O]
  J[548] += dqdci; // dwdot[HO2]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 81: CH2OH + H <=> CH2O + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[19];
  k_f = 6000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[20];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[19] * ge[20] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[19]) + (h_RT[1] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[H2]
  dqdci = -k_r * sc[20];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[46] -= dqdci; // dwdot[CH2OH]/d[H2]
  J[47] += dqdci; // dwdot[CH2O]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[19];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[73] -= dqdci; // dwdot[CH2OH]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[2];
  J[514] += dqdci; // dwdot[H2]/d[CH2OH]
  J[515] -= dqdci; // dwdot[H]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[1];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[542] -= dqdci; // dwdot[H]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 82: CH2OH + HO2 <=> CH2O + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[19];
  k_f = 12000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[7] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[20] * inv_ge[19] * ge[7])
             : exp(-g_RT[7] + g_RT[8] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[19]) + (h_RT[7] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[H2O2]
  dqdci = -k_r * sc[20];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[208] -= dqdci; // dwdot[CH2OH]/d[H2O2]
  J[209] += dqdci; // dwdot[CH2O]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[19];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[235] -= dqdci; // dwdot[CH2OH]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[8];
  J[520] += dqdci; // dwdot[H2O2]/d[CH2OH]
  J[521] -= dqdci; // dwdot[HO2]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[7];
  J[547] += dqdci; // dwdot[H2O2]/d[CH2O]
  J[548] -= dqdci; // dwdot[HO2]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 83: CH2OH + HCO <=> 2 CH2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[19] * sc[21];
  k_f = 180000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = (sc[20] * sc[20]);
  Kc =
    (kc_fast ? (inv_ge[21] * ge[20] * inv_ge[19] * ge[20])
             : exp(g_RT[19] - 2.000000 * g_RT[20] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[19] + h_RT[21]) + (2.000000 * h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[19] -= q;     // CH2OH
  wdot[20] += 2 * q; // CH2O
  wdot[21] -= q;     // HCO
  // d()/d[CH2OH]
  dqdci = +k_f * sc[21];
  J[532] -= dqdci;     // dwdot[CH2OH]/d[CH2OH]
  J[533] += 2 * dqdci; // dwdot[CH2O]/d[CH2OH]
  J[534] -= dqdci;     // dwdot[HCO]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * 2.000000 * sc[20];
  J[559] -= dqdci;     // dwdot[CH2OH]/d[CH2O]
  J[560] += 2 * dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] -= dqdci;     // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = +k_f * sc[19];
  J[586] -= dqdci;     // dwdot[CH2OH]/d[HCO]
  J[587] += 2 * dqdci; // dwdot[CH2O]/d[HCO]
  J[588] -= dqdci;     // dwdot[HCO]/d[HCO]
  // d()/dT
  J[721] -= dqdT;     // dwdot[CH2OH]/dT
  J[722] += 2 * dqdT; // dwdot[CH2O]/dT
  J[723] -= dqdT;     // dwdot[HCO]/dT

  // reaction 84: CH2OH + OH <=> CH2O + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[19];
  k_f = 24000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[19] * ge[20] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[19]) + (h_RT[5] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[H2O]
  dqdci = -k_r * sc[20];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[154] -= dqdci; // dwdot[CH2OH]/d[H2O]
  J[155] += dqdci; // dwdot[CH2O]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[19];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[181] -= dqdci; // dwdot[CH2OH]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[6];
  J[518] += dqdci; // dwdot[H2O]/d[CH2OH]
  J[519] -= dqdci; // dwdot[OH]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[5];
  J[545] += dqdci; // dwdot[H2O]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 85: CH2OH + O <=> CH2O + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[19];
  k_f = 42000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[6] * sc[20];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[19] * ge[20] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[19] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[19]) + (h_RT[6] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[O]
  dqdci = +k_f * sc[19];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[127] -= dqdci; // dwdot[CH2OH]/d[O]
  J[128] += dqdci; // dwdot[CH2O]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[20];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[181] -= dqdci; // dwdot[CH2OH]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[4];
  J[517] -= dqdci; // dwdot[O]/d[CH2OH]
  J[519] += dqdci; // dwdot[OH]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/d[CH2O]
  dqdci = -k_r * sc[6];
  J[544] -= dqdci; // dwdot[O]/d[CH2O]
  J[546] += dqdci; // dwdot[OH]/d[CH2O]
  J[559] -= dqdci; // dwdot[CH2OH]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 86: CH3O + O2 <=> CH2O + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[18];
  k_f = 4.38e-25 * exp(9.5 * logT - (-2768.19487859176) * invT);
  dlnkfdT = 9.5 * invT + (-2768.19487859176) * invT2;
  // reverse
  phi_r = sc[8] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[18] * ge[20])
             : exp(g_RT[3] - g_RT[8] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[18]) + (h_RT[8] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[O2]
  dqdci = +k_f * sc[18];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[89] += dqdci;  // dwdot[HO2]/d[O2]
  J[99] -= dqdci;  // dwdot[CH3O]/d[O2]
  J[101] += dqdci; // dwdot[CH2O]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[20];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[234] -= dqdci; // dwdot[CH3O]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH3O]
  dqdci = +k_f * sc[3];
  J[489] -= dqdci; // dwdot[O2]/d[CH3O]
  J[494] += dqdci; // dwdot[HO2]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[8];
  J[543] -= dqdci; // dwdot[O2]/d[CH2O]
  J[548] += dqdci; // dwdot[HO2]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 87: CH3O + H <=> CH2O + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[18];
  k_f = 20000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[20];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[18] * ge[20] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[18]) + (h_RT[1] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[H2]
  dqdci = -k_r * sc[20];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[45] -= dqdci; // dwdot[CH3O]/d[H2]
  J[47] += dqdci; // dwdot[CH2O]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[18];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[72] -= dqdci; // dwdot[CH3O]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[CH3O]
  dqdci = +k_f * sc[2];
  J[487] += dqdci; // dwdot[H2]/d[CH3O]
  J[488] -= dqdci; // dwdot[H]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[1];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[542] -= dqdci; // dwdot[H]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 88: CH3O + HO2 <=> CH2O + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[18];
  k_f = 301000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[7] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[8] * ge[20] * inv_ge[18] * ge[7])
             : exp(-g_RT[7] + g_RT[8] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[18]) + (h_RT[7] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[H2O2]
  dqdci = -k_r * sc[20];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[207] -= dqdci; // dwdot[CH3O]/d[H2O2]
  J[209] += dqdci; // dwdot[CH2O]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[18];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[234] -= dqdci; // dwdot[CH3O]/d[HO2]
  J[236] += dqdci; // dwdot[CH2O]/d[HO2]
  // d()/d[CH3O]
  dqdci = +k_f * sc[8];
  J[493] += dqdci; // dwdot[H2O2]/d[CH3O]
  J[494] -= dqdci; // dwdot[HO2]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[7];
  J[547] += dqdci; // dwdot[H2O2]/d[CH2O]
  J[548] -= dqdci; // dwdot[HO2]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 89: CH3 + CH3O <=> CH2O + CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[18];
  k_f = 12000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[11] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[18] * ge[11] * inv_ge[12] * ge[20])
             : exp(-g_RT[11] + g_RT[12] + g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[18]) + (h_RT[11] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[CH4]
  dqdci = -k_r * sc[20];
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[315] -= dqdci; // dwdot[CH3O]/d[CH4]
  J[317] += dqdci; // dwdot[CH2O]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[18];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[342] -= dqdci; // dwdot[CH3O]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH3O]
  dqdci = +k_f * sc[12];
  J[497] += dqdci; // dwdot[CH4]/d[CH3O]
  J[498] -= dqdci; // dwdot[CH3]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[11];
  J[551] += dqdci; // dwdot[CH4]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 90: 2 CH3O <=> CH2O + CH3OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[18] * sc[18]);
  k_f = 60300000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[17] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[18] * ge[20] * inv_ge[18] * ge[17])
             : exp(-g_RT[17] + 2.000000 * g_RT[18] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[18]) + (h_RT[17] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[17] += q;     // CH3OH
  wdot[18] -= 2 * q; // CH3O
  wdot[20] += q;     // CH2O
  // d()/d[CH3OH]
  dqdci = -k_r * sc[20];
  J[476] += dqdci;      // dwdot[CH3OH]/d[CH3OH]
  J[477] += -2 * dqdci; // dwdot[CH3O]/d[CH3OH]
  J[479] += dqdci;      // dwdot[CH2O]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = +k_f * 2.000000 * sc[18];
  J[503] += dqdci;      // dwdot[CH3OH]/d[CH3O]
  J[504] += -2 * dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci;      // dwdot[CH2O]/d[CH3O]
  // d()/d[CH2O]
  dqdci = -k_r * sc[17];
  J[557] += dqdci;      // dwdot[CH3OH]/d[CH2O]
  J[558] += -2 * dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] += dqdci;      // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[719] += dqdT;      // dwdot[CH3OH]/dT
  J[720] += -2 * dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT;      // dwdot[CH2O]/dT

  // reaction 91: CH2O + O <=> HCO + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[20];
  k_f = 6260 * exp(1.15 * logT - (1137.26966471867) * invT);
  dlnkfdT = 1.15 * invT + (1137.26966471867) * invT2;
  // reverse
  phi_r = sc[6] * sc[21];
  Kc =
    (kc_fast ? (ge[6] * ge[21] * inv_ge[20] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[20]) + (h_RT[6] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[O]
  dqdci = +k_f * sc[20];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[128] -= dqdci; // dwdot[CH2O]/d[O]
  J[129] += dqdci; // dwdot[HCO]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[21];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[182] -= dqdci; // dwdot[CH2O]/d[OH]
  J[183] += dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CH2O]
  dqdci = +k_f * sc[4];
  J[544] -= dqdci; // dwdot[O]/d[CH2O]
  J[546] += dqdci; // dwdot[OH]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[6];
  J[571] -= dqdci; // dwdot[O]/d[HCO]
  J[573] += dqdci; // dwdot[OH]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 92: CH2O + H <=> H2 + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[20];
  k_f = 57.4 * exp(1.9 * logT - (1378.81366430493) * invT);
  dlnkfdT = 1.9 * invT + (1378.81366430493) * invT2;
  // reverse
  phi_r = sc[1] * sc[21];
  Kc =
    (kc_fast ? (ge[1] * ge[21] * inv_ge[20] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[20]) + (h_RT[1] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[H2]
  dqdci = -k_r * sc[21];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[47] -= dqdci; // dwdot[CH2O]/d[H2]
  J[48] += dqdci; // dwdot[HCO]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[20];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[74] -= dqdci; // dwdot[CH2O]/d[H]
  J[75] += dqdci; // dwdot[HCO]/d[H]
  // d()/d[CH2O]
  dqdci = +k_f * sc[2];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[542] -= dqdci; // dwdot[H]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[1];
  J[568] += dqdci; // dwdot[H2]/d[HCO]
  J[569] -= dqdci; // dwdot[H]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 93: CH2O + OH <=> H2O + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[20];
  k_f = 78.2 * exp(1.63 * logT - (-530.893582423979) * invT);
  dlnkfdT = 1.63 * invT + (-530.893582423979) * invT2;
  // reverse
  phi_r = sc[5] * sc[21];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[21] * inv_ge[20] * ge[5])
             : exp(-g_RT[5] + g_RT[6] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[20]) + (h_RT[5] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[H2O]
  dqdci = -k_r * sc[21];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[155] -= dqdci; // dwdot[CH2O]/d[H2O]
  J[156] += dqdci; // dwdot[HCO]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[20];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[182] -= dqdci; // dwdot[CH2O]/d[OH]
  J[183] += dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CH2O]
  dqdci = +k_f * sc[6];
  J[545] += dqdci; // dwdot[H2O]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[5];
  J[572] += dqdci; // dwdot[H2O]/d[HCO]
  J[573] -= dqdci; // dwdot[OH]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 94: CH2O + HO2 <=> H2O2 + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[20];
  k_f = 0.0188 * exp(2.7 * logT - (5797.05599007037) * invT);
  dlnkfdT = 2.7 * invT + (5797.05599007037) * invT2;
  // reverse
  phi_r = sc[7] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * inv_ge[8] * ge[7] * inv_ge[20])
             : exp(-g_RT[7] + g_RT[8] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[20]) + (h_RT[7] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[H2O2]
  dqdci = -k_r * sc[21];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[209] -= dqdci; // dwdot[CH2O]/d[H2O2]
  J[210] += dqdci; // dwdot[HCO]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[20];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[236] -= dqdci; // dwdot[CH2O]/d[HO2]
  J[237] += dqdci; // dwdot[HCO]/d[HO2]
  // d()/d[CH2O]
  dqdci = +k_f * sc[8];
  J[547] += dqdci; // dwdot[H2O2]/d[CH2O]
  J[548] -= dqdci; // dwdot[HO2]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[7];
  J[574] += dqdci; // dwdot[H2O2]/d[HCO]
  J[575] -= dqdci; // dwdot[HO2]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 95: CH2O + CH3 <=> CH4 + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[20];
  k_f = 3.83e-05 * exp(3.36 * logT - (2169.87026294995) * invT);
  dlnkfdT = 3.36 * invT + (2169.87026294995) * invT2;
  // reverse
  phi_r = sc[11] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * inv_ge[12] * inv_ge[20] * ge[11])
             : exp(-g_RT[11] + g_RT[12] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[20]) + (h_RT[11] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[CH4]
  dqdci = -k_r * sc[21];
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[317] -= dqdci; // dwdot[CH2O]/d[CH4]
  J[318] += dqdci; // dwdot[HCO]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[20];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[344] -= dqdci; // dwdot[CH2O]/d[CH3]
  J[345] += dqdci; // dwdot[HCO]/d[CH3]
  // d()/d[CH2O]
  dqdci = +k_f * sc[12];
  J[551] += dqdci; // dwdot[CH4]/d[CH2O]
  J[552] -= dqdci; // dwdot[CH3]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[11];
  J[578] += dqdci; // dwdot[CH4]/d[HCO]
  J[579] -= dqdci; // dwdot[CH3]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 96: CH2O + CH3O <=> CH3OH + HCO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[18] * sc[20];
  k_f = 662000 * exp(-(1154.37903135603) * invT);
  dlnkfdT = (1154.37903135603) * invT2;
  // reverse
  phi_r = sc[17] * sc[21];
  Kc =
    (kc_fast ? (ge[21] * inv_ge[18] * inv_ge[20] * ge[17])
             : exp(-g_RT[17] + g_RT[18] + g_RT[20] - g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[18] + h_RT[20]) + (h_RT[17] + h_RT[21]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[17] += q; // CH3OH
  wdot[18] -= q; // CH3O
  wdot[20] -= q; // CH2O
  wdot[21] += q; // HCO
  // d()/d[CH3OH]
  dqdci = -k_r * sc[21];
  J[476] += dqdci; // dwdot[CH3OH]/d[CH3OH]
  J[477] -= dqdci; // dwdot[CH3O]/d[CH3OH]
  J[479] -= dqdci; // dwdot[CH2O]/d[CH3OH]
  J[480] += dqdci; // dwdot[HCO]/d[CH3OH]
  // d()/d[CH3O]
  dqdci = +k_f * sc[20];
  J[503] += dqdci; // dwdot[CH3OH]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] -= dqdci; // dwdot[CH2O]/d[CH3O]
  J[507] += dqdci; // dwdot[HCO]/d[CH3O]
  // d()/d[CH2O]
  dqdci = +k_f * sc[18];
  J[557] += dqdci; // dwdot[CH3OH]/d[CH2O]
  J[558] -= dqdci; // dwdot[CH3O]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  J[561] += dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = -k_r * sc[17];
  J[584] += dqdci; // dwdot[CH3OH]/d[HCO]
  J[585] -= dqdci; // dwdot[CH3O]/d[HCO]
  J[587] -= dqdci; // dwdot[CH2O]/d[HCO]
  J[588] += dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[719] += dqdT; // dwdot[CH3OH]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT
  J[723] += dqdT; // dwdot[HCO]/dT

  // reaction 98: HCO + O2 <=> CO + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[3] * sc[21];
  k_f = 7580000 * exp(-(206.318832979935) * invT);
  dlnkfdT = (206.318832979935) * invT2;
  // reverse
  phi_r = sc[8] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[3] * ge[8] * inv_ge[21] * ge[9])
             : exp(g_RT[3] - g_RT[8] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[3] + h_RT[21]) + (h_RT[8] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[3] -= q;  // O2
  wdot[8] += q;  // HO2
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[O2]
  dqdci = +k_f * sc[21];
  J[84] -= dqdci;  // dwdot[O2]/d[O2]
  J[89] += dqdci;  // dwdot[HO2]/d[O2]
  J[90] += dqdci;  // dwdot[CO]/d[O2]
  J[102] -= dqdci; // dwdot[HCO]/d[O2]
  // d()/d[HO2]
  dqdci = -k_r * sc[9];
  J[219] -= dqdci; // dwdot[O2]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[225] += dqdci; // dwdot[CO]/d[HO2]
  J[237] -= dqdci; // dwdot[HCO]/d[HO2]
  // d()/d[CO]
  dqdci = -k_r * sc[8];
  J[246] -= dqdci; // dwdot[O2]/d[CO]
  J[251] += dqdci; // dwdot[HO2]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[3];
  J[570] -= dqdci; // dwdot[O2]/d[HCO]
  J[575] += dqdci; // dwdot[HO2]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[705] -= dqdT; // dwdot[O2]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 99: HCO + O <=> CO + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[21];
  k_f = 30200000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[6] * sc[9];
  Kc =
    (kc_fast ? (ge[6] * inv_ge[21] * ge[9] * inv_ge[4])
             : exp(g_RT[4] - g_RT[6] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[21]) + (h_RT[6] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[6] += q;  // OH
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[O]
  dqdci = +k_f * sc[21];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[114] += dqdci; // dwdot[OH]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[129] -= dqdci; // dwdot[HCO]/d[O]
  // d()/d[OH]
  dqdci = -k_r * sc[9];
  J[166] -= dqdci; // dwdot[O]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[171] += dqdci; // dwdot[CO]/d[OH]
  J[183] -= dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CO]
  dqdci = -k_r * sc[6];
  J[247] -= dqdci; // dwdot[O]/d[CO]
  J[249] += dqdci; // dwdot[OH]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[4];
  J[571] -= dqdci; // dwdot[O]/d[HCO]
  J[573] += dqdci; // dwdot[OH]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 100: H + HCO <=> CO + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[21];
  k_f = 73400000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc =
    (kc_fast ? (ge[1] * inv_ge[21] * ge[9] * inv_ge[2])
             : exp(-g_RT[1] + g_RT[2] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[21]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] -= q;  // H
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[29] -= dqdci; // dwdot[H]/d[H2]
  J[36] += dqdci; // dwdot[CO]/d[H2]
  J[48] -= dqdci; // dwdot[HCO]/d[H2]
  // d()/d[H]
  dqdci = +k_f * sc[21];
  J[55] += dqdci; // dwdot[H2]/d[H]
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[63] += dqdci; // dwdot[CO]/d[H]
  J[75] -= dqdci; // dwdot[HCO]/d[H]
  // d()/d[CO]
  dqdci = -k_r * sc[1];
  J[244] += dqdci; // dwdot[H2]/d[CO]
  J[245] -= dqdci; // dwdot[H]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[2];
  J[568] += dqdci; // dwdot[H2]/d[HCO]
  J[569] -= dqdci; // dwdot[H]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 101: HCO + OH <=> CO + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[21];
  k_f = 30110000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[5] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[6] * inv_ge[21] * ge[9] * ge[5])
             : exp(-g_RT[5] + g_RT[6] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[21]) + (h_RT[5] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[H2O]
  dqdci = -k_r * sc[9];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[144] += dqdci; // dwdot[CO]/d[H2O]
  J[156] -= dqdci; // dwdot[HCO]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[21];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[171] += dqdci; // dwdot[CO]/d[OH]
  J[183] -= dqdci; // dwdot[HCO]/d[OH]
  // d()/d[CO]
  dqdci = -k_r * sc[5];
  J[248] += dqdci; // dwdot[H2O]/d[CO]
  J[249] -= dqdci; // dwdot[OH]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[6];
  J[572] += dqdci; // dwdot[H2O]/d[HCO]
  J[573] -= dqdci; // dwdot[OH]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 102: CH3 + HCO <=> CH4 + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[21];
  k_f = 26500000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[11];
  Kc =
    (kc_fast ? (inv_ge[21] * ge[11] * inv_ge[12] * ge[9])
             : exp(-g_RT[9] - g_RT[11] + g_RT[12] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[12] + h_RT[21]) + (h_RT[9] + h_RT[11]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[21] -= q; // HCO
  // d()/d[CO]
  dqdci = -k_r * sc[11];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[254] += dqdci; // dwdot[CH4]/d[CO]
  J[255] -= dqdci; // dwdot[CH3]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[CH4]
  dqdci = -k_r * sc[9];
  J[306] += dqdci; // dwdot[CO]/d[CH4]
  J[308] += dqdci; // dwdot[CH4]/d[CH4]
  J[309] -= dqdci; // dwdot[CH3]/d[CH4]
  J[318] -= dqdci; // dwdot[HCO]/d[CH4]
  // d()/d[CH3]
  dqdci = +k_f * sc[21];
  J[333] += dqdci; // dwdot[CO]/d[CH3]
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[345] -= dqdci; // dwdot[HCO]/d[CH3]
  // d()/d[HCO]
  dqdci = +k_f * sc[12];
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[578] += dqdci; // dwdot[CH4]/d[HCO]
  J[579] -= dqdci; // dwdot[CH3]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 103: 2 HCO <=> CH2O + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = (sc[21] * sc[21]);
  k_f = 18000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[21] * inv_ge[21] * ge[9] * ge[20])
             : exp(-g_RT[9] - g_RT[20] + 2.000000 * g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(2.000000 * h_RT[21]) + (h_RT[9] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;      // CO
  wdot[20] += q;     // CH2O
  wdot[21] -= 2 * q; // HCO
  // d()/d[CO]
  dqdci = -k_r * sc[20];
  J[252] += dqdci;      // dwdot[CO]/d[CO]
  J[263] += dqdci;      // dwdot[CH2O]/d[CO]
  J[264] += -2 * dqdci; // dwdot[HCO]/d[CO]
  // d()/d[CH2O]
  dqdci = -k_r * sc[9];
  J[549] += dqdci;      // dwdot[CO]/d[CH2O]
  J[560] += dqdci;      // dwdot[CH2O]/d[CH2O]
  J[561] += -2 * dqdci; // dwdot[HCO]/d[CH2O]
  // d()/d[HCO]
  dqdci = +k_f * 2.000000 * sc[21];
  J[576] += dqdci;      // dwdot[CO]/d[HCO]
  J[587] += dqdci;      // dwdot[CH2O]/d[HCO]
  J[588] += -2 * dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[711] += dqdT;      // dwdot[CO]/dT
  J[722] += dqdT;      // dwdot[CH2O]/dT
  J[723] += -2 * dqdT; // dwdot[HCO]/dT

  // reaction 104: HCO + O <=> CO2 + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[21];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[10];
  Kc =
    (kc_fast ? (inv_ge[21] * ge[2] * inv_ge[4] * ge[10])
             : exp(-g_RT[2] + g_RT[4] - g_RT[10] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[21]) + (h_RT[2] + h_RT[10]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[4] -= q;  // O
  wdot[10] += q; // CO2
  wdot[21] -= q; // HCO
  // d()/d[H]
  dqdci = -k_r * sc[10];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci; // dwdot[O]/d[H]
  J[64] += dqdci; // dwdot[CO2]/d[H]
  J[75] -= dqdci; // dwdot[HCO]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[21];
  J[110] += dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[118] += dqdci; // dwdot[CO2]/d[O]
  J[129] -= dqdci; // dwdot[HCO]/d[O]
  // d()/d[CO2]
  dqdci = -k_r * sc[2];
  J[272] += dqdci; // dwdot[H]/d[CO2]
  J[274] -= dqdci; // dwdot[O]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  J[291] -= dqdci; // dwdot[HCO]/d[CO2]
  // d()/d[HCO]
  dqdci = +k_f * sc[4];
  J[569] += dqdci; // dwdot[H]/d[HCO]
  J[571] -= dqdci; // dwdot[O]/d[HCO]
  J[577] += dqdci; // dwdot[CO2]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[712] += dqdT; // dwdot[CO2]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 105: HCO + HO2 => CO2 + H + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[21];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += q;  // H
  wdot[6] += q;  // OH
  wdot[8] -= q;  // HO2
  wdot[10] += q; // CO2
  wdot[21] -= q; // HCO
  // d()/d[HO2]
  dqdci = +k_f * sc[21];
  J[218] += dqdci; // dwdot[H]/d[HO2]
  J[222] += dqdci; // dwdot[OH]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[226] += dqdci; // dwdot[CO2]/d[HO2]
  J[237] -= dqdci; // dwdot[HCO]/d[HO2]
  // d()/d[HCO]
  dqdci = +k_f * sc[8];
  J[569] += dqdci; // dwdot[H]/d[HCO]
  J[573] += dqdci; // dwdot[OH]/d[HCO]
  J[575] -= dqdci; // dwdot[HO2]/d[HCO]
  J[577] += dqdci; // dwdot[CO2]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[712] += dqdT; // dwdot[CO2]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 109: CH + OH <=> C + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[16];
  k_f = 40 * exp(2 * logT - (1509.64999741416) * invT);
  dlnkfdT = 2 * invT + (1509.64999741416) * invT2;
  // reverse
  phi_r = sc[5] * sc[15];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[5] * ge[15] * inv_ge[16])
             : exp(-g_RT[5] + g_RT[6] - g_RT[15] + g_RT[16]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[16]) + (h_RT[5] + h_RT[15]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[15] += q; // C
  wdot[16] -= q; // CH
  // d()/d[H2O]
  dqdci = -k_r * sc[15];
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[141] -= dqdci; // dwdot[OH]/d[H2O]
  J[150] += dqdci; // dwdot[C]/d[H2O]
  J[151] -= dqdci; // dwdot[CH]/d[H2O]
  // d()/d[OH]
  dqdci = +k_f * sc[16];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[177] += dqdci; // dwdot[C]/d[OH]
  J[178] -= dqdci; // dwdot[CH]/d[OH]
  // d()/d[C]
  dqdci = -k_r * sc[5];
  J[410] += dqdci; // dwdot[H2O]/d[C]
  J[411] -= dqdci; // dwdot[OH]/d[C]
  J[420] += dqdci; // dwdot[C]/d[C]
  J[421] -= dqdci; // dwdot[CH]/d[C]
  // d()/d[CH]
  dqdci = +k_f * sc[6];
  J[437] += dqdci; // dwdot[H2O]/d[CH]
  J[438] -= dqdci; // dwdot[OH]/d[CH]
  J[447] += dqdci; // dwdot[C]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[717] += dqdT; // dwdot[C]/dT
  J[718] -= dqdT; // dwdot[CH]/dT

  // reaction 110: CH2 + HCO <=> CH3 + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[13] * sc[21];
  k_f = 20000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[9] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[21] * ge[12] * ge[9] * inv_ge[13])
             : exp(-g_RT[9] - g_RT[12] + g_RT[13] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[13] + h_RT[21]) + (h_RT[9] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[12] += q; // CH3
  wdot[13] -= q; // CH2
  wdot[21] -= q; // HCO
  // d()/d[CO]
  dqdci = -k_r * sc[12];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[255] += dqdci; // dwdot[CH3]/d[CO]
  J[256] -= dqdci; // dwdot[CH2]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[CH3]
  dqdci = -k_r * sc[9];
  J[333] += dqdci; // dwdot[CO]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  J[337] -= dqdci; // dwdot[CH2]/d[CH3]
  J[345] -= dqdci; // dwdot[HCO]/d[CH3]
  // d()/d[CH2]
  dqdci = +k_f * sc[21];
  J[360] += dqdci; // dwdot[CO]/d[CH2]
  J[363] += dqdci; // dwdot[CH3]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[372] -= dqdci; // dwdot[HCO]/d[CH2]
  // d()/d[HCO]
  dqdci = +k_f * sc[13];
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[579] += dqdci; // dwdot[CH3]/d[HCO]
  J[580] -= dqdci; // dwdot[CH2]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 111: CH2 + O <=> CO + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[13];
  k_f = 50000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[1] * sc[9];
  Kc =
    (kc_fast ? (ge[1] * ge[9] * inv_ge[4] * inv_ge[13])
             : exp(-g_RT[1] + g_RT[4] - g_RT[9] + g_RT[13]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[13]) + (h_RT[1] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[4] -= q;  // O
  wdot[9] += q;  // CO
  wdot[13] -= q; // CH2
  // d()/d[H2]
  dqdci = -k_r * sc[9];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[31] -= dqdci; // dwdot[O]/d[H2]
  J[36] += dqdci; // dwdot[CO]/d[H2]
  J[40] -= dqdci; // dwdot[CH2]/d[H2]
  // d()/d[O]
  dqdci = +k_f * sc[13];
  J[109] += dqdci; // dwdot[H2]/d[O]
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[117] += dqdci; // dwdot[CO]/d[O]
  J[121] -= dqdci; // dwdot[CH2]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[1];
  J[244] += dqdci; // dwdot[H2]/d[CO]
  J[247] -= dqdci; // dwdot[O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[256] -= dqdci; // dwdot[CH2]/d[CO]
  // d()/d[CH2]
  dqdci = +k_f * sc[4];
  J[352] += dqdci; // dwdot[H2]/d[CH2]
  J[355] -= dqdci; // dwdot[O]/d[CH2]
  J[360] += dqdci; // dwdot[CO]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT

  // reaction 112: CH2 + OH <=> CH2O + H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[13];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[2] * sc[20];
  Kc =
    (kc_fast ? (inv_ge[6] * ge[20] * ge[2] * inv_ge[13])
             : exp(-g_RT[2] + g_RT[6] + g_RT[13] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[6] + h_RT[13]) + (h_RT[2] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[13] -= q; // CH2
  wdot[20] += q; // CH2O
  // d()/d[H]
  dqdci = -k_r * sc[20];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[60] -= dqdci; // dwdot[OH]/d[H]
  J[67] -= dqdci; // dwdot[CH2]/d[H]
  J[74] += dqdci; // dwdot[CH2O]/d[H]
  // d()/d[OH]
  dqdci = +k_f * sc[13];
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[175] -= dqdci; // dwdot[CH2]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2]
  dqdci = +k_f * sc[6];
  J[353] += dqdci; // dwdot[H]/d[CH2]
  J[357] -= dqdci; // dwdot[OH]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[371] += dqdci; // dwdot[CH2O]/d[CH2]
  // d()/d[CH2O]
  dqdci = -k_r * sc[2];
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[553] -= dqdci; // dwdot[CH2]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 113: CH2 + CO2 <=> CH2O + CO
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[10] * sc[13];
  k_f = 110000 * exp(-(503.21666580472) * invT);
  dlnkfdT = (503.21666580472) * invT2;
  // reverse
  phi_r = sc[9] * sc[20];
  Kc =
    (kc_fast ? (ge[9] * inv_ge[10] * inv_ge[13] * ge[20])
             : exp(-g_RT[9] + g_RT[10] + g_RT[13] - g_RT[20]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[10] + h_RT[13]) + (h_RT[9] + h_RT[20]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] += q;  // CO
  wdot[10] -= q; // CO2
  wdot[13] -= q; // CH2
  wdot[20] += q; // CH2O
  // d()/d[CO]
  dqdci = -k_r * sc[20];
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[253] -= dqdci; // dwdot[CO2]/d[CO]
  J[256] -= dqdci; // dwdot[CH2]/d[CO]
  J[263] += dqdci; // dwdot[CH2O]/d[CO]
  // d()/d[CO2]
  dqdci = +k_f * sc[13];
  J[279] += dqdci; // dwdot[CO]/d[CO2]
  J[280] -= dqdci; // dwdot[CO2]/d[CO2]
  J[283] -= dqdci; // dwdot[CH2]/d[CO2]
  J[290] += dqdci; // dwdot[CH2O]/d[CO2]
  // d()/d[CH2]
  dqdci = +k_f * sc[10];
  J[360] += dqdci; // dwdot[CO]/d[CH2]
  J[361] -= dqdci; // dwdot[CO2]/d[CH2]
  J[364] -= dqdci; // dwdot[CH2]/d[CH2]
  J[371] += dqdci; // dwdot[CH2O]/d[CH2]
  // d()/d[CH2O]
  dqdci = -k_r * sc[9];
  J[549] += dqdci; // dwdot[CO]/d[CH2O]
  J[550] -= dqdci; // dwdot[CO2]/d[CH2O]
  J[553] -= dqdci; // dwdot[CH2]/d[CH2O]
  J[560] += dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[712] -= dqdT; // dwdot[CO2]/dT
  J[715] -= dqdT; // dwdot[CH2]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 114: CH2(S) + O <=> CO + 2 H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[14];
  k_f = 30000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = (sc[2] * sc[2]) * sc[9];
  Kc =
    refC * (kc_fast ? (ge[9] * ge[2] * inv_ge[4] * ge[2] * inv_ge[14])
                    : exp(-2.000000 * g_RT[2] + g_RT[4] - g_RT[9] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[4] + h_RT[14]) + (2.000000 * h_RT[2] + h_RT[9]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[4] -= q;     // O
  wdot[9] += q;     // CO
  wdot[14] -= q;    // CH2(S)
  // d()/d[H]
  dqdci = -k_r * 2.000000 * sc[2] * sc[9];
  J[56] += 2 * dqdci; // dwdot[H]/d[H]
  J[58] -= dqdci;     // dwdot[O]/d[H]
  J[63] += dqdci;     // dwdot[CO]/d[H]
  J[68] -= dqdci;     // dwdot[CH2(S)]/d[H]
  // d()/d[O]
  dqdci = +k_f * sc[14];
  J[110] += 2 * dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci;     // dwdot[O]/d[O]
  J[117] += dqdci;     // dwdot[CO]/d[O]
  J[122] -= dqdci;     // dwdot[CH2(S)]/d[O]
  // d()/d[CO]
  dqdci = -k_r * sc[2] * sc[2];
  J[245] += 2 * dqdci; // dwdot[H]/d[CO]
  J[247] -= dqdci;     // dwdot[O]/d[CO]
  J[252] += dqdci;     // dwdot[CO]/d[CO]
  J[257] -= dqdci;     // dwdot[CH2(S)]/d[CO]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[4];
  J[380] += 2 * dqdci; // dwdot[H]/d[CH2(S)]
  J[382] -= dqdci;     // dwdot[O]/d[CH2(S)]
  J[387] += dqdci;     // dwdot[CO]/d[CH2(S)]
  J[392] -= dqdci;     // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[706] -= dqdT;     // dwdot[O]/dT
  J[711] += dqdT;     // dwdot[CO]/dT
  J[716] -= dqdT;     // dwdot[CH2(S)]/dT

  // reaction 115: CH2(S) + CH4 <=> 2 CH3
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[11] * sc[14];
  k_f = 43000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = (sc[12] * sc[12]);
  Kc =
    (kc_fast ? (ge[12] * ge[12] * inv_ge[14] * inv_ge[11])
             : exp(g_RT[11] - 2.000000 * g_RT[12] + g_RT[14]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[11] + h_RT[14]) + (2.000000 * h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[11] -= q;     // CH4
  wdot[12] += 2 * q; // CH3
  wdot[14] -= q;     // CH2(S)
  // d()/d[CH4]
  dqdci = +k_f * sc[14];
  J[308] -= dqdci;     // dwdot[CH4]/d[CH4]
  J[309] += 2 * dqdci; // dwdot[CH3]/d[CH4]
  J[311] -= dqdci;     // dwdot[CH2(S)]/d[CH4]
  // d()/d[CH3]
  dqdci = -k_r * 2.000000 * sc[12];
  J[335] -= dqdci;     // dwdot[CH4]/d[CH3]
  J[336] += 2 * dqdci; // dwdot[CH3]/d[CH3]
  J[338] -= dqdci;     // dwdot[CH2(S)]/d[CH3]
  // d()/d[CH2(S)]
  dqdci = +k_f * sc[11];
  J[389] -= dqdci;     // dwdot[CH4]/d[CH2(S)]
  J[390] += 2 * dqdci; // dwdot[CH3]/d[CH2(S)]
  J[392] -= dqdci;     // dwdot[CH2(S)]/d[CH2(S)]
  // d()/dT
  J[713] -= dqdT;     // dwdot[CH4]/dT
  J[714] += 2 * dqdT; // dwdot[CH3]/dT
  J[716] -= dqdT;     // dwdot[CH2(S)]/dT

  // reaction 116: CH3OH + H => CH3 + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[17];
  k_f = 650000 * exp(-(2667.04832876501) * invT);
  dlnkfdT = (2667.04832876501) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] -= q;  // H
  wdot[5] += q;  // H2O
  wdot[12] += q; // CH3
  wdot[17] -= q; // CH3OH
  // d()/d[H]
  dqdci = +k_f * sc[17];
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[59] += dqdci; // dwdot[H2O]/d[H]
  J[66] += dqdci; // dwdot[CH3]/d[H]
  J[71] -= dqdci; // dwdot[CH3OH]/d[H]
  // d()/d[CH3OH]
  dqdci = +k_f * sc[2];
  J[461] -= dqdci; // dwdot[H]/d[CH3OH]
  J[464] += dqdci; // dwdot[H2O]/d[CH3OH]
  J[471] += dqdci; // dwdot[CH3]/d[CH3OH]
  J[476] -= dqdci; // dwdot[CH3OH]/d[CH3OH]
  // d()/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[719] -= dqdT; // dwdot[CH3OH]/dT

  // reaction 117: CH2O + O => CO2 + 2 H
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[20];
  k_f = 20000000 * exp(-(2516.0833290236) * invT);
  dlnkfdT = (2516.0833290236) * invT2;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[4] -= q;     // O
  wdot[10] += q;    // CO2
  wdot[20] -= q;    // CH2O
  // d()/d[O]
  dqdci = +k_f * sc[20];
  J[110] += 2 * dqdci; // dwdot[H]/d[O]
  J[112] -= dqdci;     // dwdot[O]/d[O]
  J[118] += dqdci;     // dwdot[CO2]/d[O]
  J[128] -= dqdci;     // dwdot[CH2O]/d[O]
  // d()/d[CH2O]
  dqdci = +k_f * sc[4];
  J[542] += 2 * dqdci; // dwdot[H]/d[CH2O]
  J[544] -= dqdci;     // dwdot[O]/d[CH2O]
  J[550] += dqdci;     // dwdot[CO2]/d[CH2O]
  J[560] -= dqdci;     // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[706] -= dqdT;     // dwdot[O]/dT
  J[712] += dqdT;     // dwdot[CO2]/dT
  J[722] -= dqdT;     // dwdot[CH2O]/dT

  // reaction 118: CH2O + OH => CO2 + H + H2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[20];
  k_f = 100000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[1] += q;  // H2
  wdot[2] += q;  // H
  wdot[6] -= q;  // OH
  wdot[10] += q; // CO2
  wdot[20] -= q; // CH2O
  // d()/d[OH]
  dqdci = +k_f * sc[20];
  J[163] += dqdci; // dwdot[H2]/d[OH]
  J[164] += dqdci; // dwdot[H]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[172] += dqdci; // dwdot[CO2]/d[OH]
  J[182] -= dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH2O]
  dqdci = +k_f * sc[6];
  J[541] += dqdci; // dwdot[H2]/d[CH2O]
  J[542] += dqdci; // dwdot[H]/d[CH2O]
  J[546] -= dqdci; // dwdot[OH]/d[CH2O]
  J[550] += dqdci; // dwdot[CO2]/d[CH2O]
  J[560] -= dqdci; // dwdot[CH2O]/d[CH2O]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[712] += dqdT; // dwdot[CO2]/dT
  J[722] -= dqdT; // dwdot[CH2O]/dT

  // reaction 119: CH3O + CO <=> CH3 + CO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[9] * sc[18];
  k_f = 500000 * exp(-(3270.90832773068) * invT);
  dlnkfdT = (3270.90832773068) * invT2;
  // reverse
  phi_r = sc[10] * sc[12];
  Kc =
    (kc_fast ? (inv_ge[18] * ge[12] * inv_ge[9] * ge[10])
             : exp(g_RT[9] - g_RT[10] - g_RT[12] + g_RT[18]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[9] + h_RT[18]) + (h_RT[10] + h_RT[12]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[9] -= q;  // CO
  wdot[10] += q; // CO2
  wdot[12] += q; // CH3
  wdot[18] -= q; // CH3O
  // d()/d[CO]
  dqdci = +k_f * sc[18];
  J[252] -= dqdci; // dwdot[CO]/d[CO]
  J[253] += dqdci; // dwdot[CO2]/d[CO]
  J[255] += dqdci; // dwdot[CH3]/d[CO]
  J[261] -= dqdci; // dwdot[CH3O]/d[CO]
  // d()/d[CO2]
  dqdci = -k_r * sc[12];
  J[279] -= dqdci; // dwdot[CO]/d[CO2]
  J[280] += dqdci; // dwdot[CO2]/d[CO2]
  J[282] += dqdci; // dwdot[CH3]/d[CO2]
  J[288] -= dqdci; // dwdot[CH3O]/d[CO2]
  // d()/d[CH3]
  dqdci = -k_r * sc[10];
  J[333] -= dqdci; // dwdot[CO]/d[CH3]
  J[334] += dqdci; // dwdot[CO2]/d[CH3]
  J[336] += dqdci; // dwdot[CH3]/d[CH3]
  J[342] -= dqdci; // dwdot[CH3O]/d[CH3]
  // d()/d[CH3O]
  dqdci = +k_f * sc[9];
  J[495] -= dqdci; // dwdot[CO]/d[CH3O]
  J[496] += dqdci; // dwdot[CO2]/d[CH3O]
  J[498] += dqdci; // dwdot[CH3]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  // d()/dT
  J[711] -= dqdT; // dwdot[CO]/dT
  J[712] += dqdT; // dwdot[CO2]/dT
  J[714] += dqdT; // dwdot[CH3]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT

  // reaction 121: HCO + HO2 <=> CO + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[8] * sc[21];
  k_f = 400000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[7] * sc[9];
  Kc =
    (kc_fast ? (inv_ge[21] * inv_ge[8] * ge[9] * ge[7])
             : exp(-g_RT[7] + g_RT[8] - g_RT[9] + g_RT[21]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[8] + h_RT[21]) + (h_RT[7] + h_RT[9]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[7] += q;  // H2O2
  wdot[8] -= q;  // HO2
  wdot[9] += q;  // CO
  wdot[21] -= q; // HCO
  // d()/d[H2O2]
  dqdci = -k_r * sc[9];
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[197] -= dqdci; // dwdot[HO2]/d[H2O2]
  J[198] += dqdci; // dwdot[CO]/d[H2O2]
  J[210] -= dqdci; // dwdot[HCO]/d[H2O2]
  // d()/d[HO2]
  dqdci = +k_f * sc[21];
  J[223] += dqdci; // dwdot[H2O2]/d[HO2]
  J[224] -= dqdci; // dwdot[HO2]/d[HO2]
  J[225] += dqdci; // dwdot[CO]/d[HO2]
  J[237] -= dqdci; // dwdot[HCO]/d[HO2]
  // d()/d[CO]
  dqdci = -k_r * sc[7];
  J[250] += dqdci; // dwdot[H2O2]/d[CO]
  J[251] -= dqdci; // dwdot[HO2]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[264] -= dqdci; // dwdot[HCO]/d[CO]
  // d()/d[HCO]
  dqdci = +k_f * sc[8];
  J[574] += dqdci; // dwdot[H2O2]/d[HCO]
  J[575] -= dqdci; // dwdot[HO2]/d[HCO]
  J[576] += dqdci; // dwdot[CO]/d[HCO]
  J[588] -= dqdci; // dwdot[HCO]/d[HCO]
  // d()/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[710] -= dqdT; // dwdot[HO2]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[723] -= dqdT; // dwdot[HCO]/dT

  // reaction 122: CH3O + OH => CH2O + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[6] * sc[18];
  k_f = 15000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[5] += q;  // H2O
  wdot[6] -= q;  // OH
  wdot[18] -= q; // CH3O
  wdot[20] += q; // CH2O
  // d()/d[OH]
  dqdci = +k_f * sc[18];
  J[167] += dqdci; // dwdot[H2O]/d[OH]
  J[168] -= dqdci; // dwdot[OH]/d[OH]
  J[180] -= dqdci; // dwdot[CH3O]/d[OH]
  J[182] += dqdci; // dwdot[CH2O]/d[OH]
  // d()/d[CH3O]
  dqdci = +k_f * sc[6];
  J[491] += dqdci; // dwdot[H2O]/d[CH3O]
  J[492] -= dqdci; // dwdot[OH]/d[CH3O]
  J[504] -= dqdci; // dwdot[CH3O]/d[CH3O]
  J[506] += dqdci; // dwdot[CH2O]/d[CH3O]
  // d()/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[708] -= dqdT; // dwdot[OH]/dT
  J[720] -= dqdT; // dwdot[CH3O]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 123: CH2OH + CH3 => CH2O + CH4
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[12] * sc[19];
  k_f = 15000000;
  dlnkfdT = 0.0;
  // rate of progress
  q = k_f * phi_f;
  dqdT = dlnkfdT * k_f * phi_f;
  // update wdot
  wdot[11] += q; // CH4
  wdot[12] -= q; // CH3
  wdot[19] -= q; // CH2OH
  wdot[20] += q; // CH2O
  // d()/d[CH3]
  dqdci = +k_f * sc[19];
  J[335] += dqdci; // dwdot[CH4]/d[CH3]
  J[336] -= dqdci; // dwdot[CH3]/d[CH3]
  J[343] -= dqdci; // dwdot[CH2OH]/d[CH3]
  J[344] += dqdci; // dwdot[CH2O]/d[CH3]
  // d()/d[CH2OH]
  dqdci = +k_f * sc[12];
  J[524] += dqdci; // dwdot[CH4]/d[CH2OH]
  J[525] -= dqdci; // dwdot[CH3]/d[CH2OH]
  J[532] -= dqdci; // dwdot[CH2OH]/d[CH2OH]
  J[533] += dqdci; // dwdot[CH2O]/d[CH2OH]
  // d()/dT
  J[713] += dqdT; // dwdot[CH4]/dT
  J[714] -= dqdT; // dwdot[CH3]/dT
  J[721] -= dqdT; // dwdot[CH2OH]/dT
  J[722] += dqdT; // dwdot[CH2O]/dT

  // reaction 124: CH + O <=> CHO+ + E
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[4] * sc[16];
  k_f = 1746000000000 * exp(-2.19 * logT - (164.999209334044) * invT);
  dlnkfdT = -2.19 * invT + (164.999209334044) * invT2;
  // reverse
  phi_r = sc[22] * sc[25];
  Kc =
    (kc_fast ? (ge[25] * inv_ge[4] * ge[22] * inv_ge[16])
             : exp(g_RT[4] + g_RT[16] - g_RT[22] - g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[4] + h_RT[16]) + (h_RT[22] + h_RT[25]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[4] -= q;  // O
  wdot[16] -= q; // CH
  wdot[22] += q; // CHO+
  wdot[25] += q; // E
  // d()/d[O]
  dqdci = +k_f * sc[16];
  J[112] -= dqdci; // dwdot[O]/d[O]
  J[124] -= dqdci; // dwdot[CH]/d[O]
  J[130] += dqdci; // dwdot[CHO+]/d[O]
  J[133] += dqdci; // dwdot[E]/d[O]
  // d()/d[CH]
  dqdci = +k_f * sc[4];
  J[436] -= dqdci; // dwdot[O]/d[CH]
  J[448] -= dqdci; // dwdot[CH]/d[CH]
  J[454] += dqdci; // dwdot[CHO+]/d[CH]
  J[457] += dqdci; // dwdot[E]/d[CH]
  // d()/d[CHO+]
  dqdci = -k_r * sc[25];
  J[598] -= dqdci; // dwdot[O]/d[CHO+]
  J[610] -= dqdci; // dwdot[CH]/d[CHO+]
  J[616] += dqdci; // dwdot[CHO+]/d[CHO+]
  J[619] += dqdci; // dwdot[E]/d[CHO+]
  // d()/d[E]
  dqdci = -k_r * sc[22];
  J[679] -= dqdci; // dwdot[O]/d[E]
  J[691] -= dqdci; // dwdot[CH]/d[E]
  J[697] += dqdci; // dwdot[CHO+]/d[E]
  J[700] += dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[706] -= dqdT; // dwdot[O]/dT
  J[718] -= dqdT; // dwdot[CH]/dT
  J[724] += dqdT; // dwdot[CHO+]/dT
  J[727] += dqdT; // dwdot[E]/dT

  // reaction 125: CHO+ + H2O <=> CO + H3O+
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[5] * sc[22];
  k_f = 26080000000 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = sc[9] * sc[23];
  Kc =
    (kc_fast ? (ge[9] * inv_ge[5] * inv_ge[22] * ge[23])
             : exp(g_RT[5] - g_RT[9] + g_RT[22] - g_RT[23]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[5] + h_RT[22]) + (h_RT[9] + h_RT[23]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[5] -= q;  // H2O
  wdot[9] += q;  // CO
  wdot[22] -= q; // CHO+
  wdot[23] += q; // H3O+
  // d()/d[H2O]
  dqdci = +k_f * sc[22];
  J[140] -= dqdci; // dwdot[H2O]/d[H2O]
  J[144] += dqdci; // dwdot[CO]/d[H2O]
  J[157] -= dqdci; // dwdot[CHO+]/d[H2O]
  J[158] += dqdci; // dwdot[H3O+]/d[H2O]
  // d()/d[CO]
  dqdci = -k_r * sc[23];
  J[248] -= dqdci; // dwdot[H2O]/d[CO]
  J[252] += dqdci; // dwdot[CO]/d[CO]
  J[265] -= dqdci; // dwdot[CHO+]/d[CO]
  J[266] += dqdci; // dwdot[H3O+]/d[CO]
  // d()/d[CHO+]
  dqdci = +k_f * sc[5];
  J[599] -= dqdci; // dwdot[H2O]/d[CHO+]
  J[603] += dqdci; // dwdot[CO]/d[CHO+]
  J[616] -= dqdci; // dwdot[CHO+]/d[CHO+]
  J[617] += dqdci; // dwdot[H3O+]/d[CHO+]
  // d()/d[H3O+]
  dqdci = -k_r * sc[9];
  J[626] -= dqdci; // dwdot[H2O]/d[H3O+]
  J[630] += dqdci; // dwdot[CO]/d[H3O+]
  J[643] -= dqdci; // dwdot[CHO+]/d[H3O+]
  J[644] += dqdci; // dwdot[H3O+]/d[H3O+]
  // d()/dT
  J[707] -= dqdT; // dwdot[H2O]/dT
  J[711] += dqdT; // dwdot[CO]/dT
  J[724] -= dqdT; // dwdot[CHO+]/dT
  J[725] += dqdT; // dwdot[H3O+]/dT

  // reaction 126: E + H3O+ <=> H + H2O
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[23] * sc[25];
  k_f = 739500000000 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = sc[2] * sc[5];
  Kc =
    (kc_fast ? (inv_ge[25] * ge[2] * ge[5] * inv_ge[23])
             : exp(-g_RT[2] - g_RT[5] + g_RT[23] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[23] + h_RT[25]) + (h_RT[2] + h_RT[5]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += q;  // H
  wdot[5] += q;  // H2O
  wdot[23] -= q; // H3O+
  wdot[25] -= q; // E
  // d()/d[H]
  dqdci = -k_r * sc[5];
  J[56] += dqdci; // dwdot[H]/d[H]
  J[59] += dqdci; // dwdot[H2O]/d[H]
  J[77] -= dqdci; // dwdot[H3O+]/d[H]
  J[79] -= dqdci; // dwdot[E]/d[H]
  // d()/d[H2O]
  dqdci = -k_r * sc[2];
  J[137] += dqdci; // dwdot[H]/d[H2O]
  J[140] += dqdci; // dwdot[H2O]/d[H2O]
  J[158] -= dqdci; // dwdot[H3O+]/d[H2O]
  J[160] -= dqdci; // dwdot[E]/d[H2O]
  // d()/d[H3O+]
  dqdci = +k_f * sc[25];
  J[623] += dqdci; // dwdot[H]/d[H3O+]
  J[626] += dqdci; // dwdot[H2O]/d[H3O+]
  J[644] -= dqdci; // dwdot[H3O+]/d[H3O+]
  J[646] -= dqdci; // dwdot[E]/d[H3O+]
  // d()/d[E]
  dqdci = +k_f * sc[23];
  J[677] += dqdci; // dwdot[H]/d[E]
  J[680] += dqdci; // dwdot[H2O]/d[E]
  J[698] -= dqdci; // dwdot[H3O+]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[704] += dqdT; // dwdot[H]/dT
  J[707] += dqdT; // dwdot[H2O]/dT
  J[725] -= dqdT; // dwdot[H3O+]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 127: E + H3O+ <=> 2 H + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[23] * sc[25];
  k_f = 3181000000000 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = (sc[2] * sc[2]) * sc[6];
  Kc =
    refC * (kc_fast ? (inv_ge[25] * ge[6] * ge[2] * inv_ge[23] * ge[2])
                    : exp(-2.000000 * g_RT[2] - g_RT[6] + g_RT[23] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT =
    invT * (-(h_RT[23] + h_RT[25]) + (2.000000 * h_RT[2] + h_RT[6]) - 1.000000);
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] += 2 * q; // H
  wdot[6] += q;     // OH
  wdot[23] -= q;    // H3O+
  wdot[25] -= q;    // E
  // d()/d[H]
  dqdci = -k_r * 2.000000 * sc[2] * sc[6];
  J[56] += 2 * dqdci; // dwdot[H]/d[H]
  J[60] += dqdci;     // dwdot[OH]/d[H]
  J[77] -= dqdci;     // dwdot[H3O+]/d[H]
  J[79] -= dqdci;     // dwdot[E]/d[H]
  // d()/d[OH]
  dqdci = -k_r * sc[2] * sc[2];
  J[164] += 2 * dqdci; // dwdot[H]/d[OH]
  J[168] += dqdci;     // dwdot[OH]/d[OH]
  J[185] -= dqdci;     // dwdot[H3O+]/d[OH]
  J[187] -= dqdci;     // dwdot[E]/d[OH]
  // d()/d[H3O+]
  dqdci = +k_f * sc[25];
  J[623] += 2 * dqdci; // dwdot[H]/d[H3O+]
  J[627] += dqdci;     // dwdot[OH]/d[H3O+]
  J[644] -= dqdci;     // dwdot[H3O+]/d[H3O+]
  J[646] -= dqdci;     // dwdot[E]/d[H3O+]
  // d()/d[E]
  dqdci = +k_f * sc[23];
  J[677] += 2 * dqdci; // dwdot[H]/d[E]
  J[681] += dqdci;     // dwdot[OH]/d[E]
  J[698] -= dqdci;     // dwdot[H3O+]/d[E]
  J[700] -= dqdci;     // dwdot[E]/d[E]
  // d()/dT
  J[704] += 2 * dqdT; // dwdot[H]/dT
  J[708] += dqdT;     // dwdot[OH]/dT
  J[725] -= dqdT;     // dwdot[H3O+]/dT
  J[727] -= dqdT;     // dwdot[E]/dT

  // reaction 128: E + H3O+ <=> H2 + OH
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[23] * sc[25];
  k_f = 560100000000 * exp(-0.5 * logT);
  dlnkfdT = -0.5 * invT;
  // reverse
  phi_r = sc[1] * sc[6];
  Kc =
    (kc_fast ? (inv_ge[25] * ge[1] * ge[6] * inv_ge[23])
             : exp(-g_RT[1] - g_RT[6] + g_RT[23] + g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[23] + h_RT[25]) + (h_RT[1] + h_RT[6]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] += q;  // H2
  wdot[6] += q;  // OH
  wdot[23] -= q; // H3O+
  wdot[25] -= q; // E
  // d()/d[H2]
  dqdci = -k_r * sc[6];
  J[28] += dqdci; // dwdot[H2]/d[H2]
  J[33] += dqdci; // dwdot[OH]/d[H2]
  J[50] -= dqdci; // dwdot[H3O+]/d[H2]
  J[52] -= dqdci; // dwdot[E]/d[H2]
  // d()/d[OH]
  dqdci = -k_r * sc[1];
  J[163] += dqdci; // dwdot[H2]/d[OH]
  J[168] += dqdci; // dwdot[OH]/d[OH]
  J[185] -= dqdci; // dwdot[H3O+]/d[OH]
  J[187] -= dqdci; // dwdot[E]/d[OH]
  // d()/d[H3O+]
  dqdci = +k_f * sc[25];
  J[622] += dqdci; // dwdot[H2]/d[H3O+]
  J[627] += dqdci; // dwdot[OH]/d[H3O+]
  J[644] -= dqdci; // dwdot[H3O+]/d[H3O+]
  J[646] -= dqdci; // dwdot[E]/d[H3O+]
  // d()/d[E]
  dqdci = +k_f * sc[23];
  J[676] += dqdci; // dwdot[H2]/d[E]
  J[681] += dqdci; // dwdot[OH]/d[E]
  J[698] -= dqdci; // dwdot[H3O+]/d[E]
  J[700] -= dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[703] += dqdT; // dwdot[H2]/dT
  J[708] += dqdT; // dwdot[OH]/dT
  J[725] -= dqdT; // dwdot[H3O+]/dT
  J[727] -= dqdT; // dwdot[E]/dT

  // reaction 129: H2 + O2- <=> E + H2O2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[1] * sc[24];
  k_f = 602000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[7] * sc[25];
  Kc =
    (kc_fast ? (ge[25] * inv_ge[1] * inv_ge[24] * ge[7])
             : exp(g_RT[1] - g_RT[7] + g_RT[24] - g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[1] + h_RT[24]) + (h_RT[7] + h_RT[25]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[1] -= q;  // H2
  wdot[7] += q;  // H2O2
  wdot[24] -= q; // O2-
  wdot[25] += q; // E
  // d()/d[H2]
  dqdci = +k_f * sc[24];
  J[28] -= dqdci; // dwdot[H2]/d[H2]
  J[34] += dqdci; // dwdot[H2O2]/d[H2]
  J[51] -= dqdci; // dwdot[O2-]/d[H2]
  J[52] += dqdci; // dwdot[E]/d[H2]
  // d()/d[H2O2]
  dqdci = -k_r * sc[25];
  J[190] -= dqdci; // dwdot[H2]/d[H2O2]
  J[196] += dqdci; // dwdot[H2O2]/d[H2O2]
  J[213] -= dqdci; // dwdot[O2-]/d[H2O2]
  J[214] += dqdci; // dwdot[E]/d[H2O2]
  // d()/d[O2-]
  dqdci = +k_f * sc[1];
  J[649] -= dqdci; // dwdot[H2]/d[O2-]
  J[655] += dqdci; // dwdot[H2O2]/d[O2-]
  J[672] -= dqdci; // dwdot[O2-]/d[O2-]
  J[673] += dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = -k_r * sc[7];
  J[676] -= dqdci; // dwdot[H2]/d[E]
  J[682] += dqdci; // dwdot[H2O2]/d[E]
  J[699] -= dqdci; // dwdot[O2-]/d[E]
  J[700] += dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[703] -= dqdT; // dwdot[H2]/dT
  J[709] += dqdT; // dwdot[H2O2]/dT
  J[726] -= dqdT; // dwdot[O2-]/dT
  J[727] += dqdT; // dwdot[E]/dT

  // reaction 130: H + O2- <=> E + HO2
  // a non-third-body and non-pressure-fall-off reaction
  // forward
  phi_f = sc[2] * sc[24];
  k_f = 723000000;
  dlnkfdT = 0.0;
  // reverse
  phi_r = sc[8] * sc[25];
  Kc =
    (kc_fast ? (ge[25] * ge[8] * inv_ge[24] * inv_ge[2])
             : exp(g_RT[2] - g_RT[8] + g_RT[24] - g_RT[25]));
  k_r = k_f / Kc;
  dlnKcdT = invT * (-(h_RT[2] + h_RT[24]) + (h_RT[8] + h_RT[25]));
  dkrdT = (dlnkfdT - dlnKcdT) * k_r;
  // rate of progress
  q = k_f * phi_f - k_r * phi_r;
  dqdT = (dlnkfdT * k_f * phi_f - dkrdT * phi_r);
  // update wdot
  wdot[2] -= q;  // H
  wdot[8] += q;  // HO2
  wdot[24] -= q; // O2-
  wdot[25] += q; // E
  // d()/d[H]
  dqdci = +k_f * sc[24];
  J[56] -= dqdci; // dwdot[H]/d[H]
  J[62] += dqdci; // dwdot[HO2]/d[H]
  J[78] -= dqdci; // dwdot[O2-]/d[H]
  J[79] += dqdci; // dwdot[E]/d[H]
  // d()/d[HO2]
  dqdci = -k_r * sc[25];
  J[218] -= dqdci; // dwdot[H]/d[HO2]
  J[224] += dqdci; // dwdot[HO2]/d[HO2]
  J[240] -= dqdci; // dwdot[O2-]/d[HO2]
  J[241] += dqdci; // dwdot[E]/d[HO2]
  // d()/d[O2-]
  dqdci = +k_f * sc[2];
  J[650] -= dqdci; // dwdot[H]/d[O2-]
  J[656] += dqdci; // dwdot[HO2]/d[O2-]
  J[672] -= dqdci; // dwdot[O2-]/d[O2-]
  J[673] += dqdci; // dwdot[E]/d[O2-]
  // d()/d[E]
  dqdci = -k_r * sc[8];
  J[677] -= dqdci; // dwdot[H]/d[E]
  J[683] += dqdci; // dwdot[HO2]/d[E]
  J[699] -= dqdci; // dwdot[O2-]/d[E]
  J[700] += dqdci; // dwdot[E]/d[E]
  // d()/dT
  J[704] -= dqdT; // dwdot[H]/dT
  J[710] += dqdT; // dwdot[HO2]/dT
  J[726] -= dqdT; // dwdot[O2-]/dT
  J[727] += dqdT; // dwdot[E]/dT

  amrex::Real c_R[26], dcRdT[26], e_RT[26];
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
  for (int k = 0; k < 26; ++k) {
    cmix += c_R[k] * sc[k];
    dcmixdT += dcRdT[k] * sc[k];
    ehmix += eh_RT[k] * wdot[k];
    dehmixdT += invT * (c_R[k] - eh_RT[k]) * wdot[k] + eh_RT[k] * J[702 + k];
  }

  amrex::Real cmixinv = 1.0 / cmix;
  amrex::Real tmp1 = ehmix * cmixinv;
  amrex::Real tmp3 = cmixinv * T;
  amrex::Real tmp2 = tmp1 * tmp3;
  amrex::Real dehmixdc;
  // dTdot/d[X]
  for (int k = 0; k < 26; ++k) {
    dehmixdc = 0.0;
    for (int m = 0; m < 26; ++m) {
      dehmixdc += eh_RT[m] * J[k * 27 + m];
    }
    J[k * 27 + 26] = tmp2 * c_R[k] - tmp3 * dehmixdc;
  }
  // dTdot/dT
  J[728] = -tmp1 + tmp2 * dcmixdT - tmp3 * dehmixdT;
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
  amrex::Real c[26];

  for (int k = 0; k < 26; k++) {
    c[k] = 1.e6 * sc[k];
  }

  aJacobian(J, c, *Tp, *consP);

  // dwdot[k]/dT
  // dTdot/d[X]
  for (int k = 0; k < 26; k++) {
    J[702 + k] *= 1.e-6;
    J[k * 27 + 26] *= 1.e6;
  }
}

// Transport function declarations

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENIMC(int* LENIMC)
{
  *LENIMC = 107;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetLENRMC(int* LENRMC)
{
  *LENRMC = 13858;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNO(int* NO)
{
  *NO = 4;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKK(int* KK)
{
  *KK = 26;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLITE(int* NLITE)
{
  *NLITE = 3;
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
  WT[0] = 2.80140000E+01;
  WT[1] = 2.01600000E+00;
  WT[2] = 1.00800000E+00;
  WT[3] = 3.19980000E+01;
  WT[4] = 1.59990000E+01;
  WT[5] = 1.80150000E+01;
  WT[6] = 1.70070000E+01;
  WT[7] = 3.40140000E+01;
  WT[8] = 3.30060000E+01;
  WT[9] = 2.80100000E+01;
  WT[10] = 4.40090000E+01;
  WT[11] = 1.60430000E+01;
  WT[12] = 1.50350000E+01;
  WT[13] = 1.40270000E+01;
  WT[14] = 1.40270000E+01;
  WT[15] = 1.20110000E+01;
  WT[16] = 1.30190000E+01;
  WT[17] = 3.20420000E+01;
  WT[18] = 3.10340000E+01;
  WT[19] = 3.10340000E+01;
  WT[20] = 3.00260000E+01;
  WT[21] = 2.90180000E+01;
  WT[22] = 2.90174514E+01;
  WT[23] = 1.90224514E+01;
  WT[24] = 3.19985486E+01;
  WT[25] = 5.48579909E-04;
}

// the lennard-jones potential well depth eps/kb in K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetEPS(amrex::Real* EPS)
{
  EPS[0] = 9.75300000E+01;
  EPS[1] = 3.80000000E+01;
  EPS[2] = 1.45000000E+02;
  EPS[3] = 1.07400000E+02;
  EPS[4] = 8.00000000E+01;
  EPS[5] = 5.72400000E+02;
  EPS[6] = 8.00000000E+01;
  EPS[7] = 1.07400000E+02;
  EPS[8] = 1.07400000E+02;
  EPS[9] = 9.81000000E+01;
  EPS[10] = 2.44000000E+02;
  EPS[11] = 1.41400000E+02;
  EPS[12] = 1.44000000E+02;
  EPS[13] = 1.44000000E+02;
  EPS[14] = 1.44000000E+02;
  EPS[15] = 7.14000000E+01;
  EPS[16] = 8.00000000E+01;
  EPS[17] = 4.81800000E+02;
  EPS[18] = 4.17000000E+02;
  EPS[19] = 4.17000000E+02;
  EPS[20] = 4.98000000E+02;
  EPS[21] = 4.98000000E+02;
  EPS[22] = 8.00000000E+01;
  EPS[23] = 1.06200000E+02;
  EPS[24] = 1.36500000E+02;
  EPS[25] = 8.50000000E+02;
}

// the lennard-jones collision diameter in Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetSIG(amrex::Real* SIG)
{
  SIG[0] = 3.62100000E+00;
  SIG[1] = 2.92000000E+00;
  SIG[2] = 2.05000000E+00;
  SIG[3] = 3.45800000E+00;
  SIG[4] = 2.75000000E+00;
  SIG[5] = 2.60500000E+00;
  SIG[6] = 2.75000000E+00;
  SIG[7] = 3.45800000E+00;
  SIG[8] = 3.45800000E+00;
  SIG[9] = 3.65000000E+00;
  SIG[10] = 3.76300000E+00;
  SIG[11] = 3.74600000E+00;
  SIG[12] = 3.80000000E+00;
  SIG[13] = 3.80000000E+00;
  SIG[14] = 3.80000000E+00;
  SIG[15] = 3.29800000E+00;
  SIG[16] = 2.75000000E+00;
  SIG[17] = 3.62600000E+00;
  SIG[18] = 3.69000000E+00;
  SIG[19] = 3.69000000E+00;
  SIG[20] = 3.59000000E+00;
  SIG[21] = 3.59000000E+00;
  SIG[22] = 2.75000000E+00;
  SIG[23] = 3.15000000E+00;
  SIG[24] = 3.33000000E+00;
  SIG[25] = 4.25000000E+02;
}

// the dipole moment in Debye
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetDIP(amrex::Real* DIP)
{
  DIP[0] = 0.00000000E+00;
  DIP[1] = 0.00000000E+00;
  DIP[2] = 0.00000000E+00;
  DIP[3] = 0.00000000E+00;
  DIP[4] = 0.00000000E+00;
  DIP[5] = 1.84400053E+00;
  DIP[6] = 0.00000000E+00;
  DIP[7] = 0.00000000E+00;
  DIP[8] = 0.00000000E+00;
  DIP[9] = 0.00000000E+00;
  DIP[10] = 0.00000000E+00;
  DIP[11] = 0.00000000E+00;
  DIP[12] = 0.00000000E+00;
  DIP[13] = 0.00000000E+00;
  DIP[14] = 0.00000000E+00;
  DIP[15] = 0.00000000E+00;
  DIP[16] = 0.00000000E+00;
  DIP[17] = 0.00000000E+00;
  DIP[18] = 1.70000049E+00;
  DIP[19] = 1.70000049E+00;
  DIP[20] = 0.00000000E+00;
  DIP[21] = 0.00000000E+00;
  DIP[22] = 0.00000000E+00;
  DIP[23] = 0.00000000E+00;
  DIP[24] = 0.00000000E+00;
  DIP[25] = 0.00000000E+00;
}

// the polarizability in cubic Angstroms
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetPOL(amrex::Real* POL)
{
  POL[0] = 1.76000000E+00;
  POL[1] = 7.90000000E-01;
  POL[2] = 0.00000000E+00;
  POL[3] = 1.60000000E+00;
  POL[4] = 0.00000000E+00;
  POL[5] = 0.00000000E+00;
  POL[6] = 0.00000000E+00;
  POL[7] = 0.00000000E+00;
  POL[8] = 0.00000000E+00;
  POL[9] = 1.95000000E+00;
  POL[10] = 2.65000000E+00;
  POL[11] = 2.60000000E+00;
  POL[12] = 0.00000000E+00;
  POL[13] = 0.00000000E+00;
  POL[14] = 0.00000000E+00;
  POL[15] = 0.00000000E+00;
  POL[16] = 0.00000000E+00;
  POL[17] = 0.00000000E+00;
  POL[18] = 0.00000000E+00;
  POL[19] = 0.00000000E+00;
  POL[20] = 0.00000000E+00;
  POL[21] = 0.00000000E+00;
  POL[22] = 1.34100000E+00;
  POL[23] = 9.64000000E-01;
  POL[24] = 1.58100000E+00;
  POL[25] = 0.00000000E+00;
}

// the rotational relaxation collision number at 298 K
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetZROT(amrex::Real* ZROT)
{
  ZROT[0] = 4.00000000E+00;
  ZROT[1] = 2.80000000E+02;
  ZROT[2] = 0.00000000E+00;
  ZROT[3] = 3.80000000E+00;
  ZROT[4] = 0.00000000E+00;
  ZROT[5] = 4.00000000E+00;
  ZROT[6] = 0.00000000E+00;
  ZROT[7] = 3.80000000E+00;
  ZROT[8] = 1.00000000E+00;
  ZROT[9] = 1.80000000E+00;
  ZROT[10] = 2.10000000E+00;
  ZROT[11] = 1.30000000E+01;
  ZROT[12] = 0.00000000E+00;
  ZROT[13] = 0.00000000E+00;
  ZROT[14] = 0.00000000E+00;
  ZROT[15] = 0.00000000E+00;
  ZROT[16] = 0.00000000E+00;
  ZROT[17] = 1.00000000E+00;
  ZROT[18] = 2.00000000E+00;
  ZROT[19] = 2.00000000E+00;
  ZROT[20] = 2.00000000E+00;
  ZROT[21] = 0.00000000E+00;
  ZROT[22] = 0.00000000E+00;
  ZROT[23] = 1.00000000E+01;
  ZROT[24] = 3.80000000E+00;
  ZROT[25] = 1.00000000E+00;
}

// 0: monoatomic, 1: linear, 2: nonlinear
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetNLIN(int* NLIN)
{
  NLIN[0] = 1;
  NLIN[1] = 1;
  NLIN[2] = 0;
  NLIN[3] = 1;
  NLIN[4] = 0;
  NLIN[5] = 2;
  NLIN[6] = 1;
  NLIN[7] = 2;
  NLIN[8] = 2;
  NLIN[9] = 1;
  NLIN[10] = 1;
  NLIN[11] = 2;
  NLIN[12] = 1;
  NLIN[13] = 1;
  NLIN[14] = 1;
  NLIN[15] = 0;
  NLIN[16] = 1;
  NLIN[17] = 2;
  NLIN[18] = 2;
  NLIN[19] = 2;
  NLIN[20] = 2;
  NLIN[21] = 2;
  NLIN[22] = 1;
  NLIN[23] = 2;
  NLIN[24] = 1;
  NLIN[25] = 0;
}

// Poly fits for the viscosities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFETA(amrex::Real* COFETA)
{
  COFETA[0] = -1.62526676E+01;
  COFETA[1] = 2.24839597E+00;
  COFETA[2] = -2.13428438E-01;
  COFETA[3] = 9.46192413E-03;
  COFETA[4] = -1.37549291E+01;
  COFETA[5] = 9.65530587E-01;
  COFETA[6] = -4.45720114E-02;
  COFETA[7] = 2.05871810E-03;
  COFETA[8] = -1.98744352E+01;
  COFETA[9] = 3.41660514E+00;
  COFETA[10] = -3.63206306E-01;
  COFETA[11] = 1.58671021E-02;
  COFETA[12] = -1.68118998E+01;
  COFETA[13] = 2.52362554E+00;
  COFETA[14] = -2.49309128E-01;
  COFETA[15] = 1.10211025E-02;
  COFETA[16] = -1.48001711E+01;
  COFETA[17] = 1.79491990E+00;
  COFETA[18] = -1.54008440E-01;
  COFETA[19] = 6.86719439E-03;
  COFETA[20] = -1.17770970E+01;
  COFETA[21] = -8.26746102E-01;
  COFETA[22] = 3.39009582E-01;
  COFETA[23] = -2.00674559E-02;
  COFETA[24] = -1.47696217E+01;
  COFETA[25] = 1.79491990E+00;
  COFETA[26] = -1.54008440E-01;
  COFETA[27] = 6.86719439E-03;
  COFETA[28] = -1.67813504E+01;
  COFETA[29] = 2.52362554E+00;
  COFETA[30] = -2.49309128E-01;
  COFETA[31] = 1.10211025E-02;
  COFETA[32] = -1.67963918E+01;
  COFETA[33] = 2.52362554E+00;
  COFETA[34] = -2.49309128E-01;
  COFETA[35] = 1.10211025E-02;
  COFETA[36] = -1.63031343E+01;
  COFETA[37] = 2.26143219E+00;
  COFETA[38] = -2.15114671E-01;
  COFETA[39] = 9.53461976E-03;
  COFETA[40] = -2.36749638E+01;
  COFETA[41] = 4.99775518E+00;
  COFETA[42] = -5.52687718E-01;
  COFETA[43] = 2.34353338E-02;
  COFETA[44] = -1.95453436E+01;
  COFETA[45] = 3.36385478E+00;
  COFETA[46] = -3.56948469E-01;
  COFETA[47] = 1.56210922E-02;
  COFETA[48] = -1.97154229E+01;
  COFETA[49] = 3.40193237E+00;
  COFETA[50] = -3.61470464E-01;
  COFETA[51] = 1.57990760E-02;
  COFETA[52] = -1.97501213E+01;
  COFETA[53] = 3.40193237E+00;
  COFETA[54] = -3.61470464E-01;
  COFETA[55] = 1.57990760E-02;
  COFETA[56] = -1.97501213E+01;
  COFETA[57] = 3.40193237E+00;
  COFETA[58] = -3.61470464E-01;
  COFETA[59] = 1.57990760E-02;
  COFETA[60] = -1.47250813E+01;
  COFETA[61] = 1.57011544E+00;
  COFETA[62] = -1.24217373E-01;
  COFETA[63] = 5.55161238E-03;
  COFETA[64] = -1.49032293E+01;
  COFETA[65] = 1.79491990E+00;
  COFETA[66] = -1.54008440E-01;
  COFETA[67] = 6.86719439E-03;
  COFETA[68] = -2.17058520E+01;
  COFETA[69] = 3.55238511E+00;
  COFETA[70] = -2.93340526E-01;
  COFETA[71] = 9.37852541E-03;
  COFETA[72] = -2.09501615E+01;
  COFETA[73] = 3.29835026E+00;
  COFETA[74] = -2.67151401E-01;
  COFETA[75] = 8.53547645E-03;
  COFETA[76] = -2.09501615E+01;
  COFETA[77] = 3.29835026E+00;
  COFETA[78] = -2.67151401E-01;
  COFETA[79] = 8.53547645E-03;
  COFETA[80] = -2.11136150E+01;
  COFETA[81] = 3.26961843E+00;
  COFETA[82] = -2.51355092E-01;
  COFETA[83] = 7.35605058E-03;
  COFETA[84] = -2.11306886E+01;
  COFETA[85] = 3.26961843E+00;
  COFETA[86] = -2.51355092E-01;
  COFETA[87] = 7.35605058E-03;
  COFETA[88] = -1.45024855E+01;
  COFETA[89] = 1.79491990E+00;
  COFETA[90] = -1.54008440E-01;
  COFETA[91] = 6.86719439E-03;
  COFETA[92] = -1.67955362E+01;
  COFETA[93] = 2.48907364E+00;
  COFETA[94] = -2.44793974E-01;
  COFETA[95] = 1.08243607E-02;
  COFETA[96] = -1.87177043E+01;
  COFETA[97] = 3.27402596E+00;
  COFETA[98] = -3.45827972E-01;
  COFETA[99] = 1.51622680E-02;
  COFETA[100] = -2.18499525E+01;
  COFETA[101] = -3.03985219E+00;
  COFETA[102] = 6.46768575E-01;
  COFETA[103] = -3.43986573E-02;
}

// Poly fits for the conductivities, dim NO*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFLAM(amrex::Real* COFLAM)
{
  COFLAM[0] = 1.15135759E+01;
  COFLAM[1] = -2.89718927E+00;
  COFLAM[2] = 5.52227164E-01;
  COFLAM[3] = -2.73618079E-02;
  COFLAM[4] = 1.05575645E+01;
  COFLAM[5] = -1.08525230E+00;
  COFLAM[6] = 2.11035028E-01;
  COFLAM[7] = -7.45871436E-03;
  COFLAM[8] = -3.24554543E-01;
  COFLAM[9] = 3.41660514E+00;
  COFLAM[10] = -3.63206306E-01;
  COFLAM[11] = 1.58671021E-02;
  COFLAM[12] = -2.39605522E+00;
  COFLAM[13] = 3.11185492E+00;
  COFLAM[14] = -3.06053822E-01;
  COFLAM[15] = 1.33590959E-02;
  COFLAM[16] = 1.98515155E+00;
  COFLAM[17] = 1.79491990E+00;
  COFLAM[18] = -1.54008440E-01;
  COFLAM[19] = 6.86719439E-03;
  COFLAM[20] = 2.10125974E+01;
  COFLAM[21] = -7.92638744E+00;
  COFLAM[22] = 1.37729291E+00;
  COFLAM[23] = -6.86156427E-02;
  COFLAM[24] = 1.50443438E+01;
  COFLAM[25] = -3.64813621E+00;
  COFLAM[26] = 5.94792348E-01;
  COFLAM[27] = -2.63489305E-02;
  COFLAM[28] = 1.90147776E+00;
  COFLAM[29] = 9.16567599E-01;
  COFLAM[30] = 6.93878492E-02;
  COFLAM[31] = -6.39769965E-03;
  COFLAM[32] = 2.91063106E+00;
  COFLAM[33] = 5.78338476E-01;
  COFLAM[34] = 9.02531383E-02;
  COFLAM[35] = -6.19296183E-03;
  COFLAM[36] = 1.00332074E+01;
  COFLAM[37] = -2.32812074E+00;
  COFLAM[38] = 4.79111919E-01;
  COFLAM[39] = -2.42630313E-02;
  COFLAM[40] = -1.19710226E+01;
  COFLAM[41] = 6.16510101E+00;
  COFLAM[42] = -6.12547921E-01;
  COFLAM[43] = 2.26087237E-02;
  COFLAM[44] = 6.90173358E+00;
  COFLAM[45] = -2.03450657E+00;
  COFLAM[46] = 5.89314746E-01;
  COFLAM[47] = -3.40238924E-02;
  COFLAM[48] = 1.24262108E+01;
  COFLAM[49] = -3.92863760E+00;
  COFLAM[50] = 7.99986093E-01;
  COFLAM[51] = -4.23406929E-02;
  COFLAM[52] = 1.26992050E+01;
  COFLAM[53] = -3.59706429E+00;
  COFLAM[54] = 6.88908305E-01;
  COFLAM[55] = -3.47879197E-02;
  COFLAM[56] = 1.34129531E+01;
  COFLAM[57] = -3.93997381E+00;
  COFLAM[58] = 7.35902066E-01;
  COFLAM[59] = -3.65948332E-02;
  COFLAM[60] = 2.34694471E+00;
  COFLAM[61] = 1.57011544E+00;
  COFLAM[62] = -1.24217373E-01;
  COFLAM[63] = 5.55161238E-03;
  COFLAM[64] = 2.19696972E+01;
  COFLAM[65] = -6.76971206E+00;
  COFLAM[66] = 1.06194842E+00;
  COFLAM[67] = -4.90083482E-02;
  COFLAM[68] = -8.91606915E+00;
  COFLAM[69] = 3.61762165E+00;
  COFLAM[70] = -8.92916386E-02;
  COFLAM[71] = -7.46114102E-03;
  COFLAM[72] = -1.33068537E+01;
  COFLAM[73] = 5.77901128E+00;
  COFLAM[74] = -4.27810572E-01;
  COFLAM[75] = 9.36447961E-03;
  COFLAM[76] = -6.42835893E+00;
  COFLAM[77] = 3.26905843E+00;
  COFLAM[78] = -1.28184954E-01;
  COFLAM[79] = -2.45699613E-03;
  COFLAM[80] = 7.06271985E-01;
  COFLAM[81] = -2.71898947E-01;
  COFLAM[82] = 4.21512856E-01;
  COFLAM[83] = -3.00411809E-02;
  COFLAM[84] = -1.49162912E+00;
  COFLAM[85] = 1.41832455E+00;
  COFLAM[86] = 6.93252840E-02;
  COFLAM[87] = -8.61179989E-03;
  COFLAM[88] = 6.28525712E+00;
  COFLAM[89] = -8.51059745E-01;
  COFLAM[90] = 3.29554354E-01;
  COFLAM[91] = -1.93574144E-02;
  COFLAM[92] = 1.18912374E+01;
  COFLAM[93] = -3.44418514E+00;
  COFLAM[94] = 7.12558841E-01;
  COFLAM[95] = -3.73179377E-02;
  COFLAM[96] = -6.15617763E+00;
  COFLAM[97] = 4.62492578E+00;
  COFLAM[98] = -4.96911868E-01;
  COFLAM[99] = 2.09308888E-02;
  COFLAM[100] = 5.21607397E+00;
  COFLAM[101] = -3.03985219E+00;
  COFLAM[102] = 6.46768575E-01;
  COFLAM[103] = -3.43986573E-02;
}

// Poly fits for the diffusion coefficients, dim NO*KK*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFD(amrex::Real* COFD)
{
  COFD[0] = -1.47639411E+01;
  COFD[1] = 3.15955654E+00;
  COFD[2] = -1.97590757E-01;
  COFD[3] = 8.64692156E-03;
  COFD[4] = -1.13253618E+01;
  COFD[5] = 2.31195095E+00;
  COFD[6] = -8.63988037E-02;
  COFD[7] = 3.77573452E-03;
  COFD[8] = -1.02395385E+01;
  COFD[9] = 2.15403244E+00;
  COFD[10] = -6.97480266E-02;
  COFD[11] = 3.23666871E-03;
  COFD[12] = -1.40298991E+01;
  COFD[13] = 3.55837688E+00;
  COFD[14] = -2.47785790E-01;
  COFD[15] = 1.07555332E-02;
  COFD[16] = -1.11808845E+01;
  COFD[17] = 2.66936727E+00;
  COFD[18] = -1.34411514E-01;
  COFD[19] = 5.92957488E-03;
  COFD[20] = -1.43693219E+01;
  COFD[21] = 4.03992999E+00;
  COFD[22] = -3.08044800E-01;
  COFD[23] = 1.32757775E-02;
  COFD[24] = -1.50096252E+01;
  COFD[25] = 3.25515933E+00;
  COFD[26] = -2.09710110E-01;
  COFD[27] = 9.15941830E-03;
  COFD[28] = -1.15797897E+01;
  COFD[29] = 2.43235504E+00;
  COFD[30] = -1.02890179E-01;
  COFD[31] = 4.52903603E-03;
  COFD[32] = -1.43713019E+01;
  COFD[33] = 3.70920439E+00;
  COFD[34] = -2.67274113E-01;
  COFD[35] = 1.15967481E-02;
  COFD[36] = -1.53110597E+01;
  COFD[37] = 3.37317428E+00;
  COFD[38] = -2.24900439E-01;
  COFD[39] = 9.81228151E-03;
  COFD[40] = -1.38756380E+01;
  COFD[41] = 2.98558426E+00;
  COFD[42] = -1.75507216E-01;
  COFD[43] = 7.71173691E-03;
  COFD[44] = -1.06250314E+01;
  COFD[45] = 2.15849701E+00;
  COFD[46] = -6.53886401E-02;
  COFD[47] = 2.81453370E-03;
  COFD[48] = -1.31860264E+01;
  COFD[49] = 3.38003453E+00;
  COFD[50] = -2.25783856E-01;
  COFD[51] = 9.85028660E-03;
  COFD[52] = -1.40864784E+01;
  COFD[53] = 3.07458927E+00;
  COFD[54] = -1.86899591E-01;
  COFD[55] = 8.19829781E-03;
  COFD[56] = -1.29877255E+01;
  COFD[57] = 2.80841511E+00;
  COFD[58] = -1.52629888E-01;
  COFD[59] = 6.72604927E-03;
  COFD[60] = -2.07768629E+01;
  COFD[61] = 5.42470187E+00;
  COFD[62] = -4.69700452E-01;
  COFD[63] = 1.95706916E-02;
  COFD[64] = -1.68277339E+01;
  COFD[65] = 4.49460364E+00;
  COFD[66] = -3.64766207E-01;
  COFD[67] = 1.56457185E-02;
  COFD[68] = -1.93611201E+01;
  COFD[69] = 5.51579726E+00;
  COFD[70] = -4.76061961E-01;
  COFD[71] = 1.96329391E-02;
  COFD[72] = -2.10287060E+01;
  COFD[73] = 5.50980708E+00;
  COFD[74] = -4.78335495E-01;
  COFD[75] = 1.98515433E-02;
  COFD[76] = -1.91096700E+01;
  COFD[77] = 5.02608697E+00;
  COFD[78] = -4.26959993E-01;
  COFD[79] = 1.80709910E-02;
  COFD[80] = -1.31492477E+01;
  COFD[81] = 1.48003884E+00;
  COFD[82] = 1.60500200E-01;
  COFD[83] = -1.19765987E-02;
  COFD[84] = -1.38948654E+01;
  COFD[85] = 2.98558426E+00;
  COFD[86] = -1.75507216E-01;
  COFD[87] = 7.71173691E-03;
  COFD[88] = -1.06283588E+01;
  COFD[89] = 2.15849701E+00;
  COFD[90] = -6.53886401E-02;
  COFD[91] = 2.81453370E-03;
  COFD[92] = -1.31877859E+01;
  COFD[93] = 3.38003453E+00;
  COFD[94] = -2.25783856E-01;
  COFD[95] = 9.85028660E-03;
  COFD[96] = -1.41066358E+01;
  COFD[97] = 3.07458927E+00;
  COFD[98] = -1.86899591E-01;
  COFD[99] = 8.19829781E-03;
  COFD[100] = -1.30027669E+01;
  COFD[101] = 2.80841511E+00;
  COFD[102] = -1.52629888E-01;
  COFD[103] = 6.72604927E-03;
  COFD[104] = -1.91256173E+01;
  COFD[105] = 5.02608697E+00;
  COFD[106] = -4.26959993E-01;
  COFD[107] = 1.80709910E-02;
  COFD[108] = -1.30182749E+01;
  COFD[109] = 2.80841511E+00;
  COFD[110] = -1.52629888E-01;
  COFD[111] = 6.72604927E-03;
  COFD[112] = -1.50236540E+01;
  COFD[113] = 3.25515933E+00;
  COFD[114] = -2.09710110E-01;
  COFD[115] = 9.15941830E-03;
  COFD[116] = -1.15815492E+01;
  COFD[117] = 2.43235504E+00;
  COFD[118] = -1.02890179E-01;
  COFD[119] = 4.52903603E-03;
  COFD[120] = -1.43722077E+01;
  COFD[121] = 3.70920439E+00;
  COFD[122] = -2.67274113E-01;
  COFD[123] = 1.15967481E-02;
  COFD[124] = -1.53261011E+01;
  COFD[125] = 3.37317428E+00;
  COFD[126] = -2.24900439E-01;
  COFD[127] = 9.81228151E-03;
  COFD[128] = -1.40964555E+01;
  COFD[129] = 3.07458927E+00;
  COFD[130] = -1.86899591E-01;
  COFD[131] = 8.19829781E-03;
  COFD[132] = -2.04229988E+01;
  COFD[133] = 5.31457079E+00;
  COFD[134] = -4.58216496E-01;
  COFD[135] = 1.91825910E-02;
  COFD[136] = -1.41170278E+01;
  COFD[137] = 3.07458927E+00;
  COFD[138] = -1.86899591E-01;
  COFD[139] = 8.19829781E-03;
  COFD[140] = -1.53416091E+01;
  COFD[141] = 3.37317428E+00;
  COFD[142] = -2.24900439E-01;
  COFD[143] = 9.81228151E-03;
  COFD[144] = -1.50168047E+01;
  COFD[145] = 3.25515933E+00;
  COFD[146] = -2.09710110E-01;
  COFD[147] = 9.15941830E-03;
  COFD[148] = -1.15806955E+01;
  COFD[149] = 2.43235504E+00;
  COFD[150] = -1.02890179E-01;
  COFD[151] = 4.52903603E-03;
  COFD[152] = -1.43717684E+01;
  COFD[153] = 3.70920439E+00;
  COFD[154] = -2.67274113E-01;
  COFD[155] = 1.15967481E-02;
  COFD[156] = -1.53187536E+01;
  COFD[157] = 3.37317428E+00;
  COFD[158] = -2.24900439E-01;
  COFD[159] = 9.81228151E-03;
  COFD[160] = -1.40915944E+01;
  COFD[161] = 3.07458927E+00;
  COFD[162] = -1.86899591E-01;
  COFD[163] = 8.19829781E-03;
  COFD[164] = -2.04177394E+01;
  COFD[165] = 5.31457079E+00;
  COFD[166] = -4.58216496E-01;
  COFD[167] = 1.91825910E-02;
  COFD[168] = -1.41119635E+01;
  COFD[169] = 3.07458927E+00;
  COFD[170] = -1.86899591E-01;
  COFD[171] = 8.19829781E-03;
  COFD[172] = -1.53340319E+01;
  COFD[173] = 3.37317428E+00;
  COFD[174] = -2.24900439E-01;
  COFD[175] = 9.81228151E-03;
  COFD[176] = -1.53265677E+01;
  COFD[177] = 3.37317428E+00;
  COFD[178] = -2.24900439E-01;
  COFD[179] = 9.81228151E-03;
  COFD[180] = -1.47850505E+01;
  COFD[181] = 3.16433919E+00;
  COFD[182] = -1.98191564E-01;
  COFD[183] = 8.67209742E-03;
  COFD[184] = -1.13541221E+01;
  COFD[185] = 2.31999438E+00;
  COFD[186] = -8.75064804E-02;
  COFD[187] = 3.82656365E-03;
  COFD[188] = -1.40524219E+01;
  COFD[189] = 3.56261348E+00;
  COFD[190] = -2.48287981E-01;
  COFD[191] = 1.07752947E-02;
  COFD[192] = -1.50371688E+01;
  COFD[193] = 3.26249588E+00;
  COFD[194] = -2.10658287E-01;
  COFD[195] = 9.20032462E-03;
  COFD[196] = -1.39007309E+01;
  COFD[197] = 2.99164244E+00;
  COFD[198] = -1.76293106E-01;
  COFD[199] = 7.74575100E-03;
  COFD[200] = -2.08561923E+01;
  COFD[201] = 5.44718685E+00;
  COFD[202] = -4.72082987E-01;
  COFD[203] = 1.96531332E-02;
  COFD[204] = -1.39199573E+01;
  COFD[205] = 2.99164244E+00;
  COFD[206] = -1.76293106E-01;
  COFD[207] = 7.74575100E-03;
  COFD[208] = -1.50511964E+01;
  COFD[209] = 3.26249588E+00;
  COFD[210] = -2.10658287E-01;
  COFD[211] = 9.20032462E-03;
  COFD[212] = -1.50443476E+01;
  COFD[213] = 3.26249588E+00;
  COFD[214] = -2.10658287E-01;
  COFD[215] = 9.20032462E-03;
  COFD[216] = -1.48061406E+01;
  COFD[217] = 3.16912473E+00;
  COFD[218] = -1.98792456E-01;
  COFD[219] = 8.69726395E-03;
  COFD[220] = -1.77350630E+01;
  COFD[221] = 4.19328271E+00;
  COFD[222] = -3.26911461E-01;
  COFD[223] = 1.40520357E-02;
  COFD[224] = -1.35545391E+01;
  COFD[225] = 3.13878730E+00;
  COFD[226] = -1.94980335E-01;
  COFD[227] = 8.53744486E-03;
  COFD[228] = -1.72994129E+01;
  COFD[229] = 4.71931868E+00;
  COFD[230] = -3.91258152E-01;
  COFD[231] = 1.66866639E-02;
  COFD[232] = -1.81197250E+01;
  COFD[233] = 4.33684042E+00;
  COFD[234] = -3.44981265E-01;
  COFD[235] = 1.48142449E-02;
  COFD[236] = -1.67115471E+01;
  COFD[237] = 3.98859394E+00;
  COFD[238] = -3.02316219E-01;
  COFD[239] = 1.30661099E-02;
  COFD[240] = -2.11719362E+01;
  COFD[241] = 5.20775012E+00;
  COFD[242] = -4.07348260E-01;
  COFD[243] = 1.55473249E-02;
  COFD[244] = -1.67337674E+01;
  COFD[245] = 3.98859394E+00;
  COFD[246] = -3.02316219E-01;
  COFD[247] = 1.30661099E-02;
  COFD[248] = -1.81371853E+01;
  COFD[249] = 4.33684042E+00;
  COFD[250] = -3.44981265E-01;
  COFD[251] = 1.48142449E-02;
  COFD[252] = -1.81286456E+01;
  COFD[253] = 4.33684042E+00;
  COFD[254] = -3.44981265E-01;
  COFD[255] = 1.48142449E-02;
  COFD[256] = -1.77672912E+01;
  COFD[257] = 4.20234040E+00;
  COFD[258] = -3.28057658E-01;
  COFD[259] = 1.41006192E-02;
  COFD[260] = -2.10907633E+01;
  COFD[261] = 5.29211327E+00;
  COFD[262] = -4.56068366E-01;
  COFD[263] = 1.91195062E-02;
  COFD[264] = -1.56756124E+01;
  COFD[265] = 3.54035770E+00;
  COFD[266] = -2.45653442E-01;
  COFD[267] = 1.06717969E-02;
  COFD[268] = -1.22114409E+01;
  COFD[269] = 2.65215786E+00;
  COFD[270] = -1.32206766E-01;
  COFD[271] = 5.83532580E-03;
  COFD[272] = -1.53241298E+01;
  COFD[273] = 4.02249494E+00;
  COFD[274] = -3.06112211E-01;
  COFD[275] = 1.32056535E-02;
  COFD[276] = -1.59878735E+01;
  COFD[277] = 3.66478157E+00;
  COFD[278] = -2.61506432E-01;
  COFD[279] = 1.13466231E-02;
  COFD[280] = -1.47765660E+01;
  COFD[281] = 3.34934830E+00;
  COFD[282] = -2.21831026E-01;
  COFD[283] = 9.68019791E-03;
  COFD[284] = -2.12561676E+01;
  COFD[285] = 5.52385259E+00;
  COFD[286] = -4.69683808E-01;
  COFD[287] = 1.90677248E-02;
  COFD[288] = -1.47916284E+01;
  COFD[289] = 3.34934830E+00;
  COFD[290] = -2.21831026E-01;
  COFD[291] = 9.68019791E-03;
  COFD[292] = -1.59978691E+01;
  COFD[293] = 3.66478157E+00;
  COFD[294] = -2.61506432E-01;
  COFD[295] = 1.13466231E-02;
  COFD[296] = -1.59929990E+01;
  COFD[297] = 3.66478157E+00;
  COFD[298] = -2.61506432E-01;
  COFD[299] = 1.13466231E-02;
  COFD[300] = -1.56953785E+01;
  COFD[301] = 3.54443207E+00;
  COFD[302] = -2.46133003E-01;
  COFD[303] = 1.06905018E-02;
  COFD[304] = -1.89594762E+01;
  COFD[305] = 4.68735220E+00;
  COFD[306] = -3.87449514E-01;
  COFD[307] = 1.65352261E-02;
  COFD[308] = -1.68532873E+01;
  COFD[309] = 4.00572564E+00;
  COFD[310] = -3.04255586E-01;
  COFD[311] = 1.31384083E-02;
  COFD[312] = -1.57070757E+01;
  COFD[313] = 3.55327776E+00;
  COFD[314] = -2.47179250E-01;
  COFD[315] = 1.07315653E-02;
  COFD[316] = -1.22583533E+01;
  COFD[317] = 2.66488613E+00;
  COFD[318] = -1.33839930E-01;
  COFD[319] = 5.90524087E-03;
  COFD[320] = -1.53791907E+01;
  COFD[321] = 4.03496578E+00;
  COFD[322] = -3.07489899E-01;
  COFD[323] = 1.32554084E-02;
  COFD[324] = -1.60662884E+01;
  COFD[325] = 3.69744368E+00;
  COFD[326] = -2.65754489E-01;
  COFD[327] = 1.15311681E-02;
  COFD[328] = -1.48358026E+01;
  COFD[329] = 3.37160686E+00;
  COFD[330] = -2.24698307E-01;
  COFD[331] = 9.80357301E-03;
  COFD[332] = -2.10691956E+01;
  COFD[333] = 5.51604369E+00;
  COFD[334] = -4.76337283E-01;
  COFD[335] = 1.96544379E-02;
  COFD[336] = -1.48503699E+01;
  COFD[337] = 3.37160686E+00;
  COFD[338] = -2.24698307E-01;
  COFD[339] = 9.80357301E-03;
  COFD[340] = -1.60758526E+01;
  COFD[341] = 3.69744368E+00;
  COFD[342] = -2.65754489E-01;
  COFD[343] = 1.15311681E-02;
  COFD[344] = -1.60711937E+01;
  COFD[345] = 3.69744368E+00;
  COFD[346] = -2.65754489E-01;
  COFD[347] = 1.15311681E-02;
  COFD[348] = -1.57272832E+01;
  COFD[349] = 3.55756440E+00;
  COFD[350] = -2.47689003E-01;
  COFD[351] = 1.07517019E-02;
  COFD[352] = -1.90110170E+01;
  COFD[353] = 4.70842146E+00;
  COFD[354] = -3.89924637E-01;
  COFD[355] = 1.66320934E-02;
  COFD[356] = -1.68880017E+01;
  COFD[357] = 4.01759713E+00;
  COFD[358] = -3.05560618E-01;
  COFD[359] = 1.31852111E-02;
  COFD[360] = -1.69251551E+01;
  COFD[361] = 4.03028423E+00;
  COFD[362] = -3.06976249E-01;
  COFD[363] = 1.32370318E-02;
  COFD[364] = -1.56842242E+01;
  COFD[365] = 3.55327776E+00;
  COFD[366] = -2.47179250E-01;
  COFD[367] = 1.07315653E-02;
  COFD[368] = -1.22541230E+01;
  COFD[369] = 2.66488613E+00;
  COFD[370] = -1.33839930E-01;
  COFD[371] = 5.90524087E-03;
  COFD[372] = -1.53769382E+01;
  COFD[373] = 4.03496578E+00;
  COFD[374] = -3.07489899E-01;
  COFD[375] = 1.32554084E-02;
  COFD[376] = -1.60424224E+01;
  COFD[377] = 3.69744368E+00;
  COFD[378] = -2.65754489E-01;
  COFD[379] = 1.15311681E-02;
  COFD[380] = -1.48176140E+01;
  COFD[381] = 3.37160686E+00;
  COFD[382] = -2.24698307E-01;
  COFD[383] = 9.80357301E-03;
  COFD[384] = -2.10499842E+01;
  COFD[385] = 5.51604369E+00;
  COFD[386] = -4.76337283E-01;
  COFD[387] = 1.96544379E-02;
  COFD[388] = -1.48316536E+01;
  COFD[389] = 3.37160686E+00;
  COFD[390] = -2.24698307E-01;
  COFD[391] = 9.80357301E-03;
  COFD[392] = -1.60515367E+01;
  COFD[393] = 3.69744368E+00;
  COFD[394] = -2.65754489E-01;
  COFD[395] = 1.15311681E-02;
  COFD[396] = -1.60470980E+01;
  COFD[397] = 3.69744368E+00;
  COFD[398] = -2.65754489E-01;
  COFD[399] = 1.15311681E-02;
  COFD[400] = -1.57044328E+01;
  COFD[401] = 3.55756440E+00;
  COFD[402] = -2.47689003E-01;
  COFD[403] = 1.07517019E-02;
  COFD[404] = -1.89849283E+01;
  COFD[405] = 4.70842146E+00;
  COFD[406] = -3.89924637E-01;
  COFD[407] = 1.66320934E-02;
  COFD[408] = -1.68697894E+01;
  COFD[409] = 4.01759713E+00;
  COFD[410] = -3.05560618E-01;
  COFD[411] = 1.31852111E-02;
  COFD[412] = -1.69075050E+01;
  COFD[413] = 4.03028423E+00;
  COFD[414] = -3.06976249E-01;
  COFD[415] = 1.32370318E-02;
  COFD[416] = -1.68904567E+01;
  COFD[417] = 4.03028423E+00;
  COFD[418] = -3.06976249E-01;
  COFD[419] = 1.32370318E-02;
  COFD[420] = -1.56842242E+01;
  COFD[421] = 3.55327776E+00;
  COFD[422] = -2.47179250E-01;
  COFD[423] = 1.07315653E-02;
  COFD[424] = -1.22541230E+01;
  COFD[425] = 2.66488613E+00;
  COFD[426] = -1.33839930E-01;
  COFD[427] = 5.90524087E-03;
  COFD[428] = -1.53769382E+01;
  COFD[429] = 4.03496578E+00;
  COFD[430] = -3.07489899E-01;
  COFD[431] = 1.32554084E-02;
  COFD[432] = -1.60424224E+01;
  COFD[433] = 3.69744368E+00;
  COFD[434] = -2.65754489E-01;
  COFD[435] = 1.15311681E-02;
  COFD[436] = -1.48176140E+01;
  COFD[437] = 3.37160686E+00;
  COFD[438] = -2.24698307E-01;
  COFD[439] = 9.80357301E-03;
  COFD[440] = -2.10499842E+01;
  COFD[441] = 5.51604369E+00;
  COFD[442] = -4.76337283E-01;
  COFD[443] = 1.96544379E-02;
  COFD[444] = -1.48316536E+01;
  COFD[445] = 3.37160686E+00;
  COFD[446] = -2.24698307E-01;
  COFD[447] = 9.80357301E-03;
  COFD[448] = -1.60515367E+01;
  COFD[449] = 3.69744368E+00;
  COFD[450] = -2.65754489E-01;
  COFD[451] = 1.15311681E-02;
  COFD[452] = -1.60470980E+01;
  COFD[453] = 3.69744368E+00;
  COFD[454] = -2.65754489E-01;
  COFD[455] = 1.15311681E-02;
  COFD[456] = -1.57044328E+01;
  COFD[457] = 3.55756440E+00;
  COFD[458] = -2.47689003E-01;
  COFD[459] = 1.07517019E-02;
  COFD[460] = -1.89849283E+01;
  COFD[461] = 4.70842146E+00;
  COFD[462] = -3.89924637E-01;
  COFD[463] = 1.66320934E-02;
  COFD[464] = -1.68697894E+01;
  COFD[465] = 4.01759713E+00;
  COFD[466] = -3.05560618E-01;
  COFD[467] = 1.31852111E-02;
  COFD[468] = -1.69075050E+01;
  COFD[469] = 4.03028423E+00;
  COFD[470] = -3.06976249E-01;
  COFD[471] = 1.32370318E-02;
  COFD[472] = -1.68904567E+01;
  COFD[473] = 4.03028423E+00;
  COFD[474] = -3.06976249E-01;
  COFD[475] = 1.32370318E-02;
  COFD[476] = -1.68904567E+01;
  COFD[477] = 4.03028423E+00;
  COFD[478] = -3.06976249E-01;
  COFD[479] = 1.32370318E-02;
  COFD[480] = -1.36591073E+01;
  COFD[481] = 2.87846903E+00;
  COFD[482] = -1.61681103E-01;
  COFD[483] = 7.11635115E-03;
  COFD[484] = -1.06186344E+01;
  COFD[485] = 2.09311029E+00;
  COFD[486] = -5.66715295E-02;
  COFD[487] = 2.42607686E-03;
  COFD[488] = -1.30215958E+01;
  COFD[489] = 3.24041848E+00;
  COFD[490] = -2.07804876E-01;
  COFD[491] = 9.07722400E-03;
  COFD[492] = -1.38694480E+01;
  COFD[493] = 2.96800700E+00;
  COFD[494] = -1.73235360E-01;
  COFD[495] = 7.61377726E-03;
  COFD[496] = -1.28155338E+01;
  COFD[497] = 2.69967657E+00;
  COFD[498] = -1.38362256E-01;
  COFD[499] = 6.10143187E-03;
  COFD[500] = -1.88701223E+01;
  COFD[501] = 4.90944038E+00;
  COFD[502] = -4.13866957E-01;
  COFD[503] = 1.75844447E-02;
  COFD[504] = -1.28284059E+01;
  COFD[505] = 2.69967657E+00;
  COFD[506] = -1.38362256E-01;
  COFD[507] = 6.10143187E-03;
  COFD[508] = -1.38776021E+01;
  COFD[509] = 2.96800700E+00;
  COFD[510] = -1.73235360E-01;
  COFD[511] = 7.61377726E-03;
  COFD[512] = -1.38736329E+01;
  COFD[513] = 2.96800700E+00;
  COFD[514] = -1.73235360E-01;
  COFD[515] = 7.61377726E-03;
  COFD[516] = -1.36800096E+01;
  COFD[517] = 2.88311045E+00;
  COFD[518] = -1.62274683E-01;
  COFD[519] = 7.14167394E-03;
  COFD[520] = -1.64069432E+01;
  COFD[521] = 3.85900456E+00;
  COFD[522] = -2.86199757E-01;
  COFD[523] = 1.23960157E-02;
  COFD[524] = -1.44868574E+01;
  COFD[525] = 3.20806285E+00;
  COFD[526] = -2.03619458E-01;
  COFD[527] = 8.89655203E-03;
  COFD[528] = -1.45499799E+01;
  COFD[529] = 3.23154119E+00;
  COFD[530] = -2.06656216E-01;
  COFD[531] = 9.02762284E-03;
  COFD[532] = -1.45342725E+01;
  COFD[533] = 3.23154119E+00;
  COFD[534] = -2.06656216E-01;
  COFD[535] = 9.02762284E-03;
  COFD[536] = -1.45342725E+01;
  COFD[537] = 3.23154119E+00;
  COFD[538] = -2.06656216E-01;
  COFD[539] = 9.02762284E-03;
  COFD[540] = -1.27188437E+01;
  COFD[541] = 2.62486277E+00;
  COFD[542] = -1.28774228E-01;
  COFD[543] = 5.69134090E-03;
  COFD[544] = -1.38076340E+01;
  COFD[545] = 2.98558426E+00;
  COFD[546] = -1.75507216E-01;
  COFD[547] = 7.71173691E-03;
  COFD[548] = -1.06123852E+01;
  COFD[549] = 2.15849701E+00;
  COFD[550] = -6.53886401E-02;
  COFD[551] = 2.81453370E-03;
  COFD[552] = -1.31792887E+01;
  COFD[553] = 3.38003453E+00;
  COFD[554] = -2.25783856E-01;
  COFD[555] = 9.85028660E-03;
  COFD[556] = -1.40154693E+01;
  COFD[557] = 3.07458927E+00;
  COFD[558] = -1.86899591E-01;
  COFD[559] = 8.19829781E-03;
  COFD[560] = -1.29335458E+01;
  COFD[561] = 2.80841511E+00;
  COFD[562] = -1.52629888E-01;
  COFD[563] = 6.72604927E-03;
  COFD[564] = -1.90524563E+01;
  COFD[565] = 5.02608697E+00;
  COFD[566] = -4.26959993E-01;
  COFD[567] = 1.80709910E-02;
  COFD[568] = -1.29470215E+01;
  COFD[569] = 2.80841511E+00;
  COFD[570] = -1.52629888E-01;
  COFD[571] = 6.72604927E-03;
  COFD[572] = -1.40241141E+01;
  COFD[573] = 3.07458927E+00;
  COFD[574] = -1.86899591E-01;
  COFD[575] = 8.19829781E-03;
  COFD[576] = -1.40199050E+01;
  COFD[577] = 3.07458927E+00;
  COFD[578] = -1.86899591E-01;
  COFD[579] = 8.19829781E-03;
  COFD[580] = -1.38327302E+01;
  COFD[581] = 2.99164244E+00;
  COFD[582] = -1.76293106E-01;
  COFD[583] = 7.74575100E-03;
  COFD[584] = -1.66339567E+01;
  COFD[585] = 3.98859394E+00;
  COFD[586] = -3.02316219E-01;
  COFD[587] = 1.30661099E-02;
  COFD[588] = -1.47223159E+01;
  COFD[589] = 3.34934830E+00;
  COFD[590] = -2.21831026E-01;
  COFD[591] = 9.68019791E-03;
  COFD[592] = -1.47832205E+01;
  COFD[593] = 3.37160686E+00;
  COFD[594] = -2.24698307E-01;
  COFD[595] = 9.80357301E-03;
  COFD[596] = -1.47668181E+01;
  COFD[597] = 3.37160686E+00;
  COFD[598] = -2.24698307E-01;
  COFD[599] = 9.80357301E-03;
  COFD[600] = -1.47668181E+01;
  COFD[601] = 3.37160686E+00;
  COFD[602] = -2.24698307E-01;
  COFD[603] = 9.80357301E-03;
  COFD[604] = -1.27687189E+01;
  COFD[605] = 2.69967657E+00;
  COFD[606] = -1.38362256E-01;
  COFD[607] = 6.10143187E-03;
  COFD[608] = -1.28846673E+01;
  COFD[609] = 2.80841511E+00;
  COFD[610] = -1.52629888E-01;
  COFD[611] = 6.72604927E-03;
  COFD[612] = -2.01091568E+01;
  COFD[613] = 5.06508846E+00;
  COFD[614] = -4.31649695E-01;
  COFD[615] = 1.82598046E-02;
  COFD[616] = -1.56250770E+01;
  COFD[617] = 3.92232623E+00;
  COFD[618] = -2.94183188E-01;
  COFD[619] = 1.27325506E-02;
  COFD[620] = -1.94386101E+01;
  COFD[621] = 5.43670321E+00;
  COFD[622] = -4.70985776E-01;
  COFD[623] = 1.96158515E-02;
  COFD[624] = -2.04422993E+01;
  COFD[625] = 5.18126516E+00;
  COFD[626] = -4.45006101E-01;
  COFD[627] = 1.87713513E-02;
  COFD[628] = -1.90163591E+01;
  COFD[629] = 4.83982263E+00;
  COFD[630] = -4.05783981E-01;
  COFD[631] = 1.72721335E-02;
  COFD[632] = -1.89495722E+01;
  COFD[633] = 4.10779436E+00;
  COFD[634] = -2.35955221E-01;
  COFD[635] = 7.03410627E-03;
  COFD[636] = -1.90365260E+01;
  COFD[637] = 4.83982263E+00;
  COFD[638] = -4.05783981E-01;
  COFD[639] = 1.72721335E-02;
  COFD[640] = -2.04573512E+01;
  COFD[641] = 5.18126516E+00;
  COFD[642] = -4.45006101E-01;
  COFD[643] = 1.87713513E-02;
  COFD[644] = -2.04499985E+01;
  COFD[645] = 5.18126516E+00;
  COFD[646] = -4.45006101E-01;
  COFD[647] = 1.87713513E-02;
  COFD[648] = -2.01404432E+01;
  COFD[649] = 5.07318708E+00;
  COFD[650] = -4.32608485E-01;
  COFD[651] = 1.82977801E-02;
  COFD[652] = -2.21011224E+01;
  COFD[653] = 5.52816937E+00;
  COFD[654] = -4.65357386E-01;
  COFD[655] = 1.86939300E-02;
  COFD[656] = -2.10732098E+01;
  COFD[657] = 5.40974993E+00;
  COFD[658] = -4.68080393E-01;
  COFD[659] = 1.95128787E-02;
  COFD[660] = -2.11276735E+01;
  COFD[661] = 5.42949713E+00;
  COFD[662] = -4.70215729E-01;
  COFD[663] = 1.95888774E-02;
  COFD[664] = -2.11037973E+01;
  COFD[665] = 5.42949713E+00;
  COFD[666] = -4.70215729E-01;
  COFD[667] = 1.95888774E-02;
  COFD[668] = -2.11037973E+01;
  COFD[669] = 5.42949713E+00;
  COFD[670] = -4.70215729E-01;
  COFD[671] = 1.95888774E-02;
  COFD[672] = -1.86387511E+01;
  COFD[673] = 4.68388273E+00;
  COFD[674] = -3.87039310E-01;
  COFD[675] = 1.65190492E-02;
  COFD[676] = -1.89453198E+01;
  COFD[677] = 4.83982263E+00;
  COFD[678] = -4.05783981E-01;
  COFD[679] = 1.72721335E-02;
  COFD[680] = -2.03539177E+01;
  COFD[681] = 4.56892447E+00;
  COFD[682] = -3.06147917E-01;
  COFD[683] = 1.04644656E-02;
  COFD[684] = -1.98811260E+01;
  COFD[685] = 4.98725744E+00;
  COFD[686] = -4.22747564E-01;
  COFD[687] = 1.79211872E-02;
  COFD[688] = -1.54930014E+01;
  COFD[689] = 3.86994367E+00;
  COFD[690] = -2.87570272E-01;
  COFD[691] = 1.24533966E-02;
  COFD[692] = -1.90281731E+01;
  COFD[693] = 5.30278138E+00;
  COFD[694] = -4.57079018E-01;
  COFD[695] = 1.91486428E-02;
  COFD[696] = -2.02095101E+01;
  COFD[697] = 5.10234461E+00;
  COFD[698] = -4.36039243E-01;
  COFD[699] = 1.84326773E-02;
  COFD[700] = -1.84871270E+01;
  COFD[701] = 4.64907964E+00;
  COFD[702] = -3.82960584E-01;
  COFD[703] = 1.63598480E-02;
  COFD[704] = -1.75035239E+01;
  COFD[705] = 3.41230274E+00;
  COFD[706] = -1.32406923E-01;
  COFD[707] = 2.10435795E-03;
  COFD[708] = -1.85070738E+01;
  COFD[709] = 4.64907964E+00;
  COFD[710] = -3.82960584E-01;
  COFD[711] = 1.63598480E-02;
  COFD[712] = -1.99478621E+01;
  COFD[713] = 5.00532670E+00;
  COFD[714] = -4.24686880E-01;
  COFD[715] = 1.79891229E-02;
  COFD[716] = -1.99406295E+01;
  COFD[717] = 5.00532670E+00;
  COFD[718] = -4.24686880E-01;
  COFD[719] = 1.79891229E-02;
  COFD[720] = -1.99207369E+01;
  COFD[721] = 4.99746599E+00;
  COFD[722] = -4.23838967E-01;
  COFD[723] = 1.79592096E-02;
  COFD[724] = -2.20169041E+01;
  COFD[725] = 5.52121111E+00;
  COFD[726] = -4.67224743E-01;
  COFD[727] = 1.88799020E-02;
  COFD[728] = -2.09034978E+01;
  COFD[729] = 5.35659328E+00;
  COFD[730] = -4.62684017E-01;
  COFD[731] = 1.93375284E-02;
  COFD[732] = -2.07138428E+01;
  COFD[733] = 5.29812901E+00;
  COFD[734] = -4.56644617E-01;
  COFD[735] = 1.91364773E-02;
  COFD[736] = -2.06902060E+01;
  COFD[737] = 5.29812901E+00;
  COFD[738] = -4.56644617E-01;
  COFD[739] = 1.91364773E-02;
  COFD[740] = -2.06902060E+01;
  COFD[741] = 5.29812901E+00;
  COFD[742] = -4.56644617E-01;
  COFD[743] = 1.91364773E-02;
  COFD[744] = -1.82026567E+01;
  COFD[745] = 4.53059303E+00;
  COFD[746] = -3.69227245E-01;
  COFD[747] = 1.58309435E-02;
  COFD[748] = -1.84167969E+01;
  COFD[749] = 4.64907964E+00;
  COFD[750] = -3.82960584E-01;
  COFD[751] = 1.63598480E-02;
  COFD[752] = -2.10007650E+01;
  COFD[753] = 4.89469279E+00;
  COFD[754] = -3.56802629E-01;
  COFD[755] = 1.29773618E-02;
  COFD[756] = -2.02181133E+01;
  COFD[757] = 4.55977855E+00;
  COFD[758] = -3.10217483E-01;
  COFD[759] = 1.08560899E-02;
  COFD[760] = -1.98811260E+01;
  COFD[761] = 4.98725744E+00;
  COFD[762] = -4.22747564E-01;
  COFD[763] = 1.79211872E-02;
  COFD[764] = -1.54930014E+01;
  COFD[765] = 3.86994367E+00;
  COFD[766] = -2.87570272E-01;
  COFD[767] = 1.24533966E-02;
  COFD[768] = -1.90281731E+01;
  COFD[769] = 5.30278138E+00;
  COFD[770] = -4.57079018E-01;
  COFD[771] = 1.91486428E-02;
  COFD[772] = -2.02095101E+01;
  COFD[773] = 5.10234461E+00;
  COFD[774] = -4.36039243E-01;
  COFD[775] = 1.84326773E-02;
  COFD[776] = -1.84871270E+01;
  COFD[777] = 4.64907964E+00;
  COFD[778] = -3.82960584E-01;
  COFD[779] = 1.63598480E-02;
  COFD[780] = -1.75035239E+01;
  COFD[781] = 3.41230274E+00;
  COFD[782] = -1.32406923E-01;
  COFD[783] = 2.10435795E-03;
  COFD[784] = -1.85070738E+01;
  COFD[785] = 4.64907964E+00;
  COFD[786] = -3.82960584E-01;
  COFD[787] = 1.63598480E-02;
  COFD[788] = -1.99478621E+01;
  COFD[789] = 5.00532670E+00;
  COFD[790] = -4.24686880E-01;
  COFD[791] = 1.79891229E-02;
  COFD[792] = -1.99406295E+01;
  COFD[793] = 5.00532670E+00;
  COFD[794] = -4.24686880E-01;
  COFD[795] = 1.79891229E-02;
  COFD[796] = -1.99207369E+01;
  COFD[797] = 4.99746599E+00;
  COFD[798] = -4.23838967E-01;
  COFD[799] = 1.79592096E-02;
  COFD[800] = -2.20169041E+01;
  COFD[801] = 5.52121111E+00;
  COFD[802] = -4.67224743E-01;
  COFD[803] = 1.88799020E-02;
  COFD[804] = -2.09034978E+01;
  COFD[805] = 5.35659328E+00;
  COFD[806] = -4.62684017E-01;
  COFD[807] = 1.93375284E-02;
  COFD[808] = -2.07138428E+01;
  COFD[809] = 5.29812901E+00;
  COFD[810] = -4.56644617E-01;
  COFD[811] = 1.91364773E-02;
  COFD[812] = -2.06902060E+01;
  COFD[813] = 5.29812901E+00;
  COFD[814] = -4.56644617E-01;
  COFD[815] = 1.91364773E-02;
  COFD[816] = -2.06902060E+01;
  COFD[817] = 5.29812901E+00;
  COFD[818] = -4.56644617E-01;
  COFD[819] = 1.91364773E-02;
  COFD[820] = -1.82026567E+01;
  COFD[821] = 4.53059303E+00;
  COFD[822] = -3.69227245E-01;
  COFD[823] = 1.58309435E-02;
  COFD[824] = -1.84167969E+01;
  COFD[825] = 4.64907964E+00;
  COFD[826] = -3.82960584E-01;
  COFD[827] = 1.63598480E-02;
  COFD[828] = -2.10007650E+01;
  COFD[829] = 4.89469279E+00;
  COFD[830] = -3.56802629E-01;
  COFD[831] = 1.29773618E-02;
  COFD[832] = -2.02181133E+01;
  COFD[833] = 4.55977855E+00;
  COFD[834] = -3.10217483E-01;
  COFD[835] = 1.08560899E-02;
  COFD[836] = -2.02181133E+01;
  COFD[837] = 4.55977855E+00;
  COFD[838] = -3.10217483E-01;
  COFD[839] = 1.08560899E-02;
  COFD[840] = -2.02136057E+01;
  COFD[841] = 5.10993120E+00;
  COFD[842] = -4.36931630E-01;
  COFD[843] = 1.84677592E-02;
  COFD[844] = -1.57172268E+01;
  COFD[845] = 3.96062263E+00;
  COFD[846] = -2.98964970E-01;
  COFD[847] = 1.29322565E-02;
  COFD[848] = -1.95318517E+01;
  COFD[849] = 5.47046983E+00;
  COFD[850] = -4.74577605E-01;
  COFD[851] = 1.97408822E-02;
  COFD[852] = -2.05196205E+01;
  COFD[853] = 5.21473296E+00;
  COFD[854] = -4.48646311E-01;
  COFD[855] = 1.89013813E-02;
  COFD[856] = -1.91105534E+01;
  COFD[857] = 4.87977047E+00;
  COFD[858] = -4.10448693E-01;
  COFD[859] = 1.74535827E-02;
  COFD[860] = -1.87236049E+01;
  COFD[861] = 4.00967621E+00;
  COFD[862] = -2.21153539E-01;
  COFD[863] = 6.31528745E-03;
  COFD[864] = -1.91302705E+01;
  COFD[865] = 4.87977047E+00;
  COFD[866] = -4.10448693E-01;
  COFD[867] = 1.74535827E-02;
  COFD[868] = -2.05341767E+01;
  COFD[869] = 5.21473296E+00;
  COFD[870] = -4.48646311E-01;
  COFD[871] = 1.89013813E-02;
  COFD[872] = -2.05270679E+01;
  COFD[873] = 5.21473296E+00;
  COFD[874] = -4.48646311E-01;
  COFD[875] = 1.89013813E-02;
  COFD[876] = -2.02444913E+01;
  COFD[877] = 5.11785645E+00;
  COFD[878] = -4.37867828E-01;
  COFD[879] = 1.85047543E-02;
  COFD[880] = -2.20860863E+01;
  COFD[881] = 5.52171573E+00;
  COFD[882] = -4.63284984E-01;
  COFD[883] = 1.85570924E-02;
  COFD[884] = -2.11618589E+01;
  COFD[885] = 5.44451952E+00;
  COFD[886] = -4.71804032E-01;
  COFD[887] = 1.96436657E-02;
  COFD[888] = -2.12140281E+01;
  COFD[889] = 5.46309033E+00;
  COFD[890] = -4.73777184E-01;
  COFD[891] = 1.97122249E-02;
  COFD[892] = -2.11906415E+01;
  COFD[893] = 5.46309033E+00;
  COFD[894] = -4.73777184E-01;
  COFD[895] = 1.97122249E-02;
  COFD[896] = -2.11906415E+01;
  COFD[897] = 5.46309033E+00;
  COFD[898] = -4.73777184E-01;
  COFD[899] = 1.97122249E-02;
  COFD[900] = -1.87435094E+01;
  COFD[901] = 4.72753865E+00;
  COFD[902] = -3.92270598E-01;
  COFD[903] = 1.67283839E-02;
  COFD[904] = -1.90409646E+01;
  COFD[905] = 4.87977047E+00;
  COFD[906] = -4.10448693E-01;
  COFD[907] = 1.74535827E-02;
  COFD[908] = -2.01405949E+01;
  COFD[909] = 4.47715839E+00;
  COFD[910] = -2.92174559E-01;
  COFD[911] = 9.78054024E-03;
  COFD[912] = -2.08446140E+01;
  COFD[913] = 4.82681292E+00;
  COFD[914] = -3.46134170E-01;
  COFD[915] = 1.24444585E-02;
  COFD[916] = -2.08446140E+01;
  COFD[917] = 4.82681292E+00;
  COFD[918] = -3.46134170E-01;
  COFD[919] = 1.24444585E-02;
  COFD[920] = -1.99154422E+01;
  COFD[921] = 4.38041133E+00;
  COFD[922] = -2.77538214E-01;
  COFD[923] = 9.06748822E-03;
  COFD[924] = -2.02052920E+01;
  COFD[925] = 5.10993120E+00;
  COFD[926] = -4.36931630E-01;
  COFD[927] = 1.84677592E-02;
  COFD[928] = -1.57161352E+01;
  COFD[929] = 3.96062263E+00;
  COFD[930] = -2.98964970E-01;
  COFD[931] = 1.29322565E-02;
  COFD[932] = -1.95312879E+01;
  COFD[933] = 5.47046983E+00;
  COFD[934] = -4.74577605E-01;
  COFD[935] = 1.97408822E-02;
  COFD[936] = -2.05107395E+01;
  COFD[937] = 5.21473296E+00;
  COFD[938] = -4.48646311E-01;
  COFD[939] = 1.89013813E-02;
  COFD[940] = -1.91045520E+01;
  COFD[941] = 4.87977047E+00;
  COFD[942] = -4.10448693E-01;
  COFD[943] = 1.74535827E-02;
  COFD[944] = -1.87171338E+01;
  COFD[945] = 4.00967621E+00;
  COFD[946] = -2.21153539E-01;
  COFD[947] = 6.31528745E-03;
  COFD[948] = -1.91240292E+01;
  COFD[949] = 4.87977047E+00;
  COFD[950] = -4.10448693E-01;
  COFD[951] = 1.74535827E-02;
  COFD[952] = -2.05250357E+01;
  COFD[953] = 5.21473296E+00;
  COFD[954] = -4.48646311E-01;
  COFD[955] = 1.89013813E-02;
  COFD[956] = -2.05180548E+01;
  COFD[957] = 5.21473296E+00;
  COFD[958] = -4.48646311E-01;
  COFD[959] = 1.89013813E-02;
  COFD[960] = -2.02361782E+01;
  COFD[961] = 5.11785645E+00;
  COFD[962] = -4.37867828E-01;
  COFD[963] = 1.85047543E-02;
  COFD[964] = -2.20758670E+01;
  COFD[965] = 5.52171573E+00;
  COFD[966] = -4.63284984E-01;
  COFD[967] = 1.85570924E-02;
  COFD[968] = -2.11558468E+01;
  COFD[969] = 5.44451952E+00;
  COFD[970] = -4.71804032E-01;
  COFD[971] = 1.96436657E-02;
  COFD[972] = -2.12082662E+01;
  COFD[973] = 5.46309033E+00;
  COFD[974] = -4.73777184E-01;
  COFD[975] = 1.97122249E-02;
  COFD[976] = -2.11851415E+01;
  COFD[977] = 5.46309033E+00;
  COFD[978] = -4.73777184E-01;
  COFD[979] = 1.97122249E-02;
  COFD[980] = -2.11851415E+01;
  COFD[981] = 5.46309033E+00;
  COFD[982] = -4.73777184E-01;
  COFD[983] = 1.97122249E-02;
  COFD[984] = -1.87385713E+01;
  COFD[985] = 4.72753865E+00;
  COFD[986] = -3.92270598E-01;
  COFD[987] = 1.67283839E-02;
  COFD[988] = -1.90357388E+01;
  COFD[989] = 4.87977047E+00;
  COFD[990] = -4.10448693E-01;
  COFD[991] = 1.74535827E-02;
  COFD[992] = -2.01317080E+01;
  COFD[993] = 4.47715839E+00;
  COFD[994] = -2.92174559E-01;
  COFD[995] = 9.78054024E-03;
  COFD[996] = -2.08358634E+01;
  COFD[997] = 4.82681292E+00;
  COFD[998] = -3.46134170E-01;
  COFD[999] = 1.24444585E-02;
  COFD[1000] = -2.08358634E+01;
  COFD[1001] = 4.82681292E+00;
  COFD[1002] = -3.46134170E-01;
  COFD[1003] = 1.24444585E-02;
  COFD[1004] = -1.99068325E+01;
  COFD[1005] = 4.38041133E+00;
  COFD[1006] = -2.77538214E-01;
  COFD[1007] = 9.06748822E-03;
  COFD[1008] = -1.98983685E+01;
  COFD[1009] = 4.38041133E+00;
  COFD[1010] = -2.77538214E-01;
  COFD[1011] = 9.06748822E-03;
  COFD[1012] = -1.40437647E+01;
  COFD[1013] = 2.98558426E+00;
  COFD[1014] = -1.75507216E-01;
  COFD[1015] = 7.71173691E-03;
  COFD[1016] = -1.06507866E+01;
  COFD[1017] = 2.15849701E+00;
  COFD[1018] = -6.53886401E-02;
  COFD[1019] = 2.81453370E-03;
  COFD[1020] = -1.31995018E+01;
  COFD[1021] = 3.38003453E+00;
  COFD[1022] = -2.25783856E-01;
  COFD[1023] = 9.85028660E-03;
  COFD[1024] = -1.42641696E+01;
  COFD[1025] = 3.07458927E+00;
  COFD[1026] = -1.86899591E-01;
  COFD[1027] = 8.19829781E-03;
  COFD[1028] = -1.31147338E+01;
  COFD[1029] = 2.80841511E+00;
  COFD[1030] = -1.52629888E-01;
  COFD[1031] = 6.72604927E-03;
  COFD[1032] = -2.04268198E+01;
  COFD[1033] = 5.41191809E+00;
  COFD[1034] = -4.68316503E-01;
  COFD[1035] = 1.95213604E-02;
  COFD[1036] = -1.31342108E+01;
  COFD[1037] = 2.80841511E+00;
  COFD[1038] = -1.52629888E-01;
  COFD[1039] = 6.72604927E-03;
  COFD[1040] = -1.42784657E+01;
  COFD[1041] = 3.07458927E+00;
  COFD[1042] = -1.86899591E-01;
  COFD[1043] = 8.19829781E-03;
  COFD[1044] = -1.42714849E+01;
  COFD[1045] = 3.07458927E+00;
  COFD[1046] = -1.86899591E-01;
  COFD[1047] = 8.19829781E-03;
  COFD[1048] = -1.40688472E+01;
  COFD[1049] = 2.99164244E+00;
  COFD[1050] = -1.76293106E-01;
  COFD[1051] = 7.74575100E-03;
  COFD[1052] = -1.69110608E+01;
  COFD[1053] = 3.98859394E+00;
  COFD[1054] = -3.02316219E-01;
  COFD[1055] = 1.30661099E-02;
  COFD[1056] = -1.49037729E+01;
  COFD[1057] = 3.34934830E+00;
  COFD[1058] = -2.21831026E-01;
  COFD[1059] = 9.68019791E-03;
  COFD[1060] = -1.49583394E+01;
  COFD[1061] = 3.37160686E+00;
  COFD[1062] = -2.24698307E-01;
  COFD[1063] = 9.80357301E-03;
  COFD[1064] = -1.49352148E+01;
  COFD[1065] = 3.37160686E+00;
  COFD[1066] = -2.24698307E-01;
  COFD[1067] = 9.80357301E-03;
  COFD[1068] = -1.49352148E+01;
  COFD[1069] = 3.37160686E+00;
  COFD[1070] = -2.24698307E-01;
  COFD[1071] = 9.80357301E-03;
  COFD[1072] = -1.29223673E+01;
  COFD[1073] = 2.69967657E+00;
  COFD[1074] = -1.38362256E-01;
  COFD[1075] = 6.10143187E-03;
  COFD[1076] = -1.30459210E+01;
  COFD[1077] = 2.80841511E+00;
  COFD[1078] = -1.52629888E-01;
  COFD[1079] = 6.72604927E-03;
  COFD[1080] = -1.91941481E+01;
  COFD[1081] = 4.83982263E+00;
  COFD[1082] = -4.05783981E-01;
  COFD[1083] = 1.72721335E-02;
  COFD[1084] = -1.91944180E+01;
  COFD[1085] = 4.83981923E+00;
  COFD[1086] = -4.05783571E-01;
  COFD[1087] = 1.72721170E-02;
  COFD[1088] = -1.91944180E+01;
  COFD[1089] = 4.83981923E+00;
  COFD[1090] = -4.05783571E-01;
  COFD[1091] = 1.72721170E-02;
  COFD[1092] = -1.92836946E+01;
  COFD[1093] = 4.87977047E+00;
  COFD[1094] = -4.10448693E-01;
  COFD[1095] = 1.74535827E-02;
  COFD[1096] = -1.92752307E+01;
  COFD[1097] = 4.87977047E+00;
  COFD[1098] = -4.10448693E-01;
  COFD[1099] = 1.74535827E-02;
  COFD[1100] = -1.32854111E+01;
  COFD[1101] = 2.80841511E+00;
  COFD[1102] = -1.52629888E-01;
  COFD[1103] = 6.72604927E-03;
  COFD[1104] = -1.47449891E+01;
  COFD[1105] = 3.24099818E+00;
  COFD[1106] = -2.07879868E-01;
  COFD[1107] = 9.08046161E-03;
  COFD[1108] = -1.14322126E+01;
  COFD[1109] = 2.42102995E+00;
  COFD[1110] = -1.01354401E-01;
  COFD[1111] = 4.45948069E-03;
  COFD[1112] = -1.41953189E+01;
  COFD[1113] = 3.68995707E+00;
  COFD[1114] = -2.64784442E-01;
  COFD[1115] = 1.14891794E-02;
  COFD[1116] = -1.50366933E+01;
  COFD[1117] = 3.35941694E+00;
  COFD[1118] = -2.23126552E-01;
  COFD[1119] = 9.73586922E-03;
  COFD[1120] = -1.38538393E+01;
  COFD[1121] = 3.06403296E+00;
  COFD[1122] = -1.85549879E-01;
  COFD[1123] = 8.14070073E-03;
  COFD[1124] = -2.06777636E+01;
  COFD[1125] = 5.46880564E+00;
  COFD[1126] = -4.74397826E-01;
  COFD[1127] = 1.97344800E-02;
  COFD[1128] = -1.38702007E+01;
  COFD[1129] = 3.06403296E+00;
  COFD[1130] = -1.85549879E-01;
  COFD[1131] = 8.14070073E-03;
  COFD[1132] = -1.50478662E+01;
  COFD[1133] = 3.35941694E+00;
  COFD[1134] = -2.23126552E-01;
  COFD[1135] = 9.73586922E-03;
  COFD[1136] = -1.50424192E+01;
  COFD[1137] = 3.35941694E+00;
  COFD[1138] = -2.23126552E-01;
  COFD[1139] = 9.73586922E-03;
  COFD[1140] = -1.47729612E+01;
  COFD[1141] = 3.24835477E+00;
  COFD[1142] = -2.08830747E-01;
  COFD[1143] = 9.12148470E-03;
  COFD[1144] = -1.78202286E+01;
  COFD[1145] = 4.31993263E+00;
  COFD[1146] = -3.42851630E-01;
  COFD[1147] = 1.47243122E-02;
  COFD[1148] = -1.57442430E+01;
  COFD[1149] = 3.64418347E+00;
  COFD[1150] = -2.58820093E-01;
  COFD[1151] = 1.12296295E-02;
  COFD[1152] = -1.58288917E+01;
  COFD[1153] = 3.67755209E+00;
  COFD[1154] = -2.63170905E-01;
  COFD[1155] = 1.14190784E-02;
  COFD[1156] = -1.58092152E+01;
  COFD[1157] = 3.67755209E+00;
  COFD[1158] = -2.63170905E-01;
  COFD[1159] = 1.14190784E-02;
  COFD[1160] = -1.58092152E+01;
  COFD[1161] = 3.67755209E+00;
  COFD[1162] = -2.63170905E-01;
  COFD[1163] = 1.14190784E-02;
  COFD[1164] = -1.36610957E+01;
  COFD[1165] = 2.95689281E+00;
  COFD[1166] = -1.71804447E-01;
  COFD[1167] = 7.55232843E-03;
  COFD[1168] = -1.37952463E+01;
  COFD[1169] = 3.06403296E+00;
  COFD[1170] = -1.85549879E-01;
  COFD[1171] = 8.14070073E-03;
  COFD[1172] = -2.01735995E+01;
  COFD[1173] = 5.17081088E+00;
  COFD[1174] = -4.43888632E-01;
  COFD[1175] = 1.87323735E-02;
  COFD[1176] = -1.98752847E+01;
  COFD[1177] = 5.06690169E+00;
  COFD[1178] = -4.31863731E-01;
  COFD[1179] = 1.82682527E-02;
  COFD[1180] = -1.98752847E+01;
  COFD[1181] = 5.06690169E+00;
  COFD[1182] = -4.31863731E-01;
  COFD[1183] = 1.82682527E-02;
  COFD[1184] = -2.02538837E+01;
  COFD[1185] = 5.20405648E+00;
  COFD[1186] = -4.47505457E-01;
  COFD[1187] = 1.88615428E-02;
  COFD[1188] = -2.02471926E+01;
  COFD[1189] = 5.20405648E+00;
  COFD[1190] = -4.47505457E-01;
  COFD[1191] = 1.88615428E-02;
  COFD[1192] = -1.39934893E+01;
  COFD[1193] = 3.06403296E+00;
  COFD[1194] = -1.85549879E-01;
  COFD[1195] = 8.14070073E-03;
  COFD[1196] = -1.47917050E+01;
  COFD[1197] = 3.34581072E+00;
  COFD[1198] = -2.21377649E-01;
  COFD[1199] = 9.66079817E-03;
  COFD[1200] = -1.56749318E+01;
  COFD[1201] = 3.51447210E+00;
  COFD[1202] = -2.42579007E-01;
  COFD[1203] = 1.05506318E-02;
  COFD[1204] = -1.20579435E+01;
  COFD[1205] = 2.63303536E+00;
  COFD[1206] = -1.29792632E-01;
  COFD[1207] = 5.73363738E-03;
  COFD[1208] = -1.51177800E+01;
  COFD[1209] = 3.99904647E+00;
  COFD[1210] = -3.03517220E-01;
  COFD[1211] = 1.31117363E-02;
  COFD[1212] = -1.59068122E+01;
  COFD[1213] = 3.60186887E+00;
  COFD[1214] = -2.53302622E-01;
  COFD[1215] = 1.09893496E-02;
  COFD[1216] = -1.46739282E+01;
  COFD[1217] = 3.30683499E+00;
  COFD[1218] = -2.16378602E-01;
  COFD[1219] = 9.44670561E-03;
  COFD[1220] = -2.12365298E+01;
  COFD[1221] = 5.53137765E+00;
  COFD[1222] = -4.73118467E-01;
  COFD[1223] = 1.93125852E-02;
  COFD[1224] = -1.46940858E+01;
  COFD[1225] = 3.30683499E+00;
  COFD[1226] = -2.16378602E-01;
  COFD[1227] = 9.44670561E-03;
  COFD[1228] = -1.59218538E+01;
  COFD[1229] = 3.60186887E+00;
  COFD[1230] = -2.53302622E-01;
  COFD[1231] = 1.09893496E-02;
  COFD[1232] = -1.59145062E+01;
  COFD[1233] = 3.60186887E+00;
  COFD[1234] = -2.53302622E-01;
  COFD[1235] = 1.09893496E-02;
  COFD[1236] = -1.56952963E+01;
  COFD[1237] = 3.51861272E+00;
  COFD[1238] = -2.43068621E-01;
  COFD[1239] = 1.05698368E-02;
  COFD[1240] = -1.89571393E+01;
  COFD[1241] = 4.64763677E+00;
  COFD[1242] = -3.82799418E-01;
  COFD[1243] = 1.63539171E-02;
  COFD[1244] = -1.68027382E+01;
  COFD[1245] = 3.97899248E+00;
  COFD[1246] = -3.01178156E-01;
  COFD[1247] = 1.30211921E-02;
  COFD[1248] = -1.68409184E+01;
  COFD[1249] = 3.99442024E+00;
  COFD[1250] = -3.03003880E-01;
  COFD[1251] = 1.30931075E-02;
  COFD[1252] = -1.68170523E+01;
  COFD[1253] = 3.99442024E+00;
  COFD[1254] = -3.03003880E-01;
  COFD[1255] = 1.30931075E-02;
  COFD[1256] = -1.68170523E+01;
  COFD[1257] = 3.99442024E+00;
  COFD[1258] = -3.03003880E-01;
  COFD[1259] = 1.30931075E-02;
  COFD[1260] = -1.44057543E+01;
  COFD[1261] = 3.17964349E+00;
  COFD[1262] = -2.00113588E-01;
  COFD[1263] = 8.75259314E-03;
  COFD[1264] = -1.46029188E+01;
  COFD[1265] = 3.30683499E+00;
  COFD[1266] = -2.16378602E-01;
  COFD[1267] = 9.44670561E-03;
  COFD[1268] = -2.10549431E+01;
  COFD[1269] = 5.37765260E+00;
  COFD[1270] = -4.64755221E-01;
  COFD[1271] = 1.94011994E-02;
  COFD[1272] = -2.08483029E+01;
  COFD[1273] = 5.31248163E+00;
  COFD[1274] = -4.58014127E-01;
  COFD[1275] = 1.91765075E-02;
  COFD[1276] = -2.08483029E+01;
  COFD[1277] = 5.31248163E+00;
  COFD[1278] = -4.58014127E-01;
  COFD[1279] = 1.91765075E-02;
  COFD[1280] = -2.11253124E+01;
  COFD[1281] = 5.40762714E+00;
  COFD[1282] = -4.67856822E-01;
  COFD[1283] = 1.95051950E-02;
  COFD[1284] = -2.11164313E+01;
  COFD[1285] = 5.40762714E+00;
  COFD[1286] = -4.67856822E-01;
  COFD[1287] = 1.95051950E-02;
  COFD[1288] = -1.48516207E+01;
  COFD[1289] = 3.30683499E+00;
  COFD[1290] = -2.16378602E-01;
  COFD[1291] = 9.44670561E-03;
  COFD[1292] = -1.56182178E+01;
  COFD[1293] = 3.58311501E+00;
  COFD[1294] = -2.50860151E-01;
  COFD[1295] = 1.08830877E-02;
  COFD[1296] = -1.67835296E+01;
  COFD[1297] = 3.94346012E+00;
  COFD[1298] = -2.96835271E-01;
  COFD[1299] = 1.28438696E-02;
  COFD[1300] = -2.46746425E+01;
  COFD[1301] = 5.51584151E+00;
  COFD[1302] = -4.76108968E-01;
  COFD[1303] = 1.96365975E-02;
  COFD[1304] = -2.18596191E+01;
  COFD[1305] = 4.62023553E+00;
  COFD[1306] = -3.79733248E-01;
  COFD[1307] = 1.62407913E-02;
  COFD[1308] = -2.50118194E+01;
  COFD[1309] = 5.51533435E+00;
  COFD[1310] = -4.61886312E-01;
  COFD[1311] = 1.84746261E-02;
  COFD[1312] = -2.47604202E+01;
  COFD[1313] = 5.51868821E+00;
  COFD[1314] = -4.73225401E-01;
  COFD[1315] = 1.93849523E-02;
  COFD[1316] = -2.42503398E+01;
  COFD[1317] = 5.40793558E+00;
  COFD[1318] = -4.67888803E-01;
  COFD[1319] = 1.95062704E-02;
  COFD[1320] = -1.78173107E+01;
  COFD[1321] = 1.97328753E+00;
  COFD[1322] = 7.64115525E-02;
  COFD[1323] = -7.78429580E-03;
  COFD[1324] = -2.42503408E+01;
  COFD[1325] = 5.40793558E+00;
  COFD[1326] = -4.67888803E-01;
  COFD[1327] = 1.95062704E-02;
  COFD[1328] = -2.47604207E+01;
  COFD[1329] = 5.51868821E+00;
  COFD[1330] = -4.73225401E-01;
  COFD[1331] = 1.93849523E-02;
  COFD[1332] = -2.47604204E+01;
  COFD[1333] = 5.51868821E+00;
  COFD[1334] = -4.73225401E-01;
  COFD[1335] = 1.93849523E-02;
  COFD[1336] = -2.46791388E+01;
  COFD[1337] = 5.51562323E+00;
  COFD[1338] = -4.75876136E-01;
  COFD[1339] = 1.96184735E-02;
  COFD[1340] = -2.38843968E+01;
  COFD[1341] = 4.82941387E+00;
  COFD[1342] = -3.46543093E-01;
  COFD[1343] = 1.24648862E-02;
  COFD[1344] = -2.50221925E+01;
  COFD[1345] = 5.52564127E+00;
  COFD[1346] = -4.64221065E-01;
  COFD[1347] = 1.86139427E-02;
  COFD[1348] = -2.50224826E+01;
  COFD[1349] = 5.51888287E+00;
  COFD[1350] = -4.62631724E-01;
  COFD[1351] = 1.85178829E-02;
  COFD[1352] = -2.50224813E+01;
  COFD[1353] = 5.51888287E+00;
  COFD[1354] = -4.62631724E-01;
  COFD[1355] = 1.85178829E-02;
  COFD[1356] = -2.50224813E+01;
  COFD[1357] = 5.51888287E+00;
  COFD[1358] = -4.62631724E-01;
  COFD[1359] = 1.85178829E-02;
  COFD[1360] = -2.39197394E+01;
  COFD[1361] = 5.30539812E+00;
  COFD[1362] = -4.57330353E-01;
  COFD[1363] = 1.91560814E-02;
  COFD[1364] = -2.42503358E+01;
  COFD[1365] = 5.40793558E+00;
  COFD[1366] = -4.67888803E-01;
  COFD[1367] = 1.95062704E-02;
  COFD[1368] = -1.94498128E+01;
  COFD[1369] = 2.70960748E+00;
  COFD[1370] = -2.96143895E-02;
  COFD[1371] = -2.82061574E-03;
  COFD[1372] = -2.05464448E+01;
  COFD[1373] = 3.22046026E+00;
  COFD[1374] = -1.04593424E-01;
  COFD[1375] = 7.40271888E-04;
  COFD[1376] = -2.05464448E+01;
  COFD[1377] = 3.22046026E+00;
  COFD[1378] = -1.04593424E-01;
  COFD[1379] = 7.40271888E-04;
  COFD[1380] = -1.91495340E+01;
  COFD[1381] = 2.57271534E+00;
  COFD[1382] = -9.78488237E-03;
  COFD[1383] = -3.75334124E-03;
  COFD[1384] = -1.91495337E+01;
  COFD[1385] = 2.57271534E+00;
  COFD[1386] = -9.78488237E-03;
  COFD[1387] = -3.75334124E-03;
  COFD[1388] = -2.42503475E+01;
  COFD[1389] = 5.40793558E+00;
  COFD[1390] = -4.67888803E-01;
  COFD[1391] = 1.95062704E-02;
  COFD[1392] = -2.47385674E+01;
  COFD[1393] = 5.51399489E+00;
  COFD[1394] = -4.72948445E-01;
  COFD[1395] = 1.93856424E-02;
  COFD[1396] = -2.49944573E+01;
  COFD[1397] = 5.52779193E+00;
  COFD[1398] = -4.65764754E-01;
  COFD[1399] = 1.87288224E-02;
  COFD[1400] = -1.47803548E+01;
  COFD[1401] = 1.70835496E-01;
  COFD[1402] = 3.31563490E-01;
  COFD[1403] = -1.95427081E-02;
}

// List of specs with small weight, dim NLITE
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetKTDIF(int* KTDIF)
{
  KTDIF[0] = 1;
  KTDIF[1] = 2;
  KTDIF[2] = 25;
}

// Poly fits for thermal diff ratios, dim NO*NLITE*KK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
egtransetCOFTD(amrex::Real* COFTD)
{
  COFTD[0] = 4.45261429E-01;
  COFTD[1] = 4.94696577E-05;
  COFTD[2] = -2.63023124E-08;
  COFTD[3] = 4.90305626E-12;
  COFTD[4] = 0.00000000E+00;
  COFTD[5] = 0.00000000E+00;
  COFTD[6] = 0.00000000E+00;
  COFTD[7] = 0.00000000E+00;
  COFTD[8] = -1.52534742E-01;
  COFTD[9] = -5.46404022E-05;
  COFTD[10] = 2.93412470E-08;
  COFTD[11] = -4.87091914E-12;
  COFTD[12] = 4.42736017E-01;
  COFTD[13] = 7.11765887E-05;
  COFTD[14] = -3.84765396E-08;
  COFTD[15] = 6.86318682E-12;
  COFTD[16] = 4.15577509E-01;
  COFTD[17] = 1.09736860E-05;
  COFTD[18] = -3.96016410E-09;
  COFTD[19] = 1.14412838E-12;
  COFTD[20] = 6.02020725E-02;
  COFTD[21] = 5.61555766E-04;
  COFTD[22] = -2.55370076E-07;
  COFTD[23] = 3.63385938E-11;
  COFTD[24] = 4.21927217E-01;
  COFTD[25] = 1.11413555E-05;
  COFTD[26] = -4.02067239E-09;
  COFTD[27] = 1.16160979E-12;
  COFTD[28] = 4.46067449E-01;
  COFTD[29] = 7.17121674E-05;
  COFTD[30] = -3.87660620E-08;
  COFTD[31] = 6.91482989E-12;
  COFTD[32] = 4.44449675E-01;
  COFTD[33] = 7.14520855E-05;
  COFTD[34] = -3.86254673E-08;
  COFTD[35] = 6.88975155E-12;
  COFTD[36] = 4.44650439E-01;
  COFTD[37] = 5.06628083E-05;
  COFTD[38] = -2.69818971E-08;
  COFTD[39] = 5.01286176E-12;
  COFTD[40] = 3.25740914E-01;
  COFTD[41] = 3.03631979E-04;
  COFTD[42] = -1.55289598E-07;
  COFTD[43] = 2.41465297E-11;
  COFTD[44] = 3.58655012E-01;
  COFTD[45] = 1.21524029E-04;
  COFTD[46] = -6.53752231E-08;
  COFTD[47] = 1.09001717E-11;
  COFTD[48] = 3.50272421E-01;
  COFTD[49] = 1.23583915E-04;
  COFTD[50] = -6.63969060E-08;
  COFTD[51] = 1.10354999E-11;
  COFTD[52] = 3.43456512E-01;
  COFTD[53] = 1.21179111E-04;
  COFTD[54] = -6.51048965E-08;
  COFTD[55] = 1.08207614E-11;
  COFTD[56] = 3.43456512E-01;
  COFTD[57] = 1.21179111E-04;
  COFTD[58] = -6.51048965E-08;
  COFTD[59] = 1.08207614E-11;
  COFTD[60] = 3.88529645E-01;
  COFTD[61] = -4.91887876E-06;
  COFTD[62] = 5.69539773E-09;
  COFTD[63] = -5.68222508E-13;
  COFTD[64] = 3.91826338E-01;
  COFTD[65] = 1.03465156E-05;
  COFTD[66] = -3.73383198E-09;
  COFTD[67] = 1.07873892E-12;
  COFTD[68] = 1.70978021E-01;
  COFTD[69] = 4.98264784E-04;
  COFTD[70] = -2.37738756E-07;
  COFTD[71] = 3.49395808E-11;
  COFTD[72] = 1.79391459E-01;
  COFTD[73] = 4.84490026E-04;
  COFTD[74] = -2.32134571E-07;
  COFTD[75] = 3.42178255E-11;
  COFTD[76] = 1.79391459E-01;
  COFTD[77] = 4.84490026E-04;
  COFTD[78] = -2.32134571E-07;
  COFTD[79] = 3.42178255E-11;
  COFTD[80] = 1.62149345E-01;
  COFTD[81] = 5.03434613E-04;
  COFTD[82] = -2.39391657E-07;
  COFTD[83] = 3.50983128E-11;
  COFTD[84] = 1.61391212E-01;
  COFTD[85] = 5.01080793E-04;
  COFTD[86] = -2.38272376E-07;
  COFTD[87] = 3.49342098E-11;
  COFTD[88] = 4.65846649E-01;
  COFTD[89] = 1.23010864E-05;
  COFTD[90] = -4.43919397E-09;
  COFTD[91] = 1.28252458E-12;
  COFTD[92] = 4.07197076E-01;
  COFTD[93] = 6.29805002E-05;
  COFTD[94] = -3.40117037E-08;
  COFTD[95] = 6.09207140E-12;
  COFTD[96] = 4.12032475E-01;
  COFTD[97] = 1.28802637E-04;
  COFTD[98] = -6.94527177E-08;
  COFTD[99] = 1.16546716E-11;
  COFTD[100] = -4.83575595E-02;
  COFTD[101] = -7.29902876E-04;
  COFTD[102] = 3.28156716E-07;
  COFTD[103] = -4.63569261E-11;
  COFTD[104] = 2.40744276E-01;
  COFTD[105] = 4.45343183E-04;
  COFTD[106] = -2.18173742E-07;
  COFTD[107] = 3.26958310E-11;
  COFTD[108] = 1.52534742E-01;
  COFTD[109] = 5.46404022E-05;
  COFTD[110] = -2.93412470E-08;
  COFTD[111] = 4.87091914E-12;
  COFTD[112] = 0.00000000E+00;
  COFTD[113] = 0.00000000E+00;
  COFTD[114] = 0.00000000E+00;
  COFTD[115] = 0.00000000E+00;
  COFTD[116] = 2.20482081E-01;
  COFTD[117] = 4.80162630E-04;
  COFTD[118] = -2.32927140E-07;
  COFTD[119] = 3.46469239E-11;
  COFTD[120] = 2.70008280E-01;
  COFTD[121] = 3.61552588E-04;
  COFTD[122] = -1.80743499E-07;
  COFTD[123] = 2.75319341E-11;
  COFTD[124] = -1.41882969E-01;
  COFTD[125] = 7.66554625E-04;
  COFTD[126] = -3.06548330E-07;
  COFTD[127] = 4.02957302E-11;
  COFTD[128] = 2.72039997E-01;
  COFTD[129] = 3.64273144E-04;
  COFTD[130] = -1.82103531E-07;
  COFTD[131] = 2.77391022E-11;
  COFTD[132] = 2.21307723E-01;
  COFTD[133] = 4.81960701E-04;
  COFTD[134] = -2.33799385E-07;
  COFTD[135] = 3.47766667E-11;
  COFTD[136] = 2.20907136E-01;
  COFTD[137] = 4.81088308E-04;
  COFTD[138] = -2.33376187E-07;
  COFTD[139] = 3.47137178E-11;
  COFTD[140] = 2.39409087E-01;
  COFTD[141] = 4.47195587E-04;
  COFTD[142] = -2.18950923E-07;
  COFTD[143] = 3.27972343E-11;
  COFTD[144] = 2.44368810E-02;
  COFTD[145] = 7.18240808E-04;
  COFTD[146] = -3.19717751E-07;
  COFTD[147] = 4.48827629E-11;
  COFTD[148] = 1.45433118E-01;
  COFTD[149] = 5.30347483E-04;
  COFTD[150] = -2.50118653E-07;
  COFTD[151] = 3.64613999E-11;
  COFTD[152] = 1.40109849E-01;
  COFTD[153] = 5.30876682E-04;
  COFTD[154] = -2.49898194E-07;
  COFTD[155] = 3.63822692E-11;
  COFTD[156] = 1.38759794E-01;
  COFTD[157] = 5.25761319E-04;
  COFTD[158] = -2.47490252E-07;
  COFTD[159] = 3.60317009E-11;
  COFTD[160] = 1.38759794E-01;
  COFTD[161] = 5.25761319E-04;
  COFTD[162] = -2.47490252E-07;
  COFTD[163] = 3.60317009E-11;
  COFTD[164] = 2.81033340E-01;
  COFTD[165] = 3.13353760E-04;
  COFTD[166] = -1.58475652E-07;
  COFTD[167] = 2.43852063E-11;
  COFTD[168] = 2.62294123E-01;
  COFTD[169] = 3.51223003E-04;
  COFTD[170] = -1.75579644E-07;
  COFTD[171] = 2.67453446E-11;
  COFTD[172] = -1.19542204E-01;
  COFTD[173] = 8.00169896E-04;
  COFTD[174] = -3.27980898E-07;
  COFTD[175] = 4.37706567E-11;
  COFTD[176] = -9.21021062E-02;
  COFTD[177] = 7.88085864E-04;
  COFTD[178] = -3.29283183E-07;
  COFTD[179] = 4.44614948E-11;
  COFTD[180] = -9.21021062E-02;
  COFTD[181] = 7.88085864E-04;
  COFTD[182] = -3.29283183E-07;
  COFTD[183] = 4.44614948E-11;
  COFTD[184] = -1.24938974E-01;
  COFTD[185] = 7.98385053E-04;
  COFTD[186] = -3.25757471E-07;
  COFTD[187] = 4.33526419E-11;
  COFTD[188] = -1.24647578E-01;
  COFTD[189] = 7.96522972E-04;
  COFTD[190] = -3.24997704E-07;
  COFTD[191] = 4.32515301E-11;
  COFTD[192] = 2.85751979E-01;
  COFTD[193] = 3.82634072E-04;
  COFTD[194] = -1.91282329E-07;
  COFTD[195] = 2.91372719E-11;
  COFTD[196] = 2.13713591E-01;
  COFTD[197] = 4.56510862E-04;
  COFTD[198] = -2.21708032E-07;
  COFTD[199] = 3.30064357E-11;
  COFTD[200] = 1.63366239E-01;
  COFTD[201] = 5.54240001E-04;
  COFTD[202] = -2.62337753E-07;
  COFTD[203] = 3.83388181E-11;
  COFTD[204] = 2.14687863E-01;
  COFTD[205] = -8.33408992E-04;
  COFTD[206] = 3.11551008E-07;
  COFTD[207] = -3.92419743E-11;
  COFTD[208] = -1.58494270E-01;
  COFTD[209] = 8.57382603E-04;
  COFTD[210] = -3.42931140E-07;
  COFTD[211] = 4.50831409E-11;
  COFTD[212] = 4.83575595E-02;
  COFTD[213] = 7.29902876E-04;
  COFTD[214] = -3.28156716E-07;
  COFTD[215] = 4.63569261E-11;
  COFTD[216] = -2.14687863E-01;
  COFTD[217] = 8.33408992E-04;
  COFTD[218] = -3.11551008E-07;
  COFTD[219] = 3.92419743E-11;
  COFTD[220] = -1.74367468E-01;
  COFTD[221] = 8.56399915E-04;
  COFTD[222] = -3.37494379E-07;
  COFTD[223] = 4.39622394E-11;
  COFTD[224] = -1.22051466E-01;
  COFTD[225] = 8.50463321E-04;
  COFTD[226] = -3.49880208E-07;
  COFTD[227] = 4.67982394E-11;
  COFTD[228] = -1.82074509E-01;
  COFTD[229] = 3.58494859E-04;
  COFTD[230] = -4.14797572E-08;
  COFTD[231] = -1.91123311E-12;
  COFTD[232] = -1.22051962E-01;
  COFTD[233] = 8.50466778E-04;
  COFTD[234] = -3.49881630E-07;
  COFTD[235] = 4.67984296E-11;
  COFTD[236] = -1.74367822E-01;
  COFTD[237] = 8.56401656E-04;
  COFTD[238] = -3.37495065E-07;
  COFTD[239] = 4.39623287E-11;
  COFTD[240] = -1.74367651E-01;
  COFTD[241] = 8.56400812E-04;
  COFTD[242] = -3.37494732E-07;
  COFTD[243] = 4.39622854E-11;
  COFTD[244] = -1.59489983E-01;
  COFTD[245] = 8.57405115E-04;
  COFTD[246] = -3.42640753E-07;
  COFTD[247] = 4.50206824E-11;
  COFTD[248] = -2.48073493E-01;
  COFTD[249] = 7.25180496E-04;
  COFTD[250] = -2.36929722E-07;
  COFTD[251] = 2.71093741E-11;
  COFTD[252] = -2.12064065E-01;
  COFTD[253] = 8.37240651E-04;
  COFTD[254] = -3.14517782E-07;
  COFTD[255] = 3.97420162E-11;
  COFTD[256] = -2.14134032E-01;
  COFTD[257] = 8.35097175E-04;
  COFTD[258] = -3.12605803E-07;
  COFTD[259] = 3.94096885E-11;
  COFTD[260] = -2.14132909E-01;
  COFTD[261] = 8.35092796E-04;
  COFTD[262] = -3.12604164E-07;
  COFTD[263] = 3.94094818E-11;
  COFTD[264] = -2.14132909E-01;
  COFTD[265] = 8.35092796E-04;
  COFTD[266] = -3.12604164E-07;
  COFTD[267] = 3.94094818E-11;
  COFTD[268] = -9.90513879E-02;
  COFTD[269] = 8.41283392E-04;
  COFTD[270] = -3.51342290E-07;
  COFTD[271] = 4.74261956E-11;
  COFTD[272] = -1.22049550E-01;
  COFTD[273] = 8.50449971E-04;
  COFTD[274] = -3.49874716E-07;
  COFTD[275] = 4.67975048E-11;
  COFTD[276] = -2.08192097E-01;
  COFTD[277] = 4.49774819E-04;
  COFTD[278] = -8.69551363E-08;
  COFTD[279] = 4.64889109E-12;
  COFTD[280] = -2.25054527E-01;
  COFTD[281] = 5.20024585E-04;
  COFTD[282] = -1.23031218E-07;
  COFTD[283] = 9.92192373E-12;
  COFTD[284] = -2.25054527E-01;
  COFTD[285] = 5.20024585E-04;
  COFTD[286] = -1.23031218E-07;
  COFTD[287] = 9.92192373E-12;
  COFTD[288] = -2.03684878E-01;
  COFTD[289] = 4.32867875E-04;
  COFTD[290] = -7.84216895E-08;
  COFTD[291] = 3.41085687E-12;
  COFTD[292] = -2.03684620E-01;
  COFTD[293] = 4.32867326E-04;
  COFTD[294] = -7.84215900E-08;
  COFTD[295] = 3.41085254E-12;
  COFTD[296] = -1.22055221E-01;
  COFTD[297] = 8.50489487E-04;
  COFTD[298] = -3.49890973E-07;
  COFTD[299] = 4.67996793E-11;
  COFTD[300] = -1.72580282E-01;
  COFTD[301] = 8.56644825E-04;
  COFTD[302] = -3.38190010E-07;
  COFTD[303] = 4.41015047E-11;
  COFTD[304] = -2.07905569E-01;
  COFTD[305] = 8.41099546E-04;
  COFTD[306] = -3.18089807E-07;
  COFTD[307] = 4.03679127E-11;
  COFTD[308] = 0.00000000E+00;
  COFTD[309] = 0.00000000E+00;
  COFTD[310] = 0.00000000E+00;
  COFTD[311] = 0.00000000E+00;
}

// compute the critical parameters for each species
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS(
  amrex::Real* Tci, amrex::Real* ai, amrex::Real* bi, amrex::Real* acentric_i)
{

  amrex::Real EPS[26];
  amrex::Real SIG[26];
  amrex::Real wt[26];
  amrex::Real Rcst = 83.144598; // in bar [CGS] !
  amrex::Real avogadro = 6.02214199e23;
  amrex::Real boltzmann = 1.3806503e-16; // we work in CGS

  egtransetEPS(EPS);
  egtransetSIG(SIG);
  get_mw(wt);

  // species 0: N2
  // Imported from NIST
  Tci[0] = 126.192000;
  ai[0] = 1e6 * 0.42748 * Rcst * Rcst * Tci[0] * Tci[0] /
          (28.013400 * 28.013400 * 33.958000);
  bi[0] = 0.08664 * Rcst * Tci[0] / (28.013400 * 33.958000);
  acentric_i[0] = 0.037200;

  // species 1: H2
  // Imported from NIST
  Tci[1] = 33.145000;
  ai[1] = 1e6 * 0.42748 * Rcst * Rcst * Tci[1] * Tci[1] /
          (2.015880 * 2.015880 * 12.964000);
  bi[1] = 0.08664 * Rcst * Tci[1] / (2.015880 * 12.964000);
  acentric_i[1] = -0.219000;

  // species 2: H
  Tci[2] = 1.316 * EPS[2];
  ai[2] = (5.55 * avogadro * avogadro * EPS[2] * boltzmann * 1e-24 * SIG[2] *
           SIG[2] * SIG[2]) /
          (wt[2] * wt[2]);
  bi[2] = 0.855 * avogadro * 1e-24 * SIG[2] * SIG[2] * SIG[2] / (wt[2]);
  acentric_i[2] = 0.0;

  // species 3: O2
  // Imported from NIST
  Tci[3] = 154.581000;
  ai[3] = 1e6 * 0.42748 * Rcst * Rcst * Tci[3] * Tci[3] /
          (31.998800 * 31.998800 * 50.430466);
  bi[3] = 0.08664 * Rcst * Tci[3] / (31.998800 * 50.430466);
  acentric_i[3] = 0.022200;

  // species 4: O
  Tci[4] = 1.316 * EPS[4];
  ai[4] = (5.55 * avogadro * avogadro * EPS[4] * boltzmann * 1e-24 * SIG[4] *
           SIG[4] * SIG[4]) /
          (wt[4] * wt[4]);
  bi[4] = 0.855 * avogadro * 1e-24 * SIG[4] * SIG[4] * SIG[4] / (wt[4]);
  acentric_i[4] = 0.0;

  // species 5: H2O
  // Imported from NIST
  Tci[5] = 647.096000;
  ai[5] = 1e6 * 0.42748 * Rcst * Rcst * Tci[5] * Tci[5] /
          (18.015340 * 18.015340 * 220.640000);
  bi[5] = 0.08664 * Rcst * Tci[5] / (18.015340 * 220.640000);
  acentric_i[5] = 0.344300;

  // species 6: OH
  Tci[6] = 1.316 * EPS[6];
  ai[6] = (5.55 * avogadro * avogadro * EPS[6] * boltzmann * 1e-24 * SIG[6] *
           SIG[6] * SIG[6]) /
          (wt[6] * wt[6]);
  bi[6] = 0.855 * avogadro * 1e-24 * SIG[6] * SIG[6] * SIG[6] / (wt[6]);
  acentric_i[6] = 0.0;

  // species 7: H2O2
  Tci[7] = 1.316 * EPS[7];
  ai[7] = (5.55 * avogadro * avogadro * EPS[7] * boltzmann * 1e-24 * SIG[7] *
           SIG[7] * SIG[7]) /
          (wt[7] * wt[7]);
  bi[7] = 0.855 * avogadro * 1e-24 * SIG[7] * SIG[7] * SIG[7] / (wt[7]);
  acentric_i[7] = 0.0;

  // species 8: HO2
  Tci[8] = 1.316 * EPS[8];
  ai[8] = (5.55 * avogadro * avogadro * EPS[8] * boltzmann * 1e-24 * SIG[8] *
           SIG[8] * SIG[8]) /
          (wt[8] * wt[8]);
  bi[8] = 0.855 * avogadro * 1e-24 * SIG[8] * SIG[8] * SIG[8] / (wt[8]);
  acentric_i[8] = 0.0;

  // species 9: CO
  // Imported from NIST
  Tci[9] = 132.850000;
  ai[9] = 1e6 * 0.42748 * Rcst * Rcst * Tci[9] * Tci[9] /
          (28.010000 * 28.010000 * 34.940000);
  bi[9] = 0.08664 * Rcst * Tci[9] / (28.010000 * 34.940000);
  acentric_i[9] = 0.045000;

  // species 10: CO2
  // Imported from NIST
  Tci[10] = 304.120000;
  ai[10] = 1e6 * 0.42748 * Rcst * Rcst * Tci[10] * Tci[10] /
           (44.009950 * 44.009950 * 73.740000);
  bi[10] = 0.08664 * Rcst * Tci[10] / (44.009950 * 73.740000);
  acentric_i[10] = 0.225000;

  // species 11: CH4
  // Imported from NIST
  Tci[11] = 190.560000;
  ai[11] = 1e6 * 0.42748 * Rcst * Rcst * Tci[11] * Tci[11] /
           (16.043030 * 16.043030 * 45.990000);
  bi[11] = 0.08664 * Rcst * Tci[11] / (16.043030 * 45.990000);
  acentric_i[11] = 0.011000;

  // species 12: CH3
  Tci[12] = 1.316 * EPS[12];
  ai[12] = (5.55 * avogadro * avogadro * EPS[12] * boltzmann * 1e-24 * SIG[12] *
            SIG[12] * SIG[12]) /
           (wt[12] * wt[12]);
  bi[12] = 0.855 * avogadro * 1e-24 * SIG[12] * SIG[12] * SIG[12] / (wt[12]);
  acentric_i[12] = 0.0;

  // species 13: CH2
  Tci[13] = 1.316 * EPS[13];
  ai[13] = (5.55 * avogadro * avogadro * EPS[13] * boltzmann * 1e-24 * SIG[13] *
            SIG[13] * SIG[13]) /
           (wt[13] * wt[13]);
  bi[13] = 0.855 * avogadro * 1e-24 * SIG[13] * SIG[13] * SIG[13] / (wt[13]);
  acentric_i[13] = 0.0;

  // species 14: CH2(S)
  Tci[14] = 1.316 * EPS[14];
  ai[14] = (5.55 * avogadro * avogadro * EPS[14] * boltzmann * 1e-24 * SIG[14] *
            SIG[14] * SIG[14]) /
           (wt[14] * wt[14]);
  bi[14] = 0.855 * avogadro * 1e-24 * SIG[14] * SIG[14] * SIG[14] / (wt[14]);
  acentric_i[14] = 0.0;

  // species 15: C
  Tci[15] = 1.316 * EPS[15];
  ai[15] = (5.55 * avogadro * avogadro * EPS[15] * boltzmann * 1e-24 * SIG[15] *
            SIG[15] * SIG[15]) /
           (wt[15] * wt[15]);
  bi[15] = 0.855 * avogadro * 1e-24 * SIG[15] * SIG[15] * SIG[15] / (wt[15]);
  acentric_i[15] = 0.0;

  // species 16: CH
  Tci[16] = 1.316 * EPS[16];
  ai[16] = (5.55 * avogadro * avogadro * EPS[16] * boltzmann * 1e-24 * SIG[16] *
            SIG[16] * SIG[16]) /
           (wt[16] * wt[16]);
  bi[16] = 0.855 * avogadro * 1e-24 * SIG[16] * SIG[16] * SIG[16] / (wt[16]);
  acentric_i[16] = 0.0;

  // species 17: CH3OH
  // Imported from NIST
  Tci[17] = 512.640000;
  ai[17] = 1e6 * 0.42748 * Rcst * Rcst * Tci[17] * Tci[17] /
           (32.042000 * 32.042000 * 80.970000);
  bi[17] = 0.08664 * Rcst * Tci[17] / (32.042000 * 80.970000);
  acentric_i[17] = 0.565000;

  // species 18: CH3O
  Tci[18] = 1.316 * EPS[18];
  ai[18] = (5.55 * avogadro * avogadro * EPS[18] * boltzmann * 1e-24 * SIG[18] *
            SIG[18] * SIG[18]) /
           (wt[18] * wt[18]);
  bi[18] = 0.855 * avogadro * 1e-24 * SIG[18] * SIG[18] * SIG[18] / (wt[18]);
  acentric_i[18] = 0.0;

  // species 19: CH2OH
  Tci[19] = 1.316 * EPS[19];
  ai[19] = (5.55 * avogadro * avogadro * EPS[19] * boltzmann * 1e-24 * SIG[19] *
            SIG[19] * SIG[19]) /
           (wt[19] * wt[19]);
  bi[19] = 0.855 * avogadro * 1e-24 * SIG[19] * SIG[19] * SIG[19] / (wt[19]);
  acentric_i[19] = 0.0;

  // species 20: CH2O
  Tci[20] = 1.316 * EPS[20];
  ai[20] = (5.55 * avogadro * avogadro * EPS[20] * boltzmann * 1e-24 * SIG[20] *
            SIG[20] * SIG[20]) /
           (wt[20] * wt[20]);
  bi[20] = 0.855 * avogadro * 1e-24 * SIG[20] * SIG[20] * SIG[20] / (wt[20]);
  acentric_i[20] = 0.0;

  // species 21: HCO
  Tci[21] = 1.316 * EPS[21];
  ai[21] = (5.55 * avogadro * avogadro * EPS[21] * boltzmann * 1e-24 * SIG[21] *
            SIG[21] * SIG[21]) /
           (wt[21] * wt[21]);
  bi[21] = 0.855 * avogadro * 1e-24 * SIG[21] * SIG[21] * SIG[21] / (wt[21]);
  acentric_i[21] = 0.0;

  // species 22: CHO+
  Tci[22] = 1.316 * EPS[22];
  ai[22] = (5.55 * avogadro * avogadro * EPS[22] * boltzmann * 1e-24 * SIG[22] *
            SIG[22] * SIG[22]) /
           (wt[22] * wt[22]);
  bi[22] = 0.855 * avogadro * 1e-24 * SIG[22] * SIG[22] * SIG[22] / (wt[22]);
  acentric_i[22] = 0.0;

  // species 23: H3O+
  Tci[23] = 1.316 * EPS[23];
  ai[23] = (5.55 * avogadro * avogadro * EPS[23] * boltzmann * 1e-24 * SIG[23] *
            SIG[23] * SIG[23]) /
           (wt[23] * wt[23]);
  bi[23] = 0.855 * avogadro * 1e-24 * SIG[23] * SIG[23] * SIG[23] / (wt[23]);
  acentric_i[23] = 0.0;

  // species 24: O2-
  Tci[24] = 1.316 * EPS[24];
  ai[24] = (5.55 * avogadro * avogadro * EPS[24] * boltzmann * 1e-24 * SIG[24] *
            SIG[24] * SIG[24]) /
           (wt[24] * wt[24]);
  bi[24] = 0.855 * avogadro * 1e-24 * SIG[24] * SIG[24] * SIG[24] / (wt[24]);
  acentric_i[24] = 0.0;

  // species 25: E
  Tci[25] = 1.316 * EPS[25];
  ai[25] = (5.55 * avogadro * avogadro * EPS[25] * boltzmann * 1e-24 * SIG[25] *
            SIG[25] * SIG[25]) /
           (wt[25] * wt[25]);
  bi[25] = 0.855 * avogadro * 1e-24 * SIG[25] * SIG[25] * SIG[25] / (wt[25]);
  acentric_i[25] = 0.0;
}

// compute the critical parameter quantities for each species for SRK
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
GET_CRITPARAMS_SRK(
  amrex::Real* sqrtOneOverTc,
  amrex::Real* sqrtAsti,
  amrex::Real* Bi,
  amrex::Real* Fomega)
{

  // species 0: N2
  // Imported from NIST
  sqrtOneOverTc[0] = 8.9019282240563e-02;
  sqrtAsti[0] = 4.2022983929562e+04;
  Bi[0] = 9.5560052221563e-01;
  Fomega[0] = 5.4259343186608e-01;

  // species 1: H2
  // Imported from NIST
  sqrtOneOverTc[1] = 1.7369646834812e-01;
  sqrtAsti[1] = 2.4824229305911e+05;
  Bi[1] = 9.1362254169923e+00;
  Fomega[1] = 1.3798618890700e-01;

  // species 2: H
  // Computed from Lennard-Jones
  sqrtOneOverTc[2] = 7.2391579785311e-02;
  sqrtAsti[2] = 5.8451221185793e+05;
  Bi[2] = 4.4006634563410e+00;
  Fomega[2] = 4.8508000000000e-01;

  // species 3: O2
  // Imported from NIST
  sqrtOneOverTc[3] = 8.0430717653241e-02;
  sqrtAsti[3] = 3.6980081924740e+04;
  Bi[3] = 6.9005216444999e-01;
  Fomega[3] = 5.1945301904908e-01;

  // species 4: O
  // Computed from Lennard-Jones
  sqrtOneOverTc[4] = 9.7460146948181e-02;
  sqrtAsti[4] = 4.2500156012386e+04;
  Bi[4] = 6.6930234851885e-01;
  Fomega[4] = 4.8508000000000e-01;

  // species 5: H2O
  // Imported from NIST
  sqrtOneOverTc[5] = 3.9311140369715e-02;
  sqrtAsti[5] = 1.3145468419592e+05;
  Bi[5] = 1.1727204136223e+00;
  Fomega[5] = 1.0013577274636e+00;

  // species 6: OH
  // Computed from Lennard-Jones
  sqrtOneOverTc[6] = 9.7460146948181e-02;
  sqrtAsti[6] = 3.9981183985545e+04;
  Bi[6] = 6.2963299076575e-01;
  Fomega[6] = 4.8508000000000e-01;

  // species 7: H2O2
  // Computed from Lennard-Jones
  sqrtOneOverTc[7] = 8.4114315417892e-02;
  sqrtAsti[7] = 3.2660380884830e+04;
  Bi[7] = 6.2594239211036e-01;
  Fomega[7] = 4.8508000000000e-01;

  // species 8: HO2
  // Computed from Lennard-Jones
  sqrtOneOverTc[8] = 8.4114315417892e-02;
  sqrtAsti[8] = 3.3657825710980e+04;
  Bi[8] = 6.4505861132042e-01;
  Fomega[8] = 4.8508000000000e-01;

  // species 9: CO
  // Imported from NIST
  sqrtOneOverTc[9] = 8.6759935530451e-02;
  sqrtAsti[9] = 4.3619324712923e+04;
  Bi[9] = 9.7786303823350e-01;
  Fomega[9] = 5.5459948367500e-01;

  // species 10: CO2
  // Imported from NIST
  sqrtOneOverTc[10] = 5.7342616962522e-02;
  sqrtAsti[10] = 4.3745610142919e+04;
  Bi[10] = 6.7506167841745e-01;
  Fomega[10] = 8.2653709187500e-01;

  // species 11: CH4
  // Imported from NIST
  sqrtOneOverTc[11] = 7.2440948474968e-02;
  sqrtAsti[11] = 9.5215046085708e+04;
  Bi[11] = 1.8605203320660e+00;
  Fomega[11] = 5.0213035482700e-01;

  // species 12: CH3
  // Computed from Lennard-Jones
  sqrtOneOverTc[12] = 7.2642504557751e-02;
  sqrtAsti[12] = 9.8558148656101e+04;
  Bi[12] = 1.8791630452967e+00;
  Fomega[12] = 4.8508000000000e-01;

  // species 13: CH2
  // Computed from Lennard-Jones
  sqrtOneOverTc[13] = 7.2642504557751e-02;
  sqrtAsti[13] = 1.0564067619908e+05;
  Bi[13] = 2.0142023516102e+00;
  Fomega[13] = 4.8508000000000e-01;

  // species 14: CH2(S)
  // Computed from Lennard-Jones
  sqrtOneOverTc[14] = 7.2642504557751e-02;
  sqrtAsti[14] = 1.0564067619908e+05;
  Bi[14] = 2.0142023516102e+00;
  Fomega[14] = 4.8508000000000e-01;

  // species 15: C
  // Computed from Lennard-Jones
  sqrtOneOverTc[15] = 1.0316275924140e-01;
  sqrtAsti[15] = 7.0240119914876e+04;
  Bi[15] = 1.5377647194458e+00;
  Fomega[15] = 4.8508000000000e-01;

  // species 16: CH
  // Computed from Lennard-Jones
  sqrtOneOverTc[16] = 9.7460146948181e-02;
  sqrtAsti[16] = 5.2228281438065e+04;
  Bi[16] = 8.2250313188056e-01;
  Fomega[16] = 4.8508000000000e-01;

  // species 17: CH3OH
  // Imported from NIST
  sqrtOneOverTc[17] = 4.4166578333598e-02;
  sqrtAsti[17] = 9.6654647783180e+04;
  Bi[17] = 1.4233811698894e+00;
  Fomega[17] = 1.3133918400750e+00;

  // species 18: CH3O
  // Computed from Lennard-Jones
  sqrtOneOverTc[18] = 4.2687825036676e-02;
  sqrtAsti[18] = 7.7751566794340e+04;
  Bi[18] = 8.3360142526260e-01;
  Fomega[18] = 4.8508000000000e-01;

  // species 19: CH2OH
  // Computed from Lennard-Jones
  sqrtOneOverTc[19] = 4.2687825036676e-02;
  sqrtAsti[19] = 7.7751566794340e+04;
  Bi[19] = 8.3360142526260e-01;
  Fomega[19] = 4.8508000000000e-01;

  // species 20: CH2O
  // Computed from Lennard-Jones
  sqrtOneOverTc[20] = 3.9062261583604e-02;
  sqrtAsti[20] = 8.4274889632487e+04;
  Bi[20] = 7.9341968505345e-01;
  Fomega[20] = 4.8508000000000e-01;

  // species 21: HCO
  // Computed from Lennard-Jones
  sqrtOneOverTc[21] = 3.9062261583604e-02;
  sqrtAsti[21] = 8.7202351509582e+04;
  Bi[21] = 8.2098075206475e-01;
  Fomega[21] = 4.8508000000000e-01;

  // species 22: CHO+
  // Computed from Lennard-Jones
  sqrtOneOverTc[22] = 9.7460146948181e-02;
  sqrtAsti[22] = 2.3432795189290e+04;
  Bi[22] = 3.6902511247211e-01;
  Fomega[22] = 4.8508000000000e-01;

  // species 23: H3O+
  // Computed from Lennard-Jones
  sqrtOneOverTc[23] = 8.4588202634765e-02;
  sqrtAsti[23] = 5.0489484185304e+04;
  Bi[23] = 8.4602322126222e-01;
  Fomega[23] = 4.8508000000000e-01;

  // species 24: O2-
  // Computed from Lennard-Jones
  sqrtOneOverTc[24] = 7.4611492866627e-02;
  sqrtAsti[24] = 3.6986371266605e+04;
  Bi[24] = 5.9418204849386e-01;
  Fomega[24] = 4.8508000000000e-01;

  // species 25: E
  // Computed from Lennard-Jones
  sqrtOneOverTc[25] = 2.9899408206870e-02;
  sqrtAsti[25] = 7.7623333318521e+12;
  Bi[25] = 7.2051661156636e+10;
  Fomega[25] = 4.8508000000000e-01;
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
