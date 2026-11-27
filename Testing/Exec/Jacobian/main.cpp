#include <iostream>
#include <vector>

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>

#ifdef AMREX_USE_GPU
#include <AMReX_SUNMemory.H>
#endif

#include "mechanism.H"

#include <PelePhysics.H>
#include <ReactorBase.H>
#include <random>

void
GotoNextLine(std::istream& is)
{
  constexpr std::streamsize bl_ignore_max{100000};
  is.ignore(bl_ignore_max, '\n');
}

void
perturb_sc(
  amrex::Real* sc1,
  amrex::Real* sc2,
  amrex::Real* sc,
  amrex::Real perturb,
  int index)
{
  for (int i = 0; i < NUM_SPECIES; ++i) {
    sc1[i] = sc[i];
    sc2[i] = sc[i];
  }
  sc1[index] = sc1[index] + perturb;
  sc2[index] = sc2[index] - perturb;
}
void
perturb_T(amrex::Real* T1, amrex::Real* T2, amrex::Real T, amrex::Real perturb)
{
  *T1 = T + perturb;
  *T2 = T - perturb;
}
amrex::Real
frobenius(amrex::Real* J, size_t size)
{
  amrex::Real norm = 0;
  for (size_t i = 0; i < size; ++i) {
    norm += J[i] * J[i];
  }
  return norm;
}

int
main()
{

  int NUM_JAC_ENTRIES = (NUM_SPECIES + 1) * (NUM_SPECIES + 1);
  amrex::Real T = 1000.0;
  amrex::Real* sc = new amrex::Real[NUM_SPECIES];
  amrex::Real* sc_pert1 = new amrex::Real[NUM_SPECIES];
  amrex::Real* sc_pert2 = new amrex::Real[NUM_SPECIES];
  amrex::Real* wdot_pert1 = new amrex::Real[NUM_SPECIES];
  amrex::Real* wdot_pert2 = new amrex::Real[NUM_SPECIES];
  amrex::Real* J_analytical = new amrex::Real[NUM_JAC_ENTRIES];
  amrex::Real* J_finite_diff = new amrex::Real[NUM_JAC_ENTRIES];
  amrex::Real pertMag = 1e-4;
  amrex::Real* error = new amrex::Real[NUM_JAC_ENTRIES];
  int index_sc;
  int index_wdot;
  int index_J;
  const int consP = 1;

  // Set sc randomly
  // Reproducibility
  srand(42);
  for (int i = 0; i < NUM_SPECIES; ++i) {
    // dummy numbers
    sc[i] = ((double)rand() / (RAND_MAX)) + 1e-4; //(i+1)*0.02;
  }

  // Compute analytically
  aJacobian(J_analytical, sc, T, consP);

  // Compute dwdot/dsc by finite difference
  for (int i = 0; i < NUM_SPECIES; ++i) {
    for (int j = 0; j < NUM_SPECIES; ++j) {
      index_sc = i;
      index_wdot = j;
      index_J = index_sc * (NUM_SPECIES + 1) + index_wdot;
      perturb_sc(sc_pert1, sc_pert2, sc, pertMag, index_sc);
      productionRate(wdot_pert1, sc_pert1, T);
      productionRate(wdot_pert2, sc_pert2, T);
      J_finite_diff[index_J] =
        (wdot_pert1[index_wdot] - wdot_pert2[index_wdot]) / (2.0 * pertMag);
      error[index_J] = std::abs(J_finite_diff[index_J] - J_analytical[index_J]);
    }
  }

  // Check error
  amrex::Real norm_error = frobenius(error, NUM_JAC_ENTRIES);
  amrex::Real norm_J = frobenius(J_finite_diff, NUM_JAC_ENTRIES);

  return (norm_error / norm_J > 1e-12) ? 1 : 0;
}
