#ifndef REACTORUTILSCVODE_H
#define REACTORUTILSCVODE_H
#include "ReactorTypes.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <cvode/cvode.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include "ReactorUtils.H"

#ifdef AMREX_USE_CUDA
#include <cusolverSp.h>
#endif

#ifdef PELE_USE_KLU
#include "klu.h"
#include <sunlinsol/sunlinsol_klu.h>
#endif

#ifdef AMREX_USE_GPU
#if defined(AMREX_USE_CUDA)
#define CVODE_NB_THREADS 32
#elif defined(AMREX_USE_HIP)
#define CVODE_NB_THREADS 64
#else
#define CVODE_NB_THREADS 32
#endif
#endif

namespace pele::physics::reactions {

struct CVODEUserData
{
  amrex::Real dt_save;     // Internal cvode dt holder
  int ncells;              // Number of cells in solve
  int verbose;             // Verbose
  int maxOrder;            // CVODE max order
  int reactor_type;        // Either HP (LM) or UV (C)
  int analytical_jacobian; // Analytical Jacobian 'On' flag
  int solve_type;          // Type of linear solve for Newton direction
  int precond_type;        // Type of preconditioner (if iterative solve)
  int NNZ; // Number of non-zero entry per cell for sparse representations
  amrex::Real* rhoe_init = nullptr; // Initial energy (rhoE for C, rhoH for LM)
  amrex::Real* rhoesrc_ext = nullptr; // External energy forcing
  amrex::Real* rYsrc_ext = nullptr;   // External species forcing
  int* mask =
    nullptr; // Masking tagging cells where integration should not be performed
  int* FCunt = nullptr; // Number of RHS evaluations (not used on GPU)
  amrex::Real gamma;    // System Chem. jacobian coefficient
  int nbBlocks;         // GPU kernel launch parameter
  int nbThreads;        // GPU kernel launch parameter

#ifdef AMREX_USE_GPU
  // Matrix data
  int* csr_row_count_h;   // Host CSR sparse mat
  int* csr_col_index_h;   // Host CSR sparse mat
  int* csr_row_count_d;   // Device CSR sparse mat
  int* csr_col_index_d;   // Device CSR sparse mat
  amrex::Real* csr_val_d; // Syst. Jac. CSR sparse mat values
  amrex::Real* csr_jac_d; // Chem. Jac. CSR sparse mat values

#ifdef AMREX_USE_CUDA
  // Batched QR preconditioner data
  void* buffer_qr = NULL;    // cuSolver internal work arrays
  csrqrInfo_t info;          // QR solve info
  cusparseMatDescr_t descrA; // Sparse mat. descriptor
  cusolverSpHandle_t cusolverHandle;
  cusparseHandle_t cuSPHandle;
#endif
  amrex::gpuStream_t stream;

#else
  bool FirstTimePrecond;
  amrex::Real** Jdata = nullptr;
  // Sparse direct solve data
  // Ptrs to CSR/CSC matrix for each submatrices (cells)
  SUNMatrix* PS;
  int** rowVals = nullptr;
  int** rowPtrs = nullptr;
  int** colPtrs = nullptr;
  int** colVals = nullptr;
  int* indx = nullptr;
  amrex::Real** JSPSmat = nullptr;
#ifdef PELE_USE_KLU
  klu_common* Common;
  klu_symbolic** Symbolic;
  klu_numeric** Numeric;
#endif
  // Matrix data for denseSimpleAjac preconditioner
  amrex::Real**** Jbd;
  amrex::Real**** P;
  sunindextype*** pivot;
  // Custom direct linear solve: a CSR SUNMatrix and ptrs
  SUNMatrix PSc;
  int* colVals_c;
  int* rowPtrs_c;
#endif
};

namespace cvode {

#ifdef AMREX_USE_GPU
enum linSolveType {
  fixedPoint,
  sparseDirect,
  customDirect,
  magmaDirect,
  GMRES,
  precGMRES
};
enum precondType { sparseSimpleAJac };
#else
enum linSolveType {
  fixedPoint,
  denseFDDirect,
  denseDirect,
  sparseDirect,
  customDirect,
  GMRES,
  precGMRES,
  hackDumpSparsePattern
};
enum precondType { denseSimpleAJac, sparseSimpleAJac, customSimpleAJac };
#endif

// Error function for CVODE
void cvodeErrHandler(
  int error_code,
  const char* /*module*/,
  const char* /*function*/,
  char* msg,
  void* /*eh_data*/);

#ifdef AMREX_USE_GPU
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeallAJ(
  int ncell,
  int NNZ,
  int reactor_type,
  amrex::Real gamma,
  void* user_data,
  amrex::Real* u_d,
  amrex::Real* csr_val_arg)
{
  CVODEUserData* udata = static_cast<CVODEUserData*>(user_data);
  int neqs = NUM_SPECIES + 1;
  auto csr_jac_d = udata->csr_jac_d;
  auto csr_row_count_d = udata->csr_row_count_d;
  auto csr_col_index_d = udata->csr_col_index_d;

  amrex::GpuArray<amrex::Real, NUM_SPECIES> activity{0.0};
  amrex::GpuArray<amrex::Real, (NUM_SPECIES + 1) * (NUM_SPECIES + 1)> Jmat_pt{
    0.0};

  int u_offset = ncell * neqs;
  amrex::Real* u_curr = u_d + u_offset;

  amrex::Real rho_pt = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rho_pt = rho_pt + u_curr[n];
  }

  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac{0.0};
  for (int i = 0; i < NUM_SPECIES; i++) {
    massfrac[i] = u_curr[i] / rho_pt;
  }

  amrex::Real temp_pt = u_curr[NUM_SPECIES];

  auto eos = pele::physics::PhysicsType::eos();
  eos.RTY2C(rho_pt, temp_pt, massfrac.arr, activity.arr);

  int consP = reactor_type == ReactorTypes::h_reactor_type;
  DWDOT_SIMPLIFIED(Jmat_pt.arr, activity.arr, &temp_pt, &consP);

  int jac_offset = ncell * NNZ;
  amrex::Real* csr_jac_cell = csr_jac_d + jac_offset;
  amrex::Real* csr_val_cell = csr_val_arg + jac_offset;
  for (int i = 0; i < NUM_SPECIES; i++) {
    for (int k = 0; k < NUM_SPECIES; k++) {
      Jmat_pt[k * neqs + i] *= mw(i) * imw(k);
    }
    Jmat_pt[i * neqs + NUM_SPECIES] *= imw(i);
    Jmat_pt[NUM_SPECIES * neqs + i] *= mw(i);
  }
  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    int nbVals = csr_row_count_d[i] - csr_row_count_d[i - 1];
    for (int j = 0; j < nbVals; j++) {
      int idx = csr_col_index_d[csr_row_count_d[i - 1] + j - 1] - 1;
      if (idx == (i - 1)) {
        csr_val_cell[csr_row_count_d[i - 1] + j - 1] =
          1.0 - gamma * Jmat_pt[idx * neqs + idx];
        csr_jac_cell[csr_row_count_d[i - 1] + j - 1] =
          Jmat_pt[idx * neqs + idx];
      } else {
        csr_val_cell[csr_row_count_d[i - 1] + j - 1] =
          -gamma * Jmat_pt[idx * neqs + i - 1];
        csr_jac_cell[csr_row_count_d[i - 1] + j - 1] =
          Jmat_pt[idx * neqs + i - 1];
      }
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeAJsys(
  int ncell,
  int NNZ,
  amrex::Real gamma,
  void* user_data,
  amrex::Real* u_d,
  amrex::Real* csr_val_arg)
{
  CVODEUserData* udata = static_cast<CVODEUserData*>(user_data);
  auto csr_jac_d = udata->csr_jac_d;
  auto csr_row_count_d = udata->csr_row_count_d;
  auto csr_col_index_d = udata->csr_col_index_d;

  int jac_offset = ncell * NNZ;
  amrex::Real* csr_jac_cell = csr_jac_d + jac_offset;
  amrex::Real* csr_val_cell = csr_val_arg + jac_offset;

  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    int nbVals = csr_row_count_d[i] - csr_row_count_d[i - 1];
    for (int j = 0; j < nbVals; j++) {
      int idx = csr_col_index_d[csr_row_count_d[i - 1] + j - 1] - 1;
      if (idx == (i - 1)) {
        csr_val_cell[csr_row_count_d[i - 1] + j - 1] =
          1.0 - gamma * csr_jac_cell[csr_row_count_d[i - 1] + j - 1];
      } else {
        csr_val_cell[csr_row_count_d[i - 1] + j - 1] =
          -gamma * csr_jac_cell[csr_row_count_d[i - 1] + j - 1];
      }
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelComputeAJchem(
  int icell,
  int NNZ,
  int reactor_type,
  int* csr_row_count_d,
  int* csr_col_index_d,
  amrex::Real* u_d,
  amrex::Real* Jdata)
{
  const int neqs = NUM_SPECIES + 1;
  int u_offset = icell * neqs;
  amrex::Real* u_curr = u_d + u_offset;

  amrex::Real rho_pt = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rho_pt = rho_pt + u_curr[n];
  }

  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {0.0};
  for (int n = 0; n < NUM_SPECIES; n++) {
    massfrac[n] = u_curr[n] / rho_pt;
  }
  amrex::Real temp_pt = u_curr[NUM_SPECIES];

  const int consP = reactor_type == ReactorTypes::h_reactor_type;
  amrex::GpuArray<amrex::Real, neqs * neqs> Jmat_pt = {0.0};
  auto eos = pele::physics::PhysicsType::eos();
  eos.RTY2JAC(rho_pt, temp_pt, massfrac.arr, Jmat_pt.arr, consP);

  // Scale Jacobian
  for (int i = 0; i < NUM_SPECIES; i++) {
    for (int k = 0; k < NUM_SPECIES; k++) {
      Jmat_pt[k * neqs + i] *= mw(i) * imw(k);
    }
    Jmat_pt[i * neqs + NUM_SPECIES] *= imw(i);
    Jmat_pt[NUM_SPECIES * neqs + i] *= mw(i);
  }

  // Fill the sparse outgoing Jacobian matrix
  int jac_offset = icell * NNZ;
  amrex::Real* Jcurr = Jdata + jac_offset;

  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    int nbVals = csr_row_count_d[i] - csr_row_count_d[i - 1];
    for (int j = 0; j < nbVals; j++) {
      int idx_cell = csr_col_index_d[csr_row_count_d[i - 1] + j];
      Jcurr[csr_row_count_d[i - 1] + j] = Jmat_pt[idx_cell * neqs + i - 1];
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
fKernelDenseAJchem(
  int icell, int reactor_type, amrex::Real* u_d, amrex::Real* Jdata)
{
  const int neqs = NUM_SPECIES + 1;

  int u_offset = icell * neqs;
  amrex::Real* u_curr = u_d + u_offset;

  amrex::Real rho_pt = 0.0;
  for (int n = 0; n < NUM_SPECIES; n++) {
    rho_pt = rho_pt + u_curr[n];
  }

  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {0.0};
  for (int n = 0; n < NUM_SPECIES; n++) {
    massfrac[n] = u_curr[n] / rho_pt;
  }
  amrex::Real temp_pt = u_curr[NUM_SPECIES];

  const int consP = reactor_type == ReactorTypes::h_reactor_type;
  amrex::GpuArray<amrex::Real, neqs * neqs> Jmat_pt = {0.0};
  auto eos = pele::physics::PhysicsType::eos();
  eos.RTY2JAC(rho_pt, temp_pt, massfrac.arr, Jmat_pt.arr, consP);

  // Scale Jacobian and pass into outgoing data ptr.
  int jac_offset = icell * (neqs * neqs);
  amrex::Real* Jcurr = Jdata + jac_offset;

  for (int i = 0; i < NUM_SPECIES; i++) {
    for (int k = 0; k < NUM_SPECIES; k++) {
      Jcurr[k * neqs + i] = Jmat_pt[k * neqs + i] * mw(i) * imw(k);
    }
    Jcurr[i * neqs + NUM_SPECIES] = Jmat_pt[i * neqs + NUM_SPECIES] * imw(i);
    Jcurr[NUM_SPECIES * neqs + i] = Jmat_pt[NUM_SPECIES * neqs + i] * mw(i);
  }
}

#endif
} // namespace cvode
} // namespace pele::physics::reactions
#endif
