#include "ReactorCvodePreconditioner.H"

namespace pele::physics::reactions::cvode {
#ifdef AMREX_USE_GPU
#ifdef AMREX_USE_CUDA
int
Precond(
  amrex::Real /*tn*/,
  N_Vector u,
  N_Vector /*fu*/,
  sunbooleantype jok,
  sunbooleantype* jcurPtr,
  amrex::Real gamma,
  void* user_data)
{
  BL_PROFILE("Pele::ReactorCvode::Precond()");

  amrex::Real* u_d = N_VGetDeviceArrayPointer_Cuda(u);

  CVODEUserData* udata = static_cast<CVODEUserData*>(user_data);
  udata->gamma = gamma;

  // Get data out of udata to minimize global memory access
  auto ncells = udata->ncells;
  auto stream = udata->stream;
  auto nbThreads = udata->nbThreads;
  auto nbBlocks = udata->nbBlocks;
  auto csr_val_d = udata->csr_val_d;
  // auto csr_row_count_d = udata->csr_row_count_d;
  // auto csr_col_index_d = udata->csr_col_index_d;
  auto NNZ = udata->NNZ;
  auto react_type = udata->reactor_type;

  BL_PROFILE_VAR("Pele::ReactorCvode::fKernelComputeAJ()", fKernelComputeAJ);
  AMREX_ALWAYS_ASSERT(nbThreads == CVODE_NB_THREADS);
  if (jok) {
    const auto ec = amrex::Gpu::ExecutionConfig(ncells);
    amrex::launch_global<CVODE_NB_THREADS>
      <<<nbBlocks, CVODE_NB_THREADS, ec.sharedMem, stream>>>(
        [=] AMREX_GPU_DEVICE() noexcept {
          for (int icell = blockDim.x * blockIdx.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;
               icell < ncells; icell += stride) {
            fKernelComputeAJsys(icell, NNZ, gamma, user_data, u_d, csr_val_d);
          }
        });
    *jcurPtr = SUNFALSE;
  } else {
    const auto ec = amrex::Gpu::ExecutionConfig(ncells);
    amrex::launch_global<CVODE_NB_THREADS>
      <<<nbBlocks, CVODE_NB_THREADS, ec.sharedMem, stream>>>(
        [=] AMREX_GPU_DEVICE() noexcept {
          for (int icell = blockDim.x * blockIdx.x + threadIdx.x,
                   stride = blockDim.x * gridDim.x;
               icell < ncells; icell += stride) {
            fKernelComputeallAJ(
              icell, NNZ, react_type, gamma, user_data, u_d, csr_val_d);
          }
        });
    *jcurPtr = SUNTRUE;
  }
  cudaError_t cuda_status = cudaStreamSynchronize(stream);
  AMREX_ASSERT(cuda_status == cudaSuccess);
  BL_PROFILE_VAR_STOP(fKernelComputeAJ);

  /*
  Somehow this function crash when called again after Psolve
  but is it really necessary ? The code solver behaves well ...
  BL_PROFILE_VAR("InfoBatched(inPrecond)", InfoBatched);
  size_t workspaceInBytes = 0;
  size_t internalDataInBytes = 0;
  cusolverStatus_t cuS_st = CUSOLVER_STATUS_SUCCESS;
  cuS_st = cusolverSpDcsrqrBufferInfoBatched(udata->cusolverHandle,
                                             NUM_SPECIES + 1, NUM_SPECIES + 1,
                                             NNZ, udata->descrA, csr_val_d,
                                             csr_row_count_d, csr_col_index_d,
                                             ncells, udata->info,
                                             &internalDataInBytes,
                                             &workspaceInBytes);
  Print() << " BufferInfo workspaceInBytes " << workspaceInBytes << "\n";
  AMREX_ASSERT(cuS_st == CUSOLVER_STATUS_SUCCESS);
  */

  cuda_status = cudaDeviceSynchronize();
  AMREX_ASSERT(cuda_status == cudaSuccess);
  return (0);
}

int
PSolve(
  amrex::Real /*tn*/,
  N_Vector /*u*/,
  N_Vector /*fu*/,
  N_Vector r,
  N_Vector z,
  amrex::Real /*gamma*/,
  amrex::Real /*delta*/,
  int /*lr*/,
  void* user_data)
{
  BL_PROFILE("Pele::ReactorCvode::PSolve()");

  // Get data out of udata to minimize global memory access
  CVODEUserData* udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells;
  auto csr_val_d = udata->csr_val_d;
  auto csr_row_count_d = udata->csr_row_count_d;
  auto csr_col_index_d = udata->csr_col_index_d;
  auto NNZ = udata->NNZ;

  amrex::Real* z_d = N_VGetDeviceArrayPointer_Cuda(z);
  amrex::Real* r_d = N_VGetDeviceArrayPointer_Cuda(r);

  cusolverStatus_t cuS_st = cusolverSpDcsrqrsvBatched(
    udata->cusolverHandle, NUM_SPECIES + 1, NUM_SPECIES + 1, NNZ, udata->descrA,
    csr_val_d, csr_row_count_d, csr_col_index_d, r_d, z_d, ncells, udata->info,
    udata->buffer_qr);
  AMREX_ASSERT(cuS_st == CUSOLVER_STATUS_SUCCESS);

  cudaError_t cuda_status = cudaDeviceSynchronize();
  AMREX_ASSERT(cuda_status == cudaSuccess);

  N_VCopyFromDevice_Cuda(z);
  N_VCopyFromDevice_Cuda(r);

  return (0);
}

#else

int
Precond(
  amrex::Real /* tn */,
  N_Vector /* u */,
  N_Vector /* fu */,
  sunbooleantype /* jok */,
  sunbooleantype* /* jcurPtr */,
  amrex::Real /* gamma */,
  void* /* user_data */)
{
  amrex::Abort("Only implemented for CUDA.");
  return 1;
}

int
PSolve(
  amrex::Real /* tn */,
  N_Vector /* u */,
  N_Vector /* fu */,
  N_Vector /* r */,
  N_Vector /* z */,
  amrex::Real /* gamma */,
  amrex::Real /* delta */,
  int /* lr */,
  void* /* user_data */)
{
  amrex::Abort("Only implemented for CUDA.");
  return 1;
}

#endif

#else

// Preconditioner setup routine for GMRES solver when no sparse mode is
// activated Generate and preprocess P
int
Precond(
  amrex::Real /* tn */,
  N_Vector u,
  N_Vector /* fu */,
  sunbooleantype jok,
  sunbooleantype* jcurPtr,
  amrex::Real gamma,
  void* user_data)
{
  BL_PROFILE("Pele::ReactorCvode::Precond()");
  // Make local copies of pointers to input data
  amrex::Real* u_d = N_VGetArrayPointer(u);

  // Make local copies of pointers in user_data
  auto* udata = static_cast<CVODEUserData*>(user_data);
  auto reactor_type = udata->reactor_type;
  auto* P = udata->P;
  auto* Jbd = udata->Jbd;
  auto* pivot = udata->pivot;

  // MW CGS
  amrex::Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  if (jok != 0) {
    // jok = SUNTRUE: Copy Jbd to P
    SUNDlsMat_denseCopy(Jbd[0][0], P[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1);
    *jcurPtr = SUNFALSE;
  } else {
    // rho MKS
    amrex::Real rho = 0.0;
    for (int i = 0; i < NUM_SPECIES; i++) {
      rho = rho + u_d[i];
    }
    // Yks
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    amrex::Real rhoinv = 1.0 / rho;
    for (int i = 0; i < NUM_SPECIES; i++) {
      massfrac[i] = u_d[i] * rhoinv;
    }
    amrex::Real temp = u_d[NUM_SPECIES];
    // Activities
    amrex::Real activity[NUM_SPECIES] = {0.0};
    auto eos = pele::physics::PhysicsType::eos();
    eos.RTY2C(rho, temp, massfrac, activity);
    int consP = static_cast<int>(reactor_type == ReactorTypes::h_reactor_type);
    amrex::Real Jmat[(NUM_SPECIES + 1) * (NUM_SPECIES + 1)] = {0.0};
    DWDOT_SIMPLIFIED(Jmat, activity, &temp, &consP);

    // Scale Jacobian.  Load into P.
    SUNDlsMat_denseScale(0.0, Jbd[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1);
    for (int i = 0; i < NUM_SPECIES; i++) {
      for (int k = 0; k < NUM_SPECIES; k++) {
        (Jbd[0][0])[k][i] = Jmat[k * (NUM_SPECIES + 1) + i] * mw[i] / mw[k];
      }
      (Jbd[0][0])[i][NUM_SPECIES] =
        Jmat[i * (NUM_SPECIES + 1) + NUM_SPECIES] / mw[i];
    }
    for (int i = 0; i < NUM_SPECIES; i++) {
      (Jbd[0][0])[NUM_SPECIES][i] =
        Jmat[NUM_SPECIES * (NUM_SPECIES + 1) + i] * mw[i];
    }
    (Jbd[0][0])[NUM_SPECIES][NUM_SPECIES] =
      Jmat[(NUM_SPECIES + 1) * (NUM_SPECIES + 1) - 1];

    SUNDlsMat_denseCopy(Jbd[0][0], P[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1);

    *jcurPtr = SUNTRUE;
  }

  // Scale by -gamma
  SUNDlsMat_denseScale(-gamma, P[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1);

  // Add identity matrix and do LU decompositions on blocks in place.
  SUNDlsMat_denseAddIdentity(P[0][0], NUM_SPECIES + 1);
  sunindextype ierr = SUNDlsMat_denseGETRF(
    P[0][0], NUM_SPECIES + 1, NUM_SPECIES + 1, pivot[0][0]);
  if (ierr != 0) {
    return (1);
  }

  return (0);
}

int
PSolve(
  amrex::Real /* tn */,
  N_Vector /* u */,
  N_Vector /* fu */,
  N_Vector r,
  N_Vector z,
  amrex::Real /* gamma */,
  amrex::Real /* delta */,
  int /* lr */,
  void* user_data)
{
  BL_PROFILE("Pele::ReactorCvode::PSolve()");
  // Make local copies of pointers to input data
  amrex::Real* zdata = N_VGetArrayPointer(z);

  // Extract the P and pivot arrays from user_data.
  auto* udata = static_cast<CVODEUserData*>(user_data);
  auto* P = udata->P;
  auto* pivot = udata->pivot;

  N_VScale(1.0, r, z);

  // Solve the block-diagonal system Pz = r using LU factors stored
  //   in P and pivot data in pivot, and return the solution in z.
  amrex::Real* v = zdata;
  SUNDlsMat_denseGETRS(P[0][0], NUM_SPECIES + 1, pivot[0][0], v);

  return (0);
}

#ifdef PELE_USE_KLU
// Preconditioner setup routine for GMRES solver when KLU sparse mode is
// activated Generate and preprocess P
int
Precond_sparse(
  amrex::Real /* tn */,
  N_Vector u,
  N_Vector /* fu */,
  sunbooleantype jok,
  sunbooleantype* jcurPtr,
  amrex::Real gamma,
  void* user_data)
{
  BL_PROFILE("Pele::ReactorCvode::Precond_sparse()");
  // Make local copies of pointers to input data (big M)
  amrex::Real* u_d = N_VGetArrayPointer(u);

  // Make local copies of pointers in user_data
  CVODEUserData* udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells;
  auto reactor_type = udata->reactor_type;
  auto JSPSmat = udata->JSPSmat;
  auto colPtrs = udata->colPtrs;
  auto rowVals = udata->rowVals;
  auto Jdata = udata->Jdata;
  auto Symbolic = udata->Symbolic;
  auto Numeric = udata->Numeric;
  auto Common = udata->Common;
  auto FirstTimePrecond = udata->FirstTimePrecond;

  // MW CGS
  // MW CGS
  amrex::Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  // Check if Jac is stale
  if (jok) {
    // jok = SUNTRUE: Copy Jbd to P
    *jcurPtr = SUNFALSE;
  } else {
    // Save Jac from cell to cell if more than one
    amrex::Real temp_save_lcl = 0.0;
    for (int tid = 0; tid < ncells; tid++) {
      // Offset in case several cells
      int offset = tid * (NUM_SPECIES + 1);
      // rho MKS
      amrex::Real rho = 0.0;
      for (int i = 0; i < NUM_SPECIES; i++) {
        rho = rho + u_d[offset + i];
      }
      // Yks
      amrex::Real rhoinv = 1.0 / rho;
      amrex::Real massfrac[NUM_SPECIES] = {0.0};
      for (int i = 0; i < NUM_SPECIES; i++) {
        massfrac[i] = u_d[offset + i] * rhoinv;
      }
      amrex::Real temp = u_d[offset + NUM_SPECIES];
      // Activities
      amrex::Real activity[NUM_SPECIES] = {0.0};
      auto eos = pele::physics::PhysicsType::eos();
      eos.RTY2C(rho, temp, massfrac, activity);

      // Do we recompute Jac ?
      if (fabs(temp - temp_save_lcl) > 1.0) {
        // Formalism
        int consP = reactor_type == ReactorTypes::h_reactor_type;
        DWDOT_SIMPLIFIED(JSPSmat[tid], activity, &temp, &consP);

        for (int i = 0; i < NUM_SPECIES; i++) {
          for (int k = 0; k < NUM_SPECIES; k++) {
            (JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] *= mw[i] / mw[k];
          }
          (JSPSmat[tid])[i * (NUM_SPECIES + 1) + NUM_SPECIES] /= mw[i];
        }
        for (int i = 0; i < NUM_SPECIES; i++) {
          (JSPSmat[tid])[NUM_SPECIES * (NUM_SPECIES + 1) + i] *= mw[i];
        }
        temp_save_lcl = temp;
      } else {
        // if not: copy the one from prev cell
        for (int i = 0; i < NUM_SPECIES + 1; i++) {
          for (int k = 0; k < NUM_SPECIES + 1; k++) {
            (JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] =
              (JSPSmat[tid - 1])[k * (NUM_SPECIES + 1) + i];
          }
        }
      }
    }

    *jcurPtr = SUNTRUE;
  }

  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    // nb non zeros elem should be the same for all cells
    int nbVals = colPtrs[0][i] - colPtrs[0][i - 1];
    for (int j = 0; j < nbVals; j++) {
      // row of non zero elem should be the same for all cells
      int idx = rowVals[0][colPtrs[0][i - 1] + j];
      // Scale by -gamma
      // Add identity matrix
      for (int tid = 0; tid < ncells; tid++) {
        if (idx == (i - 1)) {
          Jdata[tid][colPtrs[tid][i - 1] + j] =
            1.0 - gamma * (JSPSmat[tid])[idx * (NUM_SPECIES + 1) + idx];
        } else {
          Jdata[tid][colPtrs[tid][i - 1] + j] =
            -gamma * (JSPSmat[tid])[(i - 1) * (NUM_SPECIES + 1) + idx];
        }
      }
    }
  }

  BL_PROFILE_VAR("Pele::ReactorCvode::KLU_factorization", KLU_factor);
  if (!(FirstTimePrecond)) {
    for (int tid = 0; tid < ncells; tid++) {
      klu_refactor(
        colPtrs[tid], rowVals[tid], Jdata[tid], Symbolic[tid], Numeric[tid],
        &(Common[tid]));
    }
  } else {
    for (int tid = 0; tid < ncells; tid++) {
      Numeric[tid] = klu_factor(
        colPtrs[tid], rowVals[tid], Jdata[tid], Symbolic[tid], &(Common[tid]));
    }
    udata->FirstTimePrecond = false;
  }
  BL_PROFILE_VAR_STOP(KLU_factor);

  return (0);
}

int
PSolve_sparse(
  amrex::Real /* tn */,
  N_Vector /* u */,
  N_Vector /* fu */,
  N_Vector r,
  N_Vector z,
  amrex::Real /* gamma */,
  amrex::Real /* delta */,
  int /* lr */,
  void* user_data)
{
  BL_PROFILE("Pele::ReactorCvode::PSolve_sparse()");
  // Make local copies of pointers in user_data
  CVODEUserData* udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells;
  auto Symbolic = udata->Symbolic;
  auto Numeric = udata->Numeric;
  auto Common = udata->Common;

  // Make local copies of pointers to input data (big M)
  amrex::Real* zdata = N_VGetArrayPointer(z);

  BL_PROFILE_VAR("Pele::ReactorCvode::KLU_inversion", PSolve_sparse);
  N_VScale(1.0, r, z);

  // Solve the block-diagonal system Pz = r using LU factors stored
  //   in P and pivot data in pivot, and return the solution in z.
  amrex::Real zdata_cell[NUM_SPECIES + 1];
  for (int tid = 0; tid < ncells; tid++) {
    int offset_beg = tid * (NUM_SPECIES + 1);
    std::memcpy(
      zdata_cell, zdata + offset_beg, (NUM_SPECIES + 1) * sizeof(amrex::Real));
    klu_solve(
      Symbolic[tid], Numeric[tid], NUM_SPECIES + 1, 1, zdata_cell,
      &(Common[tid]));
    std::memcpy(
      zdata + offset_beg, zdata_cell, (NUM_SPECIES + 1) * sizeof(amrex::Real));
  }
  BL_PROFILE_VAR_STOP(PSolve_sparse);

  return (0);
}
#endif

// Preconditioner setup routine for GMRES solver when custom sparse mode is
// activated Generate and preprocess P
int
Precond_custom(
  amrex::Real /* tn */,
  N_Vector u,
  N_Vector /* fu */,
  sunbooleantype jok,
  sunbooleantype* jcurPtr,
  amrex::Real gamma,
  void* user_data)
{
  BL_PROFILE("Pele::ReactorCvode::Precond_custom()");
  // Make local copies of pointers to input data
  amrex::Real* u_d = N_VGetArrayPointer(u);

  // Make local copies of pointers in user_data
  auto* udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells;
  auto reactor_type = udata->reactor_type;
  auto* JSPSmat = udata->JSPSmat;
  auto* rowPtrs = udata->rowPtrs;
  auto* colVals = udata->colVals;
  auto* Jdata = udata->Jdata;

  // MW CGS
  amrex::Real mw[NUM_SPECIES] = {0.0};
  get_mw(mw);

  // Check if Jac is stale
  if (jok != 0) {
    // jok = SUNTRUE: Copy Jbd to P
    *jcurPtr = SUNFALSE;
  } else {
    // Save Jac from cell to cell if more than one
    amrex::Real temp_save_lcl = 0.0;
    for (int tid = 0; tid < ncells; tid++) {
      // Offset in case several cells
      int offset = tid * (NUM_SPECIES + 1);
      // rho MKS
      amrex::Real rho = 0.0;
      for (int i = 0; i < NUM_SPECIES; i++) {
        rho = rho + u_d[offset + i];
      }
      // Yks
      amrex::Real massfrac[NUM_SPECIES] = {0.0};
      amrex::Real rhoinv = 1.0 / rho;
      for (int i = 0; i < NUM_SPECIES; i++) {
        massfrac[i] = u_d[offset + i] * rhoinv;
      }
      amrex::Real temp = u_d[offset + NUM_SPECIES];
      // Activities
      amrex::Real activity[NUM_SPECIES] = {0.0};
      auto eos = pele::physics::PhysicsType::eos();
      eos.RTY2C(rho, temp, massfrac, activity);

      // Do we recompute Jac ?
      if (fabs(temp - temp_save_lcl) > 1.0) {
        // Formalism
        int consP =
          static_cast<int>(reactor_type == ReactorTypes::h_reactor_type);
        DWDOT_SIMPLIFIED(JSPSmat[tid], activity, &temp, &consP);

        for (int i = 0; i < NUM_SPECIES; i++) {
          for (int k = 0; k < NUM_SPECIES; k++) {
            (JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] *= mw[i] / mw[k];
          }
          (JSPSmat[tid])[i * (NUM_SPECIES + 1) + NUM_SPECIES] /= mw[i];
        }
        for (int i = 0; i < NUM_SPECIES; i++) {
          (JSPSmat[tid])[NUM_SPECIES * (NUM_SPECIES + 1) + i] *= mw[i];
        }
        temp_save_lcl = temp;
      } else {
        // if not: copy the one from prev cell
        for (int i = 0; i < NUM_SPECIES + 1; i++) {
          for (int k = 0; k < NUM_SPECIES + 1; k++) {
            (JSPSmat[tid])[k * (NUM_SPECIES + 1) + i] =
              (JSPSmat[tid - 1])[k * (NUM_SPECIES + 1) + i];
          }
        }
      }
    }
    *jcurPtr = SUNTRUE;
  }

  for (int i = 1; i < NUM_SPECIES + 2; i++) {
    // nb non zeros elem should be the same for all cells
    int nbVals = rowPtrs[0][i] - rowPtrs[0][i - 1];
    for (int j = 0; j < nbVals; j++) {
      // row of non zero elem should be the same for all cells
      int idx = colVals[0][rowPtrs[0][i - 1] + j];
      // Scale by -gamma
      // Add identity matrix
      for (int tid = 0; tid < ncells; tid++) {
        if (idx == (i - 1)) {
          Jdata[tid][rowPtrs[tid][i - 1] + j] =
            1.0 - gamma * (JSPSmat[tid])[idx * (NUM_SPECIES + 1) + idx];
        } else {
          Jdata[tid][rowPtrs[tid][i - 1] + j] =
            -gamma * (JSPSmat[tid])[(i - 1) + (NUM_SPECIES + 1) * idx];
        }
      }
    }
  }

  return (0);
}

int
PSolve_custom(
  amrex::Real /* tn */,
  N_Vector /* u */,
  N_Vector /* fu */,
  N_Vector r,
  N_Vector z,
  amrex::Real /* gamma */,
  amrex::Real /* delta */,
  int /* lr */,
  void* user_data)
{
  BL_PROFILE("Pele::ReactorCvode::PSolve_custom()");
  // Make local copies of pointers in user_data
  auto* udata = static_cast<CVODEUserData*>(user_data);
  auto ncells = udata->ncells;
  auto* Jdata = udata->Jdata;

  // Make local copies of pointers to input data
  amrex::Real* zdata = N_VGetArrayPointer(z);
  amrex::Real* rdata = N_VGetArrayPointer(r);

  N_VScale(1.0, r, z);

  // Solve the block-diagonal system Pz = r using LU factors stored
  // in P and pivot data in pivot, and return the solution in z.
  BL_PROFILE_VAR("Pele::ReactorCvode::GaussSolver", GaussSolver);
  for (int tid = 0; tid < ncells; tid++) {
    int offset = tid * (NUM_SPECIES + 1);
    amrex::Real* z_d_offset = zdata + offset;
    amrex::Real* r_d_offset = rdata + offset;
    sgjsolve_simplified(Jdata[tid], z_d_offset, r_d_offset);
  }
  BL_PROFILE_VAR_STOP(GaussSolver);

  return (0);
}

#endif
} // namespace pele::physics::reactions::cvode
