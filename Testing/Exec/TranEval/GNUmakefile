# AMReX
DIM        = 2
PRECISION  = DOUBLE
PROFILE    = FALSE
VERBOSE    = FALSE
DEBUG      = FALSE

# Compiler
COMP       = gnu
FCOMP      = gfortran
USE_MPI    = FALSE
USE_OMP    = FALSE
USE_CUDA   = FALSE
USE_HIP    = FALSE

# PelePhysics
FUEGO_GAS  = TRUE
TINY_PROFILE = FALSE
# Multicomponent (LCT) conductivity models of Simple transport
USE_LCT_TRANSPORT = TRUE

# define the location of the PELE_PHYSICS top directory
PELE_PHYSICS_HOME    ?= ../../..

# this flag activates the subcycling mode in the D/Cvode routines
DEFINES  += -DMOD_REACTOR

# Activates use of SUNDIALS
ifeq ($(USE_CUDA), TRUE)
  PELE_USE_KLU = FALSE
else
  ifeq ($(USE_HIP), TRUE)
    PELE_USE_KLU = FALSE
  else
    PELE_USE_KLU = FALSE
  endif
endif

ifeq ($(FUEGO_GAS), TRUE)
  Eos_Model       = Fuego
  Chemistry_Model = air
  Transport_Model = Simple
else
  Eos_Model       = GammaLaw
  Chemistry_Model = Null
  Transport_Model = Constant
endif


USE_MANIFOLD = FALSE
ifeq ($(USE_MANIFOLD), TRUE)
  Eos_Model       = Manifold
  Chemistry_Model = Null
  Transport_Model = Manifold
endif
Manifold_Dim = 2

Bpack   := ./Make.package
Blocs   := .

include $(PELE_PHYSICS_HOME)/Testing/Exec/Make.PelePhysics
