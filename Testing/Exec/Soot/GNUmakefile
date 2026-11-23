# AMReX
DIM          = 2
PRECISION    = DOUBLE
PROFILE      = FALSE
VERBOSE      = FALSE
DEBUG        = FALSE

# Compiler
COMP       = gnu
FCOMP      = gfortran
USE_MPI    = FALSE
USE_OMP    = FALSE
USE_CUDA   = FALSE
USE_HIP    = FALSE

# PelePhysics
Eos_Model       = Fuego
Chemistry_Model = SootReaction
Transport_Model = Constant
TINY_PROFILE    = FALSE

# Soot (PeleC conventions: CGS, primitive state with mass fractions)
NUM_SOOT_MOMENTS = 3
DEFINES += -DPELE_USE_SOOT -DNUM_SOOT_MOMENTS=$(NUM_SOOT_MOMENTS)

# define the location of the PELE_PHYSICS top directory
PELE_PHYSICS_HOME    ?= ../../..

Bpack   := ./Make.package $(PELE_PHYSICS_HOME)/Source/Soot/Make.package
Blocs   := . $(PELE_PHYSICS_HOME)/Source/Soot $(PELE_PHYSICS_HOME)/Testing/Exec

include $(PELE_PHYSICS_HOME)/Testing/Exec/Make.PelePhysics
