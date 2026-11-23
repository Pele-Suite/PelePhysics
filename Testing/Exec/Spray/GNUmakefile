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
Chemistry_Model = heptane_3sp
Transport_Model = Simple
TINY_PROFILE    = FALSE

# Spray (PeleC conventions: CGS units, conservative state)
USE_PARTICLES  = TRUE
SPRAY_FUEL_NUM = 1
DEFINES += -DPELE_USE_SPRAY -DSPRAY_FUEL_NUM=$(SPRAY_FUEL_NUM)

# define the location of the PELE_PHYSICS top directory
PELE_PHYSICS_HOME    ?= ../../..
AMREX_HOME           ?= $(abspath $(PELE_PHYSICS_HOME)/Submodules/amrex)

SPRAY_HOME := $(PELE_PHYSICS_HOME)/Source/Spray
Bpack   := ./Make.package $(SPRAY_HOME)/Make.package \
           $(SPRAY_HOME)/Distribution/Make.package \
           $(SPRAY_HOME)/BreakupSplash/Make.package
Blocs   := . $(SPRAY_HOME) $(SPRAY_HOME)/Distribution $(SPRAY_HOME)/BreakupSplash \
           $(PELE_PHYSICS_HOME)/Testing/Exec
# The AMReX particles, unless they come with the AMReX library
ifneq ($(USE_AMREX_LIB), TRUE)
  Bpack += $(AMREX_HOME)/Src/Particle/Make.package
  Blocs += $(AMREX_HOME)/Src/Particle
endif

include $(PELE_PHYSICS_HOME)/Testing/Exec/Make.PelePhysics
