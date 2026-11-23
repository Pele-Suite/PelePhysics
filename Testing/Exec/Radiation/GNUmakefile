# Builds one radiation test, tst$(TEST).cpp, into tst$(TEST)2d.<...>.ex:
# RadProps (PlanckMean.H, SpectralModels.H), RadCoupling (PeleLMRad.H) and
# the P1 solver tests. The EB tests and POneMulti are 3D only. As in
# CMakeLists.txt, the tests need AMReX and Source/Radiation only.
TEST ?= POneSingle

# AMReX
DIM          = 2
PRECISION    = DOUBLE
PROFILE      = FALSE
VERBOSE      = FALSE
DEBUG        = FALSE
TINY_PROFILE = FALSE

# Compiler
COMP       = gnu
USE_MPI    = FALSE
USE_OMP    = FALSE
USE_CUDA   = FALSE
USE_HIP    = FALSE
BL_NO_FORT = TRUE
CXXSTD     = c++20

# define the location of the PELE_PHYSICS top directory
PELE_PHYSICS_HOME ?= ../../..
AMREX_HOME        ?= $(abspath $(PELE_PHYSICS_HOME)/Submodules/amrex)

EBASE = tst$(TEST)
include $(AMREX_HOME)/Tools/GNUMake/Make.defs

Bpack := ./Make.package $(PELE_PHYSICS_HOME)/Source/Radiation/Make.package
Blocs := . $(PELE_PHYSICS_HOME)/Source/Radiation $(PELE_PHYSICS_HOME)/Testing/Exec
ifeq ($(USE_AMREX_LIB), TRUE)
  AMREX_LIBRARY_HOME ?= $(AMREX_HOME)
  LIBDIR := $(AMREX_LIBRARY_HOME)/lib
  INCDIR := $(AMREX_LIBRARY_HOME)/include
  COMPILE_CPP_FLAGS ?= $(shell awk '/Cflags:/ {$$1=$$2=""; print $$0}' $(LIBDIR)/pkgconfig/amrex.pc)
  COMPILE_LIB_FLAGS ?= $(shell awk '/Libs:/ {$$1=$$2=""; print $$0}' $(LIBDIR)/pkgconfig/amrex.pc)
  CXXFLAGS += -I$(INCDIR) $(COMPILE_CPP_FLAGS)
  LIBRARIES += -L$(LIBDIR) $(COMPILE_LIB_FLAGS)
  # An AMReX library built with SUNDIALS (as for the CI job's other suites)
  # needs it at link time
  ifeq ($(USE_SUNDIALS), TRUE)
    include $(PELE_PHYSICS_HOME)/ThirdParty/Make.ThirdParty
  endif
else
  Pdirs := Base Boundary AmrCore LinearSolvers
  Bpack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
  Blocs += $(foreach dir, Base Boundary AmrCore LinearSolvers/MLMG, $(AMREX_HOME)/Src/$(dir))
endif

include $(Bpack)
INCLUDE_LOCATIONS += $(Blocs)
VPATH_LOCATIONS   += $(Blocs)

all: $(executable)
	@echo SUCCESS

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
