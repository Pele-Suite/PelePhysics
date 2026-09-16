AMREX_HOME   ?= ../../Submodules/amrex
PELE_PHYSICS_HOME ?= ../..

# Local
EBASE = PeleTurb

# AMReX
DIM             = 3
DEBUG           = FALSE
PRECISION       = DOUBLE
TINY_PROFILE    = FALSE

# Compilation
COMP            = gnu
USE_MPI         = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

Bpack   += ./Make.package

Pdirs   := Base Boundary Extern/amrdata AmrCore
Bpack   += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)
Blocs   += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir))

PPdirs  := Utility/PltFileManager Utility/MeshMap
Bpack   += $(foreach dir, $(PPdirs), $(PELE_PHYSICS_HOME)/Source/$(dir)/Make.package)
Blocs   += $(foreach dir, $(PPdirs), $(PELE_PHYSICS_HOME)/Source/$(dir))

include $(Bpack)
INCLUDE_LOCATIONS += $(Blocs)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
