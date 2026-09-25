COMP            ?= gcc
DEBUG           ?= FALSE
USE_CUDA        ?= FALSE
PELE_USE_KLU    ?= FALSE
PELE_USE_MAGMA  ?= FALSE
CUDA_ARCH       ?= 70
AMD_ARCH        ?= gfx90a

SS_VERSION=5.4.0
MAGMA_VERSION=2.9.0

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

lowercase_hcomp = $(shell echo $(COMP) | tr A-Z a-z)
ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),gcc gnu g++ mpicc mpicxx))
  ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),mpi))
    CCOMPILER = $(shell $(COMP) --show | awk '{print $$1}')
    CXXCOMPILER = $(shell $(COMP) --show | awk '{print $$1}')
  else
    CCOMPILER = gcc
    CXXCOMPILER = g++
  endif
else
  ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),intel-classic))
    CCOMPILER = icc
    CXXCOMPILER = icpc
  else
    ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),pgi))
      CCOMPILER = pgicc
      CXXCOMPILER = pgiCC
    else
      ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),llvm clang clang++))
        CCOMPILER = clang
        CXXCOMPILER = clang++
      else
        ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),hip))
           CCOMPILER = hipcc
           CXXCOMPILER = hipcc
        else
          ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),sycl))
             CCOMPILER = icx
             CXXCOMPILER = icpx
	  else
            ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),intel-llvm))
               CCOMPILER = icx
               CXXCOMPILER = icpx
	    else
              ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),intel))
                $(error for intel, specify COMP as intel-classic or intel-llvm)
              else
                ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),cray))
                   CCOMPILER = craycc
                   CXXCOMPILER = craycxx
                else
                  ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),nvhpc))
                    CCOMPILER = nvc
                    CXXCOMPILER = nvc++
                  else
                    $(error Unknown COMP setting)
                  endif
                endif
              endif
	    endif
	  endif
        endif
      endif
    endif
  endif
endif

ifeq ($(USE_HIP),TRUE)
  ifeq ($(which_computer),$(filter $(which_computer),frontier crusher))
    ifeq ($(findstring 5.1, $(ROCM_PATH)), 5.1)
      $(error Incompatible rocm version. Use >= 5.2)
    endif
  endif
endif

tp_suffix=$(COMP)$(PrecisionSuffix)$(DebugSuffix)$(CUDASuffix)$(HIPSuffix)$(SYCLSuffix)$(KLUSuffix)$(MAGMASuffix)
INSTALL_PREFIX=$(PWD)/INSTALL/$(tp_suffix)

ifeq ($(PELE_USE_KLU),TRUE)
  KLUSuffix   := .KLU
  KLU_DEFINES=-DENABLE_KLU:BOOL=ON -DKLU_INCLUDE_DIR=${INSTALL_PREFIX}/include -DKLU_LIBRARY_DIR=${INSTALL_PREFIX}/lib
  SS_BUILD_PREFIX=$(PWD)/BUILD/SUITESPARSE

  SS_DIST_DIR=$(SS_BUILD_PREFIX)/dist
  SS_SOURCE_DIR=$(SS_BUILD_PREFIX)/SuiteSparse-$(SS_VERSION)
  SS_BUILD_DIR=$(SS_BUILD_PREFIX)/build/$(tp_suffix)

  ONE_SS_SRC_FILE=$(SS_SOURCE_DIR)/LICENSE.txt
  ONE_SS_LIB_FILE=$(INSTALL_PREFIX)/libamd.so
  SS_DIST_FILE=v$(SS_VERSION).tar.gz
else
  KLUSuffix=
  KLU_DEFINES=-DENABLE_KLU:BOOL=OFF
  ONE_SS_SRC_FILE=
  ONE_SS_LIB_FILE=
endif

ifeq ($(PELE_USE_MAGMA),TRUE)
  MAGMASuffix   := .MAGMA
  MAGMA_BUILD_PREFIX=$(PWD)/BUILD/MAGMA
  MAGMA_DEFINES=-DENABLE_MAGMA:BOOL=ON -DMAGMA_DIR=${INSTALL_PREFIX}/ -DMAGMA_WORKS=TRUE
ifeq ($(USE_CUDA),TRUE)
  MAGMA_DEFINES+=-DSUNDIALS_MAGMA_BACKENDS:STRING=CUDA
  ifeq ($(which_computer),perlmutter)
    MAGMA_DEFINES+=-DCMAKE_PREFIX_PATH="${CUDATOOLKIT_HOME}/../../math_libs/"
  endif
endif
ifeq ($(USE_HIP),TRUE)
  MAGMA_DEFINES+=-DSUNDIALS_MAGMA_BACKENDS:STRING=HIP
  ifeq ($(which_computer),$(filter $(which_computer),frontier crusher))
    MAGMA_DEFINES+=-DCMAKE_C_FLAGS="-I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64"
    MAGMA_DEFINES+=-DCMAKE_CXX_FLAGS="-I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64"
  endif
endif

  MAGMA_DIST_DIR=$(MAGMA_BUILD_PREFIX)/dist
  MAGMA_SOURCE_DIR=$(MAGMA_BUILD_PREFIX)/magma-$(MAGMA_VERSION)
  MAGMA_BUILD_DIR=$(MAGMA_BUILD_PREFIX)/build/$(tp_suffix)

  ONE_MAGMA_SRC_FILE=$(MAGMA_SOURCE_DIR)/COPYRIGHT
  ONE_MAGMA_LIB_FILE=$(INSTALL_PREFIX)/libmagma.so
  MAGMA_DIST_FILE=magma-$(MAGMA_VERSION).tar.gz
  # Fixes for newer toolchains, applied once after extraction (if present)
  MAGMA_PATCH_FILE=$(wildcard $(PWD)/patches/magma-$(MAGMA_VERSION).patch)

ifeq ($(USE_CUDA),TRUE)
  MAGMA_COMPILER_OPT=-DMAGMA_ENABLE_CUDA=ON -DGPU_TARGET=sm_$(CUDA_ARCH) -DCMAKE_CUDA_COMPILER:STRING=nvcc -DCMAKE_CUDA_HOST_COMPILER:STRING=$(CXXCOMPILER)
  ifeq ($(which_computer),perlmutter)
    MAGMA_COMPILER_OPT+=-DCMAKE_PREFIX_PATH="$(CUDATOOLKIT_HOME)/../../math_libs/"
  endif
  ifeq ($(which_computer),kestrel)
    MAGMA_COMPILER_OPT+=-DLAPACK_LIBRARIES=${CRAY_PE_LIBSCI_PREFIX_DIR}/lib/libsci_gnu.so
  endif
endif
ifeq ($(USE_HIP),TRUE)
  MAGMA_COMPILER_OPT=-DMAGMA_ENABLE_HIP:BOOL=ON -DCMAKE_HIP_ARCHITECTURES:STRING=$(AMD_ARCH) -DAMDGPU_TARGETS:STRING=$(AMD_ARCH) -DGPU_TARGETS:STRING=$(AMD_ARCH) -DGPU_TARGET:STRING=$(AMD_ARCH)
  ifeq ($(lowercase_hcomp),$(filter $(lowercase_hcomp),hip))
    MAGMA_COMPILER_OPT+=-DLAPACK_LIBRARIES=${CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_amd.so
  else
    MAGMA_COMPILER_OPT+=-DLAPACK_LIBRARIES=${CRAY_LIBSCI_PREFIX_DIR}/lib/libsci_cray.so
  endif
  ifeq ($(which_computer),$(filter $(which_computer),frontier crusher))
    MAGMA_COMPILER_OPT+=-DCMAKE_C_FLAGS="-I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64"
    MAGMA_COMPILER_OPT+=-DCMAKE_CXX_FLAGS="-I${ROCM_PATH}/include -L${ROCM_PATH}/lib -lamdhip64"
  endif
endif
else
  MAGMASuffix=
  MAGMA_DEFINES=-DENABLE_MAGMA:BOOL=OFF
  ONE_MAGMA_SRC_FILE=
  ONE_MAGMA_LIB_FILE=
endif

ifeq ($(PRECISION),FLOAT)
  SUNDIALS_PRECISION=SINGLE
  PrecisionSuffix=.SINGLE
else
  SUNDIALS_PRECISION=DOUBLE
  PrecisionSuffix=
endif

ifeq ($(DEBUG),FALSE)
  BUILD_TYPE=Release
  DebugSuffix=
else
  BUILD_TYPE=Debug
  DebugSuffix=.DEBUG
endif

ifeq ($(USE_CUDA),TRUE)
  CUDA_DEFINES=-DENABLE_CUDA:BOOL=ON -DCMAKE_CUDA_ARCHITECTURES:STRING=$(CUDA_ARCH) -DCMAKE_CUDA_COMPILER:STRING=nvcc -DCMAKE_CUDA_HOST_COMPILER:STRING=$(CXXCOMPILER)
  CUDASuffix=.CUDA
  ifeq ($(which_computer),perlmutter)
    CUDA_DEFINES+=-DCMAKE_PREFIX_PATH="$(CUDATOOLKIT_HOME)/../../math_libs/"
  endif
else
  CUDA_DEFINES=-DENABLE_CUDA:BOOL=OFF
  CUDASuffix=
endif
ifeq ($(USE_HIP),TRUE)
  HIP_DEFINES=-DENABLE_HIP:BOOL=ON -DCMAKE_HIP_ARCHITECTURES:STRING=$(AMD_ARCH) -DAMDGPU_TARGETS:STRING=$(AMD_ARCH) -DGPU_TARGETS:STRING=$(AMD_ARCH)
  HIPSuffix=.HIP
else
  HIP_DEFINES=-DENABLE_HIP:BOOL=OFF
  HIPSuffix=
endif
ifeq ($(USE_SYCL),TRUE)
  SYCL_DEFINES=-DENABLE_SYCL:BOOL=ON -DCMAKE_CXX_FLAGS=-fsycl
  SYCLSuffix=.SYCL
else
  SYCL_DEFINES=-DENABLE_SYCL:BOOL=OFF
  SYCLSuffix=
endif

SUNDIALS_PROFILING ?= OFF
SUNDIALS_BUILD_PREFIX=$(PWD)/BUILD/SUNDIALS
SUNDIALS_BUILD_DIR=$(SUNDIALS_BUILD_PREFIX)/build/$(tp_suffix)
ONE_SUNDIALS_LIB_FILE=$(INSTALL_PREFIX)/lib/libsundials_cvode.a

sundials: $(ONE_SUNDIALS_LIB_FILE)

tp_suffix:
	@echo $(tp_suffix)

$(ONE_SUNDIALS_LIB_FILE): $(ONE_SS_LIB_FILE) $(ONE_MAGMA_LIB_FILE)
	@mkdir -p $(SUNDIALS_BUILD_DIR)
	cd $(SUNDIALS_BUILD_DIR) && \
        cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON \
          -DCMAKE_INSTALL_PREFIX:PATH=$(INSTALL_PREFIX) \
          -DCMAKE_INSTALL_LIBDIR:PATH=$(INSTALL_PREFIX)/lib \
          -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
          -DCMAKE_C_COMPILER:STRING=$(CCOMPILER) \
          -DCMAKE_CXX_COMPILER:STRING=$(CXXCOMPILER) \
          -DCMAKE_C_STANDARD:STRING=99 \
          -DCMAKE_CXX_STANDARD:STRING=17 \
          -DSUNDIALS_PRECISION:STRING=$(SUNDIALS_PRECISION) \
          -DCMAKE_BUILD_TYPE:STRING=$(BUILD_TYPE) \
          -DENABLE_MPI:BOOL=OFF \
          -DENABLE_OPENMP:BOOL=OFF \
          -DEXAMPLES_INSTALL:BOOL=OFF \
          -DEXAMPLES_ENABLE_C:BOOL=OFF \
          -DEXAMPLES_ENABLE_CXX:BOOL=OFF \
          -DEXAMPLES_ENABLE_CUDA:BOOL=OFF \
          -DBUILD_CVODES:BOOL=OFF \
          -DBUILD_IDA:BOOL=OFF \
          -DBUILD_IDAS:BOOL=OFF \
          -DBUILD_KINSOL:BOOL=OFF \
          -DBUILD_SHARED_LIBS:BOOL=ON \
          -DSUNDIALS_BUILD_WITH_PROFILING:BOOL=$(SUNDIALS_PROFILING) \
          -DSUNDIALS_INDEX_SIZE:INT=32 \
          $(CUDA_DEFINES) $(HIP_DEFINES) $(SYCL_DEFINES) \
          $(KLU_DEFINES) $(MAGMA_DEFINES) \
          $(SUNDIALS_HOME) && \
        cmake --build $(SUNDIALS_BUILD_DIR) --parallel 16 && \
        cmake --install $(SUNDIALS_BUILD_DIR)

$(ONE_MAGMA_LIB_FILE): $(ONE_MAGMA_SRC_FILE)
	@echo "Building MAGMA"
	#@cd $(MAGMA_SOURCE_DIR); sed -i '389s/#//' CMakeLists.txt
	#@cd $(MAGMA_SOURCE_DIR); sed -i '374s/90c/90c;90a/' CMakeLists.txt
	@echo ${CCOMPILER}
	@echo ${CXXCOMPILER}
	@mkdir -p $(MAGMA_BUILD_DIR)
	@cmake --version
	cd $(MAGMA_BUILD_DIR) && \
        cmake \
          -DCMAKE_BUILD_TYPE:STRING=Release \
          -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} \
          -DCMAKE_C_COMPILER:STRING=$(CCOMPILER) \
          -DCMAKE_CXX_COMPILER:STRING=$(CXXCOMPILER) \
          -DUSE_FORTRAN:BOOL=OFF \
          -DBUILD_SHARED_LIBS:BOOL=ON \
          $(MAGMA_COMPILER_OPT) \
          $(MAGMA_SOURCE_DIR) && \
        cmake --build $(MAGMA_BUILD_DIR) --parallel 10 && \
        cmake --install $(MAGMA_BUILD_DIR)

$(ONE_MAGMA_SRC_FILE): $(MAGMA_DIST_DIR)/$(MAGMA_DIST_FILE) $(MAGMA_PATCH_FILE)
	@cd $(MAGMA_DIST_DIR)/..; tar xf $(MAGMA_DIST_DIR)/$(MAGMA_DIST_FILE)
	$(if $(MAGMA_PATCH_FILE),cd $(MAGMA_SOURCE_DIR) && patch -p1 < $(MAGMA_PATCH_FILE))
	@find $(MAGMA_SOURCE_DIR) -type f -exec touch {} +

$(MAGMA_DIST_DIR)/$(MAGMA_DIST_FILE):
	@mkdir -p $(MAGMA_DIST_DIR)
	@cd $(MAGMA_DIST_DIR); wget https://icl.utk.edu/projectsfiles/magma/downloads/$(MAGMA_DIST_FILE)

$(ONE_SS_LIB_FILE): $(ONE_SS_SRC_FILE)
	@cd $(SS_SOURCE_DIR)/SuiteSparse_config; $(MAKE) install BLAS=-lblas INSTALL=$(INSTALL_PREFIX)
	@cd $(SS_SOURCE_DIR)/AMD; $(MAKE) install BLAS=-lblas INSTALL=$(INSTALL_PREFIX)
	@cd $(SS_SOURCE_DIR)/COLAMD; $(MAKE) install BLAS=-lblas INSTALL=$(INSTALL_PREFIX)
	@cd $(SS_SOURCE_DIR)/BTF; $(MAKE) install BLAS=-lblas INSTALL=$(INSTALL_PREFIX)
	@cd $(SS_SOURCE_DIR)/metis-5.1.0; $(MAKE) config BLAS=-lblas prefix=$(INSTALL_PREFIX) shared=1; $(MAKE) install
	@cd $(SS_SOURCE_DIR)/CCOLAMD; $(MAKE) install BLAS=-lblas INSTALL=$(INSTALL_PREFIX)
	@cd $(SS_SOURCE_DIR)/CAMD; $(MAKE) install BLAS=-lblas INSTALL=$(INSTALL_PREFIX)
	@cd $(SS_SOURCE_DIR)/KLU; $(MAKE) install BLAS=-lblas INSTALL=$(INSTALL_PREFIX)
	# For some reason, the static libraries are built but not installed. Add
	# a workaround here for now, but this should be fixed in SuiteSparse.
	@install -m 644 $(SS_SOURCE_DIR)/SuiteSparse_config/libsuitesparseconfig.a $(INSTALL_PREFIX)/lib
	@install -m 644 $(SS_SOURCE_DIR)/*/Lib/*.a $(INSTALL_PREFIX)/lib

$(ONE_SS_SRC_FILE): $(SS_DIST_DIR)/$(SS_DIST_FILE)
	@cd $(SS_DIST_DIR)/..; tar zxf $(SS_DIST_DIR)/$(SS_DIST_FILE); find $(SS_SOURCE_DIR) -type f -exec touch {} +

$(SS_DIST_DIR)/$(SS_DIST_FILE):
	@mkdir -p $(SS_DIST_DIR)
	@cd $(SS_DIST_DIR); wget https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/$(SS_DIST_FILE)

ifeq ($(PELE_USE_KLU),TRUE)
  clean_SS:
	rm -rf $(SS_BUILD_PREFIX)
else
  clean_SS:
endif

ifeq ($(PELE_USE_MAGMA),TRUE)
  clean_MAGMA:
	rm -rf $(MAGMA_BUILD_PREFIX)
else
  clean_MAGMA:
endif

clean: clean_SS clean_MAGMA
	rm -rf $(INSTALL_PREFIX) $(SUNDIALS_BUILD_PREFIX)

realclean:
	rm -rf $(PWD)/INSTALL $(PWD)/BUILD
#-----------------------------------------------------------------------------
# for debugging.  To see the value of a Makefile variable,
# e.g. libraries, simply do "make print-libraries".  This will
# print out the value.
print-%:
	@echo $* is '$($*)'
	@echo '    origin = $(origin $*)'
	@echo '     value = $(value  $*)'
