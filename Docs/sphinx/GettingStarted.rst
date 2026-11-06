.. highlight:: rst

************************
`PelePhysics` Quickstart
************************

Greetings users. As a word of caution, this documentation is in progress. Some parts of the code remain undocumented,
and some parts of the documentation are out of data. If you are confused by something you read here, or otherwise
need help with `PelePhysics`, the best course of action is to open a Discussion on the `GitHub page <https://github.com/Pele-Suite/PelePhysics/discussions>`_,
so the development team and other users can help.

- If you are a complete beginner, I urge you to carefully read the two following chapters :ref:`sec:GetPP` and :ref:`sec:GetCVODE`, to properly set-up your working environment.
- If you are familiar with `PelePhysics`, have it installed already and would simply like to know which chemistry-related keywords and/or environment variables to set in your various input files to perform a simulation with one of the codes available in the `PeleSuite`, then I invite you to directly skip to section :ref:`sec:subsPPOptions`.
- If you are in a hurry but still would like more context, visit section :ref:`sec:subsWD` to be referred to portions of this document that are of interest to you.

.. _sec:GetPP:

Obtaining `PelePhysics`
=======================

PelePhysics is primarily intended as a library for use in the other `Pele codes <https://Pele-Suite.github.io>`_, and is automatically downloaded as
a submodule of both PeleC and PeleLMeX. However, it can also be used as a stand-alone solver for chemical reactions and thermodynamic properties,
or as a library for other codes. Instructions for how to obtain `PelePhysics` for these purposes are provided here.

First, make sure that "Git" is installed on your machine---we recommend version 1.7.x or higher. Then...

1. Clone the `PelePhysics` repository and its submodules: ::

    git clone --recursive https://github.com/Pele-Suite/PelePhysics.git

  This will create a ``PelePhysics`` directory on your machine. The ``--recursive`` option ensures that the required :ref:`sec:GetCVODE` are also downloaded to the
  ``PelePhysics/Submodules`` directory. Set the environment variable ``PELE_PHYSICS_HOME`` to point to the location of this folder (``export PELE_PHYSICS_HOME=$(pwd)/PelePhysics``)

2. Periodically update the repository and its dependencies by typing ``git pull && git submodule update`` within the repository.


.. _sec:GetCVODE:

Dependencies
============

PelePhysics has two required dependencies: `AMReX <https://amrex-codes.github.io/amrex/>`_ and `SUNDIALS <https://github.com/LLNL/sundials>`_.
These dependencies are shipped with PelePhysics as git submodules and the proper versions are cloned to the ``PelePhysics/Submodules/`` directory automatically when
doing the recursive git clone described in :ref:`sec:GetPP`. Users also have the option to download their own versions of these dependencies elsewhere on their machine, in which case
the paths to these libraries must be specified by exporting ``AMREX_HOME`` and ``SUNDIALS_HOME`` environment variables or defining these variables in
the ``GNUmakefile`` when building.

1. `AMReX` is a library that provides data structures and methods for operating on data in the context of
   block-structured adaptive mesh refinement that are used throughout the Pele suite of codes.

2. `SUNDIALS` is a library of differential and algebraic equation solvers that is used by PelePhysics
   primarily for its CVODE package for implicit solution of stiff ODE systems. CVODE is used for integration of stiff chemical reaction systems in PelePhysics.

PelePhysics has two optional dependencies: `SuiteSparse <http://faculty.cse.tamu.edu/davis/suitesparse.html>`_ and `MAGMA <https://icl.utk.edu/magma/>`_.
These dependencies are not shipped with PelePhysics by default, but the proper versions are automatically downloaded if needed during the building
process (:ref:`sec:BuildingRunning`). Both of these dependencies are libraries that aid in solving the linear systems that arise during implicit integration of a chemical system using CVODE.
There are several other options available for solvers within CVODE, so these libraries are not strictly required, but they may enable options that
lead to better performance on certain computing architectures.

1. `SuiteSparse` is a suite of Sparse Matrix software that is very handy when dealing with big kinetic mechanisms (the Jacobian of which are usually very sparse).
   In such a case, CVODE can make use of the KLU library, which is part of `SuiteSparse`, to perform sparse linear algebra.
   Documentation and further information can be found on `SuiteSparse website <http://faculty.cse.tamu.edu/davis/suitesparse.html>`_.

2. `MAGMA` is a collection of linear algebra libraries for heterogeneous computing.


.. _sec:BuildingRunning:

Building and Running Test Cases
===============================

PelePhysics has several short programs that are used to test its various capabilities, located in the ``Testing/Exec`` directory. For example,
we will consider the `ReactEval` case, which tests the chemical reaction integration capability. ::

  cd ${PELE_PHYSICS_HOME}/Testing/Exec/ReactEval

The ``GNUmakefile`` in this directory specifies several key build options, like the compiler (``COMP``) and whether it will be a debug (``DEBUG``) build.

First, build the necessary dependencies. `SUNDIALS` is always built.
`SuiteSparse` is downloaded and built if ``PELE_USE_KLU = TRUE`` is specified in the ``GNUmakefile``.
`MAGMA` is downloaded and built if ``PELE_USE_MAGMA = TRUE`` is specified in the ``GNUmakefile``.
All dependencies are installed in the ${PELE_PHYSICS_HOME}/ThirdParty directory. For a given set of
compile-time options, this step only needs to be done once, but it needs to be redone whenever compile-time
options are changed ::

  make TPL

Now, build the `ReactEval` executable (the ``-j 4`` option specified to compile in parallel on 4 processors): ::

  make -j 4

To run the program, execute: ::

  ./Pele3d.gnu.ex inputs.3d-regt_GPU

If you need to clean your build, you can run ::

  make TPLrealclean && make realclean
