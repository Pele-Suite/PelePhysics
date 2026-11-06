.. highlight:: rst

.. _sec:subsWD:

************
Introduction
************

Objectives and State-Of-The-Art
===============================

What we will call the `PeleSuite` is currently composed of 3 separate codes:

- `PelePhysics <https://github.com/Pele-Suite/PelePhysics>`_ is a repository of physics databases and implementation code for use within the other `Pele` codes. In particular, the choice of chemistry and transport models as well as associated functions and capabilities are managed in `PelePhysics`.
- `PeleLM <https://github.com/AMReX-Combustion/PeleLM>`_ is an adaptive-mesh Low-Mach number hydrodynamics code for reacting flows. It has a sibling, `PeleLMeX <https://github.com/Pele-Suite/PeleLMeX>`_, that solves for the same type of flow using a subtly different numerical approach.
- `PeleC <https://github.com/Pele-Suite/PeleC>`_ is an adaptive-mesh compressible hydrodynamics code for reacting flows.

All three codes rely on `AMReX <https://amrex-codes.github.io/amrex>`_, which is a software frameworks that provides the data structure and enable massive parallelization.

.. |a| image:: ./Visualization/PeleSuite.png

.. table:: 
   :align: center

   +-----+
   | |a| |
   +-----+



`PelePhysics` (as well as the `ChemDriver` object of `PeleLM` ) used to rely upon DVODE [VODE1989]_ 
to perform the chemistry integration, no matter the problem at hand. 
DVODE is a very robust, but slightly outdated, variable-coefficient Ordinary Differential Equation (ODE) solver written in Fortran 77. 
At its core, it uses a direct dense linear solver. DVODE is very efficient in the resolution of small stiff systems 
of equations but can become prohibitively expensive when dealing with bigger systems of equations, such as those frequently encountered in combustion systems. 

In recent years, the Sundials team at LLNL [LLNL2005]_ has been involved in the active development of a modern, 
C++ version, of DVODE called CVODE. 
CVODE implements the same functionalities as those available in DVODE, but offers much more flexibility through 
its user-friendly set of interface routines. Additionally, other linear solvers are available, 
such as iterative or sparse solvers which can prove to be very efficient in handling "larger" systems of equations.

The objective of this user-guide is to document the CVODE-based chemistry integration implemented in `PelePhysics`. Although it is possible to use CVODE in `PeleC`, the following is mainly intended for `PeleLM` users. This user-guide will cover:

- ODE equations (`reactor type`)
- Default settings (tolerances/order/...)
- Linear solvers available --along with examples of performance
- Setting-up a `PelePhysics` test case
- ...



How to naviguate this documentation
===================================

`This section provides a short overview of the chemistry-related features of PelePhysics. For an in-depth discussion, relevant references to specific sections of this manuscript are given.`


- In `PelePhysics`, the user can select between **two different reactor types**. 
  The first type is a `constant volume` reactor, were the choice of fixed variables are the internal energy and the density, 
  to comply with `PeleC` transported variables. This reproduces what was originally 
  implemented with DVODE in `PelePhysics`. 
  A second reactor type formulated in terms of enthalpy and density has been put in place, to comply with `PeleLM`. 
  Both reactors are available in DVODE (in fortran 90) and CVODE (in cpp).
  See sections :ref:`sec:subsDiffReacts` for additional details, 
  and :ref:`sec:subsPPOptions` to see how to activate one or the other via the input files.

- With both reactors, it is possible to use an **Analytical Jacobian** (depending upon the choice of linear solver, see section :ref:`sec:subslinalg`.)

- Three different types of **linear solvers** are implemented, see section :ref:`sec:subslinalg` for additional details, and :ref:`sec:subsPPOptions` to see how to make a selection:
 
    - a dense direct solver -- with or without Analytical Jacobian
    - a sparse direct solver (requires the KLU library) -- always requires an Analytical Jacobian
    - a couple of sparse iterative solvers from the GMRES family -- preconditioned or not

- **Regression testings** have been put in place in `PelePhysics` to test the CVODE integration. See section :ref:`sec:subsubValidCVreact` for validations, and section :ref:`sec:subsReactEvalCvode` for a step-by-step example.

- A CVODE version running on **GPU** is also technically available but the documentation is a WIP. The interested user can   
  contact code developers for additional information

- `PelePhysics` uses an automatic chemistry-related routine generator:
  **CEPTR**. CEPTR is part of the sources of `PelePhysics`. With
  CEPTR, a unique chemistry file is generated for a specific kinetic
  scheme. Instructions to generate your own chemistry files (all you
  need are chemistry files in the famous `Cantera
  <https://cantera.org>`_ yaml format) are discussed in section
  :ref:`sec:ceptr`.  Note that `PelePhysics` already offers a large
  choice of more than 20 different kinetic schemes.
