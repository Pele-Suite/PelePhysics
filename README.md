# PelePhysics

[![AMReX Badge](https://img.shields.io/static/v1?label=%22powered%20by%22&message=%22AMReX%22&color=%22blue%22)](https://amrex-codes.github.io/amrex/)
[![Exascale Computing Project](https://img.shields.io/badge/supported%20by-ECP-blue)](https://www.exascaleproject.org/research-project/combustion-pele/)

## Overview

PelePhysics is a repository of physics databases and implementation code for use with the [Pele suite of codes](https://Pele-Suite.github.io),
primarily the compressible solver [PeleC](https://Pele-Suite.github.io/PeleC/) and low-Mach solver [PeleLMeX](https://Pele-Suite.github.io/PeleLMeX/).

PelePhysics contains C++ source code for the following physics and other modules, in the `Source` directory:

* *EOS*: Equation of State, include multi-species ideal and real gas options
* *Transport*: Routines to evaluate multi-species transport properties
* *Reactions*: Routines for substepped integration of stiff chemical systems, including with CVODE
* *Spray*: Lagrangian spray droplet library, formerly part of [PeleMP](https://github.com/Pele-Suite/PeleMP)
* *Soot*: An implementation of the Hybrid Method of Moments soot model, formerly part of [PeleMP](https://github.com/Pele-Suite/PeleMP)
* *Radiation*: Radiative heat transfer model based on the spherical harmonics method, formerly [PeleRad](https://github.com/Pele-Suite/PeleRad)
* *Utility*: Several handy modules for data management across other Pele codes

Additionally, PelePhysics contains a variety of stand-alone tools to aid in Pele simulation workflows, found in the `Support` directory.
Most notable among these is CEPTR, described further below, a tool that converts chemical mechanisms from Cantera format to efficient
source code files for use in the Pele codes.

For convenience, the `Mechanisms` directory contains the CEPTR-generated source code for a number of chemical mechanisms, as well as both
Cantera and CHEMKIN format data files for each mechanism.

Documentation for these various capabilities, to varying degrees, can be found in the `Docs` directory or online
at https://Pele-Suite.github.io/PelePhysics/. The papers listed below under the Citation heading may also be valuable resources.

PelePhysics has two required dependencies, AMReX and SUNDIALS, which are both git submodules located the `Submodules` directory.
The `ThirdParty` directory is used when building these submodules, and for downloading and building other optional submodules.

Finally, the `Testing` directory includes several stand-alone codes for exercising various aspects of the physics modules mentioned above.

## Getting Started

PelePhysics is primarily intended as a library for use with other Pele codes. For more information on how to download PelePhysics and
its dependencies, and how to build the stand-alone test codes, consult the [online documentation](https://Pele-Suite.github.io/PelePhysics/GettingStarted.html).

## Acknowledgment

This research was supported by the Exascale Computing Project (ECP), Project
Number: 17-SC-20-SC, a collaborative effort of two DOE organizations -- the
Office of Science and the National Nuclear Security Administration --
responsible for the planning and preparation of a capable exascale ecosystem --
including software, applications, hardware, advanced system engineering, and
early testbed platforms -- to support the nation's exascale computing
imperative.

## Citation

To cite PelePhysics as a whole and its use as part of the broader Pele
suite, please use the following [SIAM Parallel
Processing](https://doi.org/10.1137/1.9781611977967.2) and [Combustion
and Flame](https://doi.org/10.1016/j.combustflame.2024.113740)
articles:

```
@article{PeleSoftware,
  author = {Marc T. {Henry de Frahan} and Lucas Esclapez and Jon Rood and Nicholas T. Wimer and Paul Mullowney and Bruce A. Perry and Landon Owen and Hariswaran Sitaraman and Shashank Yellapantula and Malik Hassanaly and Mohammad J. Rahimi and Michael J. Martin and Olga A. Doronina and Sreejith N. A. and Martin Rieth and Wenjun Ge and Ramanan Sankaran and Ann S. Almgren and Weiqun Zhang and John B. Bell and Ray Grout and Marc S. Day and Jacqueline H. Chen},
  title = {The Pele Simulation Suite for Reacting Flows at Exascale},
  booktitle = {Proceedings of the 2024 SIAM Conference on Parallel Processing for Scientific Computing},
  journal = {Proceedings of the 2024 SIAM Conference on Parallel Processing for Scientific Computing},
  chapter = {},
  pages = {13-25},
  doi = {10.1137/1.9781611977967.2},
  URL = {https://epubs.siam.org/doi/abs/10.1137/1.9781611977967.2},
  eprint = {https://epubs.siam.org/doi/pdf/10.1137/1.9781611977967.2},
  year = {2024},
  publisher = {Proceedings of the 2024 SIAM Conference on Parallel Processing for Scientific Computing}
}

@article{Hassanaly2024,
  author = {Malik Hassanaly and Nicholas T. Wimer and Anne Felden and Lucas Esclapez and Julia Ream and Marc T. {Henry de Frahan} and Jon Rood and Marc Day},
  title = {Symbolic construction of the chemical Jacobian of quasi-steady state (QSS) chemistries for Exascale computing platforms},
  journal = {Combustion and Flame},
  volume = {270},
  pages = {113740},
  year = {2024},
  issn = {0010-2180},
  doi = {https://doi.org/10.1016/j.combustflame.2024.113740},
  url = {https://www.sciencedirect.com/science/article/pii/S0010218024004498},
}
```

To cite the multi-physics (soot, spray, radiation) capabilities in PelePhysics that were imported from
PeleMP and PeleRad, please use the following [Journal of Fluids Engineering article](https://doi.org/10.1115/1.4064494):
```
@article{owen2023pelemp,
  title={PeleMP: The Multiphysics Solver for the Combustion Pele Adaptive Mesh Refinement Code Suite},
  author={Owen, Landon D and Ge, Wenjun and Rieth, Martin and Arienti, Marco and Esclapez, Lucas and S Soriano, Bruno and Mueller, Michael E and Day, Marc and Sankaran, Ramanan and Chen, Jacqueline H},
  journal={Journal of Fluids Engineering},
  pages={1--41},
  year={2023}
}
```

A full list of publications documenting the development of the Pele suite and its
application to various reacting flow and other simulations is available on the main
[Pele suite page](https://Pele-Suite.github.io/pubs.html). After publication,
if you'd like your work to be included on that list, you can request to have it added
[here](https://github.com/Pele-Suite/Pele-Suite.github.io/discussions/3).

## Getting help

Do you have a question ? Found an issue ? Please use the [GitHub Discussions](https://github.com/Pele-Suite/PelePhysics/discussions) to engage
with the development team or open a new [GitHub issue](https://github.com/Pele-Suite/PelePhysics/issues) to report a bug. The development team
also encourages users to take an active role in respectfully answering each other's questions in these spaces. When reporting a bug, it is helpful
to provide as much detail as possible, including a case description and the major compile and runtime options being used. Though not required,
it is most effective to create a fork of this repository and share a branch of that fork with a case that minimally reproduces the error.

New contributions to *PelePhysics* are welcome ! Contributing Guidelines are provided in [CONTRIBUTING.md](CONTRIBUTING.md).

## Versioning

PelePhysics now uses uses a type of semantic versioning to help users navigate different versions of the code,
which are labeled with [GitHub tags](https://github.com/Pele-Suite/PelePhysics/releases/). These tagged versions are not exhaustive, but they adhere to
the following convention. Given a version number MAJOR.MINOR.PATCH:
1. MAJOR version for changes to key aspects of the solver affecting input/source files for all cases, when a key model is changed to significantly affect results of simulations, when a major new capability is added
2. MINOR version for when a significant feature is added (in a backward compatible manner), accumulation of smaller features, or changes to input file compatibility for less central aspects of the solver (e.g., post-processing) or aspects not affecting all cases
3. PATCH version for backward compatible bug fixes and minor features

PelePhysics previously used YY.MM formatting for versions. These should be interpreted as version 0 subversions,
e.g. v25.04 is equivalent to v0.25.04.

## CEPTR

This `PelePhysics` repository contains the CEPTR source code generation tool in order to support the inclusion of chemical models specified in the Cantera yaml format. CETPR derives from FUEGO, which was originally created by Michael Aivazis at CalTech, and donated to CCSE in 2001.  Originally, FUEGO was part of a larger Python-based set of workflow tools Michael developed for managing and publishing simulation-based studies.  FUEGO itself was developed as a drop-in replacement for the CHEMKIN library, and provided "hand-coded" replacement routines for evaluation of EOS thermodynamic functions that are considerably more efficient than their CHEMKIN counterparts.  Since 2001, CCSE has continued to modify FUEGO independently for its own use so that the current version here bears little resemblance to Michael's original code, or its intentions. CEPTR has adapted FUEGO code to use the Cantera yaml format and is now the preferred way of generating mechanism files. If using CEPTR, please cite the above `Hassanaly2024` journal article.

Typically, Cantera *mechanisms* (combustion models) are transmitted via a set of yaml files, which provide data for
* *Species* - definition of chemical species, as composed by fundamental elements
* *Chemistry* - modified Arrhenius reaction parameters, including third body efficiencies and pressure dependencies
* *Thermodynamics* - NASA-formatted polynomial expressions, assuming a mixture of ideal gases
* *Transport* - physical parameters for each of the chemical species required for the `EGLib` (or comparable) library for computing species- and temperature-dependent molecular transport coefficients

This information is read by the CEPTR tool, and source code files are generated that are to linked directly into an executable in the `Pele` code suite in order to provide data and functions necessary to evaluate chemistry, transport and thermodynamic relationships. `PelePhysics` supports the following in addition to standard Cantera style models:
* *Soave-Redlich-Kwong* real gas equation of state, implemented to be consistent with corresponding ideal gas model at suitable conditions
* *Gamma-law* gas equation of state
* *Constant coefficient* transport
* *Simple* A simplified mixture-averaged transport model consistent with EGLib under suitable conditions
* *Inert* Chemistry setting to ignore reactions
* Mixture-averaged thermal diffusion coefficients are also available using the transport.use_soret flag (see PeleLMeX implementation for more information)
* Utilization of auxiliary source files (C++) that follow the Cantera species production rate function interface (e.g., QSS approximations)

If the `Pele` codes are built with `Eos_Model = Fuego`, the make system variable `Chemistry_Model` must be set to one of the models (subfolders) that exist in the `${PELE_PHYSICS_HOME}/Mechanisms` folder. The repository currently provides (multiple) models for each of the following mixtures:
* air
* hydrogen
* methane
* ethane
* *n*-dodecane
* di-methyl ether
* ... more can be added (see below)


### NOTE: Non-ideal equations of state

`PelePhysics` currently supports a cubic EOS model: `Soave-Redlich-Kwong`.  It is built on top of the ideal gas models, and is selected by specifying its name as the `Eos_Model` during the build (the make system requires that both `Eos_Model` and `Chemistry_Model` be specified).  Any additional parameters required for the EOS (e.g., attractions, repulsions, critical states) are either included in the underlying CEPTR database used to generate the source file model implementation, or else are inferred from the input model data.

### Model generation procedures
This repository provides the tools necessary for generating new Pele-compatible combustion mechanisms. Please refer to the [CEPTR documentation](https://Pele-Suite.github.io/PelePhysics/Ceptr.html) for instructions on generating mechanism models. Make sure that you edit the `GNUmakefile` where you want to use this (in, e.g., `PeleC/Exec`) so that `Chemistry_Model` is `XXX`.  In `PeleC/Exec/Make.PeleC`, the model is expected to be in the folder `${PELE_PHYSICS_HOME}/Mechanisms/$(Chemistry_Model)`, and it is expected that the folder contains a `Make.package` file to include, so make sure things are where they need to be. Refer to other mechanisms for additional guidance.
