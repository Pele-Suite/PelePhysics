.. highlight:: rst

.. _sec:Support:

*******
Support
*******

The main purpose of PelePhysics is to contain a library of routines for shared use by the other Pele codes,
which are contained in the ``Source`` directory. In the ``Support`` directory, PelePhysics also contains
several stand-alone tools and scripts that are used to generate data and source code that support these purposes.
These tools include:

* ``ceptr``: Chemistry Evaluation for Pele Through Recasting, machine generation of mechanism code
* ``TurbFileHIT``: Isotropic turbulence initial conditions and inflows
* ``MechanismPAH``: Append PAH module to chemical mechanism data
* ``liquidProp``: Fits to NIST data for condensed phase species

ceptr
=====

CEPTR is used to generate the C++ mechanism files from Cantera-format mechanism data for all the chemical
mechanisms in the ``Mechanisms/`` directory. For a full description of CEPTR and how to use it, see the
:ref:`CEPTR subsection <sec:ceptr>` of the Chemistry section of this documentation.

.. _sec_turbfile:

TurbFileGenerator
=================

This support code contains two separate pieces: a python script that generates a synthetic 3D isotropic turbulence velocity field and
an AMReX-based C++ code that converts data from a variety of sources into a format that can be used by the PelePhysics
:ref:`TurbInflow Utility <sec_turbinflow>` to provide turbulent inflow boundary conditions in either PeleC or PeleLMeX.

Generating an HIT File
~~~~~~~~~~~~~~~~~~~~~~

The ``gen_hit_ic.py`` python script creates a synthetic isotropic turbulence velocity field based on the methodology described
in the appendix of `Johnsen et al. (2010) J. Comp Phys. <http://dx.doi.org/10.1016/j.jcp.2009.10.028>`_. The resulting velocity
field is stored in files with names following the pattern ``hit_ic_k0_N.dat``, where ``k0`` is the most
energetic wave number and ``N`` is the size of the grid. The initial
condition for a grid of size :math:`N^3` is generated as follows:

1. velocity fluctuations generated on a :math:`N_k^3` grid in wavenumber space (default :math:`N_k = 512`)
2. Coefficients associated to wavenumbers that cannot be represented on the desired grid are set to 0 (sharp wavenumber cutoff)
3. inverse Fourier transform of the velocity fluctuations (:math:`N_k^3` grid)
4. velocity fluctuations resampled on the desired grid (:math:`N^3`)

This script accepts the following options: ::

  ./gen_hit_ic.py --help
  usage: gen_hit_ic.py [-h] [-k0 K0] [-N N] [-Nk NK] [-s SEED] [-p]

  Generate the velocity fluctuations for the HIT IC

  options:
    -h, --help            show this help message and exit
    -k0 K0                Wave number containing highest energy
    -N N                  Resolution
    -Nk NK                Resolution in wavenumber space for intermediate step
    -s SEED, --seed SEED  Random number generator seed
    -p, --plot            Save a plot of the x-velocity

Generating an initial condition file is as easy as: ::

  ./gen_hit_ic.py -N 16

.. Note::

   This script utilizes standard python libraries such as SciPy and Matplotlib. If you'd like to ensure that you are using appropriate
   versions of the dependencies, follow the :ref:`CEPTR instructions <sec_ceptr_software>` for setting up ``poetry`` to  manage
   dependencies, then run the script using ``poetry run -C ../ceptr/ python gen_hit_ic.py -N 16``.

Generating Inflow Files
-----------------------

The ``TurbFileHIT`` directory also contains C++ source code for a small program that reads data from a variety of sources and converts
it into a format interpretable by PelePhysics (planes of data saved as many AMReX FArrayBox in binary format in a `DAT` file with a text
`HDR` file with metadata that allows the binary data to be read properly).

Build options can be specified in the ``GNUmakefile``. Notably, if you did not recursively clone PelePhysics with the AMReX submodule, you
will need to specify a path to AMReX (``AMREX_HOME``). But in most cases, you should be able to directly build the utility simply by running: ::

  make -j

Then run the resulting executable without specifying an inputs to see a full list of options: ::

  ./PeleTurb3d.gnu.ex

This capability currently supports three different data sources, specified with the ``type`` keyword on the command line or in an
input file. The currently supported data types are :

1. ``type=turb_box``, which uses the ``.dat`` files created using the python script above. With this input type, the TurbInflow utility
   uses Taylor's hypothesis to provide temporally varying boundary data by marching through the 3rd dimension of the HIT files at a
   constant velocity. A sample ``input`` file for this type is included and can be used by running ``./PeleTurb3d.gnu.ex input``.
   Adjust the ``hit_file`` and ``input_ncell`` parameters in the input file to match the file name and number of cells from the previous
   step. The ``urms0`` parameter is a scale factor to rescale the velocity fluctuations. The length scale of the turbulence data will be 2pi
   in length and periodic in all directions.
2. ``type=periodic_plt``, which uses a PeleLMeX plot file as the input (``ifile``) that must be periodic in the designated ``normal``
   direction and contains at least the following variables: `x_velocity`, `y_velocity`, `z_velocity`. Similarly to the option above,
   the TurbInflow utility uses Taylor's hypothesis to provide temporally varying boundary data by marching through the normal dimension
   of the plt files at a constant velocity. The data files that get generated are single-level, but the user can specify which level
   of the plt file this data should match with the ``level`` keyword. The user may also specify which directions are periodic in the
   input plt file; the normal direction must be peirodic. For the other directions, ghost cells are populated by first-order extrapolation
   if not periodic. The dimensions of the generated turbulence files will match the plt file used.
3. ``type=diag_frame_plane``, which compiles temporally evolving data from a set of planes specified with the ``ifiles`` keyword, saved as 3D
   AMReX plt files with a single cell in the z-direction. Appropriate files may be saved from a PeleLMeX simulation using
   the :ref:`DiagFramePlane <sec_diagnostics>` capability, specifying the ``dump_flat_3D_plotfile=1`` option for that diagnostic.
   These files must contain at least the following variables: `x_velocity`, `y_velocity`, `z_velocity`. The ``normal`` direction used
   for generating the DiagFramePlanes should be specified.  The user may also specify which directions are periodic in the
   input plt files; ghost cells are populated by first-order extrapolation if not periodic. The dimensions of the generated turbulence files
   will match the plt files used. The valid time range also comes from these plt files. Due to the interpolation stencils used,
   the turbinflow will be valid from the timestamp of the first plt file used (inclusive) through the second last timestamp (exclusive).

For all of the above data types, an ``ofile`` keyword must be specified, which is the name of the directory where the
output files will be saved. After successful execution, the output directory should contain two files: ``HDR`` and ``DAT``.

MechanismPAH
============

This functionality takes an existing set of mechanism, transport, and thermodynamic data files in Chemkin format and attempts to add a PAH module to it. It checks if any species are duplicated by comparing the atomic composition and the enthalpy curves. It check if any reactions are duplicated. If any species or reactions are duplicated, these are skipped. Once the new yaml file is created, ignition delay time at 1 and 20 atm is compared between the original and new mechanism to see what impact the PAH module has on the mechanism. Plot files of these values are created for you to decide if the differences are significant.

Usage
~~~~~

You need to have the NumPy, Cantera, and MatPlotLib python modules. In order to run, use the following ::

     python addPAHmech.py --mech origmech.inp --transport origtrans.dat --thermo origthermo.dat --fuelname NC10H22

where ``origmech.inp``, ``origthermo.dat``, and ``origtrans.dat`` are the initial mechanism, thermodynamic,
and transport files to have the PAH module amended to.

Disclaimer
~~~~~~~~~~

The resulting IDT should be studied to determine if the new mechanism is now compromised with the addition of the PAH module. This is left up to the user's discretion. This has only been tested on a few mechanisms and might have bugs.

liquidProp
==========

This is a python script that reads in an NIST property file for a condensed or saturated phase of a species.
Files for rho, mu, and lambda (thermal conductivity) should be provided in a single directory and
named ``rho.dat``, ``mu.dat``, and ``lambda.dat``. The usage for this script is::

  $ python propcoeff.py -h

  usage: propcoeff.py [-h] --species NC10H22 [--file_loc FILE_LOC] [--units UNITS] [--vars VARS [VARS ...]]

  options:
    -h, --help            show this help message and exit
    --species NC10H22     Species name
    --file_loc FILE_LOC   Location of data files. Files should be called rho.dat, mu.dat, and/or lambda.dat
    --units UNITS         Units, either MKS or CGS
    --vars VARS [VARS ...]
                          Which variables to fit, ex. mu lambda rho
