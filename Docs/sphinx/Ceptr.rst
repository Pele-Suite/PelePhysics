.. highlight:: rst

.. _sec:ceptr:

CEPTR: Chemistry Evaluation for Pele Through Recasting
======================================================

We use CEPTR to generate C++ mechanism files from `Cantera <https://cantera.org>`_ YAML chemistry files. CEPTR is a python package part of the PelePhysics source code.

.. _sec_ceptr_software:

Software requirements
---------------------

The CEPTR package uses `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ to manage the Python dependencies. uv can typically be installed through system package managers (e.g. HomeBrew) or following the instructions in uv's documentation.

To install CEPTR and its dependencies::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ uv sync --locked

This installs the exact versions recorded in ``Support/ceptr/uv.lock``, so that every user generates the same mechanisms. uv also installs a suitable Python version (given in ``Support/ceptr/.python-version``) if none is found on the system.

The plotting packages (matplotlib, networkx and SciPy) are an optional ``plot`` extra: they are only needed for
``qssa --visualize`` and for the plotting scripts elsewhere in PelePhysics (``plotPMF.py``, ``gen_hit_ic.py``).
To install them too::

  $ uv sync --locked --extra plot

``uv sync`` removes what it is not asked for, so a later ``uv sync --locked`` uninstalls them again: give
``--extra plot`` to every sync (or to ``uv run``) when you need them.

.. note:: CEPTR is a standard Python package, so it can also be installed with ``pip install -e .`` from ``Support/ceptr``. pip then picks any versions allowed by ``pyproject.toml``, which may generate mechanisms that differ from the committed ones.

Usage
-----

Generating YAML chemistry files: converting from CHEMKIN files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _sec_convertCK:

We rely on Cantera's ``ck2yaml`` utility to convert CHEMKIN files to the Cantera YAML format::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ uv run ck2yaml --input=${PATH_TO_CHEMKIN_DIR}/mechanism.inp --thermo=${PATH_TO_CHEMKIN_DIR}/therm.dat --transport=${PATH_TO_CHEMKIN_DIR}/tran.dat --permissive

Of course, the files ``tran.dat`` and ``therm.dat`` are optional if already included in the ``mechanism.inp`` file.


Generating Pele-compatible mechanisms for a single chemistry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three ways to use CEPTR to generate C++ mechanism files for a given chemistry.

1. Using CEPTR directly:

   Executed from the ``${PELE_PHYSICS_HOME}/Support/ceptr`` directory, the most general usage of CEPTR is::

     $ uv run convert -f ${PELE_PHYSICS_HOME}/Mechanisms/${chemistry}/mechanism.yaml \
       --chemistry {chemistry-type} \
       --gas_name {gas-name} \
       --interface_name {interface-name} \
       --plog_pressure {value in Pa}

   The ``--chemistry`` (or ``-c``) argument allows users to convey if the ``${chemistry}`` of interest is either one of two valid options, namely, ``homogeneous`` or ``heterogeneous``. The default value for ``{chemistry-type}`` is ``homogeneous``.
   The ``--gas_name`` argument allows users to specify the names of the homogeneous phase to use, as several different ones can be defined in the corresponding ``mechanism.yaml`` file (under the ``phases:`` item). The default value for ``{gas-name}`` is ``gas``.
   Finally, the ``--interface_name`` arguments allow users to specify the name of the gas-solid interface, also prescribed in the corresponding ``mechanism.yaml`` file. The default value for ``{interface-name}`` is ``None``.

   Note that if ``--chemistry heterogeneous`` is specified, the user must necessarily specify a corresponding ``{interface-name}``.

   An example of directly using CEPTR for homogeneous mechanisms is::

     $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
     $ uv run convert -f ${PELE_PHYSICS_HOME}/Mechanisms/LiDryer/mechanism.yaml

   .. note:: CEPTR interpretations of heterogeneous mechanisms is currently a work in progress.

   .. note:: CEPTR offers only partial support for mechanisms with pressure-dependent Arrhenius (PLOG) reactions.
             These reactions are fixed to a constant pressure, specified with the ``--plog_pressure`` argument,
             in the CEPTR-generated source code. Mechanisms with PLOG reactions generated in this way are limited
             in validity to that specific pressure and would not be suitable for compressible flows with significant
             pressure variation. For that reason, source code for these mechanisms is saved in pressure-specific directories, for
             example ``${PELE_PHYSICS_HOME}/Mechanisms/POLIMI2020/1_000atm/``.


2. Using a helper script in the directory containing the ``mechanism.yaml`` file::

     $ ./convert.sh

   .. note:: The helper scripts run CEPTR with ``uv run --locked``, installing the dependencies on first use. Without uv,
             they fall back to ``python3`` in the current environment, which then needs CEPTR and its dependencies
             installed (e.g. with ``pip install -e ${PELE_PHYSICS_HOME}/Support/ceptr``).

3. Using a helper script in the ``Mechanisms`` directory::

     $ bash ${PELE_PHYSICS_HOME}/Mechanisms/converter.sh -f ./LiDryer/mechanism.yaml


Generating a reduced, QSS chemistry file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step consists in generating a QSS chemistry YAML file from a skeletal or a detailed YAML file. To do so, one needs: the mechanism YAML file ``skeletal.yaml``, as well as a list of non-QSS species, ``non_qssa_list.yaml``. The following command will generate a QSS YAML file, ``qssa.yaml``::

    $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
    $ uv run qssa -f ${PATH_TO_YAML}/skeletal.yaml -n ${PATH_TO_YAML}/non_qssa_list.yaml

The full list of options is::

  $ uv run qssa -h
  usage: qssa [-h] -f FNAME -n NQSSA [-m {0,1,2}] [-v]

  Mechanism converter

  optional arguments:
    -h, --help            show this help message and exit
    -f FNAME, --fname FNAME
                          Mechanism file
    -n NQSSA, --nqssa NQSSA
                          Non-QSSA species list
    -m {0,1,2}, --method {0,1,2}
                          QSSA method (default: 2)
    -v, --visualize       Visualize quadratic coupling and QSSA dependencies

For a detailed description of these options and a further information on the way QSS mechanism are treated in CEPTR the reader may consult :ref:`the QSS section <sec_qss>`.

To generate a QSS C++ mechanism from the ``.yaml`` file thus created, tailored to your needs, please refer to Tutorials :ref:`Generating NC12H26 QSS mechanism with analytical jacobian <sec_tutqss1>` and :ref:`Generating NC12H26 QSS mechanism without analytical jacobian <sec_tutqss2>`.

Batched generation of Pele-compatible mechanisms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   If you are using batched generation as outlined here, it will automatically use multiprocessing to generate the files in parallel using all CPUs detected on the machine. If you want to change that you can pass the optional argument ``-n NPCU``, wheren ``NCPU`` is an integer indicating the number of processes you want to use.


For non-reduced chemistries, CEPTR can take a file with a list of ``mechanism.yaml`` files to convert::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ uv run convert -l ${PELE_PHYSICS_HOME}/Mechanisms/list_mech

The list files take one mechanism per line; blank lines and lines starting with ``#`` are skipped. A mechanism with PLOG
reactions needs its pressure in Pa, given on its line as ``mechanism.yaml --plog=101325.0``; for the lines without it,
``--plog_pressure`` gives the pressure for the whole list.

For reduced chemistries, CEPTR can take a file with a list of ``qssa.yaml`` and ``qssa_input.toml`` to convert::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ uv run convert -lq ${PELE_PHYSICS_HOME}/Mechanisms/list_qss_mech

For generating ``qssa.yaml`` for reduced chemistries, CEPTR can take a file with a list of ``skeletal.yaml`` and ``non_qssa_list.yaml``::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ uv run qssa -lq ${PELE_PHYSICS_HOME}/Mechanisms/list_qss_mech

To generate all mechanisms::

  $ uv run convert -l ${PELE_PHYSICS_HOME}/Mechanisms/list_mech
  $ uv run qssa -lq ${PELE_PHYSICS_HOME}/Mechanisms/list_qss_mech
  $ uv run convert -lq ${PELE_PHYSICS_HOME}/Mechanisms/list_qss_mech
