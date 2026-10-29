.. role:: cpp(code)
   :language: c++

.. _sec:tutorials:

Tutorials
=========

.. toctree::
   :maxdepth: 1

Tutorial 1 - Generating NC12H26 QSS mechanism with analytical Jacobian
----------------------------------------------------------------------
.. _sec_tutqss1:

Update ``poetry``::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ poetry update  

Make sure the list of non-QSS species is correct in ``${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu_qss/non_qssa_list.yaml``


In the next step, ``skeletal.yaml`` is the ``mechanism.yaml`` of the skeletal version of the mechanism (here available under ``${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu``).
If only CHEMKIN files are available for the skeletal mechanism, see :ref:`Converting CHEMKIN files <sec_convertCK>` for generating ``skeletal.yaml``.

Generate ``qssa.yaml`` from ``skeletal.yaml`` and ``non_qssa_list.yaml``::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ poetry run qssa -f ${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu_qss/skeletal.yaml -n ${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu_qss/non_qssa_list.yaml

Generate ``mechanism.H`` and ``mechanism.cpp`` from ``qssa.yaml``::
  
  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ poetry run convert -f ${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu_qss/qssa.yaml --qss_format_input ${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu_qss/qssa_input.toml --qss_symbolic_jacobian

We recommend using the following ``qssa_input.toml``::
 
  $ [Readability]
    hformat =                       "gpu"
    
    [Arithmetic]
    remove_1 =                      true
    remove_pow =                    true
    remove_pow10 =                  true
    
    [Replacement]
    min_op_count =                  0
    min_op_count_all =              10
    gradual_op_count =              true
    remove_single_symbols_cse =     true
    
    [Recycle]
    store_in_jacobian =             true
    recycle_cse =                   true
    
    [Characters]
    round_decimals =                true


Tutorial 2 - Generating NC12H26 QSS mechanism without analytical Jacobian
-------------------------------------------------------------------------
.. _sec_tutqss2:

Update ``poetry``::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ poetry update  

Make sure the list of non-QSS species is correct in ``Mechanisms/dodecane_lu_qss/non_qssa_list.yaml``

In the next step, ``skeletal.yaml`` is the ``mechanism.yaml`` of the skeletal version of the mechanism (here available under ``${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu``).
If only CHEMKIN files are available for the skeletal mechanism, see :ref:`Converting CHEMKIN files <sec_convertCK>` for generating ``skeletal.yaml``.

Generate ``qssa.yaml`` from ``skeletal.yaml`` and ``non_qssa_list.yaml``::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ poetry run qssa -f ${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu_qss/skeletal.yaml -n ${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu_qss/non_qssa_list.yaml

Generate ``mechanism.H`` and ``mechanism.cpp`` from ``qssa.yaml``::
  
  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ poetry run convert -f ${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu_qss/qssa.yaml

Tutorial 3 - Generating NC12H26 Skeletal mechanism
--------------------------------------------------
.. _sec_tutskel3:

Update ``poetry``::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ poetry update  

For all the available mechanisms, a Cantera yaml format is provided.
If only CHEMKIN files are available, see :ref:`Converting CHEMKIN files <sec_convertCK>` for generating ``mechanism.yaml``.

Generate ``mechanism.H`` and ``mechanism.cpp`` from ``mechanism.yaml``::

  $ cd ${PELE_PHYSICS_HOME}/Support/ceptr
  $ poetry run convert -f ${PELE_PHYSICS_HOME}/Mechanisms/dodecane_lu/mechanism.yaml

