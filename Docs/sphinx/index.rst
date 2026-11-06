.. PelePhysics documentation master file, created by
   sphinx-quickstart on Sat Oct 20 12:17:48 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========
PelePhysics
===========

`PelePhysics` is a repository of physics databases and implementation code for use within the other `Pele` codes. In particular, the choice of chemistry and transport models as well as associated functions and capabilities are managed in `PelePhysics`.  `PelePhysics` has an official project `homepage <https://Pele-Suite.github.io/>`_, and can be obtained via
`GitHub <https://github.com/Pele-Suite/PelePhysics>`_.  The documentation pages appearing here are distributed with the code in the ``Docs/sphinx`` folder as "restructured text" files.
The html is built automatically with certain pushes to the `PelePhysics` GibHub repository.  A local version can also be built as follows ::

    cd ${PELE_PHYSICS_DIR}/build
    sphinx-build -M html ../Docs/sphinx .

where ``PELE_PHYSICS_DIR`` is the location of your clone of the `PelePhysics` repository.  To view the local pages,
point your web browser at the file ``${PELE_PHYSICS_DIR}/build/html/index.html``.

.. |a| image:: ./Visualization/PeleSuite.png

.. table::
   :align: center

   +-----+
   | |a| |
   +-----+


.. toctree::
   :maxdepth: 2
   :caption: Documentation contents:

   GettingStarted.rst
   Introduction.rst
   Chemistry.rst
   Transport.rst
   Thermodynamics.rst
   EOS.rst
   Spray.rst
   Soot.rst
   Utility.rst
   Support.rst
   Tutorials.rst
   DeveloperGuide.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
