.. highlight:: rst


Developer Guidelines
====================

The following developer guidelines apply to all code that is to be committed to the `Pele` suite. 

Before adding files for a commit to be pushed to a branch of `PelePhysics`, first run the following formatting tools depending on the type of code:


C++ Code
--------
All C++ code should be processed by the Clang formatter prior to being added for commit.

Run ``clang-format``::

    clang-format -i FILE.cpp
    clang-format -i FILE.H

Will apply all of the correct formatting and make the changes directly to the cpp source files.


Python
------

The tools necessary to format Python code (currently only used within CEPTR) are maintained through uv. They
are `ruff <https://docs.astral.sh/ruff/>`_ commands, run from ``Support/ceptr``.

1) Run the linter, fixing what it can (including the sorting of imports)::

    uv run ruff check --fix .

This lists the remaining issues that need to be addressed to adhere to current Python best practices.

2) Format the code::

    uv run ruff format .

Once all ``ruff check`` messages have been addressed, the code will match the `Pele` suite standard.

