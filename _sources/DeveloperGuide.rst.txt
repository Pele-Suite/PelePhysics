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

The tools necessary to format Python code (currently only used within CEPTR) are maintained through Poetry.

1) Run ``isort``::

    poetry run isort . 

This will perform proper sorting of the installed Python libraries.

2) Run ``black``::

    poetry run black . 

This will perform proper formatting of all Python files to be consistent with all current files.

3) Run ``flake8``::

    poetry run flake8 . 

This will run diagnostics on all the Python files and will list a series of issues that need to be addressed to adhere to current Python best practices.


Once all ``flake8`` messages have been addressed, the code will match the `Pele` suite standard.

