User's Guide
==============================

Structure and philosphy
------------------------------
ThemisPy is a library of utility functions for postprocessing Themis_ analyses.
These have been organized into four packages:

* chain -- Objects and methods for manipulating MCMC chains.
* data -- Objects and methods for processing observational data.
* diag -- Diagnostics.
* vis -- Visualizations of Themis_ analyses.

In addition there two additional top-level directories:

* docs -- Documentation (e.g., what you are reading now).
* scripts -- Python scripts that make use of ThemisPy library functions.


Packages
------------------------------
Individual package documentation organized by area.

.. toctree::
   :maxdepth: 3

   ./chain
   ./data
   ./diag
   ./vis


Command line tools
------------------------------
A set of executable python scripts are installed that provide both
commonly used utilities and examples.  All can be found in the scripts
subdirectory.  All are installed system wide during package installation.

.. toctree::
   :maxdepth: 1

   ./scripts

   
.. _Themis: https://github.com/PerimeterInstitute/Themis
   
