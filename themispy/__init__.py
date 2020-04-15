"""
This is a set of postprocessing tools for Themis analyses.

Tools may be accessed via

>>> import themispy as ty

Available subpackages
---------------------
diag
   Chain, sampler, and general management diagnostics
data
   Residual, systematics, calibration
modvis
   Model image visulalization
chain
   Tools for navigating and modifying chains
"""

__author__="Themis Development Team"
__bibtex__ = r"""@Article{Themis:2020,
  %%% Fill in from ADS!
}"""


__all__=['chain', 'diag', 'data', 'vis']
from . import *


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


