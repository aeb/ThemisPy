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

import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return 'WARNING: %s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line


__all__=['chain'] #'diag', 'chain', 'data', 'modvis']
from . import *



from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

