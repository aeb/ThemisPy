###########################
#
# Package:
#   utils
#
# Provides:
#   Utility functions used generally across ThemisPy
#   


# Warnings
import warnings
def _warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """
    A one-line warning format.
    """
    return 'WARNING: %s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = _warning_on_one_line

# Get a global version
from ._version import get_versions
themispy_version = get_versions()['version']
del get_versions


# Read in ehtim, if possible
import io
from contextlib import redirect_stdout
try:
    trap=io.StringIO()
    with redirect_stdout(trap) :
        import ehtim as eh
    ehtim_found = True
except:
    warnings.warn("Package ehtim not found.  Some functionality will not be available.  If this is necessary, please ensure ehtim is installed.", Warning)
    ehtim_found = False

    
# A simple progress bar
from sys import stderr as pbo
class progress_bar :
    """
    A simple ASCII progress bar to provide confidence that something is happening.

    Args:
      preamble (str): String to print before the bar. Default: 'Progress'.
      char (str): Character from which the progress bar is composed. Default: '='.
      length (int): Number of instances of the character that correspond to 100%. Default: 10.
    """
    
    def __init__(self,preamble='Progress',char='=',length=10) :

        self.preamble=preamble
        self.char = char
        self.length = length
        self.fmt = preamble + ' |' + '%' + '%i'%(self.length) + 's| %3i%%'
        self.full_length = len(self.fmt%(self.length*len(self.char)*' ',0))

    def increment(self,completion_fraction) :
        """
        Increments the progress bar.

        Args: 
          completion_fraction (float): Fraction of the job to indicate complete. Must be between [0,1].
        """
        
        ncomp = int(min(1.0,completion_fraction)*self.length)
        bar = ncomp*self.char + (self.length-ncomp)*' '
        pbo.write(self.full_length*'\b')
        pbo.write(self.fmt%(bar,int(100*completion_fraction)))
        pbo.flush()
        
        
    def start(self) :
        """
        Starts the progress bar. Alias for increment(0).
        """
        self.increment(0)

    def finish(self, epilog='DONE!') :
        """
        Finishes the progress bar and prints an epilog.

        Args:
          epilog (str): Epilog message to print. Default: 'DONE!'.
        """
        
        self.increment(1.0)
        pbo.write('  %s\n'%(epilog))
        pbo.flush()
        
