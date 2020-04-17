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
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return 'WARNING: %s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line


# A simple progress bar
from sys import stderr as pbo
class progress_bar :
    
    def __init__(self,preamble='Progress',char='=',length=10) :

        self.preamble=preamble
        self.char = char
        self.length = length
        self.fmt = preamble + ' |' + '%' + '%i'%(self.length) + 's| %3i%%'
        self.full_length = len(self.fmt%(self.length*' ',0))

    def increment(self,completion_fraction) :
        ncomp = int(completion_fraction*self.length)
        bar = ncomp*self.char + (self.length-ncomp)*' '
        pbo.write(self.full_length*'\b')
        pbo.write(self.fmt%(bar,int(100*completion_fraction)))
        pbo.flush()
        
        
    def start(self) :
        self.increment(0)

    def finish(self) :
        self.increment(1.0)
        pbo.write('  DONE! \n')
        pbo.flush()
        
