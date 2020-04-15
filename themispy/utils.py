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


