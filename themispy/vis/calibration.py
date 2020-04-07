###########################
#
# Package:
#   calibration
#
# Provides:
#   Functions for loading for calibration quantities from Themis models
#   


import numpy as np

# Read in ehtim, if possible
try:
    import ehtim as eh
    ehtim_found = True
except:
    warnings.warn("Package ehtim not found.  Some functionality will not be available.  If this is necessary, please ensure ehtim is installed.", Warning)
    ehtim_found = False


def read_gain_amplitude_correction_file(gainfile) :
    """
    Reads in a gain amplitude correction file and returns an appropriately formated dictionary.

    Args:
      gainfile (str): Name of the file containing the gain correction amplitude data.

    Returns:
      (numpy.ndarray): Indexed by keys []
    """

    # Read header information
    infile = open(gainfile,'r')
    t0=float(infile.readline().split()[5])
    n_ind_gains=int(infile.readline().split()[5])
    station_names=infile.readline().split()[6:]
    infile.close()
    print("Start time (s from J2000):",t0)
    print("Number of independent gains:",n_ind_gains)
    print("List of gains:",station_names)

    # Read correction information
    d = np.loadtxt(gainfile,skiprows=3)
    ts=d[:,0]/3600. # Epoch start time, converted to hours
    te=d[:,1]/3600. # Epoch end time, converted to hours
    gains=d[:,2:] # Gain corrections
    
    # Generate a dictionary
    gain_data = { 'tstart':ts, 'tend':te }
    for k in range(len(station_names)) :
        gain_data[station_names[k]] = gains[:,k]*(1.0+0.j)

    return gain_data


def read_gain_file(gainfile) :

    # Read header information
    infile = open(gainfile,'r')
    t0=float(infile.readline().split()[5])
    n_ind_gains=int(infile.readline().split()[5])
    station_names=infile.readline().split()[6::2]
    for k,sn in enumerate(station_names) :
        station_names[k] = sn.replace('.real','')
    infile.close()
    print("Start time (s from J2000):",t0)
    print("Number of independent gains:",n_ind_gains)
    print("List of gains:",station_names)

    # Read correction information
    d = np.loadtxt(gainfile,skiprows=3)
    ts=d[:,0]/3600. # Epoch start time, converted to hours
    te=d[:,1]/3600. # Epoch end time, converted to hours
    gains=d[:,2::2]+1.j*d[:,3::2] # Gain corrections
    
    # Generate a dictionary
    gain_data = { 'tstart':ts, 'tend':te }
    for k in range(len(station_names)) :
        gain_data[station_names[k]] = gains[:,k]

    return gain_data


    
