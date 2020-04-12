###########################
#
# Package:
#   calibration
#
# Provides:
#   Functions for loading for calibration quantities from Themis models
#   

import numpy as np
import copy
from matplotlib import cm
from matplotlib.colors import is_color_like, ListedColormap, to_rgba
import matplotlib.pyplot as plt

from themispy import vis as typlt

# Read in astropy, if possible
try:
    from astropy.time import Time
    astropy_found = True
except:
    warnings.warn("Package astropy not found.  Some functionality will not be available.  If this is necessary, please ensure astropy is installed.", Warning)
    astropy_found = False


# # Read in ehtim, if possible
# try:
#     import ehtim as eh
#     ehtim_found = True
# except:
#     warnings.warn("Package ehtim not found.  Some functionality will not be available.  If this is necessary, please ensure ehtim is installed.", Warning)
#     ehtim_found = False


def read_gain_amplitude_correction_file(gainfile) :
    """
    Reads in a gain amplitude correction file and returns an appropriately formatted dictionary.

    Warning:
      If astropy is unavailable, toffset will return the number of seconds since 12am, Jan 1, 2000 instead of an astropy.Time object.

    Args:
      gainfile (str): Name of the file containing the gain correction amplitude data.

    Returns:
      (numpy.ndarray): Indexed dictionary of start and end times (relative to the first scan time) complex gains and the  by keys ['tstart','tend','stations','toffset','nindependent',<station codes>]
    """

    # Read header information
    infile = open(gainfile,'r')
    t0=float(infile.readline().split()[5])
    n_ind_gains=int(infile.readline().split()[5])
    station_names=infile.readline().split()[6:]
    infile.close()

    # If possible, generate astropy time object with the offset time
    if (astropy_found) :
        tJ2000 = Time('J2000.0',format='jyear_str').mjd
        time_offset = Time(t0/(24*3600.0)+tJ2000,format='mjd')
    else :
        warn.warning('Because astropy is not found, functionality that depends on the toffset will be unavailable.',Warning)
        time_offset = t0
        
    # Read correction information
    d = np.loadtxt(gainfile,skiprows=3)
    ts=d[:,0]/3600. # Epoch start time, converted to hours
    te=d[:,1]/3600. # Epoch end time, converted to hours
    gains=d[:,2:] # Gain corrections
    
    # Generate a dictionary
    gain_data = { 'tstart':ts, 'tend':te }
    for k in range(len(station_names)) :
        gain_data[station_names[k]] = gains[:,k] + 1.0+0.j
    gain_data['stations'] = station_names
    gain_data['toffset'] = time_offset
    gain_data['nindependent'] = n_ind_gains
    
    return gain_data


def read_complex_gain_file(gainfile) :
    """
    Reads in a complex gain file and returns an appropriately formatted dictionary.

    Warning:
      If astropy is unavailable, toffset will return the number of seconds since 12am, Jan 1, 2000 instead of an astropy.Time object.

    Args:
      gainfile (str): Name of the file containing the complex gain data.

    Returns:
      (numpy.ndarray): Indexed dictionary of start and end times (relative to the first scan time) complex gains and the  by keys ['tstart','tend','stations','toffset','nindependent',<station codes>]
    """

    # Read header information
    infile = open(gainfile,'r')
    t0=float(infile.readline().split()[5])
    n_ind_gains=int(infile.readline().split()[5])
    station_names=infile.readline().split()[6::2]
    for k,sn in enumerate(station_names) :
        station_names[k] = sn.replace('.real','')
    infile.close()

    # If possible, generate astropy time object with the offset time
    if (astropy_found) :
        tJ2000 = Time('J2000.0',format='jyear_str').mjd
        time_offset = Time(t0/(24*3600.0)+tJ2000,format='mjd')
    else :
        warn.warning('Because astropy is not found, functionality that depends on the toffset will be unavailable.',Warning)
        time_offset = t0
    
    # Read correction information
    d = np.loadtxt(gainfile,skiprows=3)
    ts=d[:,0]/3600. # Epoch start time, converted to hours
    te=d[:,1]/3600. # Epoch end time, converted to hours
    gains=d[:,2::2]+1.j*d[:,3::2] # Gain corrections
    
    # Generate a dictionary
    gain_data = { 'tstart':ts, 'tend':te }
    for k in range(len(station_names)) :
        gain_data[station_names[k]] = gains[:,k]
    gain_data['stations'] = station_names
    gain_data['toffset'] = time_offset
    gain_data['nindependent'] = n_ind_gains
    
    return gain_data


def read_gain_file(gainfile) :
    """
    Reads in a gain file and returns an appropriately formatted dictionary. 
    The file type is determined automatically from the headers.

    Args:
      gainfile (str): Name of the file containing the gain data.

    Returns:
      (numpy.ndarray): Indexed dictionary of start and end times (relative to the first scan time) complex gains and the  by keys ['tstart','tend','stations','toffset','nindependent',<station codes>]
    """

    # Read header information
    infile = open(gainfile,'r')
    t0=float(infile.readline().split()[5])
    n_ind_gains=int(infile.readline().split()[5])
    station_names=infile.readline().split()[6:]
    infile.close()

    # Look to see if headers are in the .real/.imag format or not
    nreal=0
    nimag=0
    for sn in station_names :
        if ('.real' in sn) :
            nreal += 1
        if ('.imag' in sn) :
            nimag += 1
            
    # Read in the data with the appropriate function
    if (nreal==len(station_names)//2 and nimag==len(station_names)//2) :
        gain_data = read_complex_gain_file(gainfile)
    else :
        gain_data = read_gain_amplitude_correction_file(gainfile)

    return gain_data


def annotate_gain_data(gain_data) :
    """
    Adds elements to the gain_data dict that indicate the uniqueness of the gain solution.  
    This is indicated by an integer code under the key "<station>.status" which translates to:
    
    * 0 ... Solved for and uniquely determined.
    * 1 ... Solved for but degenerate with another value.
    * 2 ... Not being solved for.

    These are determined via "tells" in the gain solution (too close to unity, phase to close to 
    zero), and are therefore susceptable to rare instances of confusion.

    Args:
      gain_data (dict): Gain data as produced by :func:`read_gain_file`.

    Returns:
      (dict): Gain data with additional entries.
    """

    # Make a deep copy
    gain_data = copy.deepcopy(gain_data)
    
    # Get list of station names
    station_names = gain_data['stations']

    # Make sure this is not already annotated
    status_already_exists=False
    for sn in station_names :
        status_already_exists = status_already_exists or (sn+'.status' in station_names)
    if (status_already_exists) :
        raise RuntimeError("It appears that you are attempting to annotate already annotated gain data.  Cowardly refusing.")

    # Identify if gains were set
    is_gain_set = np.zeros((len(station_names),len(gain_data['tstart'])))
    for k,sn in enumerate(station_names) :
        is_gain_set[k,:] = (np.abs(gain_data[sn])-1>1e-10)+(np.abs(np.angle(gain_data[sn]))>1e-10)
    is_gain_degenerate = (np.sum(is_gain_set,axis=0)<=2)

    # Add annotation
    for k,sn in enumerate(station_names) :
        gain_data[sn+'.status'] = 2*(is_gain_set[k,:]==False) + is_gain_degenerate*is_gain_set[k,:]
    
    return gain_data


def plot_station_gain_amplitudes(gain_data, invert=False, add_station_labels=True,  add_quant_labels=False, absolute_time=True, colormap='jet', grid=True) :
    """
    Plots station gain amplitudes relative to the expected values of 1.  Optionally adds quantitative labels.

    Args:
      gain_data (dict): Gain data as produced by :func:`read_gain_file`.
      invert (bool): If True inverts the gains to be applied to the data (heresy!) instead of the model. Default: False.
      add_station_labels (bool): Adds station names as labels to the right side of the plot. Default: True.
      add_quant_labels (bool): Adds quantitative labels to the right margin, shifts axes positions accordingly. Default: False.
      absolute_time (bool): If True, plots against the hour in UTC.  If False, plots against the time after the first gain reconstruction epoch begins. Default: True.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm` or acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'b'.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.    

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    # Get information about which gains are set and degenerate
    annotated_gain_data = annotate_gain_data(gain_data)

    # If absolute_time is set but astropy is not found, unset it and warn
    if ( absolute_time and (not astropy_found) ) :
        warn.warning('Because astropy is not found, absolute_time is unavailable',Warning)
        absolute_time=False

    # Get the times at which to plot points
    if (absolute_time) :
        time_offset = annotated_gain_data['toffset']
        hour_offset = int(time_offset.yday.split(':')[2]) + int(time_offset.yday.split(':')[3])/60.0 + float(time_offset.yday.split(':')[4])/3600.0 # In hrs
    else :
        hour_offset = 0.0
    tc = 0.5*(annotated_gain_data['tend']+annotated_gain_data['tstart']) + hour_offset # In hrs
    tw = 0.5*(annotated_gain_data['tend']-annotated_gain_data['tstart']) # In hrs

    # Get colors
    number_of_stations = len(annotated_gain_data['stations'])

    # Ensure an axes is set
    plt.sca(plt.gca())

    # Get colors
    if ( not is_color_like(colormap) ) :
        cmap = cm.get_cmap(colormap)

    # Shift the axes if station labels are desired (always -0.0")
    if (add_station_labels) :
        typlt.axes_adjust(shift_width=-0.0)

    # Shift the axes if station labels are desired (always -0.75")
    if (add_quant_labels==True) :
        typlt.axes_adjust(shift_width=-0.75)
    
    # Loop over the stations and plot
    for k,sn in enumerate(annotated_gain_data['stations']) :

        if ( not is_color_like(colormap) ) :
            color = cmap(k/float(number_of_stations-1))
        else :
            color = colormap

        # Get the relevant gains (and flip if desired)
        g = annotated_gain_data[sn]
        if ( invert ):
            g = 1.0/g

        # Plot zero-line
        plt.plot(tc,0*tc+k,':',color=color,linewidth=1)

        # Plot non-degenerate, defined gains
        use = (annotated_gain_data[sn+'.status']==0)
        plt.errorbar(tc[use],np.abs(g[use])-1+k,xerr=tw[use],color=color,marker='.',capthick=0,linewidth=2,linestyle='none')

        # Plot degenerate, defined gains
        use = (annotated_gain_data[sn+'.status']==1)
        plt.errorbar(tc[use],np.abs(g[use])-1+k,xerr=tw[use],color=[0.5,0.5,0.5],marker='.',capthick=0,linewidth=2,linestyle='none')

        # Add station name
        if (add_station_labels) :
            plt.text(tc[-1]+tw[-1]+0.25 + 0.02*(tc[-1]-tc[0]+0.5),k,'%s'%sn,color=color,va='center')

        # Add quantification labels if desired
        if (add_quant_labels) :
            use = (annotated_gain_data[sn+'.status']<=1)
            gtmp = np.abs(g[use])-1
            if (gtmp.size>0) :
                vals = np.sort(gtmp)
                inds = np.floor(np.array([0.25,0.5,0.75])*vals.size).astype(int) 
                quants = vals[ inds ] * 100
                #print(quants, np.median(gtmp))
                plt.text(tc[-1]+tw[-1]+0.25 + 0.1*(tc[-1]-tc[0]+0.5),k,r'$%6.3g$%%$^{+%6.3g}_{-%6.3g}$'%(quants[1],quants[2]-quants[1],quants[1]-quants[0]),color=[0.5,0.5,0.5],va='center',fontsize=10)
            

    ylims = plt.gca().get_ylim()
    ymin = min(ylims[0],-1)
    ymax = max(ylims[1],number_of_stations)
    plt.ylim((ymin,ymax))
    plt.xlim((tc[0]-tw[0]-0.25,tc[-1]+tw[-1]+0.25))
    if (absolute_time) :
        plt.xlabel(r'$t~({\rm UTC})$')
    else :
        plt.xlabel(r'$\Delta t~({\rm hr})$')
        
    plt.ylabel(r'$|G|-1$')

    plt.gca().grid(grid)

    
def plot_station_gain_phases(gain_data, invert=False, add_station_labels=True,  absolute_time=True, colormap='jet', grid=True) :
    """
    Plots station gain phases.

    Args:
      gain_data (dict): Gain data as produced by :func:`read_gain_file`.
      invert (bool): If True inverts the gains to be applied to the data (heresy!) instead of the model. Default: False.
      add_station_labels (bool): Adds station names as labels to the right side of the plot. Default: True.
      absolute_time (bool): If True, plots against the hour in UTC.  If False, plots against the time after the first gain reconstruction epoch begins. Default: True.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm` or acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'b'.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
    
    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    # Get information about which gains are set and degenerate
    annotated_gain_data = annotate_gain_data(gain_data)

    # If absolute_time is set but astropy is not found, unset it and warn
    if ( absolute_time and (not astropy_found) ) :
        warn.warning('Because astropy is not found, absolute_time is unavailable',Warning)
        absolute_time=False

    # Get the times at which to plot points
    if (absolute_time) :
        time_offset = annotated_gain_data['toffset']
        hour_offset = int(time_offset.yday.split(':')[2]) + int(time_offset.yday.split(':')[3])/60.0 + float(time_offset.yday.split(':')[4])/3600.0 # In hrs
    else :
        hour_offset = 0.0
    tc = 0.5*(annotated_gain_data['tend']+annotated_gain_data['tstart']) + hour_offset # In hrs
    tw = 0.5*(annotated_gain_data['tend']-annotated_gain_data['tstart']) # In hrs

    # Get colors
    number_of_stations = len(annotated_gain_data['stations'])

    # Ensure an axes is set
    plt.sca(plt.gca())

    # Get colors
    if ( not is_color_like(colormap) ) :
        cmap = cm.get_cmap(colormap)

    # Shift the axes if station labels are desired (always -0.0")
    if (add_station_labels) :
        typlt.axes_adjust(shift_width=-0.0)

    # Loop over the stations and plot
    for k,sn in enumerate(annotated_gain_data['stations']) :

        if ( not is_color_like(colormap) ) :
            color = cmap(k/float(number_of_stations-1))
        else :
            color = colormap

        # Get the relevant gains (and flip if desired)
        g = annotated_gain_data[sn]
        if ( invert ):
            g = 1.0/g

        # Plot zero-line
        plt.plot(tc,0*tc+k,':',color=color,linewidth=1)

        # Plot non-degenerate, defined gains
        use = (annotated_gain_data[sn+'.status']==0)
        plt.errorbar(tc[use],np.angle(g[use])/(2.0*np.pi)+k,xerr=tw[use],color=color,marker='.',capthick=0,linewidth=2,linestyle='none')

        # Plot degenerate, defined gains
        use = (annotated_gain_data[sn+'.status']==1)
        plt.errorbar(tc[use],np.angle(g[use])/(2.0*np.pi)+k,xerr=tw[use],color=[0.5,0.5,0.5],marker='.',capthick=0,linewidth=2,linestyle='none')

        # Add station name
        if (add_station_labels) :
            plt.text(tc[-1]+tw[-1]+0.25 + 0.02*(tc[-1]-tc[0]+0.5),k,'%s'%sn,color=color,va='center')


    ylims = plt.gca().get_ylim()
    ymin = min(ylims[0],-1)
    ymax = max(ylims[1],number_of_stations)
    plt.ylim((ymin,ymax))
    plt.xlim((tc[0]-tw[0]-0.25,tc[-1]+tw[-1]+0.25))
    if (absolute_time) :
        plt.xlabel(r'$t~({\rm UTC})$')
    else :
        plt.xlabel(r'$\Delta t~({\rm hr})$')
        
    plt.ylabel(r'${\rm arg}(G)/(2\pi)$')

    plt.gca().grid(grid)

