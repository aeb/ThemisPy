###########################
#
# Package:
#   residual_plots
#
# Provides:
#   Functions for residual plots from residual files produced by Themis.
#

from themispy.utils import *

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.colors import is_color_like, to_rgba
import copy
import cmath

from themispy import data as tydata


def read_residuals(resfile_name, datafile_list=None, datafile=None, verbosity=0) :
    """
    Reads in a Themis residual file and returns a dictionary object with 
    the assiciated information.  If a datafile_list or datafile is provided, 
    will associate line-by-line time, baseline, source.  If provided, the 
    number of combined lines must match the number of data points in the 
    residual file.

    Args:
      resfile_file (str): Residual output file.
      datafile_list (str,list): Either the name of a file containing a list of Themis-formatted data files (e.g., v_file_list), or a list of data files.  Default: None.
      datafile (str): Name of file containing Themis-formatted data.  Default: None.
      verbosity (int): Verbosity level.  Default: 0.
    
    Returns:
      (dict): Dictionary of residual information, with specific structure dependent on the type of residuals.
    """

    
    # Check datafile_list vs datafile.
    if (not datafile_list is None) :
        if (not isinstance(datafile_list,list)) :
            datafile_list_name=datafile_list
            datafile_list = []
            for l in open(datafile_list_name,'r') :
                datafile_list.append(l.rstrip())
    elif (not datafile is None) :
        datafile_list = [datafile]
    if (verbosity>0) :
        if (not datafile_list is None) :
            print("Reading data files:")
            for datafile in datafile_list :
                print("  %s"%datafile)

    # Find residual file
    with open(resfile_name,'r') as f :
        restype = f.readline().split()[1]
    if (verbosity>0) :
        print("Reading residual file of type %s"%restype)

    # Dictionary
    resdata={}

    # Run through kinds of residual files
    if (restype=='likelihood_visibility') :
        d = np.loadtxt(resfile_name)
        resdata['type']='visibility'
        resdata['u']=d[:,0]
        resdata['v']=d[:,1]
        resdata['data']=d[:,2]+1.0j*d[:,6]
        resdata['error']=d[:,3]+1.0j*d[:,7]
        resdata['model']=d[:,4]+1.0j*d[:,8]
        resdata['residual']=d[:,5]+1.0j*d[:,9]

        # Read in data files
        if (not datafile_list is None) :
            for k,datafile in enumerate(datafile_list) :
                datasub = tydata.read_visibilities(datafile)
                if (k==0) :
                    data = datasub
                else :
                    for key in data.keys() :
                        data[key] = data[key].append(data[key],datasub[key])
                        
            if (len(resdata['residual'])!=len(data['source'])) :
                raise RuntimeError("Number of points in data file does not match number of points in residual file.  Expected %i, found %i."%(len(resdata['residual']),len(data['source'])))

            for key in ['source','year','day','time','baseline'] :
                resdata[key] = data[key]

        #for key in resdata.keys() :
        #    resdata[key] = np.array(resdata[key])    

    elif (restype=='likelihood_crosshand_visibilities') :
        d = np.loadtxt(resfile_name)
        resdata['type']='crosshand'
        resdata['u']=d[:,0]
        resdata['v']=d[:,1]
        resdata['field rotation 1']=d[:,2]
        resdata['field rotation 2']=d[:,3]
        resdata['data']={'RR':d[:,4]+1.0j*d[:,8], 'LL':d[:,12]+1.0j*d[:,16], 'RL':d[:,20]+1.0j*d[:,24], 'LR':d[:,28]+1.0j*d[:,32]}
        resdata['error']={'RR':d[:,5]+1.0j*d[:,9], 'LL':d[:,13]+1.0j*d[:,17], 'RL':d[:,21]+1.0j*d[:,25], 'LR':d[:,29]+1.0j*d[:,33]}
        resdata['model']={'RR':d[:,6]+1.0j*d[:,10], 'LL':d[:,14]+1.0j*d[:,18], 'RL':d[:,22]+1.0j*d[:,26], 'LR':d[:,30]+1.0j*d[:,34]}
        resdata['residual']={'RR':d[:,7]+1.0j*d[:,11], 'LL':d[:,15]+1.0j*d[:,19], 'RL':d[:,23]+1.0j*d[:,27], 'LR':d[:,31]+1.0j*d[:,35]}

        # Read in data files
        if (not datafile_list is None) :
            for k,datafile in enumerate(datafile_list) :
                datasub = tydata.read_crosshand_visibilities(datafile)
                if (k==0) :
                    data = datasub
                else :
                    for key in data.keys() :
                        data[key] = data[key].append(data[key],datasub[key])
                        
            if (len(resdata['residual']['RR'])!=len(data['source'])) :
                raise RuntimeError("Number of points in data file does not match number of points in residual file.  Expected %i, found %i."%(len(resdata['residual']),len(data['source'])))

            for key in ['source','year','day','time','baseline'] :
                resdata[key] = data[key]
        
                    
    elif (restype=='likelihood_visibility_amplitude' \
    or restype=='likelihood_marginalized_visibility_amplitude' \
    or restype=='likelihood_optimal_gain_correction_visibility_amplitude') :
        d = np.loadtxt(resfile_name)
        resdata['type']='amplitude'
        resdata['u']=d[:,0]
        resdata['v']=d[:,1]
        resdata['data']=d[:,2]
        resdata['error']=d[:,3]
        resdata['model']=d[:,4]
        resdata['residual']=d[:,5]

        # Read in data files
        if (not datafile_list is None) :
            for k,datafile in enumerate(datafile_list) :
                datasub = tydata.read_amplitudes(datafile)
                if (k==0) :
                    data = datasub
                else :
                    for key in data.keys() :
                        data[key] = data[key].append(data[key],datasub[key])
                        
            if (len(resdata['residual'])!=len(data['source'])) :
                raise RuntimeError("Number of points in data file does not match number of points in residual file.  Expected %i, found %i."%(len(resdata['residual']),len(data['source'])))

            for key in ['source','year','day','time','baseline'] :
                resdata[key] = data[key]

        #for key in resdata.keys() :
        #    resdata[key] = np.array(resdata[key])    
                
    elif (restype=='likelihood_closure_phase' \
    or restype=='likelihood_marginalized_closure_phase') :
        d = np.loadtxt(resfile_name)
        resdata['type']='closure phase'
        resdata['u1']=d[:,0]
        resdata['v1']=d[:,1]
        resdata['u2']=d[:,2]
        resdata['v2']=d[:,3]
        resdata['u3']=d[:,4]
        resdata['v3']=d[:,5]
        resdata['data']=d[:,6]
        resdata['error']=d[:,7]
        resdata['model']=d[:,8]
        resdata['residual']=d[:,9]

        # Read in data files
        if (not datafile_list is None) :
            for k,datafile in enumerate(datafile_list) :
                datasub = tydata.read_closure_phases(datafile)
                if (k==0) :
                    data = datasub
                else :
                    for key in data.keys() :
                        data[key] = data[key].append(data[key],datasub[key])
                        
            if (len(resdata['residual'])!=len(data['source'])) :
                raise RuntimeError("Number of points in data file does not match number of points in residual file.  Expected %i, found %i."%(len(resdata['residual']),len(data['source'])))

            for key in ['source','year','day','time','triangle'] :
                resdata[key] = data[key]

    elif (restype=='likelihood_closure_amplitude') :
        d = np.loadtxt(resfile_name)
        resdata['type']='closure amplitude'
        resdata['u1']=d[:,0]
        resdata['v1']=d[:,1]
        resdata['u2']=d[:,2]
        resdata['v2']=d[:,3]
        resdata['u3']=d[:,4]
        resdata['v3']=d[:,5]
        resdata['u4']=d[:,6]
        resdata['v4']=d[:,7]
        resdata['data']=d[:,8]
        resdata['error']=d[:,9]
        resdata['model']=d[:,10]
        resdata['residual']=d[:,11]

        # Read in data files
        if (not datafile_list is None) :
            raise RuntimeError("Reading closure amplitude data has not been implemented.")
            
            for k,datafile in enumerate(datafile_list) :
                datasub = tydata.read_closure_amplitudes(datafile)
                if (k==0) :
                    data = datasub
                else :
                    for key in data.keys() :
                        data[key] = data[key].append(data[key],datasub[key])
                        
            if (len(resdata['residual'])!=len(data['source'])) :
                raise RuntimeError("Number of points in data file does not match number of points in residual file.  Expected %i, found %i."%(len(resdata['residual']),len(data['source'])))

            for key in ['source','year','day','time','baseline'] :
                resdata[key] = data[key]

        #for key in resdata.keys() :
        #    resdata[key] = np.array(resdata[key])    

    elif (restype=='likelihood_log_closure_amplitude') :
        d = np.loadtxt(resfile_name)
        resdata['type']='log closure amplitude'
        resdata['u1']=d[:,0]
        resdata['v1']=d[:,1]
        resdata['u2']=d[:,2]
        resdata['v2']=d[:,3]
        resdata['u3']=d[:,4]
        resdata['v3']=d[:,5]
        resdata['u4']=d[:,6]
        resdata['v4']=d[:,7]
        resdata['data']=d[:,8]
        resdata['error']=d[:,9]
        resdata['model']=d[:,10]
        resdata['residual']=d[:,11]

        # Read in data files
        if (not datafile_list is None) :
            raise RuntimeError("Reading log closure amplitude data has not been implemented.")
            
            for k,datafile in enumerate(datafile_list) :
                datasub = tydata.read_log_closure_amplitudes(datafile)
                if (k==0) :
                    data = datasub
                else :
                    for key in data.keys() :
                        data[key] = data[key].append(data[key],datasub[key])
                        
            if (len(resdata['residual'])!=len(data['source'])) :
                raise RuntimeError("Number of points in data file does not match number of points in residual file.  Expected %i, found %i."%(len(resdata['residual']),len(data['source'])))

            for key in ['source','year','day','time','baseline'] :
                resdata[key] = data[key]

        #for key in resdata.keys() :
        #    resdata[key] = np.array(resdata[key])    
        
    else :
        raise RuntimeError("Unrecognized residual file type %s"%(restype))


    return resdata

def _station_codes_from_baseline(baseline) :
    """
    Returns station names from baseline glob.

    Args:
      baseline (str): Name of baseline (e.g., 'AAAP')
    
    Returns:
      (str,str): 
    """

    snl = len(baseline)//2
    return baseline[:snl],baseline[snl:]


def _station_codes_from_triangle(triangle) :
    """
    Returns station names from triangle glob.

    Args:
      baseline (str): Name of baseline (e.g., 'AAAP')
    
    Returns:
      (str,str): 
    """

    snl = len(triangle)//3
    return triangle[:snl],triangle[snl:2*snl],triangle[2*snl:]


def _station_codes_from_quadangle(quadangle) :
    """
    Returns station names from triangle glob.

    Args:
      baseline (str): Name of baseline (e.g., 'AAAP')
    
    Returns:
      (str,str): 
    """

    snl = len(triangle)//4
    return quadangle[:snl],quadangle[snl:2*snl],quadangle[2*snl:3*snl],quadangle[3*snl:]


def plot_amplitude_residuals(resdata, plot_type='uvamp', gain_data=None, station_list=None, residuals=True, resdist=2, datafmt='o', datacolor='b', modelfmt='.', modelcolor='r', grid=True, xscale='linear', yscale='linear') :
    """
    Plots comparison between the visibility amplitudes from the data and model in a variety of possible formats.

    Args:
      resdata (dict): Dictionary object containing the residual data as generated, e.g., by :func:`read_residuals`.
      plot_type (str): Type of residual plot to generate. Options are 'uvamp', 'u', 'v', 'time', 'amplitude', 'snr'. Default: 'uvamp'.
      gain_data (dict): Gain data object as generated, e.g., by :func:`read_gain_file`. If provided, data will be calibrated prior to residuals being plotted. Default: None.
      station_list (list,str): Station or list of stations to either exclude or restrict the residual plot to. Requires resdata to contain a key 'baselines'.  Station names prepended with '!' will be excluded.  If any station names without '!' are given, will show *only* those stations. Default: None.
      residuals (bool): If True produces a sub-panel with the error-weighted residuals plotted underneath the comparison plot. Default: True.
      resdist (int): If not None, produces a sub-panel with the distribution of residuals compared to a unit-variance Gaussian. If an int value is passed, it will set the number of bins per unit standard deviation. Default: 2.
      datafmt (str): Format specifier for the data points. Default: 'o'.
      datacolor (str,list): Color of the data points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'b'.
      modelfmt (str): Format specifier for the model points. Default: '.'.
      modelcolor (str,list): Color of the model points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.    
      xscale (str): The x-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.
      yscale (str): The y-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.

    Returns:
      (matplotlib.figure.Figure, list): Figure and list of axes handles.

    """

    # Create local copy so that we can modify it
    resdata_local = copy.deepcopy(resdata)
    
    
    # Apply gains
    if (not gain_data is None) :
        if (not 'time' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include time")

        tstart = gain_data['tstart']
        tend = gain_data['tend']
        toffset = ((gain_data['toffset'].mjd)%1)*24
        
        for j in range(len(resdata_local['baseline'])) :
            station1,station2 = _station_codes_from_baseline(resdata_local['baseline'][j])

            current_epoch = (gain_data['tstart']<(resdata_local['time'][j]-toffset))*((resdata_local['time'][j]-toffset)<=gain_data['tend'])
            GA = gain_data[station1][current_epoch]
            GB = gain_data[station2][current_epoch]
            
            for key in ['data','model','error','residual'] :
                resdata_local[key][j] = np.abs( (1.0/GA) * (1.0/np.conj(GB)) ) * resdata_local[key][j]
            

    # Apply station list flagging
    if (not station_list is None) :
        if (not 'baseline' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include baselines")
        if (not isinstance(station_list,list)) :
            station_list = [station_list]

        station_include_list = []
        station_exclude_list = []
        for station in station_list :
            if (station[0]=='!') :
                station_exclude_list.append(station[1:])
            else :
                station_include_list.append(station)


        keep = np.array([False]*len(resdata_local['baseline']))

        for j in range(len(resdata_local['baseline'])) :
            station1,station2 = _station_codes_from_baseline(resdata_local['baseline'][j])
            
            if (len(station_include_list)==0) :
                keep[j] = True
            else :
                keep[j] = ((station1 in station_include_list) or (station2 in station_include_list))

            if (len(station_exclude_list)>0) :
                keep[j] = keep[j] and not ((station1 in station_exclude_list) or (station2 in station_exclude_list))

        for key in ['time','u','v','data','model','residual','error'] :
            resdata_local[key] = resdata_local[key][keep]

    # Select coordinate
    if (plot_type=='uvamp') :
        x=np.sqrt(resdata_local['u']**2+resdata_local['v']**2)
        xlbl=r'$|u|$ (G$\lambda$)'
    elif (plot_type=='u') :
        x=resdata_local['u']
        xlbl=r'$u$ (G$\lambda$)'
    elif (plot_type=='v') :
        x=resdata_local['v']
        xlbl=r'$v$ (G$\lambda$)'
    elif (plot_type=='time') :
        if (not 'time' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include time")
        x=resdata_local['time']
        xlbl=r'$t$ (UTC)'
    elif (plot_type=='amplitude') :
        x=np.abs(resdata_local['data'])
        xlbl=r'$V$ (Jy)'
    elif (plot_type=='snr') :
        x=np.abs(resdata_local['data']/resdata_local['error'])
        xlbl=r'$S/N$'
    else :
        raise RuntimeError("Unrecognized plot type %s"%(plot_type))

    # Check if a residual distribution is desired and set defaults
    if (resdist is None) :
        resdist_numbins=2
        resdist = False
    else :
        resdist_numbins=resdist
        resdist = True
    
    # Create figure and axes objects
    if (residuals) :
        if (not resdist) :
            fig = plt.figure(figsize=[6.,5.])
            axs_res = plt.axes([0.15,0.10,0.83,0.25])
            axs_comp = plt.axes([0.15,0.38,0.83,0.57])
            axs_list = [axs_comp,axs_res]
        else :
            fig = plt.figure(figsize=[6.5,5.])
            axs_res = plt.axes([0.1385,0.10,0.7662,0.25])
            axs_comp = plt.axes([0.1385,0.38,0.7662,0.57])
            axs_resdist = plt.axes([0.915,0.10,0.07,0.25])
            axs_list = [axs_comp,axs_res,axs_resdist]
            
    else :
        fig = plt.figure(figsize=[6.,5.])
        axs_comp = plt.axes([0.15, 0.10, 0.83, 0.83])
        axs_list = [axs_comp]

    # Plot the data comparision
    plt.sca(axs_comp)
    plt.errorbar(x,resdata_local['data'],yerr=resdata_local['error'],fmt=datafmt,color=datacolor,markersize=4,zorder=10)
    plt.plot(x,resdata_local['model'],modelfmt,color=modelcolor,markersize=2,zorder=20)
    plt.ylabel(r'$|V|$ (Jy)')
    plt.grid(grid)
    axs_comp.set_xscale(xscale)
    axs_comp.set_yscale(yscale)

    # Plot the residuals if desired
    if (residuals) :
        axs_comp.xaxis.set_ticklabels([])
        plt.sca(axs_res)
        # Sigma guides
        plt.axhline(0,linestyle='-',color='r')
        plt.axhline(1.0,linestyle=':',color='r')
        plt.axhline(-1.0,linestyle=':',color='r')
        # plot the markers
        plt.plot(x,resdata_local['residual']/resdata_local['error'],datafmt,color=datacolor,markersize=4)
        plt.ylabel(r'Res.')
        plt.grid(grid)
        axs_res.set_xscale(xscale)

        if (resdist) :
            rr = resdata_local['residual']/resdata_local['error']
            ylim = axs_res.get_ylim()
            plt.sca(axs_resdist)
            xtmp = np.linspace(ylim[0],ylim[1],256)
            ytmp = np.exp(-xtmp**2/2.0)/np.sqrt(2*np.pi)
            plt.plot(ytmp,xtmp,'-r',alpha=0.5)
            var = np.var(rr)
            mean = np.average(rr)
            ytmp = np.exp(-(xtmp-mean)**2/(2.0*var))/np.sqrt(2*np.pi*var)
            plt.plot(ytmp,xtmp,':g',alpha=0.5)
            bins=np.arange(int(np.floor(ylim[0]*resdist_numbins)),int(np.ceil(ylim[1]*resdist_numbins)))/resdist_numbins
            plt.hist(rr,bins=bins,density=True,orientation="horizontal",color=datacolor,alpha=0.5)
            plt.ylim(ylim)
            plt.gca().yaxis.set_ticklabels([])
            plt.gca().xaxis.set_ticklabels([])            
            plt.grid(grid)
            plt.sca(axs_res)
        
    # Add the xlabels
    plt.xlabel(xlbl)

    return plt.gcf(),axs_list


def plot_closure_phase_residuals(resdata, plot_type='perimeter', gain_data=None, station_list=None, residuals=True, resdist=2, resdist_numbins=2, datafmt='o', datacolor='b', modelfmt='.', modelcolor='r', grid=True, xscale='linear', yscale='linear') :
    """
    Plots comparison between the closure phases from the data and model in a variety of possible formats.

    Args:
      resdata (dict): Dictionary object containing the residual data as generated, e.g., by :func:`read_residuals`.
      plot_type (str): Type of residual plot to generate. Options are 'perimeter', 'arear', 'uvmax', 'uvmin', 'umax', 'umin', 'vmax', 'vmin', 'time', 'snr'. Default: 'perimeter'.
      station_list (list,str): Station or list of stations to either exclude or restrict the residual plot to. Requires resdata to contain a key 'baselines'.  Station names prepended with '!' will be excluded.  If any station names without '!' are given, will show *only* those stations. Default: None.
      residuals (bool): If True produces a sub-panel with the error-weighted residuals plotted underneath the comparison plot. Default: True.
      resdist (int): If not None, produces a sub-panel with the distribution of residuals compared to a unit-variance Gaussian. If an int value is passed, it will set the number of bins per unit standard deviation. Default: 2.
      datafmt (str): Format specifier for the data points. Default: 'o'.
      datacolor (str,list): Color of the data points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'b'.
      modelfmt (str): Format specifier for the model points. Default: '.'.
      modelcolor (str,list): Color of the model points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.    
      xscale (str): The x-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.
      yscale (str): The y-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.

    Returns:
      (matplotlib.figure.Figure, list): Figure and list of axes handles.

    """

    # Create local copy so that we can modify it
    resdata_local = copy.deepcopy(resdata)
    
    
    # Apply station list flagging
    if (not station_list is None) :
        if (not 'triangle' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include triangles")
        if (not isinstance(station_list,list)) :
            station_list = [station_list]

        station_include_list = []
        station_exclude_list = []
        for station in station_list :
            if (station[0]=='!') :
                station_exclude_list.append(station[1:])
            else :
                station_include_list.append(station)


        keep = np.array([False]*len(resdata_local['triangle']))

        for j in range(len(resdata_local['triangle'])) :
            station1,station2,station3 = _station_codes_from_triangle(resdata_local['triangle'][j])
            
            if (len(station_include_list)==0) :
                keep[j] = True
            else :
                keep[j] = ((station1 in station_include_list) or (station2 in station_include_list) or (station3 in station_include_list))

            if (len(station_exclude_list)>0) :
                keep[j] = keep[j] and not ((station1 in station_exclude_list) or (station2 in station_exclude_list) or (station3 in station_exclude_list))

        for key in ['time','u1','v1','u2','v2','u3','v3','data','model','residual','error'] :
            resdata_local[key] = resdata_local[key][keep]

    # Select coordinate
    if (plot_type=='perimeter') :
        x=np.sqrt(resdata_local['u1']**2+resdata_local['v1']**2+resdata_local['u2']**2+resdata_local['v2']**2+resdata_local['u3']**2+resdata_local['v3']**2)
        xlbl=r'Perimeter (G$\lambda$)'
    elif (plot_type=='area') :
        u1amp = np.sqrt(resdata_local['u1']**2+resdata_local['v1']**2)
        u2amp = np.sqrt(resdata_local['u2']**2+resdata_local['v2']**2)
        u1du2 = resdata_local['u1']*resdata_local['u2']+resdata_local['v1']*resdata_local['v2']
        x = 0.5*np.sqrt( (u1amp*u2amp)**2 - u1du2**2 )
        xlbl=r'Area (G$\lambda^2$)'
    elif (plot_type=='uvmax') :
        u1amp = np.sqrt(resdata_local['u1']**2+resdata_local['v1']**2)
        u2amp = np.sqrt(resdata_local['u2']**2+resdata_local['v2']**2)
        u3amp = np.sqrt(resdata_local['u3']**2+resdata_local['v3']**2)
        x = np.maximum(u1amp,np.maximum(u2amp,u3amp))
        xlbl=r'$|u|_{max}$ (G$\lambda$)'
    elif (plot_type=='uvmin') :
        u1amp = np.sqrt(resdata_local['u1']**2+resdata_local['v1']**2)
        u2amp = np.sqrt(resdata_local['u2']**2+resdata_local['v2']**2)
        u3amp = np.sqrt(resdata_local['u3']**2+resdata_local['v3']**2)
        x = np.minimum(u1amp,np.minimum(u2amp,u3amp))
        xlbl=r'$|u|_{min}$ (G$\lambda$)'
    elif (plot_type=='umax') :
        x=np.maximum(resdata_local['u1'],np.maximum(resdata_local['u2'],resdata_local['u3']))
        xlbl=r'$u_{max}$ (G$\lambda$)'
    elif (plot_type=='umin') :
        x=np.minimum(resdata_local['u1'],np.minimum(resdata_local['u2'],resdata_local['u3']))
        xlbl=r'$u_{min}$ (G$\lambda$)'
    elif (plot_type=='vmax') :
        x=np.maximum(resdata_local['v1'],np.maximum(resdata_local['v2'],resdata_local['v3']))
        xlbl=r'$v_{max}$ (G$\lambda$)'
    elif (plot_type=='vmin') :
        x=np.minimum(resdata_local['v1'],np.minimum(resdata_local['v2'],resdata_local['v3']))
        xlbl=r'$v_{min}$ (G$\lambda$)'
    elif (plot_type=='time') :
        if (not 'time' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include time")
        x=resdata_local['time']
        xlbl=r'$t$ (UTC)'
    elif (plot_type=='snr') :
        x=np.abs(resdata_local['model']/resdata_local['error'])
        xlbl=r'$S/N$'
    else :
        raise RuntimeError("Unrecognized plot type %s"%(plot_type))

    # Check if a residual distribution is desired and set defaults
    if (resdist is None) :
        resdist_numbins=2
        resdist = False
    else :
        resdist_numbins=resdist
        resdist = True
    
    # Create figure and axes objects
    if (residuals) :
        if (not resdist) :
            fig = plt.figure(figsize=[6.,5.])
            axs_res = plt.axes([0.15,0.10,0.83,0.25])
            axs_comp = plt.axes([0.15,0.38,0.83,0.57])
            axs_list = [axs_comp,axs_res]
        else :
            fig = plt.figure(figsize=[6.5,5.])
            axs_res = plt.axes([0.1385,0.10,0.7662,0.25])
            axs_comp = plt.axes([0.1385,0.38,0.7662,0.57])
            axs_resdist = plt.axes([0.915,0.10,0.07,0.25])
            axs_list = [axs_comp,axs_res,axs_resdist]
            
    else :
        fig = plt.figure(figsize=[6.,5.])
        axs_comp = plt.axes([0.15, 0.10, 0.83, 0.83])
        axs_list = [axs_comp]

    # Plot the data comparision
    plt.sca(axs_comp)
    plt.errorbar(x,resdata_local['data'],yerr=resdata_local['error'],fmt=datafmt,color=datacolor,markersize=4,zorder=10)
    plt.plot(x,resdata_local['model'],modelfmt,color=modelcolor,markersize=2,zorder=20)
    plt.ylabel(r'Closure Phase (deg)')
    plt.grid(grid)
    axs_comp.set_xscale(xscale)
    axs_comp.set_yscale(yscale)

    # Plot the residuals if desired
    if (residuals) :
        axs_comp.xaxis.set_ticklabels([])
        plt.sca(axs_res)
        # Sigma guides
        plt.axhline(0,linestyle='-',color='r')
        plt.axhline(1.0,linestyle=':',color='r')
        plt.axhline(-1.0,linestyle=':',color='r')
        # plot the markers
        plt.plot(x,resdata_local['residual']/resdata_local['error'],datafmt,color=datacolor,markersize=4)
        plt.ylabel(r'Res.')
        plt.grid(grid)
        axs_res.set_xscale(xscale)

        if (resdist) :
            rr = resdata_local['residual']/resdata_local['error']
            ylim = axs_res.get_ylim()
            plt.sca(axs_resdist)
            xtmp = np.linspace(ylim[0],ylim[1],256)
            ytmp = np.exp(-xtmp**2/2.0)/np.sqrt(2*np.pi)
            plt.plot(ytmp,xtmp,'-r',alpha=0.5)
            var = np.var(rr)
            mean = np.average(rr)
            ytmp = np.exp(-(xtmp-mean)**2/(2.0*var))/np.sqrt(2*np.pi*var)
            plt.plot(ytmp,xtmp,':g',alpha=0.5)
            bins=np.arange(int(np.floor(ylim[0]*resdist_numbins)),int(np.ceil(ylim[1]*resdist_numbins)))/resdist_numbins
            plt.hist(rr,bins=bins,density=True,orientation="horizontal",color=datacolor,alpha=0.5)
            plt.ylim(ylim)
            plt.gca().yaxis.set_ticklabels([])
            plt.gca().xaxis.set_ticklabels([])            
            plt.grid(grid)
            plt.sca(axs_res)
            
    # Add the xlabels
    plt.xlabel(xlbl)

    return plt.gcf(),axs_list



def plot_log_closure_amplitude_residuals(resdata, plot_type='perimeter', gain_data=None, station_list=None, residuals=True, resdist=2, resdist_numbins=2, datafmt='o', datacolor='b', modelfmt='.', modelcolor='r', grid=True, xscale='linear', yscale='linear') :
    """
    Plots comparison between the log closure amplitudes from the data and model in a variety of possible formats.

    Args:
      resdata (dict): Dictionary object containing the residual data as generated, e.g., by :func:`read_residuals`.
      plot_type (str): Type of residual plot to generate. Options are 'perimeter', 'arear', 'uvmax', 'uvmin', 'umax', 'umin', 'vmax', 'vmin', 'time', 'snr'. Default: 'perimeter'.
      station_list (list,str): Station or list of stations to either exclude or restrict the residual plot to. Requires resdata to contain a key 'baselines'.  Station names prepended with '!' will be excluded.  If any station names without '!' are given, will show *only* those stations. Default: None.
      residuals (bool): If True produces a sub-panel with the error-weighted residuals plotted underneath the comparison plot. Default: True.
      resdist (int): If not None, produces a sub-panel with the distribution of residuals compared to a unit-variance Gaussian. If an int value is passed, it will set the number of bins per unit standard deviation. Default: 2.
      datafmt (str): Format specifier for the data points. Default: 'o'.
      datacolor (str,list): Color of the data points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'b'.
      modelfmt (str): Format specifier for the model points. Default: '.'.
      modelcolor (str,list): Color of the model points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.    
      xscale (str): The x-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.
      yscale (str): The y-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.

    Returns:
      (matplotlib.figure.Figure, list): Figure and list of axes handles.

    """

    # Create local copy so that we can modify it
    resdata_local = copy.deepcopy(resdata)
    
    
    # Apply station list flagging
    if (not station_list is None) :
        if (not 'quadrangle' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include quadrangles")
        if (not isinstance(station_list,list)) :
            station_list = [station_list]

        station_include_list = []
        station_exclude_list = []
        for station in station_list :
            if (station[0]=='!') :
                station_exclude_list.append(station[1:])
            else :
                station_include_list.append(station)


        keep = np.array([False]*len(resdata_local['quadrangle']))

        for j in range(len(resdata_local['quadrangle'])) :
            station1,station2,station3,station4 = _station_codes_from_quadangle(resdata_local['quadangle'][j])
            
            if (len(station_include_list)==0) :
                keep[j] = True
            else :
                keep[j] = ((station1 in station_include_list) or (station2 in station_include_list) or (station3 in station_include_list) or (station4 in station_include_list))

            if (len(station_exclude_list)>0) :
                keep[j] = keep[j] and not ((station1 in station_exclude_list) or (station2 in station_exclude_list) or (station3 in station_exclude_list) or (station4 in station_exclude_list))

        for key in ['time','u1','v1','u2','v2','u3','v3','u4','v4','data','model','residual','error'] :
            resdata_local[key] = resdata_local[key][keep]

    # Select coordinate
    if (plot_type=='perimeter') :
        x=np.sqrt(resdata_local['u1']**2+resdata_local['v1']**2+resdata_local['u2']**2+resdata_local['v2']**2+resdata_local['u3']**2+resdata_local['v3']**2+resdata_local['u4']**2+resdata_local['v4']**2)
        xlbl=r'Perimeter (G$\lambda$)'
    elif (plot_type=='area') :
        raise RuntimeError("area is not yet implmented for log closure amplitude quadrangles.")
        # u1amp = np.sqrt(resdata_local['u1']**2+resdata_local['v1']**2)
        # u2amp = np.sqrt(resdata_local['u2']**2+resdata_local['v2']**2)
        # u1du2 = resdata_local['u1']*resdata_local['u2']+resdata_local['v1']*resdata_local['v2']
        # x = 0.5*np.sqrt( (u1amp*u2amp)**2 - u1du2**2 )
        # xlbl=r'Area (G$\lambda^2$)'
    elif (plot_type=='uvmax') :
        u1amp = np.sqrt(resdata_local['u1']**2+resdata_local['v1']**2)
        u2amp = np.sqrt(resdata_local['u2']**2+resdata_local['v2']**2)
        u3amp = np.sqrt(resdata_local['u3']**2+resdata_local['v3']**2)
        u4amp = np.sqrt(resdata_local['u4']**2+resdata_local['v4']**2)
        x = np.maximum(u1amp,np.maximum(u2amp,np.maximum(u3amp,u4amp)))
        xlbl=r'$|u|_{max}$ (G$\lambda$)'
    elif (plot_type=='uvmin') :
        u1amp = np.sqrt(resdata_local['u1']**2+resdata_local['v1']**2)
        u2amp = np.sqrt(resdata_local['u2']**2+resdata_local['v2']**2)
        u3amp = np.sqrt(resdata_local['u3']**2+resdata_local['v3']**2)
        u4amp = np.sqrt(resdata_local['u4']**2+resdata_local['v4']**2)
        x = np.minimum(u1amp,np.minimum(u2amp,np.minimum(u3amp,u4amp)))
        xlbl=r'$|u|_{min}$ (G$\lambda$)'
    elif (plot_type=='umax') :
        x=np.maximum(resdata_local['u1'],np.maximum(resdata_local['u2'],np.maximum(resdata_local['u3'],resdata_local['u4'])))
        xlbl=r'$u_{max}$ (G$\lambda$)'
    elif (plot_type=='umin') :
        x=np.minimum(resdata_local['u1'],np.minimum(resdata_local['u2'],np.minimum(resdata_local['u3'],resdata_local['u4'])))
        xlbl=r'$u_{min}$ (G$\lambda$)'
    elif (plot_type=='vmax') :
        x=np.maximum(resdata_local['v1'],np.maximum(resdata_local['v2'],np.maximum(resdata_local['v3'],resdata_local['v4'])))
        xlbl=r'$v_{max}$ (G$\lambda$)'
    elif (plot_type=='vmin') :
        x=np.minimum(resdata_local['v1'],np.minimum(resdata_local['v2'],np.minimum(resdata_local['v3'],resdata_local['v4'])))
        xlbl=r'$v_{min}$ (G$\lambda$)'
    elif (plot_type=='time') :
        if (not 'time' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include time")
        x=resdata_local['time']
        xlbl=r'$t$ (UTC)'
    elif (plot_type=='snr') :
        x=np.abs(resdata_local['model']/resdata_local['error'])
        xlbl=r'$S/N$'
    else :
        raise RuntimeError("Unrecognized plot type %s"%(plot_type))

    # Check if a residual distribution is desired and set defaults
    if (resdist is None) :
        resdist_numbins=2
        resdist = False
    else :
        resdist_numbins=resdist
        resdist = True
    
    # Create figure and axes objects
    if (residuals) :
        if (not resdist) :
            fig = plt.figure(figsize=[6.,5.])
            axs_res = plt.axes([0.15,0.10,0.83,0.25])
            axs_comp = plt.axes([0.15,0.38,0.83,0.57])
            axs_list = [axs_comp,axs_res]
        else :
            fig = plt.figure(figsize=[6.5,5.])
            axs_res = plt.axes([0.1385,0.10,0.7662,0.25])
            axs_comp = plt.axes([0.1385,0.38,0.7662,0.57])
            axs_resdist = plt.axes([0.915,0.10,0.07,0.25])
            axs_list = [axs_comp,axs_res,axs_resdist]
            
    else :
        fig = plt.figure(figsize=[6.,5.])
        axs_comp = plt.axes([0.15, 0.10, 0.83, 0.83])
        axs_list = [axs_comp]

    # Plot the data comparision
    plt.sca(axs_comp)
    plt.errorbar(x,resdata_local['data'],yerr=resdata_local['error'],fmt=datafmt,color=datacolor,markersize=4,zorder=10)
    plt.plot(x,resdata_local['model'],modelfmt,color=modelcolor,markersize=2,zorder=20)
    plt.ylabel(r'Log Closure Amplitude')
    plt.grid(grid)
    axs_comp.set_xscale(xscale)
    axs_comp.set_yscale(yscale)

    # Plot the residuals if desired
    if (residuals) :
        axs_comp.xaxis.set_ticklabels([])
        plt.sca(axs_res)
        # Sigma guides
        plt.axhline(0,linestyle='-',color='r')
        plt.axhline(1.0,linestyle=':',color='r')
        plt.axhline(-1.0,linestyle=':',color='r')
        # plot the markers
        plt.plot(x,resdata_local['residual']/resdata_local['error'],datafmt,color=datacolor,markersize=4)
        plt.ylabel(r'Res.')
        plt.grid(grid)
        axs_res.set_xscale(xscale)

        if (resdist) :
            rr = resdata_local['residual']/resdata_local['error']
            ylim = axs_res.get_ylim()
            plt.sca(axs_resdist)
            xtmp = np.linspace(ylim[0],ylim[1],256)
            ytmp = np.exp(-xtmp**2/2.0)/np.sqrt(2*np.pi)
            plt.plot(ytmp,xtmp,'-r',alpha=0.5)
            var = np.var(rr)
            mean = np.average(rr)
            ytmp = np.exp(-(xtmp-mean)**2/(2.0*var))/np.sqrt(2*np.pi*var)
            plt.plot(ytmp,xtmp,':g',alpha=0.5)
            bins=np.arange(int(np.floor(ylim[0]*resdist_numbins)),int(np.ceil(ylim[1]*resdist_numbins)))/resdist_numbins
            plt.hist(rr,bins=bins,density=True,orientation="horizontal",color=datacolor,alpha=0.5)
            plt.ylim(ylim)
            plt.gca().yaxis.set_ticklabels([])
            plt.gca().xaxis.set_ticklabels([])            
            plt.grid(grid)
            plt.sca(axs_res)
            
    # Add the xlabels
    plt.xlabel(xlbl)

    return plt.gcf(),axs_list


def plot_visibility_residuals(resdata, plot_type='uvamp|complex', gain_data=None, station_list=None, residuals=True, resdist=2, datafmt='o', datacolor='b', modelfmt='.', modelcolor='r', grid=True, xscale='linear', yscale='linear', alpha=0.5) :
    """
    Plots comparison between the visibility amplitudes from the data and model in a variety of possible formats.

    Args:
      resdata (dict): Dictionary object containing the residual data as generated, e.g., by :func:`read_residuals`.
      plot_type (str): Type of residual plot to generate specified via 'xtype|ytype'. Options are xtype are 'uvamp', 'u', 'v', 'time', 'amplitude', 'snr'. Options for ytype are 'complex', 'amplitude', 'phase'. If only one type specifier is given, will attempt to intelligently interpret it, breaking ties by assigning it to the ytype. Default: 'uvamp|complex'.
      gain_data (dict): Gain data object as generated, e.g., by :func:`read_gain_file`. If provided, data will be calibrated prior to residuals being plotted. Default: None.
      station_list (list,str): Station or list of stations to either exclude or restrict the residual plot to. Requires resdata to contain a key 'baselines'.  Station names prepended with '!' will be excluded.  If any station names without '!' are given, will show *only* those stations. Default: None.
      residuals (bool): If True produces a sub-panel with the error-weighted residuals plotted underneath the comparison plot. Default: True.
      resdist (int): If not None, produces a sub-panel with the distribution of residuals compared to a unit-variance Gaussian. If an int value is passed, it will set the number of bins per unit standard deviation. Default: 2.
      datafmt (str): Format specifier for the data points. Default: 'o'.
      datacolor (str,list): Color of the data points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'b'.
      modelfmt (str): Format specifier for the model points. Default: '.'.
      modelcolor (str,list): Color of the model points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.    
      xscale (str): The x-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.
      yscale (str): The y-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.
      alpha (float): Alpha value for points. Default: 0.5.

    Returns:
      (matplotlib.figure.Figure, list): Figure and list of axes handles.

    """

    # Create local copy so that we can modify it
    resdata_local = copy.deepcopy(resdata)
    
    
    # Apply gains
    if (not gain_data is None) :
        if (not 'time' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include time")

        tstart = gain_data['tstart']
        tend = gain_data['tend']
        toffset = ((gain_data['toffset'].mjd)%1)*24
        dayoffset = np.min(resdata_local['day'])

        for j in range(len(resdata_local['baseline'])) :
            station1,station2 = _station_codes_from_baseline(resdata_local['baseline'][j])

            dt = resdata_local['time'][j] + 24*(resdata_local['day'][j]-dayoffset) - toffset
            current_epoch = (gain_data['tstart']<dt)*(dt<=gain_data['tend'])
            GA = gain_data[station1][current_epoch]
            GB = gain_data[station2][current_epoch]
            
            for key in ['data','model','residual'] :
                resdata_local[key][j] = (1.0/GA) * (1.0/np.conj(GB)) * resdata_local[key][j]
            resdata_local['error'][j] = np.abs((1.0/GA) * (1.0/np.conj(GB))) * resdata_local['error'][j]            

    # Apply station list flagging
    if (not station_list is None) :
        if (not 'baseline' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include baselines")
        if (not isinstance(station_list,list)) :
            station_list = [station_list]

        station_include_list = []
        station_exclude_list = []
        for station in station_list :
            if (station[0]=='!') :
                station_exclude_list.append(station[1:])
            else :
                station_include_list.append(station)

        keep = np.array([False]*len(resdata_local['baseline']))
        for j in range(len(resdata_local['baseline'])) :
            station1,station2 = _station_codes_from_baseline(resdata_local['baseline'][j])
            
            if (len(station_include_list)==0) :
                keep[j] = True
            else :
                keep[j] = ((station1 in station_include_list) or (station2 in station_include_list))

            if (len(station_exclude_list)>0) :
                keep[j] = keep[j] and not ((station1 in station_exclude_list) or (station2 in station_exclude_list))

                
        for key in ['time','u','v','data','model','residual','error'] :
            resdata_local[key] = resdata_local[key][keep]
            
    
    # Determine the plot type to generate
    toks = plot_type.split('|')
    if (len(toks)>1) :
        plot_type_x = toks[0]
        plot_type_y = toks[1]
    elif ( toks[0] in ['complex','amplitude','phase'] ) :
        plot_type_x = 'uvamp'
        plot_type_y = toks[0]
    else :
        plot_type_x = toks[0]
        plot_type_y = 'complex'
            
            
    # Select coordinate
    if (plot_type_x=='uvamp') :
        x=np.sqrt(resdata_local['u']**2+resdata_local['v']**2)
        xlbl=r'$|u|$ (G$\lambda$)'
    elif (plot_type_x=='u') :
        x=resdata_local['u']
        xlbl=r'$u$ (G$\lambda$)'
    elif (plot_type_x=='v') :
        x=resdata_local['v']
        xlbl=r'$v$ (G$\lambda$)'
    elif (plot_type_x=='time') :
        if (not 'time' in resdata_local.keys()) :
            raise RuntimeError("Residual data does not include time")
        x=resdata_local['time']
        xlbl=r'$t$ (UTC)'
    elif (plot_type_x=='amplitude') :
        x=np.abs(resdata_local['model'])
        xlbl=r'$|V|$ (Jy)'
    elif (plot_type_x=='snr') :
        x=np.abs(resdata_local['data']/resdata_local['error'])
        xlbl=r'$S/N$'        
    else :
        raise RuntimeError("Unrecognized plot type %s"%(plot_type_x))

    # Check if a residual distribution is desired and set defaults
    if (resdist is None) :
        resdist_numbins=2
        resdist = False
    else :
        resdist_numbins=resdist
        resdist = True
    
    # Create figure and axes objects
    if (residuals) :
        if (not resdist) :
            fig = plt.figure(figsize=[6.,5.])
            axs_res = plt.axes([0.15,0.10,0.83,0.25])
            axs_comp = plt.axes([0.15,0.38,0.83,0.57])
            axs_list = [axs_comp,axs_res]
        else :
            fig = plt.figure(figsize=[6.5,5.])
            axs_res = plt.axes([0.1385,0.10,0.7662,0.25])
            axs_comp = plt.axes([0.1385,0.38,0.7662,0.57])
            axs_resdist = plt.axes([0.915,0.10,0.07,0.25])
            axs_list = [axs_comp,axs_res,axs_resdist]
            
    else :
        fig = plt.figure(figsize=[6.,5.])
        axs_comp = plt.axes([0.15, 0.10, 0.83, 0.83])
        axs_list = [axs_comp]

    # Plot the data comparision
    plt.sca(axs_comp)

    if (plot_type_y=='complex') :
        plt.errorbar(x,resdata_local['data'].real,yerr=resdata_local['error'].real,fmt=datafmt,color=datacolor,markersize=4,alpha=alpha,zorder=10,label=r'Re($V$)')
        plt.errorbar(x,resdata_local['data'].imag,yerr=resdata_local['error'].imag,fmt=datafmt,color=datacolor,markersize=4,alpha=alpha,zorder=10,fillstyle='none',label=r'Im($V$)')
        plt.plot(x,resdata_local['model'].real,modelfmt,color=modelcolor,markersize=2,zorder=20)
        plt.plot(x,resdata_local['model'].imag,modelfmt,color=modelcolor,markersize=2,zorder=20)
        plt.ylabel(r'$V$ (Jy)')
        plt.legend()
    elif (plot_type_y=='amplitude') :
        plt.errorbar(x,np.abs(resdata_local['data']),yerr=np.abs(resdata_local['error']),fmt=datafmt,color=datacolor,markersize=4,alpha=alpha,zorder=10)
        plt.plot(x,np.abs(resdata_local['model']),modelfmt,color=modelcolor,markersize=2,zorder=20)
        plt.ylabel(r'$|V|$ (Jy)')
    elif (plot_type_y=='phase') :
        plt.errorbar(x,180.0/np.pi*np.angle(resdata_local['data']),yerr=180.0/np.pi*np.abs(resdata_local['error'])/np.abs(resdata_local['data']),fmt=datafmt,color=datacolor,markersize=4,alpha=alpha,zorder=10)
        plt.plot(x,180.0/np.pi*np.angle(resdata_local['model']),modelfmt,color=modelcolor,markersize=2,zorder=20)
        plt.ylabel(r'arg($V$) (deg)')
        plt.ylim((-200,200))
    else :
        raise RuntimeError("Unrecognized plot type %s"%(plot_type_y))
        
    plt.grid(grid)
    axs_comp.set_xscale(xscale)
    axs_comp.set_yscale(yscale)


    # Plot the residuals if desired
    if (residuals) :
        axs_comp.xaxis.set_ticklabels([])
        plt.sca(axs_res)
        # Sigma guides
        plt.axhline(0,linestyle='-',color='r')
        plt.axhline(1.0,linestyle=':',color='r')
        plt.axhline(-1.0,linestyle=':',color='r')
        # plot the markers
        if (plot_type_y=='complex') :
            res = np.append(resdata_local['residual'].real/resdata_local['error'].real,resdata_local['residual'].imag/resdata_local['error'].imag)
            plt.plot(x,resdata_local['residual'].real/resdata_local['error'].real,datafmt,color=datacolor,markersize=4,alpha=alpha)
            plt.plot(x,resdata_local['residual'].imag/resdata_local['error'].imag,datafmt,color=datacolor,markersize=4,alpha=alpha,fillstyle='none')
        elif (plot_type_y=='amplitude') :
            res = ( np.abs(resdata_local['model']+resdata_local['residual']) - np.abs(resdata_local['model']) )/( np.abs(resdata_local['error']) )
            plt.plot(x,res,datafmt,color=datacolor,markersize=4,alpha=alpha)
        elif (plot_type_y=='phase') :
            res = ( np.angle((resdata_local['model']+resdata_local['residual'])/resdata_local['model']) )/( np.abs(resdata_local['error'])/np.abs(resdata_local['data']) )
            plt.plot(x,res,datafmt,color=datacolor,markersize=4,alpha=alpha)
        plt.ylabel(r'Res.')
        plt.grid(grid)
        axs_res.set_xscale(xscale)


        if (resdist) :
            rr = res
            ylim = axs_res.get_ylim()
            plt.sca(axs_resdist)
            xtmp = np.linspace(ylim[0],ylim[1],256)
            ytmp = np.exp(-xtmp**2/2.0)/np.sqrt(2*np.pi)
            plt.plot(ytmp,xtmp,'-r',alpha=0.5)
            var = np.var(rr)
            mean = np.average(rr)
            ytmp = np.exp(-(xtmp-mean)**2/(2.0*var))/np.sqrt(2*np.pi*var)
            plt.plot(ytmp,xtmp,':g',alpha=0.5)
            bins=np.arange(int(np.floor(ylim[0]*resdist_numbins)),int(np.ceil(ylim[1]*resdist_numbins)))/resdist_numbins
            plt.hist(rr,bins=bins,density=True,orientation="horizontal",color=datacolor,alpha=0.5)
            plt.ylim(ylim)
            plt.gca().yaxis.set_ticklabels([])
            plt.gca().xaxis.set_ticklabels([])            
            plt.grid(grid)
            plt.sca(axs_res)

        
    # Add the xlabels
    plt.xlabel(xlbl)

    return plt.gcf(),axs_list



def plot_crosshand_residuals(resdata, plot_type='uvamp|complex', crosshand='all', gain_data=None, dterms=None, station_list=None, residuals=True, resdist=2, datafmt='o', datacolor='b', modelfmt='.', modelcolor='r', grid=True, xscale='linear', yscale='linear', alpha=0.5) :
    """
    Plots comparison between the visibility amplitudes from the data and model in a variety of possible formats.

    Args:
      resdata (dict): Dictionary object containing the residual data as generated, e.g., by :func:`read_residuals`.
      crosshand (str,list): Quantity or list of quantities to plot.  Recognized options are 'RR', 'LL', 'RL', 'LR', or a list of combinations thereof; 'I', 'Q', 'U', 'V', or a list of combinations thereof; 'all' which is a synonym for ['RR', 'LL', 'RL', 'LR']; or 'Stokes' which is a synonym for ['I', 'Q', 'U', 'V']. Default: 'all'.
      plot_type (str): Type of residual plot to generate specified via 'xtype|ytype'. Options are xtype are 'uvamp', 'u', 'v', 'time', 'amplitude'. Options for ytype are 'complex', 'amplitude', 'phase'. If only one type specifier is given, will attempt to intelligently interpret it, breaking ties by assigning it to the ytype. Default: 'uvamp|complex'.
      gain_data (dict): Gain data object as generated, e.g., by :func:`read_gain_file`. If provided, data will be calibrated prior to residuals being plotted. Default: None.
      dterms (dict): List of D terms as returned from :func:`dterms` in :class:`model_polarized_image`.
      station_list (list,str): Station or list of stations to either exclude or restrict the residual plot to. Requires resdata to contain a key 'baselines'.  Station names prepended with '!' will be excluded.  If any station names without '!' are given, will show *only* those stations. Default: None.
      residuals (bool): If True produces a sub-panel with the error-weighted residuals plotted underneath the comparison plot. Default: True.
      resdist (int): If not None, produces a sub-panel with the distribution of residuals compared to a unit-variance Gaussian. If an int value is passed, it will set the number of bins per unit standard deviation. Default: 2.
      datafmt (str): Format specifier for the data points. Default: 'o'.
      datacolor (str,list): Color of the data points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'b'.
      modelfmt (str): Format specifier for the model points. Default: '.'.
      modelcolor (str,list): Color of the model points in acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.    
      xscale (str): The x-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.
      yscale (str): The y-axis scaling. May be specified via any value accepted by :func:`matplotlib.axes.Axes.set_xscale`.  Default: 'linear'.
      alpha (float): Alpha value for points. Default: 0.5.

    Returns:
      (matplotlib.figure.Figure, list): Figure and list of axes handles.

    """

    # Create local copy so that we can modify it
    resdata_local = copy.deepcopy(resdata)
    
    if (not isinstance(crosshand,list)) :
        if (crosshand=='all') :
            crosshand = ['RR','LL','RL','LR']
        elif (crosshand=='Stokes') :
            crosshand = ['I', 'Q', 'U', 'V']
        else :
            crosshand = [crosshand]

    # Apply D terms if provided
    if (not dterms is None) :
        for j in range(resdata_local['u'].size) :
            stationA,stationB = _station_codes_from_baseline(resdata_local['baseline'][j])

            # Construct inverse D-term matrix for station A
            DAR = dterms[stationA][0]*np.exp(2.0j*resdata_local['field rotation 1'][j])
            DAL = dterms[stationA][1]*np.exp(-2.0j*resdata_local['field rotation 1'][j])
            iDA = np.matrix([[ 1.0+0.0j, -DAR ], [ -DAL, 1.0+0.0j ]]) / (1.0-DAR*DAL)
            
            # Construct inverse D-term matrix for station B
            DBR = dterms[stationB][0]*np.exp(2.0j*resdata_local['field rotation 2'][j])
            DBL = dterms[stationB][1]*np.exp(-2.0j*resdata_local['field rotation 2'][j])
            iDB = np.matrix([[ 1.0+0.0j, -DBR ], [ -DBL, 1.0+0.0j ]]) / (1.0-DBR*DBL)

            for key in ['data','model','residual'] :

                # Package as Jones matrix
                XAB = np.matrix([[ resdata_local[key]['RR'][j], resdata_local[key]['RL'][j] ], \
                                 [ resdata_local[key]['LR'][j], resdata_local[key]['LL'][j] ]])

                # Apply inverse D-terms
                XAB = iDA*XAB*(iDB.H)

                # Unpack back to crosshand elements
                resdata_local[key]['RR'][j] = XAB[0,0]
                resdata_local[key]['RL'][j] = XAB[0,1]
                resdata_local[key]['LR'][j] = XAB[1,0]
                resdata_local[key]['LL'][j] = XAB[1,1]

            # Generate new error estimates
            var = np.matrix([[ resdata_local['error']['RR'][j], resdata_local['error']['RL'][j] ], \
                             [ resdata_local['error']['LR'][j], resdata_local['error']['LL'][j] ]])
            for i in [0,1] :
                for k in [0,1] :
                    var[i,k] = (var[i,k].real)**2 + 1.0j*(var[i,k].imag)**2
            varnew = np.matrix([[0.0j,0.0j],[0.0j,0.0j]])
            X = np.matrix([[0.0j,0.0j],[0.0j,0.0j]])
            for i in [0,1] :
                for k in [0,1] :
                    X[i,k] = 1.0
                    Xnew =iDA*X*(iDB.H)
                    varnew[i,k] = varnew[i,k] \
                                  + (Xnew[i,k].real)**2*var[i,k].real + (Xnew[i,k].imag)**2*var[i,k].imag \
                                  + ((Xnew[i,k].real)**2*var[i,k].imag + (Xnew[i,k].imag)**2*var[i,k].real)*1.0j
                    X[i,k] = 0.0
                    
            for i in [0,1] :
                for k in [0,1] :
                    varnew[i,k] = np.sqrt(varnew[i,k].real) + 1.0j*np.sqrt(varnew[i,k].imag)
                    
            resdata_local['error']['RR'][j] = varnew[0,0]
            resdata_local['error']['RL'][j] = varnew[0,1]
            resdata_local['error']['LR'][j] = varnew[1,0]
            resdata_local['error']['LL'][j] = varnew[1,1]

    # Generate Stokes I,Q,U,V if requested
    if (not set(['I','Q','U','V']).isdisjoint(set(crosshand))) :
        for key in ['data','model','residual'] :
            resdata_local[key]['I'] = resdata_local[key]['RR']+resdata_local[key]['LL']
            resdata_local[key]['Q'] = resdata_local[key]['RL']+resdata_local[key]['LR']
            resdata_local[key]['U'] = (resdata_local[key]['RL']-resdata_local[key]['LR'])/1.0j
            resdata_local[key]['V'] = resdata_local[key]['RR']-resdata_local[key]['LL']

        resdata_local['error']['I'] = np.sqrt( (resdata_local['error']['RR'].real)**2 + (resdata_local['error']['LL'].real)**2 ) \
                               + 1.0j*np.sqrt( (resdata_local['error']['RR'].imag)**2 + (resdata_local['error']['LL'].imag)**2 )
        resdata_local['error']['Q'] = np.sqrt( (resdata_local['error']['RL'].real)**2 + (resdata_local['error']['LR'].real)**2 ) \
                               + 1.0j*np.sqrt( (resdata_local['error']['RL'].imag)**2 + (resdata_local['error']['LR'].imag)**2 )
        resdata_local['error']['U'] = np.sqrt( (resdata_local['error']['RL'].imag)**2 + (resdata_local['error']['LR'].imag)**2 ) \
                               + 1.0j*np.sqrt( (resdata_local['error']['RL'].real)**2 + (resdata_local['error']['LR'].real)**2 )
        resdata_local['error']['V'] = np.sqrt( (resdata_local['error']['RR'].real)**2 + (resdata_local['error']['LL'].real)**2 ) \
                               + 1.0j*np.sqrt( (resdata_local['error']['RR'].imag)**2 + (resdata_local['error']['LL'].imag)**2 )

        
    # Create copy for plotting visibilities
    resdata_vis = {}
    resdata_vis['type']='visibilities'
    keylist1 = ['source','year','day','time','baseline','u','v']
    keylist2 = ['data','error','model','residual']
    for key in keylist1+keylist2 :
        resdata_vis[key]=np.array([])
    for q in crosshand :
        for key in keylist1 :
            if (key in resdata_local.keys()) :
                resdata_vis[key]=np.append(resdata_vis[key],resdata_local[key])
        for key in keylist2 :
            resdata_vis[key]=np.append(resdata_vis[key],resdata_local[key][q])

    return plot_visibility_residuals(resdata_vis,plot_type=plot_type, gain_data=gain_data, station_list=station_list, residuals=residuals, resdist=resdist, datafmt=datafmt, datacolor=datacolor, modelfmt=modelfmt, modelcolor=modelcolor, grid=grid, xscale=xscale, yscale=yscale, alpha=alpha)
    

