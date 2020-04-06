###########################
#
# Package:
#   diagnostic_plots
#
# Provides:
#   Functions for diagonostic plots from MCMC chains produced by Themis.
#



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import is_color_like, ListedColormap, to_rgba
import scipy.stats as stats
import warnings
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy import interpolate as scint
from scipy.interpolate import PchipInterpolator as mcubic

from themispy import chain
from themispy import diag


def plot_rank_hist(chains, bins=30, names=None):
    """
    Computes the rank plots, where the rank is taken over all the runs. 
    The histograms should be roughly uniform. Any structure in them signals 
    that the chains aren't converged or mixing well, i.e. your Monte Carlo estimates
    will be biased.

    Args:
      chains (numpy.ndarray): Single parameter chains, stored as 2D arrays indexed by [run,sample].
      bins (int): Number of bins in which to segment data for plot.
      names (list): List of parameter names (str).  Default: None.

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    assert chains.ndim == 2, "Chains must have 2 dimensions not "+str(chains.ndim)
    ranks = stats.rankdata(chains.reshape(-1), method="average")
    ranks = ranks.reshape(chains.shape)
    bin_itr = np.linspace(ranks.min(), ranks.max(), bins)
    nplots = chains.shape[0]
    fig,axes = plt.subplots(nplots//2,2, figsize=(6,3*nplots//2),sharey=True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95, wspace=0.15, hspace=0.15)
    if (nplots>2):
        axes[0,0].set_title("Start of chain")
        axes[0,1].set_title("End of chain")
    else:
        axes[0].set_title("Start of chain")
        axes[1].set_title("End of chain")
    if (nplots>2):
        for i in range(nplots//2):
            ax = axes[i,0]
            ax.set_yticklabels([])
            ax.set_xlabel("Posterior Rank")
            ax.grid()
            ax.hist(ranks[i,:], bins=bin_itr, zorder=500)
            if (names is not None):
                ax.set_ylabel(names[i])
                    #ha="left", va="center")
            ax = axes[i,1]
            ax.set_yticklabels([])
            ax.set_xlabel("Posterior Rank")
            ax.grid()
            ax.hist(ranks[i+nplots//2,:], bins=bin_itr, zorder=500)
    else:
        ax = axes[0]
        ax.set_yticklabels([])
        ax.set_xlabel("Posterior Rank")
        ax.grid()
        ax.hist(ranks[0,:], bins=bin_itr, zorder=500)
        if (names is not None):
            ax.set_ylabel(names[0])
                #ha="left", va="center")
        ax = axes[1]
        ax.set_yticklabels([])
        ax.set_xlabel("Posterior Rank")
        ax.grid()
        ax.hist(ranks[1,:], bins=bin_itr, zorder=500)

    fig.tight_layout(rect=[0.0, 0.1, 1.0,0.9])
    return fig,axes



def plot_likelihood_trace(elklhd, colormap='plasma', step_norm=1000, grid=True, means=False, mean_color='k', alpha=0.5):
    """
    Plots to the current Axes object traces for a Themis-style ensemble likelihood 
    objects. Optionally overplots the means and standard deviations of the likelihood.

    Args:
      elklhd (numpy.ndarray): Ensemble MCMC chain likelihood, generated, e.g., from :func:`chain.mcmc_chain.read_elklhd`.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'plasma'.
      step_norm (float): Normalization of the steps to show on horizontal axis. Default: 1000.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      means (bool): Flag that determines whether or not to overplot means and standard devation ranges for the likelihood. Default: False.
      mean_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.
      alpha (float): Value of alpha for the individual likelihood traces.

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    chain_step = np.arange(elklhd.shape[0])/float(step_norm)

    # Get mean values if they are to be plotted
    if (means) :
        meanvals = np.mean(elklhd,axis=None)
        stdvals = np.std(elklhd,axis=None)
        
    cmap = cm.get_cmap(colormap)
        
    for w in range(elklhd.shape[1]) :
        plt.plot(chain_step,elklhd[:,w],'-',color=cmap(w/(elklhd.shape[1]-1.0)),alpha=alpha)

    if (means) :
        xtmp = np.array([chain_step[0], chain_step[-1]])
        ytmp = np.array([meanvals, meanvals])
        plt.plot(xtmp,ytmp,linestyle='-',alpha=0.75,color=mean_color,zorder=11)
        plt.fill_between(xtmp,ytmp+stdvals,ytmp-stdvals,alpha=0.25,color=mean_color,zorder=10)

    plt.grid(grid)
    plt.gca().set_xlabel('Sample number / %g'%(step_norm))
    plt.gca().set_ylabel('Log likelihood')

    return plt.gcf(),plt.gca()
        

def plot_likelihood_trace_list(elklhd_list, colormap='plasma', step_norm=1000, grid=True, means=False, mean_color='k', alpha=0.5):
    """
    Plots to the current Axes object traces for a list of Themis-style ensemble 
    likelihood objects. Optionally overplots the means and standard deviations 
    of the likelihood.

    Args:
      elklhd_list (list): List of ensemble MCMC chain likelihoods, generated, e.g., from :func:`chain.mcmc_chain.read_elklhd`.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'plasma'.
      step_norm (float): Normalization of the steps to show on horizontal axis. Default: 1000.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      means (bool): Flag that determines whether or not to overplot means and standard devation ranges for each parameter. Default: False.
      mean_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.
      alpha (float): Value of alpha for the individual likelihood traces.

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """


    step_offset = 0
    
    for elklhd in elklhd_list :
        
        chain_step = np.arange(elklhd.shape[0])/float(step_norm) + step_offset

        # Get mean values if they are to be plotted
        if (means) :
            meanvals = np.mean(elklhd,axis=None)
            stdvals = np.std(elklhd,axis=None)
        
        cmap = cm.get_cmap(colormap)
        
        for w in range(elklhd.shape[1]) :
            plt.plot(chain_step,elklhd[:,w],'-',color=cmap(w/(elklhd.shape[1]-1.0)),alpha=alpha)

        if (means) :
            xtmp = np.array([chain_step[0], chain_step[-1]])
            ytmp = np.array([meanvals, meanvals])
            plt.plot(xtmp,ytmp,linestyle='-',alpha=0.75,color=mean_color,zorder=11)
            plt.fill_between(xtmp,ytmp+stdvals,ytmp-stdvals,alpha=0.25,color=mean_color,zorder=10)


            
        plt.axvline(step_offset,color='k')

        step_offset = chain_step[-1]
            
            
    plt.grid(grid)
    plt.gca().set_xlabel('Sample number / %g'%(step_norm))
    plt.gca().set_ylabel('Log likelihood')
            
    return plt.gcf(),plt.gca()


def plot_parameter_trace(echain, parameter_list=None, parameter_names=None, sample_color='b', one_column=False, step_norm=1000, grid=True, means=False, mean_color='r'):
    """
    Generates a set of trace plots for a Themis-style ensemble chain object.  
    Optionally overplots the means and standard deviations for each parameter.

    Args:
      echain (numpy.ndarray): Ensemble MCMC chain, generated, e.g., from :func:`chain.mcmc_chain.read_echain`.
      parameter_list (list): List of parameter columns (zero-offset) to read in. Default: None, which reads all parameters.
      parameter_names (list): List of strings of parameter names.  Must be the same length as parameter_list. Default: None.
      sample_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`
      one_column (bool): Sets the plots to be arranged as a single column. Default: False which generates as square a grid as possible.
      step_norm (float): Normalization of the steps to show on horizontal axis. Default: 1000.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      means (bool): Flag that determines whether or not to overplot means and standard devation ranges for each parameter. Default: False.
      mean_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    # If no parameter list is passed, plot everything
    if (parameter_list is None):
        parameter_list = np.arange(echain.shape[2])
    parameter_list = np.array(parameter_list)

    # If no parameter name list is passed, create generic name list
    if (parameter_names is None):
        parameter_names=[]
        for p in parameter_list :
            parameter_names.extend(['p%i'%p])
            
    # Select relevant part of echain
    echain = echain[:,:,parameter_list]
    
    # Make plot details
    if (one_column) :
        Nwy=len(parameter_list)
        Nwx=1
    else :
        Nwx=int(np.sqrt(float(len(parameter_list))))
        Nwy=len(parameter_list)//Nwx
        if (Nwy*Nwx<len(parameter_list)) :
            Nwy += 1
    fig,ax_list=plt.subplots(Nwy,Nwx,sharex=True,figsize=(4*Nwx,2*Nwy),squeeze=False)

    chain_step = np.arange(echain.shape[0]*echain.shape[1])/float(step_norm*echain.shape[1])
    echain = echain.reshape([-1,echain.shape[2]])

    # Get mean values if they are to be plotted
    if (means) :
        meanvals = np.mean(echain,axis=0)
        stdvals = np.std(echain,axis=0)
    
    for ip in range(len(parameter_list)) :
        plt.sca(ax_list[ip//Nwx,ip%Nwx])
        plt.plot(chain_step,echain[:,ip],',',color=sample_color)
        plt.grid(grid)
        plt.gca().set_ylabel(parameter_names[ip])
        if (means) :
            plt.axhline(meanvals[ip],linestyle='-',alpha=0.5,color=mean_color,zorder=11)
            plt.axhspan(meanvals[ip]-stdvals[ip],meanvals[ip]+stdvals[ip],linestyle=None,alpha=0.1,color=mean_color,zorder=10)
            
    for iwx in range(Nwx) :
        ax_list[-1,iwx].set_xlabel('Sample number / %g'%(step_norm))

    fig.tight_layout()

    return fig,ax_list
        



def plot_parameter_trace_list(echain_list, parameter_list=None, parameter_names=None, sample_color='b', one_column=False, step_norm=1000, grid=True, means=False, mean_color='r'):
    """
    Plots to the current Axes object traces of a subset of parameters for a list of 
    Themis-style ensemble chains that are presumably temporally related in some fashion.

    Args:
      echain_list (list): List of ensemble MCMC chain, generated, e.g., from :func:`chain.mcmc_chain.read_echain`.
      parameter_list (list): List of parameter columns (zero-offset) to read in. Default: None, which reads all parameters.
      parameter_names (list): List of strings of parameter names.  Must be the same length as parameter_list. Default: None.
      sample_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`
      one_column (bool): Sets the plots to be arranged as a single column. Default: False which generates as square a grid as possible.
      step_norm (float): Normalization of the steps to show on horizontal axis. Default: 1000.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      means (bool): Flag that determines whether or not to overplot means and standard devation ranges for each parameter. Default: False.
      mean_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'r'.

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    # If no parameter list is passed, plot everything
    if (parameter_list is None):
        parameter_list = np.arange(echain_list[0].shape[2])
    parameter_list = np.array(parameter_list)

    # If no parameter name list is passed, create generic name list
    if (parameter_names is None):
        parameter_names=[]
        for p in parameter_list :
            parameter_names.extend(['p%i'%p])
            
    # Select relevant part of echain
    for echain in echain_list :
        echain = echain[:,:,parameter_list]

    # Make plot details
    if (one_column) :
        Nwy=len(parameter_list)
        Nwx=1
    else :
        Nwx=int(np.sqrt(float(len(parameter_list))))
        Nwy=len(parameter_list)//Nwx
        if (Nwy*Nwx<len(parameter_list)) :
            Nwy += 1
    fig,ax_list=plt.subplots(Nwy,Nwx,sharex=True,figsize=(4*Nwx,2*Nwy),squeeze=False)

    step_offset = 0

    for echain in echain_list :
    
        chain_step = np.arange(echain.shape[0]*echain.shape[1])/float(step_norm*echain.shape[1]) + step_offset
        echain = echain.reshape([-1,echain.shape[2]])

        # Get mean values if they are to be plotted
        if (means) :
            meanvals = np.mean(echain,axis=0)
            stdvals = np.std(echain,axis=0)
    
        for ip in range(len(parameter_list)) :
            plt.sca(ax_list[ip//Nwx,ip%Nwx])
            plt.plot(chain_step,echain[:,ip],',',color=sample_color)
            plt.grid(grid)
            if (means) :
                xtmp = np.array([chain_step[0], chain_step[-1]])
                ytmp = np.array([meanvals[ip], meanvals[ip]])
                plt.plot(xtmp,ytmp,linestyle='-',alpha=0.75,color=mean_color,zorder=11)
                plt.fill_between(xtmp,ytmp+stdvals[ip],ytmp-stdvals[ip],alpha=0.25,color=mean_color,zorder=10)
            plt.axvline(step_offset,color='k')

        step_offset = chain_step[-1]

                
    for ip in range(len(parameter_list)) :
        plt.sca(ax_list[ip//Nwx,ip%Nwx])
        plt.gca().set_ylabel(parameter_names[ip])

    for iwx in range(Nwx) :
        ax_list[-1,iwx].set_xlabel('Sample number / %g'%(step_norm))

    fig.tight_layout()

    return fig,ax_list
        

def _expand_bool_array(bv) :
    """
    Expands by one element the truth elements in a boolean array.  That is, if the input is [False,False,True,True,False] return [False,True,True,True,True].
    """
    
    # If the successive elment is true, make this element true
    bv[:-1] = bv[:-1]+bv[1:]

    # If the prior element is true, make this element true
    bv[1:] = bv[1:] + bv[:-1]

    return bv


def plot_annotated_parameter_trace(echain, elklhd, parameter_list=None, parameter_names=None, likelihood_values=4, colormap='plasma', one_column=False, step_norm=1000, grid=True, means=False, mean_color='k', add_likelihood_trace=False):
    """
    Plots traces of a subset of parameters for a Themis-style ensemble chain object. 
    Points are color-coded by likelihood, which provides a better visualizations of 
    where the high-probability regions are.

    Args:
      echain (numpy.ndarray): Ensemble MCMC chain, generated, e.g., from :func:`chain.mcmc_chain.load_erun`.
      elklhd (numpy.ndarray): Ensemble MCMC chain likelihood, generated, e.g., from :func:`chain.mcmc_chain.load_erun`.
      parameter_list (list): List of parameter columns (zero-offset) to read in. Default: None, which reads all parameters.
      parameter_names (list): List of strings of parameter names.  Must be the same length as parameter_list. Default: None.
      likelihood_values (int,list): Likelihoods at which to generate likelihood steps. If an int is passed, this is set to the 5*j for j in [0,n-1). Default: 4.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'plasma'.
      one_column (bool): Sets the plots to be arranged as a single column. Default: False which generates as square a grid as possible.
      step_norm (float): Normalization of the steps to show on horizontal axis. Default: 1000.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      means (bool): Flag that determines whether or not to overplot means and standard devation ranges for each parameter. Default: False.
      mean_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'k'.
      add_likelihood_trace (bool): Flag to determine if to add a trace plot of the likelihood. Default: False.

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    # If no parameter list is passed, plot everything
    if (parameter_list is None):
        parameter_list = np.arange(echain.shape[2])
    parameter_list = np.array(parameter_list)

    # If no parameter name list is passed, create generic name list
    if (parameter_names is None):
        parameter_names=[]
        for p in parameter_list :
            parameter_names.extend(['p%i'%p])
            
    # Select relevant part of echain
    echain = echain[:,:,parameter_list]

    # Set the likelihood levels    
    if (isinstance(likelihood_values,int)) :
        likelihood_values = np.max(elklhd) - 5*(1+np.arange(likelihood_values))
    # Order from min to max so that high likelihood values over-plot low values
    iLorder = np.argsort(likelihood_values)
    likelihood_values = likelihood_values[iLorder]
    # Set colors
    cmap  = cm.get_cmap(colormap)
    likelihood_color_indexes = np.linspace(0,1.0,len(likelihood_values))
    likelihood_colors = [ cmap(x) for x in likelihood_color_indexes ]

    # Make plot details
    number_of_windows = len(parameter_list)
    if (add_likelihood_trace) :
        number_of_windows += 1
    if (one_column) :
        Nwy=number_of_windows
        Nwx=1
    else :
        Nwx=int(np.sqrt(float(number_of_windows)))
        Nwy=number_of_windows//Nwx
        if (Nwy*Nwx<number_of_windows) :
            Nwy += 1
    fig,ax_list=plt.subplots(Nwy,Nwx,sharex=True,figsize=(4*Nwx,2*Nwy),squeeze=False)

    chain_step = np.arange(echain.shape[0]*echain.shape[1])/float(step_norm*echain.shape[1])
    echain = echain.reshape([-1,echain.shape[2]])
    elklhd_by_walker = np.copy(elklhd)
    elklhd = elklhd.reshape([-1])
    
    # Get mean values if they are to be plotted
    if (means) :
        meanvals = np.mean(echain,axis=0)
        stdvals = np.std(echain,axis=0)
    
    for ip in range(len(parameter_list)) :
        plt.sca(ax_list[ip//Nwx,ip%Nwx])

        # Below bottom
        use = (elklhd<likelihood_values[0])
        plt.plot(chain_step[use],echain[use,ip],',',color=[0.7,0.7,0.7])

        # Middle values
        for k in range(len(likelihood_values)-1) :
            use = (elklhd>=likelihood_values[k])*(elklhd<likelihood_values[k+1])
            plt.plot(chain_step[use],echain[use,ip],'.',ms=2,color=likelihood_colors[k],alpha=0.5)

        # Top value
        use = (elklhd>=likelihood_values[-1])
        plt.plot(chain_step[use],echain[use,ip],'.',ms=4,color=likelihood_colors[-1],alpha=1)
            
        plt.grid(grid)
        plt.gca().set_ylabel(parameter_names[ip])

        if (means) :
            plt.axhline(meanvals[ip],linestyle='-',alpha=0.5,color=mean_color,zorder=11)
            plt.axhspan(meanvals[ip]-stdvals[ip],meanvals[ip]+stdvals[ip],linestyle=None,alpha=0.25,color=mean_color,zorder=10)


    if (add_likelihood_trace) :

        plt.sca(ax_list[(number_of_windows-1)//Nwx,(number_of_windows-1)%Nwx])

        chain_step_by_walker = np.arange(elklhd_by_walker.shape[0])/float(step_norm)

        
        for w in range(elklhd_by_walker.shape[1]) :

            color_fluctuation_fraction = 1.0 - 0.5*np.random.rand()

            x = np.linspace(chain_step_by_walker[0],chain_step_by_walker[-1],16*chain_step_by_walker.size)
            y = np.interp(x,chain_step_by_walker,elklhd_by_walker[:,w])

            color_list = np.zeros((len(x),4))
            color_list[(y<likelihood_values[0]),:] = np.array([0.7,0.7,0.7,1])*color_fluctuation_fraction
            for k in range(len(likelihood_values)-1) :
                color_list[(y>=likelihood_values[k])*(y<likelihood_values[k+1])] = np.array(likelihood_colors[k])*color_fluctuation_fraction
            color_list[(y>=likelihood_values[-1])] = np.array(likelihood_colors[-1])*color_fluctuation_fraction

            points = np.array([x, y]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1],points[1:]],axis=1)
            lc = LineCollection(segments,colors=color_list)
            lc.set_linewidth(1)
            lc.set_alpha(0.5)
            line = plt.gca().add_collection(lc)

            #plt.gca().set_xscale(auto=True)
            ymin = np.min(elklhd)
            ymax = np.max(elklhd)
            dy = 0.1*(ymax-ymin)
            plt.gca().set_ylim((ymin-dy,ymax+dy))

            plt.grid(grid)
            plt.gca().set_ylabel(r'$\\log_{10}(L)$')
            
    for iwx in range(Nwx) :
        ax_list[-1,iwx].set_xlabel('Sample number / %g'%(step_norm))

    fig.tight_layout()

    return fig,ax_list
    

def plot_annotated_parameter_trace_list(echain_list, elklhd_list, parameter_list=None, parameter_names=None, likelihood_values=4, colormap='plasma', one_column=False, step_norm=1000, grid=True, means=False, mean_color='k', use_global_likelihoods=False, add_likelihood_trace=False):
    """
    Plots traces of a subset of parameters for a list of Themis-style ensemble chains 
    that are presumably temporally related in some fashion. Points are color-coded by 
    likelihood, which provides a better visualizations of where the high-probability 
    regions are.

    Args:
      echain_list (list): List of ensemble MCMC chain, generated, e.g., from :func:`chain.mcmc_chain.load_erun`.
      elklhd_list (list): List of ensemble MCMC chain likelihoods, generated, e.g., from :func:`chain.mcmc_chain.load_erun`.
      parameter_list (list): List of parameter columns (zero-offset) to read in. Default: None, which reads all parameters.
      parameter_names (list): List of strings of parameter names.  Must be the same length as parameter_list. Default: None.
      likelihood_values (int,list): Likelihoods at which to generate likelihood steps. If an int is passed, this is set to the 5*j for j in [0,n-1). Default: 4.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'plasma'.
      one_column (bool): Sets the plots to be arranged as a single column. Default: False which generates as square a grid as possible.
      step_norm (float): Normalization of the steps to show on horizontal axis. Default: 1000.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      means (bool): Flag that determines whether or not to overplot means and standard devation ranges for each parameter. Default: False.
      mean_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'k'.
      use_global_likelihoods (bool): Flag to determine if the color coding will be done for each element in the list individually or across all chaings in the list.  Default: False.
      add_likelihood_trace (bool): Flag to determine if to add a trace plot of the likelihood. Default: False.

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    # If no parameter list is passed, plot everything
    if (parameter_list is None):
        parameter_list = np.arange(echain_list[0].shape[2])
    parameter_list = np.array(parameter_list)

    # If no parameter name list is passed, create generic name list
    if (parameter_names is None):
        parameter_names=[]
        for p in parameter_list :
            parameter_names.extend(['p%i'%p])
            
    # Select relevant part of echain
    for echain in echain_list :
        echain = echain[:,:,parameter_list]

    # Save the likelihood values that were passed
    likelihood_values_passed = likelihood_values
    if (use_global_likelihoods) :
        # Get the max over-all likelihood
        lmax = np.max(elklhd_list[0])
        for elklhd in elklhd_list :
            lmax = max(lmax,np.max(elklhd))
        # Set the likelihood levels
        if (isinstance(likelihood_values_passed,int)) :
            likelihood_values_passed = np.array(lmax - 5*(1+np.arange(likelihood_values_passed)))
            
    # Make plot details
    number_of_windows = len(parameter_list)
    if (add_likelihood_trace) :
        number_of_windows += 1
    if (one_column) :
        Nwy=number_of_windows
        Nwx=1
    else :
        Nwx=int(np.sqrt(float(number_of_windows)))
        Nwy=number_of_windows//Nwx
        if (Nwy*Nwx<number_of_windows) :
            Nwy += 1
    fig,ax_list=plt.subplots(Nwy,Nwx,sharex=True,figsize=(4*Nwx,2*Nwy),squeeze=False)

    step_offset = 0

    for ic in range(len(echain_list)) :

        echain = echain_list[ic]
        elklhd = elklhd_list[ic]
        
        # Set the likelihood levels
        if (isinstance(likelihood_values_passed,int)) :
            likelihood_values = np.max(elklhd) - 5*(1+np.arange(likelihood_values_passed))
        else :
            likelihood_values = np.copy(likelihood_values_passed)
        # Order from min to max so that high likelihood values over-plot low values
        iLorder = np.argsort(likelihood_values)
        likelihood_values = likelihood_values[iLorder]
        # Set colors
        cmap  = cm.get_cmap(colormap)
        likelihood_colors = [ cmap(x) for x in np.linspace(0,1.0,len(likelihood_values)) ]

        chain_step = np.arange(echain.shape[0]*echain.shape[1])/float(step_norm*echain.shape[1]) + step_offset
        echain = echain.reshape([-1,echain.shape[2]])
        elklhd_by_walker = np.copy(elklhd)
        elklhd = elklhd.reshape([-1])
    
        # Get mean values if they are to be plotted
        if (means) :
            meanvals = np.mean(echain,axis=0)
            stdvals = np.std(echain,axis=0)
    
        for ip in range(len(parameter_list)) :
            plt.sca(ax_list[ip//Nwx,ip%Nwx])

            # Below bottom
            use = (elklhd<likelihood_values[0])
            plt.plot(chain_step[use],echain[use,ip],',',color=[0.7,0.7,0.7])

            # Middle values
            for k in range(len(likelihood_values)-1) :
                use = (elklhd>=likelihood_values[k])*(elklhd<likelihood_values[k+1])
                plt.plot(chain_step[use],echain[use,ip],'.',ms=2,color=likelihood_colors[k],alpha=0.5)

            # Top value
            use = (elklhd>=likelihood_values[-1])
            plt.plot(chain_step[use],echain[use,ip],'.',ms=4,color=likelihood_colors[-1],alpha=1)
            
            plt.grid(grid)
            if (means) :
                xtmp = np.array([chain_step[0], chain_step[-1]])
                ytmp = np.array([meanvals[ip], meanvals[ip]])
                plt.plot(xtmp,ytmp,linestyle='-',alpha=0.75,color=mean_color,zorder=11)
                plt.fill_between(xtmp,ytmp+stdvals[ip],ytmp-stdvals[ip],alpha=0.25,color=mean_color,zorder=10)
            plt.axvline(step_offset,color='k')

            
        if (add_likelihood_trace) :
            
            plt.sca(ax_list[(number_of_windows-1)//Nwx,(number_of_windows-1)%Nwx])

            chain_step_by_walker = np.arange(elklhd_by_walker.shape[0])/float(step_norm) + step_offset

        
            for w in range(elklhd_by_walker.shape[1]) :

                color_fluctuation_fraction = 1.0 - 0.5*np.random.rand()
                
                x = np.linspace(chain_step_by_walker[0],chain_step_by_walker[-1],16*chain_step_by_walker.size)
                y = np.interp(x,chain_step_by_walker,elklhd_by_walker[:,w])

                color_list = np.zeros((len(x),4))
                color_list[(y<likelihood_values[0]),:] = np.array([0.7,0.7,0.7,1])*color_fluctuation_fraction
                for k in range(len(likelihood_values)-1) :
                    color_list[(y>=likelihood_values[k])*(y<likelihood_values[k+1])] = np.array(likelihood_colors[k])*color_fluctuation_fraction
                color_list[(y>=likelihood_values[-1])] = np.array(likelihood_colors[-1])*color_fluctuation_fraction

                points = np.array([x, y]).T.reshape(-1,1,2)
                segments = np.concatenate([points[:-1],points[1:]],axis=1)
                lc = LineCollection(segments,colors=color_list)
                lc.set_linewidth(1)
                lc.set_alpha(0.5)
                line = plt.gca().add_collection(lc)
                    
                plt.grid(grid)

                plt.axvline(step_offset,color='k')

        step_offset = chain_step[-1]

    for ip in range(len(parameter_list)) :
        plt.sca(ax_list[ip//Nwx,ip%Nwx])
        plt.gca().set_ylabel(parameter_names[ip])

    if (add_likelihood_trace) :
        plt.sca(ax_list[(number_of_windows-1)//Nwx,(number_of_windows-1)%Nwx])
        ymin = np.min(elklhd)
        ymax = np.max(elklhd)
        dy = 0.1*(ymax-ymin)
        plt.gca().set_ylim((ymin-dy,ymax+dy))
        plt.gca().set_ylabel(r'$\\log_{10}(L)$')
        
    for iwx in range(Nwx) :
        ax_list[-1,iwx].set_xlabel('Sample number / %g'%(step_norm))

    fig.tight_layout()
        
    return fig,ax_list




def generate_streamline_colormap(colormap='plasma', number_of_streams=None, oversample_factor=4, fluctuation_factor=0.2) :
    """
    Generates a new colormap that is generated by randomly subsampling a given colormap.
    Useful for making plots that indicate flow, e.g., :func:`plot_deo_tempering_level_evolution`.
    
    Args:
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'plasma'.
    
    Returns:
      (matplotlib.colors.Colormap): A new colormap that has random rearrangements.
    """

    # Obtain colormap object
    cmap = cm.get_cmap(colormap)

    # Generate some value levels randomly bewteen 0 and 1
    if (number_of_streams is None) :
        number_of_streams = cmap.N
    vsub = np.linspace(0,1,number_of_streams) + fluctuation_factor*np.random.random(number_of_streams)
    vsub = (vsub-np.min(vsub))/(np.max(vsub)-np.min(vsub))

    # Generate an over-sampled smooth interpolated set of values
    v = scint.interp1d(np.linspace(0,1,number_of_streams),vsub,kind='cubic')(np.linspace(0,1,oversample_factor*number_of_streams))
    cvals = cmap(v)

    return ( ListedColormap(cvals) )




def plot_deo_tempering_level_evolution(beta, colormap='plasma', colormodel='tempering_level', alpha=0.5) :
    """
    Plots the tempering level with evolution with round for DEO tempered samplers in Themis.

    Args:
      beta (numpy.ndarray): :math:`\\beta` values indexed by [round,tempering level].
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'plasma'.
      colormodel (str): Determined how lines are colored.  Options are color type, 'tempering_level', 'density'.  Default: 'tempering_level'.
      alpha (float): Value of alpha for the individual tempering line curves.

    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """

    if (colormodel is 'density') :
        beta_density = np.zeros(beta.shape)
        dbdl = -(beta[:,1:]-beta[:,:-1])
        centered_density = 0.5*( 1.0/dbdl[:,1:] + 1.0/dbdl[:,:-1] )
        beta_density[:,0] = 1.0/dbdl[:,0]
        beta_density[:,1:-1] = centered_density
        beta_density[:,-1] = 1.0/dbdl[:,-1]
        beta_density = beta * beta_density # density of the log
        beta_density = beta_density / np.max(beta_density) # Normalize so peak is 1

        
    
    cmap = cm.get_cmap(colormap)
    
    round_indx = np.arange(beta.shape[0])
    for i in range(beta.shape[1]-1):
        if (colormodel is not 'density') :
            if (is_color_like(colormodel)) :
                color = colormodel
            elif (colormodel is 'tempering_level') :
                color = cmap(float(i)/float(beta.shape[1]-1))
            else :
                print('ERROR: Unknown colormodel %s.  See documentation for supported colormodel options.'%colormodel)
                quit()
            plt.plot(round_indx, beta[:,i], lw=1.0, color=color,alpha=alpha)
        else :
            plt.sca(plt.gca()) # Make sure an axis exists and grab a handle to it.
            x = np.linspace(round_indx[0],round_indx[-1],256)
            y = np.exp(np.interp(x,round_indx,np.log(beta[:,i])))
            norm = np.exp(np.interp(x,round_indx,np.log(beta_density[:,i])))
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap)
            lc.set_array(norm)
            lc.set_linewidth(1)
            lc.set_alpha(alpha)
            line = plt.gca().add_collection(lc)
            
            
    plt.gca().set_xlim((round_indx[0]-0.25,round_indx[-1]+0.25))
    betamin = np.log10(np.min(beta[:,:-1]))
    betamax = np.log10(np.max(beta[:,:-1]))
    plt.gca().set_ylim((10**(betamin-0.1*(betamax-betamin)),10**(betamax+0.1*(betamax-betamin))))
    plt.gca().set_yscale('log')
    plt.gca().set_xlabel(r"Round")
    plt.gca().grid(True)
    plt.gca().set_xticks(round_indx)
    plt.gca().set_ylabel(r"$\beta$")

    return plt.gcf(), plt.gca()

    
def plot_deo_rejection_rate(annealing_summary_data,color='b') :
    """
    Plots the rejection rate evolution with round for DEO tempered samplers in Themis.

    Args:
      annealing_summary_data (numpy.ndarray): Array of annealing run summary data as read by :func:`chain.mcmc_chain.load_deo_summary`.
      color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`
    
    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """
    
    
    plt.gca().errorbar(annealing_summary_data['round'], annealing_summary_data['AvgR'], yerr=annealing_summary_data['StdR'], capsize=2.0, color=color)
    plt.gca().set_ylabel(r"Average Rejection Rate")
    plt.gca().set_xlabel(r"Round")
    plt.gca().set_xticks(range(0,annealing_summary_data.shape[0]))
    plt.gca().grid()

    return plt.gcf(), plt.gca()


def plot_deo_lambda(annealing_summary_data,beta,R,colormap='b') :
    """
    Plots the rejection rate evolution with round for DEO tempered samplers in Themis.

    Args:
      annealing_summary_data (numpy.ndarray): Array of annealing run summary data as read by :func:`chain.mcmc_chain.load_deo_summary`.
      beta (numpy.ndarray): :math:`\\beta` values indexed by [round,tempering level], as read by :func:`chain.mcmc_chain.load_deo_summary`.
      R (nump.ndarray): :math:`R` values indexed by [round,tempering level], as read by :func:`chain.mcmc_chain.load_deo_summary`.
      color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`
    
    Returns:
      (matplotlib.figure.Figure, matplotlib.axes.Axes): Handles to the figure and array of axes objects in the plot.
    """


    if ( not is_color_like(colormap) ) :
        cmap = cm.get_cmap(colormap)

    lines = []
    nrounds = annealing_summary_data.shape[0]
    for i in range(nrounds):
        beta_r = beta[i,::-1]
        Lambda_r = np.cumsum(R[i,::-1])
        fLambda = mcubic(beta_r[1:], Lambda_r[1:])
        b = np.linspace(beta_r[1],1.0,500)
        if ( is_color_like(colormap) ) :
            color = list(to_rgba(colormap))
            color[3] = 0.05+0.95*i/(nrounds-1.0)
        else :
            color = cmap(i/(nrounds-1.0))
        im = plt.gca().semilogy(b, fLambda(b,nu=1), label="%d"%(i), color=color, linewidth=2)
        lines.append(im)
    #plt.gca().set_ylim(top=1e5,auto=True)
    plt.gca().set_xticks(np.arange(0.0,1.01,0.25))
    plt.gca().set_xlabel(r"$\beta$")
    plt.gca().set_ylabel(r"$\lambda(\beta)$")
    plt.gca().text(0.95,0.95, r"$\hat{\tau}_{\rm{opt, %d}} = %.4f$"%(nrounds-1,annealing_summary_data["rt_opt"][-1]),
               ha="right", va="center", transform=plt.gca().transAxes)
    plt.gca().text(0.95,0.85, r"$\hat{\tau}_{\rm{est, %d}} = %.4f$"%(nrounds-1,annealing_summary_data["rt_est"][-1]), 
               ha="right", va="center",transform=plt.gca().transAxes)

    plt.gca().grid(True)

    plt.gcf().legend(title="Round", bbox_to_anchor=(0.9,0.15,0.1,0.8),mode="expand", frameon=False)

    
    
    return plt.gcf(), plt.gca()




