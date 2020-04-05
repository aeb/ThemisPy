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
import scipy.stats as stats
import warnings

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
        

def plot_annotated_parameter_trace(echain, elklhd, parameter_list=None, parameter_names=None, likelihood_values=4, colormap='plasma', one_column=False, step_norm=1000, grid=True, means=False, mean_color='k'):
    """
    Plots to the current Axes object traces of a subset of parameters for a Themis-style 
    ensemble chain object. Points are color-coded by likelihood, which provides a better
    visualizations of where the high-probability regions are.

    Args:
      echain (numpy.ndarray): Ensemble MCMC chain, generated, e.g., from :func:`chain.mcmc_chain.load_erun`.
      elklhd (numpy.ndarray): Ensemble MCMC chain likelihood, generated, e.g., from :func:`chain.mcmc_chain.load_erun`.
      parameter_list (list): List of parameter columns (zero-offset) to read in. Default: None, which reads all parameters.
      parameter_names (list): List of strings of parameter names.  Must be the same length as parameter_list. Default: None.
      likelihood_values (int,list): Likelihoods at which to generate likelihood steps. If an int is passed, this is set to the 5*j for j in [0,n-1). Default: 4.
      colormap (str): A colormap name as specified in :mod:`matplotlib.cm`. Default: plasma.
      one_column (bool): Sets the plots to be arranged as a single column. Default: False which generates as square a grid as possible.
      step_norm (float): Normalization of the steps to show on horizontal axis. Default: 1000.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      means (bool): Flag that determines whether or not to overplot means and standard devation ranges for each parameter. Default: False.
      mean_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'k'.

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
    likelihood_colors = [ cmap(x) for x in np.linspace(0,1.0,len(likelihood_values)) ]

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
            
    for iwx in range(Nwx) :
        ax_list[-1,iwx].set_xlabel('Sample number / %g'%(step_norm))

    fig.tight_layout()

    return fig,ax_list


def plot_annotated_parameter_trace_list(echain_list, elklhd_list, parameter_list=None, parameter_names=None, likelihood_values=4, colormap='plasma', one_column=False, step_norm=1000, grid=True, means=False, mean_color='k', use_global_likelihoods=False):
    """
    Plots to the current Axes object traces of a subset of parameters for a list of 
    Themis-style ensemble chains that are presumably temporally related in some fashion.
    Points are color-coded by likelihood, which provides a better visualizations of 
    where the high-probability regions are.

    Args:
      echain_list (list): List of ensemble MCMC chain, generated, e.g., from :func:`chain.mcmc_chain.load_erun`.
      elklhd_list (list): List of ensemble MCMC chain likelihoods, generated, e.g., from :func:`chain.mcmc_chain.load_erun`.
      parameter_list (list): List of parameter columns (zero-offset) to read in. Default: None, which reads all parameters.
      parameter_names (list): List of strings of parameter names.  Must be the same length as parameter_list. Default: None.
      likelihood_values (int,list): Likelihoods at which to generate likelihood steps. If an int is passed, this is set to the 5*j for j in [0,n-1). Default: 4.
      colormap (str): A colormap name as specified in :mod:`matplotlib.cm`. Default: plasma.
      one_column (bool): Sets the plots to be arranged as a single column. Default: False which generates as square a grid as possible.
      step_norm (float): Normalization of the steps to show on horizontal axis. Default: 1000.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      means (bool): Flag that determines whether or not to overplot means and standard devation ranges for each parameter. Default: False.
      mean_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'k'.
      use_global_likelihoods (bool): Flag to determine if the color coding will be done for each element in the list individually or across all chaings in the list.  Default: False.

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

            step_offset = chain_step[-1]


        for ip in range(len(parameter_list)) :
            plt.sca(ax_list[ip//Nwx,ip%Nwx])
            plt.gca().set_ylabel(parameter_names[ip])
            
        for iwx in range(Nwx) :
            ax_list[-1,iwx].set_xlabel('Sample number / %g'%(step_norm))

        fig.tight_layout()



        
    return fig,ax_list

