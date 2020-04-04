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
