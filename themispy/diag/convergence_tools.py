###########################
#
# Package:
#   convergence_tools
#
# Provides:
#   Functions for computing convergence diagonstics of MCMC chains produced by Themis.
#   



import numpy as np
import scipy.stats as stats
import warnings

from themispy import chain


"""
The autocorrelation stuff has been taken from EMCEE article on IACT
for the ensemble samplers. In the future I will probably switch to the IACT formula from BDA3.
"""
def next_pow_two(n):
    """
    Returns the value of 2**n, following `emcee <https://emcee.readthedocs.io/en/stable/tutorials/autocorr>`_.

    Args:
      n (int): Power of 2
    
    Returns:
      (int): Value of 2**n
    """
    
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    """
    Computes the 1D autocorrelation function of a sequences, following 
    `emcee <https://emcee.readthedocs.io/en/stable/tutorials/autocorr>`_.  
    This is done via FFT, and thus assumes that the sequences is stationary.

    Args:
      x (numpy.ndarray): A sequence of values.
      norm (bool): If true normalizes the autocorrelation by the variance. Otherwise the zero-lag component of the autocorrelation function is the variance. Default: True.

    Returns:
      (numpy.ndarra): Autocorrlation function appropriately normalized.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function", x.shape)
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    """
    Automatic windowing procedure following `emcee <https://emcee.readthedocs.io/en/stable/tutorials/autocorr>`_, which follows `Sokal (1989) <https://pdfs.semanticscholar.org/0bfe/9e3db30605fe2d4d26e1a288a5e2997e7225.pdf>`_.

    Args:
      taus (numpy.ndarray): Estimates of the integrated autocorrelation time (IACT).
      c (float): Sokal factor, with an optimal value approximately of 5.0.  Default: 5.0.

    Returns:
      (int): Index of minimum window for computing the autocorrelation time?
    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def ensemble_autocorr(param_chain, c=5.0):
    """
    Computes the integrated autocorrelation time for the chain, following `emcee <https://emcee.readthedocs.io/en/stable/tutorials/autocorr>`_.

    Args:
      param_chain (numpy.ndarray): Ensemble MCMC chain for a single parameters, arranged as walkers,steps.
      c (float): Sokal factor, with an optimal value approximately of 5.0.  Default: 5.0.

    Returns:
      (float): Integrated autocorrelation time (IACT).
    """
    f = np.zeros(param_chain.shape[1])
    for yy in param_chain:
        f += autocorr_func_1d(yy)
    f /= len(param_chain)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    iac = taus[window]
    if (iac*50 > param_chain.shape[1]):
        print("WARNING: In order to get decent estimate of the integrated autocorrelation time (IAC), IAC*50 < number of steps. You should really run the chain longer!")
    return iac

def mean_ess(y, c=5.0):
    """
    Calculates the average ESS across chains by dividing the number of steps by the IAC.
    Note the actual ESS of the entire chain and walkers is greater than this, but likely isn't
    the number of walkers * mean_ess since the chains are correlated.

    Args:
      y (numpy.ndarray): Ensemble MCMC chain for a single parameters, arranged as walkers,steps.

    Returns:
      (float): Average number of ensemble samples across chains.
    """
    return y.shape[1]/ensemble_autocorr(y,c)

def mean_echain(echain):
    """
    Computes the average of the ensemble chain for each step. That is, 
    we average over the walkers at each step converting the chain into a
    single particle chain of the average.

    Args:
      echain (numpy.ndarray): Ensemble MCMC chain, generated, e.g., from :func:`chain.mcmc_chain.read_echain`.

    Returns:
      (numpy.ndarray): Chain data arranged as 2D array indexed by [sample, parameter] *after* averaging over walkers.
    """
    return np.mean(echain, axis=1)

def var_echain(echain):
    """
    Computes the variance of the echain for each step. That is, we compute the
    variance of the walkers at each step, converting the chain into a single 
    particle chain of the variance between chains.

    Args:
      echain (numpy.ndarray): Ensemble chain data, generated, e.g., from :func:`chain.mcmc_chain.read_echain`.

    Returns:
      (numpy.ndarray): Variance of chain data arranged as 2D array indexed by [sample, parameter] *after* computing the variance over walkers.
    """
    return np.var(echain, axis=1, ddof=1)

def ranknormalized_chains(chains):
    """
    Computes the rank-normalized chains, where the rank is taken over all the chains.
    This allows :math:`\\hat{R}` to be used when we the distribution might not have a finite
    mean or variance.

    Returns the rank normalized chains, i.e. the z-scores and the actual ranks.

    Args:
      chains (list): List of chains, stored as 2D arrays indexed by [sample,parameter].

    Returns:
      (numpy.ndarray, numpy.ndarray): Array of probability densities associated with given rank; Array of ranks for each parameter value.  Both are provided in the *same* order as the original data.
    """
    chainsary = np.asarray(chains)
    z = []
    ranks = []
    for i in range(chainsary.shape[-1]):
        size = chainsary[:,:,i].size
        rank = stats.rankdata(chainsary[:,:,i], method="average")
        zz = stats.norm.ppf((rank - 3.0/8)/(size - 0.25))
        zz = zz.reshape(chainsary.shape[0:2])
        z.append(zz)
        ranks.append(rank.reshape(chainsary.shape[0:2]))
    return np.transpose(np.array(z), (1,2,0)), np.transpose(np.array(ranks), (1,2,0))


def fold_chains(chains):
    """
    Returns the chain folded around the median of all the chains, i.e., for each
    parameter :math:`p`, every chain sample is replaced with :math:`|p-\\bar{p}|` 
    where :math:`\\bar{p}` is the median across all chains.
    This helps with measuring :math:`\\hat{R}` in the tails.

    Args:
      chains (numpy.ndarray): A single walker chain arranged as a 3D array indexed by [runs, samples, parameters].

    Returns:
      (numpy.ndarray): Folded single walker chain arranged as 3D array index by [runs, samples, parameters].
    """
    assert chains.ndim==3, "Chains must have 3 dimensions not "+str(chains.ndim)+". If ndim=4, you are probably passing an ensemble chain. In that case, please take the expectation over the walkers."
    flatchains = np.reshape(chains, (np.prod(chains.shape[:-1]), chains.shape[-1]))
    median = np.median(flatchains, axis=0)
    return np.reshape(np.abs(flatchains-median), chains.shape)


def bulk_split_rhat(chains):
    """
    Computes the bulk :math:`\\hat{R}` using rank normalized chains. The returned value is the 
    maximum of the folded :math:`\\hat{R}` and regular rank normalized :math:`\\hat{R}` as recommended in 
    `Vehtari (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190308008V/abstract>`_.

    Args:
      chains (numpy.ndarray): A single walker chain arranged as a 3D array indexed by [runs, samples, parameters].

    Returns:
      (numpy.ndarray): An array of values of the split :math:`\\hat{R}` statistic from `Gelman (2013) <http://www.stat.columbia.edu/~gelman/book/>`_ for each parameter in the chain using rank-normalized values.
    """
    
    assert chains.ndim==3, "Chains must have 3 dimensions not "+str(chains.ndim)+". If ndim=4, you are probably passing an ensemble chain. In that case, please take the expectation over the walkers."
    z,_ = ranknormalized_chains(chains)
    zfold,_ = ranknormalized_chains(fold_chains(chains))

    return np.maximum(split_rhat(z), split_rhat(zfold))


def split_rhat(chains):
    """
    Computes the split :math:`\\hat{R}` statistic from 
    `Gelman (2013) <http://www.stat.columbia.edu/~gelman/book/>`_. This is able to 
    deal with distributions that don't have finite mean and/or variance. Additionally, 
    it splits the chain in two in order to check if the beginning and end of the chain 
    have similar variance to check for intrachain convergence.

    Args:
      chains (numpy.ndarray): A single walker chain arranged as a 3D array indexed by [runs, samples, parameters].

    Returns:
      (numpy.ndarray): An array of values of the split :math:`\\hat{R}` statistic from `Gelman (2013) <http://www.stat.columbia.edu/~gelman/book/>`_ for each parameter in the chain.
    """

    assert chains.ndim==3, "Chains must have 3 dimensions not "+str(chains.ndim)+". If ndim=4, you are probably passing an ensemble chain. In that case, please take the expectation over the walkers."
    chains = np.array(chains)
    chain_maj = np.array_split(chains,2,axis=1)
    chain_maj = np.vstack(chain_maj)

    return _rhat(chain_maj)


def _rhat(chains):
    """
    Utility function
    Computes the :math:`\\hat{R}` between the chains. This only assumes that we are using a single
    chain. If using an ensemble method please first convert it into a single chain by 
    taking the expectation of a function, i.e. a mean, across the walkers.
    """ 
    M,N = chains.shape[0:2]
    #Average over each chain, so we have the average for each chain
    avg_n = np.mean(chains,axis=1)
    #Average over both
    avg_nm = np.mean(avg_n,axis=0)
    #Now calculate the between chain variance
    B = N/(M-1)*(np.sum(avg_n*avg_n,axis=0) - avg_nm*avg_nm*M)

    #Now calculate the within chain variance
    s2m = 1.0/(N-1)*(np.sum(chains*chains,axis=1) - avg_n*avg_n*N)
    W = 1.0/M*np.sum(s2m,axis=0)
    
    marginal_var = (N-1)/N*W + B/N

    return np.sqrt(marginal_var/W)


def find_eautocorr(echain):
    """
    Finds and returns the integrated autocorrelation time (IACT) for each parameter 
    in the ensemble chain.

    Args:
      echain (numpy.ndarray): Ensemble MCMC chain, generated, e.g., from :func:`chain.mcmc_chain.read_echain`.

    Returns:
      (numpy.ndarray): Array of autocorrelation times for each parameter.
    """
    
    iact = np.zeros(echain.shape[2])
    for i in range(echain.shape[2]):
        iact[i] = ensemble_autocorr(echain[:,:,i].T)
    return iact


def save_ensemble_diagnostics(out_file_name, chains, chainsname, method="rank", stride=1):
    """
    Outputs the diagnostics of the multiple chains in `chains` to the out file, 
    where the chain is kept track of in `chainsname`. Currently it only outputs 
    the chains integrated autocorrelation time for each parameter. Eventually 
    other diagnostics will be included, such as the 
    `Gelman-Rubin <https://ui.adsabs.harvard.edu/abs/1992StaSc...7..457G/abstract>`_ 
    diagnostic, i.e. split-:math:`\\hat{R}`, and ESS. Currently the default method for 
    :math:`\\hat{R}` is the rank method from
    `Vehtari (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190308008V/abstract>`_.

    Note that if the integrated autocorrelation time (IACT) is "significantly different" 
    between chains, then at least one of the chains is not converged.

    Args:
      out_file_name (str): Name of file to which to output results.
      chains (list): List of chains for which to compute diagnostics.
      chainsname (list): List of names (str) of chains for output.
      method (str): CURRENTLY UNUSED
      stride (int): Factor by which the chains have been thinned by.  Default: 1.
    """

    print("outputting to", out_file_name)
    io =  open(out_file_name, "w")
    io.write("IACT:\n")
    row_fmt = ("{:>15}"*(chains[0].shape[2])+" {:<25}")
    io.write(row_fmt.format(*map(lambda x: "p"+str(x),range(chains[0].shape[2])), "filename"))
    io.write("\n")
    for i in range(len(chains)):
        iact = find_eautocorr(chains[i])*stride
        print("IACT: ", iact, chainsname[i])
        io.write(row_fmt.format(*list(np.round(iact,2)), chainsname[i]))
        io.write("\n")
    io.write("Split-Rhat: \n")
    meanchains = np.mean(chains,axis=2)
    varchains = np.var(chains,axis=2)
    mean_rhat = bulk_split_rhat(meanchains)
    var_rhat = bulk_split_rhat(varchains)
    print("bulk Rhat mean: ", mean_rhat)
    print("bulk Rhat var: ", var_rhat)
    row_fmt = ("{:>15}"*(mean_rhat.shape[0])+" {:<25}")
    io.write(row_fmt.format(*list(np.round(mean_rhat,6)), "mean"))
    io.write("\n")
    io.write(row_fmt.format(*list(np.round(var_rhat,6)), "variance"))
    io.close()

    return iact, mean_rhat, var_rhat





