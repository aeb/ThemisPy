###########################
#
# Package:
#   mcmc_chain
#
# Provides:
#   Provides utility functions for reading and manipulating MCMC chains produced by Themis.
#

from themispy.utils import *

import numpy as np
import warnings
from matplotlib.cbook import flatten


"""
Collection of untilities for reading and manipulating MCMC chain data
produced by Themis analyses.
"""

def parse_parameter_arglist(arg_list) :
    """
    Parses a list of lists of strings containing individual numbers (e.g., 0 1 4), 
    comma separated lists (e.g., 0, 2, 5), ranges (e.g., 0-3), or combinations thereof,
    and reduces them to a unique, ordered list of integers.  Such a list of parameter
    values arises naturally from ArgumentParser append, e.g., from
    ``parser.add_argument("-p", type=str, nargs='+', action='append')``.

    Args:
      arg_list (list): List of lists with strings denoting indexes or ranges of indexes.
    
    Returns:
      (list): Sorted, unique list of integers.
    """

    plist = []
    for arg in flatten(arg_list) :
        tokens = arg.split(',')
        for token in tokens :
            if ('-' in token) : # range
                toks=token.split('-')
                plist.extend(list(range(int(toks[0]),int(toks[1])+1)))
            elif (token.isspace() or token=='') : # whitespace or empty
                continue
            else : # a number
                plist.extend([int(token)])
    plist = np.unique(np.sort(np.array(plist)))

    return list(plist)

    
    



def file_length(filename, header_string=None, comment_string=None) :
    """
    Counts the number of lines in a file.  Optionally will remove header lines, 
    marked by a prepended character, and comment lines, marked by a possibly 
    different prepended character.  These are differentiated by the presumed
    location within the file: Headers are at the beginning, ending at the first
    line to not be prepended with the header_string.  Comments can appear
    throughout the file.  In the absence of distributed comments, searching only
    for the header is significantly faster.

    Args:
      filename (str): Filename
      header_string (str): Single chararacter string that denotes header lines.  Default: None.
      comment_string (str): Single chararacter string that denotes header lines.  Default: None.

    Returns:
      (int[, int[, int]]): Number of non-header and non-commented lines; Number of header lines (only if header_string is not None); Number of commented lines (only if comment_string is not None).
    """
    
    length = 0
    nhead = 0
    ncomm = 0
    if (comment_string is None) and (header_string is None) :
        length=sum((1 for i in open(filename, 'r')))
        return length
    elif (comment_string is None) :
        length=sum((1 for i in open(filename, 'r')))
        for l in open(filename,'r') :
            if (l[0]==header_string) :
                length -= 1
                nhead += 1
            else :
                break
        return length, nhead
    elif (header_string is None) :
        for l in open(filename,'r') :
            if (l[0]==comment_string) :
                ncomm += 1
                continue
            length += 1
        return length, ncomm
    else :
        for l in open(filename,'r') :
            if (l[0]==header_string) :
                nhead += 1
                continue
            if (l[0]==comment_string) :
                ncomm += 1
                continue
            length += 1
        return length, nhead, ncomm


def read_elklhd(filename, stride=1, burn_fraction=0, skip=None, auto_warmup=False, sampler_type=None):
    """
    Reads in a Themis likelihood file. Optionally, a stride, burn-in fraction, or number of 
    *ensemble samples* to skip may be set. Returns an IndexError if fewer ensemble samples 
    exist in the chain file than skip.

    Args:
      filename (str): Filename in which likelihood data will be found (e.g., `Lklhd.dat`)
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      auto_warmup (bool): If true it will automatically sense the Stan warmup period for Stan state files and cut it. `burn_fraction` then applies to the state file with warmup already removed.
      skip (int): Number of initial *ensemble samples* to skip.  Overrides burn_fraction unless set to None. Default: None.
      sampler_type (str): Likelihood file format specifier.  If None, looks for a format at the top of the likelihood file; if not found, defaults to 'ensemble'.  Default: None.

    Returns:
      (numpy.ndarray, int): Likelihood data arranged as 2D array indexed by [sample, walker] *after* excluding the burn in period specified and applied the specified stride; Number of samples skipped.
    """

    if (sampler_type is None) :
        # Read first line and look for sampler tag
        with open(filename) as f:
            first_line = f.readline()
        if ( "lklhdfmt" in first_line.lower() ) :
            sampler_type=first_line.split()[1]
        else :
            sampler_type='ensemble'

    if (sampler_type=='ensemble') :
        return read_ensemble_lklhd(filename,stride=stride,burn_fraction=burn_fraction,skip=skip)
    elif (sampler_type=='stan') :
        state,nskip = read_stan_state(filename,stride=stride,burn_fraction=burn_fraction,skip=skip,auto_warmup=auto_warmup)
        eL = np.zeros([state.shape[0],1])
        eL[:,0] = state[:,0]
        return eL,nskip
    elif (sampler_type=='afss') :
        state,nskip = read_afss_state(filename, stride=stride, burn_fraction=burn_fraction, skip=skip)
        eL = np.zeros([state.shape[0],1])
        eL[:,0] = state[:,0]
        return eL,nskip
    else :
        raise RuntimeError("Unrecognized lklhdfmt type, %s."%(sampler_type))

    
def read_ensemble_lklhd(filename, stride=1, burn_fraction=0, skip=None):
    """
    Reads in a Themis likelihood file from an ensemble sampler.  Optionally, a 
    stride, burn-in fraction, or number of *ensemble samples* to skip may be set.
    Returns an IndexError if fewer ensemble samples exist in the chain file than skip.

    Args:
      filename (str): Filename in which likelihood data will be found (e.g., `Lklhd.dat`)
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip.  Overrides burn_fraction unless set to None. Default: None.

    Returns:
      (numpy.ndarray, int): Likelihood data arranged as 2D array indexed by [sample, walker] *after* excluding the burn in period specified and applied the specified stride; Number of samples skipped.
    """

    # Find the number of independent samples
    nsamp,nhead = file_length(filename,header_string='#')
    
    # Set the number of lines to skip
    nskip = 0
    if (skip is None) :
        nskip = int(nsamp*burn_fraction)
    else :
        nskip = skip

    # If there are too few lines in the likelihood file, raise IndexError
    if (nskip>nsamp) :
        raise IndexError

    # Number of samples to store
    nstor = (nsamp-nskip)//stride

    # Add in the header
    nskip += nhead
    
    # Loop over the lines from the file
    j=0
    for k,l in enumerate(open(filename,'r')) :
    
        # Skip the burn-in period
        if (k<nskip) :
            continue
        
        # If this is inconsistent with the thin stride skip
        if ((k-nskip)%stride!=0) :
            continue

        # If past the end, stop
        if (j>=nstor) :
            break

        # If this is the first step, allocate space for the likelihood
        if (j==0) :
            nwalk = len(l.split())
            lklhd = np.zeros((nstor,nwalk))

        # If this is consistent with the thin strid keep
        #lklhd[j,:] = np.array(l.split()).astype(float)
        tokens = l.split()
        lklhd[j,:] = np.array([float(tok) for tok in tokens]) #l.split()).astype(float)
        j += 1

    return lklhd,nskip


def read_echain(filename, walkers=1, stride=1, burn_fraction=0, skip=None, parameter_list=None, sampler_type=None):
    """
    Reads in a Themis chain file.  The number of walkers must be supplied.  Optionally, 
    a stride, burn-in fraction, number of *ensemble samples* to skip, or a set of 
    parameters may be provided.  Returns an IndexError if fewer ensemble samples exist 
    in the chain file than skip.
    
    Args:
      filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      walkers (int): Number of independent walkers in the ensemble. Default: 1.
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.
      sampler_type (str): Likelihood file format specifier.  If None, looks for a format at the top of the likelihood file; if not found, defaults to 'ensemble'.  Default: None.
 
    Returns:
      (numpy.ndarray): Chain data arranged as 3D array indexed by [sample, walker, parameter] *after* excluding the skipped lines and applying the specified stride.
    """

    if (sampler_type is None) :
        # Read first line and look for sampler tag
        with open(filename) as f:
            first_line = f.readline()
        if ( "chainfmt" in first_line.lower() ) :
            sampler_type=first_line.split()[1]
        else :
            sampler_type='ensemble'

    if (sampler_type=='ensemble') :
        return read_ensemble_chain(filename,walkers=walkers,stride=stride,burn_fraction=burn_fraction,skip=skip,parameter_list=parameter_list)
    elif (sampler_type=='stan') :
        chain = read_stan_chain(filename,stride=stride,burn_fraction=burn_fraction,skip=skip,parameter_list=parameter_list)
        echain = np.zeros([chain.shape[0],1,chain.shape[1]])
        echain[:,0,:] = chain[:,:]
        return echain
    elif (sampler_type == 'afss') :
        chain = read_stan_chain(filename,stride=stride,burn_fraction=burn_fraction,skip=skip,parameter_list=parameter_list)
        echain = np.zeros([chain.shape[0],1,chain.shape[1]])
        echain[:,0,:] = chain[:,:]
        return echain
    else :
        raise RuntimeError("Unrecognized chainfmt type, %s."%(sampler_type))


def read_ensemble_chain(filename, walkers, stride=1, burn_fraction=0, skip=None, parameter_list=None):
    """
    Reads in a Themis chain file from an ensemble sampler.  The number of walkers 
    must be supplied.  Optionally, a stride, burn-in fraction, number of *ensemble
    samples* to skip, or a set of parameters may be provided.  Returns an IndexError 
    if fewer ensemble samples exist in the chain file than skip.
    
    Args:
      filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      walkers (int): Number of independent walkers in the ensemble.
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.
 
    Returns:
      (numpy.ndarray): Chain data arranged as 3D array indexed by [sample, walker, parameter] *after* excluding the skipped lines and applying the specified stride.
    """


    # Determine if a full step has been output and remove partial outputs
    nsamp,nhead = file_length(filename,header_string='#')
    nsamp -= nsamp%walkers

    # Set the number of lines to skip
    if (skip is None) :
        nskip = int(nsamp*burn_fraction)
    else :
        nskip = skip*walkers

    
    # If there are too few lines in the chain file, raise IndexError
    if (nskip>nsamp) :
        raise IndexError

    # Read in, parse and save
    nstor = (nsamp-nskip)//(walkers*stride)

    # Add in the header
    nskip += nhead

    # Loop over the lines from the file
    j=0
    for k,l in enumerate(open(filename,'r')) :

        # Skip the burn-in period
        if (k<nskip) :
            continue

        # If this is inconsistent with the thin stride skip
        if (((k-nskip)//walkers)%stride!=0) :
            continue

        # If past the end, stop
        if (j>=nstor*walkers) :
            break

        
        # If this is the first step, allocate space for the likelihood
        if (j==0) :
            if (parameter_list is None) :
                parameter_list = list(range(len(l.split())))
            chain = np.zeros((nstor*walkers,len(parameter_list)))
        
        tokens = l.split()
        chain[j,:] = np.array([float(tokens[p]) for p in parameter_list])
        j += 1

    return chain.reshape([nstor,walkers,-1])


def load_erun(chain_filename, lklhd_filename, stride=1, burn_fraction=0, skip=None, auto_warmup=False, parameter_list=None,sampler_type=None):
    """
    Coherently loads a Themis chain and likelihood pair.
    Optionally, a stride, burn-in fraction, number of *ensemble samples* to skip, 
    or a set of parameters may be provided.  If set, the burn-in fraction is 
    computed for the *shorter* of the two files.

    Args:
      chain_filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      lklhd_filename (str): Filename in which likelihood data will be found (e.g., `Lklhd.dat`)
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      auto_warmup (bool): If true it will automatically sense the Stan warmup period for Stan state files and cut it. `burn_fraction` then applies to the state file with warmup already removed.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.
      sampler_type (str): Likelihood file format specifier.  If None, looks for a format at the top of the likelihood file; if not found, defaults to 'ensemble'.  Default: None.

    Returns:
      (numpy.ndarray, numpy.ndarray): Chain data arranged as 3D array indexed by [sample, walker, parameter] *after* excluding the skipped lines and applying the specified stride; Likelihood data arranged as 2D array indexed by [sample, walker] *after* excluding the burn in period specified and applied the specified stride.
    """

    # Find the lengths of the chain and likelihood files
    chain_nsamp,nhead = file_length(chain_filename,header_string='#')
    elklhd,_ = read_elklhd(lklhd_filename, stride=stride, skip=skip, auto_warmup=auto_warmup, sampler_type=sampler_type)
    
    # Determine the number of walkers, and set the burn-in relative to the shorter
    walkers = elklhd.shape[1]
    nsamp = min(elklhd.shape[0],chain_nsamp//(walkers*stride))
    if (skip is None) :
        skip = int(burn_fraction*nsamp)
        
    # Restrict the likelihoods
    elklhd = elklhd[0:nsamp,:]

    # Read and restrict the chain
    echain = read_echain(chain_filename, walkers, stride=stride, skip=skip, parameter_list=parameter_list)[:nsamp,:,:]
    
    return echain,elklhd
    
def sample_erun(chain_filename, lklhd_filename, samples, burn_fraction=0, skip=None, parameter_list=None):
    """
    Coherently loads and generates samples from a Themis chain and likelihood pair from
    an ensemble sampler.  Optionally, a burn-in fraction, number of *ensemble samples* 
    to skip, or a set of parameters may be provided.  If set, the burn-in fraction is 
    computed for the *shorter* of the two files.  Is faster and more memory efficient 
    than load_erun and subsequently subsampling for large chains

    Args:
      chain_filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      lklhd_filename (str): Filename in which likelihood data will be found (e.g., `Lklhd.dat`)
      samples (int): Number of samples to draw.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.

    Returns:
      (numpy.ndarray, numpy.ndarray): Chain data arranged as 2D array indexed by [sample, parameter] *after* excluding the skipped lines and applying the specified stride; Likelihood data arranged as 1D array indexed by [sample] *after* excluding the burn in period specified and applied the specified stride.
    """

    # Find the lengths of the chain and likelihood files
    chain_nsamp,nhead = file_length(chain_filename,header_string='#')
    elklhd,tmp = read_elklhd(lklhd_filename, skip=skip)


    # Determine the number of walkers, and set the burn-in relative to the shorter
    walkers = elklhd.shape[1]
    nsamp = min(elklhd.shape[0]*elklhd.shape[1],chain_nsamp)
    if (skip is None) :
        nskip = burn_fraction*nsamp
    else :
        nskip = skip*walkers
    
    # If there are too few lines in the chain file, raise IndexError
    if (nskip>nsamp) :
        raise IndexError

    # Add in the header
    nskip += nhead

    # Generate sample list
    sample_list = np.sort(np.random.randint(nskip,high=nsamp,size=samples))
    
    # Loop over the lines from the file
    j=0
    for k,l in enumerate(open(chain_filename,'r')) :
        # Skip the burn-in period
        if (k<nskip) :
            continue
                
        while (j<samples and k==sample_list[j]) :
            if (j==0) :
                if (parameter_list is None) :
                    parameter_list = list(range(len(l.split())))
                echain = np.zeros((samples,len(parameter_list)))

            tokens = l.split()
            echain[j,:] = np.array([float(tokens[p]) for p in parameter_list])
            j += 1

        if (j==len(sample_list)) :
            break
        
    # Grab the subsamples from the likelihood
    elklhd = np.array(elklhd.reshape([-1])[sample_list])
    
    return echain,elklhd

def read_afss_state(filename, stride=1, burn_fraction=0, skip=None, auto_warmup=False):
    """
    Reads in a Themis state file from the automated factor slice sampler.  Optionally, a 
    stride, burn-in fraction, or number of *Stan samples* to skip may be set.
    Returns an IndexError if fewer samples exist in the chain file than skip.

    Args:
      filename (str): Filename in which state data will be found (e.g., `State.dat`)
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip.  Overrides burn_fraction unless set to None. Default: None.

    Returns:
      (numpy.ndarray, int): Likelihood data arranged as 2D array indexed by [sample, walker] *after* excluding the burn in period specified and applied the specified stride; Number of samples skipped. 
    """

    # Find the number of independent samples
    nsamp,nhead = file_length(filename,header_string='#')
    
    warmup_index = 0 
    # Set the number of lines to skip
    nskip = 0
    if (skip is None) :
        nskip = int(nsamp*burn_fraction)+warmup_index
    else :
        nskip = skip + warmup_index

    # If there are too few lines in the state file, raise IndexError
    if (nskip>nsamp) :
        raise IndexError

    # Number of samples to store
    nstor = (nsamp-nskip)//stride
    state = np.zeros((nstor,5))    

    # Add in the header
    nskip += nhead
    
    # Loop over the lines from the file
    j=0
    for k,l in enumerate(open(filename,'r')) :
    
        # Skip the burn-in period
        if (k<nskip) :
            continue
        
        # If this is inconsistent with the thin stride skip
        if ((k-nskip)%stride!=0) :
            continue

        # If past the end, stop
        if (j>=nstor) :
            break

        # If this is consistent with the thin strid keep
        #lklhd[j,:] = np.array(l.split()).astype(float)
        tokens = l.split()
        state[j,:] = np.array([float(tok) for tok in tokens]) #l.split()).astype(float)
        j += 1

    return state,nskip


def read_stan_state(filename, stride=1, burn_fraction=0, skip=None, auto_warmup=False):
    """
    Reads in a Themis state file from a Stan sampler.  Optionally, a 
    stride, burn-in fraction, auto_warmup detection, or number of *Stan samples* to skip may be set.
    Returns an IndexError if fewer samples exist in the chain file than skip.

    Args:
      filename (str): Filename in which state data will be found (e.g., `State.dat`)
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      auto_warmup (bool): If true it will automatically sense the Stan warmup period and cut it. `burn_fraction` then applies to the state file with warmup already removed.
      skip (int): Number of initial *ensemble samples* to skip.  Overrides burn_fraction unless set to None. Default: None.

    Returns:
      (numpy.ndarray, int): Likelihood data arranged as 2D array indexed by [sample, walker] *after* excluding the burn in period specified and applied the specified stride; Number of samples skipped including warmup if auto_warmup is true.
    """

    # Find the number of independent samples
    nsamp,nhead = file_length(filename,header_string='#')
    
    warmup_index = 0 
    if (auto_warmup):
        with open(filename,"r") as f: 
             head = f.readline() 
             head = head.split()[1:] 
             i = head.index("stepsize__") 
             step = f.readline().split()[i] 
             for l in f: 
                 sstep = l.split()[i] 
                 if (sstep == step): 
                     break 
                 warmup_index+=1 
                 step = sstep

    # Set the number of lines to skip
    nskip = 0
    if (skip is None) :
        nskip = int(nsamp*burn_fraction)+warmup_index
    else :
        nskip = skip + warmup_index

    # If there are too few lines in the state file, raise IndexError
    if (nskip>nsamp) :
        raise IndexError

    # Number of samples to store
    nstor = (nsamp-nskip)//stride
    state = np.zeros((nstor,7))    

    # Add in the header
    nskip += nhead
    
    # Loop over the lines from the file
    j=0
    for k,l in enumerate(open(filename,'r')) :
    
        # Skip the burn-in period
        if (k<nskip) :
            continue
        
        # If this is inconsistent with the thin stride skip
        if ((k-nskip)%stride!=0) :
            continue

        # If past the end, stop
        if (j>=nstor) :
            break

        # If this is consistent with the thin strid keep
        #lklhd[j,:] = np.array(l.split()).astype(float)
        tokens = l.split()
        state[j,:] = np.array([float(tok) for tok in tokens]) #l.split()).astype(float)
        j += 1

    return state,nskip
            

def read_stan_chain(filename, stride=1, burn_fraction=0, skip=None, parameter_list=None):
    """
    Reads in a Themis chain file from an stan sampler.  The number of walkers 
    must be supplied.  Optionally, a stride, burn-in fraction, number of *ensemble
    samples* to skip, or a set of parameters may be provided.  Returns an IndexError 
    if fewer samples exist in the chain file than skip.
    
    Args:
      filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.
 
    Returns:
      (numpy.ndarray): Chain data arranged as 2D array indexed by [sample, parameter] *after* excluding the skipped lines and applying the specified stride.
    """


    # Determine if a full step has been output and remove partial outputs
    nsamp,nhead = file_length(filename,header_string='#')

    # Set the number of lines to skip
    if (skip is None) :
        nskip = int(nsamp*burn_fraction)
    else :
        nskip = skip

    
    # If there are too few lines in the chain file, raise IndexError
    if (nskip>nsamp) :
        raise IndexError

    # Read in, parse and save
    nstor = (nsamp-nskip)//(stride)

    # Add in the header
    nskip += nhead

    # Loop over the lines from the file
    j=0
    for k,l in enumerate(open(filename,'r')) :

        # Skip the burn-in period
        if (k<nskip) :
            continue

        # If this is inconsistent with the thin stride skip
        if ((k-nskip)%stride!=0) :
            continue

        # If past the end, stop
        if (j>=nstor) :
            break

        
        # If this is the first step, allocate space for the likelihood
        if (j==0) :
            if (parameter_list is None) :
                parameter_list = list(range(len(l.split())))
            chain = np.zeros((nstor,len(parameter_list)))
        
        tokens = l.split()
        chain[j,:] = np.array([float(tokens[p]) for p in parameter_list])
        j += 1

    return chain


def load_stan_run(chain_filename, state_filename, stride=1, burn_fraction=0, auto_warmup=True, skip=None, parameter_list=None):
    """
    Coherently loads a Themis chain and likelihood pair from an Stan Themis run.
    Optionally, a stride, burn-in fraction, number of *samples* to skip, 
    or a set of parameters may be provided.  If set, the burn-in fraction is 
    computed for the *shorter* of the two files.

    Args:
      chain_filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      state_filename (str): Filename in which state data will be found (e.g., `State.dat`)
      stride (int): Integer factor by which to step through samples *coherently* among chains.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      auto_warmup (bool): If true it will automatically sense the Stan warmup period and cut it.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.

    Returns:
      (numpy.ndarray, numpy.ndarray): Chain data arranged as 2D array indexed by [sample, parameter] *after* excluding the skipped lines and applying the specified stride; State data arranged as 2D array with columns likelihood, accept, stepsize, treedepth, nleapfrog, divergence, energy *after* excluding the burn in period specified and applied the specified stride.
    """

    # Find the lengths of the chain and likelihood files
    nsamp,nhead = file_length(chain_filename,header_string='#')
    state,_ = read_stan_state(state_filename, stride=stride, skip=skip,burn_fraction=burn_fraction, auto_warmup=auto_warmup)
    
    # Determine the number of walkers, and set the burn-in relative to the shorter
    nsamp = min(state.shape[0],nsamp//(stride))
    if (skip is None) :
        skip = int(burn_fraction*nsamp)
        
    # Restrict the likelihoods
    state = state[0:nsamp,:]

    # Read and restrict the chain
    chain = read_stan_chain(chain_filename, stride=stride, skip=skip, parameter_list=parameter_list)[:nsamp,:]

    #If warmup is removed count the number of divergences in the samples
    if (auto_warmup):
        print("The number of divergent transitions in %s = %d."%(state_filename,np.sum(state[:,-2])))
    
    return chain,state


def sample_chain(chain_filename, samples, burn_fraction=0, skip=None, parameter_list=None):
    """
    Loads and generates samples from a Themis chain a discrete sampler.  Optionally, a 
    burn-in fraction, number of *samples*  to skip, or a set of parameters may be provided.  
    Is faster and more memory efficient than load_erun and subsequently subsampling for 
    large chains.

    Args:
      chain_filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      samples (int): Number of samples to draw.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.

    Returns:
      (numpy.ndarray): Chain data arranged as 2D array indexed by [sample, parameter] *after* excluding the skipped lines and applying the specified stride
    """

    # Find the lengths of the chain and likelihood files
    nsamp,nhead = file_length(chain_filename,header_string='#')

    # Determine the number of walkers, and set the burn-in relative to the shorter
    if (skip is None) :
        nskip = int(burn_fraction*nsamp)
    else :
        nskip = skip
    
    # If there are too few lines in the chain file, raise IndexError
    if (nskip>nsamp) :
        raise IndexError

    # Add in the header
    nskip += nhead

    # Generate sample list
    sample_list = np.sort(np.random.randint(nskip,high=nsamp,size=samples))
    
    # Loop over the lines from the file
    j=0
    for k,l in enumerate(open(chain_filename,'r')) :
        # Skip the burn-in period
        if (k<nskip) :
            continue
                
        while (j<samples and k==sample_list[j]) :
            if (j==0) :
                if (parameter_list is None) :
                    parameter_list = list(range(len(l.split())))
                chain = np.zeros((samples,len(parameter_list)))

            tokens = l.split()
            chain[j,:] = np.array([float(tokens[p]) for p in parameter_list])
            j += 1

        if (j==len(sample_list)) :
            break
        
    return chain



def most_likely_erun(chain_filename, lklhd_filename, samples=1, burn_fraction=0, skip=None, parameter_list=None):
    """
    Coherently loads and returns optimal samples from a Themis chain and likelihood pair 
    from an ensemble sampler.  Optionally, a burn-in fraction, number of *ensemble samples* 
    to skip, or a set of parameters may be provided.  If set, the burn-in fraction is 
    computed for the *shorter* of the two files.  Is faster and more memory efficient 
    than load_erun and subsequently subsampling for large chains

    Args:
      chain_filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      lklhd_filename (str): Filename in which likelihood data will be found (e.g., `Lklhd.dat`)
      samples (int): Number of samples to draw. Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.

    Returns:
      (numpy.ndarray, numpy.ndarray): Chain data arranged as 2D array indexed by [sample, parameter] *after* excluding the skipped lines and applying the specified stride; Likelihood data arranged as 1D array indexed by [sample] *after* excluding the burn in period specified and applied the specified stride.
    """

    # Find the lengths of the chain and likelihood files
    chain_nsamp,nhead = file_length(chain_filename,header_string='#')
    elklhd,tmp = read_elklhd(lklhd_filename, skip=skip)

    # Determine the number of walkers, and set the burn-in relative to the shorter
    walkers = elklhd.shape[1]
    nsamp = min(elklhd.shape[0]*elklhd.shape[1],chain_nsamp)
    if (skip is None) :
        nskip = burn_fraction*nsamp
    else :
        nskip = skip*walkers
    
    # If there are too few lines in the chain file, raise IndexError
    if (nskip>nsamp) :
        raise IndexError

    # Add in the header
    nskip += nhead

    # Flatten likelihoods and shorten to the number of samples in the chain
    elklhd = np.array(elklhd.reshape([-1])[:nsamp])
    
    # Generate sample list
    sample_list = np.sort(np.argsort(-elklhd)[:samples])
    
    # Loop over the lines from the file
    j=0
    for k,l in enumerate(open(chain_filename,'r')) :
        # Skip the burn-in period
        if (k<nskip) :
            continue
        
        while (j<samples and k-nhead==sample_list[j]) :
            if (j==0) :
                if (parameter_list is None) :
                    parameter_list = list(range(len(l.split())))
                echain = np.zeros((samples,len(parameter_list)))

            tokens = l.split()
            echain[j,:] = np.array([float(tokens[p]) for p in parameter_list])
            j += 1

        if (j==len(sample_list)) :
            break
        
    # Grab the subsamples from the likelihood
    elklhd = np.array(elklhd.reshape([-1])[sample_list])

    # Re-order in likelihood
    isrt = np.argsort(-elklhd)
    elklhd = elklhd[isrt]
    echain = echain[isrt,:]

    return echain,elklhd


def join_echains(echains):
    """
    Join a list of ensemble chains from separate runs. If chains have a different
    number of steps then the largest even common size is used to join them.
    
    Args:
      echains (list): List of ensemble chains to join.

    Returns:
      (numpy.ndarray): Combined ensemble chain.
    """
    lengths = []
    for i in range(len(echains)):
        lengths.append(len(echains[i][:,0]))
    max_length = np.amin(lengths)
    if max_length%2!=0:
        max_length -=1
    jechains = np.zeros((len(echains),max_length,echains[0].shape[1], echains[0].shape[2]))
    for i in range(len(echains)):
        jechains[i,:,:,:] = echains[i][-max_length:,:,:]
    return jechains



    
    

def load_deo_summary(annealing_file="Annealing.dat", annealing_summary=None) :
    """
    Loads the DEO annealing files, which contain information about the tempering ladder 
    evolution, optimization, and performance for DEO annealing schemes.
    

    Args:
      annealing_file (str): Name of the annealing level information file generated by DEO tempered Themis analyses.  Default: "Annealing.dat".
      annealing_file_summary (str): Name of the annealing summary information file generated by DEO tempered Themis analyses.  If None will look for annealing_file+".summary".  Default: None.

    Returns:
      (numpy.ndarray,dict): Array of annealing summary information containing Lambda, rt_opt, E, rt_est, AvgR, StdR corresponding to ..., indexed by round; schedule of annealing :math:`\\beta`; :math:`R` for each tempering level and round indexed by [round,tempering level] and labelled by 'Beta' and 'R', respectively. 
    """
    
    # Read in Annealing file
    annealing_data = np.genfromtxt(annealing_file, names=True, skip_header=1)

    # Read in Annealing summary file
    if (annealing_summary is None) :
        annealing_summary = annealing_file + '.summary'
    annealing_summary_data = np.genfromtxt(annealing_summary, names=True)

    # Number of completed rounds
    nrounds = int(annealing_summary_data['round'][-1]+1)

    # Number of tempering levels
    ntemp = len((annealing_data[annealing_data['round']==0]))

    beta = np.zeros((nrounds,ntemp))
    R = np.zeros((nrounds,ntemp))

    for n in range(nrounds) :
        beta[n,:] = annealing_data['Beta'][annealing_data['round']==n]
        R[n,:] = annealing_data['R'][annealing_data['round']==n]

    annealing_data_dict = { 'Beta':beta, 'R':R }
        
        
    return annealing_summary_data,annealing_data_dict
