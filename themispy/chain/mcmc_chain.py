# Provides utility functions for reading and manipulating MCMC chain
# data produced by Themis analyses.
#
# 
#

import numpy as np
import warnings


"""
Collection of untilities for reading and manipulating MCMC chain data
produced by Themis analyses.
"""


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


def read_elklhd(filename, stride=1, burn_fraction=0, skip=None):
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
      (ndarray, int): Likelihood data arranged as 2D array indexed by [sample, walker] *after* excluding the burn in period specified and applied the specified stride; Number of samples skipped.
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
            

def read_echain(filename, walkers, stride=1, burn_fraction=0, skip=None, parameter_list=None):
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
      (ndarray): Chain data arranged as 3D array indexed by [sample, walker, parameter] *after* excluding the skipped lines and applying the specified stride.
    """


    # Determine if a full step has been output and remove partial outputs
    nsamp,nhead = file_length(filename,header_string='#')
    nsamp -= nsamp%walkers

    # Set the number of lines to skip
    if (skip is None) :
        nskip = nsamp*burn_fraction*walkers
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
        if (j>=nsamp) :
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


def load_erun(chain_filename, lklhd_filename, stride=1, burn_fraction=0, skip=None, parameter_list=None):
    """
    Coherently loads a Themis chain and likelihood pair from an ensemble sampler.
    Optionally, a stride, burn-in fraction, number of *ensemble samples* to skip, 
    or a set of parameters may be provided.  If set, the burn-in fraction is 
    computed for the *shorter* of the two files.

    Args:
      chain_filename (str): Filename in which chain data will be found (e.g., `Chain.dat`)
      lklhd_filename (str): Filename in which likelihood data will be found (e.g., `Lklhd.dat`)
      stride (int): Integer factor by which to step through samples *coherently* among walkers.  Default: 1.
      burn_fraction (float): Fraction of the total number of lines to exclude from the beginning.  Default: 0.
      skip (int): Number of initial *ensemble samples* to skip. Default: 0.
      parameter_list (list): List of parameter columns (zero-offset) to read in.  Default: None, which reads all parameters.

    Returns:
      (ndarray, ndarray): Likelihood data arranged as 2D array indexed by [sample, walker] *after* excluding the burn in period specified and applied the specified stride; Chain data arranged as 3D array indexed by [sample, walker, parameter] *after* excluding the skipped lines and applying the specified stride.
    """

    # Find the lengths of the chain and likelihood files
    chain_nsamp,nhead = file_length(chain_filename,header_string='#')
    elklhd,nhead = read_elklhd(lklhd_filename, stride=stride, skip=skip)

    # Determine the number of walkers, and set the burn-in relative to the shorter
    walkers = elklhd.shape[1]
    nsamp = min(elklhd.shape[0],chain_nsamp//walkers)
    if (skip is None) :
        skip = burn_fraction*nsamp

    # Restrict the likelihoods
    elklhd = elklhd[skip:nsamp,:]

    # Read and restrict the chain
    echain = read_echain(chain_filename, walkers, stride=stride, skip=skip, parameter_list=parameter_list)[skip:nsamp,:,:]
    
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
      (ndarray, ndarray): Likelihood data arranged as 1D array indexed by [sample] *after* excluding the burn in period specified and applied the specified stride; Chain data arranged as 2D array indexed by [sample, parameter] *after* excluding the skipped lines and applying the specified stride.
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
      (ndarray, ndarray): Likelihood data arranged as 1D array indexed by [sample] *after* excluding the burn in period specified and applied the specified stride; Chain data arranged as 2D array indexed by [sample, parameter] *after* excluding the skipped lines and applying the specified stride.
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

    # Re-order in likelihood
    isrt = np.argsort(-elklhd)
    elklhd = elklhd[isrt]
    echain = echain[isrt,:]

    return echain,elklhd
