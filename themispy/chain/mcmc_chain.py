## @package mcmc_chain
#  Provides utility functions for reading and manipulating MCMC chain
#  data produced by Themis analyses.
#
#

# MUST FIX load_erun SO THAT IT HANDLES DISPARATE SIZES
 
import numpy as np
import warnings


def file_length(fname, header_string=None, comment_string=None) :
    """
    Counts the number of lines in a file.  Optionally will remove header lines, 
    marked by a prepended character, and comment lines, marked by a possibly 
    different prepended character.

    Arguments:
     - `fname` : string of the file containing the chain.
     - `nwalkers` : number of walkers used in the ensemble sampler
     - `nskip` : number of MCMC steps to skip, i.e. warmup period.
     - `thin` : controls how much we want to thin the chain, so every thin step will be stored.
    Keyword arguments:
     - `plist` : list of integers of parameters to read in. Default is None which reads in every parameter
    Returns:
     chain file with dimensions step,walker,param indexing.


    """
    
    length=0
    if (comment_string is None) and (header_string is None) :
        length=sum((1 for i in open(fname, 'r')))
    elif (comment_string is None) :
        length=sum((1 for i in open(fname, 'r')))
        for l in open(fname,'r') :
            if (l[0]==header_string) :
                length -= 1
            else :
                break
    elif (header_string is None) :
        for l in open(fname,'r') :
            if (l[0]==comment_string) :
                continue
            length += 1
    else :
        for l in open(fname,'r') :
            if (l[0]==comment_string or l[0]==header_string) :
                continue
            length += 1
        
    return length

## Determines the number of lines in a file with potential comments
#  @param fname file name
#  @param comment_string (optional)
def file_length3(fname,comment_string=None) :

    length=0
    if (comment_string is None) :
        for length,line in enumerate(open(fname, 'r')):
            pass
    else :
        for l in open(fname,'r') :
            if (l[0]==comment_string) :
                continue
            length += 1
        
    return length


def file_length2(fname) :

    nsteps = 0
    skip_first = 0    
    with open(fname) as f:
        if f.readline().startswith("#"):
            skip_first=1
        head = [next(f) for i in range(1)]
        i = 0
        for i,l in enumerate(f):
            pass
        nsteps = i+2+1-skip_first

    return nsteps
        

def read_elklhd(lname, thin, warmup_frac):
    """
    Reads in a themis likelihood file, `lname` storing only the 1-`warmup_frac` part
    for every `thin` steps. Returns the lklhd file in an 2D array with dimensions stepsXwalkers,
    the number of steps that were skipped and the number of COMPLETED steps.
    """

    nsteps = 0
    skip_first = 0
    with open(lname) as f:
        if f.readline().startswith("#"):
            skip_first=1
        head = [next(f) for i in range(1)]
        i = 0
        for i,l in enumerate(f):
            pass
        nsteps = i+2+1-skip_first
    #Check if the last line ends with \n, if it doesn't then that step didn't finish so we will cut it.
    #Find the number of walkers
    nwalkers = len(head[0].split())
    if len(l.split()) != nwalkers:
        nsteps -= 1
    print("Reading likelihood file %s has %d steps with %d walkers"%(lname, nsteps, nwalkers))
    nskip = int(nsteps*warmup_frac)
    nstore = (nsteps-nskip)//thin
    if (nsteps-nskip)%thin > 0:
        nstore += 1
    lklhd = np.zeros((nstore,nwalkers) )
    #print()
    with open(lname) as f:
        for i,line in enumerate(f):
            if i < (nskip+skip_first):
                continue
            if (i-skip_first)%thin != 0:
                continue
            #Skip the last line if it hasn't finished outputting
            if (i > nsteps-1+skip_first):
                break
            indx = (i-nskip-skip_first)//thin
            lklhd[indx,:] = list(map(lambda x: float(x),line.split()))

    return lklhd,nskip
            

def read_echain(cname, nwalkers, nskip, thin, plist=None):
    """
    Reads in a Themis chain from the ensemble methods.
    Arguments:
     - `cname` : string of the file containing the chain.
     - `nwalkers` : number of walkers used in the ensemble sampler
     - `nskip` : number of MCMC steps to skip, i.e. warmup period.
     - `thin` : controls how much we want to thin the chain, so every thin step will be stored.
    Keyword arguments:
     - `plist` : list of integers of parameters to read in. Default is None which reads in every parameter
    Returns:
     chain file with dimensions step,walker,param indexing.
    """

    #Now we need to check if the last step finished outputting, if it doesn't then 
    #we skip it. To do this lets first find the number of lines in the Chain file
    #excluding the header if it exists
    nlines = 0
    skip_first = 0
    with open(cname) as f:
        if f.readline().startswith("#"):
            skip_first=1
        head = [next(f) for i in range(1)]
        i = 0
        for i,l in enumerate(f):
            pass
        nlines = i+2+1-skip_first
    #Find the number of walkers
    nparams = len(head[0].split())
    print("I think there are %d parameters in %s"%(nparams,cname))
    if plist is None :
        plist = list(range(nparams))
    print("Reading in parameters", plist)
    #Now if the nlines is divisible by the number of walkers then 
    #everything has outputted correctly. If not then kill the last step.
    nend = 0
    nlines -= nlines%nwalkers
    nsteps = ((nlines//nwalkers)-nskip)//thin
    if ((nlines//nwalkers-nskip)%thin > 0):
        nsteps += 1
    chain = np.zeros((nsteps*nwalkers, len(plist)))
    with open(cname) as f:
        j = -1
        for i,line in enumerate(f):
            if i < (nskip*nwalkers + skip_first):
                continue
            if (((i-skip_first)//nwalkers)%thin != 0):
                continue
            if (((i-skip_first))%nwalkers == 0):
                j+=1
            if (i > nlines-1+skip_first):
                break
            indx = j*nwalkers + (i-skip_first)%nwalkers
            chain[indx,:] = np.array(list(map(lambda x: float(x),line.split())))[plist]
    return chain.reshape(nsteps,nwalkers,len(plist))
    

def load_erun(cname, lname, thin, plist=None, warmup_frac=0.5):
    """
    Loads the ensemble chain and likelihood files
    Params:
        - cname<str> Is the name of the chains file
        - lname<str> Is the name of the lklhd file
        - thin <int> Is the thinning of the lkhd and chain
        - plist<arr<int>> Is the list of parameters to read in, e.g. [1,2,5] will read in the first second and 5 parameter from the chain file.
                          defaults to None which reads in all the parameters.
        - warmup_frac Is the fraction of the chain to remove due to warmup. The default is 0.5 so the first 50% of the chain will be removed.
    Returns:
        chain, lklhd
    Where the chain is a 3d array using the walker,steps,param indexing.
    """

    elklhd,nskip = read_elklhd(lname, thin, warmup_frac)
    echain = read_echain(cname, elklhd.shape[1], nskip, thin, plist)

    # homogenize the length of the chains, if necessary
    if (elklhd.shape[0] is not echain.shape[0]) :
        nlines = min(elklhd.shap[0],echain.shape[0])
        elklhd = elklhd[:nlines,:]
        echain = echain[:nlines,:,:]
    
    return echain,elklhd



def subsample_erun(cname, lname, samples, plist=None, warmup_frac=0.5) :
    """
    Loads a subsample of the ensemble chain and likelihood files.  Is faster and more memory efficient than load_erun and subsequently subsampling.
    Params:
        - cname <str> Is the name of the chains file
        - lname <str> Is the name of the lklhd file
        - samples <int> Is the number of samples to draw
        - plist <arr<int>> Is the list of parameters to read in, e.g. [1,2,5] will read in the first second and 5 parameter from the chain file.
                          defaults to None which reads in all the parameters.
        - warmup_frac Is the fraction of the chain to remove due to warmup. The default is 0.5 so the first 50% of the chain will be removed.
    Returns:
        chain, lklhd
    Where the chain is a 2d array using steps,param indexing and lklhd is a 1d array using steps indexing
    """


    # Count the number of samples in the chain file
    chain_length=sum((1 for i in open(cname, 'rb')))

    # Count the number of samples in the lklhd file
    lklhd,nskip = read_elklhd(lname,1,0)

    # Total sample number
    total_sample_number = min(chain_length,lklhd.shape[0]*lklhd.shape[1])

    # Generate sample list
    sample_list = np.sort(np.random.randint(int(warmup_frac*total_sample_number)+nskip,high=total_sample_number,size=samples))

    # Loop over chain file and read in desired lines
    chain=[]
    k=np.int(0)
    j=np.int(0)
    for line in open(cname,'r') :
        if (k==nskip) :
            nparams=len(line.split())
            if ( plist is None ) :
                plist = list(range(nparams))

        while (j<len(sample_list) and k==sample_list[j]) :
            chain.extend([np.array(line.split()).astype(float)[plist]])
            j += 1
        
        if (j==len(sample_list)) :
            break
            
        k += 1

    chain = np.array(chain)
    lklhd = np.array(lklhd.reshape([-1])[sample_list])
    
    return chain,lklhd


def most_likely_erun(cname, lname, samples, plist=None) :
    """
    Loads the requested number of samples of highest likelihood points from the ensemble chain and likelihood files.  Is faster and more memory efficient than load_erun and subsequently subsampling.
    Params:
        - cname <str> Is the name of the chains file
        - lname <str> Is the name of the lklhd file
        - samples <int> Is the number of samples to draw
        - plist <arr<int>> Is the list of parameters to read in, e.g. [1,2,5] will read in the first second and 5 parameter from the chain file.
                          defaults to None which reads in all the parameters.
    Returns:
        chain, lklhd
    Where the chain is a 2d array using steps,param indexing and lklhd is a 1d array using steps indexing
    """

    # Count the number of samples in the chain file
    chain_length=sum((1 for i in open(cname, 'rb')))

    # Count the number of samples in the lklhd file
    lklhd,nskip = read_elklhd(lname,1,0)
    lklhd = lklhd.reshape([-1])

    # Total sample number
    total_sample_number = min(chain_length,lklhd.size)
    lklhd = lklhd[:total_sample_number]

    # Generate sample list
    sample_list = np.sort(np.argsort(-lklhd)[:samples])
    
    # Loop over chain file and read in desired lines
    chain=[]
    k=np.int(0)
    j=np.int(0)
    for line in open(cname,'r') :
        if (k==nskip) :
            nparams=len(line.split())
            if ( plist is None ) :
                plist = list(range(nparams))

        while (j<len(sample_list) and k==sample_list[j]) :
            chain.extend([np.array(line.split()).astype(float)[plist]])
            j += 1
        
        if (j==len(sample_list)) :
            break
            
        k += 1

    chain = np.array(chain)
    lklhd = np.array(lklhd[sample_list])

    print(chain.shape,lklhd.shape)
    
    isrt = np.argsort(-lklhd)
    lklhd = lklhd[isrt]
    chain = chain[isrt,:]
    
    return chain,lklhd

