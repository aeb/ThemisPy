from matplotlib import use,rc
use('Agg')

import numpy as np
from matplotlib.cbook import flatten
import argparse

from themispy import diag as td
from themispy import chain as tc


#rc('font',**{'family':'serif','serif':['Times','Palatino','Computer Modern Roman']})
#rc('text', usetex=True)


"""
Reads in chain and likelihood files from Themis analyses and computes and plots various diagnostics.

For more information, run:

$ python3 generate_chain_diagnostics.py -h

"""


parser = argparse.ArgumentParser(description=("Computes various chain diagnostics for Themis ensemble MCMC chain data."
                                              " Diagnostics generated include split-rhat, , for Themis style arguments."))

parser.add_argument("-c","--chain",
                    type=str,
                    nargs='+',
                    action='append',
                    required=True,
                    help=("<Required> Chain files to read in, can pass multiple. Must be equal to"
                          " number of likelihood files to read in"))
parser.add_argument("-l","--lklhd",
                    type=str,
                    nargs='+',
                    action='append',
                    required=True,
                    help=("<Required> Lklhd files to read in, can pass multiple. Must be equal to"
                          " number of chain files to read in"))
parser.add_argument("-bf","--burn-fraction","-w", "--warmup",
                    type=float,
                    action='store',
                    default=0.5,
                    help=("Fraction of chain to skip.  Default: 0.5."))
parser.add_argument("--stride","--nthin",
                    type=int,
                    default=1,
                    help="How much to thin the chain by. The DEFAULT is 1, i.e. no thinning")
parser.add_argument("-p","--parameters",
                    type=str,
                    nargs='+',
                    action='append',
                    help=("Selects a subset of parameters to output.  Parameter ranges may be"
                          " specified either via individual numbers (e.g., 0 1 4), comma separated"
                          " lists (e.g., 0, 2, 5), ranges (e.g., 0-3), or combinations thereof."
                          " By default, all parameters are outputted."))
parser.add_argument("-o","--out",
                    type=str,
                    action="store",
                    default="chain_diagnostics.out",
                    help="Output file name. Default: chain_diagnostics.out")
parser.add_argument("-s","--save",
                    action="store_true",
                    help=("Flag to turn on whether the save the processed likelihood and chain files,"
                          " i.e. thinned and burn-in removed"))
parser.add_argument("--hist",
                    action="store_true",
                    help="Flag to create rank plots for the chains read in.")

args = parser.parse_args()
chainsname = list(flatten(args.chain))
lklhdname = list(flatten(args.lklhd))
nthin = args.stride
warmup_frac = args.burn_fraction
plist = args.parameters
save = args.save
plot = args.hist
if save:
    print("Will save the processed chains and likelihoods")

# Sort out which parameter columns to keep
if (args.parameters is None) :
    plist = None
else :
    plist = np.array(tc.parse_parameter_arglist(args.parameters[:]))

out = args.out
print("Reading in chains:", chainsname)
print("with Lklhd:", lklhdname)
#print("Reading in the ", plist, " columns of the chain")

print("I am cutting the first {:.2f} of the chains, and thinning by a factor of {:2d}.".format(warmup_frac, nthin))

assert len(chainsname)==len(lklhdname), "Must have equal number of chain and lklhd files to work!"

chains = []
lklhds = []
for i in range(len(chainsname)):
    c, l = tc.load_erun(chainsname[i],lklhdname[i], stride=nthin, parameter_list=plist, burn_fraction=warmup_frac)
    if (c.shape[0] != l.shape[0]):
        print("Error reading in the file! Number of steps in chain and likelihood are different")
    chains.append(c)
    #print(c.shape)
    if save:
        np.savetxt(chainsname[i]+".proc", c.reshape(c.shape[0]*c.shape[1],c.shape[2]))
        np.savetxt(lklhdname[i]+".proc", l)
    #lklhds.append(l)

chains = tc.join_echains(chains)
iact, mean_rhat, var_rhat = td.save_ensemble_diagnostics(out, chains, chainsname, stride=nthin, method="rank")
iactmax = np.argmax(iact)

if plot:
    #print(np.array(chains).shape)
    chains_split = np.vstack(np.array_split(np.array(chains),2,axis=1))
    fig,_ = td.plot_rank_hist(np.mean(chains_split[:,:,:,iactmax],axis=2), names=chainsname)
    fig.suptitle("Mean rank plots for param "+str(iactmax)
                 +"\n $\hat{R}_{\mu} = %.4f$"%(mean_rhat[iactmax]))
    #fig.tight_layout()
    outfigmean = (".".join(out.split(".")[:-1]))+"_rankmean.pdf"
    fig.savefig(outfigmean)
    fig,_ = td.plot_rank_hist(np.var(chains_split[:,:,:,iactmax],axis=2,ddof=1), names=chainsname)
    outfigvar = (".".join(out.split(".")[:-1]))+"_rankvar.pdf"
    fig.suptitle("Var rank plots for param "+str(iactmax)
                 +"\n $\hat{R}_{\sigma^2} = %.4f$"%(var_rhat[iactmax]))
    #fig.tight_layout()
    fig.savefig(outfigvar)
