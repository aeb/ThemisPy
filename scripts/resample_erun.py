import numpy as np
import themispy as ty
import argparse as ap
import warnings
import sys
from matplotlib.cbook import flatten


"""
Example ThemisPy script that makes use of the themispy.chain package.

Reads a Themis chain file and resamples it.

For more information, run:

$ python3 sample_erun.py -h

"""

# Create a parser object to manage command line options
parser = ap.ArgumentParser(description=("Reads a Themis ensemble chain file, and optionally a likelihood file"
                                        " and samples it the desired number of times after a specified burn-in"
                                        " period.  Optionally generates a resampled likelihood file."))
parser.add_argument("-c","--chain",
                    type=str,
                    nargs="+",
                    action='append',
                    required=True,
                    help=("<Required> Name of chain file to read.  If multiple files are listed, they will"
                          " independently be sampled."))
parser.add_argument("-l","--lklhd",
                    type=str,
                    nargs="+",
                    action='append',
                    default=[],
                    help=("Name of likelihoood file to read.  If multiple files are listed, they will"
                          " independently be sampled in order with the chain files.  That is, the first"
                          " likelihood file will be coherently sampled with the first chain file, the"
                          " second likelihood file will be coherently sampled with the second chain file,"
                          " etc."))
parser.add_argument("-s","--samples",
                    type=int,
                    action='store',
                    default=10000,
                    help=("Number of samples to draw from the chain."))
parser.add_argument("-bf","--burn-fraction",
                    type=float,
                    action='store',
                    default=0,
                    help=("Fraction of chain to skip."))
parser.add_argument("--postfix",
                    type=str,
                    action='store',
                    default="_sample",
                    help="Postfix to add to chain [and likelihood] file names produced after sampling.")
parser.add_argument("-p","--parameters",
                    type=str,
                    nargs='+',
                    action='append',
                    help=("Selects a subset of parameters to output.  Parameter ranges may be"
                          " specified either via individual numbers (e.g., 0 1 4), comma separated"
                          " lists (e.g., 0, 2, 5), ranges (e.g., 0-3), or combinations thereof."
                          " By default, all parameters are sampled."))


def strip_filename(filename) :
    tokens = filename.split('.')
    if (len(tokens)>1) :
        ext = tokens[-1]
    else :
        ext = ''
    stripped_filename = filename[:-(len(ext)+1)]
    return stripped_filename, ext


# Get command line options
args = parser.parse_args()


# Flatten the list of filenames
chain_file_list = list(flatten(args.chain))
lklhd_file_list = list(flatten(args.lklhd))

# Sort out which parameter columns to keep
if (args.parameters is None) :
    plist = None
else :
    plist = ty.chain.parse_parameter_arglist(args.parameters[:])

print("")
print("Will generate %i samples after skipping %.2g%% of the run"%(args.samples,100*args.burn_fraction))
print("from %i chain file(s) and %i likelihood file(s)."%(len(chain_file_list),len(lklhd_file_list)))
print("--------------------------------------------------------------")
    
# Loop over chain filenames
for j,chain_file in enumerate(chain_file_list) :

    # Get new chain file name
    new_chain_file,ext = strip_filename(chain_file)
    new_chain_file = new_chain_file+args.postfix+'.'+ext



    # If we have an accompanying likelihood file, coherently load the erun, otherwise sample the chain alone
    if (len(lklhd_file_list)>j) :
        lklhd_file = lklhd_file_list[j]

        # Get new likelihood file name
        new_lklhd_file,ext = strip_filename(lklhd_file)
        new_lklhd_file = new_lklhd_file+args.postfix+'.'+ext
        #print("Reading chain file %s and likelihood file %s"%(chain_file,lklhd_file))
        #print("Writing chain file %s and likelihood file %s"%(new_chain_file,new_lklhd_file))

        print("  (%s, %s) --> (%s, %s)"%(chain_file,lklhd_file,new_chain_file,new_lklhd_file))
        
        chain,lklhd = ty.chain.sample_erun(chain_file,lklhd_file,samples=args.samples,burn_fraction=args.burn_fraction,parameter_list=plist)
        np.savetxt(new_chain_file,chain,fmt='%15.8g')
        np.savetxt(new_lklhd_file,lklhd,fmt='%15.8g')
        
    else :

        #print("Reading chain file %s"%(chain_file))
        #print("Writing chain file %s"%(new_chain_file))

        print("  (%s) --> (%s)"%(chain_file,new_chain_file))

        chain = ty.chain.sample_chain(chain_file,samples=args.samples,burn_fraction=args.burn_fraction,parameter_list=plist)
        np.savetxt(new_chain_file,chain,fmt='%15.8g')


print("")
