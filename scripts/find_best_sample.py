import numpy as np
import themispy as ty
import argparse as ap
import warnings
import sys

"""
Example ThemisPy script that makes use of the themispy.chain package.

Reads a Themis chain file and locates the highest likelihood sample, printing it and associated quantities.

For more information, run:

$ python3 find_best_sample.py -h

"""

# Create a parser object to manage command line options
parser = ap.ArgumentParser(description=("Reads a Themis ensemble chain, likelihood file pair" 
                                        " and determines the entry with the highest likelihood."
                                        " Optionally computes the chi-squared, the reduced chi-squared,"
                                        " and/or generates a fit_summaries.txt-style file from the best fit."))
parser.add_argument("-l","--lklhd",
                    type=str,
                    action='store',
                    required=True,
                    help="<Required> Name of likelihoood file to read.")
parser.add_argument("-c","--chain",
                    type=str,
                    action='store',
                    required=True,
                    help="<Required> Name of chain file to read.")
parser.add_argument("-q","--chisq",
                    type=str,
                    action='store',
                    help=("Name of chi-squared file to read.  Does not read this"
                          " coherently with likelihood and chain."))
parser.add_argument("-d","--dof",
                    type=int,
                    action='store',
                    help=("Number of degrees of freedom to assume for reduced chi-squared"
                          " computation.  Has no effect if -q,--chisq not set."))
parser.add_argument("-f",
                    type=str,
                    nargs='?',
                    action='store',
                    const="fs_max_lklhd.txt",
                    help=("Output a fit_summaries.txt file with the highest-likelihood point."
                          " If an argument is provided, it will be assumed to be the desired"
                          " file name."))
parser.add_argument("-m",
                    type=str,
                    nargs='?',
                    action='store',
                    const="means",
                    help=("Print initialization list appropriate for cutting and pasting"
                          " into driver file.  If a string is passed, it will assume the"
                          " parameters are of the form M[0], M[1], etc., are to be set."))
parser.add_argument("-v",
                    action='store_true',
                    default=False,
                    help="Verbose mode is set, printing the chain location directly to standard out.")
parser.add_argument("-n",
                    type=int,
                    action='store',
                    default=1,
                    help="Number of optimal samples to output")
parser.add_argument("-p","--parameters",
                    type=str,
                    nargs='+',
                    action='append',
                    help=("Selects a subset of parameters to output.  Parameter ranges may be"
                          " specified either via individual numbers (e.g., 0 1 4), comma separated"
                          " lists (e.g., 0, 2, 5), ranges (e.g., 0-3), or combinations thereof."))


# Get command line options
args = parser.parse_args()

print("Reading likelihood file %s"%(args.lklhd))
print("Reading chain file %s"%(args.chain))
chain,lklhd = ty.chain.most_likely_erun(args.chain,args.lklhd,samples=args.n)
print(("Highest likelihood is" + " %f"*len(lklhd))%tuple(lklhd))

# Sort out which parameter columns to keep
if (args.parameters is None) :
    plist = np.arange(chain.shape[1])
else :
    plist = np.array(ty.chain.parse_parameter_arglist(args.parameters[:]))
    
if (args.chisq is not None) :
    print("Reading chi-squared file %s"%(args.chisq))
    chisq = np.loadtxt(args.chisq)
    chisq_min = np.min(chisq)
    print("Minimum chi-squared is %f"%(chisq_min))
    
    if (args.dof is not None) :
        print("Reduced chi-squared is %f with %i DoF"%(chisq_min/args.dof,args.dof))

if (args.f is not None) :
    print("Outputing the highest-likelihood fit to %s"%(args.f))
    with open(args.f,'w') as fout :
        fout.write('%15s'%('# index'))
        for j in plist :
            fout.write(' %15s'%('p[%i]'%j))
        fout.write(' %15s'%('maxL'))
        fout.write(' %15s'%('notes'))
        fout.write('\n')
        for k in range(args.n) :
            fout.write('%15i'%(k))
            fout.write((plist.size*' %15.8g')%tuple(chain[k][plist]))
            fout.write(' %15.8f'%(lklhd[k]))
            fout.write('  generated by ThemisPy/scripts/find_best_sample.py')
            for a in sys.argv[1:] :
                fout.write(' %s'%(a))
            fout.write('\n')
        
if (args.v) :
    for k in range(args.n) :
        print((plist.size*' %15.8g')%tuple(chain[k,plist]))
        
if (args.m is not None) :
    print("Outputting initialization lines:")
    print("--------------------------------------------")
    for j in plist :
        print("  %s[%i] = %g;"%(args.m,j,chain[0,j]))
    print("--------------------------------------------")
    
          
