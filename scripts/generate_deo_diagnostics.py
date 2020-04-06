from matplotlib import use,rc
use('Agg')

import numpy as np
from matplotlib.cbook import flatten
import argparse
import matplotlib.pyplot as plt

from themispy import diag as td
from themispy import chain as tc

rc('font',**{'family':'serif','serif':['Times','Palatino','Computer Modern Roman']})
rc('text', usetex=True)


parser = argparse.ArgumentParser(description=("Computes various diagnostics for DEO tempered Themis ensemble MCMC chain"
                                              " and associated plots."))

parser.add_argument("-ad","--annealing-data-file",
                    type=str,
                    action='store',
                    default='Annealing.dat',
                    help=("Annealing data file to be read in. Default: Annealing.dat"))
parser.add_argument("-as","--annealing-summary-file",
                    type=str,
                    action='store',
                    help=("Annealing summary file to be read in. Default: <annealing_data_file>+.summary"))
parser.add_argument("-o","--out",
                    type=str,
                    action="store",
                    default="deo_diagnostics.png",
                    help="Output file name. Default: deo_diagnostics.png")
parser.add_argument("--density",
                    action="store_true",
                    default=False,
                    help=("Sets the tempering level evolution plot to show the density of tempering levels"
                          " via the colormap. Default: False."))
parser.add_argument("--stream",
                    action="store_true",
                    default=False,
                    help=("Sets the colormap of the tempering level evolution plot to use a streamline"
                          " colormap. Default: False."))
parser.add_argument("-d","--diagnostics",
                    type=str,
                    nargs='+',
                    action='append',
                    help=("Sets the list of diagnostic plots to generate.  Takes a combination of the"
                          " letters t, r, l, which may be specified in any order and in any number of"
                          " option calls.  The appropriate plots will be made in the order trl.\n"
                          "   t ... tempering level evolution.\n"
                          "   r ... rejection rate evolution.\n"
                          "   l ... lambda vs beta.\n"
                          " Default: trl."))



# Get arguments
args = parser.parse_args()

print("Args:",args)

# Get a single diag string
if (args.diagnostics is None) :
    diagnostic_list = 'trl'
else :
    dlist = ''
    for d in flatten(args.diagnostics) :
        dlist = dlist + d
    # unique-ify
    diagnostic_list = ''
    for c in dlist :
        if (not c in diagnostic_list):
            diagnostic_list = diagnostic_list + c
        
print("Diagnostic list:",diagnostic_list)
        

figsizex = 3*len(diagnostic_list)
if ('l' in diagnostic_list) :
    figsizex = figsizex + 1
figsizey = 3

fig,axs = plt.subplots(1,len(diagnostic_list),figsize=(figsizex,figsizey),squeeze=False)
print(axs.shape)


# Read in data files
summary_data,beta,R = tc.load_deo_summary(args.annealing_data_file,args.annealing_summary_file)

###
# Start making plots

diagnostic_list_orig = diagnostic_list

# Tempering level evolution
ia=0
if ('t' in diagnostic_list) :

    if (args.density) :
        colormodel = 'density'
    else :
        colormodel = 'tempering_level'

    if (args.stream) :
        cmap = td.generate_streamline_colormap('plasma')
    else :
        cmap = 'plasma'

    print("  Generating tempering level evolution plot.")
    plt.sca(axs[0,ia])
    td.plot_deo_tempering_level_evolution(beta,colormodel=colormodel,colormap=cmap)
    diagnostic_list = diagnostic_list.replace('t','')
    ia += 1


# Rejection rate evolution
if ('r' in diagnostic_list) :
    print("  Generating rejection rate evolution plot.")
    plt.sca(axs[0,ia])
    td.plot_deo_rejection_rate(summary_data,color='k')
    diagnostic_list = diagnostic_list.replace('r','')
    ia += 1
    

# lambda vs beta plot
if ('l' in diagnostic_list) :
    print("  Generating lambda vs beta plot.")
    plt.sca(axs[0,ia])
    td.plot_deo_lambda(summary_data,beta,R,colormap='plasma')
    diagnostic_list = diagnostic_list.replace('l','')
    ia += 1
    

if (not diagnostic_list is '') :
    print("ERROR: Unrecognized diagnostics requested: %s"%(diagnostic_list))
    quit()


# Save figure
plt.tight_layout()
figsize=plt.gcf().get_size_inches()
if ('l' in diagnostic_list_orig) :
    plt.subplots_adjust(right=1.0-1.0/figsize[0])
plt.savefig(args.out,dpi=300)

    
    


