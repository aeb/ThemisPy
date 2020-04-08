from matplotlib import use,rc
use('Agg')

import numpy as np
import argparse as ap
import warnings
from matplotlib.cbook import flatten
import matplotlib.pyplot as plt

import themispy.vis as tv


"""
Reads in a Themis gain file and generates plots of the amplitude and phase corrections for each station.

For more information, run:

$ python3 generate_gain_plots.py -h

"""

# Create a parser object to manage command line options
parser = ap.ArgumentParser(description=("Reads a Themis gain file and generates plots of the amplitude and"
                                        " phase corrections for each station."))

parser.add_argument("-g","--gain-file",
                    type=str,
                    action='store',
                    required=True,
                    help=("<Required> Name of gain file to read."))
parser.add_argument("-i","--invert",
                    action='store_true',
                    default=False,
                    help=("Invert gains prior to plotting. This is as if the gains were applied to the data (heresy!)."))
parser.add_argument("-dt",
                    action='store_false',
                    default=True,
                    help=("Plots the gains gains the time difference relative to the first epoch instead of absolute time."))
parser.add_argument("-ns","--no-station-labels",
                    action='store_false',
                    default=True,
                    help=("Remove station labels"))
parser.add_argument("-gp","--gain-plots",
                    type=str,
                    nargs='+',
                    action='append',
                    help=("Sets the list of gain plots to generate.  Takes a combination of the letters 'a' (amplitudes),"
                          " 'p' (phases), which may be specified in any order and in any number of option calls.  The"
                          " appropriate plots will be made in the order ap."))
parser.add_argument("-o","--out",
                    type=str,
                    action="store",
                    default="gain_plot.png",
                    help="Output file name. Default: gain_plot.png")
parser.add_argument("--no-grid",
                    action='store_false',
                    default=True,
                    help=("Turn off grid lines. Default: True."))
parser.add_argument("--usetex",
                    action='store_true',
                    default=False,
                    help=("Turns on using LaTex in the rendering.  Note that this takes MUCH longer,"
                          " but the result is MUCH better looking."))


# Get command line options
args = parser.parse_args()


# Use LaTex or not
if (args.usetex) :
    rc('font',**{'family':'serif','serif':['Times','Palatino','Computer Modern Roman']})
    rc('text', usetex=True)


# Parse plots to generate
if (args.gain_plots is None) :
    gain_plots_list = 'ap'
else :
    gplist = ''
    for gp in flatten(args.gain_plots) :
        gplist = gplist + gp
    # unique-ify
    gain_plots_list = ''
    for c in gplist :
        if (not c in gain_plots_list):
            gain_plots_list = gain_plots_list + c
print("Gain plots to be generaed:",gain_plots_list)


# Read in the gain file
gain_data = tv.read_gain_file(args.gain_file)

# Generate plot fig/axes
figsizex = 3*len(gain_plots_list)
if ('l' in gain_plots_list) :
    figsizex = figsizex + 1
figsizey = 3
fig,axs = plt.subplots(1,len(gain_plots_list),figsize=(figsizex,figsizey),squeeze=False)

# gain amplitudes
ia=0
if ('a' in gain_plots_list) :
    print("  Generating gain amplitude plot.")
    plt.sca(axs[0,ia])
    tv.plot_station_gain_amplitudes(gain_data,absolute_time=args.dt,invert=args.invert,grid=args.no_grid,add_station_labels=args.no_station_labels)
    gain_plots_list = gain_plots_list.replace('a','')
    ia += 1
    
# gain phases
if ('p' in gain_plots_list) :
    print("  Generating gain phase plot.")
    plt.sca(axs[0,ia])
    tv.plot_station_gain_phases(gain_data,absolute_time=args.dt,invert=args.invert,grid=args.no_grid,add_station_labels=args.no_station_labels)
    gain_plots_list = gain_plots_list.replace('p','')
    ia += 1

if (not gain_plots_list is '') :
    print("ERROR: Unrecognized gain plot requested: %s"%(gain_plots_list))
    quit()


# Save figure
plt.tight_layout()
plt.savefig(args.out,dpi=300)
