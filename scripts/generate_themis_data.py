from matplotlib import use,rc
use('Agg')

import numpy as np
import argparse as ap
import warnings
from matplotlib.cbook import flatten
import matplotlib.pyplot as plt

import themispy.data as tydat

# Read in ehtim, if possible
try:
    import ehtim as eh
    ehtim_found = True
except:
    warnings.warn("Package ehtim not found.  Some functionality will not be available.  If this is necessary, please ensure ehtim is installed.", Warning)
    ehtim_found = False




"""
Reads in a uvfits files and produces the desired data files for Themis.

For more information, run:

$ python3 generate_themis_data.py -h

"""

# Create a parser object to manage command line options
parser = ap.ArgumentParser(description=("Reads in a uvfits file and produces the desired data files for Themis"
                                        " with various additions.  These include the ability to scan average the"
                                        " data, addition of fractional systematic error components, addition of"
                                        " fixed systematic error components, flagging of stations, and removal"
                                        " of baselines below some cutoff."))

parser.add_argument("uvfits_files",
                    type=str,
                    nargs='+',
                    action='append',
                    help=("<Required> Name of uvfits files to be processed."))
parser.add_argument("-d","--data-type",
                    type=str,
                    nargs='+',
                    action='append',
                    required=True,
                    help=("Data types to generate.  Accepted types include x,v,a,p for crosshands,"
                          " complex visibilities, amplitudes, closure phases.  May be specified in"
                          " independent calls. Default: None."))
parser.add_argument("-s","--scan-avg",
                    action='store_true',
                    default=False,
                    help=("Produce scan averaged data."))
parser.add_argument("--avg-time",
                    type=float,
                    action='store',
                    help=("Timescale over which to average the data in seconds. Superseded by -s,--scan-avg. Default: None."))
parser.add_argument("-f","--fractional-systematic-error",
                    type=float,
                    action='store',
                    help=("Fractional systematic error to add IN PERCENT. Default: None."))
parser.add_argument("-t","--threshold-systematic-error",
                    type=float,
                    action='store',
                    help=("Constant systematic error threshold to add in mJy. Default: None."))
parser.add_argument("-x","--exclude-station",
                    type=str,
                    nargs='+',
                    action='append',
                    help=("List of stations to flag.  Stations must be specified with the same"
                          " station codes appearing in the uvfits file.  E.g., the 2017 code list."
                          " Default: None."))
parser.add_argument("-b","--minimum-baseline",
                    type=float,
                    action='store',
                    help=("Minimum baseline, below which data are excluded from ALL data generation."
                          " Default: None."))
parser.add_argument("--no-debias",
                    action='store_false',
                    default=True,
                    help=("Do not debias visibility amplitude data. Default: debias amplitudes."))
parser.add_argument("-snr","--SNR-cut",
                    type=float,
                    action='store',
                    help=("SNR cut to apply to all data products unless specific cuts are specified."))
parser.add_argument("--crosshand-SNR-cut",
                    type=float,
                    action='store',
                    help=("SNR cut to apply to crosshand data."))
parser.add_argument("--visibility-SNR-cut",
                    type=float,
                    action='store',
                    help=("SNR cut to apply to complex visibility data."))
parser.add_argument("--amplitude-SNR-cut",
                    type=float,
                    action='store',
                    help=("SNR cut to apply to amplitude data."))
parser.add_argument("--closure-phase-SNR-cut",
                    type=float,
                    action='store',
                    help=("SNR cut to apply to closure phase data."))
parser.add_argument("-er5",
                    action='store_true',
                    default=False,
                    help=("Indicates that this is an ER5 data set that suffers from the polsolve"
                          " bug and must be corrected. Default: False."))
parser.add_argument("-xph","--exclude-partial-hands",
                    action='store_false',
                    default=True,
                    help=("Excludes partial hand data (e.g., JCMT) in the construction of crosshand"
                          " visibility data sets. Default: include partials."))


def strip_filename(filename) :
    tokens = filename.split('.')
    if (len(tokens)>1) :
        ext = tokens[-1]
    else :
        ext = ''
    stripped_filename = filename[:-(len(ext)+1)]
    return stripped_filename, ext


def read_uvfits(filename,polrep,args) :

    # load in the data
    obs = eh.obsdata.load_uvfits(filename,polrep=polrep)

    # exclude stations
    if (not args.exclude_station is None) :
        obs = obs.flag_sites(list(flatten(args.exclude_station)))

    # (scan) average the data
    if (args.scan_avg) :
        obs.add_scans()
        obs = obs.avg_coherent(0.0, scan_avg=True)
    elif (not args.avg_time is None) :
        obs = obs.avg_coherent(args.avg_time)
        
    # add systematic noise
    if (not args.fractional_systematic_error is None) :
        obs = obs.add_fractional_noise(args.fractional_systematic_error/100.0)

    # add threshold
    if (not args.threshold_systematic_error is None):
        if (polrep=='stokes') :
            obs.data['sigma'] = np.sqrt(obs.data['sigma']**2 + (args.threshold_systematic_error/1.0e3)**2)
        else :
            for sigtype in ['rrsigma','llsigma','rlsigma','lrsigma'] :
                obs.data[sigtype] = np.sqrt(obs.data[sigtype]**2 + (args.threshold_systematic_error/1.0e3)**2)

    # remove baselines below cutoff
    if (not args.minimum_baseline is None) :
        obs.flag_uvdist(args.minimum_baseline)
                
    return obs



# Check for ehtim
if (not ehtim_found) :
    print("ERROR: ehtim must be installed to generate Themis data from uvfits files currently.")
    quit()



# Get command line options
args = parser.parse_args()

# Parse data generation options
dlist = ''
for d in list(flatten(args.data_type)) :
    dlist = dlist + d
# unique-ify
data_type_list_master = ''
for d in dlist :
    if (not d in data_type_list_master):
        data_type_list_master = data_type_list_master + d
print("Data types to be generated:",data_type_list_master)

# Begin loop through uvfits files
for (iuv,uvfits) in enumerate(list(flatten(args.uvfits_files))) :

    print("Reading data from %s (%i of %i)"%(uvfits,iuv,len(list(flatten(args.uvfits_files)))))

    # Generate output file suffix
    outfile_suffix=strip_filename(uvfits)[0]
    if (args.scan_avg) :
        outfile_suffix += '_scanavg'
    elif (not args.avg_time is None) :
        outfile_suffix += '_avg%g'%(args.avg_time)
    if (not args.fractional_systematic_error is None) :
        outfile_suffix += '_f%g'%(args.fractional_systematic_error)
    if (not args.threshold_systematic_error is None) :
        outfile_suffix += '_t%gmJy'%(args.threshold_systematic_error)
    outfile_suffix += '_tygtd.dat'


    # Begin running through implemented data types
    data_type_list = data_type_list_master

    
    # Crosshands
    if ('x' in data_type_list) :

        # Generate the output file name
        outfile = 'X_'+outfile_suffix
        print("Writing crosshand data to %s"%(outfile))

        # Read observation
        obs = read_uvfits(uvfits,'circ',args)

        # Determine snrcut
        if (not args.crosshand_SNR_cut is None) :
            snrcut = args.crosshand_SNR_cut
        elif (not args.SNR_cut is None) :
            snrcut = args.SNR_cut
        else :
            snrcut = 0.0
        
        # Write data file
        tydat.write_crosshand_visibilities(obs, outfile, isER5=args.er5, snrcut=snrcut, keep_partial_hands=args.exclude_partial_hands)

        # Remove from data_type_list
        data_type_list = data_type_list.replace('x','')


    # Visibility
    if ('v' in data_type_list) :

        # Generate the output file name
        outfile = 'V_'+outfile_suffix
        print("Writing visibility data to %s"%(outfile))

        # Read observation
        obs = read_uvfits(uvfits,'stokes',args)

        # Determine snrcut
        if (not args.visibility_SNR_cut is None) :
            snrcut = args.visibility_SNR_cut
        elif (not args.SNR_cut is None) :
            snrcut = args.SNR_cut
        else :
            snrcut = 0.0
        
        # Write data file
        tydat.write_visibilities(obs, outfile, snrcut=snrcut)

        # Remove from data_type_list
        data_type_list = data_type_list.replace('v','')
        

    # Amplitudes
    if ('a' in data_type_list) :

        # Generate the output file name
        outfile = 'VM_'+outfile_suffix
        print("Writing amplitude data to %s"%(outfile))

        # Read observation
        obs = read_uvfits(uvfits,'stokes',args)

        # Determine snrcut
        if (not args.amplitude_SNR_cut is None) :
            snrcut = args.amplitude_SNR_cut
        elif (not args.SNR_cut is None) :
            snrcut = args.SNR_cut
        else :
            snrcut = 0.0
        
        # Write data file
        tydat.write_amplitudes(obs, outfile, debias_amplitudes=args.no_debias, snrcut=snrcut)

        # Remove from data_type_list
        data_type_list = data_type_list.replace('a','')

        
    # Closure phases
    if ('p' in data_type_list) :

        # Generate the output file name
        outfile = 'CP_'+outfile_suffix
        print("Writing amplitude data to %s"%(outfile))

        # Read observation
        obs = read_uvfits(uvfits,'stokes',args)

        # Determine snrcut
        if (not args.closure_phase_SNR_cut is None) :
            snrcut = args.closure_phase_SNR_cut
        elif (not args.SNR_cut is None) :
            snrcut = args.SNR_cut
        else :
            snrcut = 0.0
        
        # Write data file
        tydat.write_closure_phases(obs, outfile, snrcut=snrcut)

        # Remove from data_type_list
        data_type_list = data_type_list.replace('p','')


    if (not data_type_list is '') :
        print("ERROR: Unrecognized data types requested: %s"%(data_type_list))
        quit()
        

    
        
