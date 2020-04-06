###########################
#
# Package:
#   diagnostic_plots
#
# Provides:
#   Functions for diagonostic plots from MCMC chains produced by Themis.
#



import numpy as np
from astropy.time import Time

# Read in ehtim, if possible
try:
    import ehtim as eh
    ehtim_found = True
except:
    warnings.warn("Package ehtim not found.  Some functionality will not be available.  If this is necessary, please ensure ehtim is installed.", Warning)
    ehtim_found = False




    
def closure_phase_covariance_ordering(obs, snrcut=0) :
    """
    Generates and returns a set of closure phases with the smallest off-diagonal 
    covariances possible.  For EHT data this is possible due to the presence of
    a uniquely sensitive anchor station, ALMA, and may not be generally possible
    for other data sets.

    Warning: 
      This makes extensive use of ehtim and will not be available if ehtim is not installed.  Raises a NotImplementedError if ehtim is unavailable.

    Args:
      obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object containing the observation data (presumably repackaging a uvfits file).
      snrcut (float): A possible signal-to-noise ratio below which to reject points. Default: 0.

    Returns:
      (numpy.recarray): An ehtim closure phase array.
    """

    if (ehtim_found is False) :
        raise NotImplementedError
    
    # make a copy
    obs2 = obs.copy()
    
    # compute a maximum set of closure phases
    obs.reorder_tarr_snr()
    obs.add_cphase(count='max', snrcut=snrcut)
    
    # compute a "minimum" set of closure phases
    obs2.reorder_tarr_snr()
    obs2.add_cphase(count='min', snrcut=snrcut)
    
    # organize some info
    time_vis = obs.data['time']
    time_cp = obs.cphase['time']
    t1_vis = obs.data['t1']
    t2_vis = obs.data['t2']
    
    # Determine the number of timestamps containing a closure triangle
    timestamps_cp = np.unique(time_cp)
    N_times_cp = len(timestamps_cp)
    
    # loop over all timestamps
    obs_cphase_arr = []
    for i in np.arange(0,N_times_cp,1):
    
        # get the current timestamp
        ind_here_cp = (time_cp == timestamps_cp[i])
        time_here = time_cp[ind_here_cp]
    
        # copy the cphase table for this timestamp
        obs_cphase_orig = np.copy(obs.cphase[ind_here_cp])
    
        # sort by cphase SNR
        snr = 1.0 / ((np.pi/180.0)*obs_cphase_orig['sigmacp'])
        ind_snr = np.argsort(snr)
        obs_cphase_orig = obs_cphase_orig[ind_snr]
        snr = snr[ind_snr]
    
        # organize the closure phase stations
        cp_ant1_vec = obs_cphase_orig['t1']
        cp_ant2_vec = obs_cphase_orig['t2']
        cp_ant3_vec = obs_cphase_orig['t3']
    
        # get the number of time-matched baselines
        ind_here_bl = (time_vis == timestamps_cp[i])
        B_here = ind_here_bl.sum()
    
        # organize the time-matched baseline stations
        bl_ant1_vec = t1_vis[ind_here_bl]
        bl_ant2_vec = t2_vis[ind_here_bl]
    
        # initialize the design matrix
        design_mat = np.zeros((ind_here_cp.sum(),B_here))
    
        # fill in each row of the design matrix
        for ii in range(ind_here_cp.sum()):
    
            # determine which stations are in this triangle
            ant1_here = cp_ant1_vec[ii]
            ant2_here = cp_ant2_vec[ii]
            ant3_here = cp_ant3_vec[ii]
            
            # matrix entry for first leg of triangle
            ind1_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here))
            if ind1_here.sum() == 0.0:
                ind1_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here))
                val1_here = -1.0
            else:
                val1_here = 1.0
            design_mat[ii,ind1_here] = val1_here
            
            # matrix entry for second leg of triangle
            ind2_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here))
            if ind2_here.sum() == 0.0:
                ind2_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here))
                val2_here = -1.0
            else:
                val2_here = 1.0
            design_mat[ii,ind2_here] = val2_here
            
            # matrix entry for third leg of triangle
            ind3_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant1_here))
            if ind3_here.sum() == 0.0:
                ind3_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant3_here))
                val3_here = -1.0
            else:
                val3_here = 1.0
            design_mat[ii,ind3_here] = val3_here
    
        # determine the expected size of the minimal set
        N_min = np.linalg.matrix_rank(design_mat)
        N_assumed = len(obs2.cphase['cphase'][obs2.cphase['time'] == timestamps_cp[i]])
    
        # print some info
        print('For timestamp '+str(timestamps_cp[i])+':')
    
        # get the current stations
        stations_here = np.unique(np.concatenate((cp_ant1_vec,cp_ant2_vec,cp_ant3_vec)))
        print('Observing stations are '+str([str(station) for station in stations_here]))
    
        print('Size of maximal set of closure phases = '+str(ind_here_cp.sum())+'.')
        print('Size of minimal set of closure phases = '+str(N_min)+'.')
        print('Size of minimal set of closure phases currently calculated by ehtim = '+str(N_assumed)+'.')
        print('...')
    
        ##########################################################
        # start of loop to recover minimal set
        ##########################################################
    
        # make a mask to keep track of which cphases will stick around
        keep = np.ones(len(obs_cphase_orig),dtype=bool)
        obs_cphase = obs_cphase_orig[keep]
    
        # remember the original minimal set size
        N_min_orig = N_min
    
        # initialize the loop breaker
        good_enough = False
    
        # perform the loop
        count = 0
        ind_list = []
        while good_enough == False:
    
            # recreate the mask each time
            keep = np.ones(len(obs_cphase_orig),dtype=bool)
            keep[ind_list] = False
            obs_cphase = obs_cphase_orig[keep]
    
            # organize the closure phase stations
            cp_ant1_vec = obs_cphase['t1']
            cp_ant2_vec = obs_cphase['t2']
            cp_ant3_vec = obs_cphase['t3']
    
            # get the number of time-matched baselines
            ind_here_bl = (time_vis == timestamps_cp[i])
            B_here = ind_here_bl.sum()
    
            # organize the time-matched baseline stations
            bl_ant1_vec = t1_vis[ind_here_bl]
            bl_ant2_vec = t2_vis[ind_here_bl]
    
            # initialize the design matrix
            design_mat = np.zeros((keep.sum(),B_here))
    
            # fill in each row of the design matrix
            for ii in range(keep.sum()):
    
                # determine which stations are in this triangle
                ant1_here = cp_ant1_vec[ii]
                ant2_here = cp_ant2_vec[ii]
                ant3_here = cp_ant3_vec[ii]
                
                # matrix entry for first leg of triangle
                ind1_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant2_here))
                if ind1_here.sum() == 0.0:
                    ind1_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant1_here))
                    val1_here = -1.0
                else:
                    val1_here = 1.0
                design_mat[ii,ind1_here] = val1_here
                
                # matrix entry for second leg of triangle
                ind2_here = ((bl_ant1_vec == ant2_here) & (bl_ant2_vec == ant3_here))
                if ind2_here.sum() == 0.0:
                    ind2_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant2_here))
                    val2_here = -1.0
                else:
                    val2_here = 1.0
                design_mat[ii,ind2_here] = val2_here
                
                # matrix entry for third leg of triangle
                ind3_here = ((bl_ant1_vec == ant3_here) & (bl_ant2_vec == ant1_here))
                if ind3_here.sum() == 0.0:
                    ind3_here = ((bl_ant1_vec == ant1_here) & (bl_ant2_vec == ant3_here))
                    val3_here = -1.0
                else:
                    val3_here = 1.0
                design_mat[ii,ind3_here] = val3_here
    
            # determine the size of the minimal set
            N_min = np.linalg.matrix_rank(design_mat)
    
            if (keep.sum() == N_min_orig) & (N_min == N_min_orig):
                good_enough = True
            else:
                if N_min == N_min_orig:
                    ind_list.append(count)
                else:
                    ind_list = ind_list[:-1]
                    count -= 1
                count += 1
    
            if count > len(obs_cphase_orig):
                break
    
        # print out the size of the recovered set for double-checking
        obs_cphase = obs_cphase_orig[keep]
        if len(obs_cphase) != N_min:
            print('*****************WARNING: minimal set not found*****************')
        else:
            print('Size of recovered minimal set = '+str(len(obs_cphase))+'.')
        print('========================================================================')
    
        obs_cphase_arr.append(obs_cphase)
    
    # save an output cphase file
    obs_cphase_arr = np.concatenate(obs_cphase_arr)
    return obs_cphase_arr
