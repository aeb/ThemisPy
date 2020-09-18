###########################
#
# Package:
#   data
#
# Provides:
#   Functions for preprocessing uvfits files to produce Themis-formatted data files
#   and to generate uvfits files from Themis results.
#

from themispy.utils import *

import numpy as np

# Read in astropy, if possible
try:
    from astropy.time import Time
    astropy_found = True
except:
    warnings.warn("Package astropy not found.  Some functionality will not be available.  If this is necessary, please ensure astropy is installed.", Warning)
    astropy_found = False

# # Read in ehtim, if possible
# try:
#     import ehtim as eh
#     ehtim_found = True
# except:
#     warnings.warn("Package ehtim not found.  Some functionality will not be available.  If this is necessary, please ensure ehtim is installed.", Warning)
#     ehtim_found = False


def closure_phase_covariance_ordering(obs, snrcut=0, verbosity=0) :
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
      verbosity (int): Verbosity level. 0 prints only warnings and errors.  1 prints information about each observation period.

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
        if (verbosity>0) :
            print('For timestamp '+str(timestamps_cp[i])+':')
    
        # get the current stations
        stations_here = np.unique(np.concatenate((cp_ant1_vec,cp_ant2_vec,cp_ant3_vec)))
        if (verbosity>0) :
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
        if len(obs_cphase) != N_min :
            warnings.warn("Minimal set of closure phases not found.", Warning)
            if (verbosity>0) :
                print('*****************WARNING: minimal set not found*****************')
        else:
            if (verbosity>0) :
                print('Size of recovered minimal set = '+str(len(obs_cphase))+'.')
        if (verbosity>0) :
            print('========================================================================')
    
        obs_cphase_arr.append(obs_cphase)
    
    # save an output cphase file
    obs_cphase_arr = np.concatenate(obs_cphase_arr)
    return obs_cphase_arr


def reconstruct_field_rotation_angles(obs, isER5=False) :
    """
    Reconstructs the field rotation angle for polarization data from sub-quantities.  Also 
    implements, upon request, corrections that are appropriate for ER5 data.  Should be
    deprecated for data generated following ER5 (e.g., upon the release of ER6 and for 
    subsequent intervening releases).

    Warning: 
      This makes extensive use of ehtim and will not be available if ehtim is not installed.  Raises a NotImplementedError if ehtim is unavailable.

    Args:
      obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object containing the observation data (presumably repackaging a uvfits file).
      isER5 (bool): Flag to indicate that this is an ER5 file with polconvert errors that modify the field rotation angles. Default: False.

    Returns:
      (numpy.ndarray,numpy.ndarray): Field rotation angles for station1 and station2 for the full data set in obs.
    """

    if (ehtim_found is False) :
        raise NotImplementedError
    
    
    # An error in polconvert imparted phase errors in the ER5 R and L products that
    # propagated into RR,LL,RL,LL.  This must be fixed ONLY for ER5 data products.
    # WARNING: This should NOT be applied to any other data product!  For example,
    # simulated data or subsequent data releases (ER6).
    if (isER5) :
        warnings.warn("Correcting the absolute EVPA calibration in ER5 data.  Should not be done for any other data (simulated, later releases, etc.).", Warning)
        datadict = {t['site']:np.array([(0.0, 0.0 + 1j*1.0, 1.0 + 1j*0.0)], dtype=eh.DTCAL) for t in obs.tarr}
        caltab = eh.caltable.Caltable(obs.ra,obs.dec,obs.rf,obs.bw,datadict,obs.tarr,obs.source,obs.mjd)
        obs = caltab.applycal(obs, interp='nearest',extrapolate=True)

    ####
    ## Construct field rotation correction vectors

    # Get vector of station names
    ant1 = np.copy(obs.data['t1'])
    ant2 = np.copy(obs.data['t2'])
    
    # Get vector of elevations
    el1 = obs.unpack(['el1'],ang_unit='rad')['el1']
    el2 = obs.unpack(['el2'],ang_unit='rad')['el2']
    
    # Get vector of parallactic angles
    par1 = obs.unpack(['par_ang1'],ang_unit='rad')['par_ang1']
    par2 = obs.unpack(['par_ang2'],ang_unit='rad')['par_ang2']

    # Apparently simple linear relationship between the el,par,telescope-specific offset, and the 
    # field rotation angle. All of these single constants that depend on mount type, etc., and are -1,0,1.
    # Station 1:
    f_el1 = np.zeros_like(el1)
    f_par1 = np.zeros_like(par1)
    f_off1 = np.zeros_like(el1)
    for ia, a1 in enumerate(ant1):
        ind1 = (obs.tarr['site'] == a1)
        f_el1[ia] = obs.tarr[ind1]['fr_elev']
        f_par1[ia] = obs.tarr[ind1]['fr_par']
        f_off1[ia] = obs.tarr[ind1]['fr_off']*eh.DEGREE
    # Station 2:
    f_el2 = np.zeros_like(el2)
    f_par2 = np.zeros_like(par2)
    f_off2 = np.zeros_like(el2)
    for ia, a2 in enumerate(ant2):
        ind2 = (obs.tarr['site'] == a2)
        f_el2[ia] = obs.tarr[ind2]['fr_elev']
        f_par2[ia] = obs.tarr[ind2]['fr_par']
        f_off2[ia] = obs.tarr[ind2]['fr_off']*eh.DEGREE

    # Compute the field rotation angles, phi, for each station
    FR1 = (f_el1*el1) + (f_par1*par1) + f_off1
    FR2 = (f_el2*el2) + (f_par2*par2) + f_off2

    # Convert to radians
    #FR1 = FR1*np.pi/180.
    #FR2 = FR2*np.pi/180.

    # obs.switch_polrep('stokes')

    return FR1,FR2


def write_crosshand_visibilities(obs, outname, isER5=False, snrcut=0, keep_partial_hands=True, flip_field_rotation_angles=False, eht_field_rotation_convention=True) :
    """
    Writes complex crosshand RR,LL,RL,LR visibilities in Themis format given an :class:`ehtim.obsdata.Obsdata` object.

    Warning: 
      * The :class:`ehtim.obsdata.Obsdata` must be read in with a polrep='circ'.  If this is not the case, raises a ValueError.  DO NOT SWITCH THE POLREP AS THIS INFLATES THE ERRORS.
      * This makes extensive use of ehtim and astropy and will not be available if either is not installed.  Raises a NotImplementedError if they are is unavailable.

    Args:
      obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object containing the observation data (presumably repackaging a uvfits file).
      outname (str): Name of the output file to which to write data.
      isER5 (bool): Flag to indicate that this is an ER5 file with polconvert errors that modify the field rotation angles. Default: False.
      snrcut (float): A possible signal-to-noise ratio below which to reject points. Default: 0.
      keep_partial_hands (bool): Flag to deterime how to treat data with incomplete polarization information.  If true, the single-hand visibilities are kept, and large errors are assigned to correlation products that include other hands. Default: True.
      eht_field_rotation_convention (bool): Flag to determine if a derotation of the field rotation angles is required, as is the case for data produced by standard EHT pipelines. Default: True.
    Returns:
      None.
    """

    if (ehtim_found is False) :
        raise NotImplementedError

    if (astropy_found is False) :
        raise NotImplementedError

    # Check polrep
    if (obs.polrep!='circ') :
        warnings.warn("Must supply an obs object with polrep='circ'", Warning)
        raise ValueError

    # Get some dataset particulars
    src = obs.source
    mjd = obs.mjd
    t=Time(mjd,format='mjd')
    [year,day]=map(int,t.yday.split(':')[:2])
    
    # Get the field rotation angles
    fr1,fr2=reconstruct_field_rotation_angles(obs,isER5=isER5)

    # Flip (debugging)
    if (flip_field_rotation_angles) :
        fr1 = -fr1
        fr2 = -fr2
        
    # Write header
    out=open(outname,'w')
    out.write('#%24s %4s %4s %15s %6s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n'%('source','year',' day','time (hr)','base','u (Ml)','v (Ml)','fr1 (rad)', 'fr2 (rad)','RR.r (Jy)','RRerr.r (Jy)','RR.i (Jy)','RRerr.i (Jy)','LL.r (Jy)','LLerr.r (Jy)','LL.i (Jy)','LLerr.i (Jy)','RL.r (Jy)','RLerr.r (Jy)','RL.i (Jy)','RLerr.i (Jy)','LR.r (Jy)','LRerr.r (Jy)','LR.i (Jy)','LRerr.i (Jy)'))

    # Loop over data and output
    for ii,d in enumerate(obs.data) :

        time = d['time']
        bl = d['t1']+d['t2']
        u = d['u']/1e6
        v = d['v']/1e6
        RR = d['rrvis']
        RRerr = d['rrsigma']
        LL = d['llvis']
        LLerr = d['llsigma']
        RL = d['rlvis']
        RLerr = d['rlsigma']
        LR = d['lrvis']
        LRerr = d['lrsigma']


        # Pre-rotate by the field rotation angles if not EHT data to match the EHT definition
        if (eht_field_rotation_convention==False) :
            #efr1 = np.exp(1j*fr1[ii])
            #efr2 = np.exp(1j*fr2[ii])
            #RR = RR * efr1*np.conj(efr2)
            #LL = LL * np.conj(efr1)*efr2
            #RL = RL * efr1*efr2
            #LR = LR * np.conj(efr1)*np.conj(efr2)
            
            fr1[ii] *= 0.5
            fr2[ii] *= 0.5
            
        SNR = (np.abs(RR)+np.abs(LL))/(RRerr+LLerr)
        
        # If we want to still use only partial-hand visibilities, 
        if (keep_partial_hands) :
            if (np.isnan(RR)) :
                RR = 0.0+1j*0.0
                RRerr = 100.0
                SNR = np.abs(LL)/LLerr
            if (np.isnan(LL)) :
                LL = 0.0+1j*0.0
                LLerr = 100.0
                SNR = np.abs(RR)/RRerr
            if (np.isnan(RL)) :
                RL = 0.0+1j*0.0
                RLerr = 100.0
            if (np.isnan(LR)) :
                LR = 0.0+1j*0.0
                LRerr = 100.0
                
        # Only output data that does not include nans
        if (np.isnan([RR,LL,RL,LR]).any()==False) :
            if (SNR>snrcut) :
                out.write('%25s %4i %4i %15.8f %4s %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n'%(src,year,day,time,bl,u,v,fr1[ii],fr2[ii],RR.real,RRerr,RR.imag,RRerr,LL.real,LLerr,LL.imag,LLerr,RL.real,RLerr,RL.imag,RLerr,LR.real,LRerr,LR.imag,LRerr))
        else :
            warnings.warn("NaN crosshand visibilities found on %4s baseline, perhaps one hand is missing?"%(bl), Warning)

    out.close()




def write_visibilities(obs, outname, snrcut=0) :
    """
    Writes complex visibilities (Stokes I) in Themis format given an :class:`ehtim.obsdata.Obsdata` object.

    Warning: 
      * The :class:`ehtim.obsdata.Obsdata` must be read in with a polrep='stokes'.  If this is not the case, raises a ValueError.  DO NOT SWITCH THE POLREP AS THIS INFLATES THE ERRORS.
      * This makes extensive use of ehtim and astropy and will not be available if either is not installed.  Raises a NotImplementedError if they are is unavailable.

    Args:
      obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object containing the observation data (presumably repackaging a uvfits file).
      outname (str): Name of the output file to which to write data.
      snrcut (float): A possible signal-to-noise ratio below which to reject points. Default: 0.

    Returns:
      None.
    """

    if (ehtim_found is False) :
        raise NotImplementedError

    if (astropy_found is False) :
        raise NotImplementedError

    # Check polrep
    if (obs.polrep!='stokes') :
        warnings.warn("Must supply an obs object with polrep='stokes'", Warning)
        raise ValueError
    
    # Get some dataset particulars
    src = obs.source
    mjd = obs.mjd
    t=Time(mjd,format='mjd')
    [year,day]=map(int,t.yday.split(':')[:2])
    
    # Write header
    out=open(outname,'w')
    out.write('#%24s %4s %4s %15s %6s %15s %15s %15s %15s %15s %15s\n'%('source','year',' day','time (hr)','base','u (Ml)','v (Ml)','V.r (Jy)','err.r (Jy)','V.i (Jy)','err.i (Jy)'))

    # Write data file
    for d in obs.data :
        time = d['time']
        bl = d['t1']+d['t2']
        u = d['u']/1e6
        v = d['v']/1e6
        cv = d['vis']
        err = d['sigma']
        if ( np.abs(cv)/err >= snrcut ) :
            out.write('%25s %4i %4i %15.8f %4s %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n'%(src,year,day,time,bl,u,v,cv.real,err,cv.imag,err))
    out.close()
    

def write_amplitudes(obs, outname, debias_amplitudes=True, snrcut=0) :
    """
    Writes visibility amplitudes in Themis format given an :class:`ehtim.obsdata.Obsdata` object.

    Warning: 
      * The :class:`ehtim.obsdata.Obsdata` must be read in with a polrep='stokes'.  If this is not the case, raises a ValueError.  DO NOT SWITCH THE POLREP AS THIS INFLATES THE ERRORS.
      * This makes extensive use of ehtim and astropy and will not be available if either is not installed.  Raises a NotImplementedError if they are is unavailable.

    Args:
      obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object containing the observation data (presumably repackaging a uvfits file).
      outname (str): Name of the output file to which to write data.
      debias_amplitudes (bool): Flag that sets amplitudes should be debiased in the normal way: :math:`V_{db}=\sqrt{V^2-\sigma^2}`. Default: True.
      snrcut (float): A possible signal-to-noise ratio below which to reject points. Default: 0.

    Returns:
      None.
    """

    if (ehtim_found is False) :
        raise NotImplementedError
    
    if (astropy_found is False) :
        raise NotImplementedError

    # Check polrep
    if (obs.polrep!='stokes') :
        warnings.warn("Must supply an obs object with polrep='stokes'", Warning)
        raise ValueError

    # Get some dataset particulars
    src = obs.source
    mjd = obs.mjd
    t=Time(mjd,format='mjd')
    [year,day]=map(int,t.yday.split(':')[:2])

    # Make sure amplitudes are defined
    obs.add_amp(debias=debias_amplitudes, snrcut=snrcut)

    # Write header
    out=open(outname,'w')
    out.write('#%24s %4s %4s %15s %6s %15s %15s %15s %15s\n'%('source','year',' day','time (hr)','base','u (Ml)','v (Ml)','VA (Jy)','err (Jy)'))

    # Write data file
    for d in obs.amp :
        time = d['time']
        bl = d['t1']+d['t2']
        u = d['u']/1e6
        v = d['v']/1e6
        vm = d['amp']
        err = d['sigma']
        out.write('%25s %4i %4i %15.8f %4s %15.8f %15.8f %15.8f %15.8f\n'%(src,year,day,time,bl,u,v,vm,err))
    out.close()


def write_closure_phases(obs, outname, snrcut=0, keep_trivial_triangles=False, choose_optimal_covariance_set=True) :
    """
    Writes closure phases in Themis format given an :class:`ehtim.obsdata.Obsdata` object.

    Warning: 
      * The :class:`ehtim.obsdata.Obsdata` must be read in with a polrep='stokes'.  If this is not the case, raises a ValueError.  DO NOT SWITCH THE POLREP AS THIS INFLATES THE ERRORS.
      * This makes extensive use of ehtim and astropy and will not be available if either is not installed.  Raises a NotImplementedError if they are is unavailable.

    Args:
      obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object containing the observation data (presumably repackaging a uvfits file).
      outname (str): Name of the output file to which to write data.
      snrcut (float): A possible signal-to-noise ratio below which to reject points. Default: 0.
      keep_trivial_triangles (bool): If True, closure phases on trivial triangles will be included. Trivial triangles are defined to be the those with a baselines shorter than 100 Mlambda.  Default: False.  
      choose_optimal_covariance_set (bool): If True, returns the maximally independent set closure phases.  Default: True.

    Returns:
      None.
    """

    if (ehtim_found is False) :
        raise NotImplementedError
    
    if (astropy_found is False) :
        raise NotImplementedError

    # Check polrep
    if (obs.polrep!='stokes') :
        warnings.warn("Must supply an obs object with polrep='stokes'", Warning)
        raise ValueError
    
    # Get some dataset particulars
    src = obs.source
    mjd = obs.mjd
    t=Time(mjd,format='mjd')
    [year,day]=map(int,t.yday.split(':')[:2])


    # Remove trivial triangles if not keeping them.
    if (not keep_trivial_triangles) :
        obs.flag_uvdist(0.1e9) #Kill trivial triangles

    # Get the set of closure phase observations
    if (choose_optimal_covariance_set) :
        obs_cphase_arr = closure_phase_covariance_ordering(obs,snrcut)
    else :
        obs.add_cphase(count='max', snrcut=snrcut)
        obs_cphase_arr = obs.cphase

    # Write header
    out=open(outname,'w')
    out.write('#%24s %4s %4s %15s %6s %15s %15s %15s %15s %15s %15s\n'%('source','year',' day','time (hr)','base','u1 (Ml)','v1 (Ml)','u2 (Ml)','v2 (Ml)','CP (deg)','err (deg)'))
    
    #Now output the data
    for ii in range(len(obs_cphase_arr)):
        time = obs_cphase_arr['time'][ii]
        triangle_list = obs_cphase_arr['t1'][ii]+obs_cphase_arr['t2'][ii]+obs_cphase_arr['t3'][ii]
        u1 = obs_cphase_arr['u1'][ii]/1e6
        v1 = obs_cphase_arr['v1'][ii]/1e6
        u2 = obs_cphase_arr['u2'][ii]/1e6
        v2 = obs_cphase_arr['v2'][ii]/1e6
        u3 = obs_cphase_arr['u3'][ii]/1e6
        v3 = obs_cphase_arr['v3'][ii]/1e6
        CP = obs_cphase_arr['cphase'][ii]
        sCP = obs_cphase_arr['sigmacp'][ii]

        if (np.sqrt((u1+u2+u3)**2+(v1+v2+v3)**2)>1) :
            warnings.warn("Large triangle displacement detected: (du,dv)=(%g,%g).  This may happen for scan averaged data sets."%(u1+u2+u3,v1+v2+v3), Warning)

        out.write('%25s %4i %4i %15.8f %6s %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n'%(src,year,day,time,triangle_list,u1,v1,u2,v2,CP,sCP))
        
    out.close() 


def write_closure_amplitudes(obs,outname) :
    raise NotImplementedError


def write_polarization_fractions(obs,outname) :
    raise NotImplementedError



def read_crosshand_visibilities(filename, verbosity=0) :
    """
    Reads a themis-style amplitude data file and returns a dictionary object.

    Args:
      filename (str) :  Name of the file containing Themis-style data.
      verbosity (int) : Verbosity level. Default: 0.

    Returns:
      (dict) : Dictionary containing data with additional information.
    """
    
    datadict={}
    datadict['source']=[]
    datadict['year']=[]
    datadict['day']=[]
    datadict['time']=[]
    datadict['baseline']=[]
    datadict['u']=[]
    datadict['v']=[]
    datadict['field rotation 1']=[]
    datadict['field rotation 2']=[]
    datadict['RR']=[]
    datadict['RR error']=[]
    datadict['LL']=[]
    datadict['LL error']=[]
    datadict['RL']=[]
    datadict['RL error']=[]
    datadict['LR']=[]
    datadict['LR error']=[]
    for l in open(filename,'r') :
        # Skip headers/comment/flagged lines
        if (l[0]=='#') :
            continue
        toks=l.split()
        datadict['source'].append(toks[0])
        datadict['year'].append(int(toks[1]))
        datadict['day'].append(int(toks[2]))
        datadict['time'].append(float(toks[3]))
        datadict['baseline'].append(toks[4])
        datadict['u'].append(float(toks[5])*1e6)
        datadict['v'].append(float(toks[6])*1e6)
        datadict['field rotation 1'].append(float(toks[7]))
        datadict['field rotation 2'].append(float(toks[8]))
        datadict['RR'].append(float(toks[9])+1.0j*float(toks[11]))
        datadict['RR error'].append(float(toks[10])+1.0j*float(toks[12]))
        datadict['LL'].append(float(toks[13])+1.0j*float(toks[15]))
        datadict['LL error'].append(float(toks[14])+1.0j*float(toks[16]))
        datadict['RL'].append(float(toks[17])+1.0j*float(toks[19]))
        datadict['RL error'].append(float(toks[18])+1.0j*float(toks[20]))
        datadict['LR'].append(float(toks[21])+1.0j*float(toks[23]))
        datadict['LR error'].append(float(toks[22])+1.0j*float(toks[24]))

    for key in datadict.keys() :
        datadict[key] = np.array(datadict[key])

    if (verbosity>0) :
        for key in datadict.keys() :
            print(("%15s : ")%(key),datadict[key][:5])
        
    return datadict
        

def read_visibilities(filename, verbosity=0) :
    """
    Reads a themis-style amplitude data file and returns a dictionary object.

    Args:
      filename (str) :  Name of the file containing Themis-style data.
      verbosity (int) : Verbosity level. Default: 0.

    Returns:
      (dict) : Dictionary containing data with additional information.
    """
    
    datadict={}
    datadict['source']=[]
    datadict['year']=[]
    datadict['day']=[]
    datadict['time']=[]
    datadict['baseline']=[]
    datadict['u']=[]
    datadict['v']=[]
    datadict['visibility']=[]
    datadict['error']=[]
    for l in open(filename,'r') :
        # Skip headers/comment/flagged lines
        if (l[0]=='#') :
            continue
        toks=l.split()
        datadict['source'].append(toks[0])
        datadict['year'].append(int(toks[1]))
        datadict['day'].append(int(toks[2]))
        datadict['time'].append(float(toks[3]))
        datadict['baseline'].append(toks[4])
        datadict['u'].append(float(toks[5])*1e6)
        datadict['v'].append(float(toks[6])*1e6)
        datadict['visibility'].append(float(toks[7])+1.0j*float(toks[9]))
        datadict['error'].append(float(toks[8])+1.0j*float(toks[10]))

    for key in datadict.keys() :
        datadict[key] = np.array(datadict[key])

    if (verbosity>0) :
        for key in datadict.keys() :
            print(("%15s : ")%(key),datadict[key][:5])

    return datadict
        

def read_amplitudes(filename, verbosity=0) :
    """
    Reads a themis-style amplitude data file and returns a dictionary object.

    Args:
      filename (str) :  Name of the file containing Themis-style data.
      verbosity (int) : Verbosity level. Default: 0.

    Returns:
      (dict) : Dictionary containing data with additional information.
    """
    
    datadict={}
    datadict['source']=[]
    datadict['year']=[]
    datadict['day']=[]
    datadict['time']=[]
    datadict['baseline']=[]
    datadict['u']=[]
    datadict['v']=[]
    datadict['amplitude']=[]
    datadict['error']=[]
    for l in open(filename,'r') :
        # Skip headers/comment/flagged lines
        if (l[0]=='#') :
            continue
        toks=l.split()
        datadict['source'].append(toks[0])
        datadict['year'].append(int(toks[1]))
        datadict['day'].append(int(toks[2]))
        datadict['time'].append(float(toks[3]))
        datadict['baseline'].append(toks[4])
        datadict['u'].append(float(toks[5])*1e6)
        datadict['v'].append(float(toks[6])*1e6)
        datadict['amplitude'].append(float(toks[7]))
        datadict['error'].append(float(toks[8]))

    for key in datadict.keys() :
        datadict[key] = np.array(datadict[key])

    if (verbosity>0) :
        for key in datadict.keys() :
            print(("%15s : ")%(key),datadict[key][:5])
            
    return datadict


def read_closure_phases(filename, verbosity=0) :
    """
    Reads a themis-style closure phase data file and returns a dictionary object.

    Args:
      filename (str) :  Name of the file containing Themis-style data.
      verbosity (int) : Verbosity level. Default: 0.

    Returns:
      (dict) : Dictionary containing data with additional information.
    """
    
    datadict={}
    datadict['source']=[]
    datadict['year']=[]
    datadict['day']=[]
    datadict['time']=[]
    datadict['triangle']=[]
    datadict['u1']=[]
    datadict['v1']=[]
    datadict['u2']=[]
    datadict['v2']=[]
    datadict['u3']=[]
    datadict['v3']=[]
    datadict['closure phase']=[]
    datadict['error']=[]
    for l in open(filename,'r') :
        # Skip headers/comment/flagged lines
        if (l[0]=='#') :
            continue
        toks=l.split()
        datadict['source'].append(toks[0])
        datadict['year'].append(int(toks[1]))
        datadict['day'].append(int(toks[2]))
        datadict['time'].append(float(toks[3]))
        datadict['triangle'].append(toks[4])
        datadict['u1'].append(float(toks[5])*1e6)
        datadict['v1'].append(float(toks[6])*1e6)
        datadict['u2'].append(float(toks[7])*1e6)
        datadict['v2'].append(float(toks[8])*1e6)
        datadict['closure phase'].append(float(toks[9]))
        datadict['error'].append(float(toks[10]))

    for key in datadict.keys() :
        datadict[key] = np.array(datadict[key])
    
    datadict['u3'] = -datadict['u1']-datadict['u2']
    datadict['v3'] = -datadict['v1']-datadict['v2']

    if (verbosity>0) :
        for key in datadict.keys() :
            print(("%15s : ")%(key),datadict[key][:5])
    
    return datadict



def read_closure_amplitudes(filename, verbosity=0) :
    """
    Reads a themis-style closure amplitude data file and returns a dictionary object.

    Args:
      filename (str) :  Name of the file containing Themis-style data.
      verbosity (int) : Verbosity level. Default: 0.

    Returns:
      (dict) : Dictionary containing data with additional information.
    """

    raise NotImplementedError


def read_polarization_fractions(filename, verbosity=0) :
    """
    Reads a themis-style closure amplitude data file and returns a dictionary object.

    Args:
      filename (str) :  Name of the file containing Themis-style data.
      verbosity (int) : Verbosity level. Default: 0.

    Returns:
      (dict) : Dictionary containing data with additional information.
    """

    raise NotImplementedError


import matplotlib.pyplot as plt

def write_uvfits(obs, outname, gain_data=None, dterm_data=None, relative_timestamps=False, verbosity=0) :
    """
    Writes uvfits file given an :class:`ehtim.obsdata.Obsdata` object.  Potentially applies gains and/or dterms from a Themis analysis 

    Warning: 
      * This makes extensive use of ehtim and will not be available if ehtim is not installed.  Raises a NotImplementedError if ehtim is unavailable.
      * D term calibration not yet implemented.

    Args:
      obs (ehtim.obsdata.Obsdata): An ehtim Obsdata object containing the observation data (presumably repackaging a uvfits file).
      outname (str): Name of the output file to which to write data.
      gains (dictionary): Station gains organized as a dictionary indexed by the station codes in :class:`ehtim.obsdata.tarr`.
      dterms (dictionary): Station D terms organized as a dictionary indexed by the station codes in :class:`ehtim.obsdata.tarr`.
      relative_timestamps (bool): If True, will assume that the times in the calibration files apply to the given data set and apply in order. Requires that the number of gains must match the number of time slices in the data set. Exists prirmarily to address poor absolute time specification of earlier gain files. In almost all new Themis analyses after Apr 22, 2020, should be False.
      verbosity (int): Degree of verbosity.  0 only prints warnings and errors. 1 provides more granular output. 2 generates calibrated data plots. 3 generates cal table gain plots.  

    Returns:
      None.
    """

    ## TODO: Fix it to figure out the relative start/stop times and apply the gains in the stated time bins.
    ##

    # Flip gains from corrections to the model to adjust data, i.e., G -> 1/G
    gain_station_names = gain_data['stations']
    for sn in gain_station_names :
        gain_data[sn] = 1.0/gain_data[sn]


    if (verbosity>0) :
        print("Gain data:",gain_data)
        
    # Get the unique times
    od_time = np.unique(obs.unpack('time'))
    od_time_list = []
    for tmp in od_time :
        od_time_list.append(tmp[0])

    # Check for consistency, if they are not consistent warn
    if (relative_timestamps) :
        gain_time_list = od_time_list
        if (len(od_time_list)!=len(gain_data['tstart'])) :
            raise RuntimeError("When relative_timestamps=True, number of gain epochs (%i) must match number of observation epochs (%i)."%(len(gain_data['tstart']),len(od_time_list)))
        if ( (od_time_list[-1]-od_time_list[0]) > (gain_data['tend'][-1]-gain_data['tstart'][0]) ) :
            raise RuntimeError("gain_data does not cover the observation times.  Cowardly refusing to continue.")
    else :
        # Get the absolute times
        if (len(od_time_list)!=len(gain_data['tstart'])) :
            warnings.warn("gain_data and observation data are not the same size! Will try to apply gains nonetheless ...",Warning)
        if ( int(gain_data['toffset'].mjd) != obs.mjd ) :
            raise RuntimeError("Observation and gain reconstruction dates differ (mjd %i vs %i). Cowardly refusing to continue."%(obs.mjd,int(gain_data['toffset'].mjd)))
        gain_time_offset_hour = 24.0 * (gain_data['toffset'].mjd%1)
        gain_time_list = gain_data['tstart'] + gain_time_offset_hour
        time_precision_slop = 1e-3/3600.0 # Permit a slop of 1 ms in the time comparisons
        if ( od_time_list[0]<gain_data['tstart'][0]+gain_time_offset_hour-time_precision_slop or od_time_list[-1]>gain_data['tend'][-1]+gain_time_offset_hour+time_precision_slop ) :
            raise RuntimeError("gain_data does not cover the observation times. Cowardly refusing to continue.")

        
    # Generate Caltable object
    if (verbosity>0) :
        print("Sites:",gain_station_names)
        print("Times:",od_time_list)
    datatables = {}
    #for k in range(len(gain_station_names)):
    for station in gain_station_names :
        datatable = []
        for j in range(len(gain_time_list)):
            datatable.append(np.array((gain_time_list[j], gain_data[station][j], gain_data[station][j]), dtype=eh.DTCAL))
        datatables[station] = np.array(datatable)
    cal=eh.caltable.Caltable(obs.ra, obs.dec, obs.rf, obs.bw, datatables, obs.tarr, source=obs.source, mjd=obs.mjd, timetype=obs.timetype)

    # Calibrate the observation data (Yikes!!)
    obs_cal = cal.applycal(obs)

    # Write out new uvfits file
    obs_cal.save_uvfits(outname)

    # Make calibrated data plots if desired
    if (verbosity>2) :
        eh.plotting.comp_plots.plotall_obs_compare([obs,obs_cal],'uvdist','amp')
        
    # Make gain plots if desired
    if (verbosity>3) :
        cal.plot_gains([])
    
    # Show the plots if relevant
    if (verbosity>2) :
        plt.show()

    
