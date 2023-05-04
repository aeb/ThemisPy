###########################
#
# Package:
#   snapshots
#
# Provides:
#   Provides tools for generating and visualizing snapshot posteriors
#

from themispy.vis.typlot import *
from themispy.vis.typlot import _cdf

import numpy as np
import scipy.interpolate as sint
import scipy.stats as sstat
import matplotlib.pyplot as plt
import copy

class SnapshotPosterior :
    """
    A base class that defines generic scoring and posterior generation.

    Args:
    
    Attributes:
    """

    def __init__(self,**kwargs) :
        pass
    
    def generate(self,verbosity=0,**kwargs) :
        """
        A function that generates the necessary internal data from which to obtain
        posteriors and Bayesian evidences.

        Args:
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
        """
        raise NotImplementedError("This must be implemented in child classes.")

    def evidence(self,verbosity=0,**kwargs) :
        """
        Access to the Bayesian evidence or an appropriate proxy.

        Args: 
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float): Bayesian evidence (or proxy).
        """
        raise NotImplementedError("This must be implemented in child classes.")

    def posterior(self,params,p1,p2=None,verbosity=0,**kwargs) :
        """
        Access to the normalized 1D or 2D posterior listed in params.

        Args:
          params (list): Lists of strings of parameter names for which to evaluate the posterior.
          p1 (float,numpy.ndarray): Value of first dimension parameter at which to evaluate the posterior.
          p2 (float,numpy.ndarray): Value of second dimension parameter at which to evaluate the posterior, if 2D posterior.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float,numpy.ndarray): Value of posterior at desired parameter locations.
        """
        raise NotImplementedError("This must be implemented in child classes.")

        

class SingleEpochSnapshotPosterior(SnapshotPosterior) :
    """
    A Python class that encapsulates the post-Themis analysis of the
    Average Image Scoring (AIS) and Ensemble-based Posterior Construction (EBPC)
    methods for the analyses EHT data with a single GRMHD simulation (and
    the associated collection of snapshot images).

    Warning:
      * Formats of AIS and snapshot files must be settled upon.  Currently assumes fit_summary style.

    Args:
      ais_name (str): Name of a file containing the AIS fit information. If None, some functions will not be available. Default: None.
      snapshot_scoring_name (str): Name of a file containing the snapshot scoring fit information. If None, some functions will not be available. Default: None.
      eht_data_type (str): Type of EHT data being fitted. Options include 'V' (complex visibilities), 'VACP' (visibility amplitudes and closure phases), and 'CACP' (closure amplitudes and closure phases, uses Comrade format). Default: 'VACP'.
      spin_dir (str,float): Direction of spin in fitted images..  Options are 'N','S','E','W, or an angle measured east of north. Default: 'N'.
      themis_pa_fix (bool): If True, resets the PA to 180-PA, in accordance with the impact of the definition of PA in Themis. Default: True.
      verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

    Attributes:
      TBD
    """
    def __init__(self,ais_name=None,snapshot_scoring_name=None,eht_data_type='VACP',spin_dir='N',themis_pa_fix=True,verbosity=0,**kwargs) :

        if (not snapshot_scoring_name is None) :
            self.ebpc_summary = self.read_scoring_summary(snapshot_scoring_name,eht_data_type=eht_data_type,spin_dir=spin_dir,themis_pa_fix=themis_pa_fix,verbosity=verbosity,**kwargs)
            self.ebpc_summary_cut = None
        else :
            self._ebpc_defined = False

        if (not ais_name is None) :
            self.ais_summary = self.read_ais_summary(ais_name,eht_data_type=eht_data_type,spin_dir=spin_dir,themis_pa_fix=themis_pa_fix,verbosity=verbosity,**kwargs)
        else :
            self._ais_defined = False

        self.ebpc_posteriors = {}

        self.ais_method = None
        self.ais_quality_statistic = None
        self.ebpc_quality_cut = None
        self.ebpc_quality_statistic = None


        
            
    def read_scoring_summary(self,fsfile,eht_data_type='VACP',spin_dir='N',themis_pa_fix=True,verbosity=0,**kwargs) :
        """
        Reads the output of a scoring run, assuming a fit_summaries file format.
        
        Args:
          fsfile (str): Name of file containing AIS fit information.
          eht_data_type (str): Type of EHT data being fitted. Options include 'V' (complex visibilities), 'VACP' (visibility amplitudes and closure phases), and 'CACP' (closure amplitudes and closure phases, uses Comrade format). Default: 'VACP'.
          themis_pa_fix (bool): If True, resets the PA to 180-PA, in accordance with the impact of the definition of PA in Themis. Default: True.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
        
        Returns:
          (numpy.recarray): Structured numpy array of the values of the flux, mass/distance, position angle, likelihood, and chi-squared of the fits.
          
        """

        # Choose columns based on eht data type
        # Data read in is expected to be flux mass pa chisq likelihood
        if (eht_data_type=='V') :
            cols = [1,2,3,5,6]
            vals = np.loadtxt(fsfile,skiprows=1,usecols=cols)
        elif (eht_data_type=='VACP') :
            cols = [1,2,3,6,7]
            vals = np.loadtxt(fsfile,skiprows=1,usecols=cols)
        elif (eht_data_type=='CACP') :
            cols = [0,1,4,6]
            valssub = np.loadtxt(fsfile,skiprows=1,usecols=cols,delimiter=",")
            vals = np.zeros((valssub.shape[0],5))
            vals[:,1:] = valssub  # flux and likelihood are zeroed out.
            vals[:,1] *= 180.0*3600e6/np.pi # Convert M/D to uas
            vals[:,2] *= 180.0/np.pi # Convert PA to deg
            #vals[:,4] = -0.5*vals[:,3] # L = - 0.5*csq
        else :
            raise ValueError("Unrecognized eht_data_type, %s. Expects either 'V' or 'VACP'."%(eht_data_type))

        # Choose angle offset based on spin direction in snapshot images
        PAoffset = 0.0
        if (isinstance(spin_dir,str)) :
            if (spin_dir=='N') :
                PAoffset = 180.0
            elif (spin_dir=='S') :
                PAoffset = 0.0
            elif (spin_dir=='E') : ## Need to double check
                PAoffset = 270.0
            elif (spin_dir=='W') : ## Need to double check
                PAoffset = 90.0
            else :
                raise ValueError("Unrecognized spin_dir, %s. Expects either 'N','S','E','W' or a float."%(spin_dir))
        elif (isinstance(spin_dir,float)) :
            PAoffset = spin_dir+180.0
        else :
            raise ValueError("Unrecognized spin_dir, %s. Expects either 'N','S','E','W' or a float."%(spin_dir))            


        # Adjust to flux, mass and PA in relevant units
        flux = vals[:,0] # Jy
        mass = vals[:,1] # uas
        PA = vals[:,2] # deg
        if (themis_pa_fix) :
            PA = PAoffset - PA
        csq = vals[:,3]
        L = vals[:,4]

        PA = np.arctan2(np.sin(PA*np.pi/180.),np.cos(PA*np.pi/180.))*180./np.pi
        
        self._ebpc_defined = True
        
        return np.rec.fromarrays([flux,PA,mass,L,csq],names=['F','PA','M','Likelihood','ChiSquared'])
    

    def read_ais_summary(self,fsfile,eht_data_type='VACP',spin_dir='N',themis_pa_fix=True,verbosity=0,**kwargs) :
        """
        Reads the output of an AIS run, assuming a fit_summaries file format with the fit to data last.
        
        Args:
          fsfile (str,list): Name of file, or list of names of files, containing AIS fit information.  If single file, expects data fit to be the last entry.  If a list of file names, expects the first to be the simulation fits and the second to be the data fit.
          eht_data_type (str): Type of EHT data being fitted. Options include 'V' (complex visibilities), 'VACP' (visibility amplitudes and closure phases), and 'CACP' (closure amplitudes and closure phases, uses Comrade format). Default: 'VACP'.
          themis_pa_fix (bool): If True, resets the PA to 180-PA, in accordance with the impact of the definition of PA in Themis. Default: True.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
        
        Returns:
          (dict): A dictionary with two numpy.recarrays containing the values of the flux, mass/distance, position angle, likelihood, and chi-squared of the fits to simulated data and the real data.
        """

        # Choose columns based on eht data type
        if (eht_data_type=='V') :
            cols = [1,2,3,5,6]
            if (isinstance(fsfile,list)):
                vals_sim = np.loadtxt(fsfile[0],skiprows=1,usecols=cols)
                vals_obs = np.loadtxt(fsfile[1],skiprows=1,usecols=cols).reshape([1,len(cols)]) # Expects exactly one record
                vals = np.concatenate((vals_sim,vals_obs),axis=0)
            else :
                vals = np.loadtxt(fsfile,skiprows=1,usecols=cols)
        elif (eht_data_type=='VACP') :
            cols = [1,2,3,6,7]
            if (isinstance(fsfile,list)):
                vals_sim = np.loadtxt(fsfile[0],skiprows=1,usecols=cols)
                vals_obs = np.loadtxt(fsfile[1],skiprows=1,usecols=cols).reshape([1,len(cols)]) # Expects exactly one record
                vals = np.concatenate((vals_sim,vals_obs),axis=0)
            else :
                vals = np.loadtxt(fsfile,skiprows=1,usecols=cols)
        elif (eht_data_type=='CACP') :
            cols = [0,1,4,6]
            if (isinstance(fsfile,list)):
                valssub = np.loadtxt(fsfile[0],skiprows=1,usecols=cols,delimiter=",")
                vals_sim = np.zeros((valssub.shape[0],5))
                vals_sim[:,1:] = valssub  # flux is zeroed out.
                valssub = np.loadtxt(fsfile[1],skiprows=1,usecols=cols,delimiter=",").reshape([1,len(cols)])
                vals_obs = np.zeros((valssub.shape[0],5))
                vals_obs[:,1:] = valssub  # flux is zeroed out.
                vals = np.concatenate((vals_sim,vals_obs),axis=0)
            else :
                valssub = np.loadtxt(fsfile,skiprows=1,usecols=cols,delimiter=",")
                vals = np.zeros((valssub.shape[0],5))
                vals[:,1:] = valssub  # flux is zeroed out.
            vals[:,1] *= 180.0*3600e6/np.pi  # Convert M/D to uas
            vals[:,2] *= 180.0/np.pi # Convert PA to deg
            # vals[:,4] = -0.5*vals[:,3] # L = - 0.5*csq
        else :
            raise ValueError("Unrecognized eht_data_type, %s. Expects either 'V' or 'VACP'."%(eht_data_type))

        # Choose angle offset based on spin direction in snapshot images
        PAoffset = 0.0
        if (isinstance(spin_dir,str)) :
            if (spin_dir=='N') :
                PAoffset = 180.0
            elif (spin_dir=='S') :
                PAoffset = 0.0
            elif (spin_dir=='E') : ## Need to double check
                PAoffset = 270.0
            elif (spin_dir=='W') : ## Need to double check
                PAoffset = 90.0
            else :
                raise ValueError("Unrecognized spin_dir, %s. Expects either 'N','S','E','W' or a float."%(spin_dir))
        elif (isinstance(spin_dir,float)) :
            PAoffset = spin_dir+180.0
        else :
            raise ValueError("Unrecognized spin_dir, %s. Expects either 'N','S','E','W' or a float."%(spin_dir))

        # index flux mass pa chisq likelihood
        # if (isinstance(fsfile,list)):
        #     vals_sim = np.loadtxt(fsfile[0],skiprows=1,usecols=cols)
        #     vals_obs = np.loadtxt(fsfile[1],skiprows=1,usecols=cols).reshape([1,len(cols)]) # Expects exactly one record
        #     vals = np.concatenate((vals_sim,vals_obs),axis=0)
        # else :
        #     vals = np.loadtxt(fsfile,skiprows=1,usecols=cols)
            
        # Adjust to flux, mass and PA in relevant units
        flux = vals[:,0] # Jy
        mass = vals[:,1] # uas
        PA = vals[:,2] # deg
        if (themis_pa_fix) :
            PA = PAoffset - PA
        csq = vals[:,3]
        L = vals[:,4]

        PA = np.arctan2(np.sin(PA*np.pi/180.),np.cos(PA*np.pi/180.))*180./np.pi

        arr = np.rec.fromarrays([flux,PA,mass,L,csq],names=['F','PA','M','Likelihood','ChiSquared'])

        self._ais_defined = True
        
        return {'simulation':arr[:-1],'data':arr[-1]}


    def generate(self,method='double',ais_quality_statistic='ChiSquared',binary_ais_cut=None,ebpc_quality_cut=0.5,ebpc_quality_statistic='Likelihood',norm_size_1d=1024,norm_size_2d=512,verbosity=0,**kwargs) :
        """
        A function that generates the necessary internal data from which to obtain
        posteriors and Bayesian evidences.

        Args:
          method (str): Scoring method. Availiable options are listed in AIS_score.
          ais_quality_statistic (str): Name of the quality statistic on which to cut. Acceptable names are listed in :func:`AIS_score`. Default: 'ChiSquared'.
          binary_ais_cut (float): If not None, the evidence will be a step function in the AIS score, zero below the cut and unity above it.  If None, the evidence will be set to the AIS score. Default: None.
          ebpc_quality_cut (float): A number between (0,1] indicating the fraction of data to keep. Default: 0.5.
          ebpc_quality_statistic (str): Name of the quality statistic on which to cut. Acceptable names are 'Likelihood' and 'ChiSquared'. Default: 'Likelihood'.
          norm_size_1d (int): Number of points at which to evaluate the posterior along each direction to normalize. Default: 1024.
          norm_size_2d (int): Number of points at which to evaluate the posterior along each direction to normalize. Default: 512.        
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
        """
        self.ais_method = method
        self.ais_quality_statistic = ais_quality_statistic
        self.binary_ais_cut = binary_ais_cut
        self.ebpc_quality_cut = ebpc_quality_cut
        self.ebpc_quality_statistic = ebpc_quality_statistic
        self.ebpc_norm_size_1d = norm_size_1d
        self.ebpc_norm_size_2d = norm_size_2d
        self.ebpc_posteriors = {}
        
    def evidence(self,verbosity=0,**kwargs) :
        """
        Access to the AIS as a proxy for the Bayesian evidence. If a binary_ais_cut is set, will return 0 or 1 based on AIS score.
        The generate function must be run before evidence values can be computed.  If no AIS score is available (e.g., no AIS file
        has been read), will return 1.0..

        Args: 
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float): Bayesian evidence (or proxy).
        """

        if (self._ais_defined==False) :
            warnings.warn("No AIS statistic has been provided.  Returning Z=1.0.")
            return 1.0
        
        if (self.ais_method is None) :
            raise RuntimeError("The generate(...) function must be called before evidence can be provided.")

        ais_score = self.AIS_score(method=self.ais_method,ais_quality_statistic=self.ais_quality_statistic,verbosity=verbosity,**kwargs)
        if (self.binary_ais_cut is None) :
            return ais_score
        else :
            return float(ais_score>self.binary_ais_cut)

    def AIS_score(self,method='double',ais_quality_statistic='ChiSquared',verbosity=0,**kwargs) :
        """
        Computes and returns the AIS score using the prescribed method.

        Warning:
          * The two-sample KS test with one sample is poorly defined and may not lead to sensible behavior.  Generally, the KS test appears to be a poor metric for assessing if the recovered quality statistic (e.g., reduced chi-squared) is drawn from the theoretical distribution.  Because the theoretical distribution is (u,v)-coverage-dependent, combining multiple observations is not a viable path to improving the applicability of the KS test.

        Args:
          method (str): Scoring method, availiable types are 'double' (double-sided p-value), 'high' (high-sided p-value), 'median' (fraction of models further from the median), and 'KS' (Kolmogorov-Smirnov test).
          ais_quality_statistic (str): Name of the quality statistic on which to cut. Acceptable names are 'Likelihood' and 'ChiSquared'. Default: 'ChiSquared'.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float): AIS score.
        """
        
        if (not self._ais_defined) :
            raise RuntimeError("Because no AIS file was read, AIS-based functionality is not available.")

        if (ais_quality_statistic=='ChiSquared') :
            qsign = 1
        elif (ais_quality_statistic=='Likelihood') :
            qsign = -1
        else :
            raise ValueError("Unrecognized ais_quality_statistic found, %s.  Allowed options are 'ChiSquared' or 'Likelihood'"%(ais_quality_statistic))
        q = qsign*np.copy(self.ais_summary['simulation'][ais_quality_statistic])
        Q = qsign*self.ais_summary['data'][ais_quality_statistic]
        
        if (method=='high') :
            return max(0.5,np.sum(q>=Q))/q.size
        elif (method=='double') :
            # plo = max(0.5,np.sum(q>=Q))/q.size
            plo = np.sum(q>=Q)/q.size
            return 2*max(min(plo,1-plo),0.5/q.size)
        elif (method=='median') :
            qmed = np.median(q)
            return max(0.5,np.sum(np.abs(q-qmed)>=np.abs(Q-qmed)))/q.size
        elif (method=='KS' or method=='KolmogorovSmirnov') : # Note that for a single sample, there is a minimum p-value that can be excluded!
            #ksobj = sstat.ks_2samp([Q],q)
            #print("KS report: D=%15.8g p=%15.8g"%(ksobj.statistic,ksobj.pvalue))
            return sstat.ks_2samp([Q],q).pvalue
        else :
            raise ValueError("Unrecognized method found, %s.  Allowed options are 'high', 'double', or 'KS'"%(method))
        
    def plot_AIS_cumulative_probability(self,show_data=True,ais_quality_statistic='ChiSquared',verbosity=0,**kwargs) :
        """
        Plots cumulative probability, possibly with the data indicated.

        Args:
          show_data (bool): If True, shows the data value on the plot via a vertical line. Default: True.
          ais_quality_statistic (str): Name of the quality statistic on which to cut. Acceptable names are 'Likelihood' and 'ChiSquared'. Default: 'Likelihood'.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          None
        """
        
        if (not self._ais_defined) :
            raise RuntimeError("Because no AIS file was read, AIS-based functionality is not available.")

        if (ais_quality_statistic=='ChiSquared') :
            xlbl = r'$\chi^2_\nu$'
        elif (ais_quality_statistic=='Likelihood') :
            xlbl = r'$\log L$'
        else :
            raise ValueError("Unrecognized ais_quality_statistic found, %s.  Allowed options are 'ChiSquared' or 'Likelihood'"%(ais_quality_statistic))
        q = np.copy(self.ais_summary['simulation'][ais_quality_statistic])
        Q = self.ais_summary['data'][ais_quality_statistic]

        q = np.sort(q)
        qcum = np.arange(q.size)/q.size
        q = np.append(q,q[-1])
        qcum = np.append(qcum,1.0)
        
        plt.plot(q,qcum,'-k',ds='steps-pre')
        if (show_data) :
            plt.axvline(Q,color='b')

        plt.grid()
        plt.ylim((0,1))
        plt.xlabel(xlbl)
        plt.ylabel(r'Cumulative Prob.')
        
    def _create_unormalized_posterior(self,scoring_summary,quality_cut=0.5,dims=None,scott_factor=1.0,ebpc_quality_statistic='Likelihood',verbosity=0,**kwargs) :
        """
        Constructs a KDE-smoothed posterior function, possibly marginalized over some dimension(s).

        Warning:
          This is an internal utility function and almost certainly not what you want to access posterior data.

        Args:
          scoring_summary (numpy.recarray): The output of read_scoring_summary, containing the recovered fit parameters and quality statistics.
          quality_cut (float): A number between (0,1] indicating the fraction of data to keep. Default: 0.5.
          dims (list): List of names of variables to include in the posterior. Acceptable names include 'F','PA', and 'M'.
          scott_factor (float): A scaling of the KDE bandwidth. The KDE is very sensitive to this number, which should usually be of order unity. Default: 1.
          ebpc_quality_statistic (str): Name of the quality statistic on which to cut. Acceptable names are 'Likelihood' and 'ChiSquared'. Default: 'Likelihood'.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
        
        Returns:
          (scipy.stats.gaussian_kde): A KDE object that may be evaluated at arbitrary positions (with some effort).
        """

        if (not self._ebpc_defined) :
            raise RuntimeError("Because no snapshot scoring file was read, EBPC-based functionality is not available.")

        ss = np.copy(scoring_summary)
        
        if (not quality_cut is None) :
            if (ebpc_quality_statistic=='Likelihood') :
                cut = np.quantile(ss[ebpc_quality_statistic],q=(1-quality_cut))
                ss = ss[(ss[ebpc_quality_statistic]>cut)]
                if (verbosity>0) :
                    print("Applied quantile cut of %g in posterior construction,\n\tcorrepsonding to quality cut of log(L)>%g, leaving %i/%i snapshot fits."%(quality_cut,cut,ss.shape[0],scoring_summary.shape[0]))
            elif (ebpc_quality_statistic=='ChiSquared') :
                cut = np.quantile(ss[ebpc_quality_statistic],q=quality_cut)
                ss = ss[(ss[ebpc_quality_statistic]<cut)]
                if (verbosity>0) :
                    print("Applied quantile cut of %g in posterior construction,\n\tcorrepsonding to quality cut of reduced chi-squared<%g, leaving %i/%i snapshot fits."%(quality_cut,cut,ss.shape[0],scoring_summary.shape[0]))
            else :
                raise ValueError("Unrecognized ais_quality_statistic found, %s.  Allowed options are 'ChiSquared' or 'Likelihood'"%(ais_quality_statistic))

        # Center PAs
        self.PA_offset = np.arctan2(np.mean(np.sin(ss['PA']*np.pi/180.)),np.mean(np.cos(ss['PA']*np.pi/180.)))*180./np.pi
        ss['PA'] = ss['PA']-self.PA_offset
        
        # Make sure the PA is in the range [-180, 180)
        ss['PA'] = np.arctan2(np.sin(ss['PA']*np.pi/180.),np.cos(ss['PA']*np.pi/180.))*180./np.pi

        # Save for future access
        self.ebpc_summary_cut = np.copy(ss)
        self.ebpc_summary_cut['PA'] += self.PA_offset

        # Combine the results into a list of points that include repetitions of the PA
        pts = np.stack([ss['F'],ss['PA'],ss['M']],axis=1)
        # for PA_shift in [-360,360] :
        #     pts = np.concatenate((pts,np.stack([ss['F'],ss['PA']+PA_shift,ss['M']],axis=1)),axis=0)

        # Extract only those parameters we want to consider, marginalizing over the others
        if (not dims is None) :
            dim_map = {'F':0,'PA':1,'M':2}
            keep_dims = []
            for d in dims :
                keep_dims.append(dim_map[d])
            pts = pts[:,keep_dims]

        # Choose a KDE factor (using Scott's rule, but with a prefactor that may be set).
        # bw = scott_factor * (3*len(ss['PA']))**(-1./(pts.shape[1]+4))
        bw = scott_factor * (len(ss['PA']))**(-1./(pts.shape[1]+4))

        # Return a gaussian_kde object
        return sstat.gaussian_kde(pts.T,bw_method=bw)

    def _get_parameter_limits(self,dim,verbosity=0) :
        """
        Returns parameter limits for the specified quantity.

        Args:
          dim (str): A parameter name.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float,float): The min and max that should contain the posterior, clipped to parameter-specific default values.
        """

        if (not self._ebpc_defined) :
            raise RuntimeError("Because no snapshot scoring file was read, EBPC-based functionality is not available.")
        
        # Find min/max from sampling
        pmin = np.min(self.ebpc_summary[dim])
        pmax = np.max(self.ebpc_summary[dim])

        # Expand by 10% in each direction
        dp = pmax-pmin
        pmin -= 0.1*dp
        pmax += 0.1*dp

        # Clip to parameter-specific values
        if (dim=='F') :
            pmin = max(0.0,pmin)
            pmax = max(10.0,pmax)
        elif (dim=='M') :
            pmin = max(0.0,pmin)
            pmax = max(10.0,pmax)
        elif (dim=='PA') :
            pmin = max(-180.0,pmin)
            pmax = min(180.0,pmax)

            pmin = -180.0
            pmax = 180.0
        else :
            raise ValueError("Unrecognized parameter name %s"%(dim))            
        
        return pmin,pmax
            

    def posterior(self,dim,p1,p2=None,p3=None,verbosity=0,**kwargs) :
        """
        Evaluates the posterior after marginalizing over all but one dimension at specified parameter locations.

        Warning:
          * 3D not yet implemented.
          * Should handle a variety of types for p1,p2, but only tested on floats, numpy.arrays, and numpy.ndarrays.

        Args:
          dim (str): Name of variable to include in the posterior. Acceptable names include 'F','PA', and 'M'.
          p1 (float,numpy.array): List of first variable values at which to evaluate the posterior.
          p2 (float,numpy.array): List of second variable values at which to evaluate the posterior, if >1D. Default: None.
          p3 (float,numpy.array): List of third variable values at which to evaluate the posterior, if >2D. Default: None.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (numpy.array): Values of the normalized posterior at the parameter values.
        """

        if (self.ais_method is None) :
            raise RuntimeError("The generate(...) function must be called before posteriors can be provided.")

        if (not self._ebpc_defined) :
            raise RuntimeError("Because no snapshot scoring file was read, EBPC-based functionality is not available.")
        
        if (isinstance(dim,str)) :
            dim = [dim]
        
        if (len(dim)==1) :
            key = dim[0]

            p1val = np.copy(p1)
            
            # Lazy construction ...
            if (not key in self.ebpc_posteriors.keys()) :
                # Create KDE object
                self.ebpc_posteriors[key] = {'kde':self._create_unormalized_posterior(self.ebpc_summary,dims=[key],quality_cut=self.ebpc_quality_cut,ebpc_quality_statistic=self.ebpc_quality_statistic,verbosity=verbosity,**kwargs)}
                # Normalize
                pmin,pmax = self._get_parameter_limits(dim[0])
                ptmp = np.linspace(pmin,pmax,self.ebpc_norm_size_1d)
                self.ebpc_posteriors[key]['norm'] = 1.0/(np.sum(self.ebpc_posteriors[key]['kde'](ptmp))*(ptmp[1]-ptmp[0]))
                if (verbosity>0) :
                    print("Evaluated 1D %s Posterior"%(key))
            # Center PA for KDE
            if (dim[0]=='PA') :
                p1val -= self.PA_offset
                p1val = np.arctan2(np.sin(p1val*np.pi/180.),np.cos(p1val*np.pi/180.))*180./np.pi
            # Return values
            return self.ebpc_posteriors[key]['norm']*self.ebpc_posteriors[key]['kde'](p1val)

        elif (len(dim)==2) :

            if (p2 is None) :
                raise RuntimeError("Two parameters must be given to evaluate a 2D posterior.")

            p1val = copy.copy(p1)
            p2val = copy.copy(p2)
            
            if (not type(p1val) is np.ndarray) :
                p1val = np.array([p1val])
            if (not type(p2val) is np.ndarray) :
                p2val = np.array([p2val])
            if (len(p1val.ravel())!=len(p2val.ravel())) :
                raise ValueError("Shapes and types of the two parameters must be identical.")
            
            key = dim[0]+'-'+dim[1]
            # Lazy construction ...
            if (not key in self.ebpc_posteriors.keys()) :
                # Create KDE object
                self.ebpc_posteriors[key] = {'kde':self._create_unormalized_posterior(self.ebpc_summary,dims=dim,quality_cut=self.ebpc_quality_cut,ebpc_quality_statistic=self.ebpc_quality_statistic,verbosity=verbosity,**kwargs)}
                # Normalize
                p1min,p1max = self._get_parameter_limits(dim[0])
                p1tmp = np.linspace(p1min,p1max,self.ebpc_norm_size_2d)
                p2min,p2max = self._get_parameter_limits(dim[1])
                p2tmp = np.linspace(p2min,p2max,self.ebpc_norm_size_2d)
                X,Y = np.meshgrid(p1tmp,p2tmp)
                ptmp = np.vstack([X.ravel(), Y.ravel()])
                self.ebpc_posteriors[key]['norm'] = 1.0/(np.sum(self.ebpc_posteriors[key]['kde'](ptmp))*(p1tmp[1]-p1tmp[0])*(p2tmp[1]-p2tmp[0]))
                if (verbosity>0) :
                    print("Evaluated 1D %s Posterior"%(key))
            # Center PA for KDE
            if (dim[0]=='PA') :
                p1val -= self.PA_offset
                p1val = np.arctan2(np.sin(p1val*np.pi/180.),np.cos(p1val*np.pi/180.))*180./np.pi                
            if (dim[1]=='PA') :
                p2val -= self.PA_offset
                p2val = np.arctan2(np.sin(p2val*np.pi/180.),np.cos(p2val*np.pi/180.))*180./np.pi                
            # Reformat data into a sequence of points
            ptmp = np.vstack([p1val.ravel(), p2val.ravel()])
            # Return values after reshaping to match input data structures
            if (len(p1val)>1) :
                return self.ebpc_posteriors[key]['norm']*np.reshape(self.ebpc_posteriors[key]['kde'](ptmp).T,p1val.shape)
            else :
                return self.ebpc_posteriors[key]['norm']*np.reshape(self.ebpc_posteriors[key]['kde'](ptmp).T,p1val.shape)[0]                
        else :
            raise RuntimeError("Only 1D and 2D posteriors are defined at this time.")
        


class MultiEpochSnapshotPosterior(SnapshotPosterior) :
    """
    A Python class that encapsulates and accomodates the combination of
    multiple single-epoch snapshot analyses for a single model.

    Assumes that the joint flux prior is uninformative.
    
    Fixes the (spin) PA and M to be constant across all epochs (i.e., no precession).

    Args:
      single_epoch_list (SingleEpochSnapshotPosterior): A list of :class:`SingleEpochSnapshotPosterior` objects containing the comparisons to be combined.
      verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

    """

    def __init__(self,single_epoch_list,verbosity=0,**kwargs) :
        self.single_epoch_list = single_epoch_list
        self.Z = None
        self.posteriors = {}
        
        
    def generate(self,PA_size=360,M_size=512,generate_single_epochs=True,verbosity=0,**kwargs) :
        """
        Utility function to compute the combined AIS score and posterior in
        one step.  These are then returned using access functionis.

        Warning:
          * The two-sample KS test with one sample is poorly defined and may not lead to sensible behavior.  Generally, the KS test appears to be a poor metric for assessing if the recovered quality statistic (e.g., reduced chi-squared) is drawn from the theoretical distribution.  Because the theoretical distribution is (u,v)-coverage-dependent, combining multiple observations is not a viable path to improving the applicability of the KS test.
          * By default, combines AIS scores from single-epoch as if they are proportional to the Bayesian evidence.  If a binary choice is desired, set binary_ais_cut to appropriate value.
          * Treats ensemble-based posterior estimates as true posterior.

        Args:
          PA_size (int): Number of PA values with which to span [-180,180).
          M_size (int): Number of M values with which to span [0,10) uas.
          generate_single_epochs (bool): If True, will generate the single-epoch snapshot posteriors.  This should be the desired behavior.  However, if the single-epoch snapshot posteriors must be separately generated, this argument may be set to False. Default: True.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
          **kwargs (dict): Dictionary of key-word arguments to be passed to the SingleEpochSnapshotPosterior objects generate function.
        
        Returns:
          None
        """
        
        # Make a set of PA's and M's over which to grid the posteriors for
        # integration
        PA, M = np.meshgrid(np.linspace(-180,180,PA_size),np.linspace(0,10,M_size))
        
        # Accumulate the evidence and the combined posteriors.  This is simply a product over the list of single-epoch snapshot posteriors
        self.Z = 1.0
        p = np.ones_like(PA)
        for sesp in self.single_epoch_list :

            # Generate the relevant information in the
            if (generate_single_epochs) :
                sesp.generate(verbosity=verbosity,**kwargs)

            self.Z *= sesp.evidence(verbosity=verbosity,**kwargs)

            ptmp = sesp.posterior(['PA','M'],PA,M,verbosity=verbosity,**kwargs)
            if (verbosity>0) :
                print("sesp norm:",np.sum(ptmp)*(PA[1,1]-PA[0,0])*(M[1,1]-M[0,0]))
            p *= ptmp

        # Integrate over the parameter space and normalize
        pnorm = np.sum(p)*(PA[1,1]-PA[0,0])*(M[1,1]-M[0,0])
        self.Z *= pnorm
        p *= 1.0/pnorm

        # Construct interpolation objects for posterior
        self.posteriors['PA-M'] = sint.RectBivariateSpline(PA[0,:],M[:,0],p.T)
        self.posteriors['M-PA'] = sint.RectBivariateSpline(M[:,0],PA[0,:],p)
        self.posteriors['PA'] = sint.interp1d(PA[0,:],np.sum(p,axis=0)*(M[1,1]-M[0,0]),fill_value=0)
        self.posteriors['M'] = sint.interp1d(M[:,0],np.sum(p,axis=1)*(PA[1,1]-PA[0,0]),fill_value=0)


    def evidence(self,verbosity=0,**kwargs) :
        """
        Access to combined score computed by generate function.

        Args:
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float): AIS score.
        """

        if (self.Z is None) :
            print("Warning: combined AIS score not yet computed.  You should run generate(...) first.")
            
        return self.Z


    def posterior(self,dim,p1,p2=None,verbosity=0,**kwargs) :
        """
        Access to the normalized 1D or 2D posterior listed in params.

        Args:
          dim (list): Lists of strings of parameter names for which to evaluate the posterior.
          p1 (float,numpy.ndarray): Value of first dimension parameter at which to evaluate the posterior.
          p2 (float,numpy.ndarray): Value of second dimension parameter at which to evaluate the posterior, if 2D posterior.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float,numpy.ndarray): Value of posterior at desired parameter locations.
        """

        if (self.Z is None) :
            print("Warning: combined posterior not yet computed.  You should run generate(...) first.")

        if (isinstance(dim,str)) :
            dim = [dim]

        if (len(dim)==1) :
            # Wrap PA into tabulated ranges
            p1val = copy.copy(p1)
            if (not type(p1val) is np.ndarray) :
                p1val = np.array([p1val])
            if (dim[0]=='PA') :
                p1val = np.arctan2(np.sin(p1val*np.pi/180.),np.cos(p1val*np.pi/180.))*180./np.pi

            # Return the posterior evaluated at the proper points
            key = dim[0]
            if (key in self.posteriors.keys()) :
                return self.posteriors[key](p1val)
            else :
                raise ValueError("Unrecognized parameter name %s."%(dim))
        elif (len(dim)==2) :
            # Wrap PA into tabulated ranges
            p1val = copy.copy(p1)
            p2val = copy.copy(p2)
            if (not type(p1val) is np.ndarray) :
                p1val = np.array([p1val])
            if (not type(p2val) is np.ndarray) :
                p2val = np.array([p2val])
            if (len(p1val.ravel())!=len(p2val.ravel())) :
                raise ValueError("Shapes and types of the two parameters must be identical.")
            if (dim[0]=='PA') :
                p1val = np.arctan2(np.sin(p1val*np.pi/180.),np.cos(p1val*np.pi/180.))*180./np.pi
            if (dim[1]=='PA') :
                p2val = np.arctan2(np.sin(p2val*np.pi/180.),np.cos(p2val*np.pi/180.))*180./np.pi

            # Return the posterior evaluated at the proper points
            key = dim[0]+'-'+dim[1]            
            if (key in self.posteriors.keys()) :
                return np.reshape(self.posteriors[key](p1val.ravel(),p2val.ravel(),grid=False),p1val.shape)
            else :
                raise ValueError("Unrecognized parameter name %s %s"%(dim[0],dim[1]))
        else :
            raise RuntimeError("Only 1D and 2D posteriors are defined at this time.")
    

class MultiModelSnapshotPosterior(SnapshotPosterior) :        
    """
    A Python class that encapsulates and accomodates the combination of
    the posteriors from multiple models.  This may be done either for a
    set of single-epoch or multi-epoch models (or both, though this makes
    less sense).

    Assumes that the joint flux prior is uninformative and marginalizes
    over it.
    
    Args:
      snapshot_list (:class:`SnapshotPosterior`): A list of single-epoch or multi-epoch snapshot posterior objects containing the comparisons to be combined.
      verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
    
    """

    def __init__(self,snapshot_posterior_list,verbosity=0,**kwargs) :
        self.snapshot_posterior_list = snapshot_posterior_list
        self.Z = None
        self.Zlist = []
        self.posteriors = {}


    def generate(self,PA_size=360,M_size=512,generate_snapshots=True,verbosity=0,**kwargs) :
        """
        Utility function to compute the combined evidence and posterior in
        one step.  These are then returned using access functionis.

        Warning:
          * Treats evidence and posteriors estimates of snapshot posteriors as if they are formally real.

        Args:
          PA_size (int): Number of PA values with which to span [-180,180).
          M_size (int): Number of M values with which to span [0,10) uas.
          generate_snapshots (bool): If True, will generate the snapshot posteriors.  This should be the desired behavior.  However, if the snapshot posteriors must be separately generated, this argument may be set to False. Default: True.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
          **kwargs (dict): Dictionary of key-word arguments to be passed to the SnapshotPosterior objects generate function.
        
        Returns:
          None
        """

        # Make a set of PA's and M's over which to grid the posteriors for
        # integration
        PA, M = np.meshgrid(np.linspace(-180,180,PA_size),np.linspace(0,10,M_size))

        # Accumulate the evidence and the combined posteriors.  This is simply a product over the list of single-epoch snapshot posteriors
        self.Zlist = np.zeros(len(self.snapshot_posterior_list))

        p = np.zeros_like(PA)
        for j,sp in enumerate(self.snapshot_posterior_list) :

            if (generate_snapshots) :
                kwargs['PA_size'] = PA_size
                kwargs['M_size'] = M_size
                sp.generate(verbosity=verbosity,**kwargs)

            self.Zlist[j] = sp.evidence(verbosity=verbosity,**kwargs)
            p += self.Zlist[j] * sp.posterior(['PA','M'],PA,M,verbosity=verbosity,**kwargs)

        # Integrate over the parameter space and normalize
        pnorm = np.sum(p)*(PA[1,1]-PA[0,0])*(M[1,1]-M[0,0])
        self.Z = pnorm
        p *= 1.0/pnorm


        # Construct interpolation objects for posterior
        self.posteriors['PA-M'] = sint.RectBivariateSpline(PA[0,:],M[:,0],p.T)
        self.posteriors['M-PA'] = sint.RectBivariateSpline(M[:,0],PA[0,:],p)        
        self.posteriors['PA'] = sint.interp1d(PA[0,:],np.sum(p,axis=0)*(M[1,1]-M[0,0]),fill_value=0)
        self.posteriors['M'] = sint.interp1d(M[:,0],np.sum(p,axis=1)*(PA[1,1]-PA[0,0]),fill_value=0)

        
    def evidence(self,verbosity=0,**kwargs) :
        """
        Access to combined score computed by generate function.

        Args:
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float): AIS score.
        """

        if (self.Z is None) :
            print("Warning: combined AIS score not yet computed.  You should run generate(...) first.")
            
        return self.Z

    
    def model_evidences(self,relative_to_max=False,verbosity=0,**kwargs) :
        """
        Access to individual model evidence list.

        Args:
          relative_to_max (bool): If True, normalizes relative to the maximum value. Default: False.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (numpy.array): List of evidence values.
        """

        if (self.Z is None) :
            print("Warning: combined model evidences ot yet computed.  You should run generate(...) first.")

        model_Zs = np.copy(self.Zlist)
        if (relative_to_max) :
            model_Zs = model_Zs/np.max(model_Zs)
        return model_Zs
        

    def posterior(self,dim,p1,p2=None,verbosity=0,**kwargs) :
        """
        Access to the normalized 1D or 2D posterior listed in params.

        Args:
          dim (list): Lists of strings of parameter names for which to evaluate the posterior.
          p1 (float,numpy.ndarray): Value of first dimension parameter at which to evaluate the posterior.
          p2 (float,numpy.ndarray): Value of second dimension parameter at which to evaluate the posterior, if 2D posterior.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float,numpy.ndarray): Value of posterior at desired parameter locations.
        """

        if (self.Z is None) :
            print("Warning: combined posterior not yet computed.  You should run generate(...) first.")

        if (isinstance(dim,str)) :
            dim = [dim]

        if (len(dim)==1) :
            # Wrap PA into tabulated ranges
            p1val = copy.copy(p1)
            if (not type(p1val) is np.ndarray) :
                p1val = np.array([p1val])
            if (dim[0]=='PA') :
                p1val = np.arctan2(np.sin(p1val*np.pi/180.),np.cos(p1val*np.pi/180.))*180./np.pi

            # Return the posterior evaluated at the proper points
            key = dim[0]
            if (key in self.posteriors.keys()) :
                return self.posteriors[key](p1val)
            else :
                raise ValueError("Unrecognized parameter name %s."%(dim))
        elif (len(dim)==2) :
            # Wrap PA into tabulated ranges
            p1val = copy.copy(p1)
            p2val = copy.copy(p2)
            if (not type(p1val) is np.ndarray) :
                p1val = np.array([p1val])
            if (not type(p2val) is np.ndarray) :
                p2val = np.array([p2val])
            if (len(p1val.ravel())!=len(p2val.ravel())) :
                raise ValueError("Shapes and types of the two parameters must be identical.")
            if (dim[0]=='PA') :
                p1val = np.arctan2(np.sin(p1val*np.pi/180.),np.cos(p1val*np.pi/180.))*180./np.pi
            if (dim[1]=='PA') :
                p2val = np.arctan2(np.sin(p2val*np.pi/180.),np.cos(p2val*np.pi/180.))*180./np.pi

            # Return the posterior evaluated at the proper points
            key = dim[0]+'-'+dim[1]            
            if (key in self.posteriors.keys()) :
                return np.reshape(self.posteriors[key](p1val.ravel(),p2val.ravel(),grid=False),p1val.shape)
            else :
                raise ValueError("Unrecognized parameter name %s %s"%(dim[0],dim[1]))
        else :
            raise RuntimeError("Only 1D and 2D posteriors are defined at this time.")
    


## One-dimensional posterior, marginalized over PA
def plot_snapshot_posterior_1D(dim,posterior,pvals=None,colors='k',filled=True,fig=None,axes=None,xlabel=None,ylabel=None,labels=None,orientation='vertical',grid=True,verbosity=0) :
    """
    Plots the posterior associated with a single or multiple snapshot posteriors.

    Args:
      dim (str): The parameter to plot.  Acceptable values are those recognized by the posterior :class:`SnapshotPosterior` object.  For 'F', 'M', and 'PA', automatic labels will be generated.
      posterior (:class:`SnapshotPosterior`,list): A :class:`SnapshotPosterior` object or child class, or list of such objects, for which to plot the desired posterior.
      pvals (np.array): Array of parameter values at which to evaluate the posterior. If None, will attempt to guess based on the parameter to plot (currently defined for 'F', 'M', and 'PA'). Default: None.
      color (str,list): Color, or list of colors. Accepts any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'k'.
      filled (bool): If True, fills the posterior. Default: True.
      fig (matplotlib.figure.Figure): Handle of figure to modify. If None, use the current figure object. Default: None.
      axes (matplotlib.axes.Axes): Handle of axes to modify. If None, use the current axes object. Default: None.
      xlabel (str) : Parameter label (i.e., x-axis label if orientation is 'vertical'). If None, will attempt to guess. Default: None.
      ylabel (str) : Posterior label (i.e., y-axis label if orientation is 'vertical'). If None, will attempt to guess. Default: None.
      labels (str,list) : List of labels to placed in a legend. If None, no legend will be produced. If not None, should have one label for each :class:`SnapshotPosterior` object. Default: None.
      orientation (str) : Orientation of the plot. Options are 'vertical'/'v' and 'horizontal'/'h', corresponding placing the parameter axis on the x-axis and y-axis, respectively.
      grid (bool): If True, plots a grid. Default: True.
      verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
    
    Returns:
      (matplotlib.figure.Figure,matplotlib.axes.Axes,list) : Figure handle, axes handle, and list of handles to individual plot objects.

    """

    if (not fig is None) :
        fig_orig = plt.gcf()
        plt.figure(fig.number)
        
    if (not axes is None) :
        axes_orig = plt.gca()
        plt.sca(axes)

    if (isinstance(posterior,list)==False) :
        posterior = [posterior]

    if (labels is None) :
        labels = ''
        legend = False
    else :
        legend = True
    if (isinstance(labels,str)==True) :
        labels = len(posterior)*[labels]
    if (len(labels)!=len(posterior)) :
        raise ValueError("If not None or a string, one label must be given for each posterior.")

    if (pvals is None) :
        if (dim=='F') :
            pvals = np.linspace(0,10,256)
        elif (dim=='M') :
            pvals = np.linspace(0,10,256)
        elif (dim=='PA') :
            pvals = np.linspace(-180,180,256)
        else :
            raise ValueError("Default parameter value range is not available for dim=%s. Please specify the range of values with the 'pvals' option."%(dim))

    if (isinstance(colors,str)==True) :
        colors = len(posterior)*[colors]
    if (len(colors)!=len(posterior)) :
        raise ValueError("If not a single color name, one color must be given for each posterior.")


    if (xlabel is None) :
        if (dim=='F') :
            xlabel = r'$F~({\rm Jy})$'
        elif (dim=='M') :
            xlabel = r'$M/D~(\mu{\rm as})$'
        elif (dim=='PA') :
            xlabel = r'${\rm PA}~({\rm deg})$'
        else :
            xlabel = ''
            warnings.warn("Default xlabel is not available for dim=%s. You may specify an appropriate label with the 'xlabel' option."%(dim), Warning)

    if (ylabel is None) :
        if (dim=='F') :
            ylabel = r'$\wp(F)$'
        elif (dim=='M') :
            ylabel = r'$\wp(M/D)$'
        elif (dim=='PA') :
            ylabel = r'$\wp({\rm PA})$'
        else :
            ylabel = ''
            warnings.warn("Default ylabel is not available for dim=%s. You may specify an appropriate label with the 'ylabel' option."%(dim), Warning)
            

    plot_objs = []
    for j,sp in enumerate(posterior) :
        p = sp.posterior(dim,pvals)
        if (orientation in ['vertical','v']) :
            plot_objs.append(plt.fill_between(pvals,p,y2=0*pvals,color=colors[j],alpha=0.125))
            plot_objs.append(plt.plot(pvals,p,'-',color=colors[j],label=labels[j]))
        elif (orientation in ['horizontal','h']) :
            plot_objs.append(plt.fill_betweenx(pvals,p,x2=0*pvals,color=colors[j],alpha=0.125))
            plot_objs.append(plt.plot(p,pvals,'-',color=colors[j],label=labels[j]))
        else :
            raise ValueError("Orientation %s not recognized. Available options are 'vertical', 'v', 'horizontal', 'h'."%(orientation))

    if (orientation in ['vertical','v']) :
        plt.xlim(pvals[0],pvals[-1])
        plt.ylim(bottom=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    elif (orientation in ['horizontal','h']) :
        plt.ylim(pvals[0],pvals[-1])
        plt.xlim(left=0)
        plt.ylabel(xlabel)
        plt.xlabel(ylabel)

    if (grid) :
        plt.grid(True,alpha=0.25)
    if (legend) :
        plt.legend()

    figh = plt.gcf()
    axsh = plt.gca()
        
    if (not fig is None) :
        plt.figure(fig_orig.number)
        
    if (not axes is None) :
        plt.sca(axes_orig)


    return figh,axsh,plot_objs


## Combined two-dimensional & one-dimensional plots in triangle
def plot_snapshot_posterior_2D(dims,posterior,p1vals=None,p2vals=None,colors='k',colormaps='Greys',filled=True,fig=None,axes=None,xlabel=None,ylabel=None,labels=None,grid=True,quantiles=None,show_projections=True,verbosity=0) :
    """
    Plots the joint posterior associated with a single or multiple snapshot posteriors, with or without marginalized 1D posterior projections.

    Args:
      dims (list): A liist of two parameters to plot.  Acceptable values are those recognized by the posterior :class:`SnapshotPosterior` object.  For 'F', 'M', and 'PA', automatic labels will be generated.
      posterior (:class:`SnapshotPosterior`,list): A :class:`SnapshotPosterior` object or child class, or list of such objects, for which to plot the desired posterior.
      p1vals (np.array): Array of parameter values associated with dims[0] at which to evaluate the posterior. If None, will attempt to guess based on the parameter to plot (currently defined for 'F', 'M', and 'PA'). Default: None.
      p2vals (np.array): Array of parameter values associated with dims[1] at which to evaluate the posterior. If None, will attempt to guess based on the parameter to plot (currently defined for 'F', 'M', and 'PA'). Default: None.
      color (str,list): Color, or list of colors. Accepts any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'k'.
      colormaps (str,list): Colormap, or list of colormaps. Accepts any acceptable colormap as specified in :mod:`matplotlib.colors`. Default: 'k'.
      filled (bool): If True, fills the posterior. Default: True.
      fig (matplotlib.figure.Figure): Handle of figure to modify. If None, use the current figure object. Default: None.
      axes (matplotlib.axes.Axes,list): Handle of axes to modify (if show_projections is False) or list of handles to axes (if show_projections is True). If None, creates axes. Default: None.
      xlabel (str) : Parameter label (i.e., x-axis label if orientation is 'vertical'). If None, will attempt to guess. Default: None.
      ylabel (str) : Posterior label (i.e., y-axis label if orientation is 'vertical'). If None, will attempt to guess. Default: None.
      labels (str,list) : List of labels to placed in a legend. If None, no legend will be produced. If not None, should have one label for each :class:`SnapshotPosterior` object. Default: None.
      grid (bool): If True, plots a grid. Default: True.
      quantiles (list): List of quantiles at which to draw contours. Default: [0.99,0.90,0.50]
      show_projections (bool): If True, 1D projections will be plotted above and to the right. Default: True.
      verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
    
    Returns:
      (matplotlib.figure.Figure,matplotlib.axes.Axes,list) : Figure handle, axes handle, and list of handles to individual plot objects.

    """

    if (len(dims)!=2) :
        raise ValueError("Exactly two parameter names must be given. Received %g."%(len(dims)))
    
    if (not fig is None) :
        fig_orig = plt.gcf()
        plt.figure(fig.number)
    else :
        plt.figure(figsize=(8,8))
    
    if (not axes is None) :
        axes_orig = plt.gca()
        plt.sca(axes)

    if (isinstance(posterior,list)==False) :
        posterior = [posterior]

    if (labels is None) :
        labels = ''
        legend = False
    else :
        legend = True
    if (isinstance(labels,str)==True) :
        labels = len(posterior)*[labels]
    if (len(labels)!=len(posterior)) :
        raise ValueError("If not None or a string, one label must be given for each posterior.")

    if (p1vals is None) :
        if (dims[0]=='F') :
            p1vals = np.linspace(0,10,256)
        elif (dims[0]=='M') :
            p1vals = np.linspace(0,10,256)
        elif (dims[0]=='PA') :
            p1vals = np.linspace(-180,180,256)
        else :
            raise ValueError("Default parameter value range is not available for dims[0]=%s. Please specify the range of values with the 'p1vals' option."%(dims[0]))

    if (p2vals is None) :
        if (dims[1]=='F') :
            p2vals = np.linspace(0,10,256)
        elif (dims[1]=='M') :
            p2vals = np.linspace(0,10,256)
        elif (dims[1]=='PA') :
            p2vals = np.linspace(-180,180,256)
        else :
            raise ValueError("Default parameter value range is not available for dims[1]=%s. Please specify the range of values with the 'p2vals' option."%(dims[1]))

    if (isinstance(colors,str)==True) :
        colors = len(posterior)*[colors]
    if (len(colors)!=len(posterior)) :
        raise ValueError("If not a single color name, one color must be given for each posterior.")

    if (isinstance(colormaps,str)==True) :
        colormaps = len(posterior)*[colormaps]
    if (len(colormaps)!=len(posterior)) :
        raise ValueError("If not a single colormap, one colormap must be given for each posterior.")
    
    if (xlabel is None) :
        if (dims[0]=='F') :
            xlabel = r'$F~({\rm Jy})$'
        elif (dims[0]=='M') :
            xlabel = r'$M/D~(\mu{\rm as})$'
        elif (dims[0]=='PA') :
            xlabel = r'${\rm PA}~({\rm deg})$'
        else :
            xlabel = ''
            warnings.warn("Default xlabel is not available for dims[0]=%s. You may specify an appropriate label with the 'xlabel' option."%(dims[0]), Warning)
            
    if (ylabel is None) :
        if (dims[1]=='F') :
            ylabel = r'$F~({\rm Jy})$'
        elif (dims[1]=='M') :
            ylabel = r'$M/D~(\mu{\rm as})$'
        elif (dims[1]=='PA') :
            ylabel = r'${\rm PA}~({\rm deg})$'
        else :
            ylabel = ''
            warnings.warn("Default ylabel is not available for dims[1]=%s. You may specify an appropriate label with the 'ylabel' option."%(dims[1]), Warning)
            
    if (quantiles is None) :
        quantiles = [0.99,0.9,0.5,0]
    elif (quantiles[-1]!=0) :
        quantiles.append(0)
            
    # 2D
    if (axes is None) :
        if (show_projections) :
            plt.axes([0.1,0.1,0.6,0.6])
        else :
            plt.axes([0.1,0.1,0.85,0.85])
    else :
        if (isinstance(axes,list)) :
            plt.sca(axes[0])
        else :
            plt.sca(axes)
    axslist = [plt.gca()]
            
            
    X,Y = np.meshgrid(p1vals,p2vals)
    plot_objs = []
    for j,sp in enumerate(posterior) :
        p = sp.posterior(dims,X,Y)
        points = p.reshape(-1)
        dX = X[1,1]-X[0,0]
        dY = Y[1,1]-Y[0,0]
        norm = points.sum()*dX*dY
        levels = []
        for q in quantiles:
            try :
                levels.append(bisect(_cdf,points.min(),points.max(), args=(points,dX,dY,q*norm), xtol=points.max()*1e-10))
            except :
                warnings.warn("Could not find appropriate contour for %g quantile.  Skipping."%(q), Warning)

        if (len(levels)==0) :
            raise RuntimeError("No contour levels could be found for %i-index posterior object."%(j))
        
        lp = np.log(np.maximum(0,p)+1e-10*np.max(p))
        llevels = np.log(levels+1e-10*np.max(p))
        if (filled) :
            plot_objs.append(plt.contourf(X,Y,lp,levels=llevels,cmap=colormaps[j],alpha=0.5))
        plot_objs.append(plt.contour(X,Y,lp,levels=llevels,colors=colors[j],alpha=1,zorder=10))
        plot_objs.append(plt.plot([],[],'-',color=colors[j],label=labels[j]))


    plt.xlim(p1vals[0],p1vals[-1])
    plt.ylim(p2vals[0],p2vals[-1])
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if (grid) :
        plt.grid(True,alpha=0.25)
    if (legend) :
        plt.legend()
        

    if (show_projections) :
        # 1D (p1)
        if (axes is None) :
            plt.axes([0.1,0.725,0.6,0.25])
        else :
            if (isinstance(axes,list)) :
                plt.sca(axes[1])
        _,_,h = plot_snapshot_posterior_1D(dims[0],posterior,pvals=p1vals,colors=colors,filled=filled,fig=plt.gcf(),axes=plt.gca(),xlabel='',labels=None,grid=grid,verbosity=verbosity)
        plt.gca().xaxis.set_ticklabels([])
        axslist.append(plt.gca())
        plot_objs.append(h)

        
        # 1D (p2)
        if (axes is None) :
            plt.axes([0.725,0.1,0.25,0.6])
        else :
            if (isinstance(axes,list)) :
                plt.sca(axes[1])
        _,_,h = plot_snapshot_posterior_1D(dims[1],posterior,pvals=p2vals,colors=colors,filled=filled,fig=plt.gcf(),axes=plt.gca(),xlabel='',labels=None,grid=grid,orientation='h',verbosity=verbosity)
        plt.gca().yaxis.set_ticklabels([])
        axslist.append(plt.gca())
        plot_objs.append(h)

    figh = plt.gcf()
        
    if (not fig is None) :
        plt.figure(fig_orig.number)
        
    if (not axes is None) :
        plt.sca(axes_orig)


    if (show_projections) :
        return figh,axslist,plot_objs
    else :
        return figh,axslist[0],plot_objs


## Plot Odds Ratios
def plot_odds_ratios(posterior,colors='k',labels=None,cutoff=0.05,fig=None,axes=None,norm='max',scale='log',show_all=False,verbosity=0) :
    """
    Plots Bayesian odds ratios associated with :class:`MultiModelSnapshotPosteriors`.

    Args:
      posterior (:class:`MultiModelSnapshotPosterior`,list): A :class:`MultiModelSnapshotPosterior` object or child class, or list of such objects, for which to plot the odds ratios.
      color (str,list): Color, or list of colors. Accepts any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'k'.
      cutoff (float): Odds ratio relative to the maximum in a given :class:`MultiModelSnapshotPosterior` at which to define 'passing'.  Results a horizontal line and a bar indicating passing models across the top.
      labels (str,list) : List of labels to placed in a legend. If None, no legend will be produced. If not None, should have one label for each :class:`SnapshotPosterior` object. Default: None.
      fig (matplotlib.figure.Figure): Handle of figure to modify. If None, use the current figure object. Default: None.
      axes (matplotlib.axes.Axes,list): Handle of axes to modify (if show_projections is False) or list of handles to axes (if show_projections is True). If None, creates axes. Default: None.
      norm (str,float) : Normalization to apply. If 'max' uses the maximum.  If None, no normalization will be performed. Default: 'max'.
      scale (str) : Scale of the y-axis. Accepts any argument to :func:`axes.Axes.set_yscale`. Default: 'log'.

    Returns:
      TBD
    
    """

    if (not fig is None) :
        fig_orig = plt.gcf()
        plt.figure(fig.number)
    else :
        plt.figure(figsize=(5,4))
    
    if (not axes is None) :
        axes_orig = plt.gca()
        plt.sca(axes)
    else :
        plt.axes([0.15,0.15,0.8,0.8])        
        
    if (isinstance(posterior,list)==False) :
        posterior = [posterior]

    if (labels is None) :
        labels = ''
        legend = False
    else :
        legend = True
    if (isinstance(labels,str)==True) :
        labels = len(posterior)*[labels]
    if (len(labels)!=len(posterior)) :
        raise ValueError("If not None or a string, one label must be given for each posterior.")

    plot_objs = []
    for j,mm in enumerate(posterior) :

        if (show_all==False and j>0) :
            continue
        
        be = mm.model_evidences()
        if (norm is None) :
            pass
        elif (norm=='max') :
            be = be/np.max(be)
        elif (isinstance(norm,float)) :
            be = be/norm
        else :
            raise ValueError("Unrecognized normalization %s."%(norm))        
            
        if (scale=='linear') :
            pass
        elif (scale=='log') :
            be = np.log(be)
        else :
            raise ValueError("Unrecognized scale %s."%(scale))
            
        model_index = np.arange(len(be))
        plot_objs.append(plt.plot(model_index,be,'d',color=colors[j]))

    ylims = plt.ylim()
    if (not cutoff is None) :
        if (scale=='linear') :
            plt.axhspan(0,cutoff,color='k',alpha=0.125)
            plt.axhline(cutoff,color='k')
        elif (scale=='log') :
            plt.axhspan(ylims[0],np.log(cutoff),color='k',alpha=0.125)
            plt.axhline(np.log(cutoff),color='k')
        for j,mm in enumerate(posterior) :
            be = mm.model_evidences()
            if (norm is None) :
                pass
            elif (norm=='max') :
                be = be/np.max(be)
            elif (isinstance(norm,float)) :
                be = be/norm
            model_index = np.arange(len(be))
            yval = ylims[1] + (ylims[1]-ylims[0])*0.05*j
            plot_objs.append(plt.plot(model_index,(np.log((be>cutoff))+1)*yval,'.',color=colors[j]))

            if (True) :
                for k in model_index :
                    if (be[k]<cutoff) :
                        plt.axvspan(k-0.5,k+0.5,color=colors[j],alpha=0.125,lw=0)
                    

    if (legend) :
        for j in range(len(posterior)) :
            plot_objs.append(plt.plot([],[],'.',color=colors[j],label=labels[j]))
        plt.legend()
        
    plt.grid(True,alpha=0.25)
    plt.xlabel('Simulation Index')
    if (scale=='linear') :
        plt.ylabel('Odds Ratio')
        plt.ylim(bottom=0)
    elif (scale=='log') :
        plt.ylabel('Log Odds Ratio')
        plt.ylim(bottom=ylims[0])

    figh = plt.gcf()
    axsh = plt.gca()
        
    if (not fig is None) :
        plt.figure(fig_orig.number)
        
    if (not axes is None) :
        plt.sca(axes_orig)


    return figh,axsh,plot_objs

