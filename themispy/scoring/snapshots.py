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
      eht_data_type (str): Type of EHT data being fitted. Options include 'V' (complex visibilities) and 'VACP' (visibility amplitudes and closure phases). Default: 'V'.
      themis_pa_fix (bool): If True, resets the PA to 180-PA, in accordance with the impact of the definition of PA in Themis. Default: True.
      verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

    Attributes:
      TBD
    """
    def __init__(self,ais_name=None,snapshot_scoring_name=None,eht_data_type='V',themis_pa_fix=True,verbosity=0,**kwargs) :

        if (not snapshot_scoring_name is None) :
            self.ebpc_summary = self.read_scoring_summary(snapshot_scoring_name,eht_data_type=eht_data_type,themis_pa_fix=themis_pa_fix,verbosity=verbosity,**kwargs)
            self.ebpc_summary_cut = None
        else :
            self._ebpc_defined = False

        if (not ais_name is None) :
            self.ais_summary = self.read_ais_summary(ais_name,eht_data_type=eht_data_type,themis_pa_fix=themis_pa_fix,verbosity=verbosity,**kwargs)
        else :
            self._ais_defined = False

        self.ebpc_posteriors = {}

        self.ais_method = None
        self.ais_quality_statistic = None
        self.ebpc_quality_cut = None
        self.ebpc_quality_statistic = None


        
            
    def read_scoring_summary(self,fsfile,eht_data_type='V',themis_pa_fix=True,verbosity=0,**kwargs) :
        """
        Reads the output of a scoring run, assuming a fit_summaries file format.
        
        Args:
          fsfile (str): Name of file containing AIS fit information.
          eht_data_type (str): Type of EHT data being fitted. Options include 'V' (complex visibilities) and 'VACP' (visibility amplitudes and closure phases). Default: 'V'.
          themis_pa_fix (bool): If True, resets the PA to 180-PA, in accordance with the impact of the definition of PA in Themis. Default: True.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
        
        Returns:
          (numpy.recarray): Structured numpy array of the values of the flux, mass/distance, position angle, likelihood, and chi-squared of the fits.
          
        """

        # Choose columns based on eht data type
        if (eht_data_type=='V') :
            cols = [1,2,3,5,6]
        elif (eht_data_type=='VACP') :
            cols = [1,2,3,6,7]
        else :
            raise ValueError("Unrecognized eht_data_type, %s. Expects either 'V' or 'VACP'."%(eht_data_type))
        
        # index flux mass pa chisq likelihood
        vals = np.loadtxt(fsfile,skiprows=1,usecols=cols)

        # Adjust to flux, mass and PA in relevant units
        flux = vals[:,0] # Jy
        mass = vals[:,1] # uas
        PA = vals[:,2] # deg
        if (themis_pa_fix) :
            PA = 180.0 - PA
        csq = vals[:,3]
        L = vals[:,4]

        PA = np.arctan2(np.sin(PA*np.pi/180.),np.cos(PA*np.pi/180.))*180./np.pi
        
        self._ebpc_defined = True
        
        return np.rec.fromarrays([flux,PA,mass,L,csq],names=['F','PA','M','Likelihood','ChiSquared'])
    

    def read_ais_summary(self,fsfile,eht_data_type='V',themis_pa_fix=True,verbosity=0,**kwargs) :
        """
        Reads the output of an AIS run, assuming a fit_summaries file format with the fit to data last.
        
        Args:
          fsfile (str): Name of file containing AIS fit information.
          eht_data_type (str): Type of EHT data being fitted. Options include 'V' (complex visibilities) and 'VACP' (visibility amplitudes and closure phases). Default: 'V'.
          themis_pa_fix (bool): If True, resets the PA to 180-PA, in accordance with the impact of the definition of PA in Themis. Default: True.
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.
        
        Returns:
          (dict): A dictionary with two numpy.recarrays containing the values of the flux, mass/distance, position angle, likelihood, and chi-squared of the fits to simulated data and the real data.
        """

        # Choose columns based on eht data type
        if (eht_data_type=='V') :
            cols = [1,2,3,5,6]
        elif (eht_data_type=='VACP') :
            cols = [1,2,3,6,7]
        else :
            raise ValueError("Unrecognized eht_data_type, %s. Expects either 'V' or 'VACP'."%(eht_data_type))

        # index flux mass pa chisq likelihood
        vals = np.loadtxt(fsfile,skiprows=1,usecols=cols)

        # Adjust to flux, mass and PA in relevant units
        flux = vals[:,0] # Jy
        mass = vals[:,1] # uas
        PA = vals[:,2] # deg
        if (themis_pa_fix) :
            PA = 180.0 - PA
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

    def evidence(self,verbosity=0,**kwargs) :
        """
        Access to the AIS as a proxy for the Bayesian evidence. If a binary_ais_cut is set, will return 0 or 1 based on AIS score.
        The generate function must be run before evidence values can be computed.

        Args: 
          verbosity (int): Verbosity level. When greater than 0, various information will be provided. Default: 0.

        Returns:
          (float): Bayesian evidence (or proxy).
        """
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
            plo = max(0.5,np.sum(q>=Q))/q.size
            return 2*min(plo,1-plo)
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
            model_Zs = model_Zs - np.max(model_Zs)
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
    

        
