###########################
#
# Package:
#   image
#
# Provides:
#   Provides model image classes and functions.
#

import themispy
from themispy.utils import *

import numpy as np
import copy
import re
from scipy import interpolate as sint
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib import cm

# Some constants
rad2uas = 180.0/np.pi * 3600 * 1e6
uas2rad = np.pi/(180.0 * 3600 * 1e6)


class model_image :
    """
    Base class for Themis image models.
    This base class is intended to mirror the :cpp:class:`Themis::model_image` base class, providing 
    explicit visualizations of the output of :cpp:class:`Themis::model_image`-derived model analyses.
    
    Args:
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.

    Attributes:
      parameters (list): Parameter list for a given model. Default: None.
      size (int): Number of parameters for a given model. Default: None.
      themis_fft_sign (bool): Themis FFT sign convention flag.
      time (float): Time at which image is to be generated.
    """
    
    def __init__(self, themis_fft_sign=True) :
        self.parameters=None
        self.size=None
        self.themis_fft_sign=themis_fft_sign
        self.time=0
        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_image::generate_model`, 
        a similar function within the :cpp:class:`Themis::model_image` class.  Effectively 
        simply copies the parameters into the local list with some additional run-time checking.

        Args:
          parameters (list): Parameter list.

        Returns:
          None.
        """
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")
    
    def intensity_map(self,x,y,parameters=None,verbosity=0) :
        """
        External access function to the intenesity map.  Child classes should not overwrite 
        this function.  This implements some parameter assessment and checks the FFT sign
        convention.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          parameters (list): Parameter list. If none is provided, uses the current set of parameters. Default: None.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        # 
        if (self.themis_fft_sign==True) :
            x = -x
            y = -y
            if (verbosity>0) :
                print("Reflecting through origin to correct themis FFT sign convention")

        if (not parameters is None) :
            self.generate(parameters)

        I = self.generate_intensity_map(x,y,verbosity=verbosity)

        if (self.themis_fft_sign==True) :
            x = -x
            y = -y
            if (verbosity>0) :
                print("Unreflecting through origin to correct themis FFT sign convention")
        
        return I
    
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Hook for the intensity map function in child classes.  Assumes that the parameters
        list has been set.  This should never be called externally.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """
        
        raise NotImplementedError("intensity_map must be defined in children classes of model_image.")

    def parameter_name_list(self) :
        """
        Hook for returning a list of parameter names.  Currently just returns the list of p0-p(size-1).

        Returns:
          (list) List of strings of variable names.
        """

        return [ 'p%i'%(j) for j in range(self.size) ]
    
    
    def set_time(self,time) :
        """
        Sets time.

        Args:
          time (float): Time in hours.
        """
        
        self.time = time
    
    
class model_image_symmetric_gaussian(model_image) :
    """
    Symmetric gaussian image class that is a mirror of :cpp:class:`Themis::model_image_symmetric_gaussian`.
    Has parameters:

    * parameters[0] ... Total intensity :math:`I_0` (Jy)
    * parameters[1] ... Standard deviation :math:`\\sigma` (rad)

    and size=2.

    Args:
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    """

    def __init__(self, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=2

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """
        
        s = self.parameters[1] * rad2uas
        I0 = self.parameters[0] / (2.*np.pi*s**2)

        I = I0 * np.exp( - 0.5*( x**2 + y**2 )/s**2 )
        
        if (verbosity>0) :
            print("Symmetric Gaussian:",I0,s)
            
        return I

    
    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        return [ r'$I_0$ (Jy)', r'$\sigma$ (rad)']
    
    
class model_image_asymmetric_gaussian(model_image) :
    """
    Asymmetric gaussian image class that is a mirror of :cpp:class:`Themis::model_image_asymmetric_gaussian`.
    Has parameters:

    * parameters[0] ... Total intensity :math:`I_0` (Jy)
    * parameters[1] ... Symmetrized standard deviation :math:`\\sigma` (rad)
    * parameters[2] ... Asymmetry parameter :math:`A` in (0,1)
    * parameters[3] ... Position angle :math:`\\phi` (rad)

    and size=4.

    Args:
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    """

    def __init__(self, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=4

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """
        
        s = abs(self.parameters[1]) * rad2uas
        A = max(min(self.parameters[2],0.99),0.0)
        sm = s/np.sqrt(1.+A)
        sM = s/np.sqrt(1.-A)
        I0 = abs(self.parameters[0]) / (2.*np.pi*sm*sM)
        phi = self.parameters[3]

        c = np.cos(phi)
        s = np.sin(phi)
        dxm = c*x - s*y
        dxM = s*x + c*y

        I = I0 * np.exp( - 0.5*( dxm**2/sm**2 + dxM**2/sM**2 ) )

        if (verbosity>0) :
            print("Asymmetric Gaussian:",I0,sm,sM,phi)

        return I

    
    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        return [ r'$I_0$ (Jy)', r'$\sigma$ (rad)', r'$A$', r'$\phi$ (rad)']
    



class model_image_crescent(model_image) :
    """
    Symmetric gaussian image class that is a mirror of :cpp:class:`Themis::model_image_crescent`.
    Has parameters:

    * parameters[0] ... Total intensity :math:`I_0` (Jy)
    * parameters[1] ... Outer radius :math:`R` (rad)
    * parameters[2] ... Width paramter :math:`\\psi` in (0,1)
    * parameters[3] ... Asymmetry parmaeter :math:`\\tau` in (0,1)
    * parameters[4] ... Position angle :math:`\\phi` (rad)

    and size=5.

    Args:
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    """

    def __init__(self, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=5

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        # Make sure that delta ring is resolvable
        dx = 1.5*max(abs(x[1,1]-x[0,0]),abs(y[1,1]-y[0,0]))
        self.parameters[2] = max(self.parameters[2],dx/(self.parameters[1]*rad2uas))

        I0 = max(1e-8,self.paramters[0])
        Rp = max(1e-20,self.parameters[1]) * rad2uas
        Rn = min( max(1e-4,1-self.parameters[2]), 0.9999 ) * Rp
        d = min(max(self.parameters[3],1e-4),0.9999) * (Rp - Rn)
        a = d * np.cos(self.parameters[4])
        b = d * np.sin(self.parameters[4])
        I0 = I0/(np.pi*(Rp**2-Rn**2))
        I = I0+0*x
        I *= ((x**2+y**2)<=Rp**2)
        I *= (((x-a)**2+(y-b)**2)>=Rn**2)
        
        if (verbosity>0) :
            print("KD Crescent:",I0,Rp,Rn,a,b)

        return I    

    
    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        return [ r'$I_0$ (Jy)', r'$R_{\rm out}$ (rad)', r'$\psi$', r'$\epsilon$', r'$\tau$', r'$\phi$ (rad)']

    
class model_image_xsring(model_image) :
    """
    Symmetric gaussian image class that is a mirror of :cpp:class:`Themis::model_image_xsring`.
    Has parameters:

    * parameters[0] ... Total intensity :math:`I_0` (Jy)
    * parameters[1] ... Outer radius :math:`R` (rad)
    * parameters[2] ... Width paramter :math:`\\psi` in (0,1)
    * parameters[3] ... Eccentricity parameter :math:`\\epsilon` in (0,1)
    * parameters[4] ... Linear fade parameter math:`f` in (0,1)
    * parameters[5] ... Position angle :math:`\\phi` (rad)

    and size=6.

    Args:
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    """

    def __init__(self, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=6

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        I0 = max(1e-8,self.parameters[0])
        Rp = max(1e-20,self.parameters[1]) * rad2uas
        Rin = min( max(1e-4,1-self.parameters[2]), 0.9999 ) * Rp
        d = min(max(self.parameters[3],1e-4),0.9999) * (Rp - Rin)
        f = min(max(self.parameters[4],1e-4),0.9999);
        phi = self.parameters[5]

        Iring0 = (1-gq)*(2.*I0/np.pi) *1.0/((1.0+f)*(Rp**2 - Rin**2) - (1.0-f)*d*Rin**2/Rp);

        c = np.cos(phi)
        s = np.sin(phi)
        dx = x*c - y*s
        dy = x*s + y*c

        Iring = Iring0*(0.5*(1.0-dx/Rp) + 0.5*f*(1.0+dx/Rp)) 
        Iring *= ((dx**2+dy**2)<=Rp**2)
        Iring *= (((dx-d)**2+(dy)**2)>Rin**2)
        
        if (verbosity>0) :
            print("xsring:",I0,Rp,Rn,a,b)

        return Iring

    
    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        return [ r'$I_0$ (Jy)', r'$R_{\rm out}$ (rad)', r'$\psi$', r'$\epsilon$', r'$f$', r'$\phi$ (rad)']


class model_image_xsringauss(model_image) :
    """
    Symmetric gaussian image class that is a mirror of :cpp:class:`Themis::model_image_xsringauss`.
    Has parameters:

    * parameters[0] ... Total intensity :math:`I_0` (Jy)
    * parameters[1] ... Outer radius :math:`R` (rad)
    * parameters[2] ... Width parameter :math:`\\psi` in (0,1)
    * parameters[3] ... Eccentricity parameter :math:`\\epsilon` in (0,1)
    * parameters[4] ... Linear fade parameter math:`f` in (0,1)
    * parameters[5] ... FWHM of the main axis of the Gaussian in units of the ring outer radius math:`g_{ax}`
    * parameters[6] ... Gaussian axial ratio math:`a_q`
    * parameters[7] ... Ratio of Gaussian flux to ring flux math:`g_q`
    * parameters[8] ... Position angle :math:`\\phi` (rad)

    and size=9.

    Args:
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    """

    def __init__(self, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=9

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        # Make sure that delta ring is resolvable
        dx = 1.5*max(abs(x[1,1]-x[0,0]),abs(y[1,1]-y[0,0]))
        self.parameters[2] = max(self.parameters[2],dx/(self.parameters[1]*rad2uas))
        I0 = max(1e-8,self.parameters[0])
        Rp = max(1e-20,self.parameters[1]) * rad2uas
        Rin = min( max(1e-4,1-self.parameters[2]), 0.9999 ) * Rp
        d = min(max(self.parameters[3],1e-4),0.9999) * (Rp - Rin)
        f = min(max(self.parameters[4],1e-4),0.9999);
        sig1 = max(self.parameters[5],1e-4) * Rp;
        sig2 = max(self.parameters[6],1e-4) * sig1;
        gq = min(max(self.parameters[7],1e-4),0.9999);
        phi = self.parameters[8]


        Iring0 = (1-gq)*(2.*I0/np.pi) *1.0/((1.0+f)*(Rp**2 - Rin**2) - (1.0-f)*d*Rin**2/Rp);
        Igauss0 = gq*I0/(2.*np.pi*sig1*sig2);

        c = np.cos(phi)
        s = np.sin(phi)
        dx = x*c - y*s
        dy = x*s + y*c

        Iring = Iring0*(0.5*(1.0-dx/Rp) + 0.5*f*(1.0+dx/Rp)) 
        Igauss = Igauss0*np.exp( -0.5*((dx-(d-Rin))/sig1)**2 -0.5*((dy/sig2)**2)) 

        Iring *= ((dx**2+dy**2)<=Rp**2)
        Iring *= (((dx-d)**2+(dy)**2)>Rin**2)

        if (verbosity>0) :
            print('xsringauss:',I0, Rp, Rin, d, f, sig1, sig2, gq, phi)

        return (Iring+Igauss)


    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        return [ r'$I_0$ (Jy)', r'$R_{\rm out}$ (rad)', r'$\psi$', r'$\epsilon$', r'$f$', r'$g_{ax}$', r'$a_q$', r'$g_q$', r'$\phi$ (rad)']


class model_image_mring(model_image) :
    """
    fourier mode delta mring model that is a mirror of :cpp:class:`themis::model_image_mring`.
    has parameters:

    * parameters[0] ... total intensity :math:`i_0` (jy)
    * parameters[1] ... radius :math:`r` (rad)
    * parameters[2,3] ... coefficients (re,im) for first mring mode 
    * parameters[4,5] ... coefficients (re,im) for second mring mode
    * parameters[2n,2n+1] ... coefficients for (re,im) for n mring mode 

    and size=2+2*n.

    args:
      themis_fft_sign (bool): if true will assume the themis-default fft sign convention, which reflects the reconstructed image through the origin. default: true.
    """

    def __init__(self, nmodes, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=2+nmodes*2
        self.nmodes = nmodes

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        internal generation of the intensity map. in practice you almost certainly want to call :func:`model_image.intensity_map`.

        args:
          x (numpy.ndarray): array of -ra offsets in microarcseconds (usually plaid 2d).
          y (numpy.ndarray): array of dec offsets in microarcseconds (usually plaid 2d).
          verbosity (int): verbosity parameter. if nonzero, prints information about model properties. default: 0.

        returns:
          (numpy.ndarray) array of intensity values at positions (x,y) in :math:`jy/\\mu as^2`.
        """

        # make sure that delta ring is resolvable
        dx = 1.5*max(abs(x[1,1]-x[0,0]),abs(y[1,1]-y[0,0]))
        line_thickness=2

        i0 = max(1e-8,self.parameters[0])
        r = max(1e-20,self.parameters[1]) * rad2uas

        betalist = [self.parameters[i]+1j*self.parameters[i+1] for i in range(2,2*self.nmodes+2,2)]
        phi = np.angle(y + 1j*x)
        beta_factor = (1.0 + np.sum([2.*np.real(betalist[m-1] * np.exp(1j * m * phi)) for m in range(1,len(betalist)+1)],axis=0))

        val = (i0*dx**2/(2*np.pi*r*dx*line_thickness)
                * beta_factor
                * (r - dx*line_thickness/2 < np.sqrt((x)**2 + (y)**2))
                * (r + dx*line_thickness/2 > np.sqrt((x)**2 + (y)**2)))



        if (verbosity>0) :
            print('mring:',i0, r, betalist)

        return (val)


    def parameter_name_list(self) :
        """
        producess a lists parameter names.

        returns:
          (list) list of strings of variable names.
        """
        nr = [r"$\rm{re}(c_%d)$"%i for i in range(1,self.nmodes+1)]
        ni = [r"$\rm{im}(c_%d)$"%i for i in range(1,self.nmodes+1)]

        names =[r'$i_0$ (jy)', r'$r_{\rm delta}$ (rad)']
        for i in range(self.nmodes):
            names.append(nr[i])
            names.append(ni[i])

        return names

class model_image_mring_floor(model_image) :
    """
    fourier mode delta mring model with floor that is a mirror of :cpp:class:`themis::model_image_mring_floor`.
    has parameters:

    * parameters[0] ... total intensity :math:`i_0` (jy)
    * parameters[1] ... radius :math:`r` (rad)
    * parameters[2] ... fraction of disk flux floor
    * parameters[3,4] ... coefficients (re,im) for first mring mode 
    * parameters[5,6] ... coefficients (re,im) for second mring mode
    * parameters[3+2n,3_2n+1] ... coefficients for (re,im) for n mring mode 

    and size=3+2*n.

    args:
      themis_fft_sign (bool): if true will assume the themis-default fft sign convention, which reflects the reconstructed image through the origin. default: true.
    """

    def __init__(self, nmodes, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=3+nmodes*2
        self.nmodes = nmodes

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        internal generation of the intensity map. in practice you almost certainly want to call :func:`model_image.intensity_map`.

        args:
          x (numpy.ndarray): array of -ra offsets in microarcseconds (usually plaid 2d).
          y (numpy.ndarray): array of dec offsets in microarcseconds (usually plaid 2d).
          verbosity (int): verbosity parameter. if nonzero, prints information about model properties. default: 0.

        returns:
          (numpy.ndarray) array of intensity values at positions (x,y) in :math:`jy/\\mu as^2`.
        """

        # make sure that delta ring is resolvable
        dx = 1.5*max(abs(x[1,1]-x[0,0]),abs(y[1,1]-y[0,0]))
        line_thickness=2

        floor = self.parameters[2]
        ir = max(1e-8,self.parameters[0])*(1-floor)
        id =max(1e-8,self.parameters[0])*floor 
        r = max(1e-20,self.parameters[1]) * rad2uas

        phases = self.parameters[4:2*self.nmodes+5:2]
        amps = self.parameters[3:2*self.nmodes+4:2]
        alphas = amps*np.cos(phases)
        betas = amps*np.sin(phases)
        beta_factor = 1
        #This is because I don't include the -x in Themis and I used to opposite phase like
        #a dummy PT
        phi = np.arctan2(x,-y)
        for i in range(self.nmodes):
            c = np.cos((i+1)*phi)
            s = np.sin((i+1)*phi)
            beta_factor += alphas[i]*c - betas[i]*s
        

        val = (ir*dx**2/(2*np.pi*r*dx*line_thickness)
                * beta_factor
                * (r - dx*line_thickness/2 < np.sqrt((x)**2 + (y)**2))
                * (r + dx*line_thickness/2 > np.sqrt((x)**2 + (y)**2)))

        val += id*dx**2/(np.pi*r**2)*(x**2+y**2 < r**2)




        if (verbosity>0) :
            print('mring:',ir+id, r, floor, betalist)

        return (val)


    def parameter_name_list(self) :
        """
        producess a lists parameter names.

        returns:
          (list) list of strings of variable names.
        """
        nr = [r"$|c_%d|$"%i for i in range(1,self.nmodes+1)]
        ni = [r"$\rm{arg}(c_%d)$"%i for i in range(1,self.nmodes+1)]

        names =[r'$i_0$ (jy)', r'$r_{\rm delta}$ (rad)', "floor"]
        for i in range(self.nmodes):
            names.append(nr[i])
            names.append(ni[i])
        return names



class model_image_xsring_ellip(model_image) :
    """
    Symmetric gaussian image class that is a mirror of :cpp:class:`Themis::model_image_xsring_ellip`.
    Has parameters:

    * parameters[0] ... Total intensity :math:`I_0` (Jy)
    * parameters[1] ... Outer radius :math:`R` (rad)
    * parameters[2] ... Width parameter :math:`\\psi` in (0,1)
    * parameters[3] ... Eccentricity parameter :math:`\\epsilon` in (0,1)
    * parameters[4] ... Ellipticity parameter, ration of semi-minor to semi-major in (0,1)
    * parameters[5] ... Ellipticity orientation, (rad)
    * parameters[6] ... Linear fade parameter math:`f` in (0,1)
    * parameters[7] ... Position angle :math:`\\phi` (rad)

    and size=8.

    Args:
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    """

    def __init__(self, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=8

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        # Make sure that delta ring is resolvable
        dx = 1.5*max(abs(x[1,1]-x[0,0]),abs(y[1,1]-y[0,0]))
        self.parameters[2] = max(self.parameters[2],dx/(self.parameters[1]*rad2uas))

        I0 = max(1e-8,self.parameters[0])
        Rp = max(1e-20,self.parameters[1]) * rad2uas
        Rin = min( max(1e-4,1-self.parameters[2]), 0.9999 ) * Rp
        d = min(max(self.parameters[3],1e-4),0.9999) * (Rp - Rin)
        tau = max(self.parameters[4], 1e-4)
        xit = self.parameters[5]
        f = min(max(self.parameters[6],1e-4),0.9999);
        phi = self.parameters[7]
        scale = 1.0/(1.0-tau)
        c = np.cos(xit)
        s = np.sin(xit)
        ddx = (x*c + y*s)*scale
        ddy = (-x*s + y*c)/scale

        Iring0 = (2.*I0/np.pi) *1.0/((1.0+f)*(Rp**2 - Rin**2) - (1.0-f)*d*Rin**2/Rp);

        c = np.cos(phi)
        s = np.sin(phi)
        dx = ddx*c - ddy*s
        dy = ddx*s + ddy*c

        Iring = Iring0*(0.5*(1.0-dx/Rp) + 0.5*f*(1.0+dx/Rp)) 

        Iring *= ((dx**2+dy**2)<=Rp**2)
        Iring *= (((dx-d)**2+(dy)**2)>Rin**2)

        if (verbosity>0) :
            print('xsring_ellip:',I0, Rp, Rin, d, tau, xit, f, phi)

        return (Iring)


    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        return [ r'$I_0$ (Jy)', r'$R_{\rm out}$ (rad)', r'$\psi$', r'$\epsilon$', r'$\tau$', r'$\xi$ (rad)', r'$f$', r'$g_{ax}$', r'$a_q$', r'$g_q$', r'$\phi$ (rad)']




def direct_cubic_spline_1d(x,f,xx,a=-0.5) :
    """
    Image-space approximate cubic spline convolution along 1D.  Useful for the splined raster image classes.  
    This is VERY slow and should be used for debugging only.

    Args: 
      x (numpy.ndarray): 1D array of positions of the control points. Must have the same length as f.
      f (numpy.ndarray): 1D array of heights of the control points.  Must have the same length as x.
      xx (numpy.ndarray): 1D array of positions at which to compute inteperolated values of f.
      a (float): Cubic spline control parameter. Default: -0.5.

    Returns:
      (numpy.ndarray): 1D array with the values of f interpolated to positions xx.
    """

    Dx = x[1]-x[0]
    ff = 0*xx
    for j in range(len(xx)) :
        dx = (x-xx[j])/Dx
        w = (-a)*(dx**3+5*dx**2+8*dx+4)*(dx>=-2)*(dx<-1) + (-(a+2)*dx**3-(a+3)*dx**2+1)*(dx>=-1)*(dx<=0) + ((a+2)*dx**3-(a+3)*dx**2+1)*(dx>0)*(dx<=1) + (-a)*(-dx**3+5*dx**2-8*dx+4)*(dx>1)*(dx<=2)
        ff[j] = np.sum(w*f)

    return ff


def direct_cubic_spline_2d(x,y,f,xx,yy,rotation_angle=None,a=-0.5) :
    """
    Image-space approximate cubic spline convolution in 2D.  This is accomplished by 
    convolving along x and they y.  Expects the data to be stored in a rectalinear grid.  
    Useful for the splined raster image classes.  This is VERY slow and should be used 
    for debugging only.

    Args:
      x (numpy.ndarray): 1D array of positions of the control points in the x direction. Must have the same length as f.shape[0].
      y (numpy.ndarray): 1D array of positions of the control points in the y direction. Must have the same length as f.shape[1].
      f (numpy.ndarray): 2D array of heights of the control points.  Must have f.shape=[x.size,y.size].
      xx (numpy.ndarray): 1D array of positions at which to compute inteperolated values of f.
      ff (numpy.ndarray): 1D array of positions at which to compute inteperolated values of f.
      rotation_angle (float): Rotation angle of the field counter-clockwise in radians. If None, no rotation is performed. Default: None.
      a (float): Cubic spline control parameter. Default: -0.5.

    Returns:
      (numpy.ndarray): 2D array with the values of ff interpolated to positions [xx,yy].
    """

    # 1st pass in x direction to get up-sampled control points
    ff1=np.zeros((xx.size,f.shape[1]))
    for j in range(f.shape[1]) :
        ff1[:,j] = direct_cubic_spline_1d(x,f[:,j],xx,a=a)

    # 2nd pass in y direction to get full up-sampled data
    ff = np.zeros((xx.size,yy.size))
    for j in range(ff.shape[0]) :
        ff[j,:] = direct_cubic_spline_1d(y,ff1[j,:],yy,a=a)

    # Rotate
    if (not rotation_angle is None) :
        interp_obj = sint.RectBivariateSpline(-xx,-yy,ff)
        cpa = np.cos(rotation_angle)
        spa = np.sin(rotation_angle)
        for i in range(xx.shape[0]) :
            for j in range(yy.shape[0]) :
                xxr = cpa*xx[i] + spa*yy[j]
                yyr = -spa*xx[i] + cpa*yy[j]
                ff[i,j] = interp_obj(-xxr,-yyr)

    return ff


def W_cubic_spline_1d(k,a=-0.5) :
    """
    Fourier domain convolution function for the approximate 1D cubic spline.
    Useful for the splined raster image classes.

    Args:
      k (numpy.ndarray): Wavenumber.
      a (float): Cubic spline control parameter. Default: -0.5.

    Returns:
      (numpy.ndarray): 1D array of values of the cubic spline weight function evaluated at each value of k.
    """

    abk = np.abs(k)
    ok4 = 1.0/(abk**4+1e-10)
    ok3 = np.sign(k)/(abk**3+1e-10)

    return np.where( np.abs(k)<1e-2,
                     1 - ((2*a+1)/15.0)*k**2 + ((16*a+1)/560.0)*k**4,
                     ( 12.0*ok4*( a*(1.0-np.cos(2*k)) + 2.0*(1.0-np.cos(k)) )
                       - 4.0*ok3*np.sin(k)*( 2.0*a*np.cos(k) + (4*a+3) ) ) )


def fft_cubic_spline_2d(x,y,f,xx,yy,rotation_angle=None,a=-0.5) :
    """
    FFT cubic spline convolution in 2D. This is accomplished by directly evaluating the
    Fourier transform of f (as a sum of Fourier components), applying the cubic spline as
    a filter in the Fourier space, and then inverting the Fourier transfor with FFT. 
    Expects the data to be stored in a rectalinear grid. Useful for the splined raster 
    image classes.

    Args:
      x (numpy.ndarray): 1D array of positions of the control points in the x direction. Must have the same length as f.shape[0].
      y (numpy.ndarray): 1D array of positions of the control points in the y direction. Must have the same length as f.shape[1].
      f (numpy.ndarray): 2D array of heights of the control points.  Must have f.shape=[x.size,y.size].
      xx (numpy.ndarray): 1D array of positions at which to compute inteperolated values of f.
      yy (numpy.ndarray): 1D array of positions at which to compute inteperolated values of f.
      rotation_angle (float): Rotation angle of the field counter-clockwise in radians. If None, no rotation is performed. Default: None.
      a (float): Cubic spline control parameter. Default: -0.5.

    Returns:
      (numpy.ndarray): 2D array with the values of ff interpolated to positions [xx,yy].
    """

    # Generate Fourier grid of the rFFT of the desired image
    uu1d = np.fft.fftfreq(xx.size,xx[1]-xx[0])
    vv1d = np.fft.fftfreq(yy.size,yy[1]-yy[0])

    uu1d = np.fft.fftshift(uu1d)
    vv1d = np.fft.fftshift(vv1d)

    uu,vv = np.meshgrid(uu1d,vv1d)
    uu = np.transpose(uu)
    vv = np.transpose(vv)

    dx = x-x[0]
    dy = y-y[0]

    if (not rotation_angle is None) :
        uur = np.cos(rotation_angle)*uu + np.sin(rotation_angle)*vv
        vvr = -np.sin(rotation_angle)*uu + np.cos(rotation_angle)*vv
    else :
        uur = uu
        vvr = vv

    tpi = 2.0j*np.pi
    FF = 0.0*uu
    for ix in range(f.shape[0]) :
        for iy in range(f.shape[1]) :
            FF = FF + np.exp( -tpi*(uur*x[ix] + vvr*y[iy]) ) * f[ix,iy]
    FF = FF * np.exp(tpi*(uu*xx[0] + vv*yy[0]))
    FF = FF * W_cubic_spline_1d(2.0*np.pi*dx[1]*uur)*W_cubic_spline_1d(2.0*np.pi*dy[1]*vvr)
    FF = np.fft.ifftshift(FF)

    ff = np.fft.ifft2(FF)

    norm_factor = np.abs(((x[1]-x[0])*(y[1]-y[0]))/((xx[1]-xx[0])*(yy[1]-yy[0])))

    return np.real(ff)*norm_factor





class model_image_splined_raster(model_image) :
    """
    Splined raster image class that is a mirror of :cpp:class:`Themis::model_image_splined_raster`.
    Has parameters:

    * parameters[0] ........ Logarithm of the brightness at control point 0,0 (Jy/sr)
    * parameters[1] ........ Logarithm of the brightness at control point 1,0 (Jy/sr)

    ...

    * parameters[Nx-1] ..... Logarithm of the brightness at control point Nx-1,0 (Jy/sr)
    * parameters[Nx] ....... Logarithm of the brightness at control point 0,1 (Jy/sr)
    * parameters[Nx+1] ..... Logarithm of the brightness at control point 1,1 (Jy/sr)

    ...

    * parameters[Nx*Ny-1] .. Logarithm of the brightness at control point Nx-1,Ny-1 (Jy/sr)

    and size=Nx*Ny.

    Args:
      Nx (int): Number of control points in the -RA direction.
      Ny (int): Number of control points in the Dec direction.
      fovx (float): Field of view in the -RA direction (rad).
      fovy (float): Field of view in the Dec direction (rad).
      a (float): Spline control factor. Default: -0.5.
      spline_method (str): Spline method. Accepted values are 'fft' and 'direct'.  Default: 'fft'.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.

    Attributes:
      Nx (int): Number of control points in the -RA direction.
      Ny (int): Number of control points in the Dec direction.
      fovx (float): Field of view in the -RA direction (rad).
      fovy (float): Field of view in the Dec direction (rad).
      a (float): Spline control factor. Default: -0.5.
      spline_method (str): Spline method. Accepted values are 'fft' and 'direct'.  Default: 'fft'.
    """

    def __init__(self, Nx, Ny, fovx, fovy, a=-0.5, spline_method='fft', themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=Nx*Ny

        self.Nx = Nx
        self.Ny = Ny
        self.fovx = fovx * rad2uas
        self.fovy = fovy * rad2uas
        self.a = a
        self.spline_method = spline_method

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        #f = np.transpose(np.exp(np.array(self.parameters).reshape([self.Nx,self.Ny]))) * uas2rad**2
        #xtmp = np.linspace(-0.5*self.fovx,0.5*self.fovx,self.Nx)
        #ytmp = np.linspace(-0.5*self.fovy,0.5*self.fovy,self.Ny)
        
        f = np.transpose(np.exp(np.array(self.parameters).reshape([self.Ny,self.Nx]))) * uas2rad**2
        xtmp = np.linspace(-0.5*self.fovx,0.5*self.fovx,self.Nx)
        ytmp = np.linspace(-0.5*self.fovy,0.5*self.fovy,self.Ny)
        
        # Determine the proper transposition based on if the reshaped x is fastest or slowest
        xtest = x.reshape([-1])
        if (xtest[1]==xtest[0]) :
            transpose = False
            xx=x[:,0]
            yy=y[0,:]
        else :
            transpose = True
            xx=x[0,:]
            yy=y[:,0]

        
        if ( (abs(xx[1]-xx[0])>abs(xtmp[-1]-xtmp[0])) and (abs(yy[1]-yy[0])>abs(ytmp[-1]-ytmp[0])) ) :
            I = 0*x
        elif (self.spline_method=='fft') :
            I = fft_cubic_spline_2d(xtmp,ytmp,f,xx,yy,a=self.a)
        elif (self.spline_method=='direct') :
            I = direct_cubic_spline_2d(xtmp,ytmp,f,xx,yy,a=self.a)
        else :
            raise NotImplementedError("Only 'fft' and 'direct' methods are implemented for the spline.")


        if (transpose) :
            I = np.transpose(I)
        
        return I


    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        names = []
        for ix in range(self.Nx) :
            for iy in range(self.Ny) :
                names.append(r'$\ln(I_{%i,%i})$'%(ix,iy))
        return names

    
class model_image_adaptive_splined_raster(model_image) :
    """
    Adaptive splined raster image class that is a mirror of :cpp:class:`Themis::model_image_adaptive_splined_raster`.
    Has parameters:

    * parameters[0] ........ Logarithm of the brightness at control point 0,0 (Jy/sr)
    * parameters[1] ........ Logarithm of the brightness at control point 1,0 (Jy/sr)

    ...

    * parameters[Nx-1] ..... Logarithm of the brightness at control point Nx-1,0 (Jy/sr)
    * parameters[Nx] ....... Logarithm of the brightness at control point 0,1 (Jy/sr)
    * parameters[Nx+1] ..... Logarithm of the brightness at control point 1,1 (Jy/sr)

    ...

    * parameters[Nx*Ny-1] .. Logarithm of the brightness at control point Nx-1,Ny-1 (Jy/sr)
    * parameters[Nx*Ny] .... Field of view in the x direction (rad)
    * parameters[Nx*Ny+1] .. Field of view in the y direction (rad)
    * parameters[Nx*Ny+2] .. Orientation of the x direction (rad)

    and size=Nx*Ny.

    Args:
      Nx (int): Number of control points in the -RA direction.
      Ny (int): Number of control points in the Dec direction.
      a (float): Spline control factor. Default: -0.5.
      spline_method (str): Spline method. Accepted values are 'fft' and 'direct'.  Default: 'fft'.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True

    Attributes:
      Nx (int): Number of control points in the -RA direction.
      Ny (int): Number of control points in the Dec direction.
      a (float): Spline control factor. Default: -0.5.
      spline_method (str): Spline method. Accepted values are 'fft' and 'direct'.  Default: 'fft'.
    """

    def __init__(self, Nx, Ny, a=-0.5, spline_method='fft', themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=Nx*Ny+3

        self.Nx = Nx
        self.Ny = Ny
        self.a = a
        self.spline_method = spline_method
        

    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        #f = np.transpose(np.exp(np.array(self.parameters[:-3]).reshape([self.Nx,self.Ny]))) * uas2rad**2
        f = np.transpose(np.exp(np.array(self.parameters[:-3]).reshape([self.Ny,self.Nx]))) * uas2rad**2
        #f = np.fliplr(f)
        f = np.flipud(f)
        
        fovx = self.parameters[-3]*rad2uas
        fovy = self.parameters[-2]*rad2uas
        PA = -self.parameters[-1]
        #PA = self.parameters[-1]

        xtmp = np.linspace(-0.5*fovx,0.5*fovx,self.Nx)
        ytmp = np.linspace(-0.5*fovy,0.5*fovy,self.Ny)

        # Determine the proper transposition based on if the reshaped x is fastest or slowest
        xtest = x.reshape([-1])
        if (xtest[1]==xtest[0]) :
            transpose = False
            xx=x[:,0]
            yy=y[0,:]
        else :
            transpose = True
            xx=x[0,:]
            yy=y[:,0]

        
        if ( (abs(xx[1]-xx[0])>abs(xtmp[-1]-xtmp[0])) and (abs(yy[1]-yy[0])>abs(ytmp[-1]-ytmp[0])) ) :
            I = 0*x
        elif (self.spline_method=='fft') :
            I = fft_cubic_spline_2d(xtmp,ytmp,f,xx,yy,PA,a=self.a)
        elif (self.spline_method=='direct') :
            I = direct_cubic_spline_2d(xtmp,ytmp,f,xx,yy,PA,a=self.a)
        else :
            raise NotImplementedError("Only 'fft' and 'direct' methods are implemented for the spline.")


        if (transpose) :
            I = np.transpose(I)
        
        return I

    
    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        names = []
        for ix in range(self.Nx) :
            for iy in range(self.Ny) :
                names.append(r'$\ln(I_{%i,%i})$'%(ix,iy))
        names.append('fovx (rad)')
        names.append('fovy (rad)')
        names.append(r'$\phi$ (rad)')
        return names



class model_image_stretch(model_image):
    """
    Smoothed image class that is a mirror of :cpp:class:`Themis::model_image_stretch`. Generates a smoothed 
    image from an existing image using an asymmetric Gaussian smoothing kernel.
    Has parameters:

    * parameters[0] ............. First parameter of underlying image
    
    ...

    * parameters[image.size-1] .. Last parameter of underlying image
    * parameters[image.size] .... stretch parameter :math:`\\tau = 1 - b/a` where b is semi-minor and a is semi-major
    * parameters[image.size+1] .. Position angle of stretch :math:`\\phi` (rad)

    and size=image.size+3.

    Args:
      image (model_image): :class:`model_image` object that will be smoothed.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.

    Attributes:
      image (model_image): A preexisting :class:`model_image` object to be stretched.
    """

    def __init__(self, image, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=image.size+2
        self.image = image

        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_image_stretch::generate_model`, 
        a similar function within the :cpp:class:`Themis::model_image_stretch` class.  Effectively 
        simply copies the parameters into the local list with some additional run-time checking.

        Args:
          parameters (list): Parameter list.

        Returns:
          None        
        """
        
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")

        self.image.generate(parameters[:-2])

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        # Doesn't to boundary checking here.  Probably not a major problem, but consider it in the future.
        srmaj = 1.0/np.sqrt(1-self.parameters[-2]) 
        srmin = np.sqrt(1-self.parameters[-2]) 
        sphi = self.parameters[-1]

        # Re-orient the grind along major and minor axes of the smoothing kernel
        xMinor = np.cos(sphi)*x - np.sin(sphi)*y
        xMajor = np.sin(sphi)*x + np.cos(sphi)*y
        px = x[1,1]-x[0,0]
        py = y[1,1]-y[0,0]

        self.image.generate(self.parameters[:-2])
        I = self.image.generate_intensity_map(xMinor*srmin,xMajor*srmaj,verbosity=verbosity)
        #I = self.image.generate_intensity_map(x,y,verbosity=verbosity)

        if (verbosity>0) :
            print('Stretched:',sr1,sr2,sphi,self.parameters)
            
        return I

    
    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        names = self.image.parameter_name_list()
        names.append(r'$\tau$')
        names.append(r'$\phi_\tau$ (rad)')
        return names

    



class model_image_smooth(model_image):
    """
    Smoothed image class that is a mirror of :cpp:class:`Themis::model_image_smooth`. Generates a smoothed 
    image from an existing image using an asymmetric Gaussian smoothing kernel.
    Has parameters:

    * parameters[0] ............. First parameter of underlying image
    
    ...

    * parameters[image.size-1] .. Last parameter of underlying image
    * parameters[image.size] .... Symmetrized standard deviation of smoothing kernel :math:`\\sigma` (rad)
    * parameters[image.size+1] .. Asymmetry parameter of smoothing kernel :math:`A` in (0,1)
    * parameters[image.size+2] .. Position angle of smoothing kernel :math:`\\phi` (rad)

    and size=image.size+3.

    Args:
      image (model_image): :class:`model_image` object that will be smoothed.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.

    Attributes:
      image (model_image): A preexisting :class:`model_image` object to be smoothed.
    """

    def __init__(self, image, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.size=image.size+3
        self.image = image

        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_image_smooth::generate_model`, 
        a similar function within the :cpp:class:`Themis::model_image_smooth` class.  Effectively 
        simply copies the parameters into the local list with some additional run-time checking.

        Args:
          parameters (list): Parameter list.

        Returns:
          None        
        """
        
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")

        self.image.generate(parameters[:-3])

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        # Doesn't to boundary checking here.  Probably not a major problem, but consider it in the future.
        sr1 = self.parameters[-3] * rad2uas * np.sqrt( 1./(1.-self.parameters[-2]) )
        sr2 = self.parameters[-3] * rad2uas * np.sqrt( 1./(1.+self.parameters[-2]) )
        sphi = self.parameters[-1]

        # Check to prevent convolving with zero
        dx = 1.5*max(abs(x[1,1]-x[0,0]),abs(y[1,1]-y[0,0]))
        if (sr1<dx) :
            sr1 = dx
        if (sr2<dx) :
            sr2 = dx
        
        # Re-orient the grind along major and minor axes of the smoothing kernel
        xMajor = np.cos(sphi)*x + np.sin(sphi)*y
        xMinor = -np.sin(sphi)*x + np.cos(sphi)*y
        exponent = 0.5*(xMajor**2/sr1**2 + xMinor**2/sr2**2)
        K = 1.0/(2*np.pi*sr1*sr2)*np.exp(-exponent) 
    
        px = abs(x[1,1]-x[0,0])
        py = abs(y[1,1]-y[0,0])

        self.image.generate(self.parameters[:-3])
        I = self.image.generate_intensity_map(x,y,verbosity=verbosity)
        Ism = np.abs(fftconvolve(I*px*py,K*px*py, mode='same'))

        if (verbosity>0) :
            print('Gaussian smoothed:',sr1,sr2,sphi,self.parameters)
            
        return Ism/(px*py)

    
    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        names = self.image.parameter_name_list()
        names.append(r'$\sigma_s$ (rad)')
        names.append(r'$A_s$')
        names.append(r'$\phi_s$ (rad)')
        return names

    
    def set_time(self,time) :
        """
        Sets time.

        Args:
          time (float): Time in hours.
        """
        
        self.time = time
        self.image.set_time(time)

    
    
    
class model_image_sum(model_image) :
    """
    Summed image class that is a mirror of :cpp:class:`Themis::model_image_sum`. Generates images by summing 
    other images supplemented with some offset. The parameters list is expanded by 2 for each 
    image, corresponding to the absolute positions. These may be specified either in Cartesian 
    or polar coordinates; this selection must be made at construction and applies uniformly to 
    all components (i.e., all component positions must be specified in the same coordinates).
    
    Args:
      image_list (list): List of :class:`model_image` objects. If None, must be set prior to calling :func:`intensity_map`. Default: None.
      offset_coordinates (str): Coordinates in which to specify component shifts.  Options are 'Cartesian' and 'polar'. Default: 'Cartesian'.
      reference_component (int): Index of component to define as reference. If None, will not set any component to be the reference. Default: None.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    
    Attributes:
      image_list (list): List of :class:`model_image` objects that are to be summed.
      shift_list (list): List of shifts transformed to Cartesian coordinates.
      offset_coordinates (str): Coordinates in which to specify component shifts.
    """

    def __init__(self,image_list=None, offset_coordinates='Cartesian', reference_component=None, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)

        self.image_list=[]
        self.shift_list=[]
        
        if (not image_list is None) :
            self.image_list = copy.copy(image_list) # This is a SHALLOW copy

        self.size=0
        for image in image_list :
            self.size += image.size+2
            self.shift_list.append([0.0, 0.0])

        self.offset_coordinates = offset_coordinates
        self.reference_component = reference_component
        
            
    def add(self,image) :
        """
        Adds an model_image object or list of model_image objects to the sum.  The size is recomputed.

        Args:
          image (model_image, list): A single model_image object or list of model_image objects to be added.

        Returns:
          None
        """
        
        # Can add lists and single istances
        if (isinstance(image,list)) :
            self.image_list = image_list + image
        else :
            self.image_list.append(image)
        # Get new size and shift list
        self.size=0
        self.shift_list = []
        for image in image_list :
            self.size += image.size+2
            self.shift_list.append([0.0,0.0])

    def set_reference_component(self,reference_component) :
        self.reference_component = reference_component

            
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_image_sum::generate_model`, 
        a similar function within the :cpp:class:`Themis::model_image_sum` class.  
        Effectively simply copies the parameters into the local list, transforms
        and copies the shift vector (x0,y0) for each model_image object, and performs some run-time checking.

        Args:
          parameters (list): Parameter list.

        Returns:
          None        
        """
        
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")

        q = self.parameters
        for k,image in enumerate(self.image_list) :
            image.generate(q[:image.size])

            if (self.offset_coordinates=='Cartesian') :
                self.shift_list[k][0] = q[image.size] * rad2uas
                self.shift_list[k][1] = q[image.size+1] * rad2uas
            elif (self.offset_coordinates=='polar') :
                self.shift_list[k][0] = q[image.size]*np.cos(q[image.size+1]) * rad2uas
                self.shift_list[k][1] = q[image.size]*np.sin(q[image.size+1]) * rad2uas
                
            q = q[(image.size+2):]


        if (not (self.reference_component is None)) :
            if (self.reference_component>len(self.image_list)) :
                raise RuntimeError("Reference component %i is not in the list of images, which numbers only %i."%(self.reference_component,len(self.image_list)))
            x0 = self.shift_list[self.reference_component][0]
            y0 = self.shift_list[self.reference_component][1]
            for k in range(len(self.image_list)) :
                self.shift_list[k][0] = self.shift_list[k][0]-x0
                self.shift_list[k][1] = self.shift_list[k][1]-y0

            
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        I = np.zeros(x.shape)
        q = self.parameters
        for k,image in enumerate(self.image_list) :
            dx = x-self.shift_list[k][0]
            dy = y-self.shift_list[k][1]

            I = I+image.generate_intensity_map(dx,dy,verbosity=verbosity)

            q=self.parameters[(image.size+2):]

            if (verbosity>0) :
                print("   --> Sum shift:",self.shift_list[k])
            
        return I


    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        names = []
        for image in self.image_list :
            names.extend(image.parameter_name_list())
            if (self.offset_coordinates=='Cartesian') :
                names.append(r'$\Delta x$ (rad)')
                names.append(r'$\Delta y$ (rad)')
            elif (self.offset_coordinates=='polar') :
                names.append(r'$\Delta r$ (rad)')
                names.append(r'$\Delta \theta$ (rad)')
        return names

    
    def set_time(self,time) :
        """
        Sets time.

        Args:
          time (float): Time in hours.
        """
        
        self.time = time
        
        for image in self.image_list :
            image.set_time(time)

    

class model_image_polynomial_variable(model_image) :
    """
    Generates a :class:`model_image` object with parameters that vary polynomially with 
    time from an initially static :class:`model_image` object.

    Args:
      image (model_image): Static image from which to generate a dynamic image.
      orders (list): List of integer orders of the polynomials in time of each parameter.
      reference_time (list): Reference time that defines the time origin.
    """    
    def __init__(self, image, orders, reference_time, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)

        self.image = image
        self.orders = orders
        if (len(self.orders)<self.image.size) :
            raise RuntimeError("Length of orders list must match number of parameters in static image.")
        
        self.reference_time = reference_time/3600.0 # Reference time in hr, but provided in s
        self.size = np.sum(orders)+len(orders)
        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_image_polynomial_variable::generate_model`, 
        a similar function within the :cpp:class:`Themis::model_image_polynomial_variable` class.  

        Args:
          parameters (list): Parameter list.

        Returns:
          None        
        """
        
            
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")

        q = np.zeros(self.image.size)
        j = 0
        dt = (self.time-self.reference_time)*3600.0 # hr to s
        for k in range(self.image.size) :
            for order in range(self.orders[k]+1) :
                q[k] = q[k] + self.parameters[j] * (dt**order)
                j += 1

        self.image.generate(q)

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        if (verbosity>0) :
            print("PolynomialVariable:",self.time,self.reference_time,self.image.parameters[:3])
        
        return self.image.generate_intensity_map(x,y,verbosity=verbosity)


    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        static_names = image.parameter_name_list()
        names = []
        for k in range(self.image.size) :
            names.extend(static_names[k])
            if (orders[k]>0) :
                names.extend('$d/dt$ %s'%(static_names[k]))
            if (orders[k]>1) :
                for order in range(2,orders[k]) :
                    names.extend('$d^%i/dt^%i %s'%(order,order,static_names[k]))
        return names


    def set_time(self,time) :
        """
        Sets time.

        Args:
          time (float): Time in hours.
        """
        
        self.time = time

        self.image.set_time(time)

        self.parameters = None
        
    
class model_image_single_point_statistics(model_image) :
    """
    Generates a :class:`model_image` object that is filed with a desired set of 
    single-point statistics (mean, median, standard deviation, kurtosis, etc.) given
    a specified model_image object and appropriate chain data.
    
    Args:
      image (model_image): Static image from which to generate a dynamic image.
      stats_list (list): List of strings of the statistics to generate.  Options are limited to subsets of ['mean','median','stddev','skewness','kurtosis'].
      median_store_size (int): Number of instances about the middle to save in the construction of the median.  This is bounded from above by half of the number of chain samples.
    """

    def __init__(self,image, stats_list=None, median_store_size=16, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)

        self.image = image

        if (stats_list is None) :
            self.stats_list = ['mean', 'median', 'stddev']
        else :
            self.stats_list = stats_list
            
        self.default_stat = self.stats_list[0]
            
        self.Idict={}
        for stat in self.stats_list :
            self.Idict[stat] = None
            
        self.stats_generated = False

        self.median_store_size = median_store_size

        self.chain = None
        
            
    def generate(self,chain) :
        """
        Sets the chain from which to compute image statistics and the statistic of interest
        to be returned on the next call to :func:`intensity_map`.

        Args:
          chain (numpy.ndarray): MCMC chain of parameters ordered [samples,parameters].
        """

        if (np.all(self.chain==chain)) :
            return

        self.chain = chain
        
        self.stats_generated = False
        self.x = None
        self.y = None


    def set_default_stat(self,stat) :
        """
        Sets the statistic to be returned on the next call to :func:`intensity_map`.

        Args:
          stat (str): A statistic in the stats_list.
        """

        if (not (stat in self.stats_list)) :
            raise RuntimeError(("Requested statistic %s is not in currently computed statistic list:"+len(self.stats_list)*" %s")%(stat,tuple(self.stats_list)))
        else :
            self.default_stat = stat


    def set_median_store_size(self,median_store_size) :
        """
        Sets the number of images to remember in the computation of the median.  If 
        set to None, will store all images in the chain, possibly running afoul of 
        system memory constraints.

        Args:
          median_store_size(int): Number of images to store.
        """
        self.median_store_size = size

        
    def generate_intensity_map(self,x,y,verbosity=0) :

        # If new place, regenerate
        if (np.any(self.x!=x) or np.any(self.y!=y)) :
            self.x=x
            self.y=y
            self.stats_generated = False
            
        # If not generated, generate the relevant bits
        if (self.stats_generated==False) :

            for stat in self.Idict.keys() :
                self.Idict[stat] = np.zeros(x.shape)
            
            nparams = self.chain.shape[-1]
            flattened_chain = self.chain.reshape([-1,nparams])

            if (verbosity) :
                print("Number of parameters is %i"%(nparams))
                print("Size of chain / flattened chain :",self.chain.shape,flattened_chain.shape)

            
            # Generate room for keeping track of images for median computation.
            if ('median' in self.stats_list) :
                if ( self.median_store_size is None ) :
                    nmedstore = flattened_chain.shape[0]
                else :
                    nmedstore = self.median_store_size
                I_median_list = np.zeros((nmedstore+1,)+x.shape)
                I_median_boundary_count = np.zeros((2,)+x.shape).astype(int)

            # Determine which things to computes
            compute_median = ( 'median' in self.stats_list )
            compute_kurtosis = ( 'kurtosis' in self.stats_list )
            compute_skewness = ( 'skewness' in self.stats_list ) or compute_kurtosis
            compute_stddev = ( 'stddev' in self.stats_list ) or compute_skewness
            compute_mean = ( 'mean' in self.stats_list ) or compute_stddev

            
            # Loop over chain elmenets and accumulate quantities that will be turned into the desired statisics
            if (verbosity==0) :
                pb = progress_bar('Stats Computation:',length=40)
                pb.start()
                
            for k in range(flattened_chain.shape[0]) :

                if (verbosity>0) :
                    print("  On frame k = %i"%(k))
                else :
                    pb.increment(k/float(flattened_chain.shape[0]))
                
                #I = self.image.intensity_map(x,y,flattened_chain[k,:],verbosity=verbosity)

                self.image.generate(flattened_chain[k,:])
                I = self.image.generate_intensity_map(x,y,verbosity=verbosity)
                
                if ( compute_mean ) :
                    self.Idict['mean'] = self.Idict['mean'] + I

                if ( compute_stddev ) :
                    self.Idict['stddev'] = self.Idict['stddev'] + I**2

                if ( compute_skewness ) :
                    self.Idict['skewness'] = self.Idict['skewness'] + I**3

                if ( compute_kurtosis ) :
                    self.Idict['kurtosis'] = self.Idict['kurtosis'] + I**4

                if ( compute_median ) :
                    if (k<nmedstore) :
                        # Just store
                        I_median_list[k,:,:] = I
                    else :
                        # Store new image
                        I_median_list[-1,:,:] = I
                        # Sort the new list
                        I_median_list = np.sort(I_median_list,axis=0)
                        # Find the new median with the boundary shifts
                        index_median = (I_median_boundary_count[1,:,:]-I_median_boundary_count[0,:,:] + nmedstore )//2
                        # Remove the side furthest from the median
                        shift_median = (index_median>0.5*nmedstore).astype(int)
                        for ix in range(x.shape[0]) :
                            for iy in range(x.shape[1]) :
                                I_median_list[0:nmedstore,ix,iy] = I_median_list[(shift_median[ix,iy]):(nmedstore+shift_median[ix,iy]),ix,iy]
                        #I_median_list[0:nmedstore,:,:] = I_median_list[(shift_median):(nmedstore+shift_median),:,:]
                        # Accumulate the lost side
                        I_median_boundary_count[0,:,:] += (shift_median==1)
                        I_median_boundary_count[1,:,:] += (shift_median==0)


            # Finish the computation of the statistics
            if ( compute_mean ) :
                self.Idict['mean'] = self.Idict['mean'] / float(flattened_chain.shape[0])

            if ( compute_stddev ) :
                self.Idict['stddev'] = np.sqrt( self.Idict['stddev'] / float(flattened_chain.shape[0]) - self.Idict['mean']**2 )
                
            if ( compute_skewness ) :
                self.Idict['skewness'] = ( self.Idict['skewness'] / float(flattened_chain.shape[0]) - 3.0*self.Idict['mean']*self.Idict['stddev']**2 - self.Idict['mean']**3 )/(self.Idict['stddev']**3)
                
            if ( compute_kurtosis ) :
                self.Idict['kurtosis'] = ( self.Idict['kurtosis'] / float(flattened_chain.shape[0]) - 4.0*self.Idict['mean']*(self.Idict['stddev']**3)*self.Idict['skewness'] - 6.0*(self.Idict['mean']**2)*(self.Idict['stddev']**2)  - self.Idict['mean']**4 )/(self.Idict['stddev']**4)

            if ( compute_median ) :
                I_median_list = np.sort(I_median_list[:-1,:,:],axis=0)
                index_median = (I_median_boundary_count[1,:,:]-I_median_boundary_count[0,:,:] + nmedstore )//2
                for ix in range(x.shape[0]) :
                    for iy in range(x.shape[1]) :
                        self.Idict['median'][ix,iy] = I_median_list[index_median[ix,iy],ix,iy]
            
            if (verbosity==0) :
                pb.finish()

            self.stats_generated = True

            
        # Return desired quantities
        return self.Idict[self.default_stat]

    
    def parameter_name_list(self) :
        return self.image.parameter_name_list()

    
    def set_time(self,time) :
        """
        Sets time.

        Args:
          time (float): Time in hours.
        """
        
        self.time = time
        self.image.set_time(time)

        
    
def expand_model_glob_ccm_mexico(model_glob) :
    """
    Expands an abbreviated model glob as produced by the ccm_mexico driver 
    and generates a list of individual model specifiers. Model glob options 
    are supplemented beyond that in the ccm_mexico driver in that they permit 
    unlimited components of any type (i.e., the ccm_mexico+ standard).

    Current list of accepted model strings are:

    * 'g' ....... :class:`model_image_symmetric_gaussian`
    * 'G' ....... :class:`model_image_symmetric_gaussian`
    * 'a' ....... :class:`model_image_asymmetric_gaussian`
    * 'A' ....... :class:`model_image_asymmetric_gaussian`
    * 'c' ....... :class:`model_image_crescent`
    * 'x' ....... :class:`model_image_xsring`
    * 'X' ....... :class:`model_image_xsringauss`
    * 'r' ....... :class:`model_image_splined_raster`, hyperparameters must be provided for ths model.  Required: {'splined_raster_Nx':Nx, 'splined_raster_Ny':Ny, 'splined_raster_fovx':fovx, 'splined_raster_fovy':fovy}.  Optional: {'splined_raster_a':-0.5, 'splined_raster_fft':'fft', 'splined_raster_themis_fft_sign':True}
    * 'R' ....... :class:`model_image_adaptive_splined_raster`, hyperparameters must be provided for ths model.  Required: {'splined_raster_Nx':Nx, 'splined_raster_Ny':Ny}.  Optional: {'splined_raster_a':-0.5, 'splined_raster_fft':'fft', 'splined_raster_themis_fft_sign':True}
    * 's' ....... The immediately following specification will be assumed to be part of a :class:`model_image_smooth` object.
    * '<#>' ..... The immediately preceding model will repeated the specified number of times (e.g., 'a3' will be expanded to 'aaa', 'sX3' will be expanded to 'sXsXsX').

    Args:
      model_glob (str): Model glob string.
      hyper_parameters (dict) : Optional set of hyperparameters for image models that require them.  Where they are not provided for required arguments a RuntimeError will be raised.

    Returns:
      (list): List of individual model specifiers.
    """

    # Make sure that model_glob is a string
    if (not isinstance(model_glob,str)) :
        raise RuntimeError("Unrecognized model_glob %s. model_glob must be string formatted as described in the documentation."%(model_glob))

    # Expand model_glob to take out numbers
    expanded_model_list = []
    intsplit_model_glob = re.split('(\d+)',model_glob)
    
    ks = 0
    while ( ks<len(intsplit_model_glob) ) :
        
        ms = intsplit_model_glob[ks]

        # Peak ahead for integers
        repeat_number=1
        if ( ks+1<len(intsplit_model_glob) ) :
            if ( intsplit_model_glob[ks+1].isdigit ) :
                repeat_number = int(intsplit_model_glob[ks+1])
                ks = ks+1
        
        k=0
        while ( k<len(ms) ) :

            m = ms[k]

            # If smooth key-character passed, grab next specifier
            if (m=='s') :
                m = model_glob[k:k+2]
                k = k+1

            if (k==len(ms)-1) :
                for j in range(repeat_number) :
                    expanded_model_list.append(m)
            else :
                expanded_model_list.append(m)

            k = k+1

        ks = ks+1

    return expanded_model_list


    
def construct_model_image_from_ccm_mexico(model_glob,hyperparameters=None,verbosity=0) :
    """
    Reads a model glob as produced by the ccm_mexico driver and generates 
    a corresponding :class:`model_image` object. If more than one model
    specifier is provided, an appropriate :class:`model_image_sum` object 
    is returned. Model glob options are supplemented beyond that in the 
    ccm_mexico driver in that they permit unlimited components of any type 
    (i.e., the ccm_mexico+ standard).

    Current list of accepted model strings are listed in :func:`expand_model_glob_ccm_mexico`.  
    In addition, for the following model specifiers the associated hyperparameters
    must be provided as a dictionary:

    * 'r' ....... :class:`model_image_splined_raster`, hyperparameters must be provided for ths model.  Required: {'splined_raster_Nx':Nx, 'splined_raster_Ny':Ny, 'splined_raster_fovx':fovx, 'splined_raster_fovy':fovy}.  Optional: {'splined_raster_a':-0.5, 'splined_raster_fft':'fft', 'splined_raster_themis_fft_sign':True}
    * 'R' ....... :class:`model_image_adaptive_splined_raster`, hyperparameters must be provided for ths model.  Required: {'splined_raster_Nx':Nx, 'splined_raster_Ny':Ny}.  Optional: {'splined_raster_a':-0.5, 'splined_raster_fft':'fft', 'splined_raster_themis_fft_sign':True}

    Args:
      model_glob (str): Model glob string.
      hyper_parameters (dict) : Optional set of hyperparameters for image models that require them. This optionally includes {'offset_coordinates':'Cartesian'/'polar'} to set the coordinate type for the construction of a :class:`model_image_sum` object when more than one element is implied by the model_glob. Where they are not provided for required arguments a RuntimeError will be raised.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints model_glob information.

    Returns:
      (model_image): A corresponding model image object.
    """

    if (verbosity>0) :
        print("Generating model_image object for %s"%(model_glob))

    expanded_model_list = expand_model_glob_ccm_mexico(model_glob)
            
    # A list of the constructed image objects
    image_list=[]
    for m in expanded_model_list :
        if (verbosity>0) :
            print(m)
        if (m[-1]=='g') :
            image_list.append(model_image_symmetric_gaussian())
        elif (m[-1]=='G') :
            image_list.append(model_image_symmetric_gaussian())
        elif (m[-1]=='a') :
            image_list.append(model_image_asymmetric_gaussian())
        elif (m[-1]=='A') :
            image_list.append(model_image_asymmetric_gaussian())
        elif (m[-1]=='c') :
            image_list.append(model_image_crescent())
        elif (m[-1]=='x') :
            image_list.append(model_image_xsring())
        elif (m[-1]=='X') :
            image_list.append(model_image_xsringauss())
        elif (m[-1]=='r') :
            hpreq = ['splined_raster_Nx','splined_raster_Ny','splined_raster_fovx','splined_raster_fovy']
            hpopt = ['splined_raster_a','splined_raster_fft','splined_raster_themis_fft_sign']
            hpvals = [None,None,None,None,-0.5,'fft',True]
            if (hyperparameters is None) :
                raise RuntimeError("hyperparameters must be provided to generate a model_image_splined_raster object.")
            for j in range(len(hpreq)) :
                if ( hpreq[j] in hyperparameters.keys() ) :
                    hpvals[j] = hyperparameters[hpreq[j]]
                else :
                    raise RuntimeError("%s must be provided in the hyperparameters passed to generate a model_image_splined_raster object."%(hpreq[j]))
            for j in range(len(hpopt)) :
                if ( hpopt[j] in hyperparameters.keys() ) :
                    hpvals[j+len(hpreq)] = hyperparameters[hpopt[j]]
            image_list.append(model_image_splined_raster(hpvals[0],hpvals[1],hpvals[2],hpvals[3],a=hpvals[4],spline_method=hpvals[5],themis_fft_sign=hpvals[6]))
        elif (m[-1]=='R') :
            hpreq = ['adaptive_splined_raster_Nx','adaptive_splined_raster_Ny']
            hpopt = ['adaptive_splined_raster_a','adaptive_splined_raster_fft','adaptive_splined_raster_themis_fft_sign']
            hpvals = [None,None,-0.5,'fft',True]
            if (hyperparameters is None) :
                raise RuntimeError("hyperparameters must be provided to generate a model_image_splined_raster object.")
            for j in range(len(hpreq)) :
                if ( hpreq[j] in hyperparameters.keys() ) :
                    hpvals[j] = hyperparameters[hpreq[j]]
                else :
                    raise RuntimeError("%s must be provided in the hyperparameters passed to generate a model_image_splined_raster object."%(hpreq[j]))
            for j in range(len(hpopt)) :
                if ( hpopt[j] in hyperparameters.keys() ) :
                    hpvals[j+len(hpreq)] = hyperparameters[hpopt[j]]
            image_list.append(model_image_adaptive_splined_raster(hpvals[0],hpvals[1],a=hpvals[2],spline_method=hpvals[3],themis_fft_sign=hpvals[4]))
        else :
            raise RuntimeError("%s is not a valid model specifier in the ccm_mexico+ tag version set."%m[-1])
            
        if (m[0]=='s') :
            image_list[-1] = model_image_smooth(image_list[-1])

    if (len(image_list)==0) :
        return None
    elif (len(image_list)==1) :
        return image_list[0]
    else :
        if (not (hyperparameters is None)) and ( 'offset_coordinates' in hyperparameters.keys() ) :
            return ( model_image_sum(image_list,offset_coordinates=hyperparameters['offset_coordinates']) )
        else :
            return ( model_image_sum(image_list) )
            

def write_model_tag_file_from_ccm_mexico(model_glob,hyperparameters=None,tag_file_name='model_image.tag') :
    """
    Reads a model glob as produced by the ccm_mexico driver and writes a
    corresponding tagvers-1.0 model_image.tag file. Model glob options are 
    supplemented beyond that in the ccm_mexico driver in that they permit 
    unlimited components of any type (i.e., the ccm_mexico+ standard).

    Current list of accepted model strings are listed in :func:`expand_model_glob_ccm_mexico`.  
    In addition, for the following model specifiers the associated hyperparameters
    must be provided as a dictionary:

    * 'r' ....... :class:`model_image_splined_raster`, hyperparameters must be provided for ths model.  Required: {'splined_raster_Nx':Nx, 'splined_raster_Ny':Ny, 'splined_raster_fovx':fovx, 'splined_raster_fovy':fovy}.  Optional: {'splined_raster_a':-0.5, 'splined_raster_fft':'fft', 'splined_raster_themis_fft_sign':True}
    * 'R' ....... :class:`model_image_adaptive_splined_raster`, hyperparameters must be provided for ths model.  Required: {'splined_raster_Nx':Nx, 'splined_raster_Ny':Ny}.  Optional: {'splined_raster_a':-0.5, 'splined_raster_fft':'fft', 'splined_raster_themis_fft_sign':True}

    Args:
      model_glob (str): Model glob string.
      hyper_parameters (dict) : Optional set of hyperparameters for image models that require them.  Where they are not provided for required arguments a RuntimeError will be raised.
      tag_file_name (str): Name of tag file to be written. Default: 'model_image.tag'.
    """

    print("Generating tag file for %s"%(model_glob))
    
    # Generate
    expanded_model_list = expand_model_glob_ccm_mexico(model_glob)

    # Open tag file
    tagout=open(tag_file_name,'w')
    tagout.write("tagvers-1.0\n")
    
    # If a model_image_sum object:
    if (len(expanded_model_list)>1) :
        offset_coordinates = 'Cartesian'
        if ( not (hyperparameters is None) ):
            if ('offset_coordinates' in hyperparameters.keys()) :
                offset_coordinates = hyperparameters['offset_coordinates']
        tagout.write("model_image_sum %s\n"%(offset_coordinates))
        tagout.write("SUBTAG START\n")

    for m in expanded_model_list :
        if (m[0]=='s') :
            tagout.write("model_image_smooth\n")
            tagout.write("SUBTAG START\n")
        if (m[-1]=='g') :
            tagout.write("model_image_symmetric_gaussian\n")
        elif (m[-1]=='G') :
            tagout.write("model_image_symmetric_gaussian\n")
        elif (m[-1]=='a') :
            tagout.write("model_image_asymmetric_gaussian\n")
        elif (m[-1]=='A') :
            tagout.write("model_image_asymmetric_gaussian\n")
        elif (m[-1]=='c') :
            tagout.write("model_image_crescent\n")
        elif (m[-1]=='x') :
            tagout.write("model_image_xsring\n")
        elif (m[-1]=='X') :
            tagout.write("model_image_xsringauss\n")
        elif (m[-1]=='r') :
            hpreq = ['splined_raster_Nx','splined_raster_Ny','splined_raster_fovx','splined_raster_fovy']
            hpopt = ['splined_raster_a']
            hpvals = [None,None,None,None,-0.5]
            if (hyperparameters is None) :
                raise RuntimeError("hyperparameters must be provided to generate a model_image_splined_raster object.")
            for j in range(len(hpreq)) :
                if ( hpreq[j] in hyperparameters.keys() ) :
                    hpvals[j] = hyperparameters[hpreq[j]]
                else :
                    raise RuntimeError("%s must be provided in the hyperparameters passed to generate a model_image_splined_raster object."%(hpreq[j]))
            for j in range(len(hpopt)) :
                if ( hpopt[j] in hyperparameters.keys() ) :
                    hpvals[j+len(hpreq)] = hyperparameters[hpopt[j]]
            tagout.write("model_image_splined_raster %i %i %g %g %g\n"%(hpvals[0],hpvals[1],hpvals[2],hpvals[3],hpvals[4]))
        elif (m[-1]=='R') :
            hpreq = ['adaptive_splined_raster_Nx','adaptive_splined_raster_Ny']
            hpopt = ['adaptive_splined_raster_a']
            hpvals = [None,None,-0.5]
            if (hyperparameters is None) :
                raise RuntimeError("hyperparameters must be provided to generate a model_image_splined_raster object.")
            for j in range(len(hpreq)) :
                if ( hpreq[j] in hyperparameters.keys() ) :
                    hpvals[j] = hyperparameters[hpreq[j]]
                else :
                    raise RuntimeError("%s must be provided in the hyperparameters passed to generate a model_image_splined_raster object."%(hpreq[j]))
            for j in range(len(hpopt)) :
                if ( hpopt[j] in hyperparameters.keys() ) :
                    hpvals[j+len(hpreq)] = hyperparameters[hpopt[j]]
            tagout.write("model_image_adaptive_splined_raster %i %i %g\n"%(hpvals[0],hpvals[1],hpvals[2]))
        else :
            raise RuntimeError("%s is not a valid model specifier in the ccm_mexico+ tag version set."%m[-1])
        if (m[0]=='s') :
            tagout.write("SUBTAG FINISH\n")

    # If a model_image_sum object:
    if (len(expanded_model_list)>1) :
        tagout.write("SUBTAG FINISH\n")

    # Finish tag file
    tagout.close()
    


def tagv1_find_subtag(tag) :
    """
    Given a tag that begins with SUBTAG START, find the subtag within the matching
    SUBTAG FINISH.

    Args:
      tag (list): List of tag lines.

    Returns:
      (list): List of tag lines.
    """

    # Make sure that the first line is indeed SUBTAG START
    if (tag[0]!="SUBTAG START") :
        RuntimeError("Invalid subtag!  Expected SUBTAG START.  Found %s."%(tag[0]))
    
    # Find index of matching SUBTAG FINISH
    subtag_level=1
    k=1
    while (subtag_level>0) :
        if (tag[k]=="SUBTAG START") :
            subtag_level = subtag_level+1
        elif (tag[k]=="SUBTAG FINISH") :
            subtag_level = subtag_level-1
        k = k+1

    return tag[1:(k-1)]
    
    
def construct_model_image_from_tagv1(tag,verbosity=0) :
    """
    Parses a tagvers-1.0 Themis tag and recursively constructs a model image.

    Args:
      tag (list) : List of tags, arranged as str on separate lines.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints tag information.

    Returns:
      (model_image, list) : The first :class:`model_image` object fully described within the tag; the remaining tag lines.
    """
    #print(tag)

    if (verbosity>0) :
        for j,l in enumerate(tag) :
            print("%i : %s"%(j,tag[j]))
        print("---------------------------------")

    if (tag[0]=='model_image_symmetric_gaussian') :
        return model_image_symmetric_gaussian(),tag[1:]

    elif (tag[0]=='model_image_asymmetric_gaussian') :
        return model_image_asymmetric_gaussian(),tag[1:]

    elif (tag[0]=='model_image_crescent') :
        return model_image_crescent(),tag[1:]

    elif (tag[0]=='model_image_xsring') :
        return model_image_xsring(),tag[1:]

    elif (tag[0].split()[0]=='model_image_mring') :
        toks = tag[0].split()
        return model_image_mring(int(toks[1])),tag[1:]
    
    elif (tag[0].split()[0]=='model_image_mring_floor') :
        toks = tag[0].split()
        return model_image_mring_floor(int(toks[1])),tag[1:]
    
    elif (tag[0]=='model_image_xsringauss') :
        return model_image_xsringauss(),tag[1:]
    
    elif (tag[0]=="model_image_xsring_ellip") :
        return model_image_xsring_ellip(),tag[1:]

    elif (tag[0].split()[0]=='model_image_splined_raster') :
        toks = tag[0].split()
        # return model_image_splined_raster(float(toks[1]),float(toks[2]),int(toks[3]),float(toks[4]),float(toks[5]),int(toks[6]),float(toks[7])),tag[1:]
        # return model_image_splined_raster(int(toks[1]),int(toks[2]),float(toks[3]),float(toks[4]),float(toks[5])),tag[1:]
        return model_image_splined_raster(int(toks[3]),int(toks[6]),float(toks[2])-float(toks[1]),float(toks[5])-float(toks[4]),float(toks[7])),tag[1:]

    elif (tag[0].split()[0]=='model_image_adaptive_splined_raster') :
        toks = tag[0].split()
        return model_image_adaptive_splined_raster(int(toks[1]),int(toks[2]),float(toks[3])),tag[1:]

    elif (tag[0]=='model_image_stretch') :
        subtag = tagv1_find_subtag(tag[1:])
        subimage,_ = construct_model_image_from_tagv1(subtag,verbosity=verbosity)
        return model_image_stretch(subimage),tag[(len(subtag)+3):]
    
    elif (tag[0]=='model_image_smooth') :
        subtag = tagv1_find_subtag(tag[1:])
        subimage,_ = construct_model_image_from_tagv1(subtag,verbosity=verbosity)
        return model_image_smooth(subimage),tag[(len(subtag)+3):]

    elif (tag[0].split()[0]=='model_image_sum') :
        offset_coordinates = tag[0].split()[1]
        subtag = tagv1_find_subtag(tag[1:])
        len_subtag = len(subtag)
        image_list = []
        while (len(subtag)>0) :
            subimage,subtag = construct_model_image_from_tagv1(subtag,verbosity=verbosity)
            image_list.append(subimage)
        return model_image_sum(image_list,offset_coordinates=offset_coordinates),tag[len_subtag+3:]

    elif (tag[0].split()[0]=='model_image_polynomial_variable') :
        reference_time = float(tag[0].split()[1])
        orders = [ int(order) for order in tag[0].split()[2:]]
        subtag = tagv1_find_subtag(tag[1:])
        subimage,_ = construct_model_image_from_tagv1(subtag,verbosity=verbosity)
        return model_image_polynomial_variable(subimage,orders,reference_time),tag[(len(subtag)+3):]

    else :
        raise RuntimeError("Unrecognized model tag %s"%(tag[0]))

    
def construct_model_image_from_tag_file(tag_file_name='model_image.tag', tagversion='tagvers-1.0',verbosity=0) :
    """
    Reads tag file produced by the :cpp:func:`Themis::model_image::write_model_tag_file` function 
    and returns an appropriate model image. Model tag files can be generated from ccm_mexico+ model 
    specification strings using :func:`write_model_tag_file_from_ccm_mexico`. For details about
    the available model tag syntax, see :cpp:func:`Themis::model_image::write_model_tag_file`.

    Args:
      tag_file_name (str): Name of the tag_file. Default: 'model_image.tag'
      tagversion (str): Version of tag system. Currently only tagvers-1.0 is implemented. Default 'tagvers-1.0'.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints tag information.

    Returns:
      (model_image): :class:`model_image` object of the corresponding model.
    """


    tagin = open(tag_file_name,'r')
    tag = tagin.readlines()
    tagin.close()
    
    tag = [l.strip('\n\r') for l in tag]
    tagversion_file = tag[0]
    tag = tag[1:]

    if (tagversion_file==tagversion) :
        image,tag = construct_model_image_from_tagv1(tag,verbosity=verbosity)
        if (len(tag)>0) :
            raise RuntimeError(("Remaining tag lines:"+len(tag)*("   %s\n"))%tuple(tag))
        return image
    else:
        raise RuntimeError("Tag versions other than tagvers-1.0 are not supported at this time.")


    
def plot_intensity_map(model_image, parameters, limits=None, shape=None, colormap='afmhot', Imin=0.0, Imax=None, in_radec=True, xlabel=None, ylabel=None, return_intensity_map=False, transfer_function='linear', verbosity=0) :
    """
    Plots an intensity map associated with a given :class:`model_image` object evaluated at a 
    specified set of parameters. A number of additional practical options may be passed. Access
    to a matplotlib pcolor object, the figure handle and axes hanldes are returned. Optionally,
    the intensity map is returned.

    Args:
      model_image (model_image): :class:`model_image` object to plot.
      parameters (list): List of parameters that define model_image intensity map.
      limits (float,list): Limits in uas on image size. If a single float is given, the limits are uniformly set. If a list is given, it must be in the form [xmin,xmax,ymin,ymax]. Default: 75
      shape (int,list): Number of pixels. If a single int is given, the number of pixels are uniformly set in both directions.  If a list is given, it must be in the form [Nx,Ny]. Default: 256
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'afmhot'.
      Imin (float): Minimum intensity for colormap. Default: 0 (though see transfer_function).
      Imax (float): Maximum intensity for colormap. Default: maximum intensity.
      in_radec (bool): If True, plots in :math:`\\Delta RA` and :math:`\\Delta Dec`.  If False, plots in sky-right Cartesian coordinates. Default: True
      xlabel (str): Label for xaxis. Default: ':math:`\\Delta RA (\\mu as)` if in_radec is True, :math:`\\Delta x (\\mu as)` if in_radec is False
      ylabel (str): Label for yaxis. Default: ':math:`\\Delta Dec (\\mu as)` if in_radec is True, :math:`\\Delta y (\\mu as)` if in_radec is False
      return_intensity_map (bool): If True returns the intensity map in addition to standard return items.
      transfer_function (str): Transfer function to plot.  Options are 'linear','sqrt','log'.  If 'log', if Imin=0 it will be set to Imax/1000.0 by default, else if Imin<0 it will be assumed to be a dynamic range and Imin = Imax * 10**(Imin).
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

    Returns :
      (matplotlib.pyplot.image,matplotlib.pyplot.axes,matplotlib.pyplot.fig,[numpy.ndarray,numpy.ndarray,numpy.ndarray]): Image handle; Figure object; Axes object; Optionally intensity map (x,y,I).
    """
    
    # Shape limits
    if (limits is None) :
        limits = [-75, 75, -75, 75]
    elif (isinstance(limits,list)) :
        limits = limits
    else :
        limits = [-limits, limits, -limits, limits]

    # Set shape
    if (shape is None) :
        shape = [256, 256]
    elif (isinstance(shape,list)) :
        shape = shape
    else :
        shape = [shape, shape]

    # Set labels
    if (xlabel is None) :
        if (in_radec) :
            xlabel=r'$\Delta$RA ($\mu$as)'
        else :
            xlabel=r'$\Delta$x ($\mu$as)'
    if (ylabel is None) :
        if (in_radec) :
            ylabel=r'$\Delta$Dec ($\mu$as)'
        else :
            ylabel=r'$\Delta$y ($\mu$as)'
        

    if (verbosity>0) :
        print("limits:",limits)
        print("shape:",shape)
    
    # Generate a set of pixels
    x,y = np.mgrid[limits[0]:limits[1]:shape[0]*1j,limits[2]:limits[3]:shape[1]*1j]
    # Generate a set of intensities (Jy/uas^2)
    I = model_image.intensity_map(x,y,parameters,verbosity=verbosity)
    
    # Flip x to flip x axis so that RA decreases right to left
    if (in_radec) :
        x = -x
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])
    
    # Determine scale
    if (Imax is None) :
        Imax = np.max(I)

    if (verbosity>0) :
        print("Minimum/Maximum I: %g / %g"%(Imin,Imax))
        
    # Transfered I for plotting
    if (transfer_function=='linear') :
        tI = I/Imax
        Imin = Imin/Imax
        Imax = 1.0
    elif (transfer_function=='sqrt') :
        tI = np.sqrt(I/Imax)
        Imin = np.sqrt(Imin/Imax)
        Imax = 1.0
    elif (transfer_function=='log') :
        tI = np.log10(I/Imax)
        if (Imin==0) :
            Imin = np.log10(1.0e-3)
        elif (Imin>0) :
            Imin = np.log10(Imin/Imax)
        Imax = 0.0

    # Set background color in Axes
    cmap = cm.get_cmap(colormap)    
    plt.gca().set_facecolor(cmap(0.0))

    # Make plot
    h = plt.pcolormesh(x,y,tI,cmap=colormap,vmin=Imin,vmax=Imax,shading='auto')
    
    # Add labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Fix aspect ratio
    plt.axis('square')

    # Fix x-axis to run in sky
    if (in_radec) :
        plt.gca().invert_xaxis()

    if (return_intensity_map) :
        return h,plt.gca(),plt.gcf(),x,y,I
    else:
        return h,plt.gca(),plt.gcf()
    

