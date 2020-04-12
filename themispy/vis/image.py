###########################
#
# Package:
#   image
#
# Provides:
#   Provides model image classes and functions.
#

import numpy as np
import copy
from scipy import interpolate as sint
from scipy.signal import fftconvolve


rad2uas = 180.0/np.pi * 3600 * 1e6
uas2rad = np.pi/(180.0 * 3600 * 1e6)



## Reads in a list of alphanumeric model specifiers and expands them as
## appropriate.  This includes modifiers, models, and a number of repititions



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
    """
    
    def __init__(self, themis_fft_sign=True) :
        self.parameters=None
        self.size=None
        self.themis_fft_sign=themis_fft_sign
        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_image::generate_model`, 
        a similar a similar function within the :cpp:class:`Themis::model_image` class.  Effectively 
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
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """

        # 
        if (self.themis_fft_sign) :
            x = -x
            y = -y

        if (not parameters is None) :
            self.generate(parameters)

        I = self.generate_intensity_map(x,y,verbosity=0)

        if (self.themis_fft_sign) :
            x = -x
            y = -y
        
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
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """
        
        raise NotImplementedError("intensity_map must be defined in children classes of model_image.")

    
    
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
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """
        
        s = self.parameters[1] * rad2uas
        I0 = self.parameters[0] / (2.*np.pi*s**2)

        I = I0 * np.exp( - 0.5*( x**2 + y**2 )/s**2 )
        
        if (verbosity>0) :
            print("Symmetric Gaussian:",I0,s)
            
        return I

    
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
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """
        
        s = self.parameters[1] * rad2uas
        I0 = self.parameters[0] / (2.*np.pi*s**2)

        I = I0 * np.exp( - 0.5*( x**2 + y**2 )/s**2 )
        
        if (verbosity>0) :
            print("Symmetric Gaussian:",I0,s)


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




class model_image_crescent(model_image) :
    """
    Symmetric gaussian image class that is a mirror of :cpp:class:`Themis::model_image_crescent`.
    Has parameters:

    * parameters[0] ... Total intensity :math:`I_0` (Jy)
    * parameters[1] ... Outer radius :math:`R` (rad)
    * parameters[2] ... Width paramter :math:`\\psi` in (0,1)
    * parameters[3] ... Eccentricity parameter :math:`\\epsilon` in (0,1)
    * parameters[3] ... Asymmetry parmaeter math:`\\tau` in (0,1)
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
          (numpy.ndarray) Array of intensity values at positions (x,y).
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
          (numpy.ndarray) Array of intensity values at positions (x,y).
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
        self.size=6

        
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """

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

    norm_factor = ((x[1]-x[0])*(y[1]-y[0]))/((xx[1]-xx[0])*(yy[1]-yy[0]))

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
        self.size=6

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
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """

        f = np.array(self.parameters).reshape([self.Nx,self.Ny])

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

        
        if (self.spline_method=='fft') :
            I = fft_cubic_spline_2d(xtmp,ytmp,f,xx,yy,a=self.a)
        elif (self.spline_method=='direct') :
            I = direct_cubic_spline_2d(xtmp,ytmp,f,xx,yy,a=self.a)
        else :
            raise NotImplementedError("Only 'fft' and 'direct' methods are implemented for the spline.")


        if (transpose) :
            I = np.transpose(I)
        
        return I

    
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
        self.size=6

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
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """

        f = np.array(self.parameters[:-3]).reshape([self.Nx,self.Ny])

        fovx = self.parameters[-3]*rad2uas
        fovy = self.parameters[-2]*rad2uas
        PA = self.parameters[-1]

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

        
        if (self.spline_method=='fft') :
            I = fft_cubic_spline_2d(xtmp,ytmp,f,xx,yy,PA,a=self.a)
        elif (self.spline_method=='direct') :
            I = direct_cubic_spline_2d(xtmp,ytmp,f,xx,yy,PA,a=self.a)
        else :
            raise NotImplementedError("Only 'fft' and 'direct' methods are implemented for the spline.")


        if (transpose) :
            I = np.transpose(I)
        
        return I



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
        a similar a similar function within the :cpp:class:`Themis::model_image_smooth` class.  Effectively 
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
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """

        # Doesn't to boundary checking here.  Probably not a major problem, but consider it in the future.
        sr1 = self.parameters[0] * rad2uas * np.sqrt( 1./(1.-self.parameters[1]) )
        sr2 = self.parameters[0] * rad2uas * np.sqrt( 1./(1.+self.parameters[1]) )
        sphi = self.parameters[2]

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
    
        px = x[1,1]-x[0,0]
        py = y[1,1]-y[0,0]

        I = self.image.intensity_map(x,y,parameters=self.parameters,verbosity=verbosity)
        Ism = np.abs(fftconvolve(I*px*py,K*px*py, mode='same'))

        if (verbosity>0) :
            print('Gaussian smoothed:',sr1,sr2,sphi)
    
        return Ism/(px*py)

    
    
    
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
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.

    Attributes:
      image_list (list): List of :class:`model_image` objects that are to be summed.
      shift_list (list): List of shifts transformed to Cartesian coordinates.
      offset_coordinates (str): Coordinates in which to specify component shifts.
    """

    def __init__(self,image_list=None, offset_coordinates='Cartesian', themis_fft_sign=True) :
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

        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_image_sum::generate_model`, 
        a similar a similar function within the :cpp:class:`Themis::model_image_sum` class.  
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
            image.generate(q)

            if (self.offset_coordinates=='Cartesian') :
                self.shift_list[k][0] = q[image.size] * rad2uas
                self.shift_list[k][1] = q[image.size+1] * rad2uas
            elif (self.offset_coordinates=='polar') :
                self.shift_list[k][0] = q[image.size]*np.cos(q[image.size+1]) * rad2uas
                self.shift_list[k][1] = q[image.size]*np.sin(q[image.size+1]) * rad2uas
                
            q = q[(image.size+2):]

            
    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """

        I = np.zeros(x.shape)
        q = self.parameters
        for k,image in enumerate(self.image_list) :
            dx = x-self.x0_list[k]
            dy = y-self.y0_list[k]

            I = I+image.generate_intensity_map(dx,dy,verbosity=verbosity)

            q=self.parameters[(image.size+2):]

            if (verbosity>0) :
                print("   --> Sum shift:",self.shift_list[k],self.shift_list[k])
            
        return I
    



    
def expand_model_glob(model_glob, version='ccm_mexico') :
    """
    Reads in a model glob in the format
    """

    if (version=='ccm_mexico') :
        return expand_model_image_ccm_mexico(model_glob)
    else :
        raise RuntimeError("Unrecognized model_glob version: %s"%(version))

    
def expand_model_image_ccm_mexico(model_glob) :
    """
    Reads a model glob as produced by the ccm_mexico driver and generates a corresponding :class:`model_image` object.

    Args:
      model_glob(str): A model glob string constructed of a,A,g,G,X, following the style and details of the model glob in the m87_ccm_mexico drivers.

    Returns:
      (model_image): A corresponding model image object.
    """

    expanded_model_list = []

    for m in model_glob :

        ms = re.split('(\d+)', m)

        if (len(ms)==1) :
            n = 1
        else :
            n = int(ms[1])

        for k in range(n) :
            expanded_model_list.append(ms[0])

    return expanded_model_list


