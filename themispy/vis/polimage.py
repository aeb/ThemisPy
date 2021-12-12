###########################
#
# Package:
#   polimage
#
# Provides:
#   Provides model image classes appropriate for polarized images.
#

import themispy
from themispy.utils import *
from themispy.vis.image import *
from themispy.vis.typlot import *

import numpy as np
import copy
import re
from scipy import interpolate as sint
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import is_color_like, to_rgba


# Some constants
rad2uas = 180.0/np.pi * 3600 * 1e6
uas2rad = np.pi/(180.0 * 3600 * 1e6)


class model_polarized_image(model_image) :
    """
    Base class for Themis polarized image models.
    This base class is intended to mirror the :cpp:class:`Themis::model_polarized_image` base class,
    providing explicit visualizations of the output of :cpp:class:`Themis::model_polarized_image`-derived
    model analyses.

    Args:
      station_names (list): List of strings corresponding to station names for which D terms are being reconstructed. Default: None.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.

    Attributes:
      parameters (list): Parameter list for a given model. Default: None.
      size (int): Number of parameters for a given model. Default: None.
      number_of_dterms (int): Number of D terms. If D terms are not included, set to 0. Default: 0.
      dterm_dict (dictionary): Dictionary of D terms with keys 'station_names' and each station name.  Default: {}.
      themis_fft_sign (bool): Themis FFT sign convention flag.
      time (float): Time at which image is to be generated.
    """

    
    def __init__(self, station_names=None, themis_fft_sign=True) :
        super().__init__(themis_fft_sign)
        self.parameters=None
        self.size=0
        self.image_size=0
        self.themis_fft_sign=themis_fft_sign
        self.time=0


        self.dterm_dict={}
        self.dterm_dict['station_names'] = []
        self.number_of_dterms=0
        if (not (station_names is None)) :
            self.dterm_dict['station_names'] = station_names
            for station in station_names :
                self.dterm_dict[station]=[1.0+0.0j,1.0+0.0j]
            self.number_of_dterms=4*len(station_names)

        self.size = self.image_size+self.number_of_dterms
        
            
    def set_dterms(self,dterm_parameters) :
        """
        Sets the D terms from a list of parameters as specified directly from a
        :cpp:class:`model_polarized_image` object.

        Args:
          dterm_parameters (list): List of float values that is twice as long as the number of stations names.
        """

        if (len(dterm_parameters)!=self.number_of_dterms) :
            raise RuntimeError("Expected %i d-term parameters but %i were passed."%(self.number_of_dterms,len(dterm_parameters)))

        for k,station in enumerate(self.dterm_dict['station_names']) :
            self.dterm_dict[station] = [ dterm_parameters[4*k] + dterm_parameters[4*k+1]*1.0j, dterm_parameters[4*k+2] + dterm_parameters[4*k+3]*1.0j ]


    def dterms(self) :
        """
        Provides access to D terms and station names.

        Returns:
          (dictionary): Station D terms organized as a dictionary indexed by the station codes in :class:`ehtim.obsdata.tarr`.
        """

        return self.dterm_dict
    
            
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_image::generate_model`, 
        a similar function within the :cpp:class:`Themis::model_image` class.  Effectively 
        simply copies the parameters into the local list with some additional run-time checking.

        Args:
          parameters (list): Parameter list.
        """
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")
        self.set_dterms(parameters[self.image_size:])
        
        
    def stokes_map(self,x,y,parameters=None,kind='I',verbosity=0) :
        """
        External access function to the intenesity map.  Child classes should not overwrite 
        this function.  This implements some parameter assessment and checks the FFT sign
        convention.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          parameters (list): Parameter list. If none is provided, uses the current set of parameters. Default: None.
          kind (str): Stokes parameter to generate map for. Accepted values are 'I', 'Q', 'U', 'V', and combinations or 'all'. Default: 'I'.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (list) List of arrays of desired Stokes parameter values at positions (x,y) in :math:`Jy/\\mu as^2`. Always returned in the order IQUV.
        """

        # 
        if (self.themis_fft_sign==True) :
            x = -x
            y = -y
            if (verbosity>0) :
                print("Reflecting through origin to correct themis FFT sign convention")

        if (not parameters is None) :
            self.generate(parameters)

        S = self.generate_stokes_map(x,y,kind=kind,verbosity=verbosity)

        if (self.themis_fft_sign==True) :
            x = -x
            y = -y
            if (verbosity>0) :
                print("Unreflecting through origin to correct themis FFT sign convention")
        
        return S

    
    def generate_stokes_map(self,x,y,kind='all',verbosity=0) :
        """
        Hook for the intensity map function in child classes.  Assumes that the parameters
        list has been set.  This should never be called externally.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          kind (str): Stokes parameter to generate map for. Accepted values are 'I', 'Q', 'U', 'V', and combinations or 'all'. Default: 'all'.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (list) List of arrays of desired Stokes parameter values at positions (x,y) in :math:`Jy/\\mu as^2`. Always returned in the order IQUV.
        """
        
        raise NotImplementedError("generate_stokes_map must be defined in children classes of model_polarized_image.")

    def generate_intensity_map(self,x,y,verbosity=0) :
        """
        Specific hook for generating Stokes I map in child classes.  Assumes that the parameters
        list has been set.  This should never be called externally.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of desired Stokes I brightness values at positions (x,y) in :math:`Jy/\\mu as^2`.
        """

        return self.generate_stokes_map(x,y,kind='I',verbosity=verbosity)[0]

        
    def parameter_name_list(self) :
        """
        Hook for returning a list of parameter names.  Currently just returns the list of p0-p(size-1).

        Returns:
          (list) List of strings of variable names.
        """

        return [ 'p%i'%(j) for j in range(self.image_size) ] + self.dterm_dict['station_names']
    
    
    def set_time(self,time) :
        """
        Sets time.

        Args:
          time (float): Time in hours.
        """
        
        self.time = time



class model_polarized_image_constant_polarization(model_polarized_image) :
    """
    Polarized image class built on top of a model_image object.  Mirrors 
    :cpp:class:`Themis::model_polarized_image_constant_polarization`.

    * parameters[0] .............. First parameter of the Stokes I image.

    ...

    * parameters[image.size-1] ... Last parameter of the image.
    * parameters[image.size] ..... Polarization fraction :math:`\\sqrt{Q^2+U^2+V^2}/I`
    * parameters[image.size+1] ... Electric field polarization position angle :math:`(1/2) \\arctan(U/Q)` (rad).
    * parameters[image.size+2] ... Circular polarization fraction :math:`V/I`.

    and size=image.size+2.

    Args:
      image (model_image): Model image for the underlying intensity map.
      station_names (list): List of strings corresponding to station names for which D terms are being reconstructed. Default: None.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    """

    def __init__(self, image, station_names=None, themis_fft_sign=True) :
        super().__init__(station_names,themis_fft_sign)
        self.image_size = image.size+3
        self.image = image
        self.size = self.image_size + self.number_of_dterms
    
        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_polarized_image_constant_polarization::generate_model`, 
        a similar function within the :cpp:class:`Themis::model_polarized_image_constant_polarization` class.  Effectively 
        simply copies the parameters into the local list with some additional run-time checking.

        Args:
          parameters (list): Parameter list.
        """
        
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")

        self.image.generate(parameters[:self.image.size])
        self.set_dterms(self.parameters[self.image.size+3:])

        
    def generate_stokes_map(self,x,y,kind='all',verbosity=0) :
        """
        Internal generation of the Stokes map. In practice you almost certainly want to call :func:`model_polarized_image.stokes_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          kind (str): Stokes parameter to generate map for. Accepted values are 'I', 'Q', 'U', 'V', and combinations or 'all'. Default: 'all'.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (list) List of arrays of desired Stokes parameter values at positions (x,y) in :math:`Jy/\\mu as^2`. Always returned in the order IQUV.
        """

        S = []
        
        I = self.image.generate_intensity_map(x,y,verbosity=verbosity)

        if (('I' in kind) or (kind=='all')) :
            S.append(I)
        
        p = self.parameters[self.image.size]
        evpa = self.parameters[self.image.size+1]
        muV = self.parameters[self.image.size+2]

        if (('Q' in kind) or (kind=='all')) :
            Q = p*I*np.sqrt(1.0-muV**2)*np.cos(2.0*evpa)
            S.append(Q)
        if (('U' in kind) or (kind=='all')) :
            U = p*I*np.sqrt(1.0-muV**2)*np.sin(2.0*evpa)
            S.append(U)
        if (('V' in kind) or (kind=='all')) :
            V = p*I*muV
            S.append(V)

        return S

    
    def parameter_name_list(self) :
        """
        Hook for returning a list of parameter names.  Currently just returns the list of p0-p(size-1).

        Returns:
          (list) List of strings of variable names.
        """

        return self.image.parameter_name_list + ['m', 'EVPA (rad)', r'$\mu_V$'] + self.dterm_dict['station_names']



class model_polarized_image_adaptive_splined_raster(model_polarized_image) :
    """
    Adaptive splined raster image class that is a mirror of :cpp:class:`Themis::model_polarized_image_adaptive_splined_raster`.
    Has parameters:

    * parameters[0] ........... Logarithm of the Stokes I brightness at control point 0,0 (Jy/sr)
    * parameters[1] ........... Logarithm of the Stokes I brightness at control point 1,0 (Jy/sr)

    ...

    * parameters[Nx*Ny-1] ..... Logarithm of the Stokes I brightness at control point Nx-1,Ny-1 (Jy/sr)
    * parameters[Nx*Ny] ....... Logarithm of the polarization fraction (:math:`\\sqrt{Q^2+U^2+V^2}/I`) at control point 0,0

    ...

    * parameters[2*Nx*Ny-1] ... Logarithm of the polarization fraction (:math:`\\sqrt{Q^2+U^2+V^2}/I`) at control point Nx-1,Ny-1
    * parameters[2*Nx*Ny] ..... Electric field polarization angle of the linear polarization (:math:`(1/2) \\arctan(U/Q)` in rad) at control point 0,0
 
    ...

    * parameters[3*Nx*Ny-1] ... Electric field polarization angle of the linear polarization (:math:`(1/2) \\arctan(U/Q)` in rad) at control point Nx-1,Ny-1
    * parameters[2*Nx*Ny] ..... Circular polarization fraction (:math:`V/I`) at control point 0,0
 
    ...

    * parameters[4*Nx*Ny-1] ... Circular polarization fraction (:math:`V/I`) at control point Nx-1,Ny-1
    * parameters[4*Nx*Ny] .... Field of view in the x direction (rad)
    * parameters[4*Nx*Ny+1] .. Field of view in the y direction (rad)
    * parameters[4*Nx*Ny+2] .. Orientation of the x direction (rad)

    and size=4*Nx*Ny+3.

    Args:
      Nx (int): Number of control points in the -RA direction.
      Ny (int): Number of control points in the Dec direction.
      a (float): Spline control factor. Default: -0.5.
      spline_method (str): Spline method. Accepted values are 'fft' and 'direct'.  Default: 'fft'.
      station_names (list): List of strings corresponding to station names for which D terms are being reconstructed. Default: None.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True

    Attributes:
      Nx (int): Number of control points in the -RA direction.
      Ny (int): Number of control points in the Dec direction.
      a (float): Spline control factor. Default: -0.5.
      spline_method (str): Spline method. Accepted values are 'fft' and 'direct'.  Default: 'fft'.
    """

    def __init__(self, Nx, Ny, a=-0.5, spline_method='fft', station_names=None, themis_fft_sign=True) :
        super().__init__(station_names,themis_fft_sign)
        self.image_size = 4*Nx*Ny+3
        self.Nx = Nx
        self.Ny = Ny
        self.a = a
        self.spline_method = spline_method
        self.size = self.image_size+self.number_of_dterms
        
        
    def generate_stokes_map(self,x,y,kind='all',verbosity=0) :
        """
        Internal generation of the Stokes map. In practice you almost certainly want to call :func:`model_polarized_image.stokes_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          kind (str): Stokes parameter to generate map for. Accepted values are 'I', 'Q', 'U', 'V', and combinations or 'all'. Default: 'all'.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (list) List of arrays of desired Stokes parameter values at positions (x,y) in :math:`Jy/\\mu as^2`. Always returned in the order IQUV.
        """

        S = []

        Ndim = self.Nx*self.Ny
        fI = np.flipud(np.transpose(np.exp(np.array(self.parameters[:Ndim]).reshape([self.Ny,self.Nx])))) * uas2rad**2
        p = np.flipud(np.transpose(np.exp(np.array(self.parameters[Ndim:2*Ndim]).reshape([self.Ny,self.Nx]))))
        evpa = np.flipud(np.transpose(np.array(self.parameters[2*Ndim:3*Ndim]).reshape([self.Ny,self.Nx])))
        muV = np.flipud(np.transpose(np.array(self.parameters[3*Ndim:4*Ndim]).reshape([self.Ny,self.Nx])))
        
        fovx = self.parameters[4*Ndim]*rad2uas
        fovy = self.parameters[4*Ndim+1]*rad2uas
        PA = -self.parameters[4*Ndim+2]

        fQ = p*fI*np.sqrt(1.0-muV**2)*np.cos(2.0*evpa)
        fU = p*fI*np.sqrt(1.0-muV**2)*np.sin(2.0*evpa)
        fV = p*fI*muV
        
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
            Q = 0*x
            U = 0*x
            V = 0*x
        elif (self.spline_method=='fft') :
            I = fft_cubic_spline_2d(xtmp,ytmp,fI,xx,yy,PA,a=self.a)
            Q = fft_cubic_spline_2d(xtmp,ytmp,fQ,xx,yy,PA,a=self.a)
            U = fft_cubic_spline_2d(xtmp,ytmp,fU,xx,yy,PA,a=self.a)
            V = fft_cubic_spline_2d(xtmp,ytmp,fV,xx,yy,PA,a=self.a)
        elif (self.spline_method=='direct') :
            I = direct_cubic_spline_2d(xtmp,ytmp,fI,xx,yy,PA,a=self.a)
            Q = direct_cubic_spline_2d(xtmp,ytmp,fQ,xx,yy,PA,a=self.a)
            U = direct_cubic_spline_2d(xtmp,ytmp,fU,xx,yy,PA,a=self.a)
            V = direct_cubic_spline_2d(xtmp,ytmp,fV,xx,yy,PA,a=self.a)
        else :
            raise NotImplementedError("Only 'fft' and 'direct' methods are implemented for the spline.")

        if (transpose) :
            I = np.transpose(I)
            Q = np.transpose(Q)
            U = np.transpose(U)
            V = np.transpose(V)

        if (('I' in kind) or (kind=='all')) :
            S.append(I)
        if (('Q' in kind) or (kind=='all')) :
            S.append(Q)
        if (('U' in kind) or (kind=='all')) :
            S.append(U)
        if (('V' in kind) or (kind=='all')) :
            S.append(V)

        return S

    
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
        for ix in range(self.Nx) :
            for iy in range(self.Ny) :
                names.append(r'$\ln(p_{%i,%i})$'%(ix,iy))
        for ix in range(self.Nx) :
            for iy in range(self.Ny) :
                names.append(r'$\ln({\rm EVPA}_{%i,%i} (rad))$'%(ix,iy))
        for ix in range(self.Nx) :
            for iy in range(self.Ny) :
                names.append(r'$\ln(\\mu_{V,%i,%i})$'%(ix,iy))
        names.append('fovx (rad)')
        names.append('fovy (rad)')
        names.append(r'$\phi$ (rad)')
        return names


    

class model_polarized_image_sum(model_polarized_image) :
    """
    Summed image class that is a mirror of :cpp:class:`Themis::model_polarized_image_sum`. Generates images by summing 
    other images supplemented with some offset. The parameters list is expanded by 2 for each 
    image, corresponding to the absolute positions. These may be specified either in Cartesian 
    or polar coordinates; this selection must be made at construction and applies uniformly to 
    all components (i.e., all component positions must be specified in the same coordinates).
    
    Args:
      polarized_image_list (list): List of :class:`model_image` objects. If None, must be set prior to calling :func:`stokes_map`. Default: None.
      offset_coordinates (str): Coordinates in which to specify component shifts.  Options are 'Cartesian' and 'polar'. Default: 'Cartesian'.
      reference_component (int): Index of component to define as reference. If None, will not set any component to be the reference. Default: None.
      station_names (list): List of strings corresponding to station names for which D terms are being reconstructed. Default: None.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.
    
    Attributes:
      image_list (list): List of :class:`model_image` objects that are to be summed.
      shift_list (list): List of shifts transformed to Cartesian coordinates.
      offset_coordinates (str): Coordinates in which to specify component shifts.
    """

    def __init__(self,image_list=None, offset_coordinates='Cartesian', reference_component=None, station_names=None, themis_fft_sign=True) :
        super().__init__(station_names,themis_fft_sign)

        self.image_list=[]
        self.shift_list=[]
        
        if (not image_list is None) :
            self.image_list = copy.copy(image_list) # This is a SHALLOW copy

        self.image_size=0
        for image in image_list :
            self.image_size += image.size+2
            self.shift_list.append([0.0, 0.0])

        self.offset_coordinates = offset_coordinates
        self.reference_component = reference_component

        self.size = self.image_size+self.number_of_dterms

        
            
    def add(self,image) :
        """
        Adds an model_image object or list of model_image objects to the sum.  The size is recomputed.

        Args:
          image (model_polarized_image, list): A single model_image object or list of model_polarized_image objects to be added.

        Returns:
          None
        """
        
        # Can add lists and single istances
        if (isinstance(image,list)) :
            self.image_list = image_list + image
        else :
            self.image_list.append(image)
        # Get new size and shift list
        self.image_size=0
        self.shift_list = []
        for image in image_list :
            self.image_size += image.size+2
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
        if (len(self.parameters)<self.image_size) :
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

        self.set_dterms(q)
            
        if (not (self.reference_component is None)) :
            if (self.reference_component>len(self.image_list)) :
                raise RuntimeError("Reference component %i is not in the list of images, which numbers only %i."%(self.reference_component,len(self.image_list)))
            x0 = self.shift_list[self.reference_component][0]
            y0 = self.shift_list[self.reference_component][1]
            for k in range(len(self.image_list)) :
                self.shift_list[k][0] = self.shift_list[k][0]-x0
                self.shift_list[k][1] = self.shift_list[k][1]-y0

            
    def generate_stokes_map(self,x,y,kind='all',verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          kind (str): Stokes parameter to generate map for. Accepted values are 'I', 'Q', 'U', 'V', and combinations or 'all'. Default: 'all'.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (list) List of arrays of desired Stokes parameter values at positions (x,y) in :math:`Jy/\\mu as^2`. Always returned in the order IQUV.
        """

        S = []

        if (('I' in kind) or (kind=='all')) :
            S.append(np.zeros(x.shape))
        if (('Q' in kind) or (kind=='all')) :
            S.append(np.zeros(x.shape))
        if (('U' in kind) or (kind=='all')) :
            S.append(np.zeros(x.shape))
        if (('V' in kind) or (kind=='all')) :
            S.append(np.zeros(x.shape))
        
        q = self.parameters
        for k,image in enumerate(self.image_list) :
            dx = x-self.shift_list[k][0]
            dy = y-self.shift_list[k][1]

            Stmp = image.generate_stokes_map(dx,dy,kind=kind,verbosity=verbosity)

            for j in range(len(S)) :
                S[j] = S[j] + Stmp[j]

            q=self.parameters[(image.size+2):]

            if (verbosity>0) :
                print("   --> Sum shift:",self.shift_list[k])
            
        return S


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
        names = names + self.dterm_dict['station_names']
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
    


class model_polarized_image_smooth(model_polarized_image):
    """
    Smoothed polarized image class that is a polarized mirror of :cpp:class:`Themis::model_polarized_image_smooth`. Generates a smoothed 
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
      image (model_polarized_image): :class:`model_polarized_image` object that will be smoothed.
      station_names (list): List of strings corresponding to station names for which D terms are being reconstructed. Default: None.
      themis_fft_sign (bool): If True will assume the Themis-default FFT sign convention, which reflects the reconstructed image through the origin. Default: True.

    Attributes:
      image (model_polarized_image): A preexisting :class:`model_polarized_image` object to be smoothed.
    """

    def __init__(self, image, station_names=None, themis_fft_sign=True) :
        super().__init__(station_names,themis_fft_sign)

        self.image = image
        self.image_size = image.size + 3
        self.size = self.size + 3

        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_polarized_image_smooth::generate_model`, 
        a similar function within the :cpp:class:`Themis::model_polarized_image_smooth` class.  Effectively 
        simply copies the parameters into the local list with some additional run-time checking.

        Args:
          parameters (list): Parameter list.

        Returns:
          None        
        """
        
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")
        self.set_dterms(parameters[self.image_size:])
        self.image.generate(parameters[:self.image_size-3])

        
    def generate_stokes_map(self,x,y,kind='all',verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          kind (str): Stokes parameter to generate map for. Accepted values are 'I', 'Q', 'U', 'V', and combinations or 'all'. Default: 'all'.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (list) List of arrays of desired Stokes parameter values at positions (x,y) in :math:`Jy/\\mu as^2`. Always returned in the order IQUV.
        """

        # Doesn't to boundary checking here.  Probably not a major problem, but consider it in the future.
        sr1 = self.parameters[self.image_size-3] * rad2uas * np.sqrt( 1./(1.-self.parameters[self.image_size-2]) )
        sr2 = self.parameters[self.image_size-3] * rad2uas * np.sqrt( 1./(1.+self.parameters[self.image_size-2]) )
        sphi = self.parameters[self.image_size-1]

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

        self.image.generate(self.parameters[:self.image_size-3])
        S = self.image.generate_stokes_map(x,y,kind=kind,verbosity=verbosity)
        Ssm = []
        for s in S :
            Ssm.append( fftconvolve(s*px*py,K*px*py, mode='same')/(px*py) )

        if (verbosity>0) :
            print('Gaussian smoothed:',sr1,sr2,sphi,self.parameters)
            
        return Ssm

    
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



class model_polarized_image_Faraday_derotated(model_polarized_image):
    """
    Faraday derotated image class.  Generates a zero-wavelength image after derotated.  The Faraday rotation measure 
    (in :math:`rad m^{-2}`), and observation wavelength (in m) may be provided at construction or modified afterward.

    Image parameters are the same as those of the passed :class:`model_polarized_image`.

    Args:
      image (model_polarized_image): :class:`model_polarized_image` object that will be smoothed.
      RM (float): Faraday rotation measure of intervening screen in :math:`rad m^{-2}`. Default: 0.0.
      wavelength (float): Observation wavelength in m. Default: 1.3 mm.

    Attributes:
      image (model_polarized_image): A preexisting :class:`model_polarized_image` object to be smoothed.
      RM (float): Faraday rotation measure of intervening screen in :math:`rad m^{-2}`. Default: 0.0.
      wavelength (float): Observation wavelength in m. Default: 1.3 mm.
    """

    def __init__(self, image, RM=0.0, wavelength=1.3e-3, station_names=None, themis_fft_sign=True) :
        super().__init__(station_names,themis_fft_sign)

        self.image = image
        self.RM = RM
        self.wavelength = wavelength
        self.image_size = image.image_size
        self.size = image.size


    def set_RM(self, RM) :
        """
        Sets the Faraday rotation measure in :math:`rad m^{-2}`.

        Args:
          RM (float): Faraday rotation measure in :math:`rad m^{-2}`.

        Returns:
          None
        """
        
        self.RM = RM

        
    def set_wavelength(self, wavelength) :
        """
        Sets the obervation wavlength in m.

        Args:
          wavelength (float): Observation wavelength in m.

        Returns:
          None
        """
        
        self.wavelength = wavelength
        
        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors :cpp:func:`Themis::model_polarized_image::generate_model`, 
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
        self.set_dterms(parameters[self.image_size:])
        self.image.generate(parameters[:self.image_size])

        
    def generate_stokes_map(self,x,y,kind='all',verbosity=0) :
        """
        Internal generation of the intensity map. In practice you almost certainly want to call :func:`model_image.intensity_map`.

        Args:
          x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
          kind (str): Stokes parameter to generate map for. Accepted values are 'I', 'Q', 'U', 'V', and combinations or 'all'. Default: 'all'.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (list) List of arrays of desired Stokes parameter values at positions (x,y) in :math:`Jy/\\mu as^2`. Always returned in the order IQUV.
        """

        self.image.generate(self.parameters[:self.image_size])

        if (kind in ['Q','U']) :
            kind = 'all'
        S = self.image.generate_stokes_map(x,y,kind=kind,verbosity=verbosity)

        # Rotate by Faraday rotation measure
        Q = np.copy(S[1])
        U = np.copy(S[2])
        Psi = self.RM*self.wavelength**2
        S[1] = np.cos(2.0*Psi)*Q + np.sin(2.0*Psi)*U
        S[2] = -np.sin(2.0*Psi)*Q + np.cos(2.0*Psi)*U
        
        if (verbosity>0) :
            print('Faraday derotated:',self.RM,self.wavelength,Psi)
            
        return S

    
    def parameter_name_list(self) :
        """
        Producess a lists parameter names.

        Returns:
          (list) List of strings of variable names.
        """

        names = self.image.parameter_name_list()
        return names

    
    def set_time(self,time) :
        """
        Sets time.

        Args:
          time (float): Time in hours.
        """
        
        self.time = time
        self.image.set_time(time)

        

            

def _get_polarization_ellipse(S) :
    """
    Returns a properly oriented polarization ellipse given the Stokes parameters.

    Args:
      S (list): List of [I,Q,U,V].

    Returns:
      (numpy.ndarray,numpy.ndarray): Arrays of x and y normalized such that their major axis length is the polarization fraction and ellipticity set by V/L.
    """
    
    
    L = np.sqrt(S[1]**2+S[2]**2+S[3]**2)
    p = L/S[0]
    muV = S[2]/L
    EVPA = 0.5*np.arctan2(S[2],S[1])

    epsilon = (1.0 - np.sqrt(1-muV**2))/muV
    norm = 1.0/np.sqrt(1.0+epsilon**2)

    
    t = np.linspace(0,2.0*np.pi,128)
    x0 = 0.5*p*np.cos(t) * epsilon * norm
    y0 = 0.5*p*np.sin(t) * norm
    x = np.cos(EVPA)*x0 + np.sin(EVPA)*y0
    y = -np.sin(EVPA)*x0 + np.cos(EVPA)*y0

    return x,y
    
            

def plot_linear_polarization_map(polarized_image, parameters, limits=None, shape=None, tick_shape=None, colormap='plasma', tick_color='w', Lmin=0.0, Lmax=None, mscale=None, in_radec=True, xlabel=None, ylabel=None, return_stokes_map=False, transfer_function='linear', intensity_contours=False, intensity_colors=None, tick_fractional_intensity_floor=0.1, colorbar=None, verbosity=0) :
    """
    Plots linear polarization fraction and polarization ticks associated with a given 
    :class:`model_image` object evaluated at a specified set of parameters. A number of additional 
    practical options may be passed. Access to a matplotlib pcolor object, the figure handle and axes 
    handles are returned. Optionally, the Stokes map is returned.

    Args:
      polarized_image (model_polarized_image): :class:`model_polarized_image` object to plot.
      parameters (list): List of parameters that define model_image intensity map.
      limits (float,list): Limits in uas on image size. If a single float is given, the limits are uniformly set. If a list is given, it must be in the form [xmin,xmax,ymin,ymax]. Default: 75.
      shape (int,list): Number of pixels. If a single int is given, the number of pixels are uniformly set in both directions.  If a list is given, it must be in the form [Nx,Ny]. Default: 256.
      tick_shape (int,list): Number of ticks to plot. If a single int is given, the number of ticks are uniformly set in both directions.  If a list is given, it must be in the form [Mx,My]. Default: 16.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'afmhot'.
      tick_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`.
      Lmin (float): Minimum intensity for colormap. Default: 0 (though see transfer_function).
      Lmax (float): Maximum intensity for colormap. Default: maximum linearly polarized intensity.
      mscale (float): Polarization fraction scale for polarization ticks. If None, set by maximum polarization fraction where the polarized flux is above 1% of the maximum polarized flux. Default: None.
      in_radec (bool): If True, plots in :math:`\\Delta RA` and :math:`\\Delta Dec`.  If False, plots in sky-right Cartesian coordinates. Default: True.
      xlabel (str): Label for xaxis. Default: ':math:`\\Delta RA (\\mu as)` if in_radec is True, :math:`\\Delta x (\\mu as)` if in_radec is False.
      ylabel (str): Label for yaxis. Default: ':math:`\\Delta Dec (\\mu as)` if in_radec is True, :math:`\\Delta y (\\mu as)` if in_radec is False.
      return_stokes_map (bool): If True returns the intensity map in addition to standard return items.
      transfer_function (str): Transfer function to plot.  Options are 'linear','sqrt','log'.  If 'log', if Imin=0 it will be set to Imax/1000.0 by default, else if Imin<0 it will be assumed to be a dynamic range and Imin = Imax * 10**(Imin).
      intensity_contours (int,list,bool): May either be True, an integer, or a list of contour level values. Default: False. If True, sets 8 linearly spaced contour levels.
      intensity_colors (list): A list composed of any acceptable color type as specified in :mod:`matplotlib.colors`.
      tick_fractional_intensity_floor (float): Fraction of maximum intensity at which to stop plotting polarization ticks.  Default: 0.1.
      colorbar (str): Adds colorbar indicating magnitude of polarized flux. Options are None, 'flux', and 'frac'.  If None, no colorbar is plotted.  If 'flux' the polarized flux in Jy will be shown.  If 'frac' the fraction of the maximum intensity in polarized flux will be shown, i.e., :math:`L/I_{max}`.  Default: None.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

    Returns :
      (matplotlib.pyplot.image,matplotlib.pyplot.axes,matplotlib.pyplot.fig,[numpy.ndarray,numpy.ndarray,numpy.ndarray]): Image handle; Figure object; Axes object; Optionally intensity map (x,y,S).
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

    # Set polarization tick shape
    if (tick_shape is None) :
        tick_shape = [20,20]
    elif (isinstance(tick_shape,list)) :
        tick_shape = tick_shape
    else :
        tick_shape = [tick_shape, tick_shape]
    
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
    # Generate a set of Stokes parameters (Jy/uas^2)
    S = polarized_image.stokes_map(x,y,parameters,kind='all',verbosity=verbosity)
    
    # Flip x to flip x axis so that RA decreases right to left
    if (in_radec) :
        x = -x
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])
    
    L = np.sqrt(S[1]**2+S[2]**2)

    # Determine scale
    if (Lmax is None) :
        Lmax = np.max(L)
        
    if (verbosity>0) :
        print("Minimum/Maximum L: %g / %g"%(Lmin,Lmax))
    
    # Transfered I for plotting
    Lmin_orig = Lmin
    Lmax_orig = Lmax
    if (transfer_function=='linear') :
        tL = L/Lmax
        Lmin = Lmin/Lmax
        Lmax = 1.0
    elif (transfer_function=='sqrt') :
        tL = np.sqrt(L/Lmax)
        Lmin = np.sqrt(Lmin/Lmax)
        Lmax = 1.0
    elif (transfer_function=='log') :
        tL = np.log10(L/Lmax)
        if (Lmin==0) :
            Lmin = np.log10(1.0e-3)
            Lmin_orig = 1.0e-3
        elif (Lmin>0) :
            Lmin = np.log10(Lmin/Lmax)
        Lmax = 0.0

    # Set background color in Axes
    cmap = cm.get_cmap(colormap)    
    plt.gca().set_facecolor(cmap(0.0))

    # Make plot
    h = plt.pcolor(x,y,tL,cmap=colormap,vmin=Lmin,vmax=Lmax)

    # Add colorbar if desired
    if (not colorbar is None) :
        cb = plt.colorbar()
        if (colorbar=='flux') :
            cb.ax.set_ylabel('Polarized flux (mJy/$\mu$as$^2$)',rotation=270,va='bottom')
            cticks = cb.ax.get_yticks()
            i = np.argsort(tL.reshape([-1]))
            tL_sort = tL.reshape([-1])[i]
            L_sort = L.reshape([-1])[i]
            ctick_Lvals = np.interp(cticks,tL_sort,L_sort,left=Lmin_orig,right=Lmax_orig)
            ctlbls = [ '%5.2g'%(x*1e3) for x in ctick_Lvals ]
            cb.ax.set_yticklabels(ctlbls)
        elif (colorbar=='frac') :
            cb.ax.set_ylabel('Polarized flux ($I_{max}$)',rotation=270,va='bottom')
            #cylim = cb.ax.get_ylim()
            cticks = cb.ax.get_yticks()
            i = np.argsort(tL.reshape([-1]))
            tL_sort = tL.reshape([-1])[i]
            L_sort = L.reshape([-1])[i]
            ctick_Lvals = np.interp(cticks,tL_sort,L_sort,left=Lmin_orig,right=Lmax_orig)
            ctlbls = [ '%5.2g'%(x/np.max(S[0])) for x in ctick_Lvals ]
            cb.ax.set_yticklabels(ctlbls)
        else :
            raise RuntimeError("Unrecognized colorbar option %s"%(colorbar))

    
    # Plot tickmarks
    dxpol = (np.max(x)-np.min(x))/tick_shape[0]
    dypol = (np.max(x)-np.min(x))/tick_shape[1]
    xstride = max( 1, int(dxpol/(np.max(x)-np.min(x))*L.shape[0]+0.5) )
    ystride = max( 1, int(dypol/(np.max(y)-np.min(y))*L.shape[1]+0.5) )
    plot_polticks = (S[0]>tick_fractional_intensity_floor*np.max(S[0]))
    if (mscale is None) :
        m = np.sqrt(S[1]**2+S[2]**2)/S[0]
        mscale = min(1,np.max(m[plot_polticks]))
    pscale = np.sqrt(dxpol*dypol)/mscale
    for i in range(xstride//2,L.shape[0],xstride) :
        for j in range(ystride//2,L.shape[1],ystride) :
            if (plot_polticks[i,j]) :
                xs = x[i,j]
                ys = y[i,j]
                Qs = S[1][i,j]
                Us = S[2][i,j]
                ms = np.sqrt(S[1][i,j]**2+S[2][i,j]**2)/S[0][i,j]
                EVPA = 0.5*np.arctan2(Us,Qs)
                px = ms*np.sin(EVPA) * pscale
                py = ms*np.cos(EVPA) * pscale
                xtmp = [ (xs-0.5*px), (xs+0.5*px) ]
                ytmp = [ (ys-0.5*py), (ys+0.5*py) ]
                plt.plot(xtmp,ytmp,'-',color=tick_color,lw=1)
                

    # Add intensity contours if desired
    if (intensity_contours!=False) :
        if (intensity_contours==True) :
            intensity_contours=np.linspace(0,np.max(S[0]),10)[1:-1]
        elif (isinstance(intensity_contours,int)) :
            intensity_contours=np.linspace(0,np.max(S[0]),intensity_contours+2)[1:-1]
        elif (isinstance(intensity_contours,list)) :
            pass
        else :
            raise RuntimeError("intensity_countours must be a boolean, int, or list of intensity values at which to draw contours.")

        if (intensity_colors is None) :
            intensity_colors = [[0.75,0.75,0.75]]
        
        plt.contour(x,y,S[0],levels=intensity_contours,colors=intensity_colors,alpha=0.5)

    # Add labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Fix aspect ratio
    plt.axis('square')

    # Set limits
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])
    
    # Fix x-axis to run in sky
    if (in_radec) :
        plt.gca().invert_xaxis()        

    if (return_stokes_map) :
        return h,plt.gca(),plt.gcf(),x,y,S
    else:
        return h,plt.gca(),plt.gcf()


def plot_stokes_map(polarized_image, parameters, stokes='V', limits=None, shape=None, colormap='bwr', in_radec=True, xlabel=None, ylabel=None, return_stokes_map=False, intensity_contours=True, intensity_colors=None, stokes_fractional_intensity_floor=0.1, colorbar=None, verbosity=0) :
    """
    Plots Stokes maps with intensity contours associated with a given 
    :class:`model_image` object evaluated at a specified set of parameters. A number of additional 
    practical options may be passed. Access to a matplotlib pcolor object, the figure handle and axes 
    handles are returned. Optionally, the Stokes map is returned.

    Args:
      polarized_image (model_polarized_image): :class:`model_polarized_image` object to plot.
      parameters (list): List of parameters that define model_image intensity map.
      limits (float,list): Limits in uas on image size. If a single float is given, the limits are uniformly set. If a list is given, it must be in the form [xmin,xmax,ymin,ymax]. Default: 75.
      shape (int,list): Number of pixels. If a single int is given, the number of pixels are uniformly set in both directions.  If a list is given, it must be in the form [Nx,Ny]. Default: 256.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'afmhot'.
      in_radec (bool): If True, plots in :math:`\\Delta RA` and :math:`\\Delta Dec`.  If False, plots in sky-right Cartesian coordinates. Default: True.
      xlabel (str): Label for xaxis. Default: ':math:`\\Delta RA (\\mu as)` if in_radec is True, :math:`\\Delta x (\\mu as)` if in_radec is False.
      ylabel (str): Label for yaxis. Default: ':math:`\\Delta Dec (\\mu as)` if in_radec is True, :math:`\\Delta y (\\mu as)` if in_radec is False.
      return_stokes_map (bool): If True returns the intensity map in addition to standard return items.
      intensity_contours (int,list,bool): May either be True, an integer, or a list of contour level values. Default: True. If True, sets 8 linearly spaced contour levels.
      intensity_colors (list): A list composed of any acceptable color type as specified in :mod:`matplotlib.colors`.
      stokes_fractional_intensity_floor (float): Fraction of maximum intensity at which to stop plotting Stokes map.  Default: 0.1.
      colorbar (str): Adds colorbar indicating magnitude of polarized flux. Options are None, 'flux', and 'frac'.  If None, no colorbar is plotted.  If 'flux' the polarized flux in Jy will be shown.  If 'frac' the fraction of the maximum intensity in polarized flux will be shown, i.e., :math:`L/I_{max}`.  Default: None.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

    Returns :
      (matplotlib.pyplot.image,matplotlib.pyplot.axes,matplotlib.pyplot.fig,[numpy.ndarray,numpy.ndarray,numpy.ndarray]): Image handle; Figure object; Axes object; Optionally intensity map (x,y,S).
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
    # Generate a set of Stokes parameters (Jy/uas^2)
    S = polarized_image.stokes_map(x,y,parameters,kind='all',verbosity=verbosity)
    
    # Flip x to flip x axis so that RA decreases right to left
    if (in_radec) :
        x = -x
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])
    
    if (verbosity>0) :
        print("Minimum/Maximum L: %g / %g"%(Lmin,Lmax))

    # Select Stokes pararmeter for plotting
    I = S[0]
    if (stokes=='I') :
        L = S[0]
    elif (stokes=='Q') :
        L = S[1]
    elif (stokes=='U') :
        L = S[2]
    elif (stokes=='V') :
        L = S[3]

    # Apply the I cut
    tL = L
    for i in range(L.shape[0]) :
        for j in range(L.shape[1]) :
            if (I[i,j]<stokes_fractional_intensity_floor*np.max(I)) :
                tL[i,j] = np.nan
        
    # Set background color in Axes
    cmap = cm.get_cmap(colormap)    
    # plt.gca().set_facecolor(cmap(0.0))
    plt.gca().set_facecolor('w')

    # Make plot
    h = plt.pcolor(x,y,tL,cmap=colormap) #,vmin=Lmin,vmax=Lmax)

    # Add colorbar if desired
    if (not colorbar is None) :
        cb = plt.colorbar()
        if (colorbar=='flux') :
            cb.ax.set_ylabel('Stokes %s flux (mJy/$\mu$as$^2$)'%(stokes),rotation=270,va='bottom')
            cticks = cb.ax.get_yticks()
            i = np.argsort(tL.reshape([-1]))
            tL_sort = tL.reshape([-1])[i]
            L_sort = L.reshape([-1])[i]
            ctick_Lvals = np.interp(cticks,tL_sort,L_sort) #,left=Lmin_orig,right=Lmax_orig)
            ctlbls = [ '%5.2g'%(x*1e3) for x in ctick_Lvals ]
            cb.ax.set_yticklabels(ctlbls)
        elif (colorbar=='frac') :
            cb.ax.set_ylabel('Stokes %s flux ($I_{max}$)'%(stokes),rotation=270,va='bottom')
            #cylim = cb.ax.get_ylim()
            cticks = cb.ax.get_yticks()
            i = np.argsort(tL.reshape([-1]))
            tL_sort = tL.reshape([-1])[i]
            L_sort = L.reshape([-1])[i]
            ctick_Lvals = np.interp(cticks,tL_sort,L_sort) #,left=Lmin_orig,right=Lmax_orig)
            ctlbls = [ '%5.2g'%(x/np.max(S[0])) for x in ctick_Lvals ]
            cb.ax.set_yticklabels(ctlbls)
        else :
            raise RuntimeError("Unrecognized colorbar option %s"%(colorbar))

    
    # Add intensity contours if desired
    if (intensity_contours!=False) :
        if (intensity_contours==True) :
            intensity_contours=np.linspace(0,np.max(S[0]),10)[1:-1]
        elif (isinstance(intensity_contours,int)) :
            intensity_contours=np.linspace(0,np.max(S[0]),intensity_contours+2)[1:-1]
        elif (isinstance(intensity_contours,list)) :
            pass
        else :
            raise RuntimeError("intensity_countours must be a boolean, int, or list of intensity values at which to draw contours.")

        if (intensity_colors is None) :
            intensity_colors = [[0.75,0.75,0.75]]
        
        plt.contour(x,y,S[0],levels=intensity_contours,colors=intensity_colors,alpha=0.5)

    # Add labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Fix aspect ratio
    plt.axis('square')

    # Set limits
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])
    
    # Fix x-axis to run in sky
    if (in_radec) :
        plt.gca().invert_xaxis()        

    if (return_stokes_map) :
        return h,plt.gca(),plt.gcf(),x,y,S
    else:
        return h,plt.gca(),plt.gcf()

    

def plot_polarized_image(polarized_image, parameters, limits=None, shape=None, tick_shape=None, colormap='gray_r', tick_color='jet', Imin=0.0, Imax=None, Lmin=0.0, Lmax=None, mscale=None, in_radec=True, xlabel=None, ylabel=None, return_stokes_map=False, transfer_function='linear', tick_fractional_intensity_floor=0.1, colorbar=False, elliptical=False, verbosity=0) :
    """
    Plots linear polarization fraction, flux and intensity image associated with a given 
    :class:`model_image` object evaluated at a specified set of parameters. A number of additional 
    practical options may be passed. Access to a matplotlib pcolor object, the figure handle and axes 
    handles are returned. Optionally, the Stokes map is returned.

    Args:
      polarized_image (model_polarized_image): :class:`model_polarized_image` object to plot.
      parameters (list): List of parameters that define model_image intensity map.
      limits (float,list): Limits in uas on image size. If a single float is given, the limits are uniformly set. If a list is given, it must be in the form [xmin,xmax,ymin,ymax]. Default: 75.
      shape (int,list): Number of pixels. If a single int is given, the number of pixels are uniformly set in both directions.  If a list is given, it must be in the form [Nx,Ny]. Default: 256.
      tick_shape (int,list): Number of ticks to plot. If a single int is given, the number of ticks are uniformly set in both directions.  If a list is given, it must be in the form [Mx,My]. Default: 16.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'afmhot'.
      tick_color (matplotlib.colors.Colormap,str,list): Any acceptable colormap or color type as specified in :mod:`matplotlib.colors`.
      Imin (float): Minimum intensity for colormap. Default: 0 (though see transfer_function).
      Imax (float): Maximum intensity for colormap. Default: maximum linearly polarized intensity.
      Lmin (float): Minimum intensity for colormap. Default: 0 (though see transfer_function).
      Lmax (float): Maximum intensity for colormap. Default: maximum linearly polarized intensity.
      mscale (float): Polarization fraction scale for polarization ticks. If None, set by maximum polarization fraction where the polarized flux is above 1% of the maximum polarized flux. Default: None.
      in_radec (bool): If True, plots in :math:`\\Delta RA` and :math:`\\Delta Dec`.  If False, plots in sky-right Cartesian coordinates. Default: True.
      xlabel (str): Label for xaxis. Default: ':math:`\\Delta RA (\\mu as)` if in_radec is True, :math:`\\Delta x (\\mu as)` if in_radec is False.
      ylabel (str): Label for yaxis. Default: ':math:`\\Delta Dec (\\mu as)` if in_radec is True, :math:`\\Delta y (\\mu as)` if in_radec is False.
      return_stokes_map (bool): If True returns the intensity map in addition to standard return items.
      transfer_function (str): Transfer function to plot.  Options are 'linear','sqrt','log'.  If 'log', if Imin=0 it will be set to Imax/1000.0 by default, else if Imin<0 it will be assumed to be a dynamic range and Imin = Imax * 10**(Imin).
      tick_fractional_intensity_floor (float): Fraction of maximum intensity at which to stop plotting polarization ticks.  Default: 0.1.
      colorbar (str): Adds colorbar indicating magnitude of polarized flux. Options are None, 'flux', and 'frac'.  If None, no colorbar is plotted.  If 'flux' the polarized flux in Jy will be shown.  If 'frac' the fraction of the maximum intensity in polarized flux will be shown, i.e., :math:`L/I_{max}`.  Default: None.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

    Returns :
      (matplotlib.pyplot.image,matplotlib.pyplot.axes,matplotlib.pyplot.fig,[numpy.ndarray,numpy.ndarray,numpy.ndarray]): Image handle; Figure object; Axes object; Optionally intensity map (x,y,S).
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

    # Set polarization tick shape
    if (tick_shape is None) :
        tick_shape = [20,20]
    elif (isinstance(tick_shape,list)) :
        tick_shape = tick_shape
    else :
        tick_shape = [tick_shape, tick_shape]
    
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
    # Generate a set of Stokes parameters (Jy/uas^2)
    S = polarized_image.stokes_map(x,y,parameters,kind='all',verbosity=verbosity)
    
    # Flip x to flip x axis so that RA decreases right to left
    if (in_radec) :
        x = -x
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])

    I = S[0]
    L = np.sqrt(S[1]**2+S[2]**2)

    # Determine scale
    if (Lmax is None) :
        Lmax = np.max(L)

    # Determine scale
    if (Imax is None) :
        Imax = np.max(I)

    if (verbosity>0) :
        print("Minimum/Maximum I: %g / %g"%(Imin,Imax))
        print("Minimum/Maximum L: %g / %g"%(Lmin,Lmax))
    
    # Transfered I for plotting
    Imin_orig = Imin
    Imax_orig = Imax
    Lmin_orig = Lmin
    Lmax_orig = Lmax
    if (transfer_function=='linear') :
        tI = I/Imax
        Imin = Imin/Imax
        Imax = 1.0
        tL = L/Lmax
        Lmin = Lmin/Lmax
        Lmax = 1.0
    elif (transfer_function=='sqrt') :
        tI = np.sqrt(I/Imax)
        Imin = np.sqrt(Imin/Imax)
        Imax = 1.0
        tL = np.sqrt(L/Lmax)
        Lmin = np.sqrt(Lmin/Lmax)
        Lmax = 1.0
    elif (transfer_function=='log') :
        tI = np.log10(I/Imax)
        if (Imin==0) :
            Imin = np.log10(1.0e-3)
            Imin_orig = 1.0e-3
        elif (Imin>0) :
            Imin = np.log10(Imin/Imax)
        Imax = 0.0
        tL = np.log10(L/Lmax)
        if (Lmin==0) :
            Lmin = np.log10(1.0e-3)
            Lmin_orig = 1.0e-3
        elif (Lmin>0) :
            Lmin = np.log10(Lmin/Lmax)
        Lmax = 0.0

    # Set background color in Axes
    cmap = cm.get_cmap(colormap)    
    plt.gca().set_facecolor(cmap(0.0))

    # Make intensity plot
    h = plt.pcolor(x,y,tI,cmap=colormap,vmin=Imin,vmax=Imax)
    
    # Plot tickmarks
    dxpol = (np.max(x)-np.min(x))/tick_shape[0]
    dypol = (np.max(x)-np.min(x))/tick_shape[1]
    xstride = max( 1, int(dxpol/(np.max(x)-np.min(x))*L.shape[0]+0.5) )
    ystride = max( 1, int(dypol/(np.max(y)-np.min(y))*L.shape[1]+0.5) )
    plot_polticks = (S[0]>tick_fractional_intensity_floor*np.max(S[0]))
    if (not elliptical):
        m = np.sqrt(S[1]**2+S[2]**2)
    else :
        m = np.sqrt(S[1]**2+S[2]**2+S[3]**2)
    if (mscale is None) :
        mscale = 1.0
    pscale = 0.5*np.sqrt(dxpol*dypol)/(mscale * np.max(m[plot_polticks]))
    m = m/S[0]
    
    if (not is_color_like(tick_color)) :
        cmap = cm.get_cmap(tick_color)
    
    for i in range(xstride//2,L.shape[0],xstride) :
        for j in range(ystride//2,L.shape[1],ystride) :
            if (plot_polticks[i,j]) :
                xs = x[i,j]
                ys = y[i,j]
                ms = m[i,j]

                if (not elliptical) :
                    Qs = S[1][i,j]
                    Us = S[2][i,j]
                    EVPA = 0.5*np.arctan2(Us,Qs)
                    px = ms*np.sin(EVPA) * pscale * S[0][i,j]
                    py = ms*np.cos(EVPA) * pscale * S[0][i,j]
                    xtmp = [ (xs-0.5*px), (xs+0.5*px) ]
                    ytmp = [ (ys-0.5*py), (ys+0.5*py) ]
                    fmt = '-'
                else :
                    Ss = [S[0][i,j],S[1][i,j],S[2][i,j],S[3][i,j]]
                    xtmp,ytmp = _get_polarization_ellipse(Ss)
                    xtmp = xs + xtmp*pscale*Ss[0]
                    ytmp = ys + ytmp*pscale*Ss[0]
                    if (Ss[3]>=0) :
                        fmt = '-'
                    else :
                        fmt = '--'
                    
                if (is_color_like(tick_color)) :
                    tcolor = tick_color
                else :
                    #tcolor = cmap((tL[i,j]-Lmin)/(Lmax-Lmin))
                    tcolor = cmap(ms/mscale)

                plt.plot(xtmp,ytmp,fmt,color=tcolor,lw=1)

                

    # # Add colorbar if desired
    if ((not is_color_like(tick_color)) and colorbar) :
        lax = plt.pcolor(x,y,m*100,vmin=0,vmax=100*mscale,visible=False,cmap=tick_color)
        cb = plt.colorbar(lax)
        cb.ax.set_ylabel('Polarization fraction (%)',rotation=270,va='bottom')

    # Add labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Fix aspect ratio
    plt.axis('square')

    # Set limits
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])
    
    # Fix x-axis to run in sky
    if (in_radec) :
        plt.gca().invert_xaxis()        

    if (return_stokes_map) :
        return h,plt.gca(),plt.gcf(),x,y,S
    else:
        return h,plt.gca(),plt.gcf()


def plot_polarized_image_stats(polarized_image, chain, limits=None, shape=None, tick_shape=None, colormap='gray_r', tick_color='jet', Imin=0.0, Imax=None, Lmin=0.0, Lmax=None, mscale=None, in_radec=True, xlabel=None, ylabel=None, return_stokes_map=False, transfer_function='linear', tick_fractional_intensity_floor=0.1, colorbar=False, elliptical=False, verbosity=0) :
    """
    Plots linear polarization fraction, flux and intensity image associated with a given 
    :class:`model_image` object evaluated at a specified set of parameters. A number of additional 
    practical options may be passed. Access to a matplotlib pcolor object, the figure handle and axes 
    handles are returned. Optionally, the Stokes map is returned.

    Args:
      polarized_image (model_polarized_image): :class:`model_polarized_image` object to plot.
      chain (list): List of parameters that define model_image intensity map.
      limits (float,list): Limits in uas on image size. If a single float is given, the limits are uniformly set. If a list is given, it must be in the form [xmin,xmax,ymin,ymax]. Default: 75.
      shape (int,list): Number of pixels. If a single int is given, the number of pixels are uniformly set in both directions.  If a list is given, it must be in the form [Nx,Ny]. Default: 256.
      tick_shape (int,list): Number of ticks to plot. If a single int is given, the number of ticks are uniformly set in both directions.  If a list is given, it must be in the form [Mx,My]. Default: 16.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'afmhot'.
      tick_color (matplotlib.colors.Colormap,str,list): Any acceptable colormap or color type as specified in :mod:`matplotlib.colors`.
      Imin (float): Minimum intensity for colormap. Default: 0 (though see transfer_function).
      Imax (float): Maximum intensity for colormap. Default: maximum linearly polarized intensity.
      Lmin (float): Minimum intensity for colormap. Default: 0 (though see transfer_function).
      Lmax (float): Maximum intensity for colormap. Default: maximum linearly polarized intensity.
      mscale (float): Polarization fraction scale for polarization ticks. If None, set by maximum polarization fraction where the polarized flux is above 1% of the maximum polarized flux. Default: None.
      in_radec (bool): If True, plots in :math:`\\Delta RA` and :math:`\\Delta Dec`.  If False, plots in sky-right Cartesian coordinates. Default: True.
      xlabel (str): Label for xaxis. Default: ':math:`\\Delta RA (\\mu as)` if in_radec is True, :math:`\\Delta x (\\mu as)` if in_radec is False.
      ylabel (str): Label for yaxis. Default: ':math:`\\Delta Dec (\\mu as)` if in_radec is True, :math:`\\Delta y (\\mu as)` if in_radec is False.
      return_stokes_map (bool): If True returns the intensity map in addition to standard return items.
      transfer_function (str): Transfer function to plot.  Options are 'linear','sqrt','log'.  If 'log', if Imin=0 it will be set to Imax/1000.0 by default, else if Imin<0 it will be assumed to be a dynamic range and Imin = Imax * 10**(Imin).
      tick_fractional_intensity_floor (float): Fraction of maximum intensity at which to stop plotting polarization ticks.  Default: 0.1.
      colorbar (str): Adds colorbar indicating magnitude of polarized flux. Options are None, 'flux', and 'frac'.  If None, no colorbar is plotted.  If 'flux' the polarized flux in Jy will be shown.  If 'frac' the fraction of the maximum intensity in polarized flux will be shown, i.e., :math:`L/I_{max}`.  Default: None.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints various elements of the plotting process. Passed to :func:`model_image.generate_intensity_map`. Default: 0.

    Returns :
      (matplotlib.pyplot.image,matplotlib.pyplot.axes,matplotlib.pyplot.fig,[numpy.ndarray,numpy.ndarray,numpy.ndarray]): Image handle; Figure object; Axes object; Optionally intensity map (x,y,S).
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

    # Set polarization tick shape
    if (tick_shape is None) :
        tick_shape = [20,20]
    elif (isinstance(tick_shape,list)) :
        tick_shape = tick_shape
    else :
        tick_shape = [tick_shape, tick_shape]
    
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

    # Flip x to flip x axis so that RA decreases right to left
    if (in_radec) :
        x = -x
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])

    tIavg = 0.0*x
    
    for parameters in chain :

        # Flip x to flip x axis so that RA decreases right to left
        if (in_radec) :
            x = -x
        
        # Generate a set of Stokes parameters (Jy/uas^2)
        S = polarized_image.stokes_map(x,y,parameters,kind='all',verbosity=verbosity)

        # Flip x to flip x axis so that RA decreases right to left
        if (in_radec) :
            x = -x
        
        I = S[0]
        L = np.sqrt(S[1]**2+S[2]**2)

        # Determine scale
        if (Lmax is None) :
            Lmax = np.max(L)

        # Determine scale
        if (Imax is None) :
            Imax = np.max(I)

        if (verbosity>0) :
            print("Minimum/Maximum I: %g / %g"%(Imin,Imax))
            print("Minimum/Maximum L: %g / %g"%(Lmin,Lmax))

        # Transfered I for plotting
        Lmin_orig = Lmin
        Lmax_orig = Lmax
        if (transfer_function=='linear') :
            tI = I/Imax
            tL = L/Lmax
            Lmin = Lmin/Lmax
            Lmax = 1.0
        elif (transfer_function=='sqrt') :
            tI = np.sqrt(I/Imax)
            tL = np.sqrt(L/Lmax)
            Lmin = np.sqrt(Lmin/Lmax)
            Lmax = 1.0
        elif (transfer_function=='log') :
            tI = np.log10(I/Imax)
            tL = np.log10(L/Lmax)
            if (Lmin==0) :
                Lmin = np.log10(1.0e-3)
                Lmin_orig = 1.0e-3
            elif (Lmin>0) :
                Lmin = np.log10(Lmin/Lmax)
            Lmax = 0.0

        # Set background color in Axes
        cmap = cm.get_cmap(colormap)    
        plt.gca().set_facecolor(cmap(0.0))

        # Save intensity for intensity plot
        tIavg = tIavg+tI

        # Plot tickmarks
        dxpol = (np.max(x)-np.min(x))/tick_shape[0]
        dypol = (np.max(x)-np.min(x))/tick_shape[1]
        xstride = max( 1, int(dxpol/(np.max(x)-np.min(x))*L.shape[0]+0.5) )
        ystride = max( 1, int(dypol/(np.max(y)-np.min(y))*L.shape[1]+0.5) )
        plot_polticks = (S[0]>tick_fractional_intensity_floor*np.max(S[0]))
        if (not elliptical):
            m = np.sqrt(S[1]**2+S[2]**2)
        else :
            m = np.sqrt(S[1]**2+S[2]**2+S[3]**2)
        if (mscale is None) :
            mscale = 1.0
        pscale = 0.5*np.sqrt(dxpol*dypol)/(mscale * np.max(m[plot_polticks]))
        m = m/S[0]

        if (not is_color_like(tick_color)) :
            cmap = cm.get_cmap(tick_color)

        for i in range(xstride//2,L.shape[0],xstride) :
            for j in range(ystride//2,L.shape[1],ystride) :
                if (plot_polticks[i,j]) :
                    xs = x[i,j]
                    ys = y[i,j]
                    ms = m[i,j]

                    if (not elliptical) :
                        Qs = S[1][i,j]
                        Us = S[2][i,j]
                        EVPA = 0.5*np.arctan2(Us,Qs)
                        px = ms*np.sin(EVPA) * pscale * S[0][i,j]
                        py = ms*np.cos(EVPA) * pscale * S[0][i,j]
                        xtmp = [ (xs-0.5*px), (xs+0.5*px) ]
                        ytmp = [ (ys-0.5*py), (ys+0.5*py) ]
                        fmt = '-'
                    else :
                        Ss = [S[0][i,j],S[1][i,j],S[2][i,j],S[3][i,j]]
                        xtmp,ytmp = _get_polarization_ellipse(Ss)
                        xtmp = xs + xtmp*pscale*Ss[0]
                        ytmp = ys + ytmp*pscale*Ss[0]
                        if (Ss[3]>=0) :
                            fmt = '-'
                        else :
                            fmt = '--'

                    if (is_color_like(tick_color)) :
                        tcolor = tick_color
                    else :
                        #tcolor = cmap((tL[i,j]-Lmin)/(Lmax-Lmin))
                        tcolor = cmap(ms/mscale)

                    plt.plot(xtmp,ytmp,fmt,color=tcolor,lw=2,alpha=0.01,zorder=2)
            


    # Make intensity plot
    tI = tIavg/chain.shape[0]

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
            Imin_orig = 1.0e-3
        elif (Imin>0) :
            Imin = np.log10(Imin/Imax)
            Imax = 0.0

    #h = plt.pcolor(x,y,tI,cmap=colormap,vmin=Imin,vmax=Imax,zorder=1)
    h = plt.pcolor(x,y,tI,cmap=colormap,vmin=Imin,vmax=Imax,zorder=1)


    # # Add colorbar if desired
    if ((not is_color_like(tick_color)) and colorbar) :
        lax = plt.pcolor(x,y,m*100,vmin=0,vmax=100*mscale,visible=False,cmap=tick_color)
        cb = plt.colorbar(lax)
        cb.ax.set_ylabel('Polarization fraction (%)',rotation=270,va='bottom')

    # Add labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Fix aspect ratio
    plt.axis('square')

    # Set limits
    plt.gca().set_xlim(limits[0],limits[1])
    plt.gca().set_ylim(limits[2],limits[3])
    
    # Fix x-axis to run in sky
    if (in_radec) :
        plt.gca().invert_xaxis()        

    if (return_stokes_map) :
        return h,plt.gca(),plt.gcf(),x,y,S
    else:
        return h,plt.gca(),plt.gcf()

    

def plot_dterm_posteriors(polarized_image, chain, lcolormap='Reds', rcolormap='Greens', station=None, alpha=0.5, comparison_dterms=None, comparison_fmts=None, grid=True, fig=None, axs=None, scott_factor=2.0, nbin=128, size=None, plevels=None, station_labels=None, fill=True, edges=False, fill_zorder=None, edge_zorder=None, edge_rcolors=None, edge_lcolors=None, edge_colormap=None, edge_alpha=None, linewidth=1, verbosity=0) :
    """
    Plots D term distributions for a polarized image model from an MCMC chain.  Each station will 
    separately have its left and right D term distributions plotted in the complex plane.

    Args:
      polarized_image (model_polarized_image): :class:`model_polarized_image` object to plot.
      chain (numpy.ndarray): MCMC chain with parameters that include the fitted D terms.
      lcolor (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`.
      rcolor (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`.
      station (str,list): A single station code or list of station codes to which restrict attention or exclude.  Station codes must match those in the :class:`polarized_model_image.dterm_dict['station_names']` list for the polarized_image.  Station codes prefixed with '!' will be excluded. Default: D terms will be plotted for all stations.
      comparison_dterms (dict): Dictionary of D terms to over-plot as comparisons. Dictionary should be indexed by station code.  Two element lists will be interpreted as the (complex) left and right D terms.  Four element lists will be interpreted as the (complex) right and left D terms followed by estimates of their (complex) uncertainties.
      comparison_fmts (str,list): Formats for the comparison D term points.  These must be an acceptable format string as describe in :func:`matplotlib.pyplot.plot`. If a single format string is given it will be used for both points. If a list of two format strings are given, they will be used for the right and left points, respectively. If set to None, the formats ['>g', '<r'] are used. Default: None.
      grid (bool): Flag that determines whether or not to plot a background grid. Default: True.
      fig (matplotlib.pyplot.figure): Figure on which to add axes. Unnecessary if a list of axes are provided. If None, a new figure will be generated. Default: None.
      axs (list): List of axes generated by prior calls to :func:`plot_dterm_posteriors` to which to add posterior plots. This must match the number and order of axes that would be generated. If provided a figure need not be. If None, new axes will be generated. Default: None.
      scott_factor (float): Multiplier in the Scott KDE kernel. Default: 2.
      nbin (int): Number of bins to use to construct the contour plot.
      size (list): List of two integers specifying the number of windows in the horizontal and vertical direction. If None, the number of windows will be set to be the most square possible. Default: None.
      plevels (list): Probability levels at which to plot contours. If None, will plot the 1, 2 and 3 sigma contours. Default: None.
      station_labels (dict): Dictionary of station labels to plot organized as 'station code':'label' pairs. If None, the station codes will be used. Default: None.
      fill (bool): Determines if contour levels will be filled. Default: True.
      fill_zorder (int): Sets zorder of filled contours. Default: None.
      edges (bool): Deterines if countour lines will be plotted. Default: False.
      edge_zorder (int): Sets zorder of contour lines. Default: None.
      edge_lcolors (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors` or list of such types. If not None, overrides edge_colormap. Default: None.
      edge_rcolors (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors` or list of such types. If not None, overrides edge_colormap. Default: None.
      edge_colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. If None, uses the colormap passed by the colormap option. Default: None.
      edge_alpha (float): Value of alpha for contour lines. Only meaningful if edges=True. If None, sets the contour line alpha to that passed by alpha. Default: None.
      linewidth (float): Width of contour lines. Default: 1.
      verbosity (int): Verbosity level.
    
    Returns:
      (matplotlib.pyplot.figure,list,list): Figure handle and lists of axes and plot handles.
    """

    # Check that D terms are being fit
    if (polarized_image.number_of_dterms==0) :
        print("No D terms present in model, nothing to plot!")
        return


    # Flatten chain
    Nparam = chain.shape[-1]
    chain = np.reshape(chain,[-1,Nparam])


    # Set station names
    if (station_labels is None) :
        station_labels = {}
        for station_code in polarized_image.dterm_dict['station_names'] :
            station_labels[station_code] = station_code

    
    # Get the list of indices to plot
    if (station is None) :
        # By default include every station
        station_list = polarized_image.dterm_dict['station_names']
        station_dict = {}
        station_dict['station_list'] = []
        for station in station_list :
            station_dict['station_list'].append(station)
            station_dict[station] = ['R', 'L']
            
        
    else :
        #  If a single string is passed, then make a list of it.
        if (isinstance(station,str)) :
            station = [station]

        # Exclude those station codes prepended with '!', otherwise include them exclusively.
        #  Make a list of excluded or included stations
        station_exclude_list = []
        station_include_list = []
        #  Loop over stations in the list and check for '!', adding the station to the proper list
        if (isinstance(station,list)) :
            station_list = copy.copy(station)
            for station in station_list :
                if (station[0]=='!') :
                    if (':' in station[1:]) :
                        station_exclude_list.append( station[1:] )
                    else :
                        station_exclude_list.append( station[1:]+':R' )
                        station_exclude_list.append( station[1:]+':L' )
                else :
                    if (':' in station) :
                        station_include_list.append( station )
                    else :
                        station_include_list.append( station+':R' )
                        station_include_list.append( station+':L' )
                        
        #  If no include list is given, include every station
        if (len(station_include_list)==0) :
            for station in polarized_image.dterm_dict['station_names'] :
                station_include_list.append( station+':R' )
                station_include_list.append( station+':L' )

        #  Construct the total station list from the closure of the exclude list intersect the include list
        station_dict = {}
        station_dict['station_list'] = []
        for station in station_include_list :
            #if (not station in station_exclude_list) :
            if (not station in station_exclude_list) :
                toks = station.split(':')
                if (not toks[0] in station_dict['station_list']) :
                    station_dict['station_list'].append(toks[0])
                if (toks[0] in station_dict.keys()) :
                    station_dict[toks[0]].append(toks[1])
                else :
                    station_dict[toks[0]] = [toks[1]]

                    
    if (comparison_fmts==None) :
        comparison_fmts=['>g','<r']
    elif (isinstance(comparison_fmts,str)) :
        comparison_fmts=[comparison_fmts,comparison_fmts]
                
    if (verbosity>0) :
        print("Station list for which to plot D terms:\n",station_dict['station_list'])
        
    dterm_start_index_list = []
    for k in range(polarized_image.image_size,polarized_image.size,4) :
        if (polarized_image.dterm_dict['station_names'][(k-polarized_image.image_size)//4] in station_dict['station_list']) :
            dterm_start_index_list.append(k)

    if (verbosity>0) :
        print("Assummed starting index in parameter list:\n",dterm_start_index_list)

    # Makes sure that we are plotting something!
    if (len(dterm_start_index_list)==0) :
        warnings.warn("No D term posteriors will be generated.")
        return
    
    # Determine the windows particulars
    if (size is None) :
        Nwx = int(np.ceil(np.sqrt(len(dterm_start_index_list))))
        Nwy = int(np.ceil(len(dterm_start_index_list)/Nwx))
    else :
        Nwx = size[0]
        Nwy = size[1]
        
    wldxy = 3
    figsize = (1+Nwx*wldxy,1+Nwy*wldxy)
    if (not (fig is None)) :
        pass
    elif (not (axs is None)) :
        fig = plt.gcf()
    else :
        fig = plt.figure(figsize=figsize)
    aml = 0.9/figsize[0]
    amr = 0.1/figsize[0]
    amb = 0.9/figsize[1]
    amt = 0.1/figsize[1]
    abx = 0.2*wldxy/figsize[0]
    aby = 0.2*wldxy/figsize[1]
    awdx = wldxy/figsize[0] - abx*(Nwx-1)/Nwx
    awdy = wldxy/figsize[1] - aby*(Nwy-1)/Nwy

    # Begin looping over stations and generting axes
    axlist = []
    hlist = []
    for k,station in enumerate(station_dict['station_list']) :

        iwx = k%Nwx
        iwy = Nwy-1-k//Nwx

        # Create an axis object
        if (not (axs is None)) :
            ax = axs[k]
            plt.sca(ax)
        else :
            ax = plt.axes([ (aml+iwx*(awdx+abx)), (amb+iwy*(awdy+aby)), awdx, awdy ])
        axlist.append(ax)
        
        # Add the right D-term plots
        if ('R' in station_dict[station]) :
            DRr = chain[:,dterm_start_index_list[k]+0]
            DRi = chain[:,dterm_start_index_list[k]+1]
            limits = [[1.2*np.min(DRr)-0.2*np.max(DRr),1.2*np.max(DRr)-0.2*np.min(DRr)],[1.2*np.min(DRi)-0.2*np.max(DRi),1.2*np.max(DRi)-0.2*np.min(DRi)]]
            #print("FOO:",limits,np.min(DRr),np.max(DRr),np.min(DRi),np.max(DRi))
            #hR = kde_plot_2d(DRr,DRi,colormap=rcolormap,alpha=alpha,scott_factor=np.sqrt(2.0))
            hR = kde_plot_2d(DRr,DRi,colormap=rcolormap,alpha=alpha,scott_factor=scott_factor,nbin=nbin,plevels=plevels,fill=fill,edges=edges,fill_zorder=fill_zorder,edge_zorder=edge_zorder,edge_colors=edge_rcolors,edge_colormap=edge_colormap,edge_alpha=edge_alpha,linewidth=linewidth,limits=limits)
            hlist.append(hR)

        # Add the left D-term plots
        if ('L' in station_dict[station]) :
            DLr = chain[:,dterm_start_index_list[k]+2]
            DLi = chain[:,dterm_start_index_list[k]+3]
            limits = [[1.2*np.min(DLr)-0.2*np.max(DLr),1.2*np.max(DLr)-0.2*np.min(DLr)],[1.2*np.min(DLi)-0.2*np.max(DLi),1.2*np.max(DLi)-0.2*np.min(DLi)]]
            #print("BAR:",limits,np.min(DLr),np.max(DLr),np.min(DLi),np.max(DLi))
            #hL = kde_plot_2d(DLr,DLi,colormap=lcolormap,alpha=alpha,scott_factor=np.sqrt(2.0))
            hL = kde_plot_2d(DLr,DLi,colormap=lcolormap,alpha=alpha,scott_factor=scott_factor,nbin=nbin,plevels=plevels,fill=fill,edges=edges,fill_zorder=fill_zorder,edge_zorder=edge_zorder,edge_colors=edge_lcolors,edge_colormap=edge_colormap,edge_alpha=edge_alpha,linewidth=linewidth,limits=limits)
            hlist.append(hL)

        # Add truths if provided
        if (not comparison_dterms is None) :
            if ( station in comparison_dterms.keys() ) :
                if (len(comparison_dterms[station])==4) :
                    if ('R' in station_dict[station]) :
                        plt.errorbar(comparison_dterms[station][0].real,comparison_dterms[station][0].imag,fmt=comparison_fmts[0],xerr=comparison_dterms[station][2].real,yerr=comparison_dterms[station][2].imag)
                    if ('L' in station_dict[station]) :
                        plt.errorbar(comparison_dterms[station][1].real,comparison_dterms[station][1].imag,fmt=comparison_fmts[1],xerr=comparison_dterms[station][3].real,yerr=comparison_dterms[station][3].imag)
                if ('R' in station_dict[station]) :
                    plt.plot(comparison_dterms[station][0].real,comparison_dterms[station][0].imag,comparison_fmts[0])
                if ('L' in station_dict[station]) :
                    plt.plot(comparison_dterms[station][1].real,comparison_dterms[station][1].imag,comparison_fmts[1])

                 

        # Add grids
        plt.grid(grid)
        
        # Add label
        #print(station_labels)
        #print(station_labels[station])
        plt.text(0.95,0.95,station_labels[station],transform=ax.transAxes,ha='right',va='top')
        
        # Add labels if on the boundary
        if (iwy==0) :
            plt.xlabel(r'Re$(D_{L,R})$')
        if (iwx==0) :
            plt.ylabel(r'Im$(D_{L,R})$')

            
                
                            
    return fig,axlist,hlist
    
    
def write_dterm_statistics(polarized_image, chain, outfile="dterm_statistics.txt", station=None, verbosity=0) :
    """
    Writes a text file contaning D-term means and standard deviations.

    Args:
      polarized_image (model_polarized_image): :class:`model_polarized_image` object to plot.
      chain (numpy.ndarray): MCMC chain with parameters that include the fitted D terms.
      outfile (str): Text file in which to write the D-term statistics.  Default: dterm_statistics.txt.
      station (str,list): A single station code or list of station codes to which restrict attention or exclude.  Station codes must match those in the :class:`polarized_model_image.dterm_dict['station_names']` list for the polarized_image.  Station codes prefixed with '!' will be excluded. Default: D terms will be plotted for all stations.
      verbosity (int): Verbosity level.
    
    Returns:
    """

    # Check that D terms are being fit
    if (polarized_image.number_of_dterms==0) :
        print("No D terms present in model, nothing to plot!")
        return


    # Flatten chain
    Nparam = chain.shape[-1]
    chain = np.reshape(chain,[-1,Nparam])
    
    # Get the list of indices to plot
    if (station is None) :
        # By default include every station
        station_list = polarized_image.dterm_dict['station_names']
        station_dict = {}
        station_dict['station_list'] = []
        for station in station_list :
            station_dict['station_list'].append(station)
            station_dict[station] = ['R', 'L']
            
        
    else :
        #  If a single string is passed, then make a list of it.
        if (isinstance(station,str)) :
            station = [station]

        # Exclude those station codes prepended with '!', otherwise include them exclusively.
        #  Make a list of excluded or included stations
        station_exclude_list = []
        station_include_list = []
        #  Loop over stations in the list and check for '!', adding the station to the proper list
        if (isinstance(station,list)) :
            station_list = copy.copy(station)
            for station in station_list :
                if (station[0]=='!') :
                    if (':' in station[1:]) :
                        station_exclude_list.append( station[1:] )
                    else :
                        station_exclude_list.append( station[1:]+':R' )
                        station_exclude_list.append( station[1:]+':L' )
                else :
                    if (':' in station) :
                        station_include_list.append( station )
                    else :
                        station_include_list.append( station+':R' )
                        station_include_list.append( station+':L' )
                        
        #  If no include list is given, include every station
        if (len(station_include_list)==0) :
            for station in polarized_image.dterm_dict['station_names'] :
                station_include_list.append( station+':R' )
                station_include_list.append( station+':L' )

        #  Construct the total station list from the closure of the exclude list intersect the include list
        station_dict = {}
        station_dict['station_list'] = []
        for station in station_include_list :
            #if (not station in station_exclude_list) :
            if (not station in station_exclude_list) :
                toks = station.split(':')
                if (not toks[0] in station_dict['station_list']) :
                    station_dict['station_list'].append(toks[0])
                if (toks[0] in station_dict.keys()) :
                    station_dict[toks[0]].append(toks[1])
                else :
                    station_dict[toks[0]] = [toks[1]]

                    
    if (verbosity>0) :
        print("Station list for which to plot D terms:\n",station_dict['station_list'])
        
    dterm_start_index_list = []
    for k in range(polarized_image.image_size,polarized_image.size,4) :
        if (polarized_image.dterm_dict['station_names'][(k-polarized_image.image_size)//4] in station_dict['station_list']) :
            dterm_start_index_list.append(k)

    if (verbosity>0) :
        print("Assummed starting index in parameter list:\n",dterm_start_index_list)

    # Makes sure that we are plotting something!
    if (len(dterm_start_index_list)==0) :
        warnings.warn("No D term posteriors will be generated.")
        return

    out = open(outfile,'w')
    out.write("# %13s %15s %15s %15s %15s %15s %15s %15s %15s\n"%('Station','mean DR.real','stddev DR.real','mean DR.imag','stddev DR.imag','mean DL.real','stddev DL.real','mean DL.imag','stddev DL.imag'))
    for k,station in enumerate(station_dict['station_list']) :
        
        # Add the right D-term plots
        if ('R' in station_dict[station]) :
            DRr = chain[:,dterm_start_index_list[k]+0]
            DRi = chain[:,dterm_start_index_list[k]+1]

            DRr_mean = np.mean(DRr)
            DRr_sig = np.std(DRr)
            DRi_mean = np.mean(DRi)
            DRi_sig = np.std(DRi)
        else :
            DRr_mean = np.nan
            DRr_sig = np.nan
            DRi_mean = np.nan
            DRi_sig = np.nan
            
            
        # Add the left D-term plots
        if ('L' in station_dict[station]) :
            DLr = chain[:,dterm_start_index_list[k]+2]
            DLi = chain[:,dterm_start_index_list[k]+3]

            DLr_mean = np.mean(DLr)
            DLr_sig = np.std(DLr)
            DLi_mean = np.mean(DLi)
            DLi_sig = np.std(DLi)
        else :
            DLr_mean = np.nan
            DLr_sig = np.nan
            DLi_mean = np.nan
            DLi_sig = np.nan


        out.write("%15s %15.8g %15.8g %15.8g %15.8g %15.8g %15.8g %15.8g %15.8g\n"%(station,DRr_mean,DRr_sig,DRi_mean,DRi_sig,DLr_mean,DLr_sig,DLi_mean,DLi_sig))

    out.close()

    
    

    
def _get_station_names_from_tokens(toks) :
    """
    Gets station names from tokens.

    Args:
      toks (list): List of str tokens that should be interpreted as bool, [list of station names]

    Returns:
      (list): List of str corresponding to station names.

    """

    if (len(toks)<1) :
        raise RuntimeError("This does not appear to be a model_polarization_image tagvers 1.0 tag.")
    if (bool(int(toks[0]))) :
        station_names = toks[1:]
    else :
        station_names = None

    return station_names

    
def construct_model_polarized_image_from_tagv1(tag,verbosity=0) :
    """
    Parses a tagvers-1.0 Themis tag and recursively constructs a model polarized image.

    Args:
      tag (list) : List of tags, arranged as str on separate lines.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints tag information.

    Returns:
      (model_image, list) : The first :class:`model_polarized_image` object fully described within the tag; the remaining tag lines.
    """

    if (verbosity>0) :
        for j,l in enumerate(tag) :
            print("%i : %s"%(j,tag[j]))
        print("---------------------------------")

    # Split the current tag and look for information about D terms.
    toks = tag[0].split()

    # Loop over implemented names
    if (toks[0]=='model_polarized_image_constant_polarization') :
        subtag = tagv1_find_subtag(tag[1:])
        subimage,_ = construct_model_image_from_tagv1(subtag,verbosity=verbosity)
        return model_polarized_image_constant_polarization(subimage,station_names=_get_station_names_from_tokens(toks[1:])),tag[(len(subtag)+3):]

    elif (toks[0]=='model_polarized_image_adaptive_splined_raster') :
        return model_polarized_image_adaptive_splined_raster(int(toks[1]),int(toks[2]),a=float(toks[3]),station_names=_get_station_names_from_tokens(toks[4:])),tag[1:]

    elif (toks[0]=='model_polarized_image_sum') :
        offset_coordinates = toks[1]
        subtag = tagv1_find_subtag(tag[1:])
        len_subtag = len(subtag)
        image_list = []
        while (len(subtag)>0) :
            subimage,subtag = construct_model_polarized_image_from_tagv1(subtag,verbosity=verbosity)
            image_list.append(subimage)
        return model_polarized_image_sum(image_list,offset_coordinates=offset_coordinates,station_names=_get_station_names_from_tokens(toks[2:])),tag[len_subtag+3:]

    else :
        raise RuntimeError("Unrecognized model tag %s"%(tag[0]))

    
def construct_model_polarized_image_from_tag_file(tag_file_name='model_image.tag', tagversion='tagvers-1.0',verbosity=0) :
    """
    Reads tag file produced by the :cpp:func:`Themis::model_polarized_image::write_model_tag_file`
    function and returns an appropriate model polarized image.

    Args:
      tag_file_name (str): Name of the tag_file. Default: 'model_image.tag'
      tagversion (str): Version of tag system. Currently only tagvers-1.0 is implemented. Default 'tagvers-1.0'.
      verbosity (int): Verbosity level. 0 prints nothing. 1 prints tag information.

    Returns:
      (model_polarized_image): :class:`model_polarized_image` object of the corresponding model.
    """


    tagin = open(tag_file_name,'r')
    tag = tagin.readlines()
    tagin.close()
    
    tag = [l.strip('\n\r') for l in tag]
    tagversion_file = tag[0]
    tag = tag[1:]

    if (tagversion_file==tagversion) :
        image,tag = construct_model_polarized_image_from_tagv1(tag,verbosity=verbosity)
        if (len(tag)>0) :
            raise RuntimeError(("Remaining tag lines:"+len(tag)*("   %s\n"))%tuple(tag))
        return image
    else:
        raise RuntimeError("Tag versions other than tagvers-1.0 are not supported at this time.")

    
