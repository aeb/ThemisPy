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
        fI = np.flipud(np.transpose(np.exp(np.array(self.parameters[:Ndim]).reshape([self.Nx,self.Ny])))) * uas2rad**2
        p = np.flipud(np.transpose(np.exp(np.array(self.parameters[Ndim:2*Ndim]).reshape([self.Nx,self.Ny]))))
        evpa = np.flipud(np.transpose(np.array(self.parameters[2*Ndim:3*Ndim]).reshape([self.Nx,self.Ny])))
        muV = np.flipud(np.transpose(np.array(self.parameters[3*Ndim:4*Ndim]).reshape([self.Nx,self.Ny])))

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

        
        if (self.spline_method=='fft') :
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
    


def plot_linear_polarization_map(polarized_image, parameters, limits=None, shape=None, tick_shape=None, colormap='plasma', tick_color='w', Lmin=0.0, Lmax=None, mscale=None, in_radec=True, xlabel=None, ylabel=None, return_stokes_map=False, transfer_function='linear', intensity_contours=False, intensity_colors=None, verbosity=0) :
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
      Lmax (float): Maximum intensity for colormap. Default: maximum intensity.
      mscale (float): Polarization fraction scale for polarization ticks. If None, set by maximum polarization fraction where the polarized flux is above 1% of the maximum polarized flux. Default: None.
      in_radec (bool): If True, plots in :math:`\\Delta RA` and :math:`\\Delta Dec`.  If False, plots in sky-right Cartesian coordinates. Default: True.
      xlabel (str): Label for xaxis. Default: ':math:`\\Delta RA (\\mu as)` if in_radec is True, :math:`\\Delta x (\\mu as)` if in_radec is False.
      ylabel (str): Label for yaxis. Default: ':math:`\\Delta Dec (\\mu as)` if in_radec is True, :math:`\\Delta y (\\mu as)` if in_radec is False.
      return_stokes_map (bool): If True returns the intensity map in addition to standard return items.
      transfer_function (str): Transfer function to plot.  Options are 'linear','sqrt','log'.  If 'log', if Imin=0 it will be set to Imax/1000.0 by default, else if Imin<0 it will be assumed to be a dynamic range and Imin = Imax * 10**(Imin).
      intensity_contours (int,list,bool): May either be True, an integer, or a list of contour level values. Default: False. If True, sets 8 linearly spaced contour levels.
      intensity_colors (list): A list composed of any acceptable color type as specified in :mod:`matplotlib.colors`.
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
        elif (Lmin>0) :
            Lmin = np.log10(Lmin/Lmax)
        Lmax = 0.0

    # Set background color in Axes
    cmap = cm.get_cmap(colormap)    
    plt.gca().set_facecolor(cmap(0.0))

    # Make plot
    h = plt.pcolor(x,y,tL,cmap=colormap,vmin=Lmin,vmax=Lmax)

    # Plot tickmarks
    dxpol = (np.max(x)-np.min(x))/tick_shape[0]
    dypol = (np.max(x)-np.min(x))/tick_shape[1]
    xstride = max( 1, int(dxpol/(np.max(x)-np.min(x))*L.shape[0]+0.5) )
    ystride = max( 1, int(dypol/(np.max(y)-np.min(y))*L.shape[1]+0.5) )
    plot_polticks = (S[0]>0.01*np.max(S[0]))
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

    
