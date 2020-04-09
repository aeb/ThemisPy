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


rad2uas = 180.0/np.pi * 3600 * 1e6
uas2rad = np.pi/(180.0 * 3600 * 1e6)



class Image :
    """
    Base class for Themis image models.

    This base class is intended to mirror the Themis model_image base class, providing 
    explicit visualizations of the output of Themis model_image-derived model analyses.

    Attributes:
      parameters (list): Parameter list for a given model.
      size (int): Number of parameters for a given model.
    """
    
    def __init__(self) :
        """
        Initializer.

        By default both the parameters and size are set to None, forcing derived child
        classes to properly initialize these later.
        """
        
        self.parameters=None
        self.size=None

        
    def generate(self,parameters) :
        """
        Sets the model parameter list.  Mirrors similar a similar function within the Themis
        model_image class.  Effectively simply copies the parameters into the local list with
        some additional run-time checking.

        Args:
          parameters (list): Parameter list.

        Returns:
          None
        """
        self.parameters = np.copy(parameters)
        if (len(self.parameters)<self.size) :
            raise RuntimeError("Parameter list is inconsistent with the number of parameters expected.")
    

    def intensity_map(self,x,y,parameters=None,verbosity=0) :
        """
        Hook for the intensity map function in child classes.

        Args:
          x (numpy.ndarray): Array of RA offsets in microarcseconds (usually plaid 2D)
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D)
          parameters (list): Parameter list. If none is provided, uses the current set of parameters. Default: None.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """
        
        raise NotImplementedError("intensity_map must be defined in children classes of Image.")

    
class Image_SymmetricGaussian(Image) :
    """
    Symmetric gaussian image class that is a mirror of model_image_symmetric_gaussian.

    Attributes:
      parameters (list): Parameter list (Total intensity in Jy, standard deviation in rad).
      size (int): 2.
    """

    def __init__(self) :
        """
        Initializer.
        """
        self.size=2

        
    def intensity_map(self,x,y,parameters=None,verbosity=0) :
        """
        Returns the intensity map.

        Args:
          x (numpy.ndarray): Array of RA offsets in microarcseconds (usually plaid 2D)
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D)
          parameters (list): Parameter list. If none is provided, uses the current set of parameters. Default: None.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """

        if (not parameters is None) :
            self.generate(parameters)
    
        s = self.parameters[1] * rad2uas
        I0 = self.parameters[0] / (2.*np.pi*s**2)

        I = I0 * np.exp( - 0.5*( x**2 + y**2 )/s**2 )
        
        if (verbosity>0) :
            print("Symmetric Gaussian:",I0,s)
            
        return I


    
class Image_Sum(Image) :
    """
    Image sum class that generates images by summing other images supplemented with some offset.

    Attributes:
      image_list (list): List of Image objects that are to be summed.
      x0_list (numpy.ndarray): Array of RA shifts in rad.
      y0_list (numpy.ndarray): Array of Dec shifts in rad.
      parameters (list): Parameter list constructed by the concatenation of p1+[x1,y1] + p2+[x2,y2] + ... where pj is the parameter list of the jth Image object in the image_list.
      size (int): Sum of the sizes of the images + the number of shifts (2 times the number of images).
    """

    def __init__(self,image_list=None) :
        """
        Initializer that optionally takes a list of Image objects.
        
        Args:
          image_list (list): List of Image objects. If None, must be set prior to calling :func:`intensity_map`. Default: None.
        """
        
        self.image_list=[]
        self.x0_list=None
        self.y0_list=None
        
        if (not image_list is None) :
            self.image_list = copy.copy(image_list) # This is a SHALLOW copy

        self.size=0
        for image in image_list :
            self.size += image.size+2
        self.x0_list = np.zeros(len(image_list))
        self.y0_list = np.zeros(len(image_list))
        

    def add(self,image) :
        """
        Adds an Image object or list of Image objects to the sum.  The size is recomputed.

        Args:
          image (Image, list): A single Image object or list of Image objects to be added.

        Returns:
          None
        """
        
        # Can add lists and single istances
        if (isinstance(image,list)) :
            self.image_list = image_list + image
        else :
            self.image_list.append(image)
        # Get new size
        self.size=0
        for image in image_list :
            self.size += image.size+2
        self.x0_list = np.zeros(len(image_list))
        self.y0_list = np.zeros(len(image_list))

        
    def generate(self,parameters) :
        """
        Sets the combined model parameter list.  Mirrors similar a similar function within the Themis
        model_image_sum class.  Effectively simply copies the parameters into the local list, copies the
        shift vector (x0,y0) for each Image object, and performs some run-time checking.

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
            self.x0_list[k] = q[image.size] * rad2uas
            self.y0_list[k] = q[image.size+1] * rad2uas
            q = q[(image.size+2):]

            
    def intensity_map(self,x,y,parameters=None,verbosity=0) :
        """
        Returns the summed intensity map.

        Args:
          x (numpy.ndarray): Array of RA offsets in microarcseconds (usually plaid 2D)
          y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D)
          parameters (list): Parameter list. If none is provided, uses the current set of parameters. Default: None.
          verbosity (int): Verbosity parameter. If nonzero, prints information about model properties. Default: 0.

        Returns:
          (numpy.ndarray) Array of intensity values at positions (x,y).
        """

        if (not parameters is None) :
            self.generate(parameters)

        I = np.zeros(x.shape)
        q = self.parameters
        for k,image in enumerate(self.image_list) :
            dx = x-self.x0_list[k]
            dy = y-self.y0_list[k]

            if (self.parameters is None) :                
                I = I+image.intensity_map(dx,dy,verbosity=verbosity)
            else :
                I = I+image.intensity_map(dx,dy,parameters=q,verbosity=verbosity)

            q=self.parameters[(image.size+2):]

            if (verbosity>0) :
                print("   --> Sum shift:",self.x0_list[k],self.y0_list[k])
            
        return I
    
