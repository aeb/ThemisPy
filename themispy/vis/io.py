###########################
#
# Package:
#   io
#
# Provides:
#   Provides input/output support for model classes and functions.
#

import themispy
from themispy.utils import *
from themispy.vis.image import *
from themispy.vis.polimage import *

import numpy as np

# Read in h5py, if possible
try:
    import h5py
    h5py_found = True
except:
    warnings.warn("Package h5py not found.  Some functionality will not be available.  If this is necessary, please ensure h5py is installed.", Warning)
    h5py_found = False
    

# Some constants
rad2uas = 180.0/np.pi * 3600 * 1e6
uas2rad = np.pi/(180.0 * 3600 * 1e6)


def construct_model_image_from_glob(model_glob, glob_version, hyperparameters=None) :
    """
    Constructs a :class:`model_image` object given an appropriate model specification glob.  
    Currently accepted globs include:

    * ccm_mexico+ model specification string. These are an extended set of the strings used by the ccm_m87_mexico drivers.  For example, sXaagG.  For raster model types, hyperparameters must be supplied.  See :func:`expand_model_glob_ccm_mexico` for details regarding the required form.
    * model tag file name. These can be output by Themis directly and encompass a wider set of possible models.

    The appropriate associated glob version must be specified. For some specifications an additional set of hyperparameters may need to be supplied.

    Args:
      model_glob (str): Glob defining model image. Examples are 'sXaagG' or 'model_image.tag'
      tagversion (str): Version of tag system. Currently only tagvers-1.0 is implemented. Default 'tagvers-1.0'.
      hyperparameters (dict): Dictionary of hyperparameters if needed by the specification.

    Returns:
      (model_image): :class:`model_image` object of the corresponding model.

    """

    if (glob_version=='ccm_mexico' or glob_version=='ccm_mexico+') :
        return construct_model_image_from_ccm_mexico(model_glob,hyperparameters)

    elif (glob_version=='tagvers-1.0') :

        # Try to make a model iamge
        try :
            img = construct_model_image_from_tag_file(model_glob,glob_version)
        except RuntimeError as err :
            if ( not ("Unrecognized model tag" in format(err) ) ): 
                raise RuntimeError(err)
        else :
            return img

        # Try to make a model polarized image
        try :
            img = construct_model_polarized_image_from_tag_file(model_glob,glob_version)
        except RuntimeError as err :
            if ( not ("Unrecognized model tag" in format(err) ) ): 
                raise RuntimeError(err)
        else :
            return img


        # Return error
        raise RuntimeError("Unrecognized model tag %s"%(model_glob))
        
    else :
        raise RuntimeError("Unrecognized model_glob version: %s"%(version))



def write_fits(x,y,img,fits_filename,uvfits_filename=None,time=None,verbosity=0) :
    """
    Writes a FITS format image file given 2D image data.  

    Warning: 

      * This makes extensive use of ehtim and will not be available if ehtim is not installed.  Raises a NotImplementedError if ehtim is unavailable.
      * Due to ehtim, only *square* images are supported at this time.

    Args:
      x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
      y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
      I (numpy.ndarray,model_image): Either an array of intensity values in (Jy/uas^2) or a :class:`model_image` object.
      fits_filename (str): Name of output FITS file.
      uvfits_filename (str): Optional name of uvfits file with relevant header data.  Failing to provide this may result in unusable FITS files.
      time (float): Time in hr on the relevant observation day (set by uvfits file) represented by image data.
      verbosity (int): Verbosity parameter. If 0, no information is printed.  If >=1, prints information about sizes and values.  If >=2, generates debugging plots.
    """

    if (ehtim_found==False) :
        raise NotImplementedError("ERROR: write_fits requires ehtim to be installed.")

    I = None
    Q = None
    U = None
    V = None
    if (isinstance(img,model_polarized_image)) :
        if (time is None) :
            time = img.time
        x = -x # RA vs x
        S = img.stokes_map(x,y,kind='all',verbosity=verbosity)
        x = -x # RA vs x
        I = S[0]
        Q = S[1]
        U = S[2]
        V = S[3]

    elif (isinstance(img,model_image)) :
        if (time is None) :
            time = img.time
        x = -x # RA vs x
        I = img.intensity_map(x,y,verbosity=verbosity)
        x = -x # RA vs x
    elif (isinstance(img,np.ndarray)) :
        if (time is None) :
            time = 0
        I = img
    elif (isinstance(img,list)) :
        if (time is None) :
            time = 0
        I=img[0]
        Q=img[1]
        U=img[2]
        V=img[3]
            
    else :
        raise NotImplementedError("ERROR: Requies either an image or numpy.ndarray of values.")

    if (verbosity>0) :
        print("Shapes",x.shape,y.shape,I.shape)

    if (I.shape[0]!=I.shape[1]) :
        raise RuntimeError("ERROR: ehtim cannot handle non-square images. Your image had shape",I.shape)

    if (verbosity>2) :
        plt.figure(figsize=(5,5))
        plt.axes([0.15,0.15,0.8,0.8])
        plt.pcolor(x,y,I,cmap='afmhot')
    
    uas2rad = 1e-6 / 3600.0 * np.pi/180.0
    pixel_size = abs(x[1,1]-x[0,0])*uas2rad
    Ippx = np.transpose(I) * abs((x[1,1]-x[0,0])*(y[1,1]-y[0,0]))
    Ippx = np.flipud(Ippx)

    if (not Q is None) :
        Qppx = np.flipud( np.transpose(Q) * abs((x[1,1]-x[0,0])*(y[1,1]-y[0,0])) )
        Uppx = np.flipud( np.transpose(U) * abs((x[1,1]-x[0,0])*(y[1,1]-y[0,0])) )
        Vppx = np.flipud( np.transpose(V) * abs((x[1,1]-x[0,0])*(y[1,1]-y[0,0])) )
        
    if (uvfits_filename is None) :
        warnings.warn("No uvfits file has been specified. This will probably result in an nonfunctional FITS header.", Warning)
        ra=0.0
        dec=0.0
        rf=230e9
        src='NA'
        mjd=58583
    else :
        obs = eh.obsdata.load_uvfits(uvfits_filename)
        ra=obs.ra
        dec=obs.dec
        rf=obs.rf
        src=obs.source
        mjd=obs.mjd

    if (verbosity>0) :
        print('pixel size:',pixel_size)
        print('ra:',ra)
        print('dec:',ra)
        print('rf:',ra)
        print('src:',ra)
        print('mjd:',ra)

    if (Q is None) :
        if (verbosity>0) :
            print('Saving Stokes I')
        image = eh.image.Image(Ippx,pixel_size,ra,dec,rf=rf,source=src,mjd=mjd,time=time,pulse=eh.observing.pulses.deltaPulse2D)
    else :
        if (verbosity>0) :
            print('Saving Stokes I,Q,U,V')
        image = eh.image.Image(Ippx,pixel_size,ra,dec,rf=rf,source=src,mjd=mjd,time=time,pulse=eh.observing.pulses.deltaPulse2D,polrep='stokes')
        image.add_qu(Qppx,Uppx)
        image.add_v(Vppx)

    if (verbosity>1) :
        image.display()
        plt.show()
    
    image.save_fits(fits_filename)
    

def write_hdf5(x,y,t,mov,hdf5_filename,uvfits_filename=None,time_offset=None,method='ehtim',verbosity=0) :
    """
    Writes an HDF5 format movie file given a sequence of 2D image data

    Warning: 

      * Requires ehtim or h5py to be installed.  Raises a NotImplementedError if neither are unavailable
      * If these will be read in by ehtim later, only *square* images are permitted.  Raises a runtime warning if non-square images are requested.
      

    Args:
      x (numpy.ndarray): Array of -RA offsets in microarcseconds (usually plaid 2D).
      y (numpy.ndarray): Array of Dec offsets in microarcseconds (usually plaid 2D).
      t (list): Array of times of frames
      mov (list,model_image): Either a list of arrays of intensity values in (Jy/uas^2) computed at the times in t or a :class:`model_image` object.
      fits_filename (str): Name of output FITS file.
      uvfits_filename (str): Optional name of uvfits file with relevant header data.  Failing to provide this may result in unusable FITS files.
      time_offset (float): Time offset in hr on the relevant observation day (set by uvfits file) represented by image data.
      method (str): Method to use to construct and format hdf5 file.  Options are 'ehtim' or 'h5py', which make use of the ehtim functions or provide a separate hdf5 format, respectively. Default: 'ehtim'.
      verbosity (int): Verbosity parameter. If 0, no information is printed.  If >=1, prints information about sizes and values.  If >=2, generates debugging plots.
    """

    if (ehtim_found==False and h5py_found==False) :
        raise NotImplementedError("ERROR: write_hdf5 requires ehtim or h5py to be installed.")

    time_list=[]
    img_list=[]

    pb = progress_bar("Writing hdf5",length=40)
    pb.start()

    # Add check for time step uniformity
    
    
    if (isinstance(mov,model_image)) :
        for k,time in enumerate(t) :
            pb.increment(k/float(len(t)))
            mov.set_time(time)
            I = mov.intensity_map(x,y,verbosity=verbosity)
            time_list.append(time)
            img_list.append(I)
    elif (isinstance(mov,list)) :
        time_list = t
        img_list = mov
    else :
        raise NotImplementedError("ERROR: Requies either an image or list of numpy.ndarray of values.")

    pb.finish("--> %s"%(hdf5_filename))

    # Run over and fix units
    for k in range(len(time_list)) :

        if (not (time_offset is None) ) :
            time_list[k] = time_list[k] + time_offset

        
        if (verbosity>0) :
            print("Shapes",x.shape,y.shape,I.shape)

        if (img_list[k].shape[0]!=img_list[k].shape[1]) :
            raise warnings.warn("ehtim cannot handle non-square images. Your image had shape [%i,%i]"%(img_list[k].shape[0],img_list[k].shape[1]), Warning)
    
        uas2rad = 1e-6 / 3600.0 * np.pi/180.0
        pixel_size = abs(x[1,1]-x[0,0])*uas2rad
        Ippx = np.transpose(img_list[k]) * abs((x[1,1]-x[0,0])*(y[1,1]-y[0,0]))
        Ippx = np.flipud(Ippx)
        img_list[k] = Ippx

    # Check for uvfits files
    if (uvfits_filename is None) :
        warnings.warn("No uvfits file has been specified. This will probably result in an nonfunctional FITS header.", Warning)
        ra=0.0
        dec=0.0
        rf=230e9
        src='NA'
        mjd=58583
    else :
        obs = eh.obsdata.load_uvfits(uvfits_filename)
        ra=obs.ra
        dec=obs.dec
        rf=obs.rf
        src=obs.source
        mjd=obs.mjd

    if (verbosity>0) :
        print('pixel size:',pixel_size)
        print('ra:',ra)
        print('dec:',ra)
        print('rf:',ra)
        print('src:',ra)
        print('mjd:',ra)

    if (method=='ehtim') :
        # Create movie object
        movie = eh.movie.Movie(img_list,time_list,pixel_size,ra,dec,rf=rf,source=src,mjd=mjd,pulse=eh.observing.pulses.deltaPulse2D,polrep='stokes')
        # Save hdf5 file
        movie.save_hdf5(hdf5_filename)

    elif (method=='h5py') :
        # Open hdf5 file
        hdf5out = h5py.File(hdf5_filename,'w')

        # Output some standard info in a header
        hdf5out['pixel_size'] = pixel_size
        hdf5out['ra'] = ra
        hdf5out['dec'] = dec
        hdf5out['rf'] = rf
        hdf5out['source'] = src
        hdf5out['mjd'] = mjd
        hdf5out['polrep'] = 'stokes'

        # Output the images and time stamps
        hdf5out['times'] = np.array(time_list)
        hdf5out['frames'] = np.array(img_list)

        # Output some specific headers
        hdf5out['aspect'] = 'square'
        hdf5out['pixel size units'] = 'rad'
        hdf5out['intensity units'] = 'Jy/px'
        hdf5out['origin'] = 'ThemisPy %s'%(themispy.__version__)
        
        # Close
        hdf5out.close()


        
