###########################
#
# Package:
#   typlot
#
# Provides:
#   Provides general utility functions plotting.  Ambitiously named.
#

from themispy.utils import *

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import rc, cm, rcParams
from matplotlib.colors import to_rgb, LogNorm, ListedColormap

from scipy.special import logit
from scipy import interpolate as scint
from scipy.optimize import bisect
from scipy.stats import gaussian_kde as KDEsci
from scipy.stats import percentileofscore

# Adjust the negative linestyle
rcParams["contour.negative_linestyle"] = 'solid'


def generate_streamline_colormap(colormap='plasma', number_of_streams=None, oversample_factor=4, fluctuation_factor=0.2) :
    """
    Generates a new colormap that is generated by randomly subsampling a given colormap.
    Useful for making plots that indicate flow, e.g., :func:`plot_deo_tempering_level_evolution`.
    
    Args:
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'plasma'.
    
    Returns:
      (matplotlib.colors.Colormap): A new colormap that has random rearrangements.
    """

    # Obtain colormap object
    cmap = cm.get_cmap(colormap)

    # Generate some value levels randomly bewteen 0 and 1
    if (number_of_streams is None) :
        number_of_streams = cmap.N
    vsub = np.linspace(0,1,number_of_streams) + fluctuation_factor*np.random.random(number_of_streams)
    vsub = (vsub-np.min(vsub))/(np.max(vsub)-np.min(vsub))

    # Generate an over-sampled smooth interpolated set of values
    v = scint.interp1d(np.linspace(0,1,number_of_streams),vsub,kind='cubic')(np.linspace(0,1,oversample_factor*number_of_streams))
    cvals = cmap(v)

    return ( ListedColormap(cvals) )


def generate_monocolormap(color, N=256, basecolor='w', reverse=False) :
    """
    Generates a new colormap that interpolates linearly from the basecolor to a given color.

    Args:
      color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`.
      N (int): Number of levels in the color map. Default: 256.
      basecolor (str,list): Base color (i.e., at vmin in colormap). Accepts any acceptable color type as specified in :mod:`matplotlib.colors`. Default: 'w'.
      reverse (bool): Reverses the colormap.
    
    Returns:
      (matplotlib.colors.Colormap): A new colormap that has random rearrangements.
    """

    # Get colors
    c0 = np.array(to_rgb(basecolor))
    c1 = np.array(to_rgb(color))

    # Generate space for a list of colors
    cvals = np.zeros((N,3))
    for k,t in enumerate(np.linspace(0,1,N)) :
        cvals[k,:] = c0*(1-t)+c1*t

    if (reverse) :
        cvals = np.flip(cvals,axis=1)
        
    return ( ListedColormap(cvals) )




def axes_adjust(fig=None, axes=None, shift_x0=0, shift_y0=0, shift_width=0, shift_height=0) :
    """
    Adjusts the axes by some shifts provided in the specified units.

    Args:
      fig (matplotlib.figure.Figure): Handle of figure to modify. If None, use the current figure object. Default: None.
      axes (matplotlib.axes.Axes): Handle of axes to modify. If None, use the current axes object. Default: None.
      shift_x0 (float): Number of inches to shift the axes to the right. Default: 0.
      shift_y0 (float): Number of inches to shift the axes up. Default: 0.
      shift_width (float): Number of inches to increase the axes width. Default: 0.
      shift_height (float): Number of inches to increase the axes height. Default: 0.

    Returns:
      None
    """

    if (fig is None) :
        fig = plt.gcf()
    if (axes is None) :
        axes = plt.gca()

    figsize = fig.get_size_inches()
    current_location = plt.gca().get_position()
    axes.set_position([ current_location.x0+shift_x0/figsize[0], current_location.y0+shift_y0/figsize[1], current_location.width+shift_width/figsize[0], current_location.height+shift_height/figsize[1] ])
    

def _cdf(limit,H,dx,dy,target):
    """
    Function for the CDF in terms of the probability density, for computing the fixed CDF probability levels.

    Args:
      limit (numpy.ndarray): Array of limit value at which the CDF is desired.
      H (nump.ndarray): 2D array of probability data
      dx (float): Differential in 1st axis
      dy (float): Differential in 2nd axis
      target

    Returns:
      (float): CDF above the limit minus the target value.
    """
    w = (H>limit)
    count = H[w]
    return count.sum()*dx*dy-target


def _logit_kde(x, limits, bw) :
    """
    Constructs a kde that uses the logit transform to to ensure the 
    transform is in the (limits[0],limits[1]) region. 

    Args:
        x(numpy.ndarray): Array of samples to compute the kde for.
        limits(list): Array with the limits for the samples
        bw: Bandwidth to use. See scipy.stats.gaussian_kde for useable bw's
    Returns:
        closure to the kde function with the transform applied
    """
    _x = x[(x<limits[1])*(x>limits[0])]
    _u = (_x - limits[0])/(limits[1]-limits[0])
    _t = logit(_u)
    _kde = KDEsci(_t, bw_method=bw)
    _du = 1/(limits[1]-limits[0])

    # Now define a closure because we can and I like functional programming.
    def kde_trans(dx):
        u = (dx - limits[0])/(limits[1]-limits[0])
        f = _kde(logit(u))
        # Jacobian factor with floor for numerical reasons
        dt = np.abs(1/(u*(1.0-u)+1e-20))*_du
        return f*dt

    return kde_trans




def kde_plot_1d(x, limits=None, color='b', alpha=1.0, linewidth=1, linestyle='-', nbin=128, bw='scott', scott_factor=0, filled=False, transform=False):
    """
    Creates a 1d joint posterior plot using :func:`scipy.stats.gaussian_kde` function.

    Args:
      x (numpy.ndarray): List of data positions.
      limits (list): Limits in the form [xmin,xmax].  If set to None, will use the minimum and maximum values of x, expanded by 10% on each side.  Default: None.
      color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`.
      alpha (float): Value of alpha for the individual likelihood traces. Default: 1.0.
      linewidth (float): Linewidth to be passed to :func:`matplotlib.pyplot.plot`. Default: 1.
      linestype (str): Linestyle to be passed to :func:`matplotlib.pyplot.plot`. Default: '-'.
      nbin (int): Number of bins on which to construct KDE result. Default: 128.
      bw (str): Bandwidth method for :func:`scipy.stats.gaussian_kde`. Overidden by nonzero scott_factor. Default: 'scott'.
      scott_factor (float): Factor by which to expand the standard `scott` bandwidth factor.  Overrides bw if nonzero.  Default: 0.
      filled (bool): If true, the histogram is plotted filled. Default: False.
      transform (bool): Whether to use a logit transformed version of the KDE to prevent leakage out of the expected range. Default False.

    Returns:
      (matplotlib.lines.Line2D): A list of Line2D objects representing the plotted data, i.e., the handles returned by :func:`matplotlib.pyplot.plot`.
    """

    data = x

    if (scott_factor!=0) :
        bw = scott_factor*(float(x.size))**(-1.0/6.0)

    if (limits is None):
        xmin = x.min()
        xmax = x.max()
        dx = xmax-xmin
        xmin = xmin-0.1*dx
        xmax = xmax+0.1*dx
    else:
        xmin = limits[0]
        xmax = limits[1]

    if (transform) :
        kde = _logit_kde(data, [xmin,xmax], bw)
    else :
        kde = KDEsci(data, bw_method=bw)

    X = np.linspace(xmin,xmax,nbin)
    Z = kde(X)

    if (filled) :
        plt.fill_between(X,Z,color=color,alpha=0.25*alpha,linewidth=linewidth,linestyle=linestyle)
    h = plt.plot(X,Z,color=color,linewidth=linewidth,linestyle=linestyle,alpha=alpha)

    return h


def _logit_kde2d(x,y, lims, bw) :
    """
    Constructs a kde that uses the logit transform to to ensure the 
    transform is in the (limits[0],limits[1]) region. 

    Args:
        x(numpy.ndarray): x dimension array of samples to compute the kde.
        y(numpy.ndarray): y dimension array of samples to compute the kde.
        limits(list): Array of tuples with the limits for the samples
        bw: Bandwidth to use. See scipy.stats.gaussian_kde for useable bw's
    Returns:
        closure to the kde function with the transform applied
    """
    cond =(x<lims[0][1])*(x>lims[0][0])*(y<lims[1][1])*(y>lims[1][0]) 
    _x = x.copy()[cond]
    _y = y.copy()[cond]
    _u = (_x - lims[0][0])/(lims[0][1]-lims[0][0])
    _v = (_y - lims[1][0])/(lims[1][1]-lims[1][0])
    _t = logit(_u)
    _s = logit(_v)


    _kde = KDEsci(np.vstack([_t,_s]), bw_method=bw)
    _du = 1.0/(lims[0][1]-lims[0][0])
    _dv = 1.0/(lims[1][1]-lims[1][0])

    # Now define a closure because we can and I like functional programming.
    def kde_trans2d(data):
        dx = data[0,:]
        dy = data[1,:]
        u = (dx - lims[0][0])/(lims[0][1]-lims[0][0])
        v = (dy - lims[1][0])/(lims[1][1]-lims[1][0])
        tdata = np.vstack((logit(u), logit(v)))
        f = _kde(tdata)
        f_nan = np.isnan(f)
        f[f_nan] = 0.0
        # Jacobian factor with floor for numerical reasons
        dt = np.abs(1/(u*(1.0-u)+1e-8))*_du*np.abs(1/(v*(1.0-v)+1e-8))*_dv
        return f*dt

    return kde_trans2d

def kde_plot_2d(x, y, plevels=None, limits=None, colormap='Purples', alpha=1.0, nbin=128, bw='scott', scott_factor=0, transform=False, fill=True, edges=False, fill_zorder=None, edge_zorder=None, edge_colors=None, edge_colormap=None, edge_alpha=None, linewidth=1):
    """
    Creates a 2d joint posterior plot using :func:`scipy.stats.gaussian_kde` function and :func:`matplotlib.pyplot.contourf`.

    Args:
      x (numpy.ndarray): List of data positions in the 1st axis (horizontal).
      y (numpy.ndarray): List of data positions in the 2nd axis (vertical).
      plevels (list): List of cummulative probability levels at which to draw contours. If set to None, will set to :math:`1\\sigma, 2\\sigma, 3\\sigma`. Default: None.
      limits (list): Limits in the form [[xmin,xmax],[ymin,ymax]].  If set to None, will use the minimum and maximum values of, expanded by 10% on each side.  Default: None.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'Purples'.
      alpha (float): Value of alpha for the filled contours. Default: 1.0.
      nbin (int): Number of bins on which to construct KDE result in each dimension. Default: 128.
      bw (str): Bandwidth method for :func:`scipy.stats.gaussian_kde`. Overidden by nonzero scott_factor. Default: 'scott'.
      scott_factor (float): Factor by which to expand the standard `scott` bandwidth factor.  Overrides bw if nonzero.  Default: 0.
      transform (bool): Whether to use a logit transformed version of the KDE to prevent leakage out of the expected range. Default False.
      fill (bool): Determines if contour levels will be filled. Default: True.
      fill_zorder (int): Sets zorder of filled contours. Default: None.
      edges (bool): Deterines if countour lines will be plotted. Default: False.
      edge_zorder (int): Sets zorder of contour lines. Default: None.
      edge_colors (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors` or list of such types. If not None, overrides edge_colormap. Default: None.
      edge_colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. If None, uses the colormap passed by the colormap option. Default: None.
      edge_alpha (float): Value of alpha for contour lines. Only meaningful if edges=True. If None, sets the contour line alpha to that passed by alpha. Default: None.
      linewidth (float): Width of contour lines. Default: 1.

    Returns:
      (matplotlib.contour.QuadContourSet,list): The handles returned by :func:`matplotlib.pyplot.contourf` and :func:`matplotlib.pyplot.contour` depending on the fill and edges options.
    """
    
    if (np.any(x.shape != y.shape)) :
        print("ERROR! kde_plot_2d requires that x and y have the same shape.")
        exit(1)

    data = np.vstack([x,y])

    if (plevels is None) :
        plevels = [1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)]

    if (scott_factor!=0) :
        bw = scott_factor*(float(x.size))**(-1.0/6.0)

    if (limits is None):
        xmin = x.min()
        xmax = x.max()
        dx = xmax-xmin
        xmin = xmin-0.1*dx
        xmax = xmax+0.1*dx
        ymin = y.min()
        ymax = y.max()
        dy = ymax-ymin
        ymin = ymin-0.1*dy
        ymax = ymax+0.1*dy
    else:
        xmin = limits[0][0]
        xmax = limits[0][1]
        
        ymin = limits[1][0]
        ymax = limits[1][1]
    if transform :
        kde = _logit_kde2d(x,y, [(xmin,xmax),(ymin,ymax)], bw)
    else :
        kde = KDEsci(data, bw_method=bw)
    
    X, Y = np.mgrid[xmin:xmax:nbin*1j, ymin:ymax:nbin*1j]
    X = X[1:-1,1:-1]
    Y = Y[1:-1,1:-1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    Z = np.reshape(kde(positions).T, X.shape)
    points = Z.reshape(-1)
    dX = X[1,1]-X[0,0]
    dY = Y[1,1]-Y[0,0]
    norm = points.sum()*dX*dY
    
    levels = []
    for p in plevels:
        target = p*norm
        levels.append(bisect(_cdf,points.min(),points.max(), args=(points,dX,dY,target), xtol=points.max()*1e-10))

    if (fill) :
        hf = plt.contourf(X,Y,np.log10(Z),np.log10(levels),vmin=np.log10(levels[0]/4.0),cmap=colormap,extend='max',alpha=alpha,zorder=fill_zorder) 
    else :
        hf = None

    if (edges) :

        if (not edge_colors is None) :
            edge_colormap = None
        elif (edge_colormap is None) :
            edge_colormap = colormap
            
        if (edge_alpha is None) :
            edge_alpha = alpha
            
        he = plt.contour(X,Y,np.log10(Z),np.log10(levels),vmin=np.log10(levels[0]/4.0),cmap=edge_colormap,colors=edge_colors,extend='max',alpha=edge_alpha,zorder=edge_zorder,linewidths=linewidth,linestyles='-') 
    else :
        he = None
        

    if (he is None) :
        return hf
    elif (hf is None) :
        return he
    else :
        return [hf,he]


def _find_limits(data,quantile=0.25,factor=1.5) :
    """
    Utility function to find limits given some quantile.

    Args:
      data (numpy.ndarray): Array of data values.
      quantile (float): Quantile to use for typical value. Default: 0.25.
      factor (float): Factor by which to expand the limit on both sides. Default: 1.5.

    Returns:
      (numpy.ndarray): 1D array of [min,max] limits.
    """

    dm = np.median(data)
    dl = np.min(data)
    dh = np.max(data)

    lim = np.array([ dm+factor*(dl-dm), dm+factor*(dh-dm) ])

    return lim


def kde_triangle_plot(lower_data_array, upper_data_array=None, limits=None, transform=False, labels=None, upper_labels=None, truths=None, colormap='Blues', upper_colormap=None, color='blue', upper_color=None, truths_color='red', axis_location=None, alpha=1.0, quantiles=[0.99,0.9,0.5], nbin=128, linewidth=1, linestyle='-', truths_alpha=1.0, truths_linewidth=1, truths_linestype='-', scott_factor=1.41421, filled=False, grid=True) :
    """
    Produces a triangle plot with contours set by CDF.  Data may be plotted in both lower (required) and upper (optional) triangles.

    Args:
      lower_data_array (numpy.ndarray): Array of multidimensional data from which to generate triangle plot.
      upper_data_array (numpy.ndarray): Optional array of multidimensional data from which to generate triangle plot. If provided, must match the dimensions of the lower_data_array.  Default: None.
      labels (list): Optional list of str labels for the passed values to place on lower-triangle-plot axes. If provided, must match the dimensions of the lower_data_array. Default: None.
      upper_labels (list): Optional list of str labels for the passed values to place on upper-triangle-plot axes. If provided, must match the dimensions of the lower_data_array. Default: None.
      truths (list): Optional list of truth values. If provided, must match the dimensions of the lower_data_array. Default: None.
      colormap (matplotlib.colors.Colormap): Colormap name as specified in :mod:`matplotlib.cm`. Default: 'Blues'.
      upper_colormap (matplotlib.colors.Colormap): Colormap name as specified in :mod:`matplotlib.cm`. If set to None, uses the same colormap as the lower triangle. Default: None.
      color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors` to be used in the 1d histograms along the diagonal for the lower data. Default: 'blue'.
      upper_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors` to be used in the 1d histograms along the diagonal for the upper data. If set to None, uses the same color as the lower triangle Default: None.
      truths_color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors` to be used in the 1d histograms along the diagonal for the upper data. If set to None, uses the same color as the lower triangle Default: 'red'.
      axis_location (list): List of axis location in which to place bounding box of the entire plot. If set to None, a default size will be set. Default: None.
      alpha (float): Value of alpha for the individual likelihood traces. Default: 1.0.
      quantiles (list): List of quantiles at which to draw contours. Default: 0.99, 0.9, 0.5. Note that the defaults will change if set!
      nbin (int): Number of bins on which to construct KDE result in each dimension. Default: 128.
      linewidth (float): Linewidth to be passed to :func:`matplotlib.pyplot.plot` for diagonal 1d histograms. Default: 1.
      linestype (str): Linestyle to be passed to :func:`matplotlib.pyplot.plot` for diagonal 1d histograms. Default: '-'.
      truths_alpha (float): Value of alpha for the individual likelihood traces. Default: 1.0.
      truths_linewidth (float): Linewidth to be passed to :func:`matplotlib.pyplot.plot` for diagonal 1d histograms of lower data. Default: 1.
      truths_linestype (str): Linestyle to be passed to :func:`matplotlib.pyplot.plot` for diagonal 1d histograms of lower data. Default: '-'.
      scott_factor (float): Factor by which to expand the standard `scott` bandwidth factor.  Overrides bw if nonzero.  Default: 1.41421.
      filled (bool): If True, the 1d histograms along the diagonal are plotted filled. Default: False.
      grid (bool): If True, adds gridlines to plots. Default: True.

    Returns:
      (matplotlib.axes.Axes, list, list, list): Handles to the array of axes objects in the plot, list of handels to the 2d kde plot objects in the lower triangle, 2d kde plot objects in the upper triangle, and 1d plot objects in along the diagonal (see :func:`kde_plot_2d` and :func:`kde_plot_1d` for more information).
    """
    
    figsize = plt.gcf().get_size_inches()

    if (labels is None) :
        labels = ['']*lower_data_array.shape[1]

    if (axis_location is None) :
        lmarg = 0.625 # Margin in inches
        rmarg = 0.625 # Margin in inches
        tmarg = 0.625 # Margin in inches
        bmarg = 0.625 # Margin in inches
        ax0 = lmarg/figsize[0]
        ay0 = bmarg/figsize[1]
        axw = (figsize[0]-lmarg-rmarg)/figsize[0]
        ayw = (figsize[1]-tmarg-bmarg)/figsize[1]
        axis_location = [ax0, ay0, axw, ayw]

    # Number of rows/columns
    nrow = lower_data_array.shape[1]

    # Get window size details
    gutter_size = 0.0625 # Gutter in inches
    x_gutter = gutter_size/figsize[0]
    y_gutter = gutter_size/figsize[1]
    x_window_size = (axis_location[2] - (nrow-1)*x_gutter)/float(nrow)
    y_window_size = (axis_location[3] - (nrow-1)*y_gutter)/float(nrow)

    # Plot handles
    lower_triangle_plot_handles = {}
    upper_triangle_plot_handles = {}
    diagonal_plot_handles = {}
    axes_handles = {}


    if (len(labels)!=lower_data_array.shape[1]) :
        print("ERROR! kde_triangle_plot : Number of labels is inconsistent with the number of arrays of data passed.")
        exit(1)

    if (truths is not None) :
        if (len(truths)!=lower_data_array.shape[1]) :
            print("ERROR! kde_triangle_plot : Number of truths is inconsistent with the number of arrays of data passed.")
            exit(1)
            
    if (upper_data_array is not None) :
        if (upper_data_array.shape[1]!=lower_data_array.shape[1]) :
            print("ERROR! kde_triangle_plot : Number of arrays of data on upper and lower triangles must be the same.")
            exit(1)
            

    ########
    # Lower triangle
    for k in range(nrow) :
        for j in range(k) :
            # Find axis location with various gutters, etc.
            x_window_start = axis_location[0] + j*(x_gutter+x_window_size)
            y_window_start = axis_location[1] + axis_location[3] - y_window_size - k*(y_gutter+y_window_size)
            plt.axes([x_window_start, y_window_start, x_window_size, y_window_size])
        
            if limits is None:
                # Find limits in x/y
                limx=_find_limits(lower_data_array[:,j],quantile=1-quantiles[0])
                limy=_find_limits(lower_data_array[:,k],quantile=1-quantiles[0])
                if (upper_data_array is not None) :
                    upper_limx = _find_limits(upper_data_array[:,j],quantile=1-quantiles[0])
                    upper_limy = _find_limits(upper_data_array[:,k],quantile=1-quantiles[0])
                    limx[0] = min(limx[0],upper_limx[0])
                    limx[1] = max(limx[1],upper_limx[1])
                    limy[0] = min(limy[0],upper_limy[0])
                    limy[1] = max(limy[1],upper_limy[1])
            else:
                limx = limits[j]
                limy = limits[k]



            # Make 2d joint distribution plot
            lower_triangle_plot_handles[j,k] = kde_plot_2d(lower_data_array[:,j],lower_data_array[:,k],colormap=colormap,alpha=alpha,plevels=quantiles,limits=[limx,limy],nbin=nbin,scott_factor=scott_factor, transform=transform)
            axes_handles[j,k] = plt.gca()

            # Fix up tickmarks
            plt.gca().tick_params(labelleft=False, labelright=False, labelbottom=False, labeltop=False, bottom=True, top=True, left=True, right=True, direction='in')

            # Add labels if appropriate
            if (k==nrow-1) :
                plt.gca().tick_params(labelbottom=True)
                if (labels is not None) :
                    plt.xlabel(labels[j])
            if (j==0) :
                plt.gca().tick_params(labelleft=True)
                if (labels is not None) :
                    if (j<k) :
                        plt.ylabel(labels[k])

            # Add truths if requested
            if (truths is not None) :
                xlim=plt.xlim()
                ylim=plt.ylim()
                plt.axvline(truths[j],color=truths_color,lw=truths_linewidth,alpha=truths_alpha)
                if (j<k) :
                    plt.axhline(truths[k],color=truths_color,lw=truths_linewidth,alpha=truths_alpha)
                    plt.plot([truths[j]],[truths[k]],'s',color=truths_color)
                plt.xlim(xlim)
                plt.ylim(ylim)

            # Add grid if requested
            if (grid) :
                plt.grid(alpha=0.5)


    ########
    # Upper triangle
    if (upper_data_array is not None) :
        if (upper_colormap is None) :
            upper_colormap = colormap
        if (upper_labels is None) :
            upper_labels = labels

        for k in range(nrow) :
            for j in range(k+1,nrow) :
                # Find axis location with various gutters, etc.
                x_window_start = axis_location[0] + (j+0)*(x_gutter+x_window_size)
                y_window_start = axis_location[1] + axis_location[3] - y_window_size - k*(y_gutter+y_window_size)
                plt.axes([x_window_start, y_window_start, x_window_size, y_window_size])
            
                if limits is None:
                    # Find limits in x/y
                    limx=_find_limits(lower_data_array[:,j],quantile=1-quantiles[0])
                    limy=_find_limits(lower_data_array[:,k],quantile=1-quantiles[0])
                    upper_limx = _find_limits(upper_data_array[:,j],quantile=1-quantiles[0])
                    upper_limy = _find_limits(upper_data_array[:,k],quantile=1-quantiles[0])
                    limx[0] = min(limx[0],upper_limx[0])
                    limx[1] = max(limx[1],upper_limx[1])
                    limy[0] = min(limy[0],upper_limy[0])
                    limy[1] = max(limy[1],upper_limy[1])
                else:
                    limx = limits[j]
                    limy = limits[k]
        


                # Make 2d joint distribution plot
                upper_triangle_plot_handles[j,k] = kde_plot_2d(upper_data_array[:,j],upper_data_array[:,k],colormap=upper_colormap,alpha=alpha,plevels=quantiles,limits=[limx,limy],nbin=nbin,scott_factor=scott_factor, transform=transform)
                axes_handles[j,k] = plt.gca()

                # Fix up tickmarks
                plt.gca().tick_params(labelleft=False, labelright=False, labelbottom=False, labeltop=False, bottom=True, top=True, left=True, right=True, direction='in')

                # Add labels if appropriate
                if (k==0) :
                    plt.gca().tick_params(labeltop=True)
                    if (upper_labels is not None) :
                        plt.xlabel(upper_labels[j])
                        plt.gca().xaxis.set_label_position('top') 
                if (j==nrow-1) :
                    plt.gca().tick_params(labelright=True)
                    if (upper_labels is not None) :
                        if (j>k) :
                            plt.ylabel(upper_labels[k],rotation=90)
                            plt.gca().yaxis.set_label_position('right')

                # Add truths if requested
                if (truths is not None) :
                    xlim=plt.xlim()
                    ylim=plt.ylim()
                    plt.axvline(truths[j],color=truths_color,lw=truths_linewidth,alpha=truths_alpha)
                    if (j>k) :
                        plt.axhline(truths[k],color=truths_color,lw=truths_linewidth,alpha=truths_alpha)
                        plt.plot([truths[j]],[truths[k]],'s',color=truths_color)
                    plt.xlim(xlim)
                    plt.ylim(ylim)

                # Add grid if requested
                if (grid) :
                    plt.grid(alpha=0.5)

    ########
    # Diagonal
    for k in range(nrow) :
        # Find axis location with various gutters, etc.
        x_window_start = axis_location[0] + k*(x_gutter+x_window_size)
        y_window_start = axis_location[1] + axis_location[3] - y_window_size - k*(y_gutter+y_window_size)
        plt.axes([x_window_start, y_window_start, x_window_size, y_window_size])

        if limits is None:
            # Find limits in x/y
            limx=_find_limits(lower_data_array[:,k],quantile=1-quantiles[0])
            if (upper_data_array is not None) :
                upper_limx = _find_limits(upper_data_array[:,k],quantile=1-quantiles[0])
                limx[0] = min(limx[0],upper_limx[0])
                limx[1] = max(limx[1],upper_limx[1])
        else:
            limx = limits[k]
        # Find limits in x/y
        
        diagonal_plot_handles[k,0] = kde_plot_1d(lower_data_array[:,k],color=color,alpha=alpha,limits=limx,nbin=nbin,linewidth=linewidth,linestyle=linestyle,filled=filled, transform=transform)
        if (upper_data_array is not None) :
            diagonal_plot_handles[k,1] = kde_plot_1d(upper_data_array[:,k],color=upper_color,alpha=alpha,limits=limx,nbin=nbin,linewidth=linewidth,linestyle=linestyle,filled=filled, transform=transform)

        axes_handles[k,k] = plt.gca()

                
        ylims=plt.ylim()
        plt.ylim((0,ylims[1]))
        plt.xlim(limx)

        plt.gca().tick_params(labelleft=False, labelright=False, labelbottom=False, labeltop=False, bottom=True, top=True, left=True, right=True, direction='in')

        if (k==nrow-1) :
            plt.gca().tick_params(labelbottom=True)
            if (labels is not None) :
                plt.xlabel(labels[k])
        

        if (truths is not None) :

            xlim=plt.xlim()
            ylim=plt.ylim()
            plt.axvline(truths[k],color=truths_color,lw=truths_linewidth,alpha=truths_alpha)
            plt.xlim(xlim)
            plt.ylim(ylim)


        # Add grid if requested
        if (grid) :
            plt.grid(alpha=0.5)

            
    return (axes_handles,lower_triangle_plot_handles,upper_triangle_plot_handles,diagonal_plot_handles)



