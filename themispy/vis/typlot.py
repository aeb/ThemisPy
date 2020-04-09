###########################
#
# Package:
#   typlot
#
# Provides:
#   Provides general utility functions plotting.  Ambitiously named.
#



import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.optimize import bisect
from scipy.stats import gaussian_kde as KDEsci
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gs
from scipy.stats import percentileofscore


rc('font',**{'family':'serif','serif':['Times','Palatino','Computer Modern Roman']})
plt.rc('text', usetex=True)


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


def kde_plot_1d(x, limits=None, color='b', alpha=1.0, linewidth=1, linestyle='-', nbin=128, bw='scott', scott_factor=0, filled=False):
    """
    Creates a 1d joint posterior plot using :func:`scipy.stats.gaussian_kde` function.

    Args:
      x (numpy.array): List of data positions.
      limits (list): Limits in the form [xmin,xmax].  If set to None, will use the minimum and maximum values of x, expanded by 10% on each side.  Default: None.
      color (str,list): Any acceptable color type as specified in :mod:`matplotlib.colors`.
      alpha (float): Value of alpha for the individual likelihood traces. Default: 1.0.
      linewidth (float): Linewidth to be passed to :func:`matplotlib.pyplot.plot`. Default: 1.
      linestype (str): Linestyle to be passed to :func:`matplotlib.pyplot.plot`. Default: '-'.
      nbin (int): Number of bins on which to construct KDE result. Default: 128.
      bw (str): Bandwidth method for :func:`scipy.stats.gaussian_kde`. Overidden by nonzero scott_factor. Default: 'scott'.
      scott_factor (float): Factor by which to expand the standard `scott` bandwidth factor.  Overrides bw if nonzero.  Default: 0.
      filled (bool): If true, the histogram is plotted filled. Default: False.

    Returns:
      (matplotlib.lines.Line2D): A list of Line2D objects representing the plotted data, i.e., the handles returned by :func:`matplotlib.pyplot.plot`.
    """

    data = x

    if (scott_factor!=0) :
        bw = scott_factor*(float(x.size))**(-1.0/6.0)

    kde = KDEsci(x, bw_method=bw)
    if not limits:
        xmin = x.min()
        xmax = x.max()
        dx = xmax-xmin
        xmin = xmin-0.1*dx
        xmax = xmax+0.1*dx
    else:
        xmin = limits[0][0]
        xmax = limits[0][1]

    X = np.linspace(xmin,xmax,nbin)
    Z = kde(X)

    if (filled) :
        plt.fill_between(X,Z,color=color,alpha=0.25*alpha,linewidth=linewidth,linestyle=linestyle)
    h = plt.plot(X,Z,color=color,linewidth=linewidth,linestyle=linestyle,alpha=alpha)


    return h

def kde_plot_2d(x, y, plevels=None, limits=None, colormap='Purples', alpha=1.0, nbin=128, bw='scott', scott_factor=0):
    """
    Creates a 2d joint posterior plot using :func:`scipy.stats.gaussian_kde` function and :func:`matplotlib.pyplot.contourf`.

    Args:
      x (numpy.array): List of data positions in the 1st axis (horizontal).
      y (numpy.array): List of data positions in the 2nd axis (vertical).
      plevels (list): List of cummulative probability levels at which to draw contours. If set to None, will set to :math:`1\\sigma, 2\\sigma, 3\\sigma`. Default: None.
      limits (list): Limits in the form [[xmin,xmax],[ymin,ymax]].  If set to None, will use the minimum and maximum values of, expanded by 10% on each side.  Default: None.
      colormap (matplotlib.colors.Colormap): A colormap name as specified in :mod:`matplotlib.cm`. Default: 'Purples'.
      alpha (float): Value of alpha for the individual likelihood traces. Default: 1.0.
      nbin (int): Number of bins on which to construct KDE result in each dimension. Default: 128.
      bw (str): Bandwidth method for :func:`scipy.stats.gaussian_kde`. Overidden by nonzero scott_factor. Default: 'scott'.
      scott_factor (float): Factor by which to expand the standard `scott` bandwidth factor.  Overrides bw if nonzero.  Default: 0.

    Returns:
      (matplotlib.contour.QuadContourSet): The handles returned by :func:`matplotlib.pyplot.contourf`.
    """
    
    if (np.any(x.shape != y.shape)) :
        print("ERROR! kde_plot_2d requires that x and y have the same shape.")
        exit(1)

    data = np.vstack([x,y])

    if (plevels is None) :
        plevels = [1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)]

    if (scott_factor!=0) :
        bw = scott_factor*(float(x.size))**(-1.0/6.0)

    kde = KDEsci(data, bw_method=bw)
    if not limits:
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
    
    X, Y = np.mgrid[xmin:xmax:nbin*1j, ymin:ymax:nbin*1j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(kde(positions).T, X.shape)
    points = Z.reshape(-1)
    dX = X[1,1]-X[0,0]
    dY = Y[1,1]-Y[0,0]
    norm = points.sum()*dX*dY
    
    levels = []
    for p in plevels:
        target = p*norm
        levels.append(bisect(_cdf,points.min(),points.max(), args=(points,dX,dY,target)))

    h = plt.contourf(X,Y,np.log10(Z),np.log10(levels),vmin=np.log10(levels[0]/4.0),cmap=colormap,extend='max',alpha=alpha) 

    return h

def _find_limits(data,quantile=0.25,factor=1.5) :
    """
    Utility function to find limits given some quantile.

    Args:
      data (numpy.ndarray): Array of data values.
      quantile (float): Quantile to use for typical value. Default: 0.25.
      factor (float): Factor by which to expand the limit on both sides. Default: 1.5.

    Returns:
      (numpy.array): 1D array of [min,max] limits.
    """

    dm = np.median(data)
    dl = np.min(data)
    dh = np.max(data)

    lim = np.array([ dm+factor*(dl-dm), dm+factor*(dh-dm) ])

    return lim


def kde_triangle_plot(lower_data_array, upper_data_array=None, labels=None, upper_labels=None, truths=None, colormap='Blues', upper_colormap=None, color='blue', upper_color=None, truths_color='red', axis_location=None, alpha=1.0, quantiles=[0.99,0.9,0.5], nbin=128, linewidth=1, linestyle='-', truths_alpha=1.0, truths_linewidth=1, truths_linestype='-', scott_factor=1.41421, filled=False) :
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
      filled (bool): If true, the 1d histograms along the diagonal are plotted filled. Default: False.

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


            # Make 2d joint distribution plot
            lower_triangle_plot_handles[j,k] = kde_plot_2d(lower_data_array[:,j],lower_data_array[:,k],cmap=colormap,alpha=alpha,plevels=quantiles,limits=[limx,limy],nbin=nbin,scott_factor=scott_factor)
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
        
                # Find limits in x/y
                limx=_find_limits(lower_data_array[:,j],quantile=1-quantiles[0])
                limy=_find_limits(lower_data_array[:,k],quantile=1-quantiles[0])
                upper_limx = _find_limits(upper_data_array[:,j],quantile=1-quantiles[0])
                upper_limy = _find_limits(upper_data_array[:,k],quantile=1-quantiles[0])
                limx[0] = min(limx[0],upper_limx[0])
                limx[1] = max(limx[1],upper_limx[1])
                limy[0] = min(limy[0],upper_limy[0])
                limy[1] = max(limy[1],upper_limy[1])


                # Make 2d joint distribution plot
                upper_triangle_plot_handles[j,k] = kde_plot_2d(upper_data_array[:,j],upper_data_array[:,k],cmap=upper_colormap,alpha=alpha,plevels=quantiles,limits=[limx,limy],nbin=nbin,scott_factor=scott_factor)
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
                            plt.ylabel(upper_labels[k],rotation=-90)
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


    ########
    # Diagonal
    for k in range(nrow) :
        # Find axis location with various gutters, etc.
        x_window_start = axis_location[0] + k*(x_gutter+x_window_size)
        y_window_start = axis_location[1] + axis_location[3] - y_window_size - k*(y_gutter+y_window_size)
        plt.axes([x_window_start, y_window_start, x_window_size, y_window_size])

        # Find limits in x/y
        limx=_find_limits(lower_data_array[:,k],quantile=1-quantiles[0])
        if (upper_data_array is not None) :
            upper_limx = _find_limits(upper_data_array[:,k],quantile=1-quantiles[0])
            limx[0] = min(limx[0],upper_limx[0])
            limx[1] = max(limx[1],upper_limx[1])

        diagonal_plot_handles[k,0] = kde_plot_1d(lower_data_array[:,k],color=color,alpha=alpha,limits=limx,nbin=nbin,linewidth=linewidth,linestyle=linestyle,filled=filled)
        if (upper_data_array is not None) :
            diagonal_plot_handles[k,1] = kde_plot_1d(upper_data_array[:,k],color=upper_color,alpha=alpha,limits=limx,nbin=nbin,linewidth=linewidth,linestyle=linestyle,filled=filled)

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


    return (axes_handles,lower_triangle_plot_handles,upper_triangle_plot_handles,diagonal_plot_handles)



