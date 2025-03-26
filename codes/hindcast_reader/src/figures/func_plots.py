# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:35:44 2025

@author: gregoires
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interpn


def plot_histogram(ax: mpl.axes.Axes, data: np.ndarray, 
                   vmin: float, vmax: float, dBins: float=None) -> None:
    '''
    Wrapper to plot an histogram in a given axis.

    Parameters
    ----------
    ax : mpl.axes.Axes
        Axis where the histogram should be displayed.
    data : np.ndarray
        Data to be displayed as an histogram.
    vmin : float
        Minimum value to display.
    vmax : float
        Maximum value to display.
    dBins : float, optional
        Bins' size. The default is None.

    '''
    
    # Estimating the number of bins in the histogram
    _range = abs(vmax - vmin)
    if dBins is None:
        nBins = 100
    else:
        nBins = int(_range / dBins)
        
    # Binning the data 
    binned_data, bins = np.histogram(data, range=(vmin, vmax), bins=nBins, density=True)

    # Calculating the middle of each bin
    x_bins = np.array( [ _bin + (bins[ii+1] - _bin) / 2 for ii, _bin in enumerate(bins[:-1]) ] )
    
    # Ploting the histogram
    ax.bar(x_bins, binned_data, width=dBins, edgecolor='k', align='center')
    return


def plot_density_scatter(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, 
                         sort: bool=True, bins: int=20, **kwargs:dict) \
                         -> mpl.artist.Artist:
    '''
     Scatter plot colored by 2d histogram
     
     See https://stackoverflow.com/a/53865762/22968865
    
     Parameters
     ----------
     ax : mpl.axes.Axes
         Axis where the histogram should be displayed..
     x : np.ndarray
         Data to be displayed along the x-axis.
     y : np.ndarray
         Data to be displayed along the y-axis
     sort : bool, optional
         Sort the points by density, so that the densest points are plotted last. The default is True.
     bins : int, optional
         Number of bins used to generate the density coloring scheme. The default is 20.
     **kwargs : dict
         Any arguments that can be passed to pyplot.scatter().
    
     Returns
     -------
     scatt : mpl.artist.Artist
         Matplotlib Artist.
    
    '''
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    
    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    
    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    scatt = ax.scatter( x, y, c=z, cmap='jet', **kwargs)
    
    # norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # cbar.ax.set_ylabel('Density')
    
    return scatt


def plot_energy_spectra(ax: mpl.axes.Axes, EnergySpectra: np.ndarray, 
                        freqs: np.ndarray, theta: np.ndarray, 
                        cbar: bool=False, rad: bool=False, density: bool=False, 
                        rticks: float=225, **kwargs:dict) \
                        -> mpl.artist.Artist:
    '''
    Plot the Energy Spectra into a polar plot.

    Parameters
    ----------
    ax : mpl.axes.Axes
        Axis where the histogram should be displayed.
        
    EnergySpectra : numpy.array, [nFreqs, nTheta]
        Waves full Spectra, can either be the energy density or the energy spectra.
        
    freqs : numpy.array [nFreqs]
        Frequencies which the Wave Spectra was defined, given in Hz
        
    theta : numpy.array [nTheta]
        Angles along which the Wave Spectra was defined, given in rad or degrees
        
    cbar : bool, defualt is False
        Boolean indicating if the colorbar is added to the figure.
        
    rad : bool, default is False
        Boolean indicating if it is necessary to convert the angles to radians
        
    density : bool, default is False
        Boolean indicating if we are ploting a energy density spectra or an energy spectra
        Mainly used to changed the colorbar label.
        
    **kwargs : dict.
        Dict. of arguments that can be passed to plt.contourf

    '''
    
    # Convert the theta to radians if it is needed
    if rad is False:
        theta = np.deg2rad(theta)
        
    # Check for the last direction
    if theta[0] == 0 and theta[-1] != np.pi*2:
        theta = np.append(theta, 2 * np.pi)
        EnergySpectra = np.append(EnergySpectra, EnergySpectra[:, 0][:,None], axis=-1)
        
    # Defining the labels 
    xticks = np.deg2rad(np.linspace(0, 360, 8, endpoint=False))
    xticks_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    
    # Defining the cbar legend
    if density is False:
        cbar_lbl = r'Frequency - Directional Energy Spectrum $[m^2/Hz/rad]$'
    else:
        cbar_lbl = r'Frequency - Directional Energy Density Spectrum $[1/rad]$'
    
    
    # Mapping the coordinates into a [nFreqs, nTheta] shape
    plot_f = np.repeat(freqs, theta.shape).reshape(EnergySpectra.shape)
    plot_t = np.tile(theta, freqs.shape).reshape(EnergySpectra.shape)
    plot_ES = EnergySpectra.copy()
    
    
    # Orienting the theta axis to point toward the north
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    
    # Ploting the Spectrum as a contour plot
    im = ax.contourf(plot_t, plot_f, plot_ES, cmap='inferno', **kwargs)
    
    # Managing the grid
    ax.xaxis.grid(linewidth=0.75, linestyle='--', color='gray')
    ax.yaxis.grid(linewidth=0.75, linestyle='--', color='gray')
    
    # Managing the x-ticks
    ax.set_xticks(ticks=xticks, labels=xticks_labels)
    
    # Managing the y-ticks
    locs = ax.get_yticks()
    yticks_labels = [ '{:.2f} [Hz]'.format(v) for v in locs ]
    
    ax.set_rmax(max(freqs))
    ax.set_rticks(ticks=locs, labels=yticks_labels, color='w')
    ax.set_rlabel_position(rticks)

    # Adding a label to the colorbar
    if cbar is True:
        cbar = plt.colorbar(im, ax=ax, label=cbar_lbl, pad=0.1)

    return im



def set_subplots_numbering(axs: np.ndarray[mpl.axes.Axes] | list[mpl.axes.Axes], 
                           letters: bool=True)->None:
    '''
    Automatic subplots numbering.

    Parameters
    ----------
    axs : np.ndarray[mpl.axes.Axes] | list[mpl.axes.Axes]
        List (or array) of axes generated by matplotlib.
    letters : bool, optional
        Numbering scheme, (True) lower case letters, (False) numbers. The default is True.

    '''
    
    if letters is True:
        numbering_scheme = [ '({})'.format(chr(i)) for i in range(ord('a'),ord('z')+1)]
        
    else:
        numbering_scheme = [ '({})'.format(ii+1) for ii in range(26) ]
    
    
    if isinstance(axs, np.ndarray):
        for ii, ax in enumerate(axs.flatten()):
            ax.set_title(numbering_scheme[ii], loc='left', fontweight='bold')
            
    elif isinstance(axs, list):
        for ii, ax in enumerate(axs):
            ax.set_title(numbering_scheme[ii], loc='left', fontweight='bold')
        
    else:
        raise TypeError('Unrecognized {} type for axs parameter'.format(type(axs)))
        
    return

def plt_show_save(savename: str=None, savedir: str=None)->None:
    '''
    Wrapper used to either display or save a figure generated by matplotlib.
    The figure will only be saved if both parameters are provided. Note that, 
    if the figure is saved then it will not be displayed.
    
    Parameters
    ----------
    savename : str, optional
        Name of the figure. The default is None.
    savedir : str, optional
        Path where the figure will be saved. The default is None.

    '''
    if savedir is not None and savename is not None:
        # plt.savefig(os.path.join(savedir, savename), bbox_inches='tight')
        plt.savefig(os.path.join(savedir, savename))
        plt.close()
        print('\n{} was successfully saved in {}'.format(savename, savedir))
    else:
        plt.show()
    return

def get_discreet_cmap(nColor: int, cmap: str='jet', vlim: tuple=None, _reversed: bool=False, cList: bool=False):
    '''
    
    
    I won't write the types for the outputs as this function is messy

    Parameters
    ----------
    nColor : int
        Number of colors to be returned (i.e., nColors=20 will return a cmap with
        20 bins).
    cmap : str, optional
        Name of the colormap to discreetize. The default is 'jet'.
    vlim : tuple, optional
        Tuple following a (vmin, vmax) format. The default is None.
    _reversed : bool, optional
        Reverse the colormap if set to True (e.g., cmap='jet' would go from 
        red to blue instead of blue to red). The default is False.
    cList : bool, optional
        Return the new colormap as list of colors if set to True and as classic
        colorbar if set to False. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if _reversed:
        new_cmap = plt.cm.get_cmap(cmap, nColor).reversed()
    else:
        new_cmap = plt.cm.get_cmap(cmap, nColor)
    
    cmapList = [ new_cmap(i) for i in range(new_cmap.N) ]
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmapList, new_cmap.N)
    
    if cList is False:
        if vlim is not None:
            if nColor <= 20:
                x_ticks_labels = np.around(np.linspace(vlim[0], vlim[1], nColor, endpoint=False), decimals=2)
            else:
                x_ticks_labels = np.around(np.linspace(vlim[0], vlim[1], nColor//5, endpoint=False), decimals=2)
            return new_cmap, np.append(x_ticks_labels, vlim[1])
        else:
            return new_cmap
        
    else:
        return cmapList