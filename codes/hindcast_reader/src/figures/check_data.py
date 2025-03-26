# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:02:21 2025

@author: gregoires
"""


from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from shapely import box, get_coordinates
from scipy.integrate import simpson
import sys
import xarray as xr


from . func_plots import plot_histogram, plot_density_scatter, plot_energy_spectra, set_subplots_numbering, plt_show_save, get_discreet_cmap
from .. utils.func_tools import get_outliers, BasicEstimators, get_map_grid_definition
from .. utils.wave_spectral import EnergySpectra, JONSWAP, jonswap_spreading_parameter, cosine_squared_general




def plot_wave_parameters_timeseries(data_spec, data_ori, time, vname, dout=None):
    
    pkwargs = {'nrows' : 4,
               'ncols' : 1,
               'layout' : 'compressed',
               'figsize' : (9, 12),
               'dpi' : 200}
    
    if dout is None:
        _ = pkwargs.pop('dpi')
        _ = pkwargs.pop('figsize')
    
    savename = 'hindcast_subset_{}_comparison.jpeg'.format(vname)
    # savename = None # there are issues that need to be fixed before being able to print the figures
    
    vmax = np.ceil(max(np.nanmax(data_spec), np.nanmax(data_ori)))
    vmin = np.floor(max(np.nanmin(data_spec), np.nanmin(data_ori)))
    
    Estimator = BasicEstimators(y_est=data_spec, y_exp=data_ori)
    lbl_scatter = Estimator.to_str()
    
    fig, axs = plt.subplots(**pkwargs)
    # Distributions share the same x-axis
    axs[2].sharex(axs[1])
    axs[2].sharey(axs[1])
    
    # Scatter plot may share the same axis as the distributions
    axs[3].sharex(axs[1])
    
    
    # Ploting the timeseries
    axs[0].plot(time, data_ori, '-k', label='Hindcast')
    axs[0].plot(time, data_spec, '--r', label='Reconstructed')
    
    
    # Working on the distributions
    vmax = np.ceil(max(np.nanmax(data_spec), np.nanmax(data_ori)))
    vmin = np.floor(min(np.nanmin(data_spec), np.nanmin(data_ori)))
    _range = abs(vmax - vmin)
    dBins = min(0.05, _range/100)
    
    # Ploting the distributions
    plot_histogram(ax=axs[1], data=data_ori, vmin=vmin, vmax=vmax, dBins=dBins)
    plot_histogram(ax=axs[2], data=data_spec, vmin=vmin, vmax=vmax, dBins=dBins)
    
    # Ploting the density scatter
    scatt = plot_density_scatter(axs[3], data_ori, data_spec, label=lbl_scatter)
    axs[3].plot(np.arange(vmin, vmax+0.1, 0.1), np.arange(vmin, vmax+0.1, 0.1), '--k', label='1:1')
    
    # Setting up some nice looking thingies for the plots
    for ax in axs:
        ax.grid()
    
    axs[0].legend()
    axs[0].set(title='{} - Timeseries'.format(vname))
    
    axs[1].set(title='Original data')
    axs[2].set(title='Reconstructed data')
    
    axs[3].legend()
    axs[3].set(xlabel='Original', ylabel='Reconstructed', xlim=(vmin, vmax), ylim=(vmin, vmax))
    
    set_subplots_numbering(axs, letters=True)
    plt_show_save(savedir=dout, savename=savename)
    return


def plot_spectra_reconstruction(Hpi, Tpi, Dpi, Dspri, freqs, theta, wpar, time, dout=None, fname=None):
    # dout = None
    
    if fname is None:
        fname = 'hindcast_subset_reconstruction_steps.jpeg'
    
    title_fmt = '(Hindcast) Hs: {:.2f} [m], Tp: {:.2f} [s]\n' + \
                '(Spectrum) Hs: {:.2f} [m], Tp: {:.2f} [s]\n' + \
                'Time of estimation: {}'
    
    stitle_fmt = 'Part.{} - Hs: {:.2f} [m], Tp: {:.2f} [s]'
    stitle_fmt2 = 'Dirm: {:.2f} [°], Dspr: {:.2f} [°]'
    
    nPartitions = len(Hpi)
    Fspec_i = []
    DirDist_i = []
    Espec_i = []
    
    for ii in range(nPartitions):
        Fspec_i.append(JONSWAP(Hs=Hpi[ii], fp=1/Tpi[ii], freqs=freqs, gamma=3.3))
        s_i = jonswap_spreading_parameter(dspr=Dspri[ii], unit='degree')
        DirDist_i.append(cosine_squared_general(theta, Dpi[ii], s=s_i, unit='degree'))
        
        vEspec = Fspec_i[ii][:,None] * DirDist_i[ii][None,:]
        Espec_i.append(vEspec)
      
        
        
    Espec = np.nansum(np.array(Espec_i), axis=0)
    Fspec = simpson(Espec, x=np.deg2rad(theta), axis=-1)
    DirDist = simpson(Espec / Fspec[:,None], x=freqs, axis=0)
    
    # Appending the reconstructed spectra at the end of the list
    Fspec_i.append(Fspec)
    DirDist_i.append(DirDist)
    Espec_i.append(Espec)
    
    Spectrum = EnergySpectra(Espec, time=None, freqs=freqs, theta=theta, unit='degree')
        
    # Ploting time
    pkwargs = {'layout' : 'compressed',
               'figsize' : (13, 15),
               'dpi' : 200
               }
    
    title = title_fmt.format(wpar['Hs'], wpar['Tp'], Spectrum.wpar['HM0'][0], Spectrum.wpar['TPEAK'][0], time)
    
    if dout is None:
        _ = pkwargs.pop('dpi')
        _ = pkwargs.pop('figsize')
    
    nrows = nPartitions + 1
    ncols = 3
    
    cAx0 = []
    cAx1 = []
    cAx2 = []
    
    fig = plt.figure(**pkwargs)
    gs = GridSpec(nrows, ncols, figure=fig)
    
    for ii in range(nrows):
       cAx0.append(fig.add_subplot(gs[ii,0]))
       cAx1.append(fig.add_subplot(gs[ii,1]))
       cAx2.append(fig.add_subplot(gs[ii,2], projection='polar'))
       
       cAx0[ii].plot(freqs, Fspec_i[ii])
       
       cAx1[ii].plot(theta, DirDist_i[ii])
       
       plot_energy_spectra(ax=cAx2[ii], 
                           EnergySpectra=Espec_i[ii], 
                           freqs=freqs, 
                           theta=theta,
                           rticks=45)
       
       cAx0[ii].grid()
       cAx1[ii].grid()
       
       if ii < 6:
           cAx0[ii].set_title(stitle_fmt.format(ii, Hpi[ii], Tpi[ii]), loc='left', fontsize='small')
           cAx1[ii].set_title(stitle_fmt2.format(Dpi[ii], Dspri[ii]), loc='left', fontsize='small')
    
    fig.suptitle(title)
    plt_show_save(savename=fname, savedir=dout)
    
    return


def plot_outliers(data_spec, data_ori, time, k, dout=None, fname=None):
    
    pkwargs = {'nrows' : 2,
               'ncols' : 1,
               'sharex' : True,
               'sharey' : True,
               'layout' : 'compressed',
               'figsize' : (12, 9),
               'dpi' : 200}
    
    if dout is None:
        _ = pkwargs.pop('dpi')
        _ = pkwargs.pop('figsize')
    
    
    # Estimating the outliers
    error = data_ori - data_spec
    msk_std = get_outliers(error, k=k, method='std')
    msk_iqr = get_outliers(error, k=k, method='iqr')
    
    indeces = np.argsort(abs(error))
    
    # Estimating the min / max values
    vmax = np.ceil(max(np.nanmax(data_spec), np.nanmax(data_ori)))
    vmin = np.floor(max(np.nanmin(data_spec), np.nanmin(data_ori)))
    
    # Starting the plot
    fig, axs = plt.subplots(**pkwargs)
    
    axs[0].plot(time, data_ori, '-k', label='Hindcast')
    axs[0].plot(time, data_spec, '--r', label='Reconstructed')
    axs[0].plot(time[msk_std], data_spec[msk_std], 'og', label='Outliers')
    
    axs[1].plot(time, data_ori, '-k')
    axs[1].plot(time, data_spec, '--r')
    axs[1].plot(time[indeces[-10:]], data_spec[indeces[-10:]], 'og')
    
    # Setting up some nice looking thingies for the plots
    for ax in axs:
        ax.grid()
        
    axs[0].legend()
    axs[0].set(title='Standard Deviation Method', ylim=(vmin, vmax))
    axs[1].set(title='10 Largest outliers', ylim=(vmin, vmax))
    
    fig.suptitle('(Hm0) Outliers detection')
    plt_show_save(savedir=dout, savename=fname)
    
    
    return






def main_check(fpath, savedir):
    
    
    
    idx = 2

    with xr.open_dataset(fpath) as fncdf:
        ds = fncdf

    
    # Load the full spectra
    freqs = ds['frequency'].values.copy()
    theta = ds['theta'].values.copy()
    time = ds['time'].values.copy()
    Espec = ds['Espec'].data.copy()
    
    longitude = ds['longitude'].data.copy()
    latitude = ds['latitude'].data.copy()
    nLocs = len(longitude)
    
    # Load the wave parameters
    keys = ['Hs', 'Tp', 'Tm', 'Dir', 'Dirp', 'Dspr']
    wpar = [ { key : ds[key].values[:,ii].squeeze().copy() for key in keys } for ii in range(nLocs) ]
    
    ds = None
    
    # Instantiate the energy spectra class and store them in a list
    Spectra = [ EnergySpectra(Espec=Espec[:,ii,:,:].squeeze(), time=time, freqs=freqs, theta=theta, unit='degree') for ii in range(nLocs) ]
    
    for key, val in Spectra[idx].wpar.items():
        msk = np.isnan(val)
        print('-'*20)
        print(key)
        print('Array Shape: ', val.shape)
        print('# of NaN', len(msk[msk].flatten()))
        print('% of NaN', 100 * len(msk[msk].flatten()) / len(msk.flatten()))
        print('Mean value: ', np.nanmean(val))
        print('Min value: ', np.nanmin(val))
        print('Max value: ', np.nanmax(val))
        print()
     
    
    # Ploting the data
    arr_spec = Spectra[idx].wpar['HM0']
    plot_wave_parameters_timeseries(arr_spec, wpar[idx]['Hs'], time=time, vname='Hs', dout=savedir)
    
    arr_spec = Spectra[idx].wpar['TPEAK']
    plot_wave_parameters_timeseries(arr_spec, wpar[idx]['Tp'], time=time, vname='Tp', dout=savedir)
    
    arr_spec = Spectra[idx].wpar['TM02']
    plot_wave_parameters_timeseries(arr_spec, wpar[idx]['Tm'], time=time, vname='Tm02', dout=savedir)
    return


def reconstruction_check(fpath, savedir):
    
    idx = 5
    tidx = 125
    
    with xr.open_dataset(fpath) as fncdf:
        ds = fncdf

    # Load the full spectra
    freqs = ds['frequency'].values.copy()
    theta = ds['theta'].values.copy()
    time = ds['time'].values[tidx]
    Espec = ds['Espec'].data[tidx,idx,:,:].copy()
    
    # theta = np.round(np.rad2deg(theta))
    
    longitude = ds['longitude'].data[idx]
    latitude = ds['latitude'].data[idx]
    
    # Load the wave parameters
    keys = ['Hs', 'Tp', 'Tm', 'Dir', 'Dirp', 'Dspr']
    wpar = { key : ds[key].values[tidx,idx].squeeze().copy() for key in keys }
    

    # Gather the partitions
    Hpi = [ds['Hp1'].data[tidx,idx], ds['Hp2'].data[tidx,idx], ds['Hp3'].data[tidx,idx], \
           ds['Hp4'].data[tidx,idx], ds['Hp5'].data[tidx,idx], ds['Hp6'].data[tidx,idx]]
        
    Tpi = [ds['Tp1'].data[tidx,idx], ds['Tp2'].data[tidx,idx], ds['Tp3'].data[tidx,idx], \
           ds['Tp4'].data[tidx,idx], ds['Tp5'].data[tidx,idx], ds['Tp6'].data[tidx,idx]]
        
    Dpi = [ds['Dp1'].data[tidx,idx], ds['Dp2'].data[tidx,idx], ds['Dp3'].data[tidx,idx], \
           ds['Dp4'].data[tidx,idx], ds['Dp5'].data[tidx,idx], ds['Dp6'].data[tidx,idx]]
        
    Dspri = [ds['Dspr1'].data[tidx,idx], ds['Dspr2'].data[tidx,idx], ds['Dspr3'].data[tidx,idx], \
             ds['Dspr4'].data[tidx,idx], ds['Dspr5'].data[tidx,idx], ds['Dspr6'].data[tidx,idx]]
    
    ds = None
    
    
    # Ploting time
    plot_spectra_reconstruction(Hpi, Tpi, Dpi, Dspri, freqs, theta, time=time, dout=savedir, wpar=wpar)
    
    return


def check_directional_distribution(savedir):
    '''
    Wrapper to display the sensibility of the cosine-2s model (Mitsuyasu et al. 1975)
    to the directional spreading parameter. Due to some modifications carried out
    on the functions used (read it as I fixed the issues), the outputed figure 
    will not take into account Dspr low values.
    
    The full plot can either be obtained by commenting out the lines setting values
    of s to 1000 if the values are higher in wave_spectral.jonswap_spreading_parameter()
    or by asking greggy for the already generated figure.
    
    Parameters
    ----------
    savedir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    
    savename = 'directional_distribution_threshold_v1.jpeg'
    dtheta = 1
    theta = np.arange(0, 360+dtheta, dtheta)
    
    Dirp = 45       # Random value, it doesn't affect anything
    Dspr = np.arange(0, 96, 1)
    # Dspr = np.arange(1, 16, 1)
    
    s = jonswap_spreading_parameter(Dspr, unit='degree')
    
    dirDist = []
    for ii, s_i in enumerate(s):
        tmp = cosine_squared_general(theta, Dirp, s=s_i, unit='degree')
        dirDist.append(tmp)
    
    dirDist = np.array(dirDist)
    
    # Retrieve the colormap and the color list objects
    clist = get_discreet_cmap(nColor=len(Dspr), cmap='turbo', cList=True)
    cmap = get_discreet_cmap(nColor=len(Dspr), cmap='turbo', cList=False)
    
    
    smax_xbeach = 1000
    dspr_min_xbeach = np.rad2deg(np.sqrt(2 / (smax_xbeach + 1)))
    
    txt = 'XBeach threshold (s={}, Dspr={:.2f}°)'.format(smax_xbeach, dspr_min_xbeach)
    
    fkwargs = {'nrows' : 1,
               'ncols' : 2,
               'layout' : 'compressed',
               'figsize' : (12, 9),
               'dpi' : 200}
    
    
    fig, axs = plt.subplots(**fkwargs)
    for ii, Dspr_i in enumerate(Dspr):
        axs[0].plot(theta, dirDist[ii,:], label='Dir. spread: {}'.format(Dspr_i), color=clist[ii])
    
    axs[1].plot(Dspr, s)
    axs[1].hlines(smax_xbeach, 0, dspr_min_xbeach, linestyles='--', colors='k')
    axs[1].vlines(dspr_min_xbeach, 0, smax_xbeach, linestyles='--', colors='k')
    
    axs[1].annotate(txt, xy=(dspr_min_xbeach, smax_xbeach),
                    xytext=(dspr_min_xbeach+10, smax_xbeach+100),
                    arrowprops=dict(facecolor='black', shrink=0.05))
    
    
    for ax in axs:
        ax.grid()
        # ax.legend()
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(Dspr.min(), Dspr.max()), cmap=cmap),
                ax=axs[0], orientation='horizontal', label='Directional Spreading [°]')
    
    axs[0].set(xlim=(0, 360), title='cosine-2s model, Mitsuyasu et al. 1975')
    axs[1].set(xlabel='Directional Spreading [°]', xlim=(0, Dspr.max()), title=r'$s = \frac{2}{Dspr^2} - 1$', ylim=(0, np.nanmax(s[np.isfinite(s)])))
    
    plt_show_save(savedir=savedir, savename=savename)
    
    return


def reconstruction_check_looped(fpath, savedir, k):
    
    with xr.open_dataset(fpath) as fncdf:
        ds = fncdf

    
    # Load the full spectra
    freqs = ds['frequency'].values
    theta = ds['theta'].values
    time = ds['time'].values
    Espec = ds['Espec'].data.copy()
    
    longitude = ds['longitude'].data
    latitude = ds['latitude'].data
    nLocs = len(longitude)
    
    
    # Load the wave parameters
    for ii in range(nLocs):
        keys = ['Hs', 'Tp', 'Tm', 'Dir', 'Dirp', 'Dspr']
        wpar = { key : ds[key].values[:,ii].squeeze().copy() for key in keys }
        
        # Instantiate the energy spectra class 
        Spectra = EnergySpectra(Espec=Espec[:,ii,:,:].squeeze(), time=time, freqs=freqs, theta=theta, unit='degree')
    
        # Estimate the error
        error = wpar['Hs'] - Spectra.wpar['HM0']
        indeces = np.argsort(abs(error))
        
        fname = 'outliers_retrieval_loc_{}.jpeg'.format(ii)
        plot_outliers(data_ori=wpar['Hs'], data_spec=Spectra.wpar['HM0'], time=time, k=k, dout=savedir, fname=fname)
        
        # creating subfolder to store the plots
        sdir = os.path.join(savedir, 'outliers', 'location_{}'.format(ii))
        if not os.path.exists(sdir):
            os.mkdir(sdir)
            
        # Retrieving the 10 largest errors
        indeces = np.argsort(abs(error))
        
        for jj, idx in enumerate(indeces[-10:]):
            # Gather the partitions
            Hpi = [ds['Hp1'].data[idx,ii], ds['Hp2'].data[idx,ii], ds['Hp3'].data[idx,ii], \
                   ds['Hp4'].data[idx,ii], ds['Hp5'].data[idx,ii], ds['Hp6'].data[idx,ii]]
                
            Tpi = [ds['Tp1'].data[idx,ii], ds['Tp2'].data[idx,ii], ds['Tp3'].data[idx,ii], \
                   ds['Tp4'].data[idx,ii], ds['Tp5'].data[idx,ii], ds['Tp6'].data[idx,ii]]
                
            Dpi = [ds['Dp1'].data[idx,ii], ds['Dp2'].data[idx,ii], ds['Dp3'].data[idx,ii], \
                   ds['Dp4'].data[idx,ii], ds['Dp5'].data[idx,ii], ds['Dp6'].data[idx,ii]]
                
            Dspri = [ds['Dspr1'].data[idx,ii], ds['Dspr2'].data[idx,ii], ds['Dspr3'].data[idx,ii], \
                     ds['Dspr4'].data[idx,ii], ds['Dspr5'].data[idx,ii], ds['Dspr6'].data[idx,ii]]
            
            tmp = { key : v[idx] for key, v in wpar.items() }
            
            fname = 'outliers_detected_loc_{}_n_{}_idx_{}.jpeg'.format(ii, jj, idx)
            plot_spectra_reconstruction(Hpi, Tpi, Dpi, Dspri, freqs=freqs, theta=theta, wpar=tmp, time=time[idx], dout=sdir, fname=fname)
            
    return



def export_synthetic_spectra(savedir):
    # Hs = np.ones(2) * 4
    # Tp = np.ones(2) * 14
    # Dirm = np.array([245, 270])
    # Dspr = np.ones(2) * 25
    
    Dirm = np.arange(0, 360, 10)
    Hs = np.ones(len(Dirm)) * 4
    Tp = np.ones(len(Dirm)) * 14
    Dspr = np.ones(len(Dirm)) * 25
    
    # Creating the coordinates
    df = 0.01
    freqs = np.arange(0.01, 0.5+df, df)
    
    dtheta = 5
    theta = np.arange(0, 360, dtheta)
    
    # Random time, it doesn't matter as the spectrum is a synthetic one
    time = np.arange(0, len(Hs))
    
    Spectra = EnergySpectra.from_jonswap(Hs=Hs, Tp=Tp, Dirm=Dirm, Dspr=Dspr, \
                                         freqs=freqs, theta=theta, time=time, 
                                          gamma=3.3, unit='degree')
    
    title_fmt = 'Hs: {:.2f} [m], Tp: {:.2f} [s]\n' + \
                'Dirm: {:.2f} [°], Dspr: {:.2f} [°]'
        
    for ii, t in enumerate(Spectra.time):
        fname = 'synthetic_spectra_idx_{}.jpeg'.format(ii)
        
        fig = plt.figure(figsize=(8,8), dpi=200, layout='compressed')
        ax = fig.add_subplot(111, projection='polar')
        plot_energy_spectra(ax=ax, EnergySpectra=Spectra.Espec[ii,:,:], 
                            freqs=Spectra.freqs, theta=Spectra.theta, rad=True)
        
        ax.set(title=title_fmt.format(Hs[ii], Tp[ii], Dirm[ii], Dspr[ii]))
        plt_show_save(savename=fname, savedir=savedir)
    
    
    Espec, theta = Spectra.spec_units_conversion()
    
    ds = xr.Dataset(
    data_vars=dict(
        Espec=(["idx", "frequency", "theta"], Espec),
        Hs=(["idx"], Hs),
        Tp=(["idx"], Tp),
        Dirm=(["idx"], Dirm),
        Dspr=(["idx"], Dspr)
        ),
    coords=dict(
        idx=time,
        frequency=freqs,
        theta=theta,
        )
    )
    
    fpath = os.path.join(savedir, 'spectra_file.nc')
    ds.to_netcdf(fpath)
    return


def compare_gamma_sensibility(fdir, savedir):
    
    files = [ f for f in os.listdir(fdir) if 'gamma' in f ]
    gammas = np.array([ float('.'.join(f.split('_')[-1].split('.')[:-1])) for f in files ])
    
    indeces = np.argsort(gammas)
    gammas = gammas[indeces]
    files = [ files[ii] for ii in indeces ]
    
    
    nFile = len(files)
    nLocs = 8
    nVars = 3
    
    RMSE = np.zeros((nFile, nLocs, nVars))
    R2 = RMSE.copy()
    
    # RMSE = np.random.rand(nFile*nLocs*nVars).reshape((nFile, nLocs, nVars))
    # R2 = np.random.rand(nFile*nLocs*nVars).reshape((nFile, nLocs, nVars))
    
    for ii, file in enumerate(files):
        fpath = os.path.join(fdir, file)
        
        with xr.open_dataset(fpath) as fncdf:
            ds = fncdf

        # Load the full spectra
        freqs = ds['frequency'].values
        theta = ds['theta'].values
        time = ds['time'].values
        Espec = ds['Espec'].data.copy()
        
        ds_Hs = ds['Hs'].values.copy()
        ds_Tp = ds['Tp'].values.copy()
        ds_Tm = ds['Tm'].values.copy()
        
        ds = None
        
        # Load the wave parameters
        for jj in range(nLocs):
            Hs = ds_Hs[:,jj].copy()
            Tp = ds_Tp[:,jj].copy()
            Tm = ds_Tm[:,jj].copy()
            
            # Instantiate the energy spectra class 
            Spectra = EnergySpectra(Espec=Espec[:,jj,:,:].squeeze(), time=time, freqs=freqs, theta=theta, unit='degree')
            
            r_Hs = BasicEstimators(y_est=Spectra.wpar['HM0'], y_exp=Hs)
            r_Tp = BasicEstimators(y_est=Spectra.wpar['TPEAK'], y_exp=Tp)
            r_Tm = BasicEstimators(y_est=Spectra.wpar['TM02'], y_exp=Tm)
            
            for kk, v in enumerate([r_Hs, r_Tp, r_Tm]):
                RMSE[ii,jj,kk] = v.rmse
                R2[ii,jj,kk] = v.r2
            
    
    pkwargs = {'nrows' : nVars,
               'ncols' : 2,
               'figsize' : (12,9),
               'layout' : 'compressed',
               'sharex' : True,
               'dpi' : 200}
    
    cList = get_discreet_cmap(nColor=nLocs, cmap='jet', cList=True)
    keys = ['Hm0', 'Tpeak', 'Tm02']
    
    fname = 'gamma_sensibillity_hindcast.jpeg'
    
    fig, axs = plt.subplots(**pkwargs)
    for kk in range(nVars):
        for jj in range(nLocs):
            axs[kk,0].plot(gammas, RMSE[:,jj,kk], color=cList[jj], label='Loc. {}'.format(jj))
            axs[kk,1].plot(gammas, R2[:,jj,kk], color=cList[jj], label='Loc. {}'.format(jj))
        
    for ax in axs.flatten():
        ax.grid()
        ax.set(xlim=(0, 7))
    
    for ii in range(nVars):
        axs[ii,0].set(title='{} - RMSE'.format(keys[ii]), xlabel=r'$\gamma$')
        axs[ii,1].set(ylim=(0, 1), title='r2', xlabel=r'$\gamma$')
        
    axs[0,1].legend(loc='best')
    set_subplots_numbering(axs.flatten())
    plt_show_save(savename=fname, savedir=savedir)
    
    return



def plot_AoI_location(fpath, savedir):
    
    from mpl_toolkits.basemap import Basemap
    
    with xr.open_dataset(fpath) as fncdf:
        ds = fncdf
        
    
    longitude = ds['longitude'].data.copy()
    latitude = ds['latitude'].data.copy()
    
    nLocs = len(longitude)
    if nLocs == 9:
        nLocs = 8 # small mistake here, the last point is the same as the first one
    
    pkwargs = {'figsize': (12, 9),
               # 'dpi' : 200,
               'layout' : 'compressed'}
    
    dx = 0.25
    
    bbox = box(longitude.min() - dx, latitude.min() - dx, \
               longitude.max() + dx, latitude.max() + dx)
    
    coord = get_coordinates(bbox)
    kwargs = {'projection' : 'cyl',
              'llcrnrlat' : coord[:,1].min(),
              'urcrnrlat' : coord[:,1].max(),
              'llcrnrlon' : coord[:,0].min(),
              'urcrnrlon' : coord[:,0].max(),
              'resolution' : 'h', 
              'suppress_ticks' : True}
    
    pkwargs = {'markeredgecolor' : 'k',
               'markersize' : 7,
               'marker' : 'o',
               'linestyle' : '',
               'zorder' : 9999}
    
    #
    xticks, yticks = get_map_grid_definition(bbox=bbox)
    
    
    fig, axs = plt.subplots(figsize=(8.5, 7), dpi=200, layout='compressed')
    m = Basemap(ax=axs, **kwargs)
    m.fillcontinents(color='lightgray', zorder=1)
    m.drawcoastlines(zorder=0)
    
    m.plot(longitude, latitude, color='r', latlon=True, **pkwargs)
    
    for ii in range(nLocs):
        x,y = m(longitude[ii] + 0.01, latitude[ii] + 0.01)
        axs.text(x,y, 'Loc. {}'.format(ii))
    
    
    m.drawparallels(yticks, labels=[1,0,0,0], fmt='%.2f', fontsize=6, color='grey')                             # Latitude lines
    m.drawmeridians(xticks, labels=[0,0,0,1], fmt='%.2f', fontsize=6, color='grey', rotation=45, ha='right')    # Labels on all sides
    
    fname = 'AreaOfInterest_locations.jpeg'
    plt_show_save(savename=fname, savedir=savedir)
    
    return
