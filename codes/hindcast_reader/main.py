# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:07:35 2025

@author: gregoires
"""

from datetime import datetime
import numpy as np
import os
import sys
import pandas as pd
import src
from src.metatata.netcdf_utils import StreamDataset
from src.utils.func_tools import get_filenames, get_surrounding_positions
from src.utils.hindcast import HindcastData



def setup(run=True):
    # country can either be CK or NIUE
    country = 'NIUE'
    
    if country.upper() == 'CK':
        root = os.path.join('S:\\', 'GEM', 'FJ_NAB', 'O_M', 'Oceans', 'Active', 'MARINE_SURVEYS_SURVEYS', 'ECIKS', 'Products', 'WaveHindcasts', country, 'Rarotonga')
    elif country.upper() == 'NIUE':
        root = os.path.join('S:\\', 'GEM', 'FJ_NAB', 'O_M', 'Oceans', 'Active', 'MARINE_SURVEYS_SURVEYS', 'ECIKS', 'Products', 'WaveHindcasts', country)
    else:
        msg = 'Only "CK" or "NIUE" can be used, not {}.'.format(country)
        raise ValueError(msg)
        
    # Define the saving directory
    savedir = os.path.join('D:\\', 'cispac-5', 'CIS-PAC5-Niue-Inundation-ML', 'extras', 'spec_hindcast','subset')
    
    # Enter the year / month / day of your first and last timesteps
    t0 = datetime(year=1979, month=1, day=1)
    t1 = datetime(year=2023, month=12, day=31)
    
    # # This one is an example to have a positions along a circle drawn around a given point
    # x0, y0 = (190.15, -19.05)
    # radius = 0.25
    # angles = np.arange(0, 360, 45)
    # xi, yi = get_surrounding_positions(x0, y0, radius, angles=angles)
    
    df = pd.read_csv(r"D:\CISPac-5\CIS-PAC5-Niue-Inundation-ML\extras\spec_hindcast\location.csv")
    xi = df.X.values
    #xi=xi+360
    yi = df.Y.values
    
    
    # xi = np.array([-169.73625233, -169.74191518, -169.74160006, -169.73753673,
    #        -169.73739576, -169.73223107, -169.73055027, -169.73212948,
    #        -169.74277026, -169.74682834])
    # yi = np.array([-19.0538761 , -19.05147651, -19.05646728, -19.04852579,
    #        -19.05960041, -19.04936968, -19.05371748, -19.05812263,
    #        -19.04654219, -19.0541015 ])
    
    
    # if you dont use the previous lines (estimating the positions), you can
    # provide an array of M locations following (longitude, latitude) -> (M,2)
    # to be used. If this parameter is not provided then every points will retrieved
    lonlat = np.array([xi, yi]).T

    # List of variables to be DROPPED from the hindcast
    # /!\ DO NOT REMOVE THE PARTITIONS, otherwise there will be no spectra
    drop_vars = ['Transp_X', 'Transp_Y', 'Wlen']
    
    # List of variables to be EXPORTED (after wave spectra reconstruction)
    # If set to None, every variables not dropped with drop_vars will be exported
    varout = None
    
    # Defining some parameters
    gamma = 1.5         # Gamma used for the JONSWAP
    fmin = 0.03         # Starting frequency [Hz]
    fmax = 0.4          # Ending frequency [Hz]
    df = 0.01           # Frequency step [Hz]
    dtheta = 10         # Directional step [°]
    
    # Gathering all the inputs into a dict.
    kwargs = {'root' : root,
              'savedir' : savedir,
              't0' : t0,
              't1' : t1,
			  'country' : country,				  
              'lonlat' : lonlat,
              'drop_vars' : drop_vars,
              'varout' : varout,
              'gamma' : gamma,
              'fmin' : fmin,
              'fmax' : fmax,
              'df' : df,
              'dtheta' : dtheta}
    
    if run:
        extract_netcdf(**kwargs)
    
    return kwargs






def extract_netcdf(root, t0, t1, savedir, country, lonlat=None, drop_vars=None, varout=None, 
                   gamma=3.3, fmin=0.01, fmax=0.5, df=0.01, dtheta=10):
    
    # Define the coordinates used for the full spectra
    freqs = np.arange(fmin, fmax+df, df)
    theta = np.arange(0, 360, dtheta)
    
    # Define more parameters for the run
    files = get_filenames(fpath=root, t0=t0, t1=t1)
    
    Data = []
    start_year = t0.year
    for file in files:
        print('Working on', file)
        datestr = file.split('_')[1]
        current_year = int(datestr[:4])
        fpath = os.path.join(root, file)
        
        tmpData = HindcastData.from_netcdf(fpath=fpath, lonlat=lonlat, drop_vars=drop_vars)
        tmpData.get_spectra_from_wave_partitions(freqs, theta, gamma=gamma)
        
        if start_year != current_year:
            # Export the data
            mergedData = HindcastData.from_objects(Data)
            fname = mergedData.get_savename(t0=mergedData.time[0], t1=mergedData.time[-1], country=country)
            mergedData.export_data_to_netcdf(savedir=savedir, varout=varout, fname=fname)

            # Set back the list to an empty one
            Data = []
            start_year = current_year
            
        elif file == files[-1]:
            Data.append(tmpData)
            # Export the data
            mergedData = HindcastData.from_objects(Data)
            fname = mergedData.get_savename(t0=mergedData.time[0], t1=mergedData.time[-1], country=country)
            mergedData.export_data_to_netcdf(savedir=savedir, varout=varout, fname=fname)
            break
            
        Data.append(tmpData)
        



#if __name__ == "main":
setup(run=True)






