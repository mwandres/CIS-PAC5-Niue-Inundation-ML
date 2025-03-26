# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:58:20 2025

@author: moritzw
"""
import numpy as np
from matplotlib import pyplot as plt
import os       
import xarray as xr
    
root = os.path.join('S:\\', 'GEM', 'FJ_NAB', 'O_M', 'Oceans', 'Active', 'MARINE_SURVEYS_SURVEYS', 'ECIKS', 'Products', 'WaveHindcasts', 'NIUE')

#x, y = np.random.rand(2)
x = 190.26201707-360
y = -19.05333437
ds = xr.open_dataset(root+'/Niue_198009.nc', decode_times=False)
xi = ds['longitude'].values
xi = xi 
yi = ds['latitude'].values
# xi = np.random.rand(100)
# yi = np.random.rand(100)


def get_N_closest_point(xi, yi, x, y, N):
    
    dist = np.sqrt((xi-x)**2 + (yi-y)**2)
    indeces = np.argsort(dist)
    xout = xi[indeces[:N]]
    yout = yi[indeces[:N]]
    return xout, yout

xout, yout = get_N_closest_point(xi, yi, x, y, N=10)


fig, axs = plt.subplots()
axs.plot(xi, yi, '.k')
axs.plot(x,y, '*r')
axs.plot(xout, yout, 'og')
plt.show()