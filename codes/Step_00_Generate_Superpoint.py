# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:52:50 2025

@author: moritzw
"""
import os
import xarray as xr
import numpy as np
import wavespectra
import matplotlib.pyplot as plt
from matplotlib import gridspec

path_hindcast = r'D:\CISPac-5\CIS-PAC5-Niue-Inundation-ML\extras\spec_hindcast\subset'
start_year = 1979
end_year = 2023
out_fl_path = os.path.join(r'D:\CISPac-5\CIS-PAC5-Niue-Inundation-ML\extras\spec_hindcast', 'Niue_Superpoint.nc')


fl_path_list = []
for yy in range(start_year,end_year+1):
    fl_name = 'subset_hindcast_NIUE_' + str(yy) + '0101_' + str(yy) + '1231.nc'
    print(fl_name)
    fl_path = os.path.join(path_hindcast,fl_name)
    fl_path_list.append(fl_path)
    
ds = xr.open_mfdataset(fl_path_list, combine="by_coords")  # Auto-detects the time dimension


ds_max = ds.Espec.max(dim='node')
ds_superpoint = xr.Dataset({'efth':ds_max})


ds_superpoint.to_netcdf(out_fl_path)




def nth_largest_index(arr, n):
    """Find the index of the n-th largest value in a NumPy array."""
    if n > arr.size:
        raise ValueError("n is larger than the array size.")
    
    # Flatten the array and get the index of the n-th largest element
    flat_index = np.argpartition(arr.flatten(), -n)[-n]
    
    # Convert flat index to multi-dimensional index if needed
    return np.unravel_index(flat_index, arr.shape)

ds = ds.rename({'frequency':'freq','theta':'dir','Espec':'efth'})
ds_superpoint = ds_superpoint.rename({'frequency':'freq','theta':'dir'})

hs = ds_superpoint.spec.hs().values
ix_1 = nth_largest_index(hs, 1) # Ofa
ix_2 = nth_largest_index(hs, 11) # Heta
index_array = ix_2


ds_subset = ds.efth.isel(time=index_array[0]).to_dataset()
ds_superpoint_subset = ds_superpoint.efth.isel(time=index_array[0]).to_dataset()

efth_max = ds_superpoint_subset['efth'].max().values.item()

fig = plt.figure(figsize=[14,12])
#sp_regular_subset.spec.plot(kind="contourf",as_period=True,normalised=False, cmap="Spectral_r")
ds_superpoint_subset.spec.plot(
    as_period=True,
    normalised=False,
    cmap="Spectral_r",
    levels=np.logspace(np.log10(0.005), np.log(efth_max), 128,endpoint=True),
    cbar_ticks=[0.005,0.01, 0.1, efth_max],
)
fig.savefig('D:\CISPac-5\CIS-PAC5-Niue-Inundation-ML\extras\spec_hindcast\Spectra_Example\Heta_Superpoint_spec.png')
plt.close(fig)

for i in range(8):
    fig = plt.figure(figsize=[14,12])
    #sp_regular_subset.spec.plot(kind="contourf",as_period=True,normalised=False, cmap="Spectral_r")
    ds_subset.isel(node=i).spec.plot(
        as_period=True,
        add_colorbar=False,
        normalised=False,
        cmap="Spectral_r",
        levels=np.logspace(np.log10(0.005), np.log(efth_max), 128,endpoint=True),
        cbar_ticks=[0.005,0.01, 0.1, efth_max],
    )
    fig.savefig(r'D:\CISPac-5\CIS-PAC5-Niue-Inundation-ML\extras\spec_hindcast\Spectra_Example\HetaPoint' + str(i) + '_spec.png')
    plt.close(fig)

    
    
    
    
# ds_new = ds.isel(time=ix_2[0])
# plt.figure()
# plt.plot(ds_new.Dp2)
# plt.title('Dp2')


# fig = plt.figure(figsize=[15,10])
# gs3=gridspec.GridSpec(3,2,hspace=0.01, wspace=0.01)
# ax=fig.add_subplot(gs3[0])
# plt.plot(ds_new.Dp1)
# plt.title('Dp1')
# ax=fig.add_subplot(gs3[1])
# plt.plot(ds_new.Dp2)
# plt.title('Dp2')
# ax=fig.add_subplot(gs3[2])
# plt.plot(ds_new.Dp3)
# plt.title('Dp3')
# ax=fig.add_subplot(gs3[3])
# plt.plot(ds_new.Dp4)
# plt.title('Dp4')
# ax=fig.add_subplot(gs3[4])
# plt.plot(ds_new.Dp5)
# plt.title('Dp5')
# ax=fig.add_subplot(gs3[5])
# plt.plot(ds_new.Dp6)
# plt.title('Dp6')


# fig = plt.figure(figsize=[15,10])
# gs3=gridspec.GridSpec(3,2,hspace=0.01, wspace=0.01)
# ax=fig.add_subplot(gs3[0])
# plt.plot(ds_new.Hp1)
# plt.title('Hs1')
# ax=fig.add_subplot(gs3[1])
# plt.plot(ds_new.Hp2)
# plt.title('Hs2')
# ax=fig.add_subplot(gs3[2])
# plt.plot(ds_new.Hp3)
# plt.title('Hs3')
# ax=fig.add_subplot(gs3[3])
# plt.plot(ds_new.Hp4)
# plt.title('Hs4')
# ax=fig.add_subplot(gs3[4])
# plt.plot(ds_new.Hp5)
# plt.title('Hs5')
# ax=fig.add_subplot(gs3[5])
# plt.plot(ds_new.Hp6)
# plt.title('Hs6')

