# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:08:45 2025

@author: gregoires
"""

import numpy as np
import os
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import xarray as xr

from . wave_spectral import JONSWAP, jonswap_spreading_parameter, cosine_squared_general
from . func_tools import datenum_to_datetime
from .. metatata.netcdf_utils import StreamDataset

try:
    from typing import Self
except:
    from typing_extensions import Self


class HindcastData(StreamDataset):
    
    fname_fmt = 'subset_hindcast_{cnt}_{t0}_{t1}.nc'

    def __init__(self, time, node, latitude, longitude, depth, 
                 Hs, Hp1, Hp2, Hp3, Hp4, Hp5, Hp6,
                 Tm, Tp, Tp1, Tp2, Tp3, Tp4, Tp5, Tp6,
                 Dir, Dirp, Dp1, Dp2, Dp3, Dp4, Dp5, Dp6,
                 Fspr, Dspr, Dspr1, Dspr2, Dspr3, Dspr4, Dspr5, Dspr6,
                 Transp_X, Transp_Y, Wlen, frequency=None, theta=None, Espec=None):
        
        # Making sure everything is time-ordered
        indeces = np.argsort(time)
        
        # Creating placeholders to keep the data
        # Coordinates
        self.time = time
        self.node = node
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        
        # Significant wave height + partitions
        self.Hs = Hs
        self.Hp1 = Hp1
        self.Hp2 = Hp2
        self.Hp3 = Hp3
        self.Hp4 = Hp4
        self.Hp5 = Hp5
        self.Hp6 = Hp6
        
        # Periods + partitions
        self.Tp = Tp
        self.Tm = Tm
        self.Tp1 = Tp1
        self.Tp2 = Tp2
        self.Tp3 = Tp3
        self.Tp4 = Tp4
        self.Tp5 = Tp5
        self.Tp6 = Tp6
        
        # Direction of propagation + paritions
        self.Dir = Dir
        self.Dirp = Dirp
        self.Dp1 = Dp1
        self.Dp2 = Dp2
        self.Dp3 = Dp3
        self.Dp4 = Dp4
        self.Dp5 = Dp5
        self.Dp6 = Dp6
        
        # Directional spread + partitions
        self.Fspr = Fspr
        self.Dspr = Dspr
        self.Dspr1 = Dspr1
        self.Dspr2 = Dspr2
        self.Dspr3 = Dspr3
        self.Dspr4 = Dspr4
        self.Dspr5 = Dspr5
        self.Dspr6 = Dspr6
        
        # Others
        self.Transp_X = Transp_X
        self.Transp_Y = Transp_Y
        self.Wlen = Wlen
        
        # Adding the spectra - these values are not in the hindcast (thus why they
        # are defined as None in the class definition)
        self.frequency = frequency
        self.theta = theta                  # /!\ This one is defined in [degrees] and not in [radians]. The conversion is done later on when needed
        self.Espec = Espec
        
        self.keys = list(self.__dict__.keys())
        
        # Looping over every arrays stored by this object and making sure they are 
        # time-ordered
        for key in self.keys:
            val = getattr(self, key)
            
            if val is None:
                continue
            elif key in ['frequency', 'theta', 'node', 'longitude', 'latitude', 'depth']:
                continue
            else:
                setattr(self, key, val[indeces,...])
                
        # Initialize the StramDataset class
        super().__init__()
        
        
        
        
    @classmethod
    def from_objects(cls, Data:list[Self])->Self:
        '''
        Factory method to used to merge a list of objects of this class. Although
        not optimised, this allows to load each file independently while dumping
        them into this class before merging their content into an object belonging
        to the same class

        Parameters
        ----------
        Data : list[Self]
            List containing objects of this class. Each object should correspond
            to a netcdf file.

        Returns
        -------
        Self
            Object belonging to this class containing the merged data.
        '''
        
        merged_dict = {}
        for d in Data:
            tmp = d.to_dict()
            for key, val in tmp.items():
                if key in ['node', 'longitude', 'latitude', 'depth']:
                    merged_dict[key] = val.copy()
                    
                elif key in ['frequency', 'theta']:
                    merged_dict[key] = val.copy()
                    
                else:
                    merged_dict.setdefault(key, []).append(val)
        
        
        for key, item in merged_dict.items():
            if item is None or any([ True if v is None else False for v in item ]):
                merged_dict[key] = None
                continue
            
            elif key in ['frequency', 'theta']:
                merged_dict[key] = item.copy()
                continue
            
            elif key in ['node', 'longitude', 'latitude', 'depth']:
                merged_dict[key] = item.copy()
                continue
                
            elif key == 'time':
                tmp = np.empty((0), dtype=np.datetime64)
                
            elif key == 'Espec':
                # If Espec is not None then neither frequency nor theta should be None
                nfreqs = len(Data[0].frequency)
                ntheta = len(Data[0].theta)
                nLocs = len(Data[0].longitude)
                shape = (0, nLocs, nfreqs, ntheta)
                tmp = np.empty(shape, dtype=float)
            
            else:
                nLocs = len(Data[0].longitude)
                tmp = np.empty((0, nLocs), dtype=float)
            
            
            for v in item:
                tmp = np.append(tmp, v, axis=0)
            
            merged_dict[key] = tmp.copy()
            
        Out = cls(**merged_dict)
        return Out


    @classmethod
    def from_netcdf(cls, fpath:str, lonlat:np.ndarray=None, drop_vars:list[str]=None)->Self:
        '''
        Load and store in memory a netcdf file located at fpath. If the lonlat 
        parameter is provided, only the locations at these points will be loaded.

        Parameters
        ----------
        fpath : str, path-like
            Path pointing to the netcdf file to load.
        lonlat : np.ndarray, (M,2) optional
            Numpy array containing longitude / latitude coordinates of the points
            to extract. The first column is the longitude, the second column is the latitude.
            The default is None.
        drop_vars : list[str], optional
            List of variables to drop before loading the file. The default is None.

        Returns
        -------
        Self
            Object belonging to this class containing the merged data.

        '''
        
        msk = lonlat[:,0] > 180
        lonlat[msk,0] -= 360
        
        
        if drop_vars is None:
            drop_vars = ['triangles']
        else:
            drop_vars += ['triangles']
        
        kwargs = {'decode_times':False}
        with xr.open_dataset(fpath, **kwargs) as fncdf:
            ds = fncdf.copy()
        
        dkeys = list(ds.keys())
        ckeys = list(ds.coords.keys())
        keys = dkeys + ckeys
        data = { key : None for key in keys }
        _ = data.pop('triangles')
        
        
        if lonlat is not None:
            lon = ds['longitude'].data
            lat = ds['latitude'].data
        
        for key in keys:
            if drop_vars is not None and key in drop_vars:
                continue
            
            elif key == 'time':
                data['time'] = np.array([ datenum_to_datetime(v) for v in ds['time'].values ], dtype='datetime64[s]')
                continue
            
            elif lonlat is None:
                data[key] = ds[key].data.copy()
            
            elif key == 'depth' and lonlat is not None:
                data['depth'] = cls.interp_from_grid_to_station(lon, lat, ds['depth'].data, lonlat[:,0], lonlat[:,1]).T
            
            else:
                data[key] = cls.interp_from_grid_to_station(lon, lat, ds[key].data.T, lonlat[:,0], lonlat[:,1]).T
                

            
        # Adding the longitude / latitude in case it wasn't done before
        if lonlat is not None:
            data['longitude'] = lonlat[:,0].copy()
            data['latitude'] = lonlat[:,1].copy()
            
        return cls(**data)


    def to_dict(self)->dict:
        '''
        Return every variables stored by the object in the form of a dictionary.
        Variables can be checked by looking the self.keys attribute.

        Returns
        -------
        dict
            Dictionary containing the data stored by this object.

        '''
        out = { key : getattr(self, key) for key in self.keys }
        return out

    
    def export_data_to_netcdf(self, savedir:str, fname:str, varout:list[str]=None)->None:
        '''
        Export the data stored within this object to a netcdf file.

        Parameters
        ----------
        savedir : str, path-like
            Path pointing to the directory where the file will be saved.
        fname : str
            Name of the netcdf file.
        varout : list[str], optional
            List of variable names to export in the created netcdf. The default is None.
        
        '''
        if varout is not None:
            keys = [ key for key in varout if key in self.keys and getattr(self, key) is not None ]
            missing_keys = list(set(varout) - set(keys))
        
            if missing_keys:
                msg = 'Unable to find \'{}\'. These values will not be exported to the netcdf'.format('\', \''.join(missing_keys))
                print(msg)
        else:
            keys = [ key for key in self.keys if getattr(self, key) is not None ]
            
        
        for key in keys:
            if key in ['time', 'node', 'frequency', 'theta']:
                pass
            
            elif key in ['longitude', 'latitude', 'depth']:
                coords = { 'node' : self.node }
                data = getattr(self, key)
                self.ingest_data(data, coords, key, atype='cattrs', ktype='short_name')
                
            else:
                attrs = self.get_attrs(key, atype='vattrs', ktype='short_name')
                coords = { v.lower() : getattr(self, v.lower()) for v in attrs['coordinates'].split(' ') }
                data = getattr(self, key)
                
                # Conversion because of the integrals limits
                if key == 'Espec':
                    data *= np.pi / 180
                
                self.ingest_data(data, coords, key, atype='vattrs', ktype='short_name')
        

        # Looping over our coordinates to make sure they have their metadata
        for key in ['time', 'node', 'frequency', 'theta']:
            self.ds[key].attrs = self.get_attrs(key, atype='cattrs', ktype='short_name').copy()
        
        self.to_netcdf(savename=fname, directory=savedir)
        return


    def get_spectra_from_wave_partitions(self, freqs:np.ndarray, theta:np.ndarray,
                             gamma:float=3.3)->None:
        '''
        Reconstruct the omni-directional wave spectra based on the paritions
        present in the hindcast. The reconstruction goes through 3 steps:
                - Estimating the frequency spectrum (JONSWAP model, Hasselmann et al, 1973)
                - Estimating the directional distribution (cosine-2s model, Mitsuyasu et al, 1975.)
                - Reconstruct the full spectrum using E = D(f,theta) * S(f)
        
        Notes:
            1. In the current implementation, theta has to be provided in degrees
            as it will be converted to radians later on.
            
            2. In the current implementation, D(f,theta) is, in fact, frequency
            independent. The distribution is expressed in term of directions only.
            
            3. The results are not returned, they are stored in the object.

        Parameters
        ----------
        freqs : np.ndarray (M,)
            Numpy array containing the frequency bins used to estimate the Energy Spectra.
        theta : np.ndarray (N,)
            Numpy array containing the directions used to estimate the Energy Spectra.
        gamma : float, optional
            JONSWAP Peak enhancement factor. The default is 3.3.

        '''
        
        
        # Gather the partitions
        Hpi = [self.Hp1, self.Hp2, self.Hp3, self.Hp4, self.Hp5, self.Hp6]
        Tpi = [self.Tp1, self.Tp2, self.Tp3, self.Tp4, self.Tp5, self.Tp6]
        Dpi = [self.Dp1, self.Dp2, self.Dp3, self.Dp4, self.Dp5, self.Dp6]
        Dspri = [self.Dspr1, self.Dspr2, self.Dspr3, self.Dspr4, self.Dspr5, self.Dspr6]
        
        # Instantiating the arrays
        nTstep = len(Hpi[0])
        nLocs = Hpi[0].shape[1]
        shape_ = ((nTstep, nLocs, freqs.shape[0], theta.shape[0]))
        E_spec = np.zeros(shape_)
        
        # Broadcasting the arrays into a (TIME x LOCATION x FREQUENCIES x ANGLES)
        # shape to allow for vectorized calculation
        freqs_ = freqs[None,None,:,None] * np.ones(E_spec.shape)
        theta_ = theta[None,None,None,:] * np.ones(E_spec.shape)
        
        # Looping over each partition, time / locations are vectorized
        for ii in range(len(Hpi)):
                
            fp = 1 / Tpi[ii][:,:,None,None]
            Hs = Hpi[ii][:,:,None,None]
            Dp = Dpi[ii][:,:,None,None]
            Dspr = Dspri[ii][:,:,None,None]
            
            # Cleaning out non physical / out of range values
            # Dp[Dp > 180] -= 360
            
            fp[fp<1e-5] = np.nan
            fp[np.isinf(fp)] = np.nan
            
            # Estimating the Frequency Spectra + Directional distribution
            # The braodcasting is done to take into account the locations
            F_spec_i = JONSWAP(Hs=Hs, fp=fp, freqs=freqs_, gamma=gamma)
            s_i = jonswap_spreading_parameter(Dspr, unit='degree')
            D_theta = cosine_squared_general(theta_, Dir=Dp, s=s_i, unit='degree')
        
            tmp = F_spec_i * D_theta
            tmp[tmp<1e-5] = 0
            tmp[~np.isfinite(tmp)] = 0
            E_spec += tmp

        # Store the data in memory and return in case it is needed
        self.theta = theta
        self.frequency = freqs
        self.Espec = E_spec
        return self.Espec, self.frequency, self.theta
    
    @staticmethod
    def interp_from_grid_to_station(xgrid:np.ndarray, ygrid:np.ndarray, zgrid:np.ndarray, \
                                    x:np.ndarray, y:np.ndarray, \
                                    nearest_neighbor:bool=True)->np.ndarray:
        '''
        Wrapping function used to interpolate points data from a triangular grid.
        
        Note that it is possible to provide a time-varying grid if -and only if-
        this grid is given in a (M,N) shape with M being the position's index and
        N being the timesteps.
        
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html
        for an explanation.

        Parameters
        ----------
        xgrid : np.array (M,)
            Longitude.
        ygrid : np.array (M,)
            Latitude.
        zgrid : np.array (M,N)
            Data values at point m.
        x : np.array(L,)
            Interpolation points' Longitude.
        y : np.array (L,)
            Interpolation points' Latitude.

        Returns
        -------
        z  : np.array (L,N)
            Data interpolated.

        '''
        if nearest_neighbor:
            interp = NearestNDInterpolator(list(zip(xgrid, ygrid)), zgrid)
        else:
            interp = LinearNDInterpolator(list(zip(xgrid, ygrid)), zgrid)
        z = interp(x, y)
        return z
    
    










    
