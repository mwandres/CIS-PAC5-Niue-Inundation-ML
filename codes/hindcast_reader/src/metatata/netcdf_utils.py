# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:46:22 2024

@author: gregoires
"""


from copy import deepcopy
import datetime
import numpy as np
import os
import warnings
import xarray as xr

# Catching xarray warning when creating the datasets using a type np.datetime64 [s]
# when specifying the TIME dimension
warnings.filterwarnings('ignore', message='.*Converting non-nanosecond precision') 

from . cf_netcdf_attrs import VarAttrs, CoordAttrs, GlobAttrs, Params
from .. utils.func_tools import datetime64_to_datetime


class Dataset_VariablesFormat(VarAttrs, CoordAttrs):
    
    def __init__(self):
        VarAttrs.__init__(self)
        CoordAttrs.__init__(self)
    
    # Method used internally by the class
    def _get_cattrs(self, key: str, ktype: str) -> dict:
        
        if ktype == 'short_name':
            attrs = [ v for v in self._cattrs if v['short_name'] == key ][0]
        else:
            attrs = [ v for v in self._cattrs if v['standard_name'] == key ][0]
        return attrs

    def _get_vattrs(self, key: str, ktype: str) -> dict:
        
        if ktype == 'short_name':
            try:
                attrs = [ v for v in self._vattrs if v['short_name'] == key ][0]
            except:
                print('---> Call from netcdsf_utils')
                print('---> Unable to find ', key)
                return {'short_name': key}
        else:
            attrs = [ v for v in self._vattrs if v['standard_name'] == key ][0]
        return attrs
    
    
    # General methods
    def get_attrs(self, key: str, atype: str, ktype: str) -> dict:
        '''
        Wrapping function used to retrieve a variable / sensor / QC / coordinate
        attributes from a list of pre-defined values.

        Parameters
        ----------
        key : str
            Key used to retrieve the parameter.
            
        atype : str
            Attribute type. Authorized values are, cattrs (Coordinate Attribute),
            sattrs (Sensor Attribute), or vattrs (Variable Attribute)
            
        ktype : str
            Key type. Authorized values are short_name, standard_name, or 
            sensor_name (only if atype is sattrs).
            
        Returns
        -------
        attrs : dict
            Dictionary containing the attributes corresponding to input key value.

        '''
        if atype not in ['vattrs', 'cattrs']:
            msg = 'atype can only be vattrs or cattrs; not {}'.format(ktype)
            raise ValueError(msg)
            
        if ktype not in ['short_name', 'standard_name']:
            msg = 'ktype can only be short_name or standard_name; not {}'.format(ktype)
            raise ValueError(msg)
            
            
        if atype == 'cattrs':
            attrs = self._get_cattrs(key, ktype)
            
        elif atype == 'vattrs':
            attrs = self._get_vattrs(key, ktype)
            
        else:
            msg = 'qcattrs is not implemented yet'
            raise NotImplementedError(msg)
        
        return deepcopy(attrs)

    def fill_missing_value(self, arr: np.ndarray, fillValue: (float, int)
                           ) -> np.ndarray:
        '''
        Replace an array's incorrect values with a fillValue.
        Values are considered incorrect if:
            - every element in the array has the same value
            - np.isfinite(val) returns False (inf, NaN, ...)

        Parameters
        ----------
        arr : np.array
            Array to check for missing values.
        fillValue : int
            Default finite value to be used to replace the non-accepted values.

        Returns
        -------
        arr : np.array
            Array with its missing values replaced.

        '''
        # Look for arrays with only unique values
        if len(np.unique(arr)) == 1 and isinstance(arr, (np.generic, np.ndarray)):
            arr = np.full(arr.shape, fillValue)
            
        elif isinstance(arr, (float, int)):
            return arr
            
        # Remove NaNs and other infinite values
        msk = ~np.isfinite(arr)
        arr[msk] = fillValue
        return arr


    def var_to_Dataset(self, data: np.ndarray, coords: dict|None, key: str, atype: str,
                       ktype: str) -> xr.Dataset:
        '''
        Wrapping method to create a fully described (at the exception of the
        global attributes) xr.Dataset object using the attributes defined in the
        cf_netcdf_attrs.py file.

        Parameters
        ----------
        data : np.ndarray
            Data to to be stored into a Dataset.
            
        coords : dict|None, optional
            Dictionary containing the coordinates describing the variable. The keys
            used in the dictionary will be used as the Dataset coordinates short_names.
            
        key : str
            Key used to retrieve the parameter.
            
        atype : str
            Attribute type. Authorized values are, cattrs (Coordinate Attribute),
            qcattrs (QC Attribute), sattrs (Sensor Attribute), or vattrs (Variable Attribute)
            
        ktype : str
            Key type. Authorized values are short_name, standard_name, or 
            sensor_name (only if atype is sattrs).

        Returns
        -------
        ds_var : xr.Dataset
            Fully described variables. CF-1.6 compliant.

        '''
        # Retrieving the variable attributes
        attrs = self.get_attrs(key, atype=atype, ktype=ktype)
        
        # Removing the NaN, inf. and other useless values
        if '_FillValue' in attrs.keys() and coords is not None:
            self.fill_missing_value(arr=data, fillValue=attrs['_FillValue'])
        
        # Creating the DataArray storing the variable
        if coords is not None:
            dims = list(coords.keys())
        else:
            dims = None
        
        if key in ['depth', 'longitude', 'latitude'] and isinstance(data, (float, int)):
            da_var = xr.DataArray(data, 
                                  attrs  = attrs )
        else:
            da_var = xr.DataArray(data, 
                                  coords = coords,
                                  dims   = dims,
                                  attrs  = attrs )
        
        # Sending the data array to a dataset
        ds_var = da_var.to_dataset(name=attrs['short_name'])
        return ds_var
    

    

class Dataset_GlobalAttributesFormat(GlobAttrs, Params):
    
    gattrs = {}
    
    def __init__(self):
        GlobAttrs.__init__(self)
        self.gattrs = self._gattrs.copy()
    
    def fill_spatio_temporal_gattrs(self, time: list|tuple|np.ndarray,
                                    lon: list|tuple|np.ndarray,
                                    lat: list|tuple|np.ndarray,
                                    depth: list|tuple|np.ndarray)->None:
        
        # Make sure the time is a string
        if isinstance(time[0], datetime.datetime):
            t0 = time[0].strftime(self.dtime_fmt)
            t1 = time[-1].strftime(self.dtime_fmt)
            dt = time[1] - time[0]
            
        elif isinstance(time[0], str):
            t0 = time[0]
            t1 = time[-1]
            dt = datetime.strftime(time[1], self.dtime_fmt) - \
                     datetime.strftime(time[0], self.dtime_fmt)
                     
        elif isinstance(time[0], np.datetime64):
            t0 = datetime64_to_datetime(time[0])
            t1 = datetime64_to_datetime(time[-1]).strftime(self.dtime_fmt)
            
            dt = datetime64_to_datetime(time[1]) - t0
            t0 = t0.strftime(self.dtime_fmt)
            
        else:
            
            raise TypeError
        
        dt = dt.total_seconds()
        tnow = datetime.datetime.utcnow().strftime(self.dtime_fmt)
        
        # Setting up the missing values
        self.gattrs['geospatial_lat_min'] = round(np.nanmin(lat), 5)
        self.gattrs['geospatial_lat_max'] = round(np.nanmax(lat), 5)
        
        self.gattrs['geospatial_lon_min'] = round(np.nanmin(lon), 5)
        self.gattrs['geospatial_lon_max'] = round(np.nanmax(lon), 5)
        
        self.gattrs["geospatial_vertical_min"] = round(np.nanmin(depth), 4)
        self.gattrs["geospatial_vertical_max"] = round(np.nanmax(depth), 4)
        
        self.gattrs['time_coverage_start'] = t0
        self.gattrs['time_coverage_end'] = t1
        self.gattrs['time_coverage_resolution'] = 'PT{dt}S'.format(dt=int(dt))
        
        # Taking care of the file history
        self.gattrs['date_created'] = tnow
        self.gattrs['history'] = '{} - File creation'.format(tnow)
        
        
    def fill_missing_gattrs(self, missing_attrs: dict)->None:
        
        attrs = { key: val for key, val in missing_attrs.items() if key in self.gattrs.keys() }
        self.gattrs.update(attrs)
        
        for key, item in self.gattrs.items():
            if item is None:
                self.gattrs[key] = 'unknown'
                
        




class StreamDataset(Dataset_VariablesFormat, Dataset_GlobalAttributesFormat):
    
    ds = xr.Dataset()
    
    def __init__(self):
        Dataset_GlobalAttributesFormat.__init__(self)
        Dataset_VariablesFormat.__init__(self)
    
    def ingest_data(self, data: np.ndarray, coords: dict|None, key: str, atype: str,
                       ktype: str)->None:
        
        if coords is None and atype not in ['cattrs']:
            msg = '{} cannot be defined without its coordinates!'.format(key)
            raise ValueError(msg)            
            
        if atype == 'sattrs':
            raise ValueError('You are using a legacy value. The sattrs option doesn\'t exist when working with hindcast data.')
        
        elif atype in ['cattrs', 'vattrs']:
            Data = self.var_to_Dataset(data, coords, key, atype, ktype)
            
        else:
            msg = 'atype can only be vattrs or cattrs; not {}'.format(ktype)
            raise ValueError(msg)
            
        self.ds = xr.merge([self.ds, Data], compat='no_conflicts')

            
    def to_netcdf(self, savename: str, directory: str):#TODO: docstring
        '''
        Save the data collected by the Object into a netcdf file.

        '''
        # Exporting the data to netcdf
        fpath = os.path.join(directory, savename)
        self.ds.to_netcdf(path=fpath, format=self._gattrs['version'], encoding=self.encoding)
        
        print(savename, 'was written in', directory)
        
        
    def get_savename(self, t0, t1, country):
        
        # Define the file name
        if isinstance(t0, datetime.datetime):
            t0_str = t0.strftime(self.dtime_fmt_sname)
            t1_str = t1.strftime(self.dtime_fmt_sname)
        elif isinstance(t0, np.datetime64):
            t0_str = t0.astype(datetime.datetime).strftime(self.dtime_fmt_sname)
            t1_str = t1.astype(datetime.datetime).strftime(self.dtime_fmt_sname)
            
        fname = self.fname_fmt.format(cnt=country.upper(), t0=t0_str, t1=t1_str)
        return fname










