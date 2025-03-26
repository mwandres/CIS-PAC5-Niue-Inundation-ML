# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:56:27 2024

@author: gregoires
"""


import json
import os


class Params:
    dtime_fmt = "%Y-%m-%dT%H:%M:%SZ"
    dtime_fmt_sname = "%Y%m%d"
    fname_fmt = 'subset_hindcast_{cnt}_{t0}_{t1}.nc'
    
    encoding = { 'time': 
                    {'dtype' : 'float', 
                     'units' : "days since 1950-01-01T00:00:00Z", 
                     'calendar' : "proleptic_gregorian" }
                }

        
class BaseAttrs:
    
    @staticmethod
    def _load_attrs(fpath):
        with open(fpath, 'r') as f:
            data = f.read()
        fcontent = json.loads(data)
        return fcontent

    @staticmethod
    def _export_attrs(fcontent, fpath):
        with open(fpath, 'w') as f:
            json.dump(fcontent, f, indent=4, sort_keys=False)
        
        
class VarAttrs(BaseAttrs):
    '''
    Class containing the CF compliant Attributes related to physical parameters
    E.g.: Wave heights ...
    '''
    
    def __init__(self):
        fpath = os.path.abspath(\
                os.path.join(os.path.dirname(os.path.realpath(__file__)), 'metadata_CF', 'CF_variables_attributes.json'))  
        
        self._vattrs_fpath = fpath
        self._vattrs = self._load_attrs(fpath)
        
    
    
class CoordAttrs(BaseAttrs):
    '''
    Class containing the CF compliant Attributes related to coordinates
    E.g.: Longitude, Latitude, Time, Depth ...
    '''
    
    def __init__(self):
        fpath = os.path.abspath(\
                os.path.join(os.path.dirname(os.path.realpath(__file__)), 'metadata_CF', 'CF_coordinates_attributes.json'))  
        
        self._cattrs_fpath = fpath
        self._cattrs = self._load_attrs(fpath)



class GlobAttrs(BaseAttrs):
    '''
    Class containing the CF compliant Global Attributes used to fill a NetCDF file
    E.g.: site_code, netcdf_version, history ...
    
    Note that this class slightly differs from the others as it needs the name 
    of the global attributes set that will be used to instantiate itself. This 
    allows switching from one standard to another (not implemented yet)
    '''
    
    def __init__(self, gattrs_name='OceanSites'):
        fpath = os.path.abspath(\
                os.path.join(os.path.dirname(os.path.realpath(__file__)), 'metadata_CF', 'CF_global_attributes.json'))  
            
        self._gattrs_fpath = fpath
        
        gattrs = self._load_attrs(fpath)
        self._gattrs = gattrs[gattrs_name].copy()
        self._full_gattrs = gattrs.copy()
        
    def switch_gattrs_to(self, gattrs_name):
        self._gattrs = self._full_gattrs[gattrs_name].copy()
