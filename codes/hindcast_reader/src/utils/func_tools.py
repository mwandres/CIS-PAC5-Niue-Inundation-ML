# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:05:50 2025

@author: gregoires
"""

from datetime import datetime, timedelta
import numpy as np
from numpy.polynomial import polynomial as Poly
import os
from shapely import Point, get_coordinates, box


def get_surrounding_positions(x:float, y:float, radius:float, \
                              angles:np.ndarray, unit:str='degree') \
                            -> tuple[np.ndarray, np.ndarray]:
    '''
    Quick way to estimate some positions around a central point (x,y). The positions
    (xi,yi) are estimated along a circle of radius r, at every angles theta
    following the trigonometric convention.
    
                xi = x + np.cos(theta) * r
                yi = y + np.sin(theta) * r
                
    Note that there is a flat-earth assumption being made here. So it's better
    not to use this function for large distances.

    Parameters
    ----------
    x : float
        x-coordinate of the central point.
    y : float
        y-coordinate of the central point.
    radius : float
        Radius of the circle along which the positions will be estimated.
    angles : np.ndarray (M,)
        Numpy array of directions, e.g. angles=np.arange(0, 361, 90) will create
        one point every 90°; covering the 4 directions (E, S, W, N).
    unit : str, optional
        Unit of the angles. The default is 'degree'.

    Returns
    -------
    xi : np.ndarray (M,)
        x-coordinates corresponding to a position located on a circle of center
        x, radius r and at angle theta_i.
    yi : np.ndarray (M,)
        y-coordinates corresponding to a position located on a circle of center
        x, radius r and at angle theta_i.

    '''
    
    if unit.lower() == 'degree':
        angles = np.deg2rad(angles)
        
    xi = x + np.cos(angles) * radius
    yi = y + np.sin(angles) * radius
    return xi, yi


def get_filenames(fpath:str, t0:datetime, t1:datetime)->list[str]:
    '''
    Wrapping function to parse the netcdf files based on their filenames. The 
    following methods were implemented based on files using the following name
    format: "location_YYYYMM.nc"

    Parameters
    ----------
    fpath : str, path
        Path pointing to the directory containing the netcdf files.
    t0 : datetime.datetime
        1st date at which the data is extracted.
    t1 : datetime.datetime
        Last date at which the data is extracted.

    Returns
    -------
    fout : list[str]
        List containing the names of the netcdf files with data between t0 and t1.

    '''
    
    t0 = datetime(year=t0.year, month=t0.month, day=1)
    t1 = datetime(year=t1.year, month=t1.month, day=1)
    fout = []
    
    for f in os.listdir(fpath):
        if not f.endswith('.nc'):
            continue
        datestr = f.split('_')[1]
        date = datetime(year=int(datestr[:4]), month=int(datestr[4:6]), day=1)
    
        if date >= t0 and date <= t1:
            fout.append(f)
    return fout

def datetime64_to_datetime(datetime64_:np.datetime64)->datetime:
    '''
    Convert a np.datetime64 object into a datetime.datetime object.

    Parameters
    ----------
    datetime64_ : np.datetime64
        Original np.datetime64 object.

    Returns
    -------
    out : datetime.datetime
        Converted input into a datetime.datetime type.
    '''
    datetime64_ = np.datetime64(datetime64_, 's')
    unix_epoch = np.datetime64(0, 's')
    one_second = np.datetime64(1, 's')
    seconds_since_epoch = (datetime64_ - unix_epoch)
    out = datetime.utcfromtimestamp(seconds_since_epoch.astype(int))
    return out

def datenum_to_datetime(datenum:float)->datetime:
    '''
    Convert Matlab datenum into Python datetime.

    Parameters
    ----------
    datenum : float
        Date in datenum format.

    Returns
    -------
    datetime
        Datetime object corresponding to datenum.

    '''
    
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    
    out = datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)
    return out

def get_outliers(data:np.ndarray, k:int, method:str='std')->np.ndarray[bool]:
    '''
    Quick function used to generate a mask giving the positions of the outliers
    in a numpy array.
    
    Note that if you want to find the problematic points between two timeseries,
    you can pass the error (i.e., data = observation - prediction) to this function.

    Parameters
    ----------
    data : np.ndarray (M,)
        Array in which the outliers will be estimated.
    k : int
        Multiplier for thresholding (e.g., k=2 means mean ± 2*std).
    method : str, optional
        "std" for standard deviation method, "iqr" for Inter-Quartile Range 
        method. The default is 'std'.

    Returns
    -------
    msk : np.ndarray[bool] (M,)
        Numpy array of booleans with the same dimensions as data. Detected outliers
        are given as True.

    '''
    if method.lower() == 'std':
        err_bar = np.nanmean(data)
        err_std = np.nanstd(data)
        
        vmin = err_bar - err_std * k
        vmax = err_bar + err_std * k
        
    elif method.lower() == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        vmin = Q1 - k * IQR
        vmax = Q3 + k * IQR
        
    else:
        msg = 'method={} is not an accepted keyword'.format(method)
        raise ValueError(msg)
        
    msk = ((data > vmax) | (data < vmin))
    return msk


def get_map_grid_definition(bbox:box)->tuple[np.ndarray, np.ndarray]:
    '''
    This function was created to be used with the Basemap library. It returns
    the (x,y) coordinates of the tick labels displayed on the map.

    Parameters
    ----------
    bbox : box (shapely.box)
        box object from the shapely library.

    Returns
    -------
    xticks : np.ndarray
        x-axis ticks positions.
    yticks : np.ndarray
        y-axis ticks positions.

    '''
    coord = get_coordinates(bbox)
    x_extent = coord[:,0].max() - coord[:,0].min()
    y_extent = coord[:,1].max() - coord[:,1].min()
    
    if x_extent <= 0.75:
        dx = 0.25
    elif x_extent > 3:
        dx = 1
    else:
        dx = 0.5
        
    if y_extent <= 0.75:
        dy = 0.25
    elif y_extent > 3:
        dy = 1
    else:
        dy = 0.5
    
    # +0.1 to keep the end point in the values
    xticks = np.arange(np.floor(coord[:,0].min()), np.ceil(coord[:,0].max()) + 0.1, dx)
    yticks = np.arange(np.floor(coord[:,1].min()), np.ceil(coord[:,1].max()) + 0.1, dy)
    return xticks, yticks



class BasicEstimators:
    
    def __init__(self, y_est, y_exp):
        # Definition of the basic estimators
        self.y_est = y_est                                                      # Estimated values
        self.y_exp = y_exp                                                      # Expected values
        self.N = len(self.y_est)                                                # Size of the Sample
        
        # Estimators
        self.mse = self.MSE()                                                   # Mean Squared Error [Unit^2]
        self.rmse = self.RMSE()                                                 # Root Mean Squared Error [Unit]
        self.r2 = self.CORR_COEFF()                                             # Correlation Coefficients (Pearson's)
        self.linreg_coeff = self.LINREG_COEFF()                                 # Linear Regression Coefficients (y = ax + b, linreg_coeff[0]=b, linreg_coeff[1]=a)
        
    def MSE(self):
        ''' Mean Squared Error '''
        return np.power(self.y_exp - self.y_est, 2).mean() 
        
    def RMSE(self):
        ''' Root Mean Squared Error'''
        return np.sqrt( np.power(self.y_exp - self.y_est, 2).mean() )
    
    def CORR_COEFF(self):
        ''' Pearson Correlation coefficient (r2)'''
        return np.corrcoef(self.y_exp, self.y_est)[0, 1]
    
    def LINREG_COEFF(self):
        ''' Linear Regression Coefficients (Least Square fit) '''
        return Poly.polyfit(self.y_exp, self.y_est, deg=1)
    
    def to_str(self):
        str_fmt = 'r2: {:.3f}, rmse: {:.3f}\n' + \
                  'y = {:.3f}x + {:.3f}'
        return str_fmt.format(self.r2, self.rmse, self.linreg_coeff[1], self.linreg_coeff[0])


