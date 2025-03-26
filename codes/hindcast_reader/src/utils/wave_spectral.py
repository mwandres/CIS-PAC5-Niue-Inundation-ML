# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:26:27 2025

@author: gregoires
"""

from scipy.integrate import simpson
from scipy.special import gamma as gamma_func
import numpy as np
import sys
import warnings

# Catching numpy warnings as we are dividing by 0, multiplying with invalid values,
# and encountering overflow warnings quite a few times but I wont change my codes :)
warnings.filterwarnings('ignore', message='.*divide by zero') 
warnings.filterwarnings('ignore', message='.*invalid value') 
warnings.filterwarnings('ignore', message='.*overflow') 

try:
    from typing import Self
except:
    from typing_extensions import Self


def PiersonMoskowitz(Hs:float|np.ndarray, fp:float|np.ndarray, freqs:np.ndarray)->np.ndarray:
    '''
    Pierson-Moskovitz parametric spectrum as described in Rossi et al. 2023.
    
    Szz = 5/16 * Hs^2 * fp^4 * f^-5 * exp(-5/4 * (f/fp)^-4)
    
    Notes: 
        1. This definition doesnt take into account the wind speed. It has,
        instead, been integrated in the peak frequency component
        2. Providing numpy arrays to vectorize the calculations is doable but
        you will need to adjust each input shape so that they can be broadcasted
        into each other
    
    Parameters
    ----------
    Hs : float | np.ndarray
        Significant wave height [Meters].
    fp : float | np.ndarray
        Peak frequency (1/Tp) [Hertz].
    freqs : np.ndarray
        Set of frequencies at which the spectrum will be estimated.

    Returns
    -------
    Spm : np.ndarray
        Pierson-Moskowitz Frequency Spectrum.

    '''
    Spm = (5/16) * (Hs**2) * (fp**4) * (freqs**(-5)) * np.exp(-5/4 * (freqs/fp)**(-4))
    return Spm


def JONSWAP(Hs:float|np.ndarray, fp:float|np.ndarray, freqs:np.ndarray, gamma:float)->np.ndarray:
    '''
    JONSWAP parametric spectrum using an adaption of the Pierson-Moskowitz model
    as described in Rossi et al. 2023.
    
    Notes: 
        1. This definition doesnt take into account the wind speed. It has,
        instead, been integrated in the peak frequency component
        2. Providing numpy arrays to vectorize the calculations is doable but
        you will need to adjust each input shape so that they can be broadcasted
        into each other
    
    Parameters
    ----------
    gamma : flaot
        Peak enhancement factor ranging from 1 to 7.
    Hs : float | np.ndarray
        Significant wave height [Meters].
    fp : float | np.ndarray
        Peak frequency (1/Tp) [Hertz].
    freqs : (M,) np.array
        Set of frequencies at which the spectrum will be estimated.

    Returns
    -------
    Sj : (M,) np.array
        JONSWAP Frequency Spectrum.

    '''
    Spm = PiersonMoskowitz(Hs, fp, freqs)
    A_gamma = 1 - 0.287 * np.log(gamma)
    sigma = np.full(freqs.shape, 0.07)
    sigma[freqs>fp] = 0.09
    
    exp_ = np.exp(-0.5*np.power((freqs - fp) / (sigma * fp), 2))
    Sj = A_gamma * Spm * np.power(gamma, exp_)
    return Sj


def jonswap_spreading_parameter(dspr:float|np.ndarray, unit:str='degree') \
                    ->float|np.ndarray:
    '''
    Estimate the JONSWAP spreading parameter (s) as a function of the waves
    directional spreading following the formula provided in XBeah manual.
            
                        s = 2 / (sigma^2) - 1
    
    Notes: 
       1. Providing numpy arrays to vectorize the calculations is doable but
        you will need to adjust each input shape so that they can be broadcasted
        into each other
    
    
    
    Parameters
    ----------
    dspr : float | np.ndarray (N,)
        Waves directional spreading [rad].
        
    unit : str
        Directional spread's unit. The default is 'degree'

    Returns
    -------
    s : float | np.ndarray (N,)
        JONSWAP spreading parameter.

    '''
    if unit=='degree':
        dspr = np.deg2rad(dspr)
    elif unit=='radian':
        pass
    else:
        msg = 'Only "degree" or "radian" are accepted values, not {}'.format(unit)
        raise ValueError(msg)
    
    s = -1 + 2 / (dspr**2)
    
    # Bellow this threshold (i.e, Dspr=10°), the normalization function will blow up...
    threshold = -1 + 2 / (np.deg2rad(10)**2)
    if isinstance(s, np.ndarray):
        s[s<1] = 1
        s[s>threshold] = threshold
    elif isinstance(s, float) and s < 1:
        s = 1
    elif isinstance(s, float) and s > threshold:
        s = threshold
    
    return s


def cosine_squared_general(theta:np.ndarray, Dir:float|np.ndarray, 
                           s:float|np.ndarray=1, unit:str='degree')->np.ndarray:
    '''
    General form of the cosine-2s spreading function proposed by Mitsuyasu et al, 1975.
    Simillar forms are described in Benoit, 1992 (Eq. 20-21), Rossi, 2023 ...
    
    Notes: 
        1. Providing numpy arrays to vectorize the calculations is doable but
        you will need to adjust each input shape so that they can be broadcasted
        into each other
    
    
    Parameters
    ----------
    theta : (M,) np.array
        Set of direction at which the DSF will be estimated.
    Dir : float | np.ndarray
        Mean direction of propagation.
    s : float | np.ndarray, optional
        Factor governing the degree of spread. The default is 1.
        
    unit : str
        Unit used by theta (angles) and Dir (direction of propagation). The default 
        is 'degree'.

    Returns
    -------
    out : (M,) np.array
        Directional Spreading Function.

    '''
    if unit=='degree':
        theta = np.deg2rad(theta)
        Dir = np.deg2rad(Dir)
    elif unit=='radian':
        pass
    else:
        msg = 'Only "degree" or "radian" are accepted values, not {}'.format(unit)
        raise ValueError(msg)
        
    def normalizing_factor(s):
        G = np.power(2, 2*s-1) * np.power(gamma_func(s+1), 2) / gamma_func(2*s+1)
        return G
    
    # This one is used by SWAN but both are fine
    def normalizing_factor_v2(s):
        G = np.power(2, s) * np.power(gamma_func(0.5*s+1), 2) / gamma_func(s+1)
        return G
    
    
    pDist = np.power(abs(np.cos(0.5*(theta - Dir))), 2*s)
    out = normalizing_factor(s) * pDist / np.pi
    
    out[~np.isfinite(out)] = 0
    out[out < 1e-3] = 0
    return out
    

class FrequencySpectra:
    
    fmin_swell = 0
    fmax_swell = 0.5
    
    fmin_ig = 0.003
    fmax_ig = 0.03

    
    def __init__(self, Fspec, time, freqs):
        
        if time is None and Fspec.ndim == 1:
            Fspec = Fspec[None,...]
        
        # Storing the spectra
        self.Fspec = Fspec

        # Storing the coordinates
        self.time = time
        self.freqs = freqs
        
        # Storing the wave coordinates for the record
        self.wpar = {'HM0': self.HM0(),                                         # Significant wave height
                     'TPEAK': self.TPEAK(),                                     # Peak Period
                     'TM01': self.TM01(),                                       # Average period
                     'TM02': self.TM02(),                                       # Mean Zero-Crossing period
                     'TM-10': self.TM_10()                                      # Spectral Wave Period
                     }
        # I commented these lines out as we are working with IG waves at the moment
        # If you need the IG waves parameters, uncomment them
                     # 'IGHM0': self.HM0(ig=True),                                # Significant wave height - Infragravity waves
                     # 'IGTPEAK': self.TPEAK(ig=True),                            # Peak Period - Infragravity waves
                     # 'IGTM01': self.TM01(ig=True),                              # Average period - Infragravity waves
                     # 'IGTM02': self.TM02(ig=True)}                              # Mean Zero-Crossing period - Infragravity waves

    def get_freq_msk(self, ig:bool=False)->np.ndarray[bool]:
        '''
        Return a mask defined as True at frequency bins defined between fmin and
        fmax.
    
        Parameters
        ----------
        ig : Bool, optional
            Switch to estimate the IG waves or the swell. The default is False.
    
        Returns
        -------
        msk : (M,) np.ndarray[bool]
            True for fmin < f < fmax, False otherwise.
    
        '''
        # Lower frequency cutoff        
        if ig:
            fmin = self.fmin_ig
            fmax = self.fmax_ig
        else:
            fmin = self.fmin_swell
            fmax = self.fmax_swell
            
        # Creating the mask
        msk = self.freqs >= fmin
        msk[self.freqs > fmax] = False
        return msk
    
    def get_peak_frequency_index(self, ig:bool=False)->tuple[int, np.ndarray[bool]]:
        '''
        Return the index of peak frequency. 
        
        Parameters
        ----------
        apply_cutoff : Bool, optional
            Boolean switch used to apply the frequency cutoff values. The default is False.
        
        ig : Bool, optional
            Switch to estimate the IG waves or the swell. The default is False.

        Returns
        -------
        indp : int
            Indeces of the peak frequency.
            
        msk : (freqs,) np.ndarray
            Only returned if apply_cutoff is True. msk is the mask corresponding
            to the frequency used in the estimation of the peak frequency index.
        '''
        msk = self.get_freq_msk(ig=ig)
        indp = np.nanargmax(self.Fspec[:, msk], axis=1)
        return indp, msk
    
    def get_nth_spectral_moment(self, n:int, ig:bool=False)->np.ndarray:
        '''
        Return the Nth spectral moment

        Parameters
        ----------
        n : int
            Nth moment.
        ig : bool, optional
            Set to True to estimate IG waves instead of swell. The default is False.

        Returns
        -------
        m_nth : np.ndarray
            Nth spectral moment given as function of time.

        '''
        if n is None:
            raise ValueError
        
        # Applying a cutoff on the frequencies
        msk = self.get_freq_msk(ig)
        
        # Estimating the spectral moment
        df = np.diff(self.freqs).mean()
        m_nth = np.sum(self.Fspec[:,msk] * np.power(self.freqs[None,msk], n), axis=-1) * df
        return m_nth
        
    def TPEAK(self, ig=False):
        ''' Peak period in [s] '''
        indp, msk = self.get_peak_frequency_index(ig)
        return 1 / self.freqs[msk][indp]
    
    def HM0(self, ig=False):
        ''' Significant wave height in [m] '''
        m0 = self.get_nth_spectral_moment(0, ig)
        return 4*np.sqrt(m0)
    
    def TM_10(self, ig=False): # Tm-10, the - character can't be used in the name
        ''' Spectral wave period in [s] '''
        m0 = self.get_nth_spectral_moment(0, ig)
        m_1 = self.get_nth_spectral_moment(-1, ig)
        return m_1 / m0
    
    def TM01(self, ig=False):
        ''' Average spectral wave period in [s] '''
        m0 = self.get_nth_spectral_moment(0, ig)
        m1 = self.get_nth_spectral_moment(1, ig)
        return m0 / m1
    
    def TM02(self, ig=False):
        ''' Mean spectral ZeroCrossing period in [s] '''
        m0 = self.get_nth_spectral_moment(0, ig)
        m2 = self.get_nth_spectral_moment(2, ig)
        return np.sqrt(m0 / m2)





class EnergySpectra(FrequencySpectra):
    
    def __init__(self, Espec, time, freqs, theta, unit='degree'):
        
        if time is None and Espec.ndim == 2:
            Espec = Espec[None,...]
        
        if unit == 'degree':
            theta = np.deg2rad(theta)
        
        # Storing the spectra
        self.Espec = Espec
        self.theta = theta
        
        # Update the object to get the Frequency Spectra too
        Fspec = self.Espec_to_Fspec()
        super().__init__(Fspec=Fspec, time=time, freqs=freqs)
        

    @classmethod
    def from_jonswap(cls, Hs:float|np.ndarray, Tp:float|np.ndarray, \
                     Dirm:float|np.ndarray, Dspr:float|np.ndarray, \
                     freqs:float|np.ndarray, theta:float|np.ndarray, \
                     time:float|np.ndarray=None, gamma:float=3.3, unit:str='degree')->Self:
        '''
        Generate a (timeseries) of omni-directional variance density spectra
        given a set of wave parameters. 
        
        If only one value is fed to the function, the data will be interpreted 
        as being a timeseries of length 1 (i.e., one timestep).

        Parameters
        ----------
        Hs : float|np.ndarray
            Significant wave height used to generate the JONSWAP spectra.
        Tp : float|np.ndarray
            Peak Period used to generate the JONSWAP spectra.
        Dirm : float|np.ndarray
            Mean Direction of propagation used to generate the directional distribution.
        Dspr : float|np.ndarray
            Directional spread used to generate the directional distribution.
        freqs : float|np.ndarray
            Frequency bins.
        theta : float|np.ndarray
            Directional bins.
        time : float|np.ndarray, optional
            Time. The default is None.
        gamma : float, optional
            JONSWAP peak enhancement factor. The default is 3.3.
        unit : str, optional
            Unit used for Dspr, Dm, theta. The default is 'degree'.

        Returns
        -------
        Self
            Omni-directional variance density spectrum encapsulated in an object
            of this class.

        '''
        
        if time is None and not isinstance(Hs, np.ndarray):
            Hs = np.array(Hs)
            Tp = np.array(Tp)
            Dirm = np.array(Dirm)
            Dspr = np.array(Dspr)
        
        # Adjusting the dimensions to allow for broadcasting (time, freqs, theta)
        Hs = Hs[:,None,None]
        fp = 1 / Tp
        Dirm = Dirm[:,None,None]
        Dspr = Dspr[:,None,None]
        
        freqs_ = freqs[None,:,None] * np.ones((len(Hs), len(freqs), len(theta)))
        theta_ = theta[None,None,:] * np.ones((len(Hs), len(freqs), len(theta)))
        
        Fspec = JONSWAP(Hs=Hs, fp=fp[:,None,None], freqs=freqs_, gamma=gamma)
        s = jonswap_spreading_parameter(dspr=Dspr, unit=unit)
        DirDist = cosine_squared_general(theta_, Dirm, s=s, unit=unit)
        
        Espec = Fspec * DirDist
        return cls(Espec=Espec, time=time, freqs=freqs, theta=theta, unit=unit)

    def Espec_to_Fspec(self)->np.ndarray:
        '''
        Wrapper to estimate the frequency spectrum (S(f), variance density spectrum) 
        from the omni-directional variance density spectrum. 
        
        In practice, we simply integrate the full spectrum along its directional
        dimension.

        Returns
        -------
        Fspec : np.ndarray
            Variance density spectrum, also called Frequency Spectrum.

        '''
        Fspec = simpson(self.Espec, x=self.theta, axis=-1)
        return Fspec
    
    def spec_units_conversion(self, rad2deg:bool=True)->tuple[np.ndarray, np.ndarray]:
        '''
        So here is the tricks: the omni-directional spectra is defined with the help of 
        trigonometric functions which requires the data to be in RADIANS. However, 
        for simplicity reasons we like to work in DEGREES.
        
        Furthermore, the spectra we have are density spectra, i.e., the energy / variance [...]
        is distributed across frequency / directional bins. Thus we can't simply
        convert the coordinate theta from radians to degrees or we will mess up
        the whole distribution.
        
        The conversion is done by changing the variable: 
            - radians to degrees -> E(f,theta) = E(f,theta) * pi / 180
            - degrees to radians -> E(f,theta) = E(f,theta) * 180 / pi

        Parameters
        ----------
        rad2deg : bool, optional
            Convert from radians to degrees if True, and from Degrees to radians 
            if False. The default is True.

        Returns
        -------
        Espec : np.ndarray
            Converted omni-directional spectra.
        theta : np.ndarray
            Converted coordinates.

        '''
        if rad2deg:
            theta = np.rad2deg(self.theta)
            Espec = self.Espec * np.pi / 180
        else:
            theta = np.deg2rad(self.theta)
            Espec = self.Espec * 180 / np.pi
        return Espec, theta
    
    
    



