# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:49:10 2025

@author: gregoires
"""

import os
import sys

from src.figures import check_data


savedir = os.path.join('C:\\', 'Users', 'gregoires', 'OneDrive - SPC', 'Desktop', 'savefile', 'plotOutput', 'hindcast', 'subset_tool')

# savedir = os.path.join('S:\\', 'TEMP', 'FJ_NAB', 'Gregoire', 'ForMo', 'Spectra')

root = os.path.join('S:\\', 'GEM', 'FJ_NAB', 'O_M', 'Oceans', 'Active', 'MARINE_SURVEYS_SURVEYS', 'Greggy', 'DATA', 'dev_output')
fpath = os.path.join(root, 'subset_hindcast_Niue_19840101_19840531_v1_gamma_3.3.nc')


# check_data.export_synthetic_spectra(savedir=savedir)


# Plot the area of interest
# check_data.plot_AoI_location(fpath, savedir)

# Plot the timeseries / distributions
# check_data.main_check(fpath, savedir)

# Plot the partitions and the corresponding wave spectra
# check_data.reconstruction_check(fpath, savedir)
# check_data.reconstruction_check_looped(fpath, savedir, k=3)

# Plot the sensiblity
check_data.compare_gamma_sensibility(fdir=root, savedir=savedir)

# Plot the directional distribution's sensitivity to directional spreading
# check_data.check_directional_distribution(savedir)

