# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:20:39 2023

@author: gregoires
"""

import matplotlib.pyplot as plt

# =============================================================================
# Defining the plots parameters
# =============================================================================
SMALL_SIZE              =       8
MEDIUM_SIZE             =       10
BIGGER_SIZE             =       12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title