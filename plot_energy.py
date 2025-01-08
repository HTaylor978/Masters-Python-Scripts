# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 00:21:21 2023

@author: dhodgkin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #this is used, don't delete it
import general_function_definitions as functions
import general_constants as const
from scipy.stats import gaussian_kde
from scipy.stats import kde
import harmonic_FRD_functions
from matplotlib import colors
from scipy.interpolate import interp2d
import math

plt.close("all")
directory = "D:\\CERNBox\\laser_cooling_200MHz_det_2nJ_correct_minimum_2\\"

queue = 50
numtraj = 5

bin_width = 0.01

Ez = []
Eperp_no_U = []
t = []
U_cross = []
av_E = []
av_t = []
z = []

for seed in range(1, queue + 1):
            
    try:
        functions.read_binary_doubles(directory, "t_wallhits" , t, seed,numtraj)
        functions.read_binary_doubles(directory,"z_wallhits", z, seed,numtraj)

        functions.read_binary_doubles(directory,"Tz_cross_at_t_100.000000", Ez, seed,numtraj)
        functions.read_binary_doubles(directory,"Tperp_cross_at_t_100.000000", Eperp_no_U, seed,numtraj)
        functions.read_binary_doubles(directory,"U_cross_at_t_100.000000", U_cross, seed,numtraj)
        
        functions.read_binary_doubles(directory, "averaged_over_trajectories_E",av_E, seed,numtraj)
        functions.read_binary_doubles(directory, "averaged_over_trajectories_E",av_t, seed,numtraj)
       

    except:
        print(seed, " read failure")
    
total_E = []

for i in range(len(Ez)):
    if(Ez[i] != -1.0 and Eperp_no_U[i] != -1.0):
        total_E.append(Ez[i] + Eperp_no_U[i] + U_cross[i])

fig, ax = functions.build_fig(0.7, 0.4)
plt.hist(total_E, functions.calc_n_bins(bin_width, total_E))
plt.xlabel("simulated E(K)")

plt.title(r'$\langle E \rangle$ = '+ str(round(1e3*np.mean(total_E))) + r'$\pm$' + str(round(1e3*np.std(total_E)/np.sqrt(len(total_E)))) + "(statistical) mK")
plt.show()

fig, ax = functions.build_fig(0.7, 0.4)
plt.plot(av_t, av_E)
plt.show()