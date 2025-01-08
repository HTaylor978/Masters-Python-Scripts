# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:31:56 2023

@author: dhodgkin
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #this is used, don't delete it
import general_function_definitions as functions
import general_constants as const
from scipy.stats import gaussian_kde
from scipy.stats import kde
import matplotlib.ticker as ticker

plt.close("all")

directory = "D:\\CERNBox\\microwave_flat_2023_4_sweep_low_clearing_sweep_new_init_current_noise_10ms_average\\"

filename = "t_wallhits"
hf_filename = "initial_hf_state"

t = []
hf_state = []
queue = 100
numtraj = 20
init_E = []

for seed in range(1, queue + 1):
    try:
        functions.read_binary_doubles_change_timestep(directory, filename , t, seed,numtraj, '5e-07')
        functions.read_binary_doubles_change_timestep(directory, hf_filename , hf_state, seed, numtraj, '5e-07')
    except:
        print(seed, " read failure")
    
print(len(t), len(init_E))

c_t = []
d_t = []
for i in range(len(t)):
    if(hf_state[i] == 0.0):
        c_t.append(t[i])
    elif(hf_state[i] == 1.0):
        d_t.append(t[i])
    else:
        print("error: state: ", hf_state[i])
        
fig, ax = functions.build_fig(0.7, 0.45)

bin_width = 0.05
plt.hist(c_t, color = 'm', bins = functions.calc_n_bins(bin_width, c_t), alpha = 0.5, label = 'c')
plt.hist(d_t, color = 'c', bins = functions.calc_n_bins(bin_width, d_t), alpha = 0.5, label = 'd')



plt.axvline(5, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(54, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(63, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(112, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(62, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(111, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(120, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(53, c = 'k', linestyle = 'dashed', alpha = 0.5)
#plt.yscale("log")
plt.legend()
plt.xlabel("Time (s)")

#plt.xlim(26, 53)
#plt.ylim(0, 176)


cnt = 0

for i in range(len(c_t)):
    if(c_t[i] > 54.0 and c_t[i] < 62.0):
        cnt += 1
print(cnt)        
cnt = 0

for i in range(len(c_t)):
    if(c_t[i] > 63 and c_t[i] < 110.0):
        cnt += 1
        

        
print(cnt)

cnt = 0

for i in range(len(c_t)):
    if(c_t[i] > 24 and c_t[i] < 28.0):
        cnt += 1
        

        
print(cnt)

cnt = 0

for i in range(len(c_t)):
    if(c_t[i] > 28 and c_t[i] < 53.0):
        cnt += 1
        

        
print(cnt)


        

"""
plt.axvline(5, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(101.0, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(102.0, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(118.0, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(119.0, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(215.0, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(216.0, c = 'k', linestyle = 'dashed', alpha = 0.5)
plt.axvline(232.0, c = 'k', linestyle = 'dashed', alpha = 0.5)
#plt.yscale("log")
plt.legend()
plt.xlabel("Time (s)")
#ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
#plt.xlim(44, 48)
#plt.ylim(0, 176)

cnt = 0

for i in range(len(c_t)):
    if(c_t[i] > 102 and c_t[i] < 118.0):
        cnt += 1
        

        
print(cnt)

cnt = 0

for i in range(len(c_t)):
    if(c_t[i] > 119 and c_t[i] < 215.0):
        cnt += 1
        

        
print(cnt)

cnt = 0
for i in range(len(c_t)):
    if(c_t[i] > 44 and c_t[i] < 48.0):
        cnt += 1
        
print(cnt)

cnt = 0
for i in range(len(c_t)):
    if(c_t[i] > 48 and c_t[i] < 73):
        cnt += 1
        
print(cnt)

cnt = 0
for i in range(len(c_t)):
    if(c_t[i] > 73 and c_t[i] < 102):
        cnt += 1
        
print(cnt)

"""

