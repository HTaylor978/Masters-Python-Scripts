# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:13:41 2020

@author: dhodgkin
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # this is used, don't delete it
import general_function_definitions as functions
import general_constants as const
from scipy.stats import gaussian_kde
from scipy.stats import kde

plt.close("all")
directory = "C:/Users/harry/OneDrive/Documents/Masters/Code/\\"

# Only initial part of filenames, not seed and numtraj
filename = "average_dist_per_step"

variable = []
seed = 1
numtraj = 50

try:
    functions.read_binary_doubles_and_multiply(
        directory, filename, variable, seed, numtraj, 1)
except Exception as e:
    print(seed, " read failure:", e)

# Calculate average distance per step
average_dist_per_step = np.mean(variable)
# Calculate standard deviation
std_dev = np.std(variable)
# Calculate error on the mean (standard error)
num_data_points = len(variable)
sem = std_dev / np.sqrt(num_data_points)

# Display results
print(f"Average distance per step: {average_dist_per_step:.6f}")
print(f"Standard deviation: {std_dev:.6f}")
print(f"Standard error (error on the mean): {sem:.6f}")

# Plot histogram
plt.figure()
plt.hist(variable, 100, color='k')
plt.xlabel("Average distance travelled per time step")
plt.ylabel("Count")
plt.title("Histogram of average distance travelled per time step")
plt.show()
