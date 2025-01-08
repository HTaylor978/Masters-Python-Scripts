# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:06 2024

@author: dhodgkin
"""
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def read_binary_doubles_and_multiply(folder_name, start_filename, storage_vector, seed, itraj, multiplicative_factor):
    """
    Reads binary data from a file and appends the values (multiplied by the given factor) to the storage_vector.
    """
    full_path = os.path.join(
        folder_name, f"{start_filename}_seed{seed}_itraj{itraj}.bin")
    file_length_in_bytes = os.path.getsize(full_path)
    num_data_points = int(file_length_in_bytes / 8)

    with open(full_path, mode='rb') as file:
        for _ in range(num_data_points):
            a = struct.unpack('d', file.read(8))
            storage_vector.append(a[0] * multiplicative_factor)
    return


def calculate_median(data):
    """
    Calculates the median of a list of values.
    Returns the median if the list is not empty, otherwise None.
    """
    if len(data) == 0:
        return None
    sorted_data = sorted(data)
    mid = len(sorted_data) // 2
    if len(sorted_data) % 2 == 0:
        # Even number of elements: median is average of two middle values
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        # Odd number of elements: median is the middle value
        return sorted_data[mid]


def calculate_mean(data):
    """
    Calculates the mean of a list of values.
    Returns the mean if the list is not empty, otherwise None.
    """
    if len(data) == 0:
        return None
    return sum(data) / len(data)


def calculate_time_inside_per_second(data, total_simulation_time):
    """
    Calculates the time spent inside the beam per second.
    Assumes the diagnostic outputs cumulative time inside the beam for each crossing.
    """
    if total_simulation_time <= 0:
        raise ValueError("Total simulation time must be greater than zero.")
    if len(data) == 0:
        return None
    total_time_inside = sum(data)  # Total time spent inside the beam
    return total_time_inside / total_simulation_time  # Normalize per second


# Close all previous plots
plt.close("all")

# Directory where binary files are stored
directory = "C:/Users/harry/OneDrive/Documents/Masters/Code/z pos, n = 50, r = 0.0001, no z 1.0s"

# Base part of the filename
filename = "z_enter_beam"

# Initialize an empty list to store all times
t_inside = []

# Parameters
seed = 1  # Seed is fixed
# Replace this with the total number of trajectories you want to read
num_trajectories = 50
simulation_duration = 20.0 * num_trajectories  # Total simulation time in seconds

# Read data from all trajectory files
for itraj in range(num_trajectories):
    try:
        read_binary_doubles_and_multiply(
            directory, filename, t_inside, seed, itraj, 1)
    except FileNotFoundError:
        print(f"File for trajectory {itraj} not found. Skipping...")
    except Exception as e:
        print(f"Error reading file for trajectory {itraj}: {e}")

# Filter out negative values
# t_inside = [value for value in t_inside if value >=
#            0 and value <= 0.0008 and value >= 0.00002]

# Calculate median
median_time = calculate_median(t_inside)
if median_time is not None:
    print(
        f"The median time spent inside the beam is {median_time:.6f} seconds.")
else:
    print("No valid data found to calculate the median.")

# Calculate mean
mean_time = calculate_mean(t_inside)
if mean_time is not None:
    print(f"The mean time spent inside the beam is {mean_time:.6f} seconds.")
else:
    print("No valid data found to calculate the mean.")

# Calculate time spent inside the beam per second
try:
    time_inside_per_second = calculate_time_inside_per_second(
        t_inside, simulation_duration)
    print(
        f"The time spent inside the beam per second is {time_inside_per_second:.6f} seconds.")
except ValueError as e:
    print(f"Error calculating time inside per second: {e}")

# Plot a histogram of the data
plt.figure()
plt.hist(t_inside, bins=100, color='k', alpha=0.75)
plt.xlabel("Z position (m) (1s before data recorded)")
plt.ylabel("Frequency")
plt.title("Histogram of Z positions particle has entered/exited the beam")
plt.grid(True)
plt.show()
