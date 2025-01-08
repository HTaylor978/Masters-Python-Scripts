# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:40:55 2024

@author: harry
"""
import struct
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # this is used, don't delete it
import general_function_definitions as functions
import general_constants as const
from scipy.stats import gaussian_kde
from scipy.stats import kde

plt.close("all")
directory = "C:/Users/harry/OneDrive/Documents/Masters/Code/Sim data folders/Working batch\\"

# Only initial part of filenames, not seed and numtraj
perp_energy_filename = "Tperp_cross"
parallel_energy_filename = "Tz_cross"
magnetic_potential_energy_filename = "U_cross"

seed = 1


def read_binary_doubles_and_multiply(folder_name, start_filename, storage_vectors, seed, numtraj, multiplicative_factor):
    """
    Reads binary data line by line from a file and multiplies each entry by a given factor.
    Each line (corresponding to a trajectory) is stored in storage_vectors.
    """
    full_path = os.path.join(
        folder_name, f"{start_filename}_seed{seed}_numtraj{numtraj}.bin")
    file_length_in_bytes = os.path.getsize(full_path)
    num_data_points = int(file_length_in_bytes /
                          (8 * numtraj))  # 8 bytes per double

    # Ensure storage vectors have correct structure
    for traj_index in range(numtraj):
        if len(storage_vectors[traj_index]) != num_data_points:
            storage_vectors[traj_index] = []

    with open(full_path, mode='rb') as file:
        for _ in range(num_data_points):
            for traj_index in range(numtraj):
                a = struct.unpack('d', file.read(8))
                storage_vectors[traj_index].append(
                    a[0] * multiplicative_factor)

    return


def create_energy_plot(filename, final_time, time_step, numtraj):
    trajectories = [[] for _ in range(numtraj)]
    time_values = np.arange(time_step, final_time + time_step, time_step)

    for time in time_values:
        formatted_time = f"{time:.6f}"
        current_filename = f"{filename}_at_t_{formatted_time}"
        #print(f"Reading file: {current_filename}")

        temp_storage = [[] for _ in range(numtraj)]

        try:
            read_binary_doubles_and_multiply(
                directory, current_filename, temp_storage, seed, numtraj, 1)
        except Exception as e:
            print(f"Read failure for {current_filename}: {e}")
            continue

        # Accumulate values over time for each particle
        for i in range(numtraj):
            # Replace -1 with NaN to break the plot line for annihilated particles
            trajectories[i].extend(
                [val if val != -1 else np.nan for val in temp_storage[i]])

    plt.figure(figsize=(10, 6))
    for i, trajectory in enumerate(trajectories):
        plt.plot(time_values, trajectory[:len(
            time_values)], label=f'Trajectory {i + 1}')

    plt.xlabel('Time (s)')
    plt.ylabel(f'Axial Kinetic Energy (K)')
    plt.title(f'Axial Kinetic Energy Trajectories Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename} Energy Trajectories Over Time")
    plt.show()

    return trajectories


def plot_total_energy(perp, parallel, magnetic, time_values):
    """
    Creates seperate plots of the total energy of each particle at each readout
    time
    """
    perp = np.array(perp)
    parallel = np.array(parallel)
    magnetic = np.array(magnetic)

    # Replace -1 with NaN
    perp[perp == -1] = np.nan
    parallel[parallel == -1] = np.nan
    magnetic[magnetic == -1] = np.nan

    num_particles = perp.shape[0]  # Get number of particles

    # Calculate total energy for each particle
    for i in range(num_particles):
        # Calculate total energy as the sum of perpendicular, parallel, and magnetic energies
        total_energy = perp[i] + parallel[i] + magnetic[i]

        # Plot total energy for this particle
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, total_energy,
                 label=f'Particle {i + 1}', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Total Energy (Kelvin)')
        plt.title(f'Total Energy of Particle {i + 1} Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Total energy of Particle {i + 1} Over Time")
        plt.show()


def plot_potential_energy(perp, magnetic, time_values):
    """
    Creates plot of summed perpendicular kinetic energy and magnetic potential
    energy in order to create contrasting plot to parallel kinetic energy
    """
    perp = np.array(perp)
    magnetic = np.array(magnetic)

    # Replace -1 with NaN
    perp[perp == -1] = np.nan
    magnetic[magnetic == -1] = np.nan

    num_particles = perp.shape[0]  # Get number of particles

    # Calculate summed energy as the sum of perpendicular and magnetic energies
    summed_energy = perp + magnetic

    plt.figure(figsize=(10, 6))
    for i in range(num_particles):
        plt.plot(time_values, summed_energy[i, :], label=f'Trajectory {i + 1}')

    plt.xlabel('Time (s)')
    plt.ylabel('Perpendicular Kinetic Energy + Magnetic Potential Energy (K)')
    plt.title(
        'Summed Perpendicular Kinetic and Potential Energy Trajectories Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "Summed Perpendicular Kinetic and Potential Energy Trajectories Over Time")
    plt.show()


def main():
    # Use functions to plot evolution of 10 particles
    perp_trajectories = create_energy_plot(perp_energy_filename, 10.0, 0.1, 10)
    parallel_trajectories = create_energy_plot(
        parallel_energy_filename, 10.0, 0.1, 10)
    magnetic_trajectories = create_energy_plot(
        magnetic_potential_energy_filename, 10.0, 0.1, 10)

    plot_potential_energy(
        perp_trajectories, magnetic_trajectories, np.linspace(0.1, 10.0, 100))

    # Create plots of total energy of each particle
    plot_total_energy(perp_trajectories, parallel_trajectories,
                      magnetic_trajectories, np.linspace(0.1, 10.0, 100))


if __name__ == "__main__":
    main()
