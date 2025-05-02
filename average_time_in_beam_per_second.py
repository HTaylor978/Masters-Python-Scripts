import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def read_binary_doubles(folder):
    """Reads all binary double files in a folder."""
    storage_vector = []
    for filename in os.listdir(folder):
        # Searching for binary files that match the naming pattern.
        if filename.startswith("time_in_beam_per_second_seed") and filename.endswith(".bin"):
            full_path = os.path.join(folder, filename)
            try:
                with open(full_path, mode='rb') as file:
                    while chunk := file.read(8):
                        value = struct.unpack('d', chunk)[0]
                        storage_vector.append(value)
            except FileNotFoundError:
                print(f"File not found: {full_path}")
    return storage_vector


def process_time_inside_beam_data(base_directory, folder_names, beam_radii):
    """
    Processes data for time inside beam per second,
    computing averages and errors.
    """
    averages = []
    errors = []

    for folder_name, radius in zip(folder_names, beam_radii):
        folder_path = os.path.join(base_directory, folder_name)
        data = read_binary_doubles(folder_path)

        # Filter out negative values
        filtered_data = [x for x in data if x >= 0]
        if len(filtered_data) < len(data):
            print(
                f"Warning: Negative values detected and removed in folder {folder_path}")

        if filtered_data:  # Check if data is not empty after filtering
            avg_time = np.mean(filtered_data)
            error = np.std(filtered_data) / np.sqrt(len(filtered_data))
            averages.append(avg_time)
            errors.append(error)
        else:
            print(
                f"No valid data for time inside beam in folder: {folder_path}")
            averages.append(0)  # Append 0 for empty data
            errors.append(0)  # Append 0 for empty data

    return averages, errors


def power_law(x, a, b):
    return a * np.power(x, b)


def main():
    # Input parameters
    # Different temperature values
    temperatures = ["50K", "1K", "0.25K", "0.01K", "0.001K"]
    # Different particle numbers
    particle_counts = ["50", "50", "50", "50", "50"]
    # Colors for different plots
    colors = ['b', 'g', 'r', 'm', 'orange']
    base_directory = "C:/Users/harry/OneDrive/Documents/Masters/Code/Sim data folders/"

    # Folder names
    folder_names = ["r0_00075", "r0_0005", "r0_00025", "r0_000175", "r0_0001",
                    "r0_000075", "r0_00005", "r0_000025", "r0_0000175", "r0_00001"]  # Updated folder names (r0_x)

    beam_radii = [0.75, 0.5, 0.25, 0.175, 0.1,
                  0.075, 0.05, 0.025, 0.0175, 0.01]  # mm

    plt.figure(figsize=(10, 7))

    color_idx = 0
    for temp, particles in zip(temperatures, particle_counts):
        # Adjusted temperature and particle count folder names
        temp_directory = os.path.join(
            base_directory, f"T{temp.replace('K', '').replace('.', '_')}_n{particles}")  # Updated folder format (T0_25_n50)

        averages, errors = process_time_inside_beam_data(
            temp_directory, folder_names, beam_radii)

        # Fit power law
        popt, pcov = curve_fit(power_law, beam_radii,
                               averages, sigma=errors, absolute_sigma=True, maxfev=10000)
        fitted_x = np.logspace(np.log10(min(beam_radii)),
                               np.log10(max(beam_radii)), 100)
        fitted_y = power_law(fitted_x, *popt)

        # Plot results for each temperature and particle count
        plt.errorbar(beam_radii, averages, yerr=errors, fmt='o',
                     color=colors[color_idx % len(colors)], ecolor='gray', capsize=4, label=f"T={temp}, n={particles}")
        plt.plot(fitted_x, fitted_y, color=colors[color_idx % len(colors)],
                 linestyle='--', label=f'Fit T={temp}, n={particles}: y = {popt[0]:.3f} * x^{popt[1]:.3f}')
        color_idx += 1

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Beam Radius (mm) [log scale]")
    plt.ylabel("Time Inside Beam per Second [log scale]")
    plt.title(
        "Log-Log Plot: Time Inside Beam per Second vs Beam Radius at Different Temperatures")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
