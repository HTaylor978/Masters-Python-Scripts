import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def read_binary_doubles(folder, filename):
    """Reads binary doubles from a file."""
    full_path = os.path.join(folder, filename)
    storage_vector = []
    try:
        with open(full_path, mode='rb') as file:
            while chunk := file.read(8):
                value = struct.unpack('d', chunk)[0]
                storage_vector.append(value)
    except FileNotFoundError:
        print(f"File not found: {full_path}")
    return storage_vector


def process_crossings_data(base_directory, folder_names, beam_radii, prefixes):
    """
    Processes data for different types of crossings (total, single-step, multiple-step)
    and computes averages and errors for each type.
    """
    all_averages = {prefix: [] for prefix in prefixes}
    all_errors = {prefix: [] for prefix in prefixes}

    for folder_name, radius in zip(folder_names, beam_radii):
        folder_path = os.path.join(base_directory, folder_name)

        for prefix in prefixes:
            file_name = f"{prefix}_per_second_seed1_numtraj50.bin"
            data = read_binary_doubles(folder_path, file_name)

            if data:  # Check if data is not empty
                avg_crossings = np.mean(data)
                # Standard error of the mean
                error = np.std(data) / np.sqrt(len(data))
                all_averages[prefix].append(avg_crossings)
                all_errors[prefix].append(error)
            else:
                print(f"No data for {prefix} in folder: {folder_path}")
                all_averages[prefix].append(0)  # Append 0 for empty data
                all_errors[prefix].append(0)  # Append 0 for empty data

    return all_averages, all_errors


def main():
    # Input parameters
    base_directory = "C:/Users/harry/OneDrive/Documents/Masters/Code/T = 50K, n = 50, crossings"
    folder_names = ["r = 0.01", "r = 0.0075", "r = 0.005", "r = 0.0025", "r = 0.001",
                    "r = 0.00075", "r = 0.0005", "r = 0.00025", "r = 0.0001", "r = 0.000075", "r = 0.00005", "r = 0.000025", "r = 0.00001"]  # Names of subfolders
    # Corresponding beam radii
    beam_radii = [10, 7.5, 5.0, 2.5, 1.0, 0.75, 0.5,
                  0.25, 0.1, 0.075, 0.05, 0.025, 0.01]  # mm

    # Prefixes for the different types of crossing data
    prefixes = ["single_step_crossings",
                "multiple_step_crossings"]

    # Process crossings data for all prefixes
    all_averages, all_errors = process_crossings_data(
        base_directory, folder_names, beam_radii, prefixes)

    # Plot results for individual types of crossings
    plt.figure(figsize=(10, 7))
    colors = ['g', 'b', 'orange']  # Colors for the lines
    labels = [
        "Multiple-Step Crossings per Second",
        "Single-Step Crossings per Second",
    ]

    for prefix, color, label in zip(prefixes, colors, labels):
        averages = all_averages[prefix]
        errors = all_errors[prefix]
        plt.errorbar(beam_radii, averages, yerr=errors, fmt='o', linestyle='-',
                     color=color, ecolor='r', capsize=4, label=label)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Beam Radius (mm) [log scale]")
    plt.ylabel("Crossings per Second [log scale]")
    plt.title("Log-Log Plot: Crossings per Second vs Beam Radius")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    # Plot ratio of single-step to multiple-step crossings
    single_averages = np.array(all_averages["single_step_crossings"])
    multiple_averages = np.array(all_averages["multiple_step_crossings"])

    # Avoid division by zero
    ratios = np.divide(single_averages, multiple_averages,
                       where=multiple_averages != 0)
    # Set ratio to 0 where multiple crossings are zero
    ratios[multiple_averages == 0] = 0

    plt.figure(figsize=(10, 7))
    plt.plot(beam_radii, ratios, 'o-', color='purple',
             label="Multiple-Step/Single-Step Crossings Ratio")
    plt.xscale('log')
    plt.yscale('Log')
    plt.xlabel("Beam Radius (mm) [log scale]")
    plt.ylabel("Ratio (Multiple-Step / Single-Step)")
    plt.title(
        "Log-Log Plot: Ratio of Multiple-Step to Single-Step Crossings vs Beam Radius")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
