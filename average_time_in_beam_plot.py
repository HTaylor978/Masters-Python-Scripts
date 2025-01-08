import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def read_binary_doubles_and_ignore_negatives(folder, filename, multiplicative_factor=1):
    """Reads binary doubles from a file, ignoring negative values."""
    full_path = os.path.join(folder, filename)
    storage_vector = []
    try:
        with open(full_path, mode='rb') as file:
            while chunk := file.read(8):
                value = struct.unpack('d', chunk)[0]
                if value >= 0:
                    storage_vector.append(value * multiplicative_factor)
    except FileNotFoundError:
        print(f"File not found: {full_path}")
    return storage_vector


def calculate_total_time_in_beam(folder, file_base_name, num_files):
    """
    Calculates the total time spent in the beam for all trajectories in a folder.
    """
    total_time_in_beam = 0
    num_data_points = 0
    used_files = 0

    for itraj in range(num_files):
        filename = f"{file_base_name}_itraj{itraj}.bin"
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):  # Check if the file exists
            time_data = read_binary_doubles_and_ignore_negatives(
                folder, filename)
            # Sum up time in beam for each trajectory
            total_time_in_beam += sum(time_data)
            num_data_points += len(time_data)
            used_files += 1
        else:
            print(f"Skipping missing file: {full_path}")

    average_time_per_second = total_time_in_beam / used_files

    # Error estimation
    error = average_time_per_second / \
        np.sqrt(num_data_points) if num_data_points > 0 else 0

    return average_time_per_second, error

# Linear model function for log-transformed data


def linear_model(log_x, b, log_a):
    return b * log_x + log_a


def main():
    # Input parameters
    # Base directory containing all folders
    base_directory = "C:/Users/harry/OneDrive/Documents/Masters/Code/T = 50K, n = 50, 0.1dt"
    file_base_name = "t_inside_beam_seed1"
    num_files = 50

    folder_names = ["r = 0.00075", "r = 0.0005", "r = 0.00025", "r = 0.0001",
                    "r = 0.000075", "r = 0.00005", "r = 0.000025", "r = 0.00001"]
    beam_radii = [0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025]
    simulation_times = [1.0, 1.0, 1.5, 2.0, 2.5, 4.0, 7.5]

    # Store results
    average_times_per_second = []
    errors = []

    for folder_name, radius, simulation_time in zip(folder_names, beam_radii, simulation_times):
        folder_path = os.path.join(base_directory, folder_name)
        average_time_per_second, error = calculate_total_time_in_beam(
            folder_path, file_base_name, num_files
        )

        average_times_per_second.append(average_time_per_second)
        errors.append(error)

        print(
            f"Beam Radius: {radius}, Average Time per Second: {average_time_per_second}, Error: {error}")

    # Log-transform the data (natural log)
    log_beam_radii = np.log(beam_radii)
    log_average_times_per_second = np.log(average_times_per_second)
    # Error propagation for logs
    log_errors = np.array(errors) / np.array(average_times_per_second)

    # Fit the log-transformed data to the linear model
    popt_log, _ = curve_fit(
        linear_model, log_beam_radii, log_average_times_per_second, sigma=log_errors, absolute_sigma=True)

    # Extract fitted parameters
    b_fit, log_a_fit = popt_log
    a_fit = np.exp(log_a_fit)  # Convert log(a) back to original a

    print(f"Fitted parameters: a = {a_fit:.6f}, b = {b_fit:.6f}")

    # Generate predictions
    x_fit = np.linspace(min(beam_radii), max(beam_radii), 500)
    y_fit = a_fit * x_fit**b_fit  # Power-law fit in the original space

    # Calculate residuals (difference between actual and predicted values)
    y_pred = a_fit * np.array(beam_radii)**b_fit
    residuals = np.array(average_times_per_second) - y_pred

    # Calculate chi-squared and reduced chi-squared
    chi_squared = np.sum((residuals / np.array(errors))**2)
    degrees_of_freedom = len(beam_radii) - len(popt_log)
    reduced_chi_squared = chi_squared / degrees_of_freedom

    print(f"Chi-squared: {chi_squared:.3f}")
    print(f"Reduced Chi-squared: {reduced_chi_squared:.3f}")
    print(f"Residuals: {residuals}")

    # Plot results (log-log scale)
    plt.figure(figsize=(8, 6))
    plt.errorbar(beam_radii, average_times_per_second, yerr=errors, fmt='x',
                 color='k', ecolor='r', capsize=4, label="Data with Errors")
    plt.plot(x_fit, y_fit, color='b',
             label=f'Power-law Fit: $y = {a_fit:.3f}x^{{{b_fit:.2f}}}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Beam Radius (mm) [log scale]")
    plt.ylabel("Average Time Inside Beam per Second [log scale]")
    plt.title("Log-Log Plot: Average Time Inside Beam per Second vs Beam Radius")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
