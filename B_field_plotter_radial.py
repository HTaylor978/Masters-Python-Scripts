# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:58:36 2025

@author: harry
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the base directory where the field data files are stored
base_dir = "C:/Users/harry/OneDrive/Documents/Masters/Code/Sim data folders/B_fields/"
angles = ["-22.5", "0", "22.5"]

# Field file names
field_files = ["MIRA_1A.txt", "MIRB_1A.txt", "MIRD_1A.txt",
               "MIRE_1A.txt", "SOLA_1A.txt", "SOLB_1A.txt", "OCT_1A.txt"]

# Prefactors for each field file
prefactors = {
    "MIRA_1A.txt": 1.0,  # 490.2/2959.36,
    "MIRB_1A.txt": 1.0,  # -59.09/1479.68,
    "MIRD_1A.txt": 1.0,  # -59.13/1479.68,
    "MIRE_1A.txt": 1.0,  # 482.1/2959.36,
    "SOLA_1A.txt": 1.0,  # 250.06/1479.68,
    "SOLB_1A.txt": 1.0,  # 249.98/1479.68,
    "OCT_1A.txt": 1.0  # Adjust as needed
}

# Generate position values from 0 to 0.6 with 0.001 increments
positions = np.arange(0.0, 0.600, 0.001)

for angle in angles:
    # Initialize total field (no background contribution)
    total_field = np.zeros_like(positions)

    # Full directory path
    dir_path = os.path.join(base_dir, f"r_component, angle_{angle}")
    files = sorted(os.listdir(dir_path))  # Get all files in the directory

    total_field = np.zeros_like(positions)  # Initialize field

    for file in files:
        file_path = os.path.join(dir_path, file)  # Full file path
        try:
            field_data = np.loadtxt(file_path)
            if len(field_data) != len(positions):
                raise ValueError(f"Data length mismatch in {file_path}.")
            total_field += field_data  # Adjust if prefactors are needed
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Plot the total magnetic field for each angle
    plt.figure(figsize=(8, 5))
    plt.plot(positions, total_field, linestyle='-', label=f'Angle {angle}°')
    plt.xlabel("Radial Position (m)")
    plt.ylabel("B-Field Strength (T)")
    plt.title(f"Magnetic Field vs. Radial Position (Angle {angle}°)")
    plt.legend()
    plt.grid()
    plt.show()
