import numpy as np
import matplotlib.pyplot as plt
import os

# Define the base directory where the field data files are stored
base_dir = "C:/Users/harry/OneDrive/Documents/Masters/Code/Sim data folders/B_fields/z_component/"

# List of field file names
field_files = ["MIRA_1A.txt", "MIRB_1A.txt", "MIRC_1A.txt",
               "MIRD_1A.txt", "MIRE_1A.txt", "SOLA_1A.txt", "SOLB_1A.txt"]

# Prefactors for each field file (adjust these as needed)
prefactors = {
    "MIRA_1A.txt": 490.2/2959.36,
    "MIRB_1A.txt": -59.09/1479.68,
    "MIRC_1A.txt": -1.346/92.48,
    "MIRD_1A.txt": -59.13/1479.68,
    "MIRE_1A.txt": 482.1/2959.36,
    "SOLA_1A.txt": 250.06/1479.68,
    "SOLB_1A.txt": 249.98/1479.68
}

# Generate position values from -0.6 to 0.6 with 0.001 increments
positions = np.arange(-0.6, 0.600, 0.001)

# Initialize total field with the constant background field of 1T
total_field = np.ones_like(positions)

# Load and sum the fields from each file
for field_file in field_files:
    file_path = os.path.join(base_dir, field_file)
    try:
        field_data = np.loadtxt(file_path)
        if len(field_data) != len(positions):
            raise ValueError(f"Data length mismatch in {field_file}.")
        total_field += prefactors[field_file] * field_data
    except Exception as e:
        print(f"Error loading {field_file}: {e}")

# Plot the total magnetic field
plt.figure(figsize=(8, 5))
plt.plot(positions, total_field, linestyle='-', label='Total B-Field')
plt.xlabel("Position (m)")
plt.ylabel("B-Field Strength (T)")
plt.title("Total Magnetic Field vs. Position")
plt.legend()
plt.grid()
plt.show()


def find_potential_depth(positions, total_field):
    """Finds the depth of the potential well in the U-shaped magnetic field."""
    min_field = np.min(total_field)  # Minimum field value (bottom of U)
    max_field = np.max(total_field)  # Maximum field value (top of U)

    depth = max_field - min_field  # Difference between top and bottom

    # Convert to mK
    mu_B = 9.274e-24  # Bohr magneton in J/T
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    delta_T = (mu_B * depth) / k_B  # Temperature in Kelvin
    delta_T_mK = delta_T * 1e3  # Convert to milliKelvin
    return delta_T_mK


# Compute the depth of the potential well
depth_mK = find_potential_depth(positions, total_field)
print(f"Potential well depth: {depth_mK:.3f} mK")
