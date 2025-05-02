import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import argrelextrema

# Define the base directory
base_dir = "C:/Users/harry/OneDrive/Documents/Masters/Code/Sim data folders/B_fields_3D/"

# List of field file names
field_files = ["MIRA.txt", "MIRB.txt", "MIRC.txt", "MIRD.txt",
               "MIRE.txt", "OCT.txt", "SOLA.txt", "SOLB.txt", "BG.txt"]

# Prefactors for scaling
A = 1479.68 / 32
B = A / 2
C = A / 32
D = A / 7

prefactors = {
    "MIRA.txt": 490.2 / A,
    "MIRB.txt": -59.09 / B,
    "MIRC.txt": -1.346 / C,
    "MIRD.txt": -59.13 / B,
    "MIRE.txt": 482.1 / A,
    "OCT.txt": -900.0 / D,
    "SOLA.txt": 0,
    "SOLB.txt": 0,
    "BG.txt": 1.0
}

print(prefactors)

# Dictionary to store (r, theta, z) -> (B_r, B_theta, B_z)
B_field_dict = {}

# Process each file
for field_file in field_files:
    file_path = os.path.join(base_dir, field_file)
    try:
        data = np.loadtxt(file_path, delimiter=",")
        r_pos, theta_pos, z_pos = data[:, 0], data[:, 1] % (
            2 * np.pi), data[:, 2]
        B_r, B_theta, B_z = data[:, 3], data[:, 4], data[:, 5]
        factor = prefactors.get(field_file, 1.0)
        B_r *= factor
        B_theta *= factor
        B_z *= factor

        for i in range(len(r_pos)):
            key = (r_pos[i], theta_pos[i], z_pos[i])
            if key not in B_field_dict:
                B_field_dict[key] = np.array([0.0, 0.0, 0.0])
            B_field_dict[key] += np.array([B_r[i], B_theta[i], B_z[i]])
    except Exception as e:
        print(f"Error loading {field_file}: {e}")

# Extract sorted unique values
r_vals = sorted(set(k[0] for k in B_field_dict.keys()))
theta_vals = sorted(set(k[1] for k in B_field_dict.keys()))
z_vals = sorted(set(k[2] for k in B_field_dict.keys()))

# Create 3D arrays
B_r_values = np.zeros((len(r_vals), len(theta_vals), len(z_vals)))
B_theta_values = np.zeros((len(r_vals), len(theta_vals), len(z_vals)))
B_z_values = np.zeros((len(r_vals), len(theta_vals), len(z_vals)))

for (r, theta, z), B_vec in B_field_dict.items():
    i, j, k = r_vals.index(r), theta_vals.index(theta), z_vals.index(z)
    B_r_values[i, j, k] = B_vec[0]
    B_theta_values[i, j, k] = B_vec[1]
    B_z_values[i, j, k] = B_vec[2]

B_magnitude = np.sqrt(B_r_values**2 + B_theta_values**2 + B_z_values**2)


def get_B_field(r, theta, z):
    theta = theta % (2 * np.pi)
    if (r, theta, z) in B_field_dict:
        B_vec = B_field_dict[(r, theta, z)]
        return B_vec[0], B_vec[1], B_vec[2], np.linalg.norm(B_vec)

    if r < min(r_vals) or r > max(r_vals) or z < min(z_vals) or z > max(z_vals):
        print(f"Warning: r={r}, z={z} out of bounds.")
        return None

    B_r_interp = RegularGridInterpolator(
        (r_vals, theta_vals, z_vals), B_r_values, bounds_error=False, fill_value=None)
    B_theta_interp = RegularGridInterpolator(
        (r_vals, theta_vals, z_vals), B_theta_values, bounds_error=False, fill_value=None)
    B_z_interp = RegularGridInterpolator(
        (r_vals, theta_vals, z_vals), B_z_values, bounds_error=False, fill_value=None)

    B_r_interp_val = B_r_interp((r, theta, z))
    B_theta_interp_val = B_theta_interp((r, theta, z))
    B_z_interp_val = B_z_interp((r, theta, z))

    if B_r_interp_val is None or B_theta_interp_val is None or B_z_interp_val is None:
        print(f"Interpolation failed for (r={r}, θ={theta}, z={z})")
        return None

    return B_r_interp_val, B_theta_interp_val, B_z_interp_val, np.sqrt(B_r_interp_val**2 + B_theta_interp_val**2 + B_z_interp_val**2)


def find_minimum_escape_energy_slice(B_mag_slice, r_vals, z_vals):
    mu_B = 9.274e-24  # Bohr magneton in J/T
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    r_escape_barriers = []
    z_escape_barriers = []

    # For each fixed z (vary r)
    for z_idx in range(len(z_vals)):
        B_along_r = B_mag_slice[:, z_idx]
        if len(B_along_r) > 2:
            local_maxima_r = argrelextrema(B_along_r, np.greater)[0]
            if len(local_maxima_r) > 0:
                r_escape_barriers.append(np.max(B_along_r[local_maxima_r]))

    # For each fixed r (vary z)
    for r_idx in range(len(r_vals)):
        B_along_z = B_mag_slice[r_idx, :]
        if len(B_along_z) > 2:
            local_maxima_z = argrelextrema(B_along_z, np.greater)[0]
            if len(local_maxima_z) > 0:
                z_escape_barriers.append(np.max(B_along_z[local_maxima_z]))

    # Fallback in case no local maxima are found
    if not r_escape_barriers:
        r_escape_barriers = [np.max(B_mag_slice)]
    if not z_escape_barriers:
        z_escape_barriers = [np.max(B_mag_slice)]

    # Subtract the 1 T background before converting to energy
    min_escape_B_r = min(r_escape_barriers) - 1.0
    min_escape_B_z = min(z_escape_barriers) - 1.0

    # Avoid negative or zero field strength due to subtraction
    min_escape_B_r = max(min_escape_B_r, 0)
    min_escape_B_z = max(min_escape_B_z, 0)

    delta_T_r_mK = (mu_B * min_escape_B_r) / k_B * 1e3
    delta_T_z_mK = (mu_B * min_escape_B_z) / k_B * 1e3

    return delta_T_r_mK, delta_T_z_mK


def plot_B_field(R, Z, B, title, cmap="viridis"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(R, Z, B, cmap=cmap, edgecolor='k', alpha=0.8)
    ax.set_xlabel("r position (m)")
    ax.set_ylabel("z position (m)")
    ax.set_zlabel("B-Field Strength (T)")
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()


# ==== Plot for each theta ====
R, Z = np.meshgrid(r_vals, z_vals, indexing="ij")

num_theta = len(theta_vals)

for j, theta in enumerate(theta_vals):
    B_slice = B_magnitude[:, j, :]  # shape (r, z)
    delta_T_r_mK, delta_T_z_mK = find_minimum_escape_energy_slice(
        B_slice, r_vals, z_vals)

    print(f"\n--- θ = 2π * {j}/{num_theta} ---")
    print(f"Min escape energy (r-dir): {delta_T_r_mK:.3f} mK")
    print(f"Min escape energy (z-dir): {delta_T_z_mK:.3f} mK")

    title_str = f"|B| vs r and z at θ = 2π × {j}/{num_theta}"
    plot_B_field(R, Z, B_slice, title_str)
