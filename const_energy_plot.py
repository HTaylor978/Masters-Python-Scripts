import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.integrate import quad

# Number of particles per seed (change this as needed)
num_particles = 1000

base_dir = "C:/Users/harry/OneDrive/Documents/Masters/Code/Sim data folders/Energy_data_final/"
time_values = [1.000000, 10.000000, 50.000000,
               100.000000, 250.000000, 500.000000, 750.000000, 1000.000000, 1250.000000, 1500.000000, 1750.000000, 2000.000000, 2250.000000, 2500.000000]
seed_values = range(1, 2)
file_template = "{}_cross_at_t_{:.6f}_seed{}_numtraj" + \
    str(num_particles) + ".bin"
energy_types = ["Tperp", "Tz", "U"]
k_B = 1.0


def load_energy_file(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}, skipping.")
        return None
    try:
        data = np.fromfile(file_path, dtype=np.float64)
        if data.shape[0] != num_particles:
            print(
                f"Warning: Unexpected data shape in {file_path}, expected {num_particles} but got {data.shape[0]}, skipping.")
            return None
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}, skipping.")
        return None


def scaled_fit_function(E, A, T):
    return A * (2 / np.sqrt(np.pi)) * (1 / (T ** 1.5)) * np.sqrt(E) * np.exp(-E / T)


# Arrays to collect analysis results
temperatures = []
remaining_counts = []

for time_t in time_values:
    total_energies = []

    for seed in seed_values:
        particle_energies = np.zeros(num_particles)
        valid_data = True

        for energy_type in energy_types:
            file_name = file_template.format(energy_type, time_t, seed)
            file_path = os.path.join(base_dir, file_name)

            energy_data = load_energy_file(file_path)
            if energy_data is None:
                valid_data = False
                break

            particle_energies += energy_data

        if valid_data:
            # Only keep energies between 0 and 12 mK (0.012 K)
            valid_energies = particle_energies[(
                particle_energies >= 0) & (particle_energies <= 0.012)]
            total_energies.extend(valid_energies)

    if total_energies:
        total_energies = np.array(total_energies)
        avg_energy = np.mean(total_energies)
        variance_energy = np.var(total_energies)
        print(
            f"Time {time_t}: Avg Energy = {avg_energy:.6f} K, Variance = {variance_energy:.6f} K^2")

        hist_values, bin_edges = np.histogram(
            total_energies, bins=20, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fit_x_fine = np.linspace(0, max(bin_centers), 1000)

        plt.figure(figsize=(8, 5))
        plt.hist(total_energies, bins=20, density=False,
                 edgecolor='black', alpha=0.7, label="Data")

        try:
            popt, _ = curve_fit(scaled_fit_function, bin_centers, hist_values,
                                p0=[max(hist_values), avg_energy], maxfev=10000)
            fitted_A, fitted_T = popt
            temperatures.append((time_t, fitted_T))
            print(f"Fitted Temperature: {fitted_T*1e3:.2f} mK")
            plt.plot(fit_x_fine, scaled_fit_function(fit_x_fine, *popt),
                     'r-', label=f"Fit: T = {fitted_T*1e3:.2f} mK")
        except RuntimeError:
            print(f"Curve fit failed at t = {time_t}, showing histogram only.")
            temperatures.append((time_t, np.nan))

        plt.xlabel("Total Energy (K)")
        plt.ylabel("Number of Particles")
        plt.title(f"Total Energy Distribution at t = {time_t}")
        plt.legend()
        plt.grid()
        plt.show()

        remaining_counts.append((time_t, len(total_energies)))
    else:
        print(f"Time {time_t}: No valid data available, skipping plot.")
        temperatures.append((time_t, np.nan))
        remaining_counts.append((time_t, 0))


def plot_particles_remaining_vs_time(remaining_counts, N0=num_particles * len(seed_values)):
    times, counts = zip(*remaining_counts)
    times = np.array(times)
    counts = np.array(counts)

    # Error bars assuming binomial or Poisson statistics
    errors = np.sqrt(counts)

    # --- Plot: Particles remaining with error bars ---
    plt.figure(figsize=(8, 5))
    plt.errorbar(times, counts, yerr=errors, fmt='bo',
                 label="Simulation Data", capsize=3)

    # Fit exponential decay
    nonzero_mask = counts > 0
    times_nonzero = times[nonzero_mask]
    counts_nonzero = counts[nonzero_mask]
    log_ratio = np.log(counts_nonzero / N0)

    gamma_slope, _ = np.polyfit(times_nonzero, log_ratio, 1)
    gamma = -gamma_slope
    print(
        f"\nFitted decay rate γ = {gamma:.5f} (1/time unit) with N₀ fixed at {N0}")

    fit_times = np.linspace(min(times), max(times), 500)
    fit_counts = N0 * np.exp(-gamma * fit_times)

    plt.plot(fit_times, fit_counts, 'r--', label=f"Exp Fit (γ = {gamma:.5f})")
    plt.xlabel("Time")
    plt.ylabel("Particles Remaining in Trap")
    plt.title("Particles Remaining vs. Time with Exponential Fit")
    plt.legend()
    plt.grid()
    plt.show()

    # --- Plot: ln(N/N0) with optional errors ---
    log_errors = errors[nonzero_mask] / \
        counts_nonzero  # Δln(x) ≈ Δx / x for x >> 0

    plt.figure(figsize=(8, 5))
    plt.errorbar(times_nonzero, log_ratio, yerr=log_errors,
                 fmt='bo', capsize=3, label="ln(N/N₀)")
    plt.plot(times_nonzero, -gamma * times_nonzero,
             'r--', label=f"Linear Fit (Γ = {gamma:.5f})")
    plt.xlabel("Time")
    plt.ylabel("ln(N/N₀)")
    plt.title("Log Survival Fraction vs. Time")
    plt.legend()
    plt.grid()
    plt.show()


def temperature_ratio_theoretical(eta):
    numerator, _ = quad(lambda x: x**(1.5) * np.exp(-x), 0, eta)
    denominator, _ = quad(lambda x: x**(0.5) * np.exp(-x), 0, eta)
    return (2 / 3) * (numerator / denominator)


def fit_theoretical_temperature_ratio(temperatures, time_cutoff=100.0):
    times, temps = zip(*temperatures)
    times = np.array(times)
    temps = np.array(temps)

    # Filter out NaNs and times before cutoff
    valid_mask = (~np.isnan(temps)) & (times >= time_cutoff)
    times = times[valid_mask]
    temps = temps[valid_mask]

    if len(times) == 0:
        print("No valid temperature data after cutoff.")
        return

    T0 = temps[0]
    temp_ratios = temps / T0

    # Define model for eta(t)
    def eta_model(t, eta0, decay_const):
        return eta0 * np.exp(-decay_const * t)

    def model(t, eta0, decay_const):
        return np.array([temperature_ratio_theoretical(eta_model(ti, eta0, decay_const)) for ti in t])

    try:
        popt, _ = curve_fit(model, times, temp_ratios,
                            p0=[5, 0.001], maxfev=10000)
        eta0_fit, decay_fit = popt
        print(
            f"Fitted η(t): η₀ = {eta0_fit:.2f}, decay constant = {decay_fit:.5f}")
    except RuntimeError:
        print("Curve fitting failed.")
        return

    # Plot
    fit_times = np.linspace(min(times), max(times), 300)
    fit_ratios = model(fit_times, *popt)

    plt.figure(figsize=(8, 5))
    plt.plot(times, temp_ratios, 'o', label='Simulated T/T₀')
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature Ratio (T'/T₀)")
    plt.title(f"Analytical Fit to Temperature Ratio")
    plt.legend()
    plt.grid()
    plt.show()


# Call plotting functions
plot_particles_remaining_vs_time(remaining_counts)
fit_theoretical_temperature_ratio(temperatures)
