# import file containing function definitions for analysis program
# Danielle Hodgkinson 25/04/2019
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # this is used, don't delete it
import struct
import os
import general_constants as const
from scipy.stats import gaussian_kde
from scipy.stats import kde
import matplotlib
import pandas as pd

# latex textwidth in inches
textwidthInch = 7.05826
# font size of normal text in latex
font_size = 10
fig_dpi = 300


def read_csv(directory, col_name):
    data = pd.read_excel(directory)

    df = pd.DataFrame(data, columns=[col_name])

    array = df.to_numpy()

    drop_n = []
    for i in range(len(array)):
        drop_n.append(array[i][0])

    return drop_n


def get_AE_experimental_data():
    data_directory = "C:\\Users\\dhodgkin\\Dropbox\\Postgraduate\\Simulation and Software\\Adiabatic Expansion\\Experimental Notes and Data\\2016_offline_analysis\\"

    S622_timing = []
    S621_timing = []
    S620_timing = []

    df = pd.read_csv(data_directory + "A.csv")
    for row_i in range(len(df)):
        S622_timing.append(df.to_numpy()[row_i][0])

    df = pd.read_csv(data_directory + "B.csv")
    for row_i in range(len(df)):
        S621_timing.append(df.to_numpy()[row_i][0])

    df = pd.read_csv(data_directory + "C.csv")
    for row_i in range(len(df)):
        S620_timing.append(df.to_numpy()[row_i][0])

    return S622_timing, S621_timing, S620_timing

# formats the ticks on an axis for figures


def format_tick_params(ax):
    ax.tick_params(axis='y', direction='in', labelbottom=True, which='both')
    ax.tick_params(axis='x', direction='in', labelbottom=True, which='both')
    ax.tick_params(axis='y', direction='in',
                   labelbottom=True, which='major', length=5)
    ax.tick_params(axis='x', direction='in',
                   labelbottom=True, which='major', length=5)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')


def calc_n_bins(bin_width, data):
    return int((max(data) - min(data))/bin_width)


def build_fig(fig_xscale, fig_yscale):  # y and x scale are in units of the thesis textwidth
    fig, ax = plt.subplots(figsize=(fig_xscale*textwidthInch,
                           fig_yscale*textwidthInch), dpi=fig_dpi, tight_layout=True)
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.rcParams.update({'font.size': font_size})
    plt.minorticks_on()
    format_tick_params(ax)
    plt.xticks(fontsize=font_size - 1)
    plt.yticks(fontsize=font_size - 1)
    return fig, ax

# function to draw a 3D trap


def draw_3D_trap(fig, position):
    ax = fig.add_subplot(position, projection='3d')
    x = np.linspace(-const.big_trap_radius, const.big_trap_radius, 100)
    z = np.linspace(const.small_right_end_left, const.small_left_end_right, 5)

    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt((const.big_trap_radius**2-Xc**2))

    x_small = np.linspace(-const.small_trap_radius,
                          const.small_trap_radius, 100)
    z_small = np.linspace(const.small_left_end_left,
                          const.small_right_end_left, 5)
    Xc_small, Zc_small = np.meshgrid(x_small, z_small)
    Yc_small = np.sqrt((const.small_trap_radius**2-Xc_small**2))

    z_small_r = np.linspace(const.small_left_end_right,
                            const.small_right_end_right, 5)
    Xc_small_r, Zc_small_r = np.meshgrid(x_small, z_small_r)
    Yc_small_r = np.sqrt((const.small_trap_radius**2-Xc_small_r**2))

    # Draw parameters
    rstride = 2
    cstride = 1
    rstride_small = 2
    cstride_small = 1

    ax.plot_surface(Zc, Yc, Xc, alpha=0.4, rstride=rstride, cstride=cstride)
    ax.plot_surface(Zc, -Yc, Xc, alpha=0.4, rstride=rstride, cstride=cstride)
    ax.plot_surface(Zc_small, Yc_small, Xc_small, alpha=0.4,
                    rstride=rstride_small, cstride=cstride_small)
    ax.plot_surface(Zc_small, -Yc_small, Xc_small, alpha=0.4,
                    rstride=rstride_small, cstride=cstride_small)
    ax.plot_surface(Zc_small_r, Yc_small_r, Xc_small_r, alpha=0.4,
                    rstride=rstride_small, cstride=cstride_small)
    ax.plot_surface(Zc_small_r, -Yc_small_r, Xc_small_r, alpha=0.4,
                    rstride=rstride_small, cstride=cstride_small)
    ax.set_xlabel("Z(m)")
    ax.set_ylabel("Y(m)")
    ax.set_zlabel("X(m)")

    return ax


def read_binary_doubles(folder_name, start_filename, storage_vector, seed, numtraj):

    file_length_in_bytes = os.path.getsize(
        folder_name + "/" + start_filename + "_seed%s_numtraj%s.bin" % (seed, numtraj))
    num_data_points = int(file_length_in_bytes/8)

    with open(folder_name + "/" + start_filename + "_seed%s_numtraj%s.bin" % (seed, numtraj), mode='rb') as file:
        for i in range(num_data_points):
            a = struct.unpack('d', file.read(8))
            storage_vector.append(a[0])

    return


def read_binary_doubles_change_timestep(folder_name, start_filename, storage_vector, seed, numtraj, timestep_string):

    file_length_in_bytes = os.path.getsize(
        folder_name + "/" + start_filename + "_seed%s_numtraj%s_timestep_" % (seed, numtraj) + timestep_string + ".bin")
    num_data_points = int(file_length_in_bytes/8)

    with open(folder_name + "/" + start_filename + "_seed%s_numtraj%s_timestep_" % (seed, numtraj) + timestep_string + ".bin", mode='rb') as file:
        for i in range(num_data_points):
            a = struct.unpack('d', file.read(8))
            storage_vector.append(a[0])

    return


def read_binary_doubles_and_multiply(folder_name, start_filename, storage_vector, seed, numtraj, multiplicative_factor):
    #print(folder_name + "/" + start_filename + "_seed%s_numtraj%s.bin" %(seed, numtraj))
    file_length_in_bytes = os.path.getsize(
        folder_name + start_filename + "_seed%s_numtraj%s.bin" % (seed, numtraj))
    num_data_points = int(file_length_in_bytes/8)

    with open(folder_name + "/" + start_filename + "_seed%s_numtraj%s.bin" % (seed, numtraj), mode='rb') as file:
        for i in range(num_data_points):
            a = struct.unpack('d', file.read(8))
            storage_vector.append(a[0]*multiplicative_factor)
    file.close()
    return


def read_binary_ints(folder_name, start_filename, storage_vector, seed, numtraj):

    file_length_in_bytes = os.path.getsize(
        folder_name + start_filename + "_seed%s_numtraj%s.bin" % (seed, numtraj))
    num_data_points = int(file_length_in_bytes/4)

    with open(folder_name + "/" + start_filename + "_seed%s_numtraj%s.bin" % (seed, numtraj), mode='rb') as file:
        for i in range(num_data_points):
            a = struct.unpack('i', file.read(4))
            storage_vector.append(a[0])

    return


def read_binary_ints_change_timestep(folder_name, start_filename, storage_vector, seed, numtraj, timestep_string):
    file_length_in_bytes = os.path.getsize(
        folder_name + "/" + start_filename + "_seed%s_numtraj%s_timestep_" % (seed, numtraj) + timestep_string + ".bin")
    num_data_points = int(file_length_in_bytes/4)

    with open(folder_name + "/" + start_filename + "_seed%s_numtraj%s_timestep_" % (seed, numtraj) + timestep_string + ".bin", mode='rb') as file:
        for i in range(num_data_points):
            a = struct.unpack('i', file.read(4))
            storage_vector.append(a[0])

    return


def read_binary_bools(folder_name, start_filename, storage_vector, seed, numtraj):

    file_length_in_bytes = os.path.getsize(
        folder_name + "/" + start_filename + "_seed%s_numtraj%s.bin" % (seed, numtraj))
    num_data_points = int(file_length_in_bytes/2)

    with open(folder_name + "/" + start_filename + "_seed%s_numtraj%s.bin" % (seed, numtraj), mode='rb') as file:
        for i in range(num_data_points):
            a = struct.unpack('i', file.read(2))
            storage_vector.append(a[0])

    return


def read_bin_doubles_no_file_extension(folder_name, filename, storage_vector):

    file_length_in_bytes = os.path.getsize(folder_name + "/" + filename)
    num_data_points = int(file_length_in_bytes/8)

    with open(folder_name + "/" + filename, mode='rb') as file:
        for i in range(num_data_points):
            a = struct.unpack('d', file.read(8))
            storage_vector.append(a[0])

    return


def read_bin_ints_no_file_extension(folder_name, filename, storage_vector):

    file_length_in_bytes = os.path.getsize(folder_name + "/" + filename)
    num_data_points = int(file_length_in_bytes/4)

    with open(folder_name + "/" + filename, mode='rb') as file:
        for i in range(num_data_points):
            a = struct.unpack('i', file.read(4))
            storage_vector.append(a[0])

    return


def density_plot(x, y, nbins):
    fig, ax = plt.subplots()
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[min(x):max(x):nbins*1j, min(y):max(y):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    cax = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.inferno)

    return fig, ax


def contour_plot(x, y, nbins):
    fig, ax = plt.subplots()
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[min(x):max(x):nbins*1j, min(y):max(y):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    cax = ax.contour(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.inferno)
    return fig, ax


def contour_plot_f(x, y, nbins):
    fig, ax = plt.subplots()
    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[min(x):max(x):nbins*1j, min(y):max(y):nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    cax = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.inferno)
    return fig, ax


def flat_2D_hist(x, y, xmin, xmax, ymin, ymax, x_bins, y_bins, enable_vmin, vmin, vmax):
    xedges = []
    yedges = []
    for i in range(x_bins):
        xedges.append(i*(xmax - xmin)/x_bins)
    for j in range(y_bins):
        yedges.append(j*(ymax - ymin)/y_bins)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if(enable_vmin):
        pos = ax.imshow(H, interpolation='none', origin='low', extent=[
                        xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="Blues", vmin=vmin, vmax=vmax)
    else:
        pos = ax.imshow(H, interpolation='none', origin='low', extent=[
                        xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="Blues")

    cbar = fig.colorbar(pos, ax=ax, extend='both')

    return fig, ax
