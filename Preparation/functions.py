import numpy

from matplotlib import pyplot
from matplotlib.patches import Circle

from os import listdir
from os.path import isfile, join
import numpy as np
import h5py

from scipy.ndimage import zoom


# print the Scans with pts: LLSCC ant, LLSCC post, RLSCC ant, RLSCC post
def show_pts(volume, pts, pixel_space):
    y_row = numpy.arange(0.0, volume.shape[0] * pixel_space[0], pixel_space[0])
    x_column = numpy.arange(0.0, volume.shape[1] * pixel_space[1], pixel_space[1])
    z_slice = numpy.arange(0.0, volume.shape[2] * pixel_space[2], pixel_space[2])

    landmark_radius = volume.shape[1] * pixel_space[1] * 0.016

    fig, axs = pyplot.subplots(2, 2, sharex=True)
    fig.set_dpi(300)

    axs[0][0].set_title("LLSCC ant")
    axs[0][0].set_aspect('equal', 'datalim')
    axs[0][0].pcolormesh(x_column[:], y_row[:], volume[:, :, int(pts[0, 2])], cmap=pyplot.gray())
    llscc_ant = Circle((pts[0, 0] * pixel_space[1], pts[0, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                       edgecolor='r', lw=1)
    axs[0][0].add_patch(llscc_ant)
    axs[0][0].invert_yaxis()

    axs[0][1].set_title("LLSCC post")
    axs[0][1].set_aspect('equal', 'datalim')
    axs[0][1].pcolormesh(x_column[:], y_row[:], volume[:, :, int(pts[1, 2])], cmap=pyplot.gray())
    llscc_post = Circle((pts[1, 0] * pixel_space[1], pts[1, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                        edgecolor='r', lw=1)
    axs[0][1].add_patch(llscc_post)
    axs[0][1].invert_yaxis()

    axs[1][0].set_title("RLSCC ant")
    axs[1][0].set_aspect('equal', 'datalim')
    axs[1][0].pcolormesh(x_column[:], y_row[:], volume[:, :, int(pts[2, 2])], cmap=pyplot.gray())
    rlscc_ant = Circle((pts[2, 0] * pixel_space[1], pts[2, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                       edgecolor='r', lw=1)
    axs[1][0].add_patch(rlscc_ant)
    axs[1][0].invert_yaxis()

    axs[1][1].set_title("RLSCC post")
    axs[1][1].set_aspect('equal', 'datalim')
    axs[1][1].pcolormesh(x_column[:], y_row[:], volume[:, :, int(pts[3, 2])], cmap=pyplot.gray())
    rlscc_post = Circle((pts[3, 0] * pixel_space[1], pts[3, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                        edgecolor='r', lw=1)
    axs[1][1].add_patch(rlscc_post)
    axs[1][1].invert_yaxis()

    pyplot.show()


# Load Mat data for single patient
def load_mat_data(volume_path, pts_path="None"):
    load_mat_vol = []
    load_mat_pts = []
    if pts_path != "None":
        file_volume = h5py.File(volume_path, 'r')
        file_pts = h5py.File(pts_path, 'r')
        load_mat_vol = file_volume.get('rescaled_aug_vol')
        load_mat_pts = file_pts.get('rescaled_aug_pts')
    else:
        file_data = h5py.File(volume_path, 'r')
        load_mat_vol = file_data.get('augVol')
        load_mat_pts = file_data.get('augPts')

    volume = np.array(load_mat_vol).T
    pts = np.array(load_mat_pts).reshape(3, 4).T

    # close file automatically: file_data or file_volume&file_pts

    return volume, pts


# Load Mat data from files in directory
def load_mat_data_dir(x_base_path, y_base_path):
    # e.g. AZ_17017030_AugVol_1.mat
    # x_base_path = "/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_from_matlab/Train/Input/"
    # e.g. AZ_17017030_AugPts_1.mat
    # y_base_path = "/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_from_matlab/Train/Output/"

    x_files = [f for f in listdir(x_base_path) if isfile(join(x_base_path, f))]

    x_dataset = []
    y_dataset = []
    for x_file in x_files:
        x_file_path = join(x_base_path, x_file)
        y_file_path = join(y_base_path, x_file.replace("AugVol", "AugPts"))
        load_mat_vol, load_mat_pts = load_mat_data(x_file_path, y_file_path)
        x_dataset.append(load_mat_vol)
        y_dataset.append(load_mat_pts)

    return x_dataset, y_dataset


def rescale_3d_volume(volume, target_size=(170, 170, 30)):
    zoom_scale = np.divide(target_size, volume.shape)
    print("zoom scale is: ", zoom_scale)

    zoomed_volume = zoom(volume, zoom_scale)

    return zoomed_volume
