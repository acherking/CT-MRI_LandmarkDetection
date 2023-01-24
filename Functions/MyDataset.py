from os import listdir
from os.path import isfile, join
import numpy as np
import h5py

from scipy.ndimage import zoom


# Load Mat data for single patient
def load_mat_data(volume_path, pts_path="None", with_res=False):
    load_mat_vol = []
    load_mat_pts = []
    load_vol_res = []
    if pts_path != "None":
        file_volume = h5py.File(volume_path, 'r')
        file_pts = h5py.File(pts_path, 'r')
        load_mat_vol = file_volume.get('rescaled_aug_vol')
        load_mat_pts = file_pts.get('rescaled_aug_pts')
    else:
        file_data = h5py.File(volume_path, 'r')
        # load_mat_vol = file_data.get('augVol')
        # load_mat_pts = file_data.get('augPts')
        load_mat_vol = file_data.get('rescaled_aug_vol')
        load_mat_pts = file_data.get('rescaled_aug_pts')
        # load resolution
        if with_res:
            load_vol_res = file_data.get('pixel_distance')

    volume = np.array(load_mat_vol).T
    pts = np.array(load_mat_pts).reshape(3, 4).T
    vol_res = np.array(load_vol_res).T

    # close file automatically: file_data or file_volume&file_pts

    return volume, pts, vol_res


# Load Mat data from files in directory, X and Y in different dir
def load_mat_data_dirs(x_base_path, y_base_path):
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


# Load Mat data from files in directory, X and Y in the same dir and samefile
def load_mat_data_dir(base_path):
    # e.g. AZ_17017030_AugVol_1.mat
    # x_base_path = "/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_from_matlab/Train/Input/"
    # e.g. AZ_17017030_AugPts_1.mat
    # y_base_path = "/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_from_matlab/Train/Output/"

    files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

    x_dataset = []
    y_dataset = []
    for file in files:
        file_path = join(base_path, file)
        load_mat_vol, load_mat_pts = load_mat_data(file_path)
        x_dataset.append(load_mat_vol)
        y_dataset.append(load_mat_pts)

    return x_dataset, y_dataset


# Generally load target data from a HDF5 binary file such as .mat
def load_data(variable_list, path):
    data_list = []
    data_file = h5py.File(path, 'r')
    for var in variable_list:
        data_list.append(data_file.get(var))

    return data_list


def load_data_dir(base_path):
    variable_list = ["augPts"]
    files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

    name_list = []
    points_set = []
    for file in files:
        name_list.append(file)
        file_path = join(base_path, file)
        [pts] = load_data(variable_list, file_path)
        pts = np.array(pts).reshape(3, 4).T
        points_set.append(pts)

    points_set = np.asarray(points_set)

    return name_list, points_set


def rescale_3d_volume(volume, target_size=(170, 170, 30)):
    zoom_scale = np.divide(target_size, volume.shape)
    print("zoom scale is: ", zoom_scale)

    zoomed_volume = zoom(volume, zoom_scale)

    return zoomed_volume
