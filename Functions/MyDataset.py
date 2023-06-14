from os import listdir
from os.path import isfile, join
import numpy as np
import h5py

from scipy.ndimage import zoom

# This is the patients namelist mate
patient_names = ['AH', 'AZ', 'DE', 'DM', 'DM2', 'DGL', 'FA', 'GE', 'GM', 'GP', 'HB', 'HH',
                 'JH', 'JM', 'LG', 'LP', 'MJ', 'NV', 'PH', 'SM']


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
        load_mat_vol = file_data.get('rescaled_aug_vol')
        load_mat_pts = file_data.get('rescaled_aug_pts')
        if load_mat_vol is None:
            # Because of some Historical reasons
            load_mat_vol = file_data.get('augVol')
            load_mat_pts = file_data.get('augPts')
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


def get_pat_splits(static=False):
    pat_splits = [np.asarray([2, 4, 18, 17, 12, 10, 6, 0, 11, 16, 9, 14, 5, 19]),
                  np.asarray([3, 13]), np.asarray([8, 7, 1, 15])]

    if not static:
        pat_idx = np.arange(0, 20)
        np.random.shuffle(pat_idx)
        pat_splits = np.split(pat_idx, [int(.7 * len(pat_idx)), int(.8 * len(pat_idx))])

    return pat_splits


def get_pat_names():
    return patient_names


def get_pat_from_idx(idx, split=False):
    if not split:
        if idx >= 1000:
            print("get_pat_from_test_idx: error idx!", idx)
            return
        pt_idx = np.floor(idx / 50).astype(int)
        pt_aug_id = idx % 50 + 1
    else:
        if idx >= 2000:
            print("get_pat_from_test_idx: error idx!", idx)
            return
        pt_idx = np.floor(idx / 100).astype(int)
        pt_aug_id = np.floor(idx % 100 / 2).astype(int) + 1

    return patient_names[pt_idx], pt_aug_id


def get_pat_from_test_idx(idx, split=False):
    test_pat_idx = get_pat_splits(static=True)[2]  # 0: train, 1: val, 2: test

    if not split:
        if idx >= 200:
            print("get_pat_from_test_idx: error idx!", idx)
            return
        pt_idx = np.floor(idx / 50).astype(int)
        pt_aug_id = idx % 50 + 1
    else:
        if idx >= 400:
            print("get_pat_from_test_idx: error idx!", idx)
            return
        pt_idx = np.floor(idx / 100).astype(int)
        pt_aug_id = np.floor(idx % 100 / 2).astype(int) + 1

    pt_name = patient_names[test_pat_idx[pt_idx]]

    return pt_name, pt_aug_id


# e.g. map the landmarks points from current shape (like the rescaled 176*176*48)
# to the target shape (like before rescaled)
def map_points(points, current_volume_shape, target_volume_shape):
    # points: (num_points * 3)
    # current_volume_shape: e.g. 176*176*48
    # target_volume_shape: e.g. 1200*1100*365
    mapped_points = []

    return mapped_points
