from os import listdir
from os.path import isfile, join
import numpy as np
import h5py

from scipy.ndimage import zoom

# This is the patients namelist mate
patient_names = ['AH', 'AZ', 'DE', 'DM', 'DM2', 'DGL', 'FA', 'GE', 'GM', 'GP', 'HB', 'HH',
                 'JH', 'JM', 'LG', 'LP', 'MJ', 'NV', 'PH', 'SM']

# static K folds
k_folds_5 = [['JM', 'DM2', 'LG', 'HB'],
             ['GP', 'PH', 'AZ', 'HH'],
             ['SM', 'DM', 'AH', 'DGL'],
             ['DE', 'FA', 'GE', 'JH'],
             ['GM', 'LP', 'NV', 'MJ']]

k_folds_10 = [['LP', 'JM'],
              ['AH', 'DM'],
              ['LG', 'MJ'],
              ['NV', 'JH'],
              ['SM', 'FA'],
              ['HB', 'GM'],
              ['AZ', 'DGL'],
              ['GP', 'GE'],
              ['DE', 'PH'],
              ['HH', 'DM2']]

k_folds_20 = np.reshape(patient_names, (20, 1))


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


def get_data_splits(pat_splits, split=False):

    if split:
        idx_splits = [[list(range(i * 100, i * 100 + 100)) for i in j] for j in pat_splits]
        for i in range(0, 3):
            idx_splits[i] = [num for sublist in idx_splits[i] for num in sublist]
            idx_splits[i] = np.asarray(idx_splits[i])
    else:
        idx_splits = [[list(range(i * 50, i * 50 + 50)) for i in j] for j in pat_splits]
        for i in range(0, 3):
            idx_splits[i] = [num for sublist in idx_splits[i] for num in sublist]
            idx_splits[i] = np.asarray(idx_splits[i])

    return idx_splits


def get_pat_names():
    return patient_names


def get_idx_from_pat(pat_name, aug_id, split=False):
    pat_idx = patient_names.index(pat_name)
    if not split:
        idx = pat_idx * 50 + (aug_id - 1)
    else:
        idx = pat_idx * 100 + (aug_id - 1) * 2

    return idx


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
        pt_aug_id = np.ceil(idx % 100 / 2).astype(int)

    pt_name = patient_names[test_pat_idx[pt_idx]]

    return pt_name, pt_aug_id


def get_k_folds(k):
    if k == 5:
        return k_folds_5
    elif k == 10:
        return k_folds_10
    elif k == 20:
        return k_folds_20
    else:
        print("Error k: ", k)
        return []


def get_k_folds_data_splits(k):
    k_folds = get_k_folds(k)
    print("K folds: ", k_folds)

    # [training_dataset_id, val_dataset_id, val_dataset_id] --- just for convenient
    k_folds_idx_splits = []
    for k_i in range(k):
        test_pats = k_folds[k_i]
        test_pats_id = [patient_names.index(name) for name in test_pats]

        training_pats = k_folds[:k_i] + k_folds[k_i+1:]
        training_pats = [j for i in training_pats for j in i]  # combine sub-lists
        training_pats_id = [patient_names.index(name) for name in training_pats]

        # add val dataset
        val_pats_id = training_pats_id[0:2]
        training_pats_id = training_pats_id[2:]

        pat_splits = [training_pats_id, val_pats_id, test_pats_id]
        k_folds_idx_splits.append(get_data_splits(pat_splits, split=True))

    return k_folds_idx_splits
