import math
import os
import time
from sys import exit

import numpy as np
import tensorflow as tf
from scipy import ndimage

from common import MyCrop
from common import MyDataset


# prepare the recording directory
def get_record_dir(args_dict, get_dir=False):
    save_base_dir = args_dict.get("save_base_dir")
    dataset_tag = args_dict.get("dataset_tag")
    model_name = args_dict.get("model_name")
    # y_tag: "one_landmark_[1/2]", "two_landmarks", "mean_two_landmarks"
    y_tag = args_dict.get("y_tag")
    data_size = args_dict.get("input_shape")
    cut_layers = args_dict["cut_layers"]
    data_size = (data_size[0]-cut_layers[0][0] - cut_layers[0][1],
                 data_size[1]-cut_layers[1][0] - cut_layers[1][1],
                 data_size[2]-cut_layers[2][0] - cut_layers[2][1])
    data_size_str = f"{data_size[0]}x{data_size[1]}x{data_size[2]}"
    dataset_label_1 = args_dict.get("dataset_label_1")
    label_1 = args_dict.get("model_label_1")
    label_2 = args_dict.get("model_label_2")

    save_dir = f"{save_base_dir}/{dataset_tag}/{data_size_str}/{dataset_label_1}/{y_tag}/{model_name}"
    if len(label_1) > 0:
        save_dir = f"{save_dir}/{label_1}"
    if len(label_2) > 0:
        save_dir = f"{save_dir}/{label_2}"

    time_tag = time.strftime("%d%b%Y-%H:%M:%S")
    if args_dict.get("data_split_tag") == "cross_val":
        k_str = str(args_dict["k_cross_num"])
        k_idx_str = str(args_dict["k_cross_idx"])
        k_fold_tag = f"{k_str}fold{k_idx_str}"
        save_dir = f"{save_dir}/{time_tag}-{k_fold_tag}"
    else:
        train_id_str = str(args_dict.get("train_id", -1))
        train_id_tag = f"trainID-{train_id_str}"
        save_dir = f"{save_dir}/{time_tag}-{train_id_tag}"

    if not get_dir:
        # create the dir if not exist
        if os.path.exists(save_dir):
            print("Save model to: ", save_dir)
        else:
            os.makedirs(save_dir)
            print("Create dir and save model in it: ", save_dir)

    return save_dir


# prepare the train_dataset, val_dataset and test_dataset for all the Training
def load_dataset_manager(args_dict):
    ## load dataset
    dataset_tag = args_dict.get("dataset_tag", "divided")
    input_shape = args_dict.get("input_shape", (176, 88, 48))
    input_shape_str = f"{str(input_shape[0])}x{str(input_shape[1])}x{str(input_shape[2])}"
    base_dir = args_dict.get("base_dir", "/data/gpfs/projects/punim1836/Data")
    label_1 = args_dict.get("dataset_label_1")

    dataset_dir = f"{base_dir}/{dataset_tag}/{input_shape_str}/{label_1}/"
    print("Read dataset from: ", dataset_dir)

    x_dataset_path = dataset_dir + f"{dataset_tag}_volumes_" + input_shape_str + ".npy"
    y_dataset_path = dataset_dir + f"{dataset_tag}_points_" + input_shape_str + ".npy"
    res_dataset_path = dataset_dir + f"{dataset_tag}_res_" + input_shape_str + ".npy"

    x_dataset = np.load(x_dataset_path)
    y_dataset = np.load(y_dataset_path)
    instances_num = x_dataset.shape[0]
    res_dataset = np.repeat(np.load(res_dataset_path), 2, axis=1).reshape(instances_num, 1, 3)

    if dataset_tag == "divided":
        print("No more cook for divided dataset.")
    elif dataset_tag == "cropped":
        cut_layers = np.asarray(args_dict.get("cut_layers"))
        if not np.all(cut_layers == 0):
            x_dataset, y_dataset = \
                MyCrop.crop_outside_layers_no_length(x_dataset, y_dataset, cut_layers, keep_blank=False)
        else:
            print("No more cook for cropped dataset.")
    else:
        print("Unknown dataset tag: ", dataset_tag)
        exit(0)

    model_output_num = args_dict.get("model_output_num")
    if model_output_num == 1:
        model_y_tag = args_dict.get("y_tag")
        if model_y_tag == "one_landmark_1":
            y_dataset = np.asarray(y_dataset)[:, 0, :].reshape((instances_num, 1, 3))
        elif model_y_tag == "one_landmark_2":
            y_dataset = np.asarray(y_dataset)[:, 1, :].reshape((instances_num, 1, 3))
        elif model_y_tag == "mean_two_landmarks":
            y_dataset = np.mean(y_dataset, axis=1).reshape((instances_num, 1, 3))
        else:
            print("Error Y tag: ", model_y_tag)

    model_tag = args_dict.get("model_name").split('_')[-1]
    # adjust Y for dsnt you know, if the model is dsnt haha
    if model_tag == "dsnt":
        (row_size, column_size, slice_size) = (x_dataset.shape[1], x_dataset.shape[2], x_dataset.shape[3])
        y_dataset = (2 * y_dataset - [column_size + 1, row_size + 1, slice_size + 1]) / [column_size, row_size,
                                                                                         slice_size]
        res_dataset = (res_dataset / [2 / column_size, 2 / row_size, 2 / slice_size])

    y_dataset = y_dataset.astype(np.float32)
    res_dataset = res_dataset.astype(np.float32)

    ## get dataset splits
    data_split_tag = args_dict.get("data_split_tag")
    data_split_static = args_dict.get("data_split_static")
    if data_split_tag == "general":
        if data_split_static: print("Using static dataset split: Train, Val, Test")
        data_splits = MyDataset.get_data_splits(MyDataset.get_pat_splits(static=data_split_static), split=True)
    elif data_split_tag == "cross_val":
        k_pat_splits = MyDataset.get_k_folds_pat_splits(args_dict["k_cross_num"])
        print("K folds patients split: ", k_pat_splits[args_dict["k_cross_idx"]])
        data_splits = MyDataset.get_data_splits(k_pat_splits[args_dict["k_cross_idx"]], split=True)
    else:
        print("Unknown data_split_tag: ", data_split_tag)
        exit(0)

    (train_idx, val_idx, test_idx) = (data_splits[0], data_splits[1], data_splits[2])
    train_dataset = (x_dataset[train_idx], y_dataset[train_idx], res_dataset[train_idx])
    val_dataset = (x_dataset[val_idx], y_dataset[val_idx], res_dataset[val_idx])
    test_dataset = (x_dataset[test_idx], y_dataset[test_idx], res_dataset[test_idx])

    return train_dataset, val_dataset, test_dataset


# load dataset from the combination data files: X and Y
# idx_splits: [[train_idx], [val_idx], [test_idx]], idx from 0 to 1999
# crop_layers: ndarray shape(3*2), [[row_ascending, row_descending], [column_a, column_d], [slice_a, slice_d]]
def load_dataset_crop(x_path, y_path, length_path, idx_splits, crop_layers):
    x_dataset = np.load(x_path)
    y_dataset = np.load(y_path).astype('float32')
    length_dataset = np.load(length_path).astype('float32')

    row_num = x_dataset.shape[1]
    column_num = x_dataset.shape[2]
    slice_num = x_dataset.shape[3]

    if not np.all(crop_layers == 0):
        x_dataset = x_dataset[:,
                    crop_layers[0][0]:(row_num - crop_layers[0][1]),
                    crop_layers[1][0]:(column_num - crop_layers[1][1]),
                    crop_layers[2][0]:(slice_num - crop_layers[2][1]), :]
        y_dataset = y_dataset - [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        y_dataset = y_dataset.astype('float32')
        # left ear
        length_dataset[range(0, 2000, 2)] = \
            length_dataset[range(0, 2000, 2)] + [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        # right ear, because of the flip
        length_dataset[range(1, 2000, 2)] = \
            length_dataset[range(1, 2000, 2)] + [crop_layers[1, 1], crop_layers[0, 0], crop_layers[2, 0]]
        length_dataset = length_dataset.astype('float32')

    train_idx = idx_splits[0]
    x_train = x_dataset[train_idx]
    y_train = y_dataset[train_idx]
    length_train = length_dataset[train_idx]

    val_idx = idx_splits[1]
    x_val = x_dataset[val_idx]
    y_val = y_dataset[val_idx]
    length_val = length_dataset[val_idx]

    test_idx = idx_splits[2]
    x_test = x_dataset[test_idx]
    y_test = y_dataset[test_idx]
    length_test = length_dataset[test_idx]

    return x_train, y_train, length_train, x_val, y_val, length_val, x_test, y_test, length_test


def load_dataset_crop_no_length(x_path, y_path, idx_splits, crop_layers):
    x_dataset = np.load(x_path)
    y_dataset = np.load(y_path).astype('float32')

    row_num = x_dataset.shape[1]
    column_num = x_dataset.shape[2]
    slice_num = x_dataset.shape[3]

    if not np.all(crop_layers == 0):
        x_dataset = x_dataset[:,
                    crop_layers[0][0]:(row_num - crop_layers[0][1]),
                    crop_layers[1][0]:(column_num - crop_layers[1][1]),
                    crop_layers[2][0]:(slice_num - crop_layers[2][1]), :]
        y_dataset = y_dataset - [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        y_dataset = y_dataset.astype('float32')

    train_idx = idx_splits[0]
    x_train = x_dataset[train_idx]
    y_train = y_dataset[train_idx]

    val_idx = idx_splits[1]
    x_val = x_dataset[val_idx]
    y_val = y_dataset[val_idx]

    test_idx = idx_splits[2]
    x_test = x_dataset[test_idx]
    y_test = y_dataset[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_dataset_crop_test_only(x_path, y_path, length_path, crop_layers):
    x_dataset = np.load(x_path)
    y_dataset = np.load(y_path).astype('float32')
    length_dataset = np.load(length_path).astype('float32')

    row_num = x_dataset.shape[1]
    column_num = x_dataset.shape[2]
    slice_num = x_dataset.shape[3]

    if not np.all(crop_layers == 0):
        x_dataset = x_dataset[:,
                    crop_layers[0][0]:(row_num - crop_layers[0][1]),
                    crop_layers[1][0]:(column_num - crop_layers[1][1]),
                    crop_layers[2][0]:(slice_num - crop_layers[2][1]), :]
        y_dataset = y_dataset - [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        y_dataset = y_dataset.astype('float32')
        # left ear
        length_dataset[range(0, 2000, 2)] = \
            length_dataset[range(0, 2000, 2)] + [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        # right ear, because of the flip
        length_dataset[range(1, 2000, 2)] = \
            length_dataset[range(1, 2000, 2)] + [crop_layers[1, 1], crop_layers[0, 0], crop_layers[2, 0]]
        length_dataset = length_dataset.astype('float32')

    return x_dataset, y_dataset, length_dataset


# load dataset from the separate files: X_dir, Y_dir, Length_dir
# idx_splits: [[train_idx], [val_idx], [test_idx]], idx from 0 to 19
# crop_layers: ndarray shape(3*2), [[row_ascending, row_descending], [column_a, column_d], [slice_a, slice_d]]
def load_dataset_crop_dir(x_dir, y_dir, length_dir):
    pat_names = MyDataset.get_pat_names()

    # Combine cropped volumes
    cropped_volumes = []
    cropped_points = []
    cropped_length = []

    for pat_name in pat_names:
        for aug_id in range(1, 51):
            print("**************" + pat_name + "__" + str(aug_id) + "***************")
            cropped_volume_left_path = x_dir + pat_name + "_augVolume_" + str(aug_id) + "_cropped_left.npy"
            cropped_volume_right_path = x_dir + pat_name + "_augVolume_" + str(aug_id) + "_cropped_right.npy"
            cropped_points_left_path = y_dir + pat_name + "_augPoints_" + str(aug_id) + "_cropped_left.npy"
            cropped_length_left_path = length_dir + pat_name + "_augLength_" + str(aug_id) + "_cropped_left.npy"
            cropped_points_right_path = y_dir + pat_name + "_augPoints_" + str(aug_id) + "_cropped_right.npy"
            cropped_length_right_path = length_dir + pat_name + "_augLength_" + str(aug_id) + "_cropped_right.npy"
            cropped_volume_left = np.load(cropped_volume_left_path)
            cropped_volume_right = np.load(cropped_volume_right_path)
            cropped_points_left = np.load(cropped_points_left_path)
            cropped_length_left = np.load(cropped_length_left_path)
            cropped_points_right = np.load(cropped_points_right_path)
            cropped_length_right = np.load(cropped_length_right_path)
            cropped_volumes.append(cropped_volume_left)
            cropped_volumes.append(cropped_volume_right)
            cropped_points.append(cropped_points_left)
            cropped_points.append(cropped_points_right)
            cropped_length.append(cropped_length_left)
            cropped_length.append(cropped_length_right)

    print(len(cropped_volumes))
    print(len(cropped_points))
    print(len(cropped_length))
    instances_num = len(cropped_volumes)

    cropped_volumes = np.asarray(cropped_volumes).reshape((instances_num, 200, 200, 160, 1))
    cropped_points = np.asarray(cropped_points).reshape((instances_num, 2, 3))
    cropped_length = np.asarray(cropped_length).reshape((instances_num, 2, 3))

    # read centre shift
    centre_shift = np.load("/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/data/processed/noises/noises_s1.5_test_dis.npy")
    # centre_shift = np.zeros((2000, 1, 3))

    cropped_volumes, cropped_points, cropped_length = \
        MyCrop.crop_outside_layers_trans(cropped_volumes, cropped_points, cropped_length, centre_shift)

    crop_size = "100x100x100"
    dataset_tag = "noises_s1.5_test_dis"
    save_dir_base = f"/data/gpfs/projects/punim1836/Data/cropped/{crop_size}/{dataset_tag}"
    # create the dir if not exist
    if os.path.exists(save_dir_base): print("Save dataset to: ", save_dir_base)
    else:
        os.makedirs(save_dir_base)
        print("Create dir and save dataset in it: ", save_dir_base)

    save_volume_path = f"{save_dir_base}/cropped_volumes_{crop_size}.npy"
    save_points_path = f"{save_dir_base}/cropped_points_{crop_size}.npy"
    save_length_path = f"{save_dir_base}/cropped_length_{crop_size}.npy"
    # add res for cropped volume, yes is all 0,15 just to make it the same in the training process
    save_res_path = f"{save_dir_base}/cropped_res_{crop_size}.npy"
    np.save(save_volume_path, cropped_volumes)
    print("saved: ", save_volume_path)
    np.save(save_points_path, cropped_points)
    print("saved: ", save_points_path)
    np.save(save_length_path, cropped_length)
    print("saved: ", save_length_path)
    np.save(save_res_path, np.ones((int(instances_num/2), 1, 3)) * 0.15)
    print("saved: ", save_res_path)

    return 1


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians (Euler-Rodrigues formula).
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def augment_fun(volume, points):
    # max rotation angle: 15 Degrees; change to Radian
    max_rot_angle = 15
    # random rotation angle: -max to +max
    rand_angle_xy = 2 * np.random.rand() * max_rot_angle - max_rot_angle
    rand_angle_yz = 2 * np.random.rand() * max_rot_angle - max_rot_angle
    rand_angle_xz = 2 * np.random.rand() * max_rot_angle - max_rot_angle

    v_left_15 = ndimage.rotate(volume, rand_angle_xy, axes=(0, 1), reshape=False)
    v_left_15 = ndimage.rotate(v_left_15, rand_angle_yz, axes=(0, 2), reshape=False)
    v_left_15 = ndimage.rotate(v_left_15, rand_angle_xz, axes=(1, 2), reshape=False)

    org_centre = np.asarray([200, 200, 160]) / 2.
    rot_centre = np.asarray(v_left_15.shape) / 2.

    v = points - org_centre

    rand_angle_xy_r = rand_angle_xy * math.pi / 180
    axis_0 = np.asarray([100, 100, 70]) - org_centre
    p_new = np.dot(rotation_matrix(axis_0, rand_angle_xy_r), v.T)

    rand_angle_yz_r = rand_angle_yz * math.pi / 180
    axis_1 = np.asarray([110, 100, 80]) - org_centre
    p_new = np.dot(rotation_matrix(axis_1, rand_angle_yz_r), p_new)

    rand_angle_xz_r = rand_angle_xz * math.pi / 180
    axis_2 = np.asarray([100, 90, 80]) - org_centre
    p_new = np.dot(rotation_matrix(axis_2, rand_angle_xz_r), p_new)

    p_new = p_new.T + rot_centre

    return v_left_15, p_new, np.asarray([rand_angle_xy_r, rand_angle_yz_r, rand_angle_xz_r])


def augment_cropped_patches(x_dir, y_dir):
    pat_names = MyDataset.get_pat_names()

    # augment num
    aug_num = 50
    base_dir = "/data/gpfs/projects/punim1836/Data/cropped/based_on_truth/augment_exp_pythong"

    # landmarks diff
    diff = np.load("/data/gpfs/projects/punim1836/Training/res/diff_flip.npy")
    p_id = 0

    for pat_name in pat_names:
        aug_id = 1
        # Combine cropped volumes for patient
        cropped_volumes = []
        cropped_points = []
        rand_angle = []

        print("**************" + pat_name + "__" + str(aug_id) + "***************")
        cropped_volume_left_path = x_dir + pat_name + "_augVolume_" + str(aug_id) + "_cropped_left.npy"
        cropped_volume_right_path = x_dir + pat_name + "_augVolume_" + str(aug_id) + "_cropped_right.npy"
        cropped_points_left_path = y_dir + pat_name + "_augPoints_" + str(aug_id) + "_cropped_left.npy"
        cropped_points_right_path = y_dir + pat_name + "_augPoints_" + str(aug_id) + "_cropped_right.npy"
        cropped_volume_left = np.load(cropped_volume_left_path)
        cropped_volume_right = np.load(cropped_volume_right_path)
        cropped_points_left = np.load(cropped_points_left_path)
        cropped_points_right = np.load(cropped_points_right_path)
        # for diff
        cropped_points_left = cropped_points_left - diff[p_id, 0:2, :]
        cropped_points_right = cropped_points_right - diff[p_id, 2:4, :]
        p_id = p_id + 1

        cropped_volumes.append(cropped_volume_left)
        cropped_volumes.append(cropped_volume_right)
        cropped_points.append(cropped_points_left)
        cropped_points.append(cropped_points_right)
        rand_angle.append(np.asarray([0, 0, 0]))
        rand_angle.append(np.asarray([0, 0, 0]))

        # augmentation
        for re_aug_id in range(1, aug_num):
            print("re_aug Id: ", re_aug_id)
            aug_left_volume, aug_left_points, rand_angle_r_left = augment_fun(cropped_volume_left, cropped_points_left)
            aug_right_volume, aug_right_points, rand_angle_r_right = augment_fun(cropped_volume_right,
                                                                                 cropped_points_right)

            cropped_volumes.append(aug_left_volume)
            cropped_volumes.append(aug_right_volume)
            cropped_points.append(aug_left_points)
            cropped_points.append(aug_right_points)
            rand_angle.append(rand_angle_r_left)
            rand_angle.append(rand_angle_r_right)

        cropped_volumes = np.asarray(cropped_volumes).reshape((aug_num * 2, 200, 200, 160, 1))
        cropped_points = np.asarray(cropped_points).reshape((aug_num * 2, 2, 3))
        rand_angle = np.asarray(rand_angle).reshape((aug_num * 2, 3))
        save_volume_path = f"{base_dir}/{pat_name}_volume_patch_aug_{aug_num * 2}.npy"
        save_points_path = f"{base_dir}/{pat_name}_points_aug_{aug_num * 2}.npy"
        save_angle_path = f"{base_dir}/{pat_name}_angles_aug_{aug_num * 2}.npy"
        np.save(save_volume_path, cropped_volumes)
        np.save(save_points_path, cropped_points)
        np.save(save_angle_path, rand_angle)
        print("saved: ", save_volume_path)
        print("saved: ", save_points_path)
        print("saved: ", save_angle_path)

    return 0


def load_patch_augmentation():
    pat_names = MyDataset.get_pat_names()
    base_dir = "/data/gpfs/projects/punim1836/Data/cropped/based_on_truth/augment_exp_pythong"
    save_dir = f"{base_dir}/x7575y7575z5050"

    aug_num = 100
    # Combine cropped volumes
    cropped_volumes = []
    cropped_points = []
    rand_angles = []

    crop_outside = np.asarray([[25, 25], [25, 25], [30, 30]])

    for pat_name in pat_names:
        print("**************" + pat_name + "__" + str(aug_num * 2) + "***************")
        cropped_volume_path = base_dir + "/" + pat_name + "_volume_patch_aug_" + str(aug_num * 2) + ".npy"
        cropped_points_path = base_dir + "/" + pat_name + "_points_aug_" + str(aug_num * 2) + ".npy"
        rand_angle_path = base_dir + "/" + pat_name + "_angles_aug_" + str(aug_num * 2) + ".npy"

        volumes = np.load(cropped_volume_path)
        points = np.load(cropped_points_path)
        angles = np.load(rand_angle_path)
        # crop first for memory saving
        volumes, points = MyCrop.crop_outside_layers_no_length(volumes, points, crop_outside, keep_blank=False)
        if len(cropped_volumes) == 0:
            cropped_volumes = volumes
            cropped_points = points
            rand_angles = angles
        else:
            cropped_volumes = np.concatenate((cropped_volumes, volumes), axis=0)
            cropped_points = np.concatenate((cropped_points, points), axis=0)
            rand_angles = np.concatenate((rand_angles, angles), axis=0)

    print("Volumes Shape: ", cropped_volumes.shape)
    print("Points Shape: ", cropped_points.shape)
    print("Rand angles Shape: ", rand_angles.shape)

    save_volumes_path = f"{save_dir}/volumes_4k.npy"
    save_points_path = f"{save_dir}/points_RoI_Medium_6_4k.npy"
    save_angles_path = f"{save_dir}/angles_RoI_4k.npy"
    np.save(save_volumes_path, cropped_volumes)
    np.save(save_points_path, cropped_points)
    np.save(save_angles_path, rand_angles)
    print("saved: ", save_volumes_path)
    print("saved: ", save_points_path)
    print("saved: ", save_angles_path)

    return 0


def load_dataset_divide(dataset_dir, rescaled_size, idx_splits, no_split=False):
    size_str = f"{rescaled_size[0]}x{rescaled_size[1]}x{rescaled_size[2]}"

    x_dataset_path = dataset_dir + "divided_volumes_" + size_str + ".npy"
    y_dataset_path = dataset_dir + "divided_points_" + size_str + ".npy"
    res_dataset_path = dataset_dir + "divided_res_" + size_str + ".npy"

    x_dataset = np.load(x_dataset_path)
    y_dataset = np.load(y_dataset_path).astype('float32')
    res_dataset = np.load(res_dataset_path).astype('float32')

    res_dataset_rep = np.repeat(res_dataset, 2, axis=1).reshape(2000, 1, 3)

    # without splitting to Train, Val and Test
    if no_split:
        return x_dataset, y_dataset, res_dataset_rep

    train_idx = idx_splits[0]
    x_train = x_dataset[train_idx]
    y_train = y_dataset[train_idx]
    res_train = res_dataset_rep[train_idx]

    val_idx = idx_splits[1]
    x_val = x_dataset[val_idx]
    y_val = y_dataset[val_idx]
    res_val = res_dataset_rep[val_idx]

    test_idx = idx_splits[2]
    x_test = x_dataset[test_idx]
    y_test = y_dataset[test_idx]
    res_test = res_dataset_rep[test_idx]

    return x_train, y_train, res_train, \
        x_val, y_val, res_val, \
        x_test, y_test, res_test


@tf.function
def train_step(model, err_fun, eval_metric, optimizer, x, y, res):
    with tf.GradientTape() as tape:
        # Compute the loss value for this batch.
        y_pred = model(x, training=True)
        mse_res = err_fun(y, y_pred, res)

    # Update training metric.
    eval_metric.update_state(mse_res)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(mse_res, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return mse_res


@tf.function
def val_step(model, err_fun, eval_metric, x, y, res):
    y_pred = model(x, training=False)
    mse_res = err_fun(y, y_pred, res)
    # Update val metrics
    eval_metric.update_state(mse_res)


@tf.function
def test_step(model, err_fun, eval_metric, x, y, res):
    y_pred = model(x, training=False)
    mse_res = err_fun(y, y_pred, res)
    # Update test metrics
    eval_metric.update_state(mse_res)
    return y_pred


def my_evaluate(eval_model, err_fun, eval_metric, eval_dataset):
    # Run a test loop when meet the best val result.
    for s, (x_batch_test, y_batch_test, res_batch_test) in enumerate(eval_dataset):
        if s == 0:
            y_test_p = test_step(eval_model, err_fun, eval_metric, x_batch_test, y_batch_test, res_batch_test)
        else:
            y_test = test_step(eval_model, err_fun, eval_metric, x_batch_test, y_batch_test, res_batch_test)
            y_test_p = np.concatenate((y_test_p, y_test), axis=0)

    eval_metric_result = eval_metric.result()
    eval_metric.reset_states()
    return eval_metric_result, y_test_p

#%%
