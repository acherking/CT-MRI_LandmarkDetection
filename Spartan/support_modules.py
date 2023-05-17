import numpy as np

import Functions.MyDataset as MyDataset


# load dataset from various directories (train, val, test), volume and pts are separated.
# tvt
def load_data():
    # Set Data Path
    data_base_path = "/data/gpfs/projects/punim1836/Data/combined_aug_data/"
    x_train_path = data_base_path + "X_train_data.npy"
    y_train_path = data_base_path + "Y_train_data.npy"
    x_val_path = data_base_path + "X_val_data.npy"
    y_val_path = data_base_path + "Y_val_data.npy"
    x_test_path = data_base_path + "X_test_data.npy"
    y_test_path = data_base_path + "Y_test_data.npy"

    x_train = np.load(x_train_path, allow_pickle=True)
    y_train = np.load(y_train_path, allow_pickle=True)
    x_val = np.load(x_val_path, allow_pickle=True)
    y_val = np.load(y_val_path, allow_pickle=True)
    x_test = np.load(x_test_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    # Data shape validation
    print("X_train Shape: ", np.shape(x_train))
    print("Y_train Shape: ", np.shape(y_train))
    print("X_val Shape: ", np.shape(x_val))
    print("Y_val Shape: ", np.shape(y_val))
    print("X_test Shape: ", np.shape(x_test))
    print("Y_test Shape: ", np.shape(y_test))

    # Reshape the data (data-size, row-size?, column-size?, slice-size, channel-size)
    x_train_reshape = np.asarray(x_train).reshape((700, 170, 170, 30, 1))
    x_val_reshape = np.asarray(x_val).reshape((100, 170, 170, 30, 1))
    x_test_reshape = np.asarray(x_test).reshape((200, 170, 170, 30, 1))

    print("X_train_reshape Shape: ", np.shape(x_train))
    print("X_val_reshape Shape: ", np.shape(x_val))
    print("X_test_reshape Shape: ", np.shape(x_test))

    return x_train_reshape, y_train, x_val_reshape, y_val, x_test_reshape, y_test


# load dataset from a single directory, each file contains both volume and pts (X & Y).
def load_dataset(dir_path, size=(176, 176, 48), pat_splits=[], with_res=False, only_test=False):
    # file name format: {name}_{size}_VolPts_{id}.mat (AH_17617648_VolPts.mat)
    pat_names = ['AH', 'AZ', 'DE', 'DM', 'DM2', 'DGL', 'FA', 'GE', 'GM', 'GP', 'HB', 'HH',
                 'JH', 'JM', 'LG', 'LP', 'MJ', 'NV', 'PH', 'SM']
    str_size = str(size[0]) + str(size[1]) + str(size[2])

    if not pat_splits:
        np.random.shuffle(pat_names)
        name_splits = np.split(pat_names, [int(.7 * len(pat_names)), int(.8 * len(pat_names))])
    else:
        name_splits = [[pat_names[i] for i in pat_splits[0]],
                       [pat_names[i] for i in pat_splits[1]], [pat_names[i] for i in pat_splits[2]]]

    x_train = []
    y_train = []
    res_train = []
    if not only_test:
        for pt_name in name_splits[0]:
            for aug_id in range(1, 51):
                file_path = dir_path + pt_name + "_" + str_size + "_VolPts_" + str(aug_id) + ".mat"
                volume, pts, res = MyDataset.load_mat_data(file_path, "None", with_res)
                x_train.append(volume)
                y_train.append(pts)
                if with_res:
                    res_train.append(res)

    x_val = []
    y_val = []
    res_val = []
    if not only_test:
        for pt_name in name_splits[1]:
            for aug_id in range(1, 51):
                file_path = dir_path + pt_name + "_" + str_size + "_VolPts_" + str(aug_id) + ".mat"
                volume, pts, res = MyDataset.load_mat_data(file_path, "None", with_res)
                x_val.append(volume)
                y_val.append(pts)
                if with_res:
                    res_val.append(res)

    x_test = []
    y_test = []
    res_test = []
    for pt_name in name_splits[2]:
        for aug_id in range(1, 51):
            file_path = dir_path + pt_name + "_" + str_size + "_VolPts_" + str(aug_id) + ".mat"
            volume, pts, res = MyDataset.load_mat_data(file_path, "None", with_res)
            x_test.append(volume)
            y_test.append(pts)
            if with_res:
                res_test.append(res)

    if not only_test:
        # Data shape validation
        print("X_train Shape: ", np.shape(x_train))
        print("Y_train Shape: ", np.shape(y_train))
        print("X_val Shape: ", np.shape(x_val))
        print("Y_val Shape: ", np.shape(y_val))
    print("X_test Shape: ", np.shape(x_test))
    print("Y_test Shape: ", np.shape(y_test))

    if not only_test:
        # Reshape the data (data-size, row-size?, column-size?, slice-size, channel-size)
        x_train_reshape = np.asarray(x_train).reshape((700, size[0], size[1], size[2], 1))
        # y_train_one = np.asarray(y_train)[:, 0, :]
        y_train = np.asarray(y_train).astype('float32')
        if with_res:
            res_train = np.asarray(res_train).astype('float32').reshape((700, 1, 3))
            res_train[:, :, [0, 1]] = res_train[:, :, [1, 0]]
        x_val_reshape = np.asarray(x_val).reshape((100, size[0], size[1], size[2], 1))
        # y_val_one = np.asarray(y_val)[:, 0, :]
        y_val = np.asarray(y_val).astype('float32')
        if with_res:
            res_val = np.asarray(res_val).astype('float32').reshape((100, 1, 3))
            res_val[:, :, [0, 1]] = res_val[:, :, [1, 0]]
    x_test_reshape = np.asarray(x_test).reshape((200, size[0], size[1], size[2], 1))
    # y_test_one = np.asarray(y_test)[:, 0, :]
    y_test = np.asarray(y_test).astype('float32')
    if with_res:
        res_test = np.asarray(res_test).astype('float32').reshape((200, 1, 3))
        res_test[:, :, [0, 1]] = res_test[:, :, [1, 0]]

    if not only_test:
        print("X_train_reshape Shape: ", np.shape(x_train_reshape))
        # print("Y_train_one Shape: ", np.shape(y_train_one))
        print("Y_train Shape: ", np.shape(y_train))
        print("X_val_reshape Shape: ", np.shape(x_val_reshape))
        # print("Y_val_one Shape: ", np.shape(y_val_one))
        print("Y_val Shape: ", np.shape(y_val))
    print("X_test_reshape Shape: ", np.shape(x_test_reshape))
    # print("Y_test_one Shape: ", np.shape(y_test_one))
    print("Y_test Shape: ", np.shape(y_test))

    if not only_test:
        return x_train_reshape, y_train, res_train, x_val_reshape, y_val, res_val, x_test_reshape, y_test, res_test
    else:
        return x_test_reshape, y_test, res_test


# load dataset from the combination data files: X and Y
# idx_splits: [[train_idx], [val_idx], [test_idx]], idx from 0 to 19
# crop_layers: ndarray shape(3*2), [[row_ascending, row_descending], [column_a, column_d], [slice_a, slice_d]]
def load_dataset_crop(x_path, y_path, length_path, pat_splits, crop_layers):
    x_dataset = np.load(x_path)
    y_dataset = np.load(y_path).astype('float32')
    length_dataset = np.load(length_path).astype('float32')

    if not np.all(crop_layers == 0):
        x_dataset = x_dataset[:,
                    crop_layers[0][0]:(100 - crop_layers[0][1]),
                    crop_layers[1][0]:(100 - crop_layers[1][1]),
                    crop_layers[2][0]:(100 - crop_layers[2][1]), :]
        y_dataset = y_dataset - [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        y_dataset = y_dataset.astype('float32')
        # left ear
        length_dataset[range(0, 2000, 2)] = \
            length_dataset[range(0, 2000, 2)] + [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        # right ear, because of the flip
        length_dataset[range(1, 2000, 2)] = \
            length_dataset[range(1, 2000, 2)] + [crop_layers[1, 1], crop_layers[0, 0], crop_layers[2, 0]]
        length_dataset = length_dataset.astype('float32')

    idx_splits = [[list(range(i * 100, i * 100 + 100)) for i in j] for j in pat_splits]
    for i in range(0, 3):
        idx_splits[i] = [num for sublist in idx_splits[i] for num in sublist]
        idx_splits[i] = np.asarray(idx_splits[i])
        # np.random.shuffle(idx_splits[i])

    train_idx = idx_splits[0]
    np.random.shuffle(train_idx)
    x_train = x_dataset[train_idx]
    y_train = y_dataset[train_idx]
    length_train = length_dataset[train_idx]

    val_idx = idx_splits[1]
    np.random.shuffle(val_idx)
    x_val = x_dataset[val_idx]
    y_val = y_dataset[val_idx]
    length_val = length_dataset[val_idx]

    test_idx = idx_splits[2]
    # np.random.shuffle(test_idx)
    x_test = x_dataset[test_idx]
    y_test = y_dataset[test_idx]
    length_test = length_dataset[test_idx]

    return x_train, y_train, length_train, x_val, y_val, length_val, x_test, y_test, length_test


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

    cropped_volumes = np.asarray(cropped_volumes).reshape((2000, 200, 200, 100, 1))
    cropped_points = np.asarray(cropped_points).reshape((2000, 2, 3))
    cropped_length = np.asarray(cropped_length).reshape((2000, 2, 3))

    crop_size = "x100100y100100z5050"
    comb_tag = "truth"
    save_comb_dir = f"/data/gpfs/projects/punim1836/Data/cropped/based_on_truth/{crop_size}"
    save_volume_path = f"{save_comb_dir}/cropped_volumes_{crop_size}_{comb_tag}.npy"
    save_points_path = f"{save_comb_dir}/cropped_points_{crop_size}_{comb_tag}.npy"
    save_length_path = f"{save_comb_dir}/cropped_length_{crop_size}_{comb_tag}.npy"
    np.save(save_volume_path, cropped_volumes)
    print("saved: ", save_volume_path)
    np.save(save_points_path, cropped_points)
    print("saved: ", save_points_path)
    np.save(save_length_path, cropped_length)
    print("saved: ", save_length_path)

    return 1


def load_dataset_divide(dataset_dir, rescaled_size, pat_splits):
    size_str = f"{rescaled_size[0]}{rescaled_size[1]}{rescaled_size[2]}"

    x_dataset_path = dataset_dir + "divided_volumes_" + size_str + ".npy"
    y_dataset_path = dataset_dir + "divided_points_" + size_str + ".npy"
    length_dataset_path = dataset_dir + "divided_length_" + size_str + ".npy"
    res_dataset_path = dataset_dir + "res_array_" + size_str + ".npy"

    x_dataset = np.load(x_dataset_path)
    y_dataset = np.load(y_dataset_path).astype('float32')
    length_dataset = np.load(length_dataset_path).astype('float32')
    res_dataset = np.load(res_dataset_path).astype('float32')

    res_dataset_rep = np.repeat(res_dataset, 2, axis=1).reshape(2000, 1, 3)

    right_length = np.zeros(length_dataset.shape)
    length_dataset = np.concatenate((length_dataset, right_length), axis=1).reshape((length_dataset.shape[0] * 2, 1))

    idx_splits = [[list(range(i * 100, i * 100 + 100)) for i in j] for j in pat_splits]
    for i in range(0, 3):
        idx_splits[i] = [num for sublist in idx_splits[i] for num in sublist]
        idx_splits[i] = np.asarray(idx_splits[i])
        # np.random.shuffle(idx_splits[i])

    train_idx = idx_splits[0]
    np.random.shuffle(train_idx)
    x_train = x_dataset[train_idx]
    y_train = y_dataset[train_idx]
    res_train = res_dataset_rep[train_idx]
    length_train = length_dataset[train_idx]

    val_idx = idx_splits[1]
    np.random.shuffle(val_idx)
    x_val = x_dataset[val_idx]
    y_val = y_dataset[val_idx]
    res_val = res_dataset_rep[val_idx]
    length_val = length_dataset[val_idx]

    test_idx = idx_splits[2]
    # np.random.shuffle(test_idx)
    x_test = x_dataset[test_idx]
    y_test = y_dataset[test_idx]
    res_test = res_dataset_rep[test_idx]
    length_test = length_dataset[test_idx]

    return x_train, y_train, res_train, length_train, \
        x_val, y_val, res_val, length_val, \
        x_test, y_test, res_test, length_test
