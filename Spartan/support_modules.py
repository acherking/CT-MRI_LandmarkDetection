import numpy as np

import Functions.MyDataset as MyDataset


# load dataset from various directories (train, val, test), volume and pts are separated.
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
    y_train_one = np.asarray(y_train)[:, 0, :]
    x_val_reshape = np.asarray(x_val).reshape((100, 170, 170, 30, 1))
    y_val_one = np.asarray(y_val)[:, 0, :]
    x_test_reshape = np.asarray(x_test).reshape((200, 170, 170, 30, 1))
    y_test_one = np.asarray(y_test)[:, 0, :]

    print("X_train_reshape Shape: ", np.shape(x_train))
    print("Y_train_one Shape: ", np.shape(y_train_one))
    print("X_val_reshape Shape: ", np.shape(x_val))
    print("Y_val_one Shape: ", np.shape(y_val_one))
    print("X_test_reshape Shape: ", np.shape(x_test))
    print("Y_test_one Shape: ", np.shape(y_test_one))

    return x_train_reshape, y_train_one, x_val_reshape, y_val_one, x_test_reshape, y_test_one


# load dataset from a single directory, each file contains both volume and pts (X & Y).
def load_dataset(dir_path, size=(176, 176, 48)):
    # file name format: {name}_{size}_VolPts_{id}.mat (AH_17617648_VolPts.mat)
    pat_names = ['AH', 'AZ', 'DE', 'DM', 'DM2', 'DGL', 'FA', 'GE', 'GM', 'GP', 'HB', 'HH',
                 'JH', 'JM', 'LG', 'LP', 'MJ', 'NV', 'PH', 'SM']
    str_size = str(size[0]) + str(size[1]) + str(size[2])

    pat_names = np.asarray(pat_names)
    np.random.shuffle(pat_names)
    name_splits = np.split(pat_names, [int(.7 * len(pat_names)), int(.8 * len(pat_names))])

    x_train = []
    y_train = []
    for pt_name in name_splits[0]:
        for aug_id in range(1, 51):
            file_path = dir_path + pt_name + "_" + str_size + "_VolPts_" + str(aug_id) + ".mat"
            volume, pts = MyDataset.load_mat_data(file_path)
            x_train.append(volume)
            y_train.append(pts)

    x_val = []
    y_val = []
    for pt_name in name_splits[1]:
        for aug_id in range(1, 51):
            file_path = dir_path + pt_name + "_" + str_size + "_VolPts_" + str(aug_id) + ".mat"
            volume, pts = MyDataset.load_mat_data(file_path)
            x_val.append(volume)
            y_val.append(pts)

    x_test = []
    y_test = []
    for pt_name in name_splits[2]:
        for aug_id in range(1, 51):
            file_path = dir_path + pt_name + "_" + str_size + "_VolPts_" + str(aug_id) + ".mat"
            volume, pts = MyDataset.load_mat_data(file_path)
            x_test.append(volume)
            y_test.append(pts)

    # Data shape validation
    print("X_train Shape: ", np.shape(x_train))
    print("Y_train Shape: ", np.shape(y_train))
    print("X_val Shape: ", np.shape(x_val))
    print("Y_val Shape: ", np.shape(y_val))
    print("X_test Shape: ", np.shape(x_test))
    print("Y_test Shape: ", np.shape(y_test))

    # Reshape the data (data-size, row-size?, column-size?, slice-size, channel-size)
    x_train_reshape = np.asarray(x_train).reshape((700, size[0], size[1], size[2], 1))
    # y_train_one = np.asarray(y_train)[:, 0, :]
    y_train = np.asarray(y_train).astype('float32')
    x_val_reshape = np.asarray(x_val).reshape((100, size[0], size[1], size[2], 1))
    # y_val_one = np.asarray(y_val)[:, 0, :]
    y_val = np.asarray(y_val).astype('float32')
    x_test_reshape = np.asarray(x_test).reshape((200, size[0], size[1], size[2], 1))
    # y_test_one = np.asarray(y_test)[:, 0, :]
    y_test = np.asarray(y_test).astype('float32')

    print("X_train_reshape Shape: ", np.shape(x_train_reshape))
    # print("Y_train_one Shape: ", np.shape(y_train_one))
    print("Y_train Shape: ", np.shape(y_train))
    print("X_val_reshape Shape: ", np.shape(x_val_reshape))
    # print("Y_val_one Shape: ", np.shape(y_val_one))
    print("Y_val Shape: ", np.shape(y_val))
    print("X_test_reshape Shape: ", np.shape(x_test_reshape))
    # print("Y_test_one Shape: ", np.shape(y_test_one))
    print("Y_test Shape: ", np.shape(y_test))

    return x_train_reshape, y_train, x_val_reshape, y_val, x_test_reshape, y_test
