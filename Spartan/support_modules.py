import numpy as np


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
    x_test = np.load(x_val_path, allow_pickle=True)
    y_test = np.load(y_val_path, allow_pickle=True)

    # Data shape validation
    print("X_train Shape: ", np.shape(x_train))
    print("Y_train Shape: ", np.shape(y_train))
    print("X_val Shape: ", np.shape(x_val))
    print("Y_val Shape: ", np.shape(y_val))
    print("X_test Shape: ", np.shape(x_test))
    print("Y_test Shape: ", np.shape(y_test))

    # Reshape the data (data-size, row-size?, column-size?, slice-size, channel-size)
    x_train_reshape = np.asarray(x_train).reshape(700, 170, 170, 30, 1)
    y_train_one = np.asarray(y_train)[:, 0, :]
    x_val_reshape = np.asarray(x_val).reshape(100, 170, 170, 30, 1)
    y_val_one = np.asarray(y_val)[:, 0, :]
    x_test_reshape = np.asarray(x_val).reshape(200, 170, 170, 30, 1)
    y_test_one = np.asarray(y_val)[:, 0, :]

    print("X_train_reshape Shape: ", np.shape(x_train))
    print("Y_train_one Shape: ", np.shape(y_train_one))
    print("X_val_reshape Shape: ", np.shape(x_val))
    print("Y_val_one Shape: ", np.shape(y_val_one))
    print("X_test_reshape Shape: ", np.shape(x_test))
    print("Y_test_one Shape: ", np.shape(y_test_one))

    return x_train_reshape, y_train_one, x_val_reshape, y_val_one, x_test_reshape, y_test_one

