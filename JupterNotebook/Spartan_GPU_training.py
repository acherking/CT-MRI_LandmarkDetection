import tensorflow as tf

from os import listdir
from os.path import isfile, join
import numpy as np
import h5py

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def load_mat_data(x_base_path, y_base_path):
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
        file_vol = h5py.File(x_file_path, 'r')
        file_pts = h5py.File(y_file_path, 'r')
        load_mat_vol = file_vol.get('rescaled_aug_vol')
        load_mat_pts = file_pts.get('rescaled_aug_pts')
        x_dataset.append(np.array(load_mat_vol).T)
        y_dataset.append(np.array(load_mat_pts).reshape(3, 4).T)
        file_vol.close()
        file_pts.close()

    return x_dataset, y_dataset


# Set Data Path
X_train_base_path = "/data/gpfs/projects/punim1836/Data/augmentation_from_matlab/Train/Input/"
Y_train_base_path = "/data/gpfs/projects/punim1836/Data/augmentation_from_matlab/Train/Output/"
X_val_base_path = "/data/gpfs/projects/punim1836/Data/augmentation_from_matlab/Val/Input/"
Y_val_base_path = "/data/gpfs/projects/punim1836/Data/augmentation_from_matlab/Val/Output/"
X_test_base_path = "/data/gpfs/projects/punim1836/Data/augmentation_from_matlab/Test/Input/"
Y_test_base_path = "/data/gpfs/projects/punim1836/Data/augmentation_from_matlab/Test/Output/"

X_train, Y_train = load_mat_data(X_train_base_path, Y_train_base_path)
X_val, Y_val = load_mat_data(X_val_base_path, Y_val_base_path)
X_test, Y_test = load_mat_data(X_test_base_path, Y_test_base_path)

# Data shape validation
print("X_train Shape: ", np.shape(X_train))
print("Y_train Shape: ", np.shape(Y_train))

print("X_val Shape: ", np.shape(X_val))
print("Y_val Shape: ", np.shape(Y_val))

print("X_test Shape: ", np.shape(X_test))
print("Y_test Shape: ", np.shape(Y_test))

# Reshape the data (data-size, row-size?, column-size?, slice-size, channel-size)
X_train_reshape = np.asarray(X_train).reshape(650, 170, 170, 30, 1)
Y_train_one = np.asarray(Y_train)[:, 0, :]

X_val_reshape = np.asarray(X_val).reshape(100, 170, 170, 30, 1)
Y_val_one = np.asarray(Y_val)[:, 0, :]

X_test_reshape = np.asarray(X_test).reshape(200, 170, 170, 30, 1)
Y_test_one = np.asarray(Y_test)[:, 0, :]

print("X_train_reshape Shape: ", np.shape(X_train))
print("Y_train_one Shape: ", np.shape(Y_train))

print("X_val_reshape Shape: ", np.shape(X_val))
print("Y_val_one Shape: ", np.shape(Y_val))

print("X_test_reshape Shape: ", np.shape(X_test))
print("Y_test_one Shape: ", np.shape(Y_test))