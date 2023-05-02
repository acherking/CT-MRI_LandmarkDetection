import tensorflow as tf
from tensorflow import keras
import numpy as np

import Functions.MyDataset as MyDataset
import Functions.MyCrop as MyCrop
import support_modules as supporter


@tf.function
def predict(x, model_f):
    y_pred_f = model_f(x, training=False)
    return y_pred_f


def predict_dataset(model_path_f, x_test):
    # Load the Trained Model
    model = keras.models.load_model(model_path_f)
    model.summary()

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(2)

    for step, x_batch_test in enumerate(test_dataset):
        y = predict(x_batch_test, model)
        if step == 0:
            y_test_pred = np.copy(y)
        else:
            y_test_pred = np.concatenate((y_test_pred, np.copy(y)), axis=0)

    return y_test_pred


crop_size = (100, 100, 100)

X_path = "/data/gpfs/projects/punim1836/Data/cropped/cropped_volumes_x5050y5050z5050.npy"
Y_path = "/data/gpfs/projects/punim1836/Data/cropped/cropped_points_x5050y5050z5050.npy"
Cropped_length_path = "/data/gpfs/projects/punim1836/Data/cropped/cropped_length_x5050y5050z5050.npy"
pat_splits = MyDataset.get_pat_splits(static=True)
X_train, Y_train, length_train, X_val, Y_val, length_val, X_test, Y_test, length_test = \
    supporter.load_dataset_crop(X_path, Y_path, Cropped_length_path, pat_splits)

Y_train_one = np.asarray(Y_train)[:, 0, :].reshape((1400, 1, 3))
Y_val_one = np.asarray(Y_val)[:, 0, :].reshape((200, 1, 3))
Y_test_one = np.asarray(Y_test)[:, 0, :].reshape((400, 1, 3))

# y_tag: "one_landmark" -> OL, "two_landmarks" -> TL, "mean_two_landmarks" -> MTL
y_tag = "one_landmark"
model_name = "straight_model"
model_tag = "cropped"
model_size = f"{crop_size[0]}_{crop_size[1]}_{crop_size[2]}"
model_label = f"{model_name}_{model_tag}_{model_size}"
base_dir = "/data/gpfs/projects/punim1836/Training/trained_models/"
model_dir = f"{base_dir}{model_tag}_dataset/{model_name}/{y_tag}"
save_dir = f"{model_dir}/corroding_test/"

# some configuration (make life easier?)
model_path = f"{model_dir}/bestVal_{model_label}"

X_dataset = np.copy(X_test)
fill_val = np.min(X_dataset)
# cut layers, Shape e.g. (n, 100, 100, 100, 1)
for cut_layers_num in range(0, 25):
    X_dataset_corroded = np.ones(X_dataset.shape) * fill_val
    l_idx = cut_layers_num
    h_idx = crop_size[0] - cut_layers_num
    X_dataset_corroded[:, l_idx:h_idx, l_idx:h_idx, l_idx:h_idx, :] = \
        X_dataset[:, l_idx:h_idx, l_idx:h_idx, l_idx:h_idx, :]

