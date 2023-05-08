import tensorflow as tf
from tensorflow import keras
import numpy as np

import Functions.MyDataset as MyDataset
import Functions.MyCrop as MyCrop
import support_modules as supporter
import models


mse = tf.keras.losses.MeanSquaredError()
mse_res = models.mse_with_res
test_mse_metric = keras.metrics.Mean()


@tf.function
def test_step(x, y, res, eva_model):
    y_pred = eva_model(x, training=False)
    mse_pixel = mse_res(y, y_pred, res)
    err_mm = mse_pixel
    # Update test metrics
    test_mse_metric.update_state(mse_pixel)
    return y_pred


def my_evaluate(eva_model, test_dataset):
    # Run a test loop when meet the best val result.
    for step, (x_batch_test, y_batch_test, res_batch_test) in enumerate(test_dataset):
        y = test_step(x_batch_test, y_batch_test, res_batch_test, eva_model)
        if step == 0:
            y_test_pred = np.copy(y)
        else:
            y_test_pred = np.concatenate((y_test_pred, np.copy(y)), axis=0)

    test_mse_res_f = test_mse_metric.result()
    test_mse_metric.reset_states()
    return test_mse_res_f.numpy(), y_test_pred


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

res_train = (np.ones((1400, 1, 3)) * 0.15).astype('float32')
res_val = (np.ones((200, 1, 3)) * 0.15).astype('float32')
res_test = (np.ones((400, 1, 3)) * 0.15).astype('float32')

# y_tag: "one_landmark" -> OL, "two_landmarks" -> TL, "mean_two_landmarks" -> MTL
y_tag = "one_landmark"
model_name = "straight_model"
model_tag = "cropped"
model_size = f"{crop_size[0]}_{crop_size[1]}_{crop_size[2]}"
model_label = f"{model_name}_{model_tag}_{model_size}"
base_dir = "/data/gpfs/projects/punim1836/Training/trained_models"
model_dir = f"{base_dir}/{model_tag}_dataset/{model_name}/{y_tag}"
save_dir = f"{model_dir}/corroding_test"

err_array_file = f"{save_dir}/err_corrode51*51*51"

# some configuration (make life easier?)
model_path = f"{model_dir}/bestVal_{model_label}"
model = keras.models.load_model(model_path)

X_dataset = np.copy(X_test)
Y_dataset = np.copy(Y_test_one)
res_dataset = np.copy(res_test)

err_list = []
err_array = np.ones((15, 15, 15)) * -1
fill_val = np.min(X_dataset)
# cut layers, Shape e.g. (n, 100, 100, 100, 1)
for cut_row_num in range(0, 15):
    for cut_column_num in range(0, 15):
        for cut_slice_num in range(0, 15):
            X_dataset_corroded = np.ones(X_dataset.shape) * fill_val
            X_dataset_corroded[:,
                cut_column_num:(100-cut_column_num),
                cut_column_num:(100-cut_column_num),
                cut_slice_num:(100-cut_slice_num), :] \
                = X_dataset[:,
                  cut_column_num:(100-cut_column_num),
                  cut_column_num:(100-cut_column_num),
                  cut_slice_num:(100-cut_slice_num), :]

            # double-check the corroded data
            # print("fill val: ", fill_val)
            # print("row&slice, centre like:", X_dataset_corroded[0, 49, :, 49, 0])
            # print("column&slice, centre like:", X_dataset_corroded[0, :, 49, 49, 0])
            # print("row&column, centre like:", X_dataset_corroded[0, 49, 49, :, 0])

            dataset = tf.data.Dataset.from_tensor_slices((X_dataset_corroded, Y_dataset, res_dataset)).batch(2)
            err, _ = my_evaluate(model, dataset)
            err_array[cut_row_num, cut_column_num, cut_slice_num] = err
            print(f"({cut_row_num}:{cut_column_num}:{cut_slice_num}), MSE with res (mm^2 per 1/2 points): ", err)

np.save(err_array_file, err_array)
print("Saved: ", err_array_file)
