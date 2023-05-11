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


def corrode_sym_rcs(x_dataset, y_dataset, res_dataset, model_f, err_array_file_f):
    err_array = np.ones((15, 15, 15)) * -1
    fill_val = np.min(x_dataset)
    # cut layers, Shape e.g. (n, 100, 100, 100, 1)
    for cut_row_num in range(0, 15):
        for cut_column_num in range(0, 15):
            for cut_slice_num in range(0, 15):
                x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
                x_dataset_corroded[:,
                    cut_row_num:(100 - cut_row_num),
                    cut_column_num:(100 - cut_column_num),
                    cut_slice_num:(100 - cut_slice_num), :] \
                    = x_dataset[:,
                      cut_row_num:(100 - cut_row_num),
                      cut_column_num:(100 - cut_column_num),
                      cut_slice_num:(100 - cut_slice_num), :]

                # double-check the corroded data
                # print("fill val: ", fill_val)
                # print("row&slice, centre like:", x_dataset_corroded[0, 49, :, 49, 0])
                # print("column&slice, centre like:", x_dataset_corroded[0, :, 49, 49, 0])
                # print("row&column, centre like:", x_dataset_corroded[0, 49, 49, :, 0])

                dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
                err, _ = my_evaluate(model_f, dataset)
                err_array[cut_row_num, cut_column_num, cut_slice_num] = err
                print(f"({cut_row_num}:{cut_column_num}:{cut_slice_num}), MSE with res (mm^2 per 1/2 points): ", err)

    np.save(err_array_file_f, err_array)
    print("Saved: ", err_array_file_f)


def corrode_asym_rcs(x_dataset, y_dataset, res_dataset, model_f, err_array_file_f):
    """

    """
    err_array = np.ones((6, 50)) * -1
    fill_val = np.min(x_dataset)
    # cut layers, Shape e.g. (n, 100, 100, 100, 1)
    for surf_id in range(0, 6):
        for cut_num in range(0, 50):
            x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
            if surf_id == 0:
                corrode_info = "cut row num from ascending order"
                x_dataset_corroded[:, cut_num:, :, :, :] = x_dataset[:, cut_num:, :, :, :]
            elif surf_id == 1:
                corrode_info = "cut row num from descending order"
                x_dataset_corroded[:, 0:(100-cut_num), :, :, :] = x_dataset[:, 0:(100-cut_num), :, :, :]
            elif surf_id == 2:
                corrode_info = "cut column num from ascending order"
                x_dataset_corroded[:, :, cut_num:, :, :] = x_dataset[:, :, cut_num:, :, :]
            elif surf_id == 3:
                corrode_info = "cut column num from descending order"
                x_dataset_corroded[:, :, 0:(100-cut_num), :, :] = x_dataset[:, :, 0:(100-cut_num), :, :]
            elif surf_id == 4:
                corrode_info = "cut slice num from ascending order"
                x_dataset_corroded[:, :, :, cut_num:, :] = x_dataset[:, :, :, cut_num:, :]
            elif surf_id == 5:
                corrode_info = "cut slice num from descending order"
                x_dataset_corroded[:, :, :, 0:(100-cut_num), :] = x_dataset[:, :, :, 0:(100-cut_num), :]
            dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)

            err, _ = my_evaluate(model_f, dataset)
            err_array[surf_id, cut_num] = err
            print(f"{corrode_info}: {cut_num}, MSE with res (mm^2 per 1/2 points): ", err)

    np.save(err_array_file_f, err_array)
    print("Saved: ", err_array_file_f)

    err_array_ind = np.ones((6, 50)) * -1
    # cut row ascending
    for cut_row_a in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a:, :, :, :] = x_dataset[:, cut_row_a:, :, :, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        err, _ = my_evaluate(model_f, dataset)
        err_array_ind[0, cut_row_a] = err
        print(f"Cut row ascending: {cut_row_a}, MSE with res (mm^2 per 1/2 points): ", err)
    cut_row_a_max = np.argmin(err_array_ind, axis=1)[0].astype('int')
    # cut row descending
    for cut_row_d in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d), :, :, :] = x_dataset[:, cut_row_a_max:(100-cut_row_d), :, :, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        err, _ = my_evaluate(model_f, dataset)
        err_array_ind[1, cut_row_d] = err
        print(f"Cut row descending: {cut_row_d}, MSE with res (mm^2 per 1/2 points): ", err)
    cut_row_d_max = np.argmin(err_array_ind, axis=1)[1].astype('int')
    # cut column ascending
    for cut_column_a in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a:, :, :] = \
            x_dataset[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a:, :, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        err, _ = my_evaluate(model_f, dataset)
        err_array_ind[2, cut_column_a] = err
        print(f"Cut column ascending: {cut_column_a}, MSE with res (mm^2 per 1/2 points): ", err)
    cut_column_a_max = np.argmin(err_array_ind, axis=1)[2].astype('int')
    # cut column descending
    for cut_column_d in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d), :, :] = \
            x_dataset[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d), :, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        err, _ = my_evaluate(model_f, dataset)
        err_array_ind[3, cut_column_d] = err
        print(f"Cut column descending: {cut_column_d}, MSE with res (mm^2 per 1/2 points): ", err)
    cut_column_d_max = np.argmin(err_array_ind, axis=1)[3].astype('int')
    # cut slice ascending
    for cut_slice_a in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d_max), cut_slice_a:, :] = \
            x_dataset[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d_max), cut_slice_a:, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        err, _ = my_evaluate(model_f, dataset)
        err_array_ind[4, cut_slice_a] = err
        print(f"Cut slice ascending: {cut_slice_a}, MSE with res (mm^2 per 1/2 points): ", err)
    cut_slice_a_max = np.argmin(err_array_ind, axis=1)[4].astype('int')
    # cut slice ascending
    for cut_slice_d in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d_max), cut_slice_a_max:(100-cut_slice_d), :] = \
            x_dataset[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d_max), cut_slice_a_max:(100-cut_slice_d), :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        err, _ = my_evaluate(model_f, dataset)
        err_array_ind[5, cut_slice_d] = err
        print(f"Cut slice descending: {cut_slice_d}, MSE with res (mm^2 per 1/2 points): ", err)
    cut_slice_d_max = np.argmin(err_array_ind, axis=1)[4].astype('int')

    err_array_ind_idx = np.asarray([cut_row_a_max, cut_row_d_max, cut_column_a_max, cut_column_d_max,
                                    cut_slice_a_max, cut_slice_d_max])

    err_array_ind_f = f"{err_array_file_f}_ind"
    err_array_ind_idx_f = f"{err_array_file_f}_ind_idx"
    np.save(err_array_ind_f, err_array_ind)
    np.save(err_array_ind_idx_f, err_array_ind_idx)
    print("Saved: ", err_array_ind_f)
    print("Saved: ", err_array_ind_idx_f)


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

err_array_file = f"{save_dir}/err_corrode50x6"

# some configuration (make life easier?)
model_path = f"{model_dir}/bestVal_{model_label}"
model = keras.models.load_model(model_path)

corrode_asym_rcs(X_train, Y_train_one, res_train, model, err_array_file)
