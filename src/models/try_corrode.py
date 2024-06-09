import tensorflow as tf
from tensorflow import keras
import numpy as np

import common.MyDataset as MyDataset
import TrainingSupport
import models
import start_training
import pathlib
import loss_optimizer

# mse = tf.keras.losses.MeanSquaredError()
# mse_res = models.mse_with_res
# test_mse_metric = keras.metrics.Mean()

base_args = start_training.base_args

train_id = 5
args_updates = start_training.train_straight_model()[train_id]
base_args.update(args_updates)

date_tag = "07Jun20241451"

model_save_dir = str(pathlib.Path(TrainingSupport.get_record_dir(base_args)).parent)
weights_path = f"{model_save_dir}/{date_tag}/best_val_model.weights.h5"

model = models.model_manager(base_args)
model.load_weights(weights_path)
model_output_num = base_args.get("model_output_num")

train_dataset, val_dataset, test_dataset = TrainingSupport.load_dataset_manager(base_args)

batch_size = base_args.get("batch_size", 2)
train_num = train_dataset[0].shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
train_dataset = train_dataset.shuffle(buffer_size=train_num * 2, reshuffle_each_iteration=True).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).batch(batch_size)

eval_metrics_fn = loss_optimizer.eval_metric_manager(base_args)


@tf.function
def test_step(x):
    y_pred = model(x, training=False)
    return y_pred


def my_evaluate(eval_dataset):
    # evaluate the val or test dataset
    for s, (x_batch_test, y_batch_test, res_batch_test) in enumerate(eval_dataset):
        if s == 0:
            (y_pred, y_true, res) = (test_step(x_batch_test), y_batch_test, res_batch_test)
        else:
            y_pred = np.concatenate((y_pred, test_step(x_batch_test)), axis=0)
            y_true = np.concatenate((y_true, y_batch_test), axis=0)
            res = np.concatenate((res, res_batch_test), axis=0)
    # prepare the evaluation metrics
    evms = eval_metrics_fn(y_true, y_pred, res, base_args)

    return evms, y_true, y_pred, res


def corrode_sym_rcs(x_dataset, y_dataset, res_dataset, model_f, err_array_file_f):
    if model_output_num == 1:
        err_array = np.ones((15, 15, 15, 2)) * -1
    elif model_output_num == 2:
        err_array = np.ones((15, 15, 15, 6)) * -1
    else:
        print("wrong model_output_num:", model_output_num); exit(0)
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
                eval_metrics, _, _, _ = my_evaluate(dataset)
                err_array[cut_row_num, cut_column_num, cut_slice_num, 0] = eval_metrics.get("mean_dis_all")
                err_array[cut_row_num, cut_column_num, cut_slice_num, 1] = eval_metrics.get("std_dev_all")
                if model_output_num == 2:
                    err_array[cut_row_num, cut_column_num, cut_slice_num, 2] = eval_metrics.get("mean_dis_1")
                    err_array[cut_row_num, cut_column_num, cut_slice_num, 3] = eval_metrics.get("std_dev_1")
                    err_array[cut_row_num, cut_column_num, cut_slice_num, 4] = eval_metrics.get("mean_dis_1")
                    err_array[cut_row_num, cut_column_num, cut_slice_num, 5] = eval_metrics.get("std_dev_1")
                print(f"({cut_row_num}:{cut_column_num}:{cut_slice_num}): ", str(eval_metrics))

    np.save(err_array_file_f, err_array)
    print("Saved: ", err_array_file_f)


def corrode_asym_rcs(x_dataset, y_dataset, res_dataset, volume_shape, cut_nums, model_f, err_array_file_f):
    """
    volume_shape: (row, column, slice)
    cut_nums: (row_ascending, row_descending, column_a, column_d, slice_a, slice_d)
    """
    if model_output_num == 1:
        err_array = np.ones((6, np.max(cut_nums)+1, 2)) * -1
    elif model_output_num == 2:
        err_array = np.ones((6, np.max(cut_nums)+1, 6)) * -1
    else:
        print("wrong model_output_num:", model_output_num); exit(0)
    fill_val = np.min(x_dataset)
    # cut layers, Shape e.g. (n, 100, 100, 100, 1)
    for surf_id in range(0, 6):
        x_dataset_corroded = np.copy(x_dataset)
        for cut_num in range(0, cut_nums[surf_id]+1):
            if surf_id == 0:
                corrode_info = "cut row num from ascending order"
                if cut_num > 0:
                    x_dataset_corroded[:, cut_num-1, :, :, :] = fill_val
            elif surf_id == 1:
                corrode_info = "cut row num from descending order"
                if cut_num > 0:
                    x_dataset_corroded[:, -cut_num, :, :, :] = fill_val
            elif surf_id == 2:
                corrode_info = "cut column num from ascending order"
                if cut_num > 0:
                    x_dataset_corroded[:, :, cut_num-1, :, :] = fill_val
            elif surf_id == 3:
                corrode_info = "cut column num from descending order"
                if cut_num > 0:
                    x_dataset_corroded[:, :, -cut_num, :, :] = fill_val
            elif surf_id == 4:
                corrode_info = "cut slice num from ascending order"
                if cut_num > 0:
                    x_dataset_corroded[:, :, :, cut_num-1, :] = fill_val
            elif surf_id == 5:
                corrode_info = "cut slice num from descending order"
                if cut_num > 0:
                    x_dataset_corroded[:, :, :, -cut_num, :] = fill_val

            dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)

            eval_metrics, _, _, _ = my_evaluate(dataset)
            del dataset

            err_array[surf_id, cut_num, 0] = eval_metrics.get("mean_dis_all")
            err_array[surf_id, cut_num, 1] = eval_metrics.get("std_dev_all")
            if model_output_num == 2:
                err_array[surf_id, cut_num, 2] = eval_metrics.get("mean_dis_1")
                err_array[surf_id, cut_num, 3] = eval_metrics.get("std_dev_1")
                err_array[surf_id, cut_num, 4] = eval_metrics.get("mean_dis_1")
                err_array[surf_id, cut_num, 5] = eval_metrics.get("std_dev_1")
            print(f"({corrode_info}: {cut_num}): ", str(eval_metrics))

        del x_dataset_corroded
        np.save(err_array_file_f, err_array)
        print("Partially Saved: ", err_array_file_f)

    np.save(err_array_file_f, err_array)
    print("Final Saved: ", err_array_file_f)


def corrode_asym_rcs_independent(x_dataset, y_dataset, res_dataset, volume_shape, cut_nums, model_f, err_array_file_f):
    if model_output_num == 1:
        err_array_ind = np.ones((6, 50, 2)) * -1
    elif model_output_num == 2:
        err_array_ind = np.ones((6, 50, 6)) * -1
    else:
        print("wrong model_output_num:", model_output_num); exit(0)
    fill_val = np.min(x_dataset)
    # cut row ascending
    for cut_row_a in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a:, :, :, :] = x_dataset[:, cut_row_a:, :, :, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        eval_metrics, _, _, _ = my_evaluate(dataset)
        err_array_ind[0, cut_row_a, 0] = eval_metrics.get("mean_dis_all")
        err_array_ind[0, cut_row_a, 1] = eval_metrics.get("std_dev_all")
        if model_output_num == 2:
            err_array_ind[0, cut_row_a, 2] = eval_metrics.get("mean_dis_1")
            err_array_ind[0, cut_row_a, 3] = eval_metrics.get("std_dev_1")
            err_array_ind[0, cut_row_a, 4] = eval_metrics.get("mean_dis_1")
            err_array_ind[0, cut_row_a, 5] = eval_metrics.get("std_dev_1")
        print(f"Cut row ascending: {cut_row_a}: ", str(eval_metrics))
    cut_row_a_max = np.argmin(err_array_ind[:, :, 0], axis=1)[0].astype('int')
    # cut row descending
    for cut_row_d in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d), :, :, :] = x_dataset[:, cut_row_a_max:(100-cut_row_d), :, :, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        eval_metrics, _, _, _ = my_evaluate(dataset)
        err_array_ind[1, cut_row_d, 0] = eval_metrics.get("mean_dis_all")
        err_array_ind[1, cut_row_d, 1] = eval_metrics.get("std_dev_all")
        if model_output_num == 2:
            err_array_ind[1, cut_row_d, 2] = eval_metrics.get("mean_dis_1")
            err_array_ind[1, cut_row_d, 3] = eval_metrics.get("std_dev_1")
            err_array_ind[1, cut_row_d, 4] = eval_metrics.get("mean_dis_1")
            err_array_ind[1, cut_row_d, 5] = eval_metrics.get("std_dev_1")
        print(f"Cut row descending: {cut_row_d}: ", str(eval_metrics))
    cut_row_d_max = np.argmin(err_array_ind[:, :, 0], axis=1)[0].astype('int')
    # cut column ascending
    for cut_column_a in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a:, :, :] = \
            x_dataset[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a:, :, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        eval_metrics, _, _, _ = my_evaluate(dataset)
        err_array_ind[2, cut_column_a, 0] = eval_metrics.get("mean_dis_all")
        err_array_ind[2, cut_column_a, 1] = eval_metrics.get("std_dev_all")
        if model_output_num == 2:
            err_array_ind[2, cut_column_a, 2] = eval_metrics.get("mean_dis_1")
            err_array_ind[2, cut_column_a, 3] = eval_metrics.get("std_dev_1")
            err_array_ind[2, cut_column_a, 4] = eval_metrics.get("mean_dis_1")
            err_array_ind[2, cut_column_a, 5] = eval_metrics.get("std_dev_1")
        print(f"Cut column ascending: {cut_column_a}: ", str(eval_metrics))
    cut_column_a_max = np.argmin(err_array_ind[:, :, 0], axis=1)[0].astype('int')
    # cut column descending
    for cut_column_d in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d), :, :] = \
            x_dataset[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d), :, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        eval_metrics, _, _, _ = my_evaluate(dataset)
        err_array_ind[3, cut_column_d, 0] = eval_metrics.get("mean_dis_all")
        err_array_ind[3, cut_column_d, 1] = eval_metrics.get("std_dev_all")
        if model_output_num == 2:
            err_array_ind[3, cut_column_d, 2] = eval_metrics.get("mean_dis_1")
            err_array_ind[3, cut_column_d, 3] = eval_metrics.get("std_dev_1")
            err_array_ind[3, cut_column_d, 4] = eval_metrics.get("mean_dis_1")
            err_array_ind[3, cut_column_d, 5] = eval_metrics.get("std_dev_1")
        print(f"Cut column descending: {cut_column_d}: ", str(eval_metrics))
    cut_column_d_max = np.argmin(err_array_ind[:, :, 0], axis=1)[0].astype('int')
    # cut slice ascending
    for cut_slice_a in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d_max), cut_slice_a:, :] = \
            x_dataset[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d_max), cut_slice_a:, :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        eval_metrics, _, _, _ = my_evaluate(dataset)
        err_array_ind[4, cut_slice_a, 0] = eval_metrics.get("mean_dis_all")
        err_array_ind[4, cut_slice_a, 1] = eval_metrics.get("std_dev_all")
        if model_output_num == 2:
            err_array_ind[4, cut_slice_a, 2] = eval_metrics.get("mean_dis_1")
            err_array_ind[4, cut_slice_a, 3] = eval_metrics.get("std_dev_1")
            err_array_ind[4, cut_slice_a, 4] = eval_metrics.get("mean_dis_1")
            err_array_ind[4, cut_slice_a, 5] = eval_metrics.get("std_dev_1")
        print(f"Cut slice ascending: {cut_slice_a}: ", str(eval_metrics))
    cut_slice_a_max = np.argmin(err_array_ind[:, :, 0], axis=1)[0].astype('int')
    # cut slice ascending
    for cut_slice_d in range(0, 50):
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d_max), cut_slice_a_max:(100-cut_slice_d), :] = \
            x_dataset[:, cut_row_a_max:(100-cut_row_d_max), cut_column_a_max:(1-cut_column_d_max), cut_slice_a_max:(100-cut_slice_d), :]
        dataset = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
        eval_metrics, _, _, _ = my_evaluate(dataset)
        err_array_ind[5, cut_slice_d, 0] = eval_metrics.get("mean_dis_all")
        err_array_ind[5, cut_slice_d, 1] = eval_metrics.get("std_dev_all")
        if model_output_num == 2:
            err_array_ind[5, cut_slice_d, 2] = eval_metrics.get("mean_dis_1")
            err_array_ind[5, cut_slice_d, 3] = eval_metrics.get("std_dev_1")
            err_array_ind[5, cut_slice_d, 4] = eval_metrics.get("mean_dis_1")
            err_array_ind[5, cut_slice_d, 5] = eval_metrics.get("std_dev_1")
        print(f"Cut slice descending: {cut_slice_d}: ", str(eval_metrics))
    cut_slice_d_max = np.argmin(err_array_ind[:, :, 0], axis=1)[0].astype('int')

    err_array_ind_idx = np.asarray([cut_row_a_max, cut_row_d_max, cut_column_a_max, cut_column_d_max,
                                    cut_slice_a_max, cut_slice_d_max])

    err_array_ind_f = f"{err_array_file_f}_ind"
    err_array_ind_idx_f = f"{err_array_file_f}_ind_idx"
    np.save(err_array_ind_f, err_array_ind)
    np.save(err_array_ind_idx_f, err_array_ind_idx)
    print("Saved: ", err_array_ind_f)
    print("Saved: ", err_array_ind_idx_f)

###########################################################################
# end def corrode_asym_rcs(x_dataset, y_dataset, res_dataset, model_f, err_array_file_f):
###########################################################################


# crop_layers: (3, 2) --> row_ascending, row_descending, column_a, column_d, slice_a, slice_d
def corrode_baseline(x_dataset, y_dataset, res_dataset, model_f, crop_layers):
    """
    Check other area's contribution
    1. only target area (after corroding)
    2. only the corroded area (exclude the target area)
    3. remove all the value in the volume
    """
    dataset_org = tf.data.Dataset.from_tensor_slices((x_dataset, y_dataset, res_dataset)).batch(2)
    err_org, pred_org = my_evaluate(model_f, dataset_org)
    print("original Volume, MSE with res (mm^2 per 1/2 points): ", err_org)
    print("pred list: ", pred_org[0:5])

    fill_val = np.min(x_dataset)

    # x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
    # x_dataset_corroded[:,
    #     crop_layers[0][0]:(100 - crop_layers[0][1]),
    #     crop_layers[1][0]:(100 - crop_layers[1][1]),
    #     crop_layers[2][0]:(100 - crop_layers[2][1]), :] = \
    #     x_dataset[:,
    #     crop_layers[0][0]:(100 - crop_layers[0][1]),
    #     crop_layers[1][0]:(100 - crop_layers[1][1]),
    #     crop_layers[2][0]:(100 - crop_layers[2][1]), :]
    #
    # dataset_corroded = tf.data.Dataset.from_tensor_slices((x_dataset_corroded, y_dataset, res_dataset)).batch(2)
    # err_corroded, pred_corroded = my_evaluate(model_f, dataset_corroded)
    # print("Only target area, MSE with res (mm^2 per 1/2 points): ", err_corroded)
    # print("pred list: ", pred_corroded[0:5])
    #
    # x_dataset_surrounding = np.copy(x_dataset)
    # x_dataset_surrounding[:,
    #     crop_layers[0][0]:(100 - crop_layers[0][1]),
    #     crop_layers[1][0]:(100 - crop_layers[1][1]),
    #     crop_layers[2][0]:(100 - crop_layers[2][1]), :] = fill_val
    #
    # # double-check the corroded data
    # print("fill val: ", fill_val)
    # print("row&slice, centre like:", x_dataset_surrounding[0, 49, :, 49, 0])
    # print("column&slice, centre like:", x_dataset_surrounding[0, :, 49, 49, 0])
    # print("row&column, centre like:", x_dataset_surrounding[0, 49, 49, :, 0])
    #
    # dataset_sur = tf.data.Dataset.from_tensor_slices((x_dataset_surrounding, y_dataset, res_dataset)).batch(2)
    # err_sur, pred_sur = my_evaluate(model_f, dataset_sur)
    # print("Only surrounding area, MSE with res (mm^2 per 1/2 points): ", err_sur)
    # print("pred list: ", pred_sur[0:5])
    #
    # x_dataset_empty = np.ones(x_dataset.shape) * fill_val
    #
    # dataset_emp = tf.data.Dataset.from_tensor_slices((x_dataset_empty, y_dataset, res_dataset)).batch(2)
    # err_emp, pred_emp = my_evaluate(model_f, dataset_emp)
    # print("Blank Volume, MSE with res (mm^2 per 1/2 points): ", err_emp)
    # print("pred list: ", pred_emp[0:5])


###
# Start main process
###
crop_layers = np.asarray([[25, 25], [25, 25], [0, 0]])
crop_size = (150, 150, 100)

# crop_tag = "x5050y5050z5050"
crop_tag = "x100100y100100z5050"
base_dir = "/data/gpfs/projects/punim1836/Data/cropped/based_on_truth"

X_path = f"{base_dir}/{crop_tag}/cropped_volumes_{crop_tag}_truth.npy"
Y_path = f"{base_dir}/{crop_tag}/cropped_points_{crop_tag}_truth.npy"
Cropped_length_path = f"{base_dir}/{crop_tag}/cropped_length_{crop_tag}_truth.npy"

data_splits = MyDataset.get_data_splits(MyDataset.get_pat_splits(static=True), split=True)

X_train, Y_train, length_train, X_val, Y_val, length_val, X_test, Y_test, length_test = \
    supporter.load_dataset_crop(X_path, Y_path, Cropped_length_path, data_splits, crop_layers)

Y_train_one = np.asarray(Y_train)[:, 0, :].reshape((1400, 1, 3))
Y_val_one = np.asarray(Y_val)[:, 0, :].reshape((200, 1, 3))
Y_test_one = np.asarray(Y_test)[:, 0, :].reshape((400, 1, 3))

res_train = (np.ones((1400, 1, 3)) * 0.15).astype('float32')
res_val = (np.ones((200, 1, 3)) * 0.15).astype('float32')
res_test = (np.ones((400, 1, 3)) * 0.15).astype('float32')

# y_tag: "one_landmark" -> OL, "two_landmarks" -> TL, "mean_two_landmarks" -> MTL
y_tag = "one_landmark_res"
model_name = "straight_model"
model_tag = "cropped"
model_size = f"{crop_size[0]}x{crop_size[1]}x{crop_size[2]}"
model_label = f"{model_name}_{model_tag}_{model_size}"
base_dir = "/data/gpfs/projects/punim1836/Training/trained_models"
model_dir = f"{base_dir}/{model_tag}_dataset/{model_name}/{y_tag}/{model_size}"
save_dir = f"{model_dir}/corroding_test"

err_array_file = f"{save_dir}/err_corrode100x4and50x2_bestVal"

# some configuration (make life easier?)
model_path = f"{model_dir}/bestVal_{model_label}"
model = keras.models.load_model(model_path)

corrode_layers = np.asarray([120, 120, 120, 120, 80, 80])
corrode_asym_rcs(X_train, Y_train_one, res_train, crop_size, corrode_layers, model, err_array_file)

# crop_layers = np.asarray([[20, 10], [0, 20], [25, 18]])
# print("Train Dataset")
# corrode_baseline(X_train, Y_train_one, res_train, model, crop_layers)
# print("Test Dataset")
# corrode_baseline(X_test, Y_test_one, res_test, model, crop_layers)

