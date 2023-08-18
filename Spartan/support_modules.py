import numpy as np
import tensorflow as tf

import Functions.MyDataset as MyDataset
import Functions.MyCrop as MyCrop


# load dataset from a single directory, each file contains both volume and pts (X & Y).
def load_dataset():

    return


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

    cropped_volumes = np.asarray(cropped_volumes).reshape((2000, 200, 200, 160, 1))
    cropped_points = np.asarray(cropped_points).reshape((2000, 2, 3))
    cropped_length = np.asarray(cropped_length).reshape((2000, 2, 3))

    # read centre shift
    # centre_shift = np.load("res/noises_s1_pred_test_dis.npy")
    centre_shift = np.zeros((2000, 1, 3))

    cropped_volumes, cropped_points, cropped_length = \
        MyCrop.crop_outside_layers_trans(cropped_volumes, cropped_points, cropped_length, centre_shift)

    crop_size = "x7575y7575z5050"
    has_trans = ""  # or ""
    trans_tag = "no_trans"
    comb_tag = "truth"
    save_comb_dir = f"/data/gpfs/projects/punim1836/Data/cropped/based_on_truth/{crop_size}{has_trans}"
    save_volume_path = f"{save_comb_dir}/cropped_volumes_{crop_size}_{comb_tag}_{trans_tag}.npy"
    save_points_path = f"{save_comb_dir}/cropped_points_{crop_size}_{comb_tag}_{trans_tag}.npy"
    save_length_path = f"{save_comb_dir}/cropped_length_{crop_size}_{comb_tag}_{trans_tag}.npy"
    np.save(save_volume_path, cropped_volumes)
    print("saved: ", save_volume_path)
    np.save(save_points_path, cropped_points)
    print("saved: ", save_points_path)
    np.save(save_length_path, cropped_length)
    print("saved: ", save_length_path)

    return 1


def load_dataset_divide(dataset_dir, rescaled_size, idx_splits, no_split=False):
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

    # without splitting to Train, Val and Test
    if no_split:
        return x_dataset, y_dataset, res_dataset_rep, length_dataset

    train_idx = idx_splits[0]
    x_train = x_dataset[train_idx]
    y_train = y_dataset[train_idx]
    res_train = res_dataset_rep[train_idx]
    length_train = length_dataset[train_idx]

    val_idx = idx_splits[1]
    x_val = x_dataset[val_idx]
    y_val = y_dataset[val_idx]
    res_val = res_dataset_rep[val_idx]
    length_val = length_dataset[val_idx]

    test_idx = idx_splits[2]
    x_test = x_dataset[test_idx]
    y_test = y_dataset[test_idx]
    res_test = res_dataset_rep[test_idx]
    length_test = length_dataset[test_idx]

    return x_train, y_train, res_train, length_train, \
        x_val, y_val, res_val, length_val, \
        x_test, y_test, res_test, length_test


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
