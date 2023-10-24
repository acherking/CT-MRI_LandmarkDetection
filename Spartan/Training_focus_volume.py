import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import os
import sys

import Functions.MyDataset as MyDataset
import support_modules
import models


def train_model(data_splits, args_dict, write_log=True):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    save_dir = get_record_dir(args_dict)

    log = open(f"{save_dir}/original_log", "w")
    if write_log:
        sys.stdout = log

    dataset_tag = args_dict.get("dataset_tag")
    crop_size = args_dict.get("crop_dataset_size")
    cut = args_dict.get("cut_layers")
    has_trans = args_dict.get("has_trans")
    trans_tag = args_dict.get("trans_tag")
    base_dir = args_dict.get("base_dir")

    crop_layers = np.asarray([[cut[0], cut[1]], [cut[2], cut[3]], [cut[4], cut[5]]])
    crop_tag = f"x{crop_size[0]}{crop_size[1]}y{crop_size[2]}{crop_size[3]}z{crop_size[4]}{crop_size[5]}"
    # crop_tag_dir = crop_tag + has_trans

    # x_path = f"{base_dir}/{crop_tag_dir}/cropped_volumes_{crop_tag}_truth_{trans_tag}.npy"
    # y_path = f"{base_dir}/{crop_tag_dir}/cropped_points_{crop_tag}_truth_{trans_tag}.npy"
    x_path = f"{base_dir}/{crop_tag}/volumes_4k.npy"
    y_path = f"{base_dir}/{crop_tag}/points_4k.npy"

    print("Read Volumes from:   ", x_path)
    print("Read Points from:    ", y_path)

    x_train, y_train, x_val, y_val, x_test, y_test = \
        support_modules.load_dataset_crop_no_length(x_path, y_path, data_splits, crop_layers)

    train_num = x_train.shape[0]
    val_num = x_val.shape[0]
    test_num = x_test.shape[0]

    model_output_num = args_dict.get("model_output_num")
    print("Landmarks Num: ", model_output_num)

    model_y_tag = args_dict.get("y_tag")
    print("Y tag: ", model_y_tag)

    if model_output_num == 1:
        if model_y_tag == "one_landmark":
            y_train = np.asarray(y_train)[:, 0, :].reshape((train_num, 1, 3))
            y_val = np.asarray(y_val)[:, 0, :].reshape((val_num, 1, 3))
            y_test = np.asarray(y_test)[:, 0, :].reshape((test_num, 1, 3))
        elif model_y_tag == "mean_two_landmarks":
            y_train = np.mean(y_train, axis=1).reshape((train_num, 1, 3))
            y_val = np.mean(y_val, axis=1).reshape((val_num, 1, 3))
            y_test = np.mean(y_test, axis=1).reshape((test_num, 1, 3))

    res_train = (np.ones((train_num, 1, 3)) * 0.15).astype('float32')
    res_val = (np.ones((val_num, 1, 3)) * 0.15).astype('float32')
    res_test = (np.ones((test_num, 1, 3)) * 0.15).astype('float32')

    """ *** Training Process *** """

    batch_size = args_dict.get("batch_size", 2)
    epochs = args_dict.get("epochs", 100)
    min_val_mse = 100  # just a big number

    print(f"training process: batch_size[{batch_size}], epochs[{epochs}]")

    # Set
    # Prepare dataset used in the training process
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, res_train))
    train_dataset = train_dataset.shuffle(buffer_size=train_num*2, reshuffle_each_iteration=True).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val, res_val))
    val_dataset = val_dataset.shuffle(buffer_size=val_num*2, reshuffle_each_iteration=True).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test, res_test)).batch(batch_size)

    # Check these datasets
    for step, (x_batch_train, y_batch_train, res_batch_train) in enumerate(train_dataset):
        print("train_dataset, step: ", step)
        print("x shape: ", x_batch_train.shape, type(x_batch_train))
        print("y shape: ", y_batch_train.shape, type(y_batch_train))
        break

    for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
        print("val_dataset, step: ", step)
        print("x shape: ", x_batch_val.shape, type(x_batch_val))
        print("y shape: ", y_batch_val.shape, type(y_batch_val))
        break

    # optimizer
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # loss functions
    wing_loss = models.wing_loss
    mse = tf.keras.losses.MeanSquaredError()
    mse_res = models.mse_with_res

    # Instantiate a metric object
    train_mse_metric = keras.metrics.Mean()
    val_mse_metric = keras.metrics.Mean()
    test_mse_metric = keras.metrics.Mean()

    # Get model.
    model_name = args_dict.get("model_name")
    input_shape = (crop_size[0]+crop_size[1]-crop_layers[0, 0]-crop_layers[0, 1],
                   crop_size[2]+crop_size[3]-crop_layers[1, 0]-crop_layers[1, 1],
                   crop_size[4]+crop_size[5]-crop_layers[2, 0]-crop_layers[2, 1])

    model = models.get_model(model_name, input_shape, model_output_num)
    model.summary()

    model_size = f"{input_shape[0]}x{input_shape[1]}x{input_shape[2]}"
    model_label = f"{model_name}_{dataset_tag}_{model_size}"

    log.flush()

    train_err_array = np.zeros((2, epochs))  # 0: training err MSE over epoch, 1: val err MSE
    # Training loop
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of a dataset.
        for step, (x_batch_train, y_batch_train, res_batch_train) in enumerate(train_dataset):
            loss_mse = support_modules.train_step(model, mse_res, train_mse_metric, optimizer,
                                                  x_batch_train, y_batch_train, res_batch_train)

            # Logging every *** batches
            if step % 100 == 0:
                print("********Step ", step, " ********")
                print("Training loss (MSE with res):             %.3f" % loss_mse.numpy())
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_mse = train_mse_metric.result()
        train_err_array[0][epoch] = float(train_mse)
        print("Training (MSE with res) over epoch:       %.4f" % (float(train_mse),))

        # Reset the metric's state at the end of an epoch
        train_mse_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
            support_modules.val_step(model, mse_res, val_mse_metric, x_batch_val, y_batch_val, res_batch_val)

        val_mse = val_mse_metric.result()
        train_err_array[1][epoch] = float(val_mse)
        val_mse_metric.reset_states()

        # Try to save the Trained Model with the best Val results
        if val_mse < min_val_mse:
            min_val_mse = val_mse
            # Use Test Dataset to evaluate the best Val model (at the moment), and save the Test results
            test_mse, y_test_pred = support_modules.my_evaluate(model, mse_res, test_mse_metric, test_dataset)
            np.save(f"{save_dir}/bestVal_{model_label}_y_test", y_test_pred)
            model.save(f"{save_dir}/bestVal_{model_label}")
            print("Validation (MSE with res, saved):%.3f" % (float(val_mse),))
            print("Test (MSE with res), bestVa      %.3f" % (float(test_mse),))
        else:
            print("Validation (MSE with res):       %.3f" % (float(val_mse),))

        print("Time taken:             %.2fs" % (time.time() - start_time))

        log.flush()

    # Use Test Dataset to evaluate the final model, and save the Test results
    test_mse, y_test_pred = support_modules.my_evaluate(model, mse_res, test_mse_metric, test_dataset)
    np.save(f"{save_dir}/final_{model_label}_y_test", y_test_pred)
    print("Test (MSE with res), final       %.3f" % (float(test_mse),))

    model.save(f"{save_dir}/final_{model_label}")
    np.save(f"{save_dir}/train_val_err_array", train_err_array)

    log.close()


def get_record_dir(args_dict):
    cut = args_dict.get("cut_layers")
    crop_layers = np.asarray([[cut[0], cut[1]], [cut[2], cut[3]], [cut[4], cut[5]]])

    trans_tag = args_dict.get("trans_tag")
    dataset_tag = args_dict.get("dataset_tag")

    model_name = args_dict.get("model_name")

    crop_size = args_dict.get("crop_dataset_size")
    input_shape = (crop_size[0]+crop_size[1]-crop_layers[0, 0]-crop_layers[0, 1],
                   crop_size[2]+crop_size[3]-crop_layers[1, 0]-crop_layers[1, 1],
                   crop_size[4]+crop_size[5]-crop_layers[2, 0]-crop_layers[2, 1])
    model_size = f"{input_shape[0]}x{input_shape[1]}x{input_shape[2]}"
    # y_tag: "one_landmark", "two_landmarks", "mean_two_landmarks"
    y_tag = args_dict.get("y_tag")

    save_base = "/data/gpfs/projects/punim1836/Training/trained_models"
    save_dir = f"{save_base}/{dataset_tag}_dataset/{model_name}/{y_tag}/{model_size}/{trans_tag}"
    save_dir_extend = args_dict.get("save_dir_extend")
    save_dir = f"{save_dir}/{save_dir_extend}"

    # create the dir if not exist
    if os.path.exists(save_dir):
        print("Save model to: ", save_dir)
    else:
        os.makedirs(save_dir)
        print("Create dir and save model in it: ", save_dir)

    return save_dir


if __name__ == "__main__":

    args = {
        # prepare Dataset
        "dataset_tag": "cropped",
        "crop_dataset_size": [75, 75, 75, 75, 50, 50],
        # "cut_layers": [39, 39, 39, 39, 26, 26],
        "cut_layers": [25, 25, 25, 25, 0, 0],
        "has_trans": "",
        "trans_tag": "no_trans",
        "base_dir": "/data/gpfs/projects/punim1836/Data/cropped/based_on_truth/augment_exp_pythong",
        # training
        "batch_size": 2,
        "epochs": 100,
        # model
        "model_name": "straight_model",
        "model_output_num": 1,
        # record
        "y_tag": "one_landmark",  # "one_landmark", "two_landmarks", "mean_two_landmarks"
        "save_dir_extend": "",  # can be used for cross validation
    }

    d_splits = MyDataset.get_data_splits(MyDataset.get_pat_splits(static=True), split=True, aug_num=100)
    print("Using static dataset split: Train, Val, Test")

    train_model(d_splits, args, write_log=False)
