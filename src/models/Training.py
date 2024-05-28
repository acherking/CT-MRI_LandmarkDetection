import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

# following modules are within this project
import models
from src.models.common import TrainingSupport, MyDataset


def train_model(args_dict):
    print("whole parameters: ", args_dict)
    save_dir = TrainingSupport.get_record_dir(args_dict)

    if args_dict.get("write_log", True):
        log = open(f"{save_dir}/original_log", "w")
        sys.stdout = log

    """ *** Training Process *** """

    batch_size = args_dict.get("batch_size", 2)
    epochs = args_dict.get("epochs", 100)
    min_val_mse_res = 100  # just a big number

    print(f"training process: batch_size[{batch_size}], epochs[{epochs}]")

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
        print("res: ", res_batch_train)
        break

    for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
        print("val_dataset, step: ", step)
        print("x shape: ", x_batch_val.shape, type(x_batch_val))
        print("y shape: ", y_batch_val.shape, type(y_batch_val))
        print("res: ", res_batch_val)
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
    mse_with_res = models.mse_with_res

    # Instantiate a metric object
    train_mse_res_metric = keras.metrics.Mean()
    val_mse_res_metric = keras.metrics.Mean()
    test_mse_res_metric = keras.metrics.Mean()

    # Get model.
    model_name = args_dict.get("model_name")
    input_shape = (rescaled_size[0], np.ceil(rescaled_size[1] / 2).astype(int), rescaled_size[2])
    model_output_num = args_dict.get("model_output_num")

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
            loss_mse = support_modules.train_step(model, mse_with_res, train_mse_res_metric, optimizer,
                                                  x_batch_train, y_batch_train, res_batch_train)

            # Logging every *** batches
            if step % 100 == 0:
                print("********Step ", step, " ********")
                print("Training loss (MSE with Res):    %.3f" % loss_mse.numpy())
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_mse_res = train_mse_res_metric.result()
        train_err_array[0][epoch] = float(train_mse_res)
        print("Training (MSE Res) over epoch:   %.4f" % (float(train_mse_res),))

        # Reset the metric's state at the end of an epoch
        train_mse_res_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
            support_modules.val_step(model, mse_with_res, val_mse_res_metric, x_batch_val, y_batch_val, res_batch_val)

        val_mse_res = val_mse_res_metric.result()
        train_err_array[1][epoch] = float(val_mse_res)
        val_mse_res_metric.reset_states()

        # Try to save the Trained Model with the best Val results
        if val_mse_res < min_val_mse_res:
            min_val_mse_res = val_mse_res
            # Use Test Dataset to evaluate the best Val model (at the moment), and save the Test results
            test_mse_res, y_test_pred = support_modules.my_evaluate(model, mse_with_res, test_mse_res_metric, test_dataset)
            np.save(f"{save_dir}/bestVal_{model_label}_y_test", y_test_pred)
            model.save(f"{save_dir}/bestVal_{model_label}")
            print("Validation (MSE with Res, saved):%.3f" % (float(val_mse_res),))
            print("Test (MSE with Res), bestVa      %.3f" % (float(test_mse_res),))
        else:
            print("Validation (MSE with Res):       %.3f" % (float(val_mse_res),))
        print("Time taken:                      %.2fs" % (time.time() - start_time))

        log.flush()

    # Use Test Dataset to evaluate the final model, and save the Test results
    test_mse_res, y_test_pred = support_modules.my_evaluate(model, mse_with_res, test_mse_res_metric, test_dataset)
    np.save(f"{save_dir}/final_{model_label}_y_test", y_test_pred)
    print("Test (MSE with Res), final       %.3f" % (float(test_mse_res),))

    model.save(f"{save_dir}/final_{model_label}")
    np.save(f"{save_dir}/train_val_err_array", train_err_array)

    log.close()


if __name__ == "__main__":

    args = {
        # prepare Dataset
        "dataset_tag":          "divided",
        "rescaled_size":        (176, 176, 96),
        "dataset_label_1":      "identical_voxel_distance",  # "none", "identical_voxel_distance", "variable_voxel_distance"
        "base_dir":             "/data/gpfs/projects/punim1836/Data",
        "data_split_tag":       "general",  # "general" - train 14, val 2, test 4; "cross_val"
        "data_split_static":    True,
        # training
        "write_log":            True,
        "batch_size":           2,
        "epochs":               100,
        # model
        "model_name":           "u_net_dsnt",
        "model_output_num":     2,
        # record
        "y_tag":                "two_landmarks",  # "one_landmark_[1/2]", "two_landmarks", "mean_two_landmarks"
        "save_dir_extend":      "centre_scale",  # can be used for cross validation
    }

    train_model(args)
