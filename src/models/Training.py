import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

# following modules are within this project
import models
import loss_optimizer
import TrainingSupport


def train_model(args_dict):
    save_dir = TrainingSupport.get_record_dir(args_dict)

    orig_stdout = sys.stdout
    if args_dict.get("write_log", True):
        log = open(f"{save_dir}/original_log", "w")
        sys.stdout.flush()
        sys.stdout = log

    # record the whole parameters
    print("*** whole parameters ***")
    print(args_dict)
    print("*** *** *** *** *** ***")

    # load dataset
    train_dataset, val_dataset, test_dataset = TrainingSupport.load_dataset_manager(args_dict)

    """ *** Training Process *** """

    batch_size = args_dict.get("batch_size", 2)
    epochs = args_dict.get("epochs", 100)
    min_val_mse_res = 100  # just a big number

    # Prepare dataset used in the training process
    train_num = train_dataset[0].shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset = train_dataset.shuffle(buffer_size=train_num * 2, reshuffle_each_iteration=True).batch(batch_size)

    val_num = val_dataset[0].shape[0]
    val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset)
    val_dataset = val_dataset.shuffle(buffer_size=val_num * 2, reshuffle_each_iteration=True).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).batch(batch_size)

    # Review these datasets before the training
    print("*** review the dataset ***")
    for step, (x_batch_train, y_batch_train, res_batch_train) in enumerate(train_dataset):
        print("train_dataset, step: ", step)
        print("x shape: ", x_batch_train.shape, type(x_batch_train))
        print("y value: ", y_batch_train)
        print("res: ", res_batch_train)
        break

    for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
        print("val_dataset, step: ", step)
        print("x shape: ", x_batch_val.shape, type(x_batch_val))
        print("y value: ", y_batch_val)
        print("res: ", res_batch_val)
        break
    print("*** *** *** *** *** ***")

    optimizer = loss_optimizer.optimizer_manager(args_dict)
    loss = loss_optimizer.loss_manager(args_dict)
    mse_res_fn = loss_optimizer.mse_with_res

    # Instantiate a metric object
    train_metric = keras.metrics.Mean()
    val_metric = keras.metrics.Mean()
    test_metric = keras.metrics.Mean()

    model = models.model_manager(args_dict)
    model.summary()

    if args_dict.get("write_log", True):
        log.flush()

    # record the training process, for visualizing purpose
    # 0: training err MSE over epoch, 1: val err MSE
    train_err_array = np.zeros((2, epochs))

    @tf.function
    def train_step(x, y, res):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_val = loss(y, y_pred, res)
        gradients = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        mse_res = mse_res_fn(y, y_pred, res)
        train_metric.update_state(mse_res)
        return mse_res

    @tf.function
    def val_step(x, y, res):
        y_pred = model(x, training=False)
        mse_res = mse_res_fn(y, y_pred, res)
        val_metric.update_state(mse_res)
        return y_pred

    @tf.function
    def test_step(x, y, res):
        y_pred = model(x, training=False)
        mse_res = mse_res_fn(y, y_pred, res)
        test_metric.update_state(mse_res)
        return y_pred

    def my_evaluate(eval_dataset):
        # Run a test loop when meet the best val result.
        for s, (x_batch_test, y_batch_test, res_batch_test) in enumerate(eval_dataset):
            if s == 0:
                y_test_p = test_step(x_batch_test, y_batch_test, res_batch_test)
            else:
                y_test = test_step(x_batch_test, y_batch_test, res_batch_test)
                y_test_p = np.concatenate((y_test_p, y_test), axis=0)

        eval_metric_result = test_metric.result()
        test_metric.reset_states()
        return eval_metric_result, y_test_p

    # Training loop
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of a dataset.
        for step, (x_batch_train, y_batch_train, res_batch_train) in enumerate(train_dataset):
            loss_mse = train_step(x_batch_train, y_batch_train, res_batch_train)

            # Logging every *** batches
            if step % 100 == 0:
                print("********Step ", step, " ********")
                print("Training loss (MSE with Res):    %.3f" % loss_mse.numpy())
                print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_mse_res = train_metric.result()
        train_err_array[0][epoch] = float(train_mse_res)
        print("Training (MSE Res) over epoch:   %.4f" % (float(train_mse_res),))

        # Reset the metric's state at the end of an epoch
        train_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
            val_step(x_batch_val, y_batch_val, res_batch_val)

        val_mse_res = val_metric.result()
        train_err_array[1][epoch] = float(val_mse_res)
        val_metric.reset_states()

        # Show best Val results
        if val_mse_res < min_val_mse_res:
            min_val_mse_res = val_mse_res
            # Use Test Dataset to evaluate the best Val model (at the moment), and save the Test results
            test_mse_res, y_test_pred = my_evaluate(test_dataset)
            np.save(f"{save_dir}/best_val_Y_test_pred", y_test_pred)
            print("Validation (MSE with Res, saved):%.3f" % (float(val_mse_res),))
            print("Test (MSE with Res), bestVa      %.3f" % (float(test_mse_res),))
        else:
            print("Validation (MSE with Res):       %.3f" % (float(val_mse_res),))
        print("Time taken:                      %.2fs" % (time.time() - start_time))

        if args_dict.get("write_log", True):
            log.flush()

    # Use Test Dataset to evaluate the final model, and save the Test results
    test_mse_res, y_test_pred = my_evaluate(test_dataset)
    np.save(f"{save_dir}/final_Y_test", y_test_pred)
    print("Test (MSE with Res), final       %.3f" % (float(test_mse_res),))

    np.save(f"{save_dir}/train_val_err_array", train_err_array)

    sys.stdout = orig_stdout
    if args_dict.get("write_log", True):
        log.close()


if __name__ == "__main__":
    args = {
        # prepare Dataset
        "dataset_tag": "divided",  # "divided", "cropped"
        ## for divided dataset ##
        "input_shape": (176, 88, 48),
        "cut_layers": [[25, 25], [25, 25], [0, 0]],
        "base_dir": "/data/gpfs/projects/punim1836/Data",
        "dataset_label_1": "identical_voxel_distance",
        "data_split_tag": "general",  # "general" - train 14, val 2, test 4; "cross_val"
        "data_split_static": True,
        # training
        "write_log": True,
        "batch_size": 2,
        "epochs": 100,
        "loss_name": "MSE_res",
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "decay_steps": 10000,
        "decay_rate": 0.96,
        # model
        "model_name": "u_net_dsnt",
        "model_output_num": 2,
        # record
        "save_base_dir": "/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/models",
        "y_tag": "two_landmarks",  # "one_landmark_[1/2]", "two_landmarks", "mean_two_landmarks"
        "model_label_1": "",  # Cross validation, different parameter...
        "model_label_2": "",
    }

    train_model(args)
