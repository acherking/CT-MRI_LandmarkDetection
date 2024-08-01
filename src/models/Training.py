import sys
import time
import math

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
    min_val_mean_dis = 100  # just a big number

    # Prepare dataset used in the training process
    train_num = train_dataset[0].shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
    train_dataset_eval = train_dataset.batch(batch_size)
    train_dataset_shuffle = train_dataset.shuffle(buffer_size=train_num * 2, reshuffle_each_iteration=True).batch(
        batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_dataset).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).batch(batch_size)

    # Review these datasets before the training
    print("*** review the dataset ***")
    for step, (x_batch_train, y_batch_train, res_batch_train) in enumerate(train_dataset_shuffle):
        print("train_dataset, step: ", step)
        print("x shape: ", x_batch_train.shape, type(x_batch_train))
        print("y value: ", y_batch_train)
        print("res: ", res_batch_train)
        break
    print("*** *** *** *** *** ***")

    optimizer = loss_optimizer.optimizer_manager(args_dict)
    loss = loss_optimizer.loss_manager(args_dict)
    train_eval_fn = loss_optimizer.mean_dis_mm
    eval_metrics_fn = loss_optimizer.eval_metric_manager(args_dict)

    # Instantiate a metric object
    train_metric = keras.metrics.Mean()

    model = models.model_manager(args_dict)
    model.summary()

    if args_dict.get("write_log", True):
        log.flush()

    @tf.function
    def train_step(x, y, res):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_val = loss(y, y_pred, res)
        gradients = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        mse_res = train_eval_fn(y, y_pred, res)
        train_metric.update_state(mse_res)
        return mse_res

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
        evms = eval_metrics_fn(y_true, y_pred, res, args_dict)

        return evms, y_true, y_pred, res

    # Training loop
    for epoch in range(epochs):
        train_eval = {}
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of a dataset.
        for step, (x_batch_train, y_batch_train, res_batch_train) in enumerate(train_dataset_shuffle):
            loss_mse = train_step(x_batch_train, y_batch_train, res_batch_train)

            # Logging every *** batches
            if step % 100 == 0:
                print("********Step ", step, " ********")
                print("Training loss (mean distance):    %.3f" % loss_mse.numpy())
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
                if math.isnan(loss_mse.numpy()): break

        end_time = time.time()
        # Display metrics at the end of each epoch.
        train_mean_dis_all = float(train_metric.result())
        train_eval["mean_dis_all"] = float("{:.3f}".format(train_mean_dis_all))
        print("Training over epoch:              %.4f" % (train_mean_dis_all,))

        # Reset the metric's state at the end of an epoch
        train_metric.reset_states()

        epoch_time = end_time - start_time
        print("Time taken:                       %.2fs" % epoch_time)

        train_eval["epoch"] = epoch
        train_eval["time(h)"] = float("{:.2f}".format(epoch_time * (epoch + 1) / 3600))

        # Run a validation loop at the end of each epoch.
        train_all_eval = my_evaluate(train_dataset_eval)
        print("Train: ", train_all_eval[0])
        val_eval = my_evaluate(val_dataset)
        print("Val : ", val_eval[0])

        # Show best Val results
        val_mean_disl = val_eval[0].get("mean_dis_all")
        if epoch == 0 or val_mean_disl < min_val_mean_dis:
            min_val_mean_dis = val_mean_disl
            # Use Test Dataset to evaluate the best Val model (at the moment), and save the Test results
            test_eval = my_evaluate(test_dataset)
            np.save(f"{save_dir}/best_val_Y_test_pred", test_eval[2])
            # if args_dict.get("save_model", True): model.save_weights(f"{save_dir}/best_val_model.weights.h5")
            model.save_weights(f"{save_dir}/best_val_model.weights.h5")
            print("Test: ", test_eval[0])
            # for reviewing ...
            best_val = [train_eval.copy(), val_eval[0].copy(), test_eval[0].copy()]

        if args_dict.get("write_log", True):
            log.flush()

    # Use Test Dataset to evaluate the final model, and save the Test results
    print("*** End Training ***")
    final_test_eval = my_evaluate(test_dataset)
    print("Test final: ", final_test_eval[0])

    np.save(f"{save_dir}/final_Y_test_pred", final_test_eval[2])
    np.save(f"{save_dir}/Y_test_true", final_test_eval[1])
    np.save(f"{save_dir}/res_test", final_test_eval[3])
    # if args_dict.get("save_model", True): model.save_weights(f"{save_dir}/final_model.weights.h5")
    model.save_weights(f"{save_dir}/final_model.weights.h5")

    if args_dict.get("data_split_tag") == "cross_val":
        # do not gather results if it is cross validation
        print("cross validation processing")
    else:
        # gather results into one file (for convenient)
        dataset_tag = args_dict.get("dataset_tag")
        model_name = args_dict.get("model_name")
        train_id = str(args_dict.get("train_id"))
        save_base_dir = args_dict.get("save_base_dir")
        gather_file = open(f"{save_base_dir}/{dataset_tag}/results_all", "a")
        time_tag = time.strftime("%d%b%Y%H%M")
        gather_file.write(f"*** {model_name} *** train_id[{train_id}] *** {time_tag}\n")
        gather_file.write(f"save in: {save_dir}\n")
        gather_file.write("*** best val *** \n")
        gather_file.write("Train: " + str(best_val[0]) + "\n")
        gather_file.write("Val:   " + str(best_val[1]) + "\n")
        gather_file.write("Test:  " + str(best_val[2]) + "\n")
        gather_file.write("*** final *** \n")
        gather_file.write("Test:  " + str(final_test_eval[0]) + "\n")
        gather_file.write("*** *** *** *** *** ***\n*** *** *** *** *** ***\n\n")
        gather_file.close()

    sys.stdout = orig_stdout
    if args_dict.get("write_log", True):
        log.close()

    if math.isnan(best_val[1]["mean_dis_all"]):
        return 500
    elif 'best_val' in locals() or 'best_val' in globals():
        return best_val[1]["mean_dis_all"]
    else:
        return 600



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
        "save_model": True,
    }

    train_model(args)
