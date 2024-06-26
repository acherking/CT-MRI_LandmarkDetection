import sys
import time
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow import keras

# following modules are within this project
import models
import loss_optimizer
import TrainingSupport
import start_training


base_args = start_training.base_args

train_id = 5
args_updates = start_training.train_down_net_model()[train_id]
base_args.update(args_updates)

date_tag = "07Jun20241451"

model_save_dir = str(pathlib.Path(TrainingSupport.get_record_dir(base_args)).parent)
weights_path = f"{model_save_dir}/{date_tag}/best_val_model.weights.h5"

model = models.model_manager(base_args)
model.load_weights(weights_path)

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


evaluation_metrics, _, _, _ = my_evaluate(test_dataset)
print(evaluation_metrics)

