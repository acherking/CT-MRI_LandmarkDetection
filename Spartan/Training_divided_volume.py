import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

import Functions.MyDataset as MyDataset
import support_modules
import models

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

rescaled_size = (176, 176, 48)
str_size = str(rescaled_size[0]) + "_" + str(rescaled_size[1]) + "_" + str(rescaled_size[2])
dataset_dir = f"/data/gpfs/projects/punim1836/Data/divided/" \
              f"{str(rescaled_size[0])}{str(rescaled_size[1])}{str(rescaled_size[2])}/"

X_train, Y_train, res_train, length_train, X_val, Y_val, res_val, length_val, X_test, Y_test, res_test, length_test = \
    support_modules.load_dataset_divide(dataset_dir, rescaled_size, pat_splits=MyDataset.get_pat_splits(static=True))

Y_train_one = np.asarray(Y_train)[:, 0, :].reshape((1400, 1, 3))
Y_val_one = np.asarray(Y_val)[:, 0, :].reshape((200, 1, 3))
Y_test_one = np.asarray(Y_test)[:, 0, :].reshape((400, 1, 3))

Y_train_mean = np.mean(Y_train, axis=1).reshape((1400, 1, 3))
Y_val_mean = np.mean(Y_val, axis=1).reshape((200, 1, 3))
Y_test_mean = np.mean(Y_test, axis=1).reshape((400, 1, 3))

""" *** Training Process *** """

batch_size = 2
epochs = 100
min_val_mse_res = 400

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_mean, res_train))
train_dataset = train_dataset.shuffle(buffer_size=2800, reshuffle_each_iteration=True).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val_mean, res_val))
val_dataset = val_dataset.shuffle(buffer_size=400, reshuffle_each_iteration=True).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test_mean, res_test)).batch(batch_size)

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
train_mse_metric = keras.metrics.Mean()
train_mse_res_metric = keras.metrics.Mean()
val_mse_res_metric = keras.metrics.Mean()
test_mse_res_metric = keras.metrics.Mean()

# Get model.
w = np.ceil(rescaled_size[1]/2).astype(int)
# model = models.first_model(height=rescaled_size[0], width=w, depth=rescaled_size[2], points_num=1)
model = models.straight_model(height=rescaled_size[0], width=w, depth=rescaled_size[2], points_num=1)
model.summary()

# y_tag: "one_landmark", "two_landmarks", "mean_two_landmarks"
y_tag = "mean_two_landmarks"
model_name = "straight_model"
model_tag = "divided"
model_size = f"{rescaled_size[0]}_{w}_{rescaled_size[2]}"
model_label = f"{model_name}_{model_tag}_{model_size}"
save_dir = f"/data/gpfs/projects/punim1836/Training/trained_models/{model_tag}_dataset/{model_name}/{y_tag}/"


@tf.function
def train_step(x, y, res):
    with tf.GradientTape() as tape:
        # Compute the loss value for this batch.
        y_pred = model(x, training=True)
        mse_res = mse_with_res(y, y_pred, res)

    # record MSE in pixel distance (without resolution)
    mse_pixel = mse(y, y_pred)

    # Update training metric.
    train_mse_metric.update_state(mse_pixel)
    train_mse_res_metric.update_state(mse_res)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(mse_res, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return mse_res, mse_pixel


@tf.function
def val_step(x, y, res):
    y_pred = model(x, training=False)
    mse_res = mse_with_res(y, y_pred, res)
    # Update val metrics
    val_mse_res_metric.update_state(mse_res)


@tf.function
def test_step(x, y, res):
    y_pred = model(x, training=False)
    mse_res = mse_with_res(y, y_pred, res)
    # Update val metrics
    test_mse_res_metric.update_state(mse_res)
    return y_pred


def my_evaluate(eva_model):
    # Run a test loop when meet the best val result.
    for s, (x_batch_test, y_batch_test, res_batch_test) in enumerate(test_dataset):
        if s == 0:
            y_test_p = test_step(x_batch_test, y_batch_test, res_batch_test)
        else:
            y_test = test_step(x_batch_test, y_batch_test, res_batch_test)
            y_test_p = np.concatenate((y_test_p, y_test), axis=0)

    test_mse_res_f = test_mse_res_metric.result()
    test_mse_res_metric.reset_states()
    return test_mse_res_f, y_test_p


# Training loop
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of a dataset.
    for step, (x_batch_train, y_batch_train, res_batch_train) in enumerate(train_dataset):
        loss_value, loss_mse = train_step(x_batch_train, y_batch_train, res_batch_train)

        # Logging every *** batches
        if step % 100 == 0:
            print("********Step ", step, " ********")
            print("Training loss (MSE):             %.3f" % loss_mse.numpy())
            print("Training loss (MSE with Res):    %.3f" % loss_value.numpy())
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_mse = train_mse_metric.result()
    train_mse_res = train_mse_res_metric.result()
    print("Training (MSE) over epoch:       %.4f" % (float(train_mse),))
    print("Training (MSE Res) over epoch:   %.4f" % (float(train_mse_res),))

    # Reset the metric's state at the end of an epoch
    train_mse_metric.reset_states()
    train_mse_res_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
        val_step(x_batch_val, y_batch_val, res_batch_val)

    val_mse_res = val_mse_res_metric.result()
    val_mse_res_metric.reset_states()

    # Try to save the Trained Model with the best Val results
    if val_mse_res < min_val_mse_res:
        min_val_mse_res = val_mse_res
        # Use Test Dataset to evaluate the best Val model (at the moment), and save the Test results
        test_mse_res, y_test_pred = my_evaluate(model)
        np.save(f"{save_dir}bestVal_{model_label}_y_test", y_test_pred)
        model.save(f"{save_dir}bestVal_{model_label}")
        print("Validation (MSE with Res, saved):%.3f" % (float(val_mse_res),))
        print("Test (MSE with Res), bestVa      %.3f" % (float(test_mse_res),))
    else:
        print("Validation (MSE with Res):       %.3f" % (float(val_mse_res),))
    print("Time taken:                      %.2fs" % (time.time() - start_time))

# Use Test Dataset to evaluate the final model, and save the Test results
test_mse_res, y_test_pred = my_evaluate(model)
np.save(f"{save_dir}final_{model_label}_y_test", y_test_pred)
print("Test (MSE with Res), final       %.3f" % (float(test_mse_res),))

model.save(f"{save_dir}final_{model_label}")
