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

crop_size = (100, 100, 100)

X_path = "/data/gpfs/projects/punim1836/Data/cropped/cropped_volumes_x5050y5050z5050.npy"
Y_path = "/data/gpfs/projects/punim1836/Data/cropped/cropped_points_x5050y5050z5050.npy"
Cropped_length_path = "/data/gpfs/projects/punim1836/Data/cropped/cropped_length_x5050y5050z5050.npy"
pat_splits = MyDataset.get_pat_splits(static=True)
X_train, Y_train, length_train, X_val, Y_val, length_val, X_test, Y_test, length_test = \
    support_modules.load_dataset_crop(X_path, Y_path, Cropped_length_path, pat_splits)

Y_train_one = np.asarray(Y_train)[:, 0, :].reshape((1400, 1, 3))
Y_val_one = np.asarray(Y_val)[:, 0, :].reshape((200, 1, 3))
Y_test_one = np.asarray(Y_test)[:, 0, :].reshape((400, 1, 3))

res_train = (np.ones((1400, 1, 3)) * 0.15).astype('float32')
res_val = (np.ones((200, 1, 3)) * 0.15).astype('float32')
res_test = (np.ones((400, 1, 3)) * 0.15).astype('float32')

""" *** Training Process *** """

batch_size = 2
epochs = 100
min_val_mse = 400

# Set
# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_one, res_train))
train_dataset = train_dataset.shuffle(buffer_size=2800, reshuffle_each_iteration=True).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val_one, res_val))
val_dataset = val_dataset.shuffle(buffer_size=400, reshuffle_each_iteration=True).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test_one, res_test)).batch(batch_size)

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

# Set
# Get model.
# model = models.first_model(width=size[0], height=size[1], depth=size[2])
model = models.straight_model(height=crop_size[0], width=crop_size[1], depth=crop_size[2], points_num=1)
model.summary()

# y_tag: "one_landmark", "two_landmarks", "mean_two_landmarks"
y_tag = "one_landmark_res"
model_name = "straight_model"
model_tag = "cropped"
model_size = f"{crop_size[0]}_{crop_size[1]}_{crop_size[2]}"
model_label = f"{model_name}_{model_tag}_{model_size}"
save_dir = f"/data/gpfs/projects/punim1836/Training/trained_models/{model_tag}_dataset/{model_name}/{y_tag}/"


@tf.function
def train_step(x, y, res):
    with tf.GradientTape() as tape:
        # Compute the loss value for this batch.
        y_pred = model(x, training=True)
        err = mse_res(y, y_pred, res)

    # Update training metric.
    train_mse_metric.update_state(err)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(err, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return err


@tf.function
def val_step(x, y, res):
    y_pred = model(x, training=False)
    err = mse_res(y, y_pred, res)
    # Update val metrics
    val_mse_metric.update_state(err)


@tf.function
def test_step(x, y, res):
    y_pred = model(x, training=False)
    err = mse_res(y, y_pred, res)
    # Update test metrics
    test_mse_metric.update_state(err)
    return y_pred


def my_evaluate(eva_model):
    # Run a test loop when meet the best val result.
    for s, (x_batch_test, y_batch_test, res_batch_test) in enumerate(test_dataset):
        if s == 0:
            y_test_p = test_step(x_batch_test, y_batch_test, res_batch_test)
        else:
            y_test = test_step(x_batch_test, y_batch_test, res_batch_test)
            y_test_p = np.concatenate((y_test_p, y_test), axis=0)

    test_mse_res_f = test_mse_metric.result()
    test_mse_metric.reset_states()
    return test_mse_res_f, y_test_p


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
            print("Training loss (MSE with res):             %.3f" % loss_mse.numpy())
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_mse = train_mse_metric.result()
    print("Training (MSE with res) over epoch:       %.4f" % (float(train_mse),))

    # Reset the metric's state at the end of an epoch
    train_mse_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
        val_step(x_batch_val, y_batch_val, res_batch_val)

    val_mse = val_mse_metric.result()
    val_mse_metric.reset_states()

    # Try to save the Trained Model with the best Val results
    if val_mse < min_val_mse:
        min_val_mse = val_mse
        # Use Test Dataset to evaluate the best Val model (at the moment), and save the Test results
        test_mse, y_test_pred = my_evaluate(model)
        np.save(f"{save_dir}bestVal_{model_label}_y_test", y_test_pred)
        model.save(f"{save_dir}bestVal_{model_label}")
        print("Validation (MSE with res, saved):%.3f" % (float(val_mse),))
        print("Test (MSE with res), bestVa      %.3f" % (float(test_mse),))
    else:
        print("Validation (MSE with res):       %.3f" % (float(val_mse),))

    print("Time taken:             %.2fs" % (time.time() - start_time))

# Use Test Dataset to evaluate the final model, and save the Test results
test_mse, y_test_pred = my_evaluate(model)
np.save(f"{save_dir}final_{model_label}_y_test", y_test_pred)
print("Test (MSE with res), final       %.3f" % (float(test_mse),))

model.save(f"{save_dir}final_{model_label}")