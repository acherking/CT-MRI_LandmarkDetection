import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

import support_modules
import models

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

size = (100, 100, 100)

X_train, Y_train, X_val, Y_val, X_test, Y_test = \
    support_modules.load_dataset("/data/gpfs/projects/punim1836/Data/cropped/cropped_volumes_x5050y5050z5050.npy",
                                 "/data/gpfs/projects/punim1836/Data/cropped/cropped_points_x5050y5050z5050.npy")

Y_train_one = np.asarray(Y_train)[:, 0, :].reshape((1400, 1, 3))
Y_val_one = np.asarray(Y_val)[:, 0, :].reshape((200, 1, 3))

""" *** Training Process *** """

batch_size = 2
epochs = 100

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_one))
train_dataset = train_dataset.shuffle(buffer_size=2800, reshuffle_each_iteration=True).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val_one))
val_dataset = val_dataset.shuffle(buffer_size=400, reshuffle_each_iteration=True).batch(batch_size)

# Check these datasets
for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    print("train_dataset, step: ", step)
    print("x shape: ", x_batch_train.shape, type(x_batch_train))
    print("y shape: ", y_batch_train.shape, type(y_batch_train))
    break

for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
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

# Instantiate a metric object
train_mse_metric = keras.metrics.Mean()
val_mse_metric = keras.metrics.Mean()

# Get model.
model = models.first_model(width=size[0], height=size[1], depth=size[2])
# model = models.straight_model(width=size[0], height=size[1], depth=size[2])
model.summary()


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Compute the loss value for this batch.
        y_pred = model(x, training=True)
        mse_pixel = mse(y, y_pred)

    # Update training metric.
    train_mse_metric.update_state(mse_pixel)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(mse_pixel, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return mse_pixel


@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    mse_pixel = mse(y, y_pred)
    # Update val metrics
    val_mse_metric.update_state(mse_pixel)


# Training loop
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of a dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_mse = train_step(x_batch_train, y_batch_train)

        # Logging every *** batches
        if step % 100 == 0:
            print("********Step ", step, " ********")
            print("Training loss (MSE):             %.3f" % loss_mse.numpy())
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_mse = train_mse_metric.result()
    print("Training (MSE) over epoch:       %.4f" % (float(train_mse),))

    # Reset the metric's state at the end of an epoch
    train_mse_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        test_step(x_batch_val, y_batch_val)

    val_mse_res = val_mse_metric.result()
    val_mse_metric.reset_states()
    print("Validation (MSE):                %.3f" % (float(val_mse_res),))
    print("Time taken:                      %.2fs" % (time.time() - start_time))

# model.save("slr_model_01")
