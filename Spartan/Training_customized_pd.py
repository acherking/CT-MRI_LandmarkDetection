import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

import support_modules
import models

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

size = (176, 176, 48)
with_res = True

str_size = str(size[0]) + "_" + str(size[1]) + "_" + str(size[2])
if with_res:
    str_size = str_size + "_PD"

X_train, Y_train, res_train, X_val, Y_val, res_val, X_test, Y_test, res_test = \
    support_modules.load_dataset("/data/gpfs/projects/punim1836/Data/rescaled_data/" + str_size + "/",
                                 size, with_res=with_res)

Y_train_one = np.asarray(Y_train)[:, 0, :].reshape((700, 1, 3))
Y_val_one = np.asarray(Y_val)[:, 0, :].reshape((100, 1, 3))

""" *** Training Process *** """

batch_size = 2
epochs = 100

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_one, res_train))
train_dataset = train_dataset.shuffle(buffer_size=1400, reshuffle_each_iteration=True).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val_one, res_val))
val_dataset = val_dataset.shuffle(buffer_size=200, reshuffle_each_iteration=True).batch(batch_size)

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
loss_fn = models.two_stage_wing_loss
wing_loss = models.wing_loss
mse = tf.keras.losses.MeanSquaredError()
mse_with_res = models.mse_with_res

# Instantiate a metric object
train_mse_metric = keras.metrics.Mean()
train_mse_res_metric = keras.metrics.Mean()
val_mse_res_metric = keras.metrics.Mean()

# Get model.
model = models.first_model(width=size[0], height=size[1], depth=size[2])
model.summary()


@tf.function
def train_step(x, y, res):
    with tf.GradientTape() as tape:
        # Compute the loss value for this batch.
        y_pred = model(x, training=True)
        loss_value = mse_with_res(y, y_pred, res)

    # record MSE in pixel distance (without resolution)
    loss_mse = mse(y, y_pred)

    # Update training metric.
    train_mse_metric.update_state(loss_mse.numpy())
    train_mse_res_metric.update_state(loss_value.numpy())

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss_value, loss_mse


# Training loop
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of a dataset.
    for step, (x_batch_train, y_batch_train, res_batch_train) in enumerate(train_dataset):
        loss_value, loss_mse = train_step()

        # Logging every *** batches
        if step % 100 == 0:
            print("Epoch:", epoch, "Step:", step)
            print("Training loss (MSE):             %.3f" % loss_mse.numpy())
            print("Training loss (MSE with Res):    %.3f" % loss_value.numpy())

    # Display metrics at the end of each epoch.
    train_mse = train_mse_metric.result()
    train_mse_res = train_mse_res_metric.result()
    print("Training mse over epoch:     %.4f" % (float(train_mse),))
    print("Training mse_res over epoch: %.4f" % (float(train_mse_res),))

    # Reset the metric's state at the end of an epoch
    train_mse_metric.reset_states()
    train_mse_res_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for step, (x_batch_val, y_batch_val, res_batch_val) in enumerate(val_dataset):
        y_pred_val = model(x_batch_val, training=False)
        val_mse_res = mse_with_res(y_batch_val, y_pred_val, res_batch_val)
        # Update val metrics
        val_mse_res_metric.update_state(val_mse_res.numpy())
    val_mse_res = val_mse_res_metric.result()
    val_mse_res_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_mse_res),))
    print("Time taken: %.2fs" % (time.time() - start_time))

# model.save("slr_model_01")
