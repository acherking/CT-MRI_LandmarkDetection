import tensorflow as tf
from tensorflow import keras
import numpy as np

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

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_one, res_train))
train_dataset = train_dataset.shuffle(buffer_size=1400, reshuffle_each_iteration=True).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val_one, res_val))
val_dataset = val_dataset.shuffle(buffer_size=200, reshuffle_each_iteration=True).batch(batch_size)

# Check these datasets
for step, (x, y, res) in enumerate(train_dataset):
    print("train_dataset, step: ", step)
    print("x shape: ", x.shape, type(x))
    print("y shape: ", y.shape, type(y))
    print("res: ", res)
    break

for step, (x, y, res) in enumerate(val_dataset):
    print("val_dataset, step: ", step)
    print("x shape: ", x.shape, type(x))
    print("y shape: ", y.shape, type(y))
    print("res: ", res)
    break

initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "slr_weights.checkpoint",
    monitor="val_outputs_s2",
    save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    mode="min"
)

# Instantiate a metric object
accuracy = keras.metrics.MeanSquaredError()

loss_fn = models.two_stage_wing_loss
wing_loss = models.wing_loss
mse = tf.keras.losses.MeanSquaredError()
mse_with_res = models.mse_with_res

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Get model.
first_model = models.first_model(width=size[0], height=size[1], depth=size[2])
first_model.summary()

for epoch in range(100):
    # Iterate over the batches of a dataset.
    for step, (x, y, res) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Compute the loss value for this batch.
            y_pred = first_model(x)
            loss_value = mse_with_res(y, y_pred, res)

        # Update the state of the `accuracy` metric.
        accuracy.update_state(y, y_pred)
        loss_mse_with_res = loss_value

        # Update the weights of the model to minimize the loss value.
        gradients = tape.gradient(loss_value, first_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, first_model.trainable_weights))

        # Logging the current accuracy value so far.
        if step % 100 == 0:
            print("Epoch:", epoch, "Step:", step)
            print("accuracy (MSE) so far: %.3f" % accuracy.result())
            print("accuracy (MSE with Res: %.3f)" % loss_mse_with_res.numpy())

    # Reset the metric's state at the end of an epoch
    accuracy.reset_states()

# model.save("slr_model_01")
