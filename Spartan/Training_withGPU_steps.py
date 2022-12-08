import tensorflow as tf
from tensorflow import keras

import support_modules
import models

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

X_train, Y_train, X_val, Y_val, X_test, Y_test = \
    support_modules.load_dataset("/data/gpfs/projects/punim1836/Data/rescaled_data/176_176_48/", (176, 176, 48))

""" *** Training Process *** """

batch_size = 1

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1400, reshuffle_each_iteration=True).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_dataset = val_dataset.shuffle(buffer_size=200, reshuffle_each_iteration=True).batch(batch_size)

# Check these datasets
for step, (x, y) in enumerate(train_dataset):
    print("train_dataset, step: ", step)
    print("x shape: ", x.shape, type(x))
    print("y shape: ", y.shape, type(y))
    print(y.shape)
    break

for step, (x, y) in enumerate(val_dataset):
    print("val_dataset, step: ", step)
    print("x shape: ", x.shape, type(x))
    print("y shape: ", y.shape, type(y))
    print(y.shape)
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

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Get model.
slr_model = models.spine_lateral_radiograph_model(width=176, height=176, depth=48)
slr_model.summary()

base_cor_xyz = models.coordinate_3d(batch_size, 176, 176, 48)

for epoch in range(100):
    # Iterate over the batches of a dataset.
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            [y_pred_s1, y_pred_s2] = slr_model([x, base_cor_xyz])
            # Compute the loss value for this batch.
            loss_value = loss_fn(y, [y_pred_s1, y_pred_s2])

        # Update the state of the `accuracy` metric.
        accuracy.update_state(y, y_pred_s2)

        # Update the weights of the model to minimize the loss value.
        gradients = tape.gradient(loss_value, slr_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, slr_model.trainable_weights))

        # Logging the current accuracy value so far.
        if step % 100 == 0:
            print("Epoch:", epoch, "Step:", step)
            print("Total running accuracy so far: %.3f" % accuracy.result())

    # Reset the metric's state at the end of an epoch
    accuracy.reset_states()

slr_model.save_weights('./slr_weights')
