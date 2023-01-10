import tensorflow as tf
from tensorflow import keras
import numpy as np

import support_modules
import models

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

size = (176, 176, 48)
str_size = str(size[0]) + "_" + str(size[1]) + "_" + str(size[2])

X_train, Y_train, X_val, Y_val, X_test, Y_test = \
    support_modules.load_dataset("/data/gpfs/projects/punim1836/Data/rescaled_data/" + str_size + "/", size)
# X_train, Y_train, X_val, Y_val, X_test, Y_test = support_modules.load_data()

Y_train_one = np.asarray(Y_train)[:, 0, :]
Y_val_one = np.asarray(Y_val)[:, 0, :]

""" *** Training Process *** """

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train_one))
train_dataset = train_dataset.shuffle(buffer_size=1400).batch(2)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val_one))
val_dataset = val_dataset.shuffle(buffer_size=200).batch(2)

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

# Get model.
model = models.straight_model(width=size[0], height=size[1], depth=size[2])
model.summary()

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="mean_squared_error",
    # loss=models.wing_loss,
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["mse"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "straight_model_17617648_B2.checkpoint",
    save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=25,
    mode="min"
)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    # callbacks=[checkpoint_cb, early_stopping_cb],
    callbacks=[checkpoint_cb],
)
