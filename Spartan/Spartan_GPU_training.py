import tensorflow as tf
from tensorflow import keras

import support_modules
import models

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

X_train_reshape, Y_train_one, X_val_reshape, Y_val_one, X_test_reshape, Y_test_one = \
    support_modules.load_data()

""" *** Training Process *** """

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshape, Y_train_one))
train_dataset = train_dataset.shuffle(buffer_size=1400).batch(2)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_reshape, Y_val_one))
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

# Define data loaders. (Shawn: Looks like the same as above)
# train_loader = tf.data.Dataset.from_tensor_slices((X_train_reshape, Y_train))
# validation_loader = tf.data.Dataset.from_tensor_slices((X_val_reshape, Y_val))
#
# batch_size = 2
# train_dataset = (
#     train_loader.shuffle(len(X_train_reshape))
#     .batch(batch_size)
#     .prefetch(2)
# )
# # Only rescale.
# validation_dataset = (
#     validation_loader.shuffle(len(X_val))
#     .batch(batch_size)
#     .prefetch(2)
# )

# Get model.
my_model = models.first_model(width=170, height=170, depth=30)
my_model.summary()

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
my_model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    #metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
my_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
