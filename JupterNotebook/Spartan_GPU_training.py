import tensorflow as tf
from tensorflow import keras
import keras.layers as layers
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

""" *** Prepare Data *** """

# Set Data Path
data_base_path = "/data/gpfs/projects/punim1836/Data/combined_aug_data/"
X_train_path = data_base_path + "X_train_data.npy"
Y_train_path = data_base_path + "Y_train_data.npy"
X_val_path = data_base_path + "X_val_data.npy"
Y_val_path = data_base_path + "Y_val_data.npy"

X_train = np.load(X_train_path, allow_pickle=True)
Y_train = np.load(Y_train_path, allow_pickle=True)
X_val = np.load(X_val_path, allow_pickle=True)
Y_val = np.load(Y_val_path, allow_pickle=True)

# Data shape validation
print("X_train Shape: ", np.shape(X_train))
print("Y_train Shape: ", np.shape(Y_train))

print("X_val Shape: ", np.shape(X_val))
print("Y_val Shape: ", np.shape(Y_val))
#
# print("X_test Shape: ", np.shape(X_test))
# print("Y_test Shape: ", np.shape(Y_test))

# Reshape the data (data-size, row-size?, column-size?, slice-size, channel-size)
X_train_reshape = np.asarray(X_train).reshape(700, 170, 170, 30, 1)
Y_train_one = np.asarray(Y_train)[:, 0, :]

X_val_reshape = np.asarray(X_val).reshape(100, 170, 170, 30, 1)
Y_val_one = np.asarray(Y_val)[:, 0, :]
#
# X_test_reshape = np.asarray(X_test).reshape(200, 170, 170, 30, 1)
# Y_test_one = np.asarray(Y_test)[:, 0, :]

print("X_train_reshape Shape: ", np.shape(X_train))
print("Y_train_one Shape: ", np.shape(Y_train))

print("X_val_reshape Shape: ", np.shape(X_val))
print("Y_val_one Shape: ", np.shape(Y_val))

# print("X_test_reshape Shape: ", np.shape(X_test))
# print("Y_test_one Shape: ", np.shape(Y_test))

""" *** Training Process *** """

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshape, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1400).batch(2)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val_reshape, Y_val))
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


def get_model(width=170, height=170, depth=30):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x_hidden)
    # x = layers.MaxPool3D(pool_size=2)(x)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x_hidden)
    # x = layers.MaxPool3D(pool_size=2)(x)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.GlobalAveragePooling3D()(x_hidden)
    x_hidden = layers.Dense(units=512, activation="relu")(x_hidden)
    x_hidden = layers.Dropout(0.3)(x_hidden)

    outputs = layers.Dense(units=3, )(x_hidden)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
my_model = get_model(width=170, height=170, depth=30)
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
