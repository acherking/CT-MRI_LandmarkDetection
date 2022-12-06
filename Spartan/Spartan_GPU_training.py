import tensorflow as tf
from tensorflow import keras

import support_modules
import models

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

X_train, Y_train, X_val, Y_val, X_test, Y_test = \
    support_modules.load_data("/data/gpfs/projects/punim1836/Data/rescaled_data/17617648/", (176, 176, 48))

""" *** Training Process *** """

# Prepare dataset used in the training process
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.shuffle(buffer_size=1400).batch(2)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
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
slr_model = models.spine_lateral_radiograph_model(width=176, height=176, depth=48)
slr_model.summary()

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)
slr_model.compile(
    loss=models.two_stage_wing_loss,
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics={'outputs_s1': 'mse', 'outputs_s2': 'mse'},
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "slr_weights.checkpoint",
    monitor=models.two_stage_wing_loss,
    save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor=models.two_stage_wing_loss,
    patience=15,
    mode="min"
)

# Train the model, doing validation at the end of each epoch
epochs = 100
slr_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
