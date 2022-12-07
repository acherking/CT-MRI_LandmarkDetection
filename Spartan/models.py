import numpy as np
import math
import tensorflow as tf
# from tensorflow import keras
# for tensorflow 2.7 (on Spartan)
# import tensorflow.keras.layers as layers
# for test on tensorflow 2.9
import keras
import keras.layers as layers

from tensorflow import Tensor


# https://keras.io/examples/vision/3D_image_classification/
def first_model(width=170, height=170, depth=30):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.GlobalAveragePooling3D()(x_hidden)
    x_hidden = layers.Dense(units=512, activation="relu")(x_hidden)
    x_hidden = layers.Dropout(0.3)(x_hidden)

    outputs = layers.Dense(units=3, )(x_hidden)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3d-cnn")
    return model


# inspired by: "Deep learning approach for automatic landmark detection
# and alignment analysis in whole-spine lateral radiographs"
# Author: Yu-Cheng Yeh, Chi-Hung Weng ...
def relu_bn(inputs: Tensor) -> Tensor:
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = layers.Conv3D(kernel_size=kernel_size,
                      strides=(1 if not downsample else 2),
                      filters=filters,
                      padding="same")(x)
    y = relu_bn(y)
    y = layers.Conv3D(kernel_size=kernel_size,
                      strides=1,
                      filters=filters,
                      padding="same")(y)

    x = layers.Conv3D(kernel_size=1,
                      strides=(1 if not downsample else 2),
                      filters=filters,
                      padding="same")(x)
    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def coordinate_3d(batch_size, row_size, clown_size, slice_size):
    # pts (x, y, z) * 4
    # matrix_x -> x (clown), matrix_Y -> y (row), matrix_Z -> z (slice)
    base_array = np.ones(row_size * clown_size * slice_size).reshape((row_size, clown_size, slice_size))

    matrix_x = np.copy(base_array)
    c_0_v = [(2 * i - clown_size - 1) / clown_size for i in range(1, clown_size + 1)]
    for i in range(clown_size):
        matrix_x[:, i, :] = matrix_x[:, i, :] * c_0_v[i]

    matrix_y = np.copy(base_array)
    c_1_v = [(2 * i - row_size - 1) / row_size for i in range(1, row_size + 1)]
    for i in range(row_size):
        matrix_y[i, :, :] = matrix_y[i, :, :] * c_1_v[i]

    matrix_z = np.copy(base_array)
    c_2_v = [(2 * i - slice_size - 1) / slice_size for i in range(1, slice_size + 1)]
    for i in range(slice_size):
        matrix_z[:, :, i] = matrix_z[:, :, i] * c_2_v[i]

    matrix_x = tf.repeat(matrix_x.reshape((row_size, clown_size, slice_size, 1)), repeats=4, axis=-1).numpy() \
        .reshape((row_size, clown_size, slice_size, 4, 1))
    matrix_y = tf.repeat(matrix_y.reshape((row_size, clown_size, slice_size, 1)), repeats=4, axis=-1).numpy() \
        .reshape((row_size, clown_size, slice_size, 4, 1))
    matrix_z = tf.repeat(matrix_z.reshape((row_size, clown_size, slice_size, 1)), repeats=4, axis=-1).numpy() \
        .reshape((row_size, clown_size, slice_size, 4, 1))

    coordinate_xyz = layers.Concatenate(axis=-1)([matrix_x, matrix_y, matrix_z])

    batch_coordinate_xyz = []
    for i in range(batch_size):
        batch_coordinate_xyz.append(np.copy(coordinate_xyz))

    return np.asarray(batch_coordinate_xyz)


# y_true: batch_size*4*3 array
# y_pred: [stage1_output(batch_size*4*3 array), stage2_output(batch_size*4*3 array)]
def two_stage_wing_loss(y_true, y_pred):
    [y_stage1, y_stage2] = y_pred

    return wing_loss(y_true, y_stage1) + wing_loss(y_true, y_stage2)


def wing_fn(x, w=5, e=1):
    if abs(x) < w:
        y = w * math.log(1 + abs(x) / e)
    else:
        y = abs(x) - w + w * math.log(1 + w / e)

    return y


def wing_loss(landmarks, labels):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, dimension].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    w = 10.0
    epsilon = 2.0
    with tf.name_scope('wing_loss'):
        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.math.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss


def spine_lateral_radiograph_model(width=176, height=176, depth=48):
    """
    The original model is for 2D image, our data are 3D.
    Change it to a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))
    # e.x. batches*170*170*30*4*3, 4 type of coordinates, 3 dimensions
    base_coordinate_xyz = keras.Input((width, height, depth, 4, 3))

    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)

    # Stage 1
    x = residual_block(x, downsample=False, filters=64)
    violet_x = residual_block(x, downsample=False, filters=64)

    # x = residual_block(violet_x, downsample=False, filters=128)
    # yellow_x = residual_block(x, downsample=False, filters=128)
    #
    # x = residual_block(yellow_x, downsample=True, filters=256)
    # blue_x = residual_block(x, downsample=False, filters=256)
    #
    # x = residual_block(blue_x, downsample=True, filters=512)
    # green_x = residual_block(x, downsample=False, filters=512)
    # green_x = residual_block(green_x, downsample=False, filters=512)
    #
    # x = residual_block(green_x, downsample=False, filters=256)
    # x = layers.UpSampling3D(size=2)(x)
    # blue_x = layers.Add()([x, blue_x])
    #
    # x = residual_block(blue_x, downsample=False, filters=128)
    # x = layers.UpSampling3D(size=2)(x)
    # yellow_x = layers.Add()([x, yellow_x])

    # change ***
    # x = residual_block(yellow_x, downsample=False, filters=64)
    x = residual_block(violet_x, downsample=False, filters=64)
    # x = layers.UpSampling3D(size=2)(x)
    violet_x = layers.Add()([x, violet_x])

    x = residual_block(violet_x, downsample=False, filters=64)
    # grey_x = layers.UpSampling3D(size=2)(x)
    grey_x_s1 = x

    x = residual_block(grey_x_s1, downsample=False, filters=4)
    x = residual_block(x, downsample=False, filters=4)
    heatmap_s1 = residual_block(x, downsample=False, filters=4)

    # Stage 2
    # blue_x = residual_block(blue_x, downsample=False, filters=256)
    # blue_x = residual_block(blue_x, downsample=False, filters=256)
    # blue_x = residual_block(blue_x, downsample=False, filters=256)
    #
    # yellow_x = residual_block(yellow_x, downsample=False, filters=128)
    # yellow_x = residual_block(yellow_x, downsample=False, filters=128)

    violet_x = residual_block(violet_x, downsample=False, filters=64)

    # Upsampling & Concatenate
    # upsampling_blue_x = layers.UpSampling3D(size=2)(blue_x)
    # change ***
    # grey_x_s2 = layers.Concatenate(axis=4)([upsampling_blue_x, yellow_x, violet_x, grey_x_s1])
    grey_x_s2 = layers.Concatenate(axis=4)([violet_x, grey_x_s1])

    x = residual_block(grey_x_s2, downsample=False, filters=4)
    x = residual_block(x, downsample=False, filters=4)
    heatmap_s2 = residual_block(x, downsample=False, filters=4)

    # in our project, e.x. heatmap shape: 170*170*30*4
    pro_matrix_s1 = layers.Reshape((width, height, depth, 4, 3)) \
        (tf.repeat(layers.Softmax(axis=[0, 1, 2])(heatmap_s1), repeats=3, axis=-1))
    outputs_s1 = tf.math.reduce_sum(layers.multiply([base_coordinate_xyz, pro_matrix_s1]), axis=[1, 2, 3])
    # model_s1 = keras.Model([inputs, base_coordinate_xyz], outputs_s1, name="ResStage1")

    pro_matrix_s2 = layers.Reshape((width, height, depth, 4, 3)) \
        (tf.repeat(layers.Softmax(axis=[0, 1, 2])(heatmap_s2), repeats=3, axis=-1))
    outputs_s2 = tf.math.reduce_sum(layers.multiply([base_coordinate_xyz, pro_matrix_s2]), axis=[1, 2, 3])
    # model_s2 = keras.Model([inputs, base_coordinate_xyz], outputs_s2, name="ResStage2")

    model = keras.Model([inputs, base_coordinate_xyz], [outputs_s1, outputs_s2], name="ResModel")

    return model
