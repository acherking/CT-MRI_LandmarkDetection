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
def first_model(height=170, width=170, depth=30, points_num=4):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((height, width, depth, 1))

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.GlobalAveragePooling3D()(x_hidden)
    x_hidden = layers.Dense(units=512, activation="relu")(x_hidden)
    x_hidden = layers.Dropout(0.3)(x_hidden)

    # outputs = layers.Dense(units=1*3, )(x_hidden)
    # outputs = layers.Reshape((1, 3))(outputs)
    outputs = layers.Dense(units=points_num*3, )(x_hidden)
    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3d-cnn")
    return model


def dsnt_model(height=176, width=176, depth=48):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((height, width, depth, 1))

    # e.x. batches*170*170*30*4*3, 4 type of coordinates, 3 dimensions
    base_coordinate_xyz = keras.Input((height, width, depth, 1, 3))

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden_1 = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu")(x_hidden_1)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same", activation="relu")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same", activation="relu")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden_1 = layers.UpSampling3D(size=2)(x_hidden_1)
    x_hidden = layers.UpSampling3D(size=8)(x_hidden)
    x_hidden = layers.Add()([x_hidden, x_hidden_1])

    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same", activation="relu")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Conv3D(filters=1, kernel_size=3, padding="same", activation="relu")(x_hidden)
    heatmap = layers.BatchNormalization()(x_hidden)

    # in our project, e.x. heatmap shape: 170*170*30*4
    pro_matrix = layers.Reshape((width, height, depth, 1, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="softmax")(heatmap), repeats=3, axis=-1))
    outputs = tf.math.reduce_sum(layers.multiply([base_coordinate_xyz, pro_matrix]), axis=[1, 2, 3])

    # outputs = layers.Dense(units=3, )(x_hidden)
    # outputs = layers.Reshape((1, 3))(outputs)

    # Define the model.
    model = keras.Model([inputs, base_coordinate_xyz], outputs, name="dsnt_model")
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


def coordinate_3d(batch_size, landmarks_num, row_size, clown_size, slice_size):
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

    matrix_x = tf.repeat(matrix_x.reshape((row_size, clown_size, slice_size, 1)), repeats=landmarks_num, axis=-1)\
        .numpy().reshape((row_size, clown_size, slice_size, landmarks_num, 1))
    matrix_y = tf.repeat(matrix_y.reshape((row_size, clown_size, slice_size, 1)), repeats=landmarks_num, axis=-1)\
        .numpy().reshape((row_size, clown_size, slice_size, landmarks_num, 1))
    matrix_z = tf.repeat(matrix_z.reshape((row_size, clown_size, slice_size, 1)), repeats=landmarks_num, axis=-1)\
        .numpy().reshape((row_size, clown_size, slice_size, landmarks_num, 1))

    coordinate_xyz = layers.Concatenate(axis=-1)([matrix_x, matrix_y, matrix_z])

    batch_coordinate_xyz = []
    for i in range(batch_size):
        batch_coordinate_xyz.append(np.copy(coordinate_xyz))

    return np.asarray(batch_coordinate_xyz)


# y_true: batch_size*4*3 array
# y_pred: [stage1_output(batch_size*4*3 array), stage2_output(batch_size*4*3 array)]
def two_stage_wing_loss(y_true, y_pred, res):
    [y_stage1, y_stage2] = y_pred

    return (wing_loss(y_true, y_stage1, res) + wing_loss(y_true, y_stage2, res)) / y_true.shape[1]


def one_stage_wing_loss(y_true, y_pred, res):
    return wing_loss(y_true, y_pred, res) / y_true.shape[1]


def wing_loss(landmarks, labels, res):
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
        # repeat res to make a convenient calculation follow
        num_landmarks = x.shape[1]
        rep_res = tf.repeat(res, num_landmarks, axis=1)
        # change pixel distance to mm (kind of normalization I think)
        x = x * rep_res
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.math.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss


def mse_with_res(y_true, y_pred, res):
    """
    :param y_true: [batch_size, num_landmarks, dimension(column, row, slice)]
    :param y_pred: [batch_size, num_landmarks, dimension(column, row, slice)]
    :param res: Pixel distance in mm, [batch_size, 1, dimension(column, row, slice)]
    :return: mean square error along batch_size
    """
    with tf.name_scope('mse_res_loss'):
        err_diff = y_true - y_pred
        # repeat res to make a convenient calculation follow
        num_landmarks = err_diff.shape[1]
        rep_res = tf.repeat(res, num_landmarks, axis=1)
        # change pixel distance to mm (kind of normalization I think)
        losses = err_diff * rep_res
        square_losses = tf.pow(losses, 2)
        loss = tf.reduce_mean(tf.reduce_sum(square_losses, axis=[1, 2]), axis=0)
        return loss


def spine_lateral_radiograph_model(height=176, width=176, depth=48):
    """
    The original model is for 2D image, our data are 3D.
    Change it to a 3D convolutional neural network model."""
    inputs = keras.Input((height, width, depth, 1))
    # e.x. batches*170*170*30*4*3, 4 type of coordinates, 3 dimensions
    base_coordinate_xyz = keras.Input((width, height, depth, 4, 3))

    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)

    # Stage 1
    x = residual_block(x, downsample=True, filters=64)
    violet_x = residual_block(x, downsample=False, filters=64)

    x = residual_block(violet_x, downsample=True, filters=128)
    yellow_x = residual_block(x, downsample=False, filters=128)

    x = residual_block(yellow_x, downsample=True, filters=256)
    blue_x = residual_block(x, downsample=False, filters=256)

    x = residual_block(blue_x, downsample=True, filters=512)
    green_x = residual_block(x, downsample=False, filters=512)
    green_x = residual_block(green_x, downsample=False, filters=512)

    x = residual_block(green_x, downsample=False, filters=256)
    x = layers.UpSampling3D(size=2)(x)
    blue_x = layers.Add()([x, blue_x])

    x = residual_block(blue_x, downsample=False, filters=128)
    x = layers.UpSampling3D(size=2)(x)
    yellow_x = layers.Add()([x, yellow_x])

    x = residual_block(yellow_x, downsample=False, filters=64)
    x = layers.UpSampling3D(size=2)(x)
    violet_x = layers.Add()([x, violet_x])

    x = residual_block(violet_x, downsample=False, filters=64)
    grey_x_s1 = layers.UpSampling3D(size=2)(x)

    x = residual_block(grey_x_s1, downsample=False, filters=4)
    x = residual_block(x, downsample=False, filters=4)
    heatmap_s1 = residual_block(x, downsample=False, filters=4)

    # Stage 2
    blue_x = residual_block(blue_x, downsample=False, filters=256)
    blue_x = residual_block(blue_x, downsample=False, filters=256)
    blue_x = residual_block(blue_x, downsample=False, filters=256)

    yellow_x = residual_block(yellow_x, downsample=False, filters=128)
    yellow_x = residual_block(yellow_x, downsample=False, filters=128)

    violet_x = residual_block(violet_x, downsample=False, filters=64)

    # Upsampling & Concatenate
    upsampling_blue_x = layers.UpSampling3D(size=8)(blue_x)
    upsampling_yellow_x = layers.UpSampling3D(size=4)(yellow_x)
    upsampling_violet_x = layers.UpSampling3D(size=2)(violet_x)
    grey_x_s2 = layers.Concatenate(axis=4)([upsampling_blue_x, upsampling_yellow_x, upsampling_violet_x, grey_x_s1])

    x = residual_block(grey_x_s2, downsample=False, filters=4)
    x = residual_block(x, downsample=False, filters=4)
    heatmap_s2 = residual_block(x, downsample=False, filters=4)

    # in our project, e.x. heatmap shape: 170*170*30*4
    pro_matrix_s1 = layers.Reshape((width, height, depth, 4, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="stage1_softmax")(heatmap_s1), repeats=3, axis=-1))
    outputs_s1 = tf.math.reduce_sum(layers.multiply([base_coordinate_xyz, pro_matrix_s1]), axis=[1, 2, 3])
    # model_s1 = keras.Model([inputs, base_coordinate_xyz], outputs_s1, name="ResStage1")

    pro_matrix_s2 = layers.Reshape((width, height, depth, 4, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="stage2_softmax")(heatmap_s2), repeats=3, axis=-1))
    outputs_s2 = tf.math.reduce_sum(layers.multiply([base_coordinate_xyz, pro_matrix_s2]), axis=[1, 2, 3])
    # model_s2 = keras.Model([inputs, base_coordinate_xyz], outputs_s2, name="ResStage2")

    model = keras.Model([inputs, base_coordinate_xyz], [outputs_s1, outputs_s2], name="ResModel")

    return model


def simple_slr_model(height=176, width=176, depth=48):
    """
    This is a simplified slr model used to debug the original one.
    """
    inputs = keras.Input((height, width, depth, 1))
    # e.x. batches*170*170*30*4*3, 4 type of coordinates, 3 dimensions
    # base_coordinate_xyz = keras.Input((width, height, depth, 4, 3))

    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)

    # Stage 1
    x = residual_block(x, downsample=True, filters=64)
    violet_x = residual_block(x, downsample=False, filters=64)

    x = residual_block(violet_x, downsample=True, filters=128)
    yellow_x = residual_block(x, downsample=False, filters=128)

    x = residual_block(yellow_x, downsample=True, filters=256)
    blue_x = residual_block(x, downsample=False, filters=256)

    x = residual_block(blue_x, downsample=True, filters=512)
    green_x = residual_block(x, downsample=False, filters=512)
    green_x = residual_block(green_x, downsample=False, filters=512)

    x = residual_block(green_x, downsample=False, filters=256)
    x = layers.UpSampling3D(size=2)(x)
    blue_x = layers.Add()([x, blue_x])

    x = residual_block(blue_x, downsample=False, filters=128)
    x = layers.UpSampling3D(size=2)(x)
    yellow_x = layers.Add()([x, yellow_x])

    x = residual_block(yellow_x, downsample=False, filters=64)
    x = layers.UpSampling3D(size=2)(x)
    violet_x = layers.Add()([x, violet_x])

    x = residual_block(violet_x, downsample=False, filters=64)
    grey_x_s1 = layers.UpSampling3D(size=2)(x)

    # x = residual_block(grey_x_s1, downsample=False, filters=12)
    # x = residual_block(x, downsample=False, filters=12)
    # heatmap_s1 = residual_block(x, downsample=False, filters=12)

    # pro_matrix_s1 = layers.Reshape((width, height, depth, 4, 3)) \
    #     (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="stage1_softmax")(heatmap_s1), repeats=3, axis=-1))
    # outputs_s1 = tf.math.reduce_sum(layers.multiply([base_coordinate_xyz, pro_matrix_s1]), axis=[1, 2, 3])

    # model_s1 = keras.Model([inputs, base_coordinate_xyz], outputs_s1, name="slr_stage1")

    x_hidden = layers.Dropout(0.2)(grey_x_s1)
    x_hidden = layers.Flatten()(x_hidden)

    outputs = layers.Dense(units=3)(x_hidden)
    outputs = layers.Reshape((1, 3))(outputs)

    model_s1 = keras.Model(inputs, outputs, name="slr_stage1")

    return model_s1


def straight_model(height=176, width=176, depth=48, points_num=4):

    inputs = keras.Input((height, width, depth, 1))

    # layer 1
    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 3
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 4
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 5
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 6
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 7
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 8
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 9
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 10
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 11
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 12
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    x_hidden = layers.Dropout(0.2)(x_hidden)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num*3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-cnn")

    return model


def straight_model_bn_a(height=176, width=176, depth=48, points_num=4):

    inputs = keras.Input((height, width, depth, 1))

    # layer 1
    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 3
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 4
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 5
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 6
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 7
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 8
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 9
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 10
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 11
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 12
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    x_hidden = layers.Dropout(0.2)(x_hidden)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num*3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-cnn")

    return model
