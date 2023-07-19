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
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)
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

    matrix_x = tf.repeat(matrix_x.reshape((row_size, clown_size, slice_size, 1)), repeats=landmarks_num, axis=-1) \
        .numpy().reshape((row_size, clown_size, slice_size, landmarks_num, 1))
    matrix_y = tf.repeat(matrix_y.reshape((row_size, clown_size, slice_size, 1)), repeats=landmarks_num, axis=-1) \
        .numpy().reshape((row_size, clown_size, slice_size, landmarks_num, 1))
    matrix_z = tf.repeat(matrix_z.reshape((row_size, clown_size, slice_size, 1)), repeats=landmarks_num, axis=-1) \
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


def two_stage_mse_loss(y_true, y_pred, res):
    [y_stage1, y_stage2] = y_pred

    loss_s1 = mse_with_res(y_true, y_stage1, res)
    loss_s2 = mse_with_res(y_true, y_stage2, res)

    return [loss_s1, loss_s2]


def mse_with_res(y_true, y_pred, res):
    """
    :param y_true: [batch_size, num_landmarks, dimension(column, row, slice)]
    :param y_pred: [batch_size, num_landmarks, dimension(column, row, slice)]
    :param res: Pixel distance in mm, [batch_size, 1, dimension(column, row, slice)]
    :return: mean square error along batch_size (mm^2)
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


def down_sampling_shape(dim):
    dim_list = [dim]
    while dim > 1:
        dim = int(np.ceil(dim/2))
        dim_list.append(dim)

    return dim_list


def my_upsampling(size, x, input_shape):
    row_dim_list = down_sampling_shape(input_shape[0])
    column_dim_list = down_sampling_shape(input_shape[1])
    slice_dim_list = down_sampling_shape(input_shape[2])

    for i in range(int(np.log2(size))):
        x = layers.UpSampling3D(size=2)(x)
        row_dim = x.shape[1]
        for r_d in row_dim_list:
            if (row_dim - r_d) == 1:
                row_dim = r_d
                break
        column_dim = x.shape[2]
        for c_d in column_dim_list:
            if (column_dim - c_d) == 1:
                column_dim = c_d
                break
        slice_dim = x.shape[3]
        for s_d in slice_dim_list:
            if (slice_dim - s_d) == 1:
                slice_dim = s_d
                break
        x = x[:, 0:row_dim, 0:column_dim, 0:slice_dim, :]

    return x


# spine_lateral_radiograph_model
def slr_model(height=176, width=176, depth=48, points_num=2, batch_size=2):
    """
    The original model is for 2D image, our data are 3D.
    Change it to a 3D convolutional neural network model."""
    inputs = keras.Input((height, width, depth, 1))
    input_shape = (height, width, depth)
    # e.x. batches*170*170*30*4*3, 4 type of coordinates, 3 dimensions
    # base_coordinate_rcs = keras.Input((height, width, depth, points_num, 3))

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)

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
    x = my_upsampling(2, x, input_shape)
    blue_x = layers.Add()([x, blue_x])

    x = residual_block(blue_x, downsample=False, filters=128)
    x = my_upsampling(2, x, input_shape)
    yellow_x = layers.Add()([x, yellow_x])

    x = residual_block(yellow_x, downsample=False, filters=64)
    x = my_upsampling(2, x, input_shape)
    violet_x = layers.Add()([x, violet_x])

    x = residual_block(violet_x, downsample=False, filters=64)
    grey_x_s1 = my_upsampling(2, x, input_shape)

    x = residual_block(grey_x_s1, downsample=False, filters=points_num)
    x = residual_block(x, downsample=False, filters=points_num)
    heatmap_s1 = residual_block(x, downsample=False, filters=points_num)

    # Stage 2
    blue_x = residual_block(blue_x, downsample=False, filters=256)
    blue_x = residual_block(blue_x, downsample=False, filters=256)
    blue_x = residual_block(blue_x, downsample=False, filters=256)

    yellow_x = residual_block(yellow_x, downsample=False, filters=128)
    yellow_x = residual_block(yellow_x, downsample=False, filters=128)

    violet_x = residual_block(violet_x, downsample=False, filters=64)

    # Upsampling & Concatenate
    upsampling_blue_x = my_upsampling(8, blue_x, input_shape)
    upsampling_yellow_x = my_upsampling(4, yellow_x, input_shape)
    upsampling_violet_x = my_upsampling(2, violet_x, input_shape)
    grey_x_s2 = layers.Concatenate(axis=4)([upsampling_blue_x, upsampling_yellow_x, upsampling_violet_x, grey_x_s1])

    x = residual_block(grey_x_s2, downsample=False, filters=points_num)
    x = residual_block(x, downsample=False, filters=points_num)
    heatmap_s2 = residual_block(x, downsample=False, filters=points_num)

    # in our project, e.x. heatmap shape: 170*170*30*4
    pro_matrix_s1 = layers.Reshape((height, width, depth, points_num, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="stage1_softmax")(heatmap_s1), repeats=3, axis=-1))
    outputs_s1 = tf.math.reduce_sum(layers.multiply([base_cor_rcs, pro_matrix_s1]), axis=[1, 2, 3])
    # model_s1 = keras.Model([inputs, base_coordinate_rcs], outputs_s1, name="ResStage1")

    pro_matrix_s2 = layers.Reshape((height, width, depth, points_num, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="stage2_softmax")(heatmap_s2), repeats=3, axis=-1))
    outputs_s2 = tf.math.reduce_sum(layers.multiply([base_cor_rcs, pro_matrix_s2]), axis=[1, 2, 3])
    # model_s2 = keras.Model([inputs, base_coordinate_rcs], outputs_s2, name="ResStage2")

    model = keras.Model(inputs, [outputs_s1, outputs_s2], name="slr-model")

    return model


def slr_s1_model(height=176, width=176, depth=48, points_num=2, batch_size=2):
    """
    This is a simplified slr model used to debug the original one.
    """
    inputs = keras.Input((height, width, depth, 1))
    # e.x. batches*170*170*30*4*3, 4 type of coordinates, 3 dimensions
    # base_coordinate_xyz = keras.Input((width, height, depth, 4, 3))

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)

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

    x = residual_block(grey_x_s1, downsample=False, filters=points_num)
    x = residual_block(x, downsample=False, filters=points_num)
    heatmap_s1 = residual_block(x, downsample=False, filters=points_num)

    # add dropout?
    heatmap_s1 = layers.Dropout(0.2)(heatmap_s1)

    pro_matrix_s1 = layers.Reshape((width, height, depth, points_num, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="stage1_softmax")(heatmap_s1), repeats=3, axis=-1))
    outputs_s1 = tf.math.reduce_sum(layers.multiply([base_cor_rcs, pro_matrix_s1]), axis=[1, 2, 3])

    model_s1 = keras.Model(inputs, outputs_s1, name="slr_stage1")

    # x_hidden = layers.Dropout(0.2)(grey_x_s1)
    # x_hidden = layers.Flatten()(x_hidden)
    #
    # outputs = layers.Dense(units=3*points_num)(x_hidden)
    # outputs = layers.Reshape((points_num, 3))(outputs)
    #
    # model_s1 = keras.Model(inputs, outputs, name="slr_stage1")

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
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-cnn")

    return model


def straight_model_short(height=176, width=176, depth=48, points_num=4):
    inputs = keras.Input((height, width, depth, 1))

    # layer 1
    x_hidden = layers.Conv3D(filters=16, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 3
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # # layer 4
    # x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    # x_hidden = layers.BatchNormalization()(x_hidden)
    # x_hidden = layers.ReLU()(x_hidden)
    #
    # # layer 5
    # x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    # x_hidden = layers.BatchNormalization()(x_hidden)
    # x_hidden = layers.ReLU()(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 6
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # # layer 7
    # x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    # x_hidden = layers.BatchNormalization()(x_hidden)
    # x_hidden = layers.ReLU()(x_hidden)
    #
    # # layer 8
    # x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    # x_hidden = layers.BatchNormalization()(x_hidden)
    # x_hidden = layers.ReLU()(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    #
    # # layer 9
    # x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    # x_hidden = layers.BatchNormalization()(x_hidden)
    # x_hidden = layers.ReLU()(x_hidden)
    #
    # # layer 10
    # x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    # x_hidden = layers.BatchNormalization()(x_hidden)
    # x_hidden = layers.ReLU()(x_hidden)
    #
    # # layer 11
    # x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    # x_hidden = layers.BatchNormalization()(x_hidden)
    # x_hidden = layers.ReLU()(x_hidden)
    #
    # # layer 12
    # x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    # x_hidden = layers.BatchNormalization()(x_hidden)
    # x_hidden = layers.ReLU()(x_hidden)

    x_hidden = layers.Dropout(0.2)(x_hidden)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

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
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

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
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-cnn")

    return model


def straight_model_no_bn(height=176, width=176, depth=48, points_num=4):
    inputs = keras.Input((height, width, depth, 1))

    # layer 1
    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 3
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 4
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 5
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 6
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 7
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 8
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 9
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 10
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 11
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 12
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    x_hidden = layers.Dropout(0.2)(x_hidden)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-cnn")

    return model


def straight_model_more_dropout(height=176, width=176, depth=48, points_num=4):
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
    x_hidden = layers.Dropout(0.2)(x_hidden)

    # layer 10
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.Dropout(0.2)(x_hidden)

    # layer 11
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.Dropout(0.2)(x_hidden)

    # layer 12
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    x_hidden = layers.Dropout(0.2)(x_hidden)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-cnn")

    return model


# Convolutional Only
def cov_only_dsnt_model(height=176, width=176, depth=48, points_num=2, batch_size=2, dsnt=False):
    inputs = keras.Input((height, width, depth, 1))

    # layer 1
    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

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
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 6
    x = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # layer 7
    heatmap_s1 = residual_block(x, downsample=False, filters=points_num)

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)
    pro_matrix = layers.Reshape((width, height, depth, points_num, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="softmax")(heatmap_s1), repeats=3, axis=-1))
    outputs = tf.math.reduce_sum(layers.multiply([base_cor_rcs, pro_matrix]), axis=[1, 2, 3])

    model = keras.Model(inputs, outputs, name="cov-only-dsnt-model")

    return model


# U-Net: "End-to-End Coordinate Regression Model with Attention-Guided Mechanism
# for Landmark Localization in 3D Medical Images"
# not from the origin
# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
def double_conv_block(x, n_filters):
    # Conv3D then ReLU activation
    x = layers.Conv3D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv3D then ReLU activation
    x = layers.Conv3D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)

    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool3D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv3DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    if x.shape[1] < conv_features.shape[1]:
        p_r = (0, 1)
    else:
        p_r = (0, 0)
    if x.shape[2] < conv_features.shape[2]:
        p_c = (0, 1)
    else:
        p_c = (0, 0)
    if x.shape[3] < conv_features.shape[3]:
        p_s = (0, 1)
    else:
        p_s = (0, 0)
    padding_value = (p_r, p_c, p_s)
    x = layers.ZeroPadding3D(padding_value)(x)
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv3D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def u_net(inputs, points_num=2, dsnt=False):

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    if dsnt:
        heatmaps = layers.Conv3D(points_num, 1, padding="same", activation="softmax")(u9)
    else:
        heatmaps = layers.Conv3D(3, 3, strides=4, activation="relu")(u9)

    return heatmaps


def u_net_model(height=176, width=176, depth=48, points_num=2):
    inputs = keras.Input((height, width, depth, 1))

    reg_mat = u_net(inputs, points_num, dsnt=False)

    # calculate the coordinate directly
    # x = layers.Dropout(0.3)(reg_mat)
    # x = layers.MaxPool3D(4)(x)
    x_hidden = layers.Dropout(0.2)(reg_mat)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # unet model with Keras Functional API
    model = tf.keras.Model(inputs, outputs, name="u_net_model")

    return model


def u_net_dsnt_model(height=176, width=176, depth=48, points_num=2, batch_size=2):
    inputs = keras.Input((height, width, depth, 1))

    heatmaps = u_net(inputs, points_num, dsnt=True)

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)

    pro_matrix = layers.Reshape((width, height, depth, points_num, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="softmax")(heatmaps), repeats=3, axis=-1))
    outputs = tf.math.reduce_sum(layers.multiply([base_cor_rcs, pro_matrix]), axis=[1, 2, 3])

    # unet model with Keras Functional API
    model = tf.keras.Model(inputs, outputs, name="u_net_dsnt_model")

    return model


# https://keras.io/examples/vision/oxford_pets_image_segmentation/
def u_net_Xception_model(height=176, width=176, depth=48, points_num=2):
    inputs = keras.Input((height, width, depth, 1))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv3D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling3D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv3D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling3D(2)(x)

        # Project residual
        residual = layers.UpSampling3D(2)(previous_block_activation)
        residual = layers.Conv3D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv3D(points_num, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Spatial Configuration Net: "Regressing Heatmaps for Multiple Landmark Localization Using CNNs"
# https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression/blob/master/hand_xray/network.py
def sc_net(inputs, points_num, dsnt=False):
    num_filters = 128
    local_kernel_size = (3, 3, 3)
    spatial_kernel_size = (9, 9, 5)
    downsampling_factor = 4
    padding = 'same'
    kernel_initializer = tf.keras.initializers.HeNormal
    activation = tf.nn.relu
    heatmap_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.0001)
    local_activation = None
    spatial_activation = None
    with tf.name_scope('local_appearance'):
        node = layers.Conv3D(num_filters, kernel_size=local_kernel_size, name='conv1', activation=activation, kernel_initializer=kernel_initializer, padding=padding)(inputs)
        node = layers.Conv3D(num_filters, kernel_size=local_kernel_size, name='conv2', activation=activation, kernel_initializer=kernel_initializer, padding=padding)(node)
        node = layers.Conv3D(num_filters, kernel_size=local_kernel_size, name='conv3', activation=activation, kernel_initializer=kernel_initializer, padding=padding)(node)
        # print("points_num:", points_num)
        local_heatmaps = layers.Conv3D(points_num, kernel_size=local_kernel_size, name='local_heatmaps', activation=local_activation,  kernel_initializer=heatmap_initializer, padding=padding)(node)
    with tf.name_scope('spatial_configuration'):
        local_heatmaps_downsampled = layers.AveragePooling3D(downsampling_factor, name='local_heatmaps_downsampled')(local_heatmaps)
        print(tf.shape(local_heatmaps))
        channel_axis = -1
        local_heatmaps_downsampled_split = tf.split(local_heatmaps_downsampled, points_num, channel_axis)
        print(tf.shape(local_heatmaps_downsampled_split))
        spatial_heatmaps_downsampled_split = []
        for i in range(points_num):
            local_heatmaps_except_i = tf.concat([local_heatmaps_downsampled_split[j] for j in range(points_num) if i != j], name=f"h_app_except_{i}", axis=channel_axis)
            h_acc = layers.Conv3D(1, kernel_size=spatial_kernel_size, name='h_acc_'+str(i), activation=spatial_activation, kernel_initializer=heatmap_initializer, padding=padding)(local_heatmaps_except_i)
            spatial_heatmaps_downsampled_split.append(h_acc)
        spatial_heatmaps_downsampled = tf.concat(spatial_heatmaps_downsampled_split, name='spatial_heatmaps_downsampled', axis=channel_axis)
        spatial_heatmaps = layers.UpSampling3D(downsampling_factor, name='spatial_prediction')(spatial_heatmaps_downsampled)
    with tf.name_scope('combination'):
        if dsnt:
            heatmaps = local_heatmaps * spatial_heatmaps
        else:
            # heatmaps = tf.concat([local_heatmaps, spatial_heatmaps], axis=channel_axis)
            heatmaps = local_heatmaps_downsampled * spatial_heatmaps_downsampled
    return heatmaps


def scn_model(height=176, width=176, depth=48, points_num=2):
    inputs = keras.Input((height, width, depth, 1))

    reg_mat = sc_net(inputs, points_num, dsnt=False)

    # calculate the coordinate directly
    # x = layers.Dropout(0.3)(heatmaps)
    # x = layers.MaxPool3D(4)(x)
    x_hidden = layers.Dropout(0.2)(reg_mat)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # unet model with Keras Functional API
    model = tf.keras.Model(inputs, outputs, name="scn_model")

    return model


def scn_dsnt_model(height=176, width=176, depth=48, points_num=2, batch_size=2):
    inputs = keras.Input((height, width, depth, 1))

    heatmaps = sc_net(inputs, points_num, dsnt=True)

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)

    pro_matrix = layers.Reshape((width, height, depth, points_num, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="softmax")(heatmaps), repeats=3, axis=-1))
    outputs = tf.math.reduce_sum(layers.multiply([base_cor_rcs, pro_matrix]), axis=[1, 2, 3])

    # unet model with Keras Functional API
    model = tf.keras.Model(inputs, outputs, name="scn_dsnt_model")

    return model


def get_model(model_name, input_shape, model_output_num, batch_size=2):
    if model_name == "straight_model":
        model = straight_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "slr_model":
        model = slr_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    elif model_name == "slr_s1":
        model = slr_s1_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    elif model_name == "cov_only_dsnt":
        model = cov_only_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    elif model_name == "u_net":
        model = u_net_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "u_net_dsnt":
        model = u_net_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    elif model_name == "scn":
        model = scn_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "scn_dsnt":
        model = scn_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    else:
        print("There is no model: ", model_name)

    return model
