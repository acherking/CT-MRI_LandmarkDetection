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
    # c_0_v = [(2 * i - clown_size - 1) / clown_size for i in range(1, clown_size + 1)]
    c_0_v = [(i - clown_size/2) for i in range(1, clown_size + 1)]
    for i in range(clown_size):
        matrix_x[:, i, :] = matrix_x[:, i, :] * c_0_v[i]

    matrix_y = np.copy(base_array)
    # c_1_v = [(2 * i - row_size - 1) / row_size for i in range(1, row_size + 1)]
    c_1_v = [(i - row_size/2) for i in range(1, row_size + 1)]
    # c_1_v.reverse()
    for i in range(row_size):
        matrix_y[i, :, :] = matrix_y[i, :, :] * c_1_v[i]

    matrix_z = np.copy(base_array)
    # c_2_v = [(2 * i - slice_size - 1) / slice_size for i in range(1, slice_size + 1)]
    c_2_v = [(i - slice_size/2) for i in range(1, slice_size + 1)]
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


# input: heatmap
def dsnt_transfer(heatmap, base_cor_rcs, str_name):
    height = heatmap.shape[1]
    width = heatmap.shape[2]
    depth = heatmap.shape[3]
    points_num = heatmap.shape[4]

    pro_matrix = layers.Reshape((height, width, depth, points_num, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name=str_name)(heatmap), repeats=3, axis=-1))
    outputs = tf.math.reduce_sum(layers.multiply([base_cor_rcs, pro_matrix]), axis=[1, 2, 3])

    return outputs


# Cascaded Pyramid Network
def cpn(inputs, input_shape, points_num=2, dsnt=True):

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

    if dsnt:
        # S1 output
        x = residual_block(grey_x_s1, downsample=False, filters=points_num)
        x = residual_block(x, downsample=False, filters=points_num)
        heatmap_s1 = residual_block(x, downsample=False, filters=points_num)
        # S2 output
        x = residual_block(grey_x_s2, downsample=False, filters=points_num)
        x = residual_block(x, downsample=False, filters=points_num)
        heatmap_s2 = residual_block(x, downsample=False, filters=points_num)
    else:
        # low resolution heatmap
        # S1 output
        x = residual_block(grey_x_s1, downsample=False, filters=64)
        x = residual_block(x, downsample=True, filters=64)
        heatmap_s1 = residual_block(x, downsample=True, filters=64)
        # S2 output
        x = residual_block(grey_x_s2, downsample=False, filters=64)
        x = residual_block(x, downsample=True, filters=64)
        heatmap_s2 = residual_block(x, downsample=True, filters=64)

    return heatmap_s1, heatmap_s2


def cpn_fc_model(height=176, width=176, depth=48, points_num=2):

    inputs = keras.Input((height, width, depth, 1))
    input_shape = (height, width, depth)

    heatmap_s1, heatmap_s2 = cpn(inputs, input_shape, points_num, dsnt=False)

    x_hidden_s1 = layers.Dropout(0.2)(heatmap_s1)
    x_hidden_s1 = layers.Flatten()(x_hidden_s1)
    outputs_s1 = layers.Dense(units=3*points_num)(x_hidden_s1)
    outputs_s1 = layers.Reshape((points_num, 3))(outputs_s1)

    x_hidden_s2 = layers.Dropout(0.2)(heatmap_s2)
    x_hidden_s2 = layers.Flatten()(x_hidden_s2)
    outputs_s2 = layers.Dense(units=3*points_num)(x_hidden_s2)
    outputs_s2 = layers.Reshape((points_num, 3))(outputs_s2)

    model = keras.Model(inputs, [outputs_s1, outputs_s2], name="cpn-fc-model")

    return model


def cpn_dsnt_model(height=176, width=176, depth=48, points_num=2, batch_size=2):

    inputs = keras.Input((height, width, depth, 1))
    input_shape = (height, width, depth)

    heatmap_s1, heatmap_s2 = cpn(inputs, input_shape, points_num, dsnt=True)

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)
    output_s1 = dsnt_transfer(heatmap_s1, base_cor_rcs, "cpn-dsnt-s1")
    output_s2 = dsnt_transfer(heatmap_s1, base_cor_rcs, "cpn-dsnt-s2")

    model = keras.Model(inputs, [output_s1, output_s2], name="cpn-dsnt-model")

    return model


def cpn_s1(height=176, width=176, depth=48, points_num=2, batch_size=2):
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


# ResNet
def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same', strides = (2,2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv3D(filter, (3,3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=4)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv3D(filter, (1,1,1), strides = (2,2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def ResNet34(shape=(72, 72, 48, 1), points_num=4):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding3D((3, 3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv3D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling3D((2,2,2), padding = 'same')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    # x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    outputs = tf.keras.layers.Dense(units=points_num * 3, )(x)
    outputs = tf.keras.layers.Reshape((points_num, 3))(outputs)

    model = tf.keras.models.Model(inputs = x_input, outputs = outputs, name = "ResNet34")
    return model


def down_net_dynamic_model(hp, height=176, width=176, depth=48, points_num=4):

    inputs = keras.Input((height, width, depth, 1))
    # layer 1
    filter_num_l1 = hp.Int("filters_1", min_value=32, max_value=512, step=32)
    x_hidden = layers.Conv3D(filters=filter_num_l1, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    if hp.Boolean("max_pooling_1"):
        x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer ...
    for i in range(2, hp.Int("num_layers", 5, 12)):
        filter_num = hp.Int(f"filters_{i}", min_value=32, max_value=512, step=32)
        x_hidden = layers.Conv3D(filters=filter_num, kernel_size=3, padding="same")(x_hidden)
        x_hidden = layers.BatchNormalization()(x_hidden)
        x_hidden = layers.ReLU()(x_hidden)
        if x_hidden.shape[1] > 1 and x_hidden.shape[2] > 1 and x_hidden.shape[3] > 1:
            if hp.Boolean(f"max_pooling_{i}"):
                x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    if hp.Boolean(f"drop_out"):
        dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.3, step=0.05)
        x_hidden = layers.Dropout(dropout_rate)(x_hidden)

    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)
    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="dynamic-down-net")
    return model


def down_net_org(inputs, points_num=4, dsnt=False):
    # layer 1
    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 3
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 4
    x_hidden = layers.Conv3D(filters=64, kernel_size=1, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 5
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 6
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 7
    x_hidden = layers.Conv3D(filters=128, kernel_size=1, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 8
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 9
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 10
    x_hidden = layers.Conv3D(filters=256, kernel_size=1, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 11
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 12
    x_hidden = layers.Conv3D(filters=256, kernel_size=1, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 13
    x_hidden = layers.Conv3D(filters=512, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 14
    x_hidden = layers.Conv3D(filters=1024, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 15
    x_hidden = layers.Conv3D(filters=512, kernel_size=1, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 16
    x_hidden = layers.Conv3D(filters=1024, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 17
    x_hidden = layers.Conv3D(filters=512, kernel_size=1, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    # layer 18
    if dsnt:
        x_hidden = layers.Conv3D(filters=points_num, kernel_size=3, padding="same")(x_hidden)
    else:
        x_hidden = layers.Conv3D(filters=1024, kernel_size=3, padding="same")(x_hidden)
        x_hidden = layers.BatchNormalization()(x_hidden)
        x_hidden = layers.LeakyReLU(alpha=0.1)(x_hidden)

    return x_hidden


def down_net_org_model(height=176, width=176, depth=48, points_num=4):
    inputs = keras.Input((height, width, depth, 1))
    down_net_features = down_net_org(inputs, points_num=points_num, dsnt=False)

    x_hidden = layers.Conv3D(filters=512, kernel_size=1, padding="same")(down_net_features)
    x_hidden = layers.GlobalAveragePooling3D()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)
    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-cnn-org")
    return model


def down_net(inputs, points_num=4, dsnt=False):
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
    if dsnt:
        x_hidden = layers.Conv3D(filters=points_num, kernel_size=3, padding="same")(x_hidden)
    else:
        x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
        x_hidden = layers.BatchNormalization()(x_hidden)
        x_hidden = layers.ReLU()(x_hidden)

    return x_hidden


def down_net_model(height=176, width=176, depth=48, points_num=4):
    inputs = keras.Input((height, width, depth, 1))
    down_net_features = down_net(inputs, points_num=points_num, dsnt=False)

    x_hidden = layers.Dropout(0.2)(down_net_features)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)
    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-cnn")
    return model


def down_net_dsnt_model(height=176, width=176, depth=48, points_num=2, batch_size=2):
    inputs = keras.Input((height, width, depth, 1))

    heatmap = down_net(inputs, points_num, dsnt=True)

    base_cor_rcs = coordinate_3d(batch_size, points_num, heatmap.shape[1], heatmap.shape[2], heatmap.shape[3])
    outputs = dsnt_transfer(heatmap, base_cor_rcs, "down-net-dsnt")

    model = keras.Model(inputs, outputs, name="down-net-dsnt-model")

    return model


def down_net_short(inputs, points_num=4, dsnt=False):
    # layer 1
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 3
    x_hidden = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    # layer 4
    x_hidden = layers.Conv3D(filters=256, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)

    if dsnt:
        x_hidden = layers.Conv3D(filters=points_num, kernel_size=1, padding="same")(x_hidden)

    return x_hidden


def down_net_short_model(height=176, width=176, depth=48, points_num=4):
    inputs = keras.Input((height, width, depth, 1))
    down_net_features = down_net_short(inputs, points_num=points_num, dsnt=False)

    x_hidden = layers.Flatten()(down_net_features)
    x_hidden = layers.Dense(units=512, )(x_hidden)
    x_hidden = layers.Dropout(0.2)(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="down-net-short")
    return model


def down_net_short_dsnt_model(height=176, width=176, depth=48, points_num=2, batch_size=2):
    inputs = keras.Input((height, width, depth, 1))

    heatmap = down_net_short(inputs, points_num, dsnt=True)

    base_cor_rcs = coordinate_3d(batch_size, points_num, heatmap.shape[1], heatmap.shape[2], heatmap.shape[3])
    outputs = dsnt_transfer(heatmap, base_cor_rcs, "down-net-short-dsnt")

    model = keras.Model(inputs, outputs, name="down-net-short-dsnt-model")

    return model


def straight_model_mini(height=176, width=176, depth=48, points_num=4):
    inputs = keras.Input((height, width, depth, 1))

    # layer 1
    x_hidden = layers.Conv3D(filters=8, kernel_size=3, padding="same")(inputs)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=16, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 3
    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 4
    x_hidden = layers.Conv3D(filters=16, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 5
    x_hidden = layers.Conv3D(filters=32, kernel_size=3, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    x_hidden = layers.Dropout(0.2)(x_hidden)
    x_hidden = layers.Flatten()(x_hidden)
    outputs = layers.Dense(units=points_num * 3, )(x_hidden)

    outputs = layers.Reshape((points_num, 3))(outputs)

    # Define the model.
    model = keras.Model(inputs, outputs, name="straight-3d-mini")

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
def cov_only(inputs, kernel_size, points_num):

    # layer 1
    x_hidden = layers.Conv3D(filters=32, kernel_size=kernel_size, padding="same")(inputs)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 2
    x_hidden = layers.Conv3D(filters=64, kernel_size=kernel_size, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 3
    x_hidden = layers.Conv3D(filters=128, kernel_size=kernel_size, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 4
    x_hidden = layers.Conv3D(filters=64, kernel_size=kernel_size, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)

    # layer 5
    x_hidden = layers.Conv3D(filters=128, kernel_size=kernel_size, padding="same")(x_hidden)
    x_hidden = layers.BatchNormalization()(x_hidden)
    x_hidden = layers.ReLU()(x_hidden)
    # x_hidden = layers.MaxPool3D(pool_size=2)(x_hidden)

    # layer 6
    x = layers.Conv3D(filters=64, kernel_size=kernel_size, padding="same")(x_hidden)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # layer 7
    heatmap = residual_block(x, downsample=False, filters=points_num)

    return heatmap


def cov_only_dsnt_model(height=176, width=176, depth=48, kernel_size=3, points_num=2, batch_size=2):
    inputs = keras.Input((height, width, depth, 1))

    heatmap = cov_only(inputs, kernel_size, points_num)

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)
    outputs = dsnt_transfer(heatmap, base_cor_rcs, "cov-only-dsnt")

    model = keras.Model(inputs, outputs, name="cov-only-dsnt-model")

    return model


def cov_only_fc_model(height=176, width=176, depth=48, kernel_size=3, points_num=2):
    inputs = keras.Input((height, width, depth, 1))

    heatmap = cov_only(inputs, kernel_size, points_num)

    reg = layers.MaxPool3D(4)(heatmap)
    x_hidden = layers.Dropout(0.2)(reg)
    x_hidden = layers.Flatten()(x_hidden)

    outputs = layers.Dense(units=3*points_num)(x_hidden)
    outputs = layers.Reshape((points_num, 3))(outputs)

    model = keras.Model(inputs, outputs, name="cov-only-fc-model")

    return model


# U-Net: "End-to-End Coordinate Regression Model with Attention-Guided Mechanism
# for Landmark Localization in 3D Medical Images"
# not from the origin
# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
def double_conv_block(x, n_filters, with_bn=False):
    # Conv3D then ReLU activation
    x = layers.Conv3D(n_filters, 3, padding="same", kernel_initializer="he_normal")(x)
    if with_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Conv3D then ReLU activation
    x = layers.Conv3D(n_filters, 3, padding="same", kernel_initializer="he_normal")(x)
    if with_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def downsample_block(x, n_filters, with_bn=False):
    f = double_conv_block(x, n_filters, with_bn)
    p = layers.MaxPool3D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block(x, conv_features, n_filters, with_bn=False, up_sample_fn=0):
    # upsample
    if up_sample_fn == 0:
        x = layers.Conv3DTranspose(n_filters, 3, 2, padding="same")(x)
    elif up_sample_fn == 1:
        x = layers.UpSampling3D(2)(x)
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
    x = double_conv_block(x, n_filters, with_bn)

    return x


def u_net_mini(inputs, points_num=2, dsnt=False):

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)

    # 0 - bottleneck
    bottleneck = double_conv_block(p3, 512)

    # decoder: expanding path - upsample
    # 4 - upsample
    u4 = upsample_block(bottleneck, f3, 256)
    # 5 - upsample
    u5 = upsample_block(u4, f2, 128)
    # 6 - upsample
    u6 = upsample_block(u5, f1, 64)

    if dsnt:
        heatmaps = layers.Conv3D(points_num, 1, padding="same")(u6)
    else:
        heatmaps = layers.Conv3D(3, 3, strides=4, activation="relu")(u6)

    return heatmaps


def u_net_mini_bn(inputs, points_num=2, dsnt=False):

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64, True)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128, True)

    # 3 - bottleneck
    bottleneck = double_conv_block(p2, 256, True)

    # decoder: expanding path - upsample
    # 4 - upsample
    u4 = upsample_block(bottleneck, f2, 128, True)
    # 5 - upsample
    u5 = upsample_block(u4, f1, 64, True)

    if dsnt:
        heatmaps = layers.Conv3D(points_num, 1, padding="same", activation="softmax")(u5)
    else:
        heatmaps = layers.Conv3D(3, 3, strides=4, activation="relu")(u5)

    return heatmaps


def u_net_mini_upsample_rep(inputs, points_num=2, dsnt=False):

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64, True)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128, True)

    # 3 - bottleneck
    bottleneck = double_conv_block(p2, 256, True)

    # decoder: expanding path - upsample
    # 4 - upsample
    u4 = upsample_block(bottleneck, f2, 128, True, 1)
    # 5 - upsample
    u5 = upsample_block(u4, f1, 64, True, 1)

    if dsnt:
        heatmaps = layers.Conv3D(points_num, 1, padding="same", activation="softmax")(u5)
    else:
        heatmaps = layers.Conv3D(3, 3, strides=4, activation="relu")(u5)

    return heatmaps


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
        # heatmaps = layers.Conv3D(points_num, 1, padding="same", activation="softmax")(u9)
        heatmaps = layers.Conv3D(points_num, 1, padding="same")(u9)
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


def u_net_dsnt_model(height=176, width=176, depth=48, points_num=2, batch_size=2, u_net_id=0):
    inputs = keras.Input((height, width, depth, 1))

    if u_net_id == 0:
        heatmaps = u_net(inputs, points_num, dsnt=True)
    elif u_net_id == 1:
        heatmaps = u_net_mini(inputs, points_num, dsnt=True)
    elif u_net_id == 2:
        heatmaps = u_net_mini_bn(inputs, points_num, dsnt=True)
    elif u_net_id == 3:
        heatmaps = u_net_mini_upsample_rep(inputs, points_num, dsnt=True)
    else:
        print("u_net_dsnt_model error, unknown u_net_id: ", u_net_id)
        exit(0)

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)

    pro_matrix = layers.Reshape((height, width, depth, points_num, 3)) \
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


# upsampling padding function
def upsampling_padding(input_shape, downsampling_factor, features_downsampled):
    paddings = input_shape % downsampling_factor
    border_paddings = np.floor(paddings/2).astype("int")
    end_paddings = paddings % 2
    sum_padding = (
        (border_paddings[0], border_paddings[0]+end_paddings[0]),
        (border_paddings[1], border_paddings[1]+end_paddings[1]),
        (border_paddings[2], border_paddings[2]+end_paddings[2]))

    upsample_no_padding = layers.UpSampling3D(downsampling_factor, name='spatial_prediction')(features_downsampled)
    upsample_padding = layers.ZeroPadding3D(sum_padding)(upsample_no_padding)
    return upsample_padding


# Spatial Configuration Net: "Regressing Heatmaps for Multiple Landmark Localization Using CNNs"
# https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression/blob/master/hand_xray/network.py
def sc_net(inputs, points_num, dsnt=False, local_kernel_size=(5, 5, 5), spatial_kernel_size=(9, 9, 5)):
    num_filters = 128
    local_kernel_size = local_kernel_size
    spatial_kernel_size = spatial_kernel_size
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
        # spatial_heatmaps = layers.UpSampling3D(downsampling_factor, name='spatial_prediction')(spatial_heatmaps_downsampled)
        spatial_heatmaps = upsampling_padding(np.asarray(inputs.shape[1:-1]), downsampling_factor, spatial_heatmaps_downsampled)
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


def scn_dsnt_model(height=176, width=176, depth=48, points_num=2, local_kernel_size=(3, 3, 3), spatial_kernel_size=(9, 9, 5), batch_size=2):
    inputs = keras.Input((height, width, depth, 1))

    dsnt_tag = True
    heatmaps = sc_net(inputs, points_num, dsnt_tag, local_kernel_size, spatial_kernel_size)

    base_cor_rcs = coordinate_3d(batch_size, points_num, height, width, depth)

    pro_matrix = layers.Reshape((height, width, depth, points_num, 3)) \
        (tf.repeat(layers.Softmax(axis=[1, 2, 3], name="softmax")(heatmaps), repeats=3, axis=-1))
    outputs = tf.math.reduce_sum(layers.multiply([base_cor_rcs, pro_matrix]), axis=[1, 2, 3])

    # unet model with Keras Functional API
    model = tf.keras.Model(inputs, outputs, name="scn_dsnt_model")

    return model


## ViT
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches3D(layers.Layer):
    def __init__(self, patch_size, patch_overlap):
        super().__init__()
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.extract_volume_patches(
            input=images,
            ksizes=[1, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1],
            strides=[1, self.patch_size[0]-self.patch_overlap[0], self.patch_size[1]-self.patch_overlap[1],
                     self.patch_size[2]-self.patch_overlap[2], 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class ConvStem(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_layers = keras.Sequential(
            [
                layers.Conv3D(kernel_size=3, strides=2, filters=48, padding="same"),
                layers.Conv3D(kernel_size=3, strides=2, filters=96, padding="same"),
                layers.Conv3D(kernel_size=3, strides=2, filters=192, padding="same"),
                layers.Conv3D(kernel_size=3, strides=1, filters=384, padding="same"),
                layers.Conv3D(kernel_size=1, strides=1, filters=self.dim, padding="same")
            ]
        )
        # self.conv_0 = layers.Conv3D(kernel_size=3, strides=2, filters=48, padding="same")
        # self.conv_1 = layers.Conv3D(kernel_size=3, strides=2, filters=96, padding="same")
        # self.conv_2 = layers.Conv3D(kernel_size=3, strides=2, filters=192, padding="same")
        # self.conv_3 = layers.Conv3D(kernel_size=3, strides=2, filters=384, padding="same")

    def call(self, images):
        batch_size = tf.shape(images)[0]
        conv_x = self.conv_layers(images)

        patch_size = conv_x.shape[1] * conv_x.shape[2] * conv_x.shape[3]
        patches = tf.reshape(conv_x, [batch_size, patch_size, self.dim])

        return patches


class PatchEncoder3D(layers.Layer):
    def __init__(self, num_patches, projection_dim, if_project):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.if_project = if_project
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        if self.if_project:
            encoded = self.projection(patch) + self.position_embedding(positions)
        else:
            encoded = patch + self.position_embedding(positions)
        return encoded


def create_vit3D_regression(patch_size, patch_overlap, num_patches, projection_dim, transformer_layers,
                            num_heads, transformer_units, mlp_head_units, volume_shape, points_num):
    inputs = layers.Input(shape=(volume_shape[0], volume_shape[1], volume_shape[2], 1))
    # Create patches.
    # patches = Patches3D(patch_size, patch_overlap)(inputs)
    patches = ConvStem(projection_dim)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder3D(num_patches, projection_dim, if_project=False)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # representation = layers.Flatten()(representation)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    # logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    # model = keras.Model(inputs=inputs, outputs=logits)
    # Regression ouputs
    reg = layers.Dense(units=points_num * 3, )(features)
    reg = layers.Reshape((points_num, 3))(reg)
    model = keras.Model(inputs=inputs, outputs=reg)
    return model


def model_manager(args_dict):
    model_name = args_dict["model_name"]
    input_shape = args_dict["input_shape"]
    cut_layers = args_dict["cut_layers"]
    dataset_tag = args_dict.get("dataset_tag")
    if dataset_tag == "cropped":
        cut_layers = np.asarray(args_dict.get("cut_layers"))
        if not np.all(cut_layers == 0):
            input_shape = (input_shape[0]-cut_layers[0][0] - cut_layers[0][1],
                           input_shape[1]-cut_layers[1][0] - cut_layers[1][1],
                           input_shape[2]-cut_layers[2][0] - cut_layers[2][1])
    model_output_num = args_dict["model_output_num"]
    batch_size = args_dict["batch_size"]

    if model_name == "straight_model":
        model = down_net_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "down_net_org":
        model = down_net_org_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "down_net_dynamic":
        hp = args_dict["hp"]
        model = down_net_dynamic_model(hp, input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "down_net_dsnt":
        model = down_net_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    elif model_name == "down_net_short":
        model = down_net_short_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "down_net_short_dsnt":
        model = down_net_short_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    elif model_name == "down_net_mini":
        model = straight_model_mini(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "res_net":
        model = ResNet34((input_shape[0], input_shape[1], input_shape[2], 1), model_output_num)
    elif model_name == "cpn_fc_model":
        model = cpn_fc_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "cpn_dsnt_model":
        model = cpn_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    elif model_name == "slr_s1":
        model = cpn_s1(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size)
    elif model_name == "cov_only_dsnt":
        kernel_size = 5
        model = cov_only_dsnt_model(input_shape[0], input_shape[1], input_shape[2],
                                    kernel_size, model_output_num, batch_size)
    elif model_name == "cov_only_fc":
        kernel_size = 5
        model = cov_only_fc_model(input_shape[0], input_shape[1], input_shape[2], kernel_size, model_output_num)
    elif model_name == "u_net":
        model = u_net_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "u_net_dsnt":
        model = u_net_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size, 0)
    elif model_name == "u_net_mini_dsnt":
        model = u_net_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size, 1)
    elif model_name == "u_net_mini_bn_dsnt":
        model = u_net_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size, 2)
    elif model_name == "u_net_mini_upsample_rep_dsnt":
        model = u_net_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, batch_size, 3)
    elif model_name == "scn":
        model = scn_model(input_shape[0], input_shape[1], input_shape[2], model_output_num)
    elif model_name == "scn_dsnt":
        local_ks = args_dict.get("local_kernel_size", (3, 3, 3))
        spatial_ks = args_dict.get("spatial_kernel_size", (9, 9, 5))
        model = scn_dsnt_model(input_shape[0], input_shape[1], input_shape[2], model_output_num, local_ks, spatial_ks, batch_size)
    elif model_name == "vit_C":
        image_size = (72, 72, 48)
        patch_size = (6, 6, 4)
        patch_overlap = (2, 2, 2)
        # num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) * (image_size[2] // patch_size[2])
        num_patches = 486
        projection_dim = 256
        num_heads = 4
        transformer_units = [
            projection_dim * 2,
            projection_dim,
            ]  # Size of the transformer layers
        transformer_layers = 8
        mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
        model = create_vit3D_regression(patch_size, patch_overlap, num_patches, projection_dim, transformer_layers, num_heads,
                                        transformer_units, mlp_head_units, input_shape, model_output_num)
    else:
        print("There is no model: ", model_name)
        exit(1)

    return model
