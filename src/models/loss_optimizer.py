import math

import tensorflow as tf
from tensorflow import keras


## loss

def loss_manager(args_dict):
    loss_name = args_dict['loss_name']

    if loss_name == 'MSE_res':
        loss_fn = mse_with_res
    elif loss_name == 'MSE':
        loss_fn = mse
    else:
        print("Unknown loss function", loss_name)
        exit(0)

    return loss_fn


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


def mse(y_true, y_pred, res):
    """
    :param y_true: [batch_size, num_landmarks, dimension(column, row, slice)]
    :param y_pred: [batch_size, num_landmarks, dimension(column, row, slice)]
    :return: mean square error along batch_size (mm^2)
    """
    with tf.name_scope('mse_res_loss'):
        err_diff = y_true - y_pred
        err_diff_p2 = tf.pow(err_diff, 2)
        mse_loss = tf.reduce_mean(tf.reduce_sum(err_diff_p2, axis=[1, 2]), axis=0)
        return mse_loss


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
        err_dis = err_diff * rep_res
        err_dis_p2 = tf.pow(err_dis, 2)
        mse_res_loss = tf.reduce_mean(tf.reduce_sum(err_dis_p2, axis=[1, 2]), axis=0)
        return mse_res_loss


# optimizer

def optimizer_manager(args_dict):
    optimizer_name = args_dict.get("optimizer")
    initial_learning_rate = args_dict.get("learning_rate")
    decay_steps = args_dict.get("decay_steps")
    decay_rate = args_dict.get("decay_rate")

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

    if optimizer_name == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        print("Unknown optimizer: {}".format(optimizer_name))
        exit(0)

    return optimizer
