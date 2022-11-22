from tensorflow import keras
import tensorflow.keras.layers as layers

from tensorflow import Tensor
# from keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
#    Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

from Preparation import functions
from scipy.ndimage import zoom


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

    if downsample:
        x = layers.Conv3D(kernel_size=1,
                          strides=2,
                          filters=filters,
                          padding="same")(x)
    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out


def spine_lateral_radiograph(width=170, height=170, depth=30):
    """
    The original model is for 2D image, our data are 3D.
    Change it to a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)

    #
    x = residual_block(x, downsample=False, filters=64)
    violet_x = residual_block(x, downsample=False, filters=64)

    x = residual_block(violet_x, downsample=False, filter=128)
    yellow_x = residual_block(x, downsample=False, filter=128)

    x = residual_block(yellow_x, downsample=True, filter=256)
    blue_x = residual_block(x, downsample=False, filter=256)

    x = residual_block(blue_x, downsample=True, filter=512)
    green_x = residual_block(x, downsample=False, filter=512)
    green_x = residual_block(green_x, downsample=False, filter=512)

    x = residual_block(green_x, downsample=False, filter=256)
    x = layers.UpSampling3D(size=2)(x)
    blue_x = layers.Add()([x, blue_x])

    x = residual_block(blue_x, downsample=False, filter=128)
    x = layers.UpSampling3D(size=2)(x)
    yellow_x = layers.Add()([x, yellow_x])

    x = residual_block(yellow_x, downsample=False, filter=64)
    # x = layers.UpSampling3D(size=2)(x)
    violet_x = layers.Add()([x, violet_x])

    x = residual_block(violet_x, downsample=False, filter=64)
    # grey_x = layers.UpSampling3D(size=2)(x)
    grey_x = x

    x = residual_block(grey_x, downsample=False, filter=45)
    x = residual_block(x, downsample=False, filter=45)
    heatmap_s1 = residual_block(x, downsample=False, filter=45)

    return model
