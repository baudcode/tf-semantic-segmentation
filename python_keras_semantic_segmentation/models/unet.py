from ..layers import get_norm_by_name

from tensorflow.keras import layers, Model, Input
from tensorflow.keras import backend as K

import tensorflow as tf
import math


def conv(x, filters, kernel_size=(3, 3), norm='batch', activation='relu', padding='SAME'):
    # print(locals())
    print(x.shape, filters)
    y = layers.Conv2D(filters, kernel_size=kernel_size, activation=activation, padding=padding)(x)
    if norm:
        y = get_norm_by_name(norm)(y)
    return y


def upsample(x):
    return layers.UpSampling2D(size=(2, 2))(x)


"""
def upsample(x):
    shape = K.shape(x)
    size = (shape[0] / 2, shape[1] / 2)
    print("upsample to %s" % str(size))

    return tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR)
"""


def downsample(x, f=2):
    return layers.MaxPool2D(pool_size=(2, 2))(x)


def unet(input_shape=(256, 256, 1), num_classes=3, depth=5, num_first_filters=32):
    """ 
        https://arxiv.org/pdf/1505.04597.pdf
    """
    inputs = Input(input_shape)

    y = inputs
    layers = []

    for i in range(depth):
        num_filters = int(pow(2, math.log2(num_first_filters) + i))
        y = conv(y, num_filters)
        y = conv(y, num_filters)
        layers.append(y)
        y = downsample(y)

    num_filters = int(pow(2, math.log2(num_first_filters) + depth))
    y = conv(y, num_filters)
    y = conv(y, num_filters)

    for i in reversed(range(depth)):
        y = upsample(y)
        y = K.concatenate([y, layers[i]])
        num_filters = int(pow(2, math.log2(num_first_filters) + i))
        y = conv(y, num_filters)
        y = conv(y, num_filters)

    y = conv(y, num_classes, activation=None, norm=None)
    return Model(inputs, y)
