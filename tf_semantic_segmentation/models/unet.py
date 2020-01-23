from ..layers import get_norm_by_name, Subpixel
from ..settings import logger

from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras import backend as K

import tensorflow as tf
import math


def conv(x, filters, kernel_size=(3, 3), norm='batch', activation='relu', l2=None, padding='SAME'):
    # print(locals())
    y = layers.Conv2D(filters, kernel_size=kernel_size,
                      kernel_regularizer=regularizers.l2(l2) if l2 else None,
                      activation=activation,
                      padding=padding)(x)
    if norm:
        y = get_norm_by_name(norm)(y)
    return y


def upsample(x, method='conv', kernel_size=(3, 3), padding='SAME', activation='relu'):
    num_filters = x.shape[-1]

    methods = {
        "nearest": layers.UpSampling2D(size=(2, 2)),
        "bilinear": layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
        "conv": layers.Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=(2, 2), padding=padding, activation=activation),
        "subpixel": Subpixel(num_filters, kernel_size, 2, padding='SAME')
    }

    if method in methods:
        print("in shape: ", x.shape)
        x = methods[method](x)
        print("out shape: ", x.shape)
        return x
    else:
        raise Exception("upsample method %s not found, please use one of %s" % (method, list(methods.keys())))


def downsample(x, method='max_pool', kernel_size=(1, 1), padding='SAME', activation='relu'):
    num_filters = x.shape[-1]

    methods = {
        "max_pool": layers.MaxPooling2D(pool_size=(2, 2)),
        "avg_pool": layers.AveragePooling2D(pool_size=(2, 2)),
        "conv": layers.Conv2D(num_filters, kernel_size=kernel_size, strides=(2, 2), padding=padding, activation=activation)
    }
    if method in methods:
        return methods[method](x)
    else:
        raise Exception("upsample method %s not found, please use one of %s" % (method, list(methods.keys())))
    return


def unet(input_shape=(256, 256, 1), num_classes=3, depth=5, activation='relu', num_first_filters=64):
    """ 
        https://arxiv.org/pdf/1505.04597.pdf
    """
    logger.debug("building model unet with args %s" % (str(locals())))
    inputs = Input(input_shape)

    y = inputs
    layers = []

    features = [int(pow(2, math.log2(num_first_filters) + i)) for i in range(depth)]

    for k, num_filters in enumerate(features):
        y = conv(y, num_filters, activation=activation)
        y = conv(y, num_filters, activation=activation)
        layers.append(y)

        if k != (len(features) - 1):
            y = downsample(y, method='max_pool')

        print("encoder - features: %d, shape: %s" % (num_filters, str(y.shape)))

    for k, num_filters in enumerate(reversed(features[:-1])):
        y = upsample(y, method='conv', activation=activation)
        y = K.concatenate([y, layers[-(k + 2)]])
        y = conv(y, num_filters, activation=activation)
        y = conv(y, num_filters, activation=activation)
        print("decoder - features: %d, shape: %s" % (num_filters, str(y.shape)))

    base_model = Model(inputs, y)
    y = conv(y, num_classes, kernel_size=(1, 1), activation=None, norm=None)
    return Model(inputs, y), base_model


def unet_v2(input_shape=(256, 256, 1), num_classes=2, activation='relu', upsampling_method='bilinear',
            downsampling_method='max_pool', depth=5, num_first_filters=32):
    """
    Modified version of the original unet with more convolutions and bilinear upsampling
    """
    logger.debug("building model unet_v2 with args %s" % (str(locals())))
    inputs = Input(input_shape)

    y = inputs
    layers = []

    features = [int(pow(2, math.log2(num_first_filters))) for i in range(depth)]

    for k, f in enumerate(features):
        print("encoder k=%d, features=%d" % (k, f))
        y = conv(y, f, activation=activation)
        y = conv(y, f, activation=activation)
        layers.append(y)
        if k != (len(features) - 1):
            y = downsample(y, method=downsampling_method, activation=activation)

    for k, f in enumerate(reversed(features[:-1])):
        print('decoder: k=%d, features=%d' % (k, f))
        y = upsample(y, method=upsampling_method, activation=activation)
        y = conv(y, f, kernel_size=(1, 1), norm=None, activation=activation)
        y = K.concatenate([layers[-(k + 2)], y])
        y = conv(y, f, norm=None, activation=activation)
        y = conv(y, f, activation=activation)

    y = conv(y, features[0], norm=False, activation=activation)
    y = conv(y, features[0], norm=False, activation=activation)

    base_model = Model(inputs, y)
    y = conv(y, num_classes, kernel_size=(1, 1), activation=None, norm=None)
    return Model(inputs, y), base_model


if __name__ == "__main__":
    from ..activations import *

    for upsample_method in ['subpixel', 'bilinear', 'conv', 'nearest']:
        print("trying usampling method %s" % upsample_method)
        unet_v2(upsampling_method=upsample_method, downsampling_method='conv', activation='mish')

    for downsample_method in ['max_pool', 'avg_pool', 'conv']:
        print("trying downsampling method %s" % downsample_method)
        unet_v2(upsampling_method='bilinear', downsampling_method=downsample_method, activation='mish')
