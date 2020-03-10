from ..layers import get_norm_by_name, Subpixel, Fire
from ..settings import logger

from tensorflow.keras import layers, Model, Input, regularizers
from tensorflow.keras import backend as K

import tensorflow as tf
import math


def conv(x, filters, kernel_size=(3, 3), strides=(1, 1), norm='batch', activation='relu', l2=None, padding='SAME', conv_type='conv'):
    # print(locals())
    if conv_type == 'conv':
        y = layers.Conv2D(filters, kernel_size=kernel_size,
                          strides=strides,
                          kernel_regularizer=regularizers.l2(l2) if l2 else None,
                          activation=activation,
                          padding=padding)(x)
    elif conv_type == 'fire':
        y = Fire(filters // 4, filters, padding=padding, activation=activation, norm=norm)(x)

    if norm:
        y = get_norm_by_name(norm)(y)
    return y


def upsample(x, method='conv', kernel_size=(3, 3), padding='SAME', activation='relu', l2=None):
    num_filters = x.shape[-1]

    methods = {
        "nearest": layers.UpSampling2D(size=(2, 2)),
        "bilinear": layers.UpSampling2D(size=(2, 2), interpolation='bilinear'),
        "conv": layers.Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=(2, 2), padding=padding,
                                       activation=activation, kernel_regularizer=regularizers.l2(l2) if l2 else None),
        "subpixel": Subpixel(num_filters, kernel_size, 2, padding='SAME')
    }

    if method in methods:
        x = methods[method](x)
        return x
    else:
        raise Exception("upsample method %s not found, please use one of %s" % (method, list(methods.keys())))


def downsample(x, method='max_pool', kernel_size=(1, 1), padding='SAME', activation='relu', l2=None):
    num_filters = x.shape[-1]

    methods = {
        "max_pool": layers.MaxPooling2D(pool_size=(2, 2)),
        "avg_pool": layers.AveragePooling2D(pool_size=(2, 2)),
        "conv": layers.Conv2D(num_filters, kernel_size=kernel_size, strides=(2, 2), padding=padding, activation=activation,
                              kernel_regularizer=regularizers.l2(l2) if l2 else None)
    }
    if method in methods:
        return methods[method](x)
    else:
        raise Exception("upsample method %s not found, please use one of %s" % (method, list(methods.keys())))
    return


def unet(input_shape=(256, 256, 1), num_classes=3, depth=5, activation='relu', num_first_filters=64, l2=None,
         upsampling_method='conv', downsampling_method='max_pool', conv_type='conv'):
    """ 
        https://arxiv.org/pdf/1505.04597.pdf
    """
    logger.debug("building model unet with args %s" % (str(locals())))
    inputs = Input(input_shape)

    y = inputs
    layers = []

    features = [int(pow(2, math.log2(num_first_filters) + i)) for i in range(depth)]

    for k, num_filters in enumerate(features):
        y = conv(y, num_filters, activation=activation, l2=l2, conv_type=conv_type)
        y = conv(y, num_filters, activation=activation, l2=l2, conv_type=conv_type)
        layers.append(y)

        if k != (len(features) - 1):
            y = downsample(y, method=downsampling_method, activation=activation, l2=l2)

        logger.debug("encoder - features: %d, shape: %s" % (num_filters, str(y.shape)))

    for k, num_filters in enumerate(reversed(features[:-1])):
        y = upsample(y, method=upsampling_method, activation=activation, l2=l2)
        y = conv(y, num_filters, kernel_size=(2, 2), activation=activation, l2=l2, conv_type=conv_type)
        y = K.concatenate([layers[-(k + 2)], y])
        logger.debug("concat shape: %s" % str(y.shape))
        y = conv(y, num_filters, activation=activation, l2=l2, conv_type=conv_type)
        y = conv(y, num_filters, activation=activation, l2=l2, conv_type=conv_type)
        logger.debug("decoder - features: %d, shape: %s" % (num_filters, str(y.shape)))

    y = conv(y, num_classes, kernel_size=(1, 1), activation=None, norm=None)
    return Model(inputs, y)


if __name__ == "__main__":
    from ..activations import *
    import logging

    logger.setLevel(logging.INFO)

    # tf.keras.utils.plot_model(unet(upsampling_method='bilinear', downsampling_method='conv')[0], to_file='unet.png', show_shapes=True)
    # tf.keras.utils.plot_model(unet_v2(upsampling_method='bilinear', downsampling_method='conv')[0], to_file='unetv2.png', show_shapes=True)

    for upsample_method in ['subpixel', 'bilinear', 'conv', 'nearest']:
        model = unet(upsampling_method=upsample_method, downsampling_method='conv', activation='mish', conv_type='fire')
        print("params: ", model.count_params(), 'upsample method: ', upsample_method)

    for downsample_method in ['max_pool', 'avg_pool', 'conv']:
        model = unet(upsampling_method='bilinear', downsampling_method=downsample_method, activation='mish')
        print("params: ", model.count_params(), 'downsample method: ', downsample_method)
