from .unet import conv, upsample, downsample, logger
import math
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf


def attention(filters, x, g):
    # conv, norm on g with stride=1, kernel_size=1, no activation
    g2 = conv(g, filters, kernel_size=(1, 1), strides=(1, 1), activation=None, norm='batch')
    # conv norm on x
    x2 = conv(x, filters, kernel_size=(1, 1), strides=(1, 1), activation=None, norm='batch')

    psi = tf.nn.relu(x2 + g2)
    psi = conv(psi, 1, kernel_size=(1, 1), strides=(1, 1), activation=None, norm='batch')
    psi = tf.nn.sigmoid(psi)
    return x * psi


def attention_unet(input_shape=(256, 256, 1), num_classes=3, depth=5, activation='relu', num_first_filters=64, l2=None,
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
        # y = conv(y, num_filters, kernel_size=(2, 2), activation=activation, l2=l2, conv_type=conv_type)
        att = attention(num_filters, layers[-(k + 2)], y)
        y = K.concatenate([att, y])
        logger.debug("concat shape: %s" % str(y.shape))
        y = conv(y, num_filters, activation=activation, l2=l2, conv_type=conv_type)
        y = conv(y, num_filters, activation=activation, l2=l2, conv_type=conv_type)
        logger.debug("decoder - features: %d, shape: %s" % (num_filters, str(y.shape)))

    y = conv(y, num_classes, kernel_size=(1, 1), activation=None, norm=None)
    return Model(inputs, y)


if __name__ == "__main__":
    model = attention_unet()
    model.summary()
