# https://arxiv.org/pdf/1706.05587.pdf
# PASCAL VOC: 77.21 (val set)
# Cityscpes: 79.30 % IOU (val), 81.3 % (test)
# Crop Size: 513
# output stride=16
# eval output stride: 8

from tf_semantic_segmentation.models.apps.resnet50 import resnet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from ..layers import get_norm_by_name


def conv(x, filters, kernel_size=(3, 3), dilation=1, activation='relu', norm='batch'):
    y = layers.Conv2D(filters, kernel_size=kernel_size, dilation_rate=dilation, activation=None, padding='same')(x)
    y = get_norm_by_name(norm)(y)
    if activation:
        y = layers.Activation(activation)(y)
    return y


def atrous_spatial_pyramid_pooling(x, depth=256, norm='batch'):
    shape = x.shape

    # pooling
    pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    pool = conv(pool, depth, kernel_size=(1, 1))
    pool = tf.image.resize(pool, (shape[1], shape[2]))

    # 4 level pyramid
    l1 = conv(x, depth, (1, 1), norm=norm)
    l2 = conv(x, depth, (3, 3), dilation=6, norm=norm)
    l3 = conv(x, depth, (3, 3), dilation=12, norm=norm)
    l4 = conv(x, depth, (3, 3), dilation=18, norm=norm)

    # concat features and 1x1 conv
    y = tf.concat([l1, l2, l3, l4, pool], axis=-1)
    y = conv(y, depth, kernel_size=(1, 1))
    return y


def deeplabv3(input_shape=(512, 512, 3), num_classes=2, encoder_weights='imagenet'):

    base_model = resnet50(input_shape=input_shape, encoder_weights=encoder_weights)

    y = base_model.outputs[-3]
    y = atrous_spatial_pyramid_pooling(y)
    y = layers.Conv2D(num_classes, kernel_size=1)(y)

    # upsampling
    y = tf.image.resize(y, input_shape[:2])
    return Model(inputs=base_model.inputs, outputs=y)


if __name__ == "__main__":
    model = deeplabv3()
    model.summary()
