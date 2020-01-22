
"""
Code is adopted from https://github.com/divelab/dilated/blob/master/dilated.py
Transformed functions to keras layers, works on latest tensorflow version.

This scripts covers the dilated convolutions proposed in this paper
https://www.kdd.org/kdd2018/accepted-papers/view/smoothed-dilated-convolutions-for-improved-dense-prediction"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
import six
import numpy as np


class DilatedConv(Layer):

    """
    Regular dilated conv2d without BN or relu.
    """

    def __init__(self, filters, kernel_size, dilation_factor, biased=False, initializer=None):
        super(DilatedConv, self).__init__()
        self.__name__ = "DilatedConv"
        self.initializer = initializer
        self.filters = filters
        if type(kernel_size) != tuple and type(kernel_size) != list:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.dilation_factor = dilation_factor
        self.biased = biased

    def call(self, x, training=None):
        cur_filters = x.shape[3]
        w = self._get_variable([self.kernel_size[0], self.kernel_size[1], cur_filters, self.filters], 'weights')
        o = tf.nn.atrous_conv2d(x, w, self.dilation_factor, padding='SAME')
        if self.biased:
            b = self._get_variable([self.filters], 'bias')
            o = tf.nn.bias_add(o, b)
        return o

    def _get_variable(self, shape, name, initializer=None):
        if initializer is not None:
            init = initializer
        elif self.initializer is not None:
            init = self.initializer
        else:
            init = tf.initializers.GlorotUniform()

        var = tf.Variable(init(shape=shape))
        return var

    def get_config(self):

        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_factor': self.dilation_factor,
            'biased': self.biased
        }
        base_config = super(DilatedConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecomposedDilatedConv(DilatedConv):
    """
    Regular dilated conv2d without BN or relu.
    """

    def call(self, x, training=False):
        H, W = x.shape[1:3]

        # padding so that the input dims are multiples of dilation_factor
        pad_bottom = (self.dilation_factor - H % self.dilation_factor) if H % self.dilation_factor != 0 else 0
        pad_right = (self.dilation_factor - W % self.dilation_factor) if W % self.dilation_factor != 0 else 0
        pad = [[0, pad_bottom], [0, pad_right]]

        # decomposition to smaller-sized feature maps
        # [N,H,W,C] -> [N*d*d, H/d, W/d, C]
        o = tf.compat.v1.space_to_batch(x, block_size=self.dilation_factor, paddings=pad)
        # perform regular conv2d
        cur_filters = x.shape[3]

        w = self._get_variable([self.kernel_size[0], self.kernel_size[1], cur_filters, self.filters], 'weights')
        s = [1, 1, 1, 1]
        o = tf.nn.conv2d(o, w, s, padding='SAME')
        if self.biased:
            b = self._get_variable([self.filters], 'bias')
            o = tf.nn.bias_add(o, b)

        o = tf.compat.v1.batch_to_space(o, block_size=self.dilation_factor, crops=pad)
        return o


class SmooothGIDilatedConv(DilatedConv):

    """
    Smoothed dilated conv2d via the Group Interaction (GI) layer without BN or relu.
    """

    def call(self, x, training=False):
        H, W = x.shape[1:3]

        # padding so that the input dims are multiples of dilation_factor
        pad_bottom = (self.dilation_factor - H % self.dilation_factor) if H % self.dilation_factor != 0 else 0
        pad_right = (self.dilation_factor - W % self.dilation_factor) if W % self.dilation_factor != 0 else 0
        pad = [[0, pad_bottom], [0, pad_right]]
        # decomposition to smaller-sized feature maps
        # [N,H,W,C] -> [N*d*d, H/d, W/d, C]
        o = tf.compat.v1.space_to_batch(x, paddings=pad, block_size=self.dilation_factor)
        # perform regular conv2d
        cur_filters = x.shape[3]

        # variable scope begins
        w = self._get_variable([self.kernel_size[0], self.kernel_size[1], cur_filters, self.filters], 'weights')
        s = [1, 1, 1, 1]
        o = tf.nn.conv2d(o, w, s, padding='SAME')
        fix_w = tf.Variable(tf.eye(self.dilation_factor * self.dilation_factor), name='fix_w')
        l = tf.split(o, self.dilation_factor * self.dilation_factor, axis=0)
        os = []
        for i in six.moves.range(0, self.dilation_factor * self.dilation_factor):
            os.append(fix_w[0, i] * l[i])
            for j in six.moves.range(1, self.dilation_factor * self.dilation_factor):
                os[i] += fix_w[j, i] * l[j]
        o = tf.concat(os, axis=0)
        if self.biased:
            b = self._get_variable([cur_filters], 'biases')
            o = tf.nn.bias_add(o, b)

        o = tf.compat.v1.batch_to_space(o, block_size=self.dilation_factor, crops=pad)
        return o


class SmoothSSCDilatedConv(DilatedConv):

    """
    Smoothed dilated conv2d via the Separable and Shared Convolution (SSC) without BN or relu.
    """

    def call(self, x, training=False):
        cur_filters = x.shape[3]
        fix_w_size = self.dilation_factor * 2 - 1

        fix_w = self._get_variable([fix_w_size, fix_w_size, 1, 1, 1], 'fix_w', initializer=tf.initializers.Zeros())
        mask = np.zeros([fix_w_size, fix_w_size, 1, 1, 1], dtype=np.float32)
        mask[self.dilation_factor - 1, self.dilation_factor - 1, 0, 0, 0] = 1

        fix_w = tf.add(fix_w, tf.constant(mask, dtype=tf.float32))

        o = tf.expand_dims(x, -1)
        o = tf.nn.conv3d(o, fix_w, strides=[1, 1, 1, 1, 1], padding='SAME')
        o = tf.squeeze(o, -1)
        w = self._get_variable([self.kernel_size[0], self.kernel_size[0], cur_filters, self.filters], 'weights')
        o = tf.nn.atrous_conv2d(o, w, self.dilation_factor, padding='SAME')
        if self.biased:
            b = self._get_variable([cur_filters], 'biases')
            o = tf.nn.bias_add(o, b)
        return o


if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    from ..utils import get_random_image
    from ..visualizations import show

    image = get_random_image(width=64, height=64)
    image = tf.convert_to_tensor(image, tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)

    image_batch = tf.expand_dims(image, axis=0)

    dilated = DilatedConv(3, (3, 3), 5, biased=True)(image_batch)
    dilated = DecomposedDilatedConv(3, (3, 3), 4, biased=True)(dilated)
    dilated = SmooothGIDilatedConv(3, (3, 3), 8, biased=True)(dilated)
    dilated = SmoothSSCDilatedConv(3, (3, 3), 8, biased=True)(dilated)
    print(image_batch.shape, dilated.shape)

    dilated = dilated.numpy()
    image_batch = image_batch.numpy()
    print(dilated.max(), dilated.min())
    show.show_images([image_batch[0, :, :, :], dilated[0, :, :, :]])
