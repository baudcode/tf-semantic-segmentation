from .fire import Fire
from .minibatchstddev import MiniBatchStdDev
from .pixel_norm import PixelNorm
from .subpixel import Subpixel

import tensorflow as tf
import tensorflow_addons as tfa


def get_norm_by_name(name):
    if name == 'batch':
        return tf.keras.layers.BatchNormalization(axis=-1)
    elif name == 'instance':
        return tfa.layers.InstanceNormalization(axis=-1)
    else:
        raise Exception("unknown norm name %s" % name)


__all__ = ['Fire', "MiniBatchStdDev", "PixelNorm", "Subpixel", 'get_norm_by_name']
