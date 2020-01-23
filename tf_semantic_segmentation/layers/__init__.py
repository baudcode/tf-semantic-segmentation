from .fire import Fire
from .minibatchstddev import MiniBatchStdDev
from .pixel_norm import PixelNorm
from .subpixel import Subpixel
from .mixconv import MixConv
from ..settings import logger

import tensorflow as tf


def get_norm_by_name(name):
    if name == 'batch':
        return tf.keras.layers.BatchNormalization(axis=-1)
    elif name == 'instance':
        import tensorflow_addons as tfa
        return tfa.layers.InstanceNormalization(axis=-1)
    elif name == 'pixel':
        return PixelNorm()
    else:
        raise Exception("unknown norm name %s" % name)


__all__ = ['Fire', "MiniBatchStdDev", "PixelNorm", "MixConv", "Subpixel", 'get_norm_by_name']
