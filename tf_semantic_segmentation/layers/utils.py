import tensorflow as tf
from .pixel_norm import PixelNorm


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
