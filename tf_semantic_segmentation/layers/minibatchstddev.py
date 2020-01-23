"""
Code taken from https://github.com/johnryh/Face_Embedding_GAN/blob/master/network_utility.py
"""

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class MiniBatchStdDev(Layer):
    """
    https://arxiv.org/abs/1710.10196

    It computes the standard deviations of the 
    feature map pixels across the batch, and appends them as an extra channel.
    """

    def __init__(self, group_size=4):
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size

    def call(self, x, training=None):
        group_size = K.minimum(self.group_size, K.shape(x)[0])  # Minibatch must be divisible by (or smaller than) group_size.
        s = K.shape(x)                                             # [NCHW]  Input shape.
        y = K.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
        y = K.cast(y, 'float32')                              # [GMCHW] Cast to FP32.
        y -= K.mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = K.mean(K.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = K.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = K.mean(y, axis=[1, 2, 3], keepdims=True)    # [M111]  Take average over fmaps and pixels.
        y = K.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = K.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
        return K.concatenate([x, y], axis=-1)

    def get_config(self):
        config = {
            "group_size": self.group_size,
        }
        base_config = super(MiniBatchStdDev, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
