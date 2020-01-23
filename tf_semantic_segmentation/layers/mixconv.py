from tensorflow.keras.layers import Conv2D, Add, Concatenate, Layer, DepthwiseConv2D
from tensorflow.keras import backend as K
import tensorflow as tf

# TODO: check if mixConv is working


class MixConv(Layer):

    """ https://medium.com/@lessw/meet-mixnet-google-brains-new-state-of-the-art-mobile-ai-architecture-bd6c37abfa3a """

    def __init__(self, filters, **args):
        self.filters = filters
        self.args = args

    def call(self, x, training=None):

        G = len(self.filters)
        y = []
        for xi, fi in zip(tf.split(x, G, axis=-1), self.filters):
            o = DepthwiseConv2D(fi, **self.args)(xi)
            y.append(o)
        return Concatenate(axis=-1)(y)

    def get_config(self):

        config = {
            'filters': self.filters,
        }
        base_config = super(MixConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
