from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
import tensorflow as tf


class Mish(Activation):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(lambda x: x * K.tanh(K.softplus(x)), **kwargs)
        self.__name__ = "Mish"


class ReLU6(Activation):

    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(lambda x: K.maximum(x, 6), **kwargs)
        self.__name__ = 'ReLU6'


class Swish(Activation):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(lambda x: x * K.sigmoid(x), **kwargs)
        self.__name__ = "Switch"


class LeakyReLU(Activation):

    def __init__(self, alpha=0.2, **kwargs):

        super(LeakyReLU, self).__init__(lambda x: tf.nn.leaky_relu(x, alpha=alpha), **kwargs)
        self.__name__ = "LeakyReLU"


custom_objects = {'relu6': ReLU6(), 'mish': Mish(), 'swish': Swish(), 'leaky_relu': LeakyReLU()}
get_custom_objects().update(custom_objects)

__all__ = ['Mish', "ReLU6", "Swish"]
