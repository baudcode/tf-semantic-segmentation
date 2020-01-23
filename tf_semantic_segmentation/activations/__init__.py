from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation


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


custom_objects = {'relu6': ReLU6(), 'mish': Mish(), 'swish': Swish()}
get_custom_objects().update(custom_objects)

__all__ = ['Mish', "ReLU6", "Swish"]
