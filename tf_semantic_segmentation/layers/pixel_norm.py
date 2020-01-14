from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class PixelNorm(Layer):
    """ https://arxiv.org/abs/1710.10196 

    It normalizes the feature vector in each pixel to unit length, and is applied after the convolutional layers. 
    This is done to prevent signal magnitudes from spiraling out of control during training.    
    """

    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def call(self, x, training=True):
        return x / K.sqrt(K.mean(K.square(x), axis=-1, keepdims=True) + self.epsilon)

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
        }
        base_config = super(PixelNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
