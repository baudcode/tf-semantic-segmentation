""" Code taken from https://github.com/atriumlts/subpixel/blob/a2d9e9a163db9e3268df1b7109672eea814ee356/keras_subpixel.py """


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D


class Subpixel(Conv2D):

    """ https://arxiv.org/abs/1609.05158

        Subpixel Layer as a child class of Conv2D. This layer accepts all normal
        arguments, with the exception of dilation_rate(). The argument r indicates
        the upsampling factor, which is applied to the normal output of Conv2D.
        The output of this layer will have the same number of channels as the
        indicated filter field, and thus works for grayscale, color, or as a a
        hidden layer.

        Arguments:
            *see Keras Docs for Conv2D args, noting that dilation_rate() is removed*
            r: upscaling factor, which is applied to the output of normal Conv2D
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r * r * filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, K.cast(c / (r * r), 'int32'), r, r])  # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:, i, :, :, :, :] for i in range(a)]  # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:, i, :, :, :] for i in range(b)]  # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r * unshifted[1], self.r * unshifted[2], unshifted[3] / (self.r * self.r))

    def get_config(self):
        config = super(Subpixel, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters'] /= self.r * self.r
        config['r'] = self.r
        return config
