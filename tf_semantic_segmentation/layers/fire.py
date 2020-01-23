from tensorflow.keras.layers import Conv2D, Add, Concatenate, Layer


class Fire(Layer):

    """ https://arxiv.org/abs/1602.07360 """

    def __init__(self, squeeze, expand=None, use_bypass=True, padding='same', activation='relu', norm=None, **kwargs):
        super(Fire, self).__init__(**kwargs)
        self.squeeze = squeeze
        if expand:
            self.expand = expand
        else:
            self.expand = self.squeeze * 4

        self.activation = activation
        self.padding = padding
        self.norm = norm
        self.use_bypass = use_bypass

    def call(self, x, training=None):

        num_features = K.int_shape(x)[-1]

        if self.expand > num_features or self.expand < num_features:
            x = Conv2D(self.expand, kernel_size=(1, 1),
                       activation=self.activation, padding=self.padding)(x)

        squeeze = Conv2D(self.squeeze, (1, 1), activation=self.activation,
                         padding='same', name='squeeze')(x)
        if self.norm:
            squeeze = self.norm(squeeze)

        expand_1x1 = Conv2D(self.expand, (1, 1), strides=(
            1, 1), activation=self.activation, padding=self.padding, name='expand_1x1')(squeeze)
        expand_3x3 = Conv2D(self.expand, (3, 3), strides=(
            1, 1), activation=self.activation, padding=self.padding, name='expand_3x3')(squeeze)

        if self.use_bypass:
            x_ret = Add(name='concatenate_bypass')([expand_1x1, expand_3x3, x])
        else:
            x_ret = Concatenate(
                axis=-1, name='concatenate')([expand_1x1, expand_3x3])

        return x_ret

    def get_config(self):

        config = {
            'squeeze': self.squeeze,
            'expand': self.expand
        }
        base_config = super(Fire, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
