from ..layers import get_norm_by_name

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


def conv(x, filters, kernel_size, strides=1, norm='batch', activation='relu', l2=None, rate=1, deconv=False):

    if deconv:
        output_shape = list(K.int_shape(x)[1:])
        output_shape[0] = int(output_shape[0] * strides)
        output_shape[1] = int(output_shape[1] * strides)
        output_shape[2] = filters
        print(output_shape)

        y = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='SAME', activation=activation,
                                   dilation_rate=rate, kernel_regularizer=regularizers.l2(l2) if l2 else None)(x)
        # y = K.reshape(y, [-1] + output_shape)
        # y = layers.Reshape(output_shape)(y)
    else:
        y = layers.Conv2D(filters, kernel_size, strides=strides, padding='SAME', activation=activation,
                          kernel_regularizer=regularizers.l2(l2) if l2 else None,
                          dilation_rate=rate)(x)

     # kernel_regularizer=regularizers.l2(l2) if l2 else None
    if norm:
        y = get_norm_by_name(norm)(y)

    return y


def factorized_module(x, dropout=0.3, dilation=[1, 1], l2=None):
    print("factorized: ", locals())
    n = K.int_shape(x)[-1]
    y = conv(x, n, [3, 1], rate=dilation[0], norm=None, l2=l2)
    y = conv(y, n, [1, 3], rate=dilation[0], l2=l2)
    y = conv(y, n, [3, 1], rate=dilation[1], norm=None, l2=l2)
    y = conv(y, n, [1, 3], rate=dilation[1], l2=l2)
    y = layers.Dropout(dropout)(y)
    y = layers.Add()([x, y])
    return y


def downsample(x, n, activation='relu', norm=None, l2=None):
    print('downsample: ', locals())
    f_in = int(K.int_shape(x)[-1])
    f_conv = int(n - f_in)
    branch_1 = conv(x, f_conv, 3, strides=2, l2=l2)
    branch_2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(x)
    return layers.Concatenate(axis=-1)([branch_1, branch_2])


def upsample(x, n, norm=None, activation=None, l2=None):
    return conv(x, n, 3, strides=2, deconv=True, l2=l2)


def erfnet(input_shape=(256, 256, 1), num_classes=3, l2=None):
    x = layers.Input(shape=input_shape, name='inputs')

    y = downsample(x, 16, l2=l2)
    y = downsample(y, 64, l2=l2)

    for i in range(5):
        y = factorized_module(y, dilation=[1, 1], l2=l2)

    y = downsample(y, 128, l2=l2)
    for k in range(2):
        for i in range(4):
            y = factorized_module(y, dilation=[1, pow(2, i + 1)], l2=l2)

    print("upsample me...")
    y = upsample(y, 64)
    for i in range(2):
        y = factorized_module(y, dilation=[1, 1], l2=l2)

    y = upsample(y, 16)
    for i in range(2):
        y = factorized_module(y, dilation=[1, 1], l2=l2)

    base_model = Model(inputs=x, outputs=y)
    y = upsample(y, num_classes, l2=l2)

    return Model(inputs=x, outputs=y), base_model


if __name__ == "__main__":
    erfnet()
