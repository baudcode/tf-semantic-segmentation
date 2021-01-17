import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model


def conv(x, filters=3, dirate=1, kernel_size=3, l2=None):
    return layers.Conv2D(filters, kernel_size=kernel_size,
                         kernel_regularizer=regularizers.l2(l2) if l2 else None,
                         activation=None,
                         dilation_rate=dirate,
                         padding='same')(x)


def rebnconv(x, filters=3, dirate=1, l2=None):
    y = conv(x, filters, dirate=dirate, l2=l2)
    y = layers.BatchNormalization(axis=-1)(y)
    return layers.Activation("relu")(y)


def upsample(x, factor=2):
    # size=tar.shape[2:]
    if factor == 1:
        return x
    return layers.UpSampling2D(size=(factor, factor), interpolation='bilinear')(x)


def pool(x):
    return layers.MaxPooling2D(2)(x)


def resu(x, iterations, mid_filters=12, out_filters=3):
    layers = []

    y_in = rebnconv(x, filters=out_filters, dirate=1)
    layers.append(y_in)

    y = y_in

    for i in range(iterations):
        y = rebnconv(y, filters=mid_filters, dirate=1)
        layers.append(y)
        y = pool(y)

    y = rebnconv(y, filters=mid_filters, dirate=1)
    layers.append(y)

    y = rebnconv(y, filters=mid_filters, dirate=2)

    for i in range(iterations):
        y = rebnconv(tf.concat([y, layers[-(i + 1)]], axis=-1), filters=mid_filters, dirate=1)
        y = upsample(y)

    y = rebnconv(tf.concat([y, layers[0]], axis=-1), filters=out_filters, dirate=1)
    return y + y_in


def resu4f(x, mid_filters, out_filters=3):
    y_in = rebnconv(x, filters=out_filters, dirate=1)

    y1 = rebnconv(y_in, mid_filters, dirate=1)
    y2 = rebnconv(y1, mid_filters, dirate=2)
    y3 = rebnconv(y2, mid_filters, dirate=4)
    y4 = rebnconv(y3, mid_filters, dirate=8)

    y = rebnconv(tf.concat([y4, y3], axis=-1), filters=mid_filters, dirate=4)
    y = rebnconv(tf.concat([y, y2], axis=-1), filters=mid_filters, dirate=2)
    y = rebnconv(tf.concat([y, y1], axis=-1), filters=out_filters, dirate=1)

    return y + y_in


def resu_builder(x, name, mid_filters=12, out_filters=3):
    if name == 'resu7':
        return resu(x, 5, mid_filters=mid_filters, out_filters=out_filters)
    elif name == 'resu6':
        return resu(x, 4, mid_filters=mid_filters, out_filters=out_filters)
    elif name == 'resu5':
        return resu(x, 3, mid_filters=mid_filters, out_filters=out_filters)
    elif name == 'resu4':
        return resu(x, 2, mid_filters=mid_filters, out_filters=out_filters)
    elif name == 'resu4f':
        return resu4f(x, mid_filters=mid_filters, out_filters=out_filters)
    else:
        raise Exception("known resu name %s" % name)


def u2net_builder(encoder, decoder, input_shape=(256, 256, 3), num_classes=2):
    assert(encoder[-1] == decoder[0])

    x = layers.Input(shape=input_shape, name='inputs')

    _layers = []
    y = x

    for name, midf, outf in encoder:
        print("encoder", name, midf, outf)
        y = resu_builder(y, name, mid_filters=midf, out_filters=outf)
        _layers.append(y)
        y = pool(y)

    y = resu_builder(y, decoder[0][0], decoder[0][1], decoder[0][2])

    decoder_layers = []

    for i, (name, midf, outf) in enumerate(decoder):
        print("decoder", y.shape, name, midf, outf)
        y = upsample(y)
        y = resu_builder(tf.concat([y, _layers[-(i + 1)]], axis=-1), name, midf, outf)
        decoder_layers.append(y)

    side_layers = []
    for i, dout in enumerate(reversed(decoder_layers)):
        s = conv(dout, num_classes, dirate=1)
        side = upsample(s, pow(2, i))
        print("side: ", i, s.shape, "->", side.shape)
        side_layers.append(side)

    out = conv(tf.concat(side_layers, axis=-1), num_classes, kernel_size=1)
    return Model(outputs=out, inputs=x)


def u2net(input_shape=(256, 256, 3), num_classes=2):
    encoder = [
        ["resu7", 32, 64],
        ["resu6", 64, 128],
        ["resu5", 128, 256],
        ["resu4", 256, 512],
        ["resu4f", 256, 512],
    ]

    decoder = [
        ["resu4f", 256, 512],
        ["resu4", 128, 256],
        ["resu5", 64, 128],
        ["resu6", 32, 64],
        ["resu7", 16, 64],
    ]

    return u2net_builder(encoder, decoder, input_shape, num_classes)


def u2netp(input_shape=(256, 256, 3), num_classes=2):

    encoder = [
        ["resu7", 16, 64],
        ["resu6", 16, 64],
        ["resu5", 16, 64],
        ["resu4", 16, 64],
        ["resu4f", 16, 64],
    ]

    decoder = [
        ["resu4f", 16, 64],
        ["resu4", 16, 64],
        ["resu5", 16, 64],
        ["resu6", 16, 64],
        ["resu7", 16, 64],
    ]

    return u2net_builder(encoder, decoder, input_shape, num_classes)


if __name__ == "__main__":
    model = u2net()
    model.summary()

    model = u2netp()
    model.summary()
