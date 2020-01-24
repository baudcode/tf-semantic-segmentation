from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.models import Model

""" code is heavily borrowed from https://github.com/nibtehaz/MultiResUNet/blob/master/MultiResUNet.py """


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if activation is None:
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters

    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})

    Returns:
        [keras layer] -- [output layer]
    '''
    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    return x


def multires_block(U, inp, alpha=1.67):
    '''
    MultiRes Block

    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer

    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W * 0.167) + int(W * 0.333) +
                         int(W * 0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W * 0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W * 0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W * 0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def respath(x, filters, length):
    '''
    respath

    Arguments:
        x {keras layer} -- input layer
        filters {int} -- [description]
        length {int} -- length of respath

    Returns:
        [keras layer] -- [output layer]
    '''
    shortcut = x
    shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')

    out = conv2d_bn(x, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length - 1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def multires_unet(input_shape=(256, 256, 1), num_classes=2):
    '''
    MultiResUNet

    Arguments:
        height {int} -- height of image
        width {int} -- width of image
        n_channels {int} -- number of channels in image

    Returns:
        [keras model] -- MultiResUNet model
    '''

    inputs = Input(input_shape)

    y = inputs
    paths = []
    for i in range(4):
        features = 16 * (i + 1) * 2
        print("encoding i=%d features=%d" % (i, features))
        mres = multires_block(features, y)
        y = MaxPooling2D(pool_size=(2, 2))(mres)
        paths.append(respath(mres, features, 4 - i))

    # add middle block
    y = multires_block(32 * 16, y)

    for i in reversed(range(4)):
        features = 16 * (i + 1) * 2
        print("decoding i=%d features=%d" % (i, features))

        y = concatenate([Conv2DTranspose(features, (2, 2), strides=(2, 2), padding='same')(y), paths[i]], axis=3)
        y = multires_block(features, y)

    # 1x1 for classification
    base_model = Model(inputs=[inputs], outputs=[y])

    y = conv2d_bn(y, num_classes, 1, 1, activation=None)
    model = Model(inputs=[inputs], outputs=[y])

    return model, base_model


def main():
    # Define the model
    model, _ = multires_unet((128, 128, 3), num_classes=3)
    print(model.summary())


if __name__ == '__main__':
    main()
