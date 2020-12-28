from tensorflow.keras import layers, regularizers
from ..layers import get_norm_by_name
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


def conv(x, filters, kernel_size=(3, 3), l2=None, padding='SAME', activation='relu'):
    y = layers.Conv2D(filters, kernel_size=kernel_size,
                      kernel_regularizer=regularizers.l2(l2) if l2 else None,
                      activation=None,
                      padding=padding)(x)
    y = get_norm_by_name('batch')(y)
    y = layers.Activation(activation)(y)
    return y


def nested_conv(x, filters, residual=False):
    y = conv(x, filters)
    y = conv(y, filters)
    if residual:
        return x + y
    else:
        return y


def nested_unet(input_shape=(256, 256, 3), num_classes=2, num_first_filters=64, depth=4):
    """UNet++ aims to improve segmentation accuracy, with a series of nested, dense skip pathways.

    Redesigned skip pathways made optimisation easier with the semantically similar feature maps.
    Dense skip connections improve segmentation accuracy and improve gradient flow.

    https://arxiv.org/abs/1807.10165
    """
    n1 = num_first_filters
    filters = [num_first_filters * pow(2, i) for i in range(5)]
    assert(depth in [3, 4, 5]), 'depth has to be either 3, 4 or 5'

    pool = layers.MaxPooling2D(pool_size=(2, 2))
    up = layers.UpSampling2D((2, 2), interpolation='bilinear')

    x = layers.Input(shape=input_shape, name='inputs')
    x0_0 = nested_conv(x, filters[0])

    x1_0 = nested_conv(pool(x0_0), filters[1])
    x0_1 = nested_conv(K.concatenate([x0_0, up(x1_0)]), filters[0])

    x2_0 = nested_conv(pool(x1_0), filters[1])
    x1_1 = nested_conv(K.concatenate([x1_0, up(x2_0)]), filters[1])
    x0_2 = nested_conv(K.concatenate([x0_0, x0_1, up(x1_1)]), filters[0])

    # 3
    x3_0 = nested_conv(pool(x2_0), filters[2])
    x2_1 = nested_conv(K.concatenate([x2_0, up(x3_0)]), filters[2])
    x1_2 = nested_conv(K.concatenate([x1_0, x1_1, up(x2_1)]), filters[1])
    x0_3 = nested_conv(K.concatenate([x0_0, x0_1, x0_2, up(x1_2)]), filters[0])

    # 4
    x4_0 = nested_conv(pool(x3_0), filters[3])
    x3_1 = nested_conv(K.concatenate([x3_0, up(x4_0)]), filters[3])
    x2_2 = nested_conv(K.concatenate([x2_0, x2_1, up(x3_1)]), filters[2])
    x1_3 = nested_conv(K.concatenate([x1_0, x1_1, x1_2, up(x2_2)]), filters[1])
    x0_4 = nested_conv(K.concatenate([x0_0, x0_1, x0_2, x0_3, up(x1_3)]), filters[0])

    # 5
    x5_0 = nested_conv(pool(x4_0), filters[4])
    x4_1 = nested_conv(K.concatenate([x4_0, up(x5_0)]), filters[4])
    x3_2 = nested_conv(K.concatenate([x3_0, x3_1, up(x4_1)]), filters[3])
    x2_3 = nested_conv(K.concatenate([x2_0, x2_1, x2_2, up(x3_2)]), filters[2])
    x1_4 = nested_conv(K.concatenate([x1_0, x1_1, x1_2, x1_3, up(x2_3)]), filters[1])
    x0_5 = nested_conv(K.concatenate([x0_0, x0_1, x0_2, x0_3, x0_4, up(x1_4)]), filters[0])

    if depth == 3:
        output = layers.Conv2D(num_classes, kernel_size=(1, 1), activation=None)(x0_3)
    elif depth == 4:
        output = layers.Conv2D(num_classes, kernel_size=(1, 1), activation=None)(x0_4)
    elif depth == 5:
        output = layers.Conv2D(num_classes, kernel_size=(1, 1), activation=None)(x0_5)
    else:
        raise Exception("depth %d is invalid" % depth)

    return Model(outputs=output, inputs=x)


if __name__ == "__main__":
    model = nested_unet(depth=4)
    model.summary()
