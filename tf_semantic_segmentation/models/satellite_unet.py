""" https://github.com/reachsumit/deep-unet-for-satellite-image-segmentation/blob/master/unet_model_deeper.py """
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, MaxPooling2D, Concatenate, UpSampling2D, Dropout
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def satellite_unet(input_shape=(256, 256, 1), num_classes=3, num_first_filters=32,
                   growth_factor=2, upconv=True, dropout=0.25, activation='relu'):

    n_filters = num_first_filters
    inputs = Input(input_shape)
    #inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv4_0)
    pool4_0 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
    pool4_0 = Dropout(dropout)(pool4_0)

    n_filters *= growth_factor
    pool4_0 = BatchNormalization()(pool4_0)
    conv4_1 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(pool4_0)
    conv4_1 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv4_1)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
    pool4_1 = Dropout(dropout)(pool4_1)

    n_filters *= growth_factor
    pool4_1 = BatchNormalization()(pool4_1)
    conv4_2 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(pool4_1)
    conv4_2 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv4_2)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_2)
    pool4_2 = Dropout(dropout)(pool4_2)

    n_filters *= growth_factor
    pool4_2 = BatchNormalization()(pool4_2)
    conv5 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(pool4_2)
    conv5 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv5)
    conv5 = Dropout(dropout)(conv5)

    n_filters //= growth_factor
    if upconv:
        up6 = K.concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_2])
    else:
        up6 = K.concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_2])

    up6 = BatchNormalization()(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv6)
    conv6 = Dropout(dropout)(conv6)

    n_filters //= growth_factor
    if upconv:
        up6_1 = K.concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv4_1])
    else:
        up6_1 = K.concatenate([UpSampling2D(size=(2, 2))(conv6), conv4_1])
    up6_1 = BatchNormalization()(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv6_1)
    conv6_1 = Dropout(dropout)(conv6_1)

    n_filters //= growth_factor
    if upconv:
        up6_2 = K.concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
    else:
        up6_2 = K.concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
    up6_2 = BatchNormalization()(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv6_2)
    conv6_2 = Dropout(dropout)(conv6_2)

    n_filters //= growth_factor
    if upconv:
        up7 = K.concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
    else:
        up7 = K.concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv7)
    conv7 = Dropout(dropout)(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = K.concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = K.concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv8)
    conv8 = Dropout(dropout)(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = K.concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = K.concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])

    up9 = BatchNormalization()(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation=activation, padding='same')(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation=None)(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model


if __name__ == '__main__':
    satellite_unet(input_shape=(256, 256, 3), num_classes=3, num_first_filters=16).summary()
