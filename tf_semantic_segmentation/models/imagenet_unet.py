""" Code adopted from https://github.com/killthekitten/kaggle-carvana-2017/blob/master/models.py"""
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, Activation, SpatialDropout2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.applications import vgg16
from tensorflow.keras import backend as K

from .apps import resnet50, mobilenet, inception


def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1), activation='relu'):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation(activation, name=prefix + "_activation")(conv)
    return conv


def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1), activation='relu'):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation(activation, name=prefix + "_activation")(conv)
    return conv


def unet_resnet(input_shape=(256, 256, 3), num_classes=8, encoder_weights='imagenet'):

    base_model = resnet50.ResNet50(input_shape=input_shape, include_top=False, weights=encoder_weights)

    for l in base_model.layers:
        l.trainable = True

    conv0 = base_model.get_layer("activation").output
    conv1 = base_model.get_layer("activation_1").output
    conv2 = base_model.get_layer("activation_10").output
    conv3 = base_model.get_layer("activation_22").output
    conv4 = base_model.get_layer("activation_40").output
    conv5 = base_model.get_layer("activation_48").output

    # (None, 128, 128, 64) (None, 64, 64, 128) (None, 32, 32, 256) (None, 16, 16, 512) (None, 16, 16, 2048)
    # print(conv1.shape, conv2.shape, conv3.shape, conv4.shape, conv5.shape)

    up6 = K.concatenate([conv5, conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = K.concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = K.concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = K.concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up9x = K.concatenate([UpSampling2D()(conv9), conv0], axis=-1)
    conv9x = conv_block_simple(up9x, 64, "conv9x_1")
    conv9x = conv_block_simple(conv9x, 64, "conv9x_2")

    vgg = vgg16.VGG16(input_shape=input_shape, input_tensor=base_model.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False

    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = K.concatenate([UpSampling2D()(conv9x), vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)

    x = Conv2D(num_classes, (1, 1), activation=None, name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model


def unet_mobilenet(input_shape=(256, 256, 3), num_classes=3, encoder_weights='imagenet'):
    base_model = mobilenet.MobileNet(include_top=False, input_shape=input_shape, weights=encoder_weights)

    conv1 = base_model.get_layer('conv_pw_1_relu').output
    conv2 = base_model.get_layer('conv_pw_3_relu').output
    conv3 = base_model.get_layer('conv_pw_5_relu').output
    conv4 = base_model.get_layer('conv_pw_11_relu').output
    conv5 = base_model.get_layer('conv_pw_13_relu').output

    up6 = K.concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = K.concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = K.concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 192, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = K.concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 96, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = K.concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)

    x = Conv2D(num_classes, (1, 1), activation=None, name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model


def unet_inception_resnet_v2(input_shape=(256, 256, 3), num_classes=6, encoder_weights='imagenet'):
    base_model = inception.InceptionResNetV2(include_top=False, input_shape=input_shape, weights=encoder_weights)

    conv0 = base_model.get_layer('activation_2').output  # activation_2 (None, 512, 512, 64)
    conv1 = base_model.get_layer('activation_3').output  # activation_3 (None, 256, 256, 80)
    conv2 = base_model.get_layer('activation_5').output  # activation_5 (None, 128, 128, 96)
    conv2x = base_model.get_layer('block35_10_ac').output  # block35_10_ac (None, 128, 128, 320)
    conv4 = base_model.get_layer('block17_20_ac').output  # block17_20_ac (None, 64, 64, 1088)
    conv5 = base_model.get_layer('conv_7b_ac').output

    # print(conv1.shape, conv2.shape, conv2x.shape, conv4.shape, conv5.shape)

    up6 = K.concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = K.concatenate([UpSampling2D()(conv6), conv2, conv2x], axis=-1)
    conv7 = conv_block_simple(up7, 128, "conv7_1")
    conv7 = conv_block_simple(conv7, 128, "conv7_2")

    up8 = K.concatenate([UpSampling2D()(conv7), conv1], axis=-1)
    conv8 = conv_block_simple(up8, 64, "conv8_1")
    conv8 = conv_block_simple(conv8, 64, "conv8_2")

    up9 = K.concatenate([UpSampling2D()(conv8), conv0], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = K.concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.4)(conv10)

    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation=None, name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model


if __name__ == "__main__":

    # model = unet_resnet((1024, 1024, 3))
    # model.summary()
    # (960, 480)

    model = unet_inception_resnet_v2((480, 960, 3), num_classes=8)
    K.clear_session()

    model = unet_resnet((480, 960, 3), num_classes=8)
    K.clear_session()

    model = unet_mobilenet((480, 960, 3), num_classes=8)
    K.clear_session()

    for layer in model.layers:
        print(layer.name, layer.output.shape)
    # model.summary()
