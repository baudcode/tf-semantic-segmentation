from tf_semantic_segmentation import activations
from tf_semantic_segmentation.layers.utils import get_norm_by_name
from tensorflow.keras import layers
import tensorflow as tf

from .apps import resnet50
from tensorflow.keras.models import Model


def pool_conv_up(x, pool, filters, norm='batch', activation='relu'):
    from tensorflow_addons.layers import AdaptiveAveragePooling2D
    y = AdaptiveAveragePooling2D(pool)(x)

    y = layers.Conv2D(filters, kernel_size=1, use_bias=False)(y)
    y = get_norm_by_name(norm)(y)
    y = layers.Activation(activation)

    y = tf.image.resize(x, size=x.shape[1:3], method='bilinear')
    return y


def pyramid_pooling(x, norm='batch', activation='relu'):
    out_channels = int(x.shape[-1] / 4)
    f1 = pool_conv_up(x, 1, out_channels, norm=norm, activation=activation)
    f2 = pool_conv_up(x, 2, out_channels, norm=norm, activation=activation)
    f3 = pool_conv_up(x, 3, out_channels, norm=norm, activation=activation)
    f4 = pool_conv_up(x, 6, out_channels, norm=norm, activation=activation)
    return tf.concat([f1, f2, f3, f4], axis=-1)

# class PyramidPooling(Module):
#     """
#     Reference:
#         Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
#     """

#     def __init__(self, in_channels, norm_layer, up_kwargs):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = AdaptiveAvgPool2d(1)  # GlobalAveragePooling
#         self.pool2 = AdaptiveAvgPool2d(2)
#         self.pool3 = AdaptiveAvgPool2d(3)
#         self.pool4 = AdaptiveAvgPool2d(6)

#         out_channels = int(in_channels / 4)
#         self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
#                                 norm_layer(out_channels),
#                                 ReLU(True))
#         self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
#                                 norm_layer(out_channels),
#                                 ReLU(True))
#         self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
#                                 norm_layer(out_channels),
#                                 ReLU(True))
#         self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
#                                 norm_layer(out_channels),
#                                 ReLU(True))
#         # bilinear upsample options
#         self._up_kwargs = up_kwargs

#     def forward(self, x):
#         _, _, h, w = x.size()
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)


def psp_head(x, out_channels, norm='batch', activation='relu'):
    inter_channels = x.shape[-1] // 4
    y = pyramid_pooling(x, norm, activation)
    y = layers.Conv2D(inter_channels, kernel_size=3, padding='same', use_bias=False)(y)
    y = get_norm_by_name(norm)(y)
    y = layers.Activation(activation)(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Conv2D(out_channels, kernel_size=1, padding='same')(y)
    return y


def get_closest(x):
    if x <= 18:
        return 18
    elif x <= 24:
        return 24
    elif x <= 36:
        return 36
    elif x <= 48:
        return 48
    elif x <= 72:
        return 72
    elif x <= 92:
        return 96


def psp(input_shape=(512, 512, 3), num_classes=8, depth=5, norm='batch', activation='relu', encoder_weights='imagenet'):
    base_model = resnet50.ResNet50(input_shape=input_shape, include_top=False, weights=encoder_weights)

    for l in base_model.layers:
        l.trainable = True

    # y = tf.image.resize(conv3, (48, 48), method='nearest')

    # conv = base_model.get_layer("activation_10").output
    # conv = base_model.get_layer("activation_40").output
    # conv = base_model.get_layer("activation_22").output
    depth2layer = {
        2: "activation_10",
        3: "activation_22",
        4: "activation_40",
        5: "activation_48"
    }
    if depth not in depth2layer:
        raise Exception("invalid depth for psp net")

    name = depth2layer[depth]
    conv = base_model.get_layer(name).output

    size = (get_closest(conv.shape[1]), get_closest(conv.shape[2]))
    # to use global average pooling channels must be evenly divisible by 3
    y = tf.image.resize(conv, size, method='bilinear')

    y = psp_head(y, num_classes, norm=norm, activation=activation)
    y = tf.image.resize(y, (input_shape[0], input_shape[1]), method='bilinear')
    return Model(inputs=base_model.inputs, outputs=y)


if __name__ == "__main__":
    model = psp()
    model.summary()
