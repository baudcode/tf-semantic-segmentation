from tf_semantic_segmentation.layers.utils import get_norm_by_name
from tensorflow.keras import layers
from tensorflow.keras import Model
from .apps import resnet50


def fcn_head(x, features, acivation='relu', norm='batch'):
    inter_channels = x.shape[-1] // 4
    y = layers.Conv2D(inter_channels, kernel_size=3, padding="SAME", use_bias=False)(x)
    y = get_norm_by_name(norm)(y)
    y = layers.Activation(acivation)(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Conv2D(features, kernel_size=1)(y)
    return y


def fcn(input_shape=(256, 256, 3), num_classes=8, upsample_factor=32, activation='relu', norm='batch', encoder_weights='imagenet'):
    base_model = resnet50.ResNet50(input_shape=input_shape, include_top=False, weights=encoder_weights)

    for l in base_model.layers:
        l.trainable = True

    # conv0 = base_model.get_layer("activation").output
    # conv1 = base_model.get_layer("activation_1").output
    conv2 = base_model.get_layer("activation_10").output
    conv3 = base_model.get_layer("activation_22").output
    # conv4 = base_model.get_layer("activation_40").output
    conv5 = base_model.get_layer("activation_48").output

    if upsample_factor == 8:
        y = fcn_head(conv2, 2048, acivation=activation, norm=norm)
        y = layers.UpSampling2D(size=(upsample_factor, upsample_factor), interpolation='bilinear')(y)
    elif upsample_factor == 16:
        y = fcn_head(conv3, 2048, acivation=activation, norm=norm)
        y = layers.UpSampling2D(size=(upsample_factor, upsample_factor), interpolation='bilinear')(y)
    elif upsample_factor == 32:
        y = fcn_head(conv5, 2048, acivation=activation, norm=norm)
        y = layers.UpSampling2D(size=(upsample_factor, upsample_factor), interpolation='bilinear')(y)
    else:
        raise Exception("upsample factor %d is invalid" % upsample_factor)

    y = layers.Conv2D(num_classes, kernel_size=1)(y)
    print("shape:", y.shape)
    # x = interpolate(x, imsize, **self._up_kwargs)
    return Model(inputs=base_model.inputs, outputs=y)


if __name__ == "__main__":
    fcn(input_shape=(256, 256, 3), num_classes=3, upsample_factor=32).summary()
