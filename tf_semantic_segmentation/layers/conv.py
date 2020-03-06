from tensorflow.keras.layers import Conv2D, Add, Concatenate, Layer, DepthwiseConv2D
from tensorflow.keras import backend as K
import tensorflow as tf

from ..settings import logger
from .utils import get_norm_by_name


def fire(x, squeeze, expand=None, use_bypass=False, padding='same', activation='relu', norm='batch'):
    if not expand:
        expand = squeeze * 4

    num_features = K.int_shape(x)[-1]

    if expand != num_features:
        x = Conv2D(expand, kernel_size=(1, 1), activation=activation, padding=padding)(x)

    # squeeze 1x1 conv
    squeeze = Conv2D(squeeze, (1, 1), activation=activation, padding='same')(x)

    # apply norm
    if norm:
        squeeze = get_norm_by_name(norm)(squeeze)

    # expand 1x1 and 3x3
    expand_1x1 = Conv2D(expand, (1, 1), strides=(1, 1), activation=activation, padding=padding)(squeeze)
    expand_3x3 = Conv2D(expand, (3, 3), strides=(1, 1), activation=activation, padding=padding)(squeeze)

    if use_bypass:
        x_ret = Add()([expand_1x1, expand_3x3, x])
    else:
        x_ret = Concatenate(axis=-1)([expand_1x1, expand_3x3])

    return x_ret


def _split_channels(total_filters, num_groups):
    split = [total_filters // num_groups for _ in range(num_groups)]
    split[0] += total_filters - sum(split)
    return split


def grouped_conv_2d(x, filters, kernel_size, **kwargs):
    """Groupped convolution.
    Currently tf.keras and tf.layers don't support group convolution, so here we
    use split/concat to implement this op. It reuses kernel_size for group
    definition, where len(kernel_size) is number of groups. Notably, it allows
    different group has different kernel size.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or a list. If it is a single integer, then it is
        same as the original Conv2D. If it is a list, then we split the channels
        and perform different kernel for each group.
        use_keras: An boolean value, whether to use keras layer.
        **kwargs: other parameters passed to the original conv2d layer.
    """

    groups = len(kernel_size)
    channel_axis = -1

    convs = []
    splits = _split_channels(filters, groups)

    for i in range(groups):
        convs.append(Conv2D(splits[0], kernel_size[i], **kwargs))

    if len(convs) == 1:
        return convs[0](x)

    filters = x.shape[channel_axis]

    splits = _split_channels(filters, len(convs))
    x_splits = tf.split(x, splits, channel_axis)
    x_outputs = [c(x) for x, c in zip(x_splits, convs)]
    x = tf.concat(x_outputs, channel_axis)
    return x


def mixconv(x, kernel_size, strides, dilated=False, **kwargs):
    """MixConv with mixed depthwise convolutional kernels.
    MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
    3x3, 5x5, etc). Right now, we use an naive implementation that split channels
    into multiple groups and perform different kernels for each group.
    See Mixnet paper for more details.

    Most of args are the same as tf.keras.layers.DepthwiseConv2D except it has
    an extra parameter "dilated" to indicate whether to use dilated conv to
    simulate large kernel size. If dilated=True, then dilation_rate is ignored.
    Args:
        kernel_size: An integer or a list. If it is a single integer, then it is
        same as the original tf.keras.layers.DepthwiseConv2D. If it is a list,
        then we split the channels and perform different kernel for each group.

        strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width.

        dilated: Bool. indicate whether to use dilated conv to simulate large
        kernel size.
        **kwargs: other parameters passed to the original depthwise_conv layer.
    """

    channel_axis = -1

    convs = []
    for s in kernel_size:
        d = 1
        if strides[0] == 1 and dilated:
            # Only apply dilated conv for stride 1 if needed.
            d, s = (s - 1) // 2, 3
            logger.info('Use dilated conv with dilation rate = {}'.format(d))

        convs.append(tf.keras.layers.DepthwiseConv2D(s, strides=strides, dilation_rate=d, **kwargs))

    if len(convs) == 1:
        return convs[0](x)

    filters = x.shape[channel_axis]
    splits = _split_channels(filters, len(convs))
    x_splits = tf.split(x, splits, channel_axis)
    x_outputs = [c(x) for x, c in zip(x_splits, convs)]
    x = tf.concat(x_outputs, channel_axis)
    return x


class Fire(object):

    def __init__(self, squeeze, expand=None, use_bypass=False, padding='SAME', activation='relu', norm='batch'):
        self.squeeze = squeeze
        self.expand = expand
        self.use_bypass = use_bypass
        self.padding = padding
        self.activation = activation
        self.norm = norm

    def __call__(self, x):
        return fire(x, self.squeeze, self.expand, self.use_bypass, self.padding, self.activation, self.norm)


class GroupedConv2D(object):

    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.kwargs = kwargs

    def __call__(self, x):
        return grouped_conv_2d(x, self.filters, self.kernel_size, **self.kwargs)


class MixConv(object):

    def __init__(self, kernel_size, strides, dilated=False, **kwargs):
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilated = dilated
        self.kwargs = kwargs

    def __call__(self, x):
        return mixconv(x, self.kernel_size, self.strides, self.dilated, **self.kwargs)


if __name__ == "__main__":

    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input

    inputs = Input((32, 32, 3))
    x = Conv2D(32, padding='same', kernel_size=(3, 3), strides=(1, 1), activation='relu')(inputs)

    x = MixConv([3, 5, 7], strides=(1, 1), padding='same')(x)
    x = GroupedConv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = Fire(32, 64)(x)
    m = Model(inputs=inputs, outputs=x)
    m.compile(loss='mse')
    m.summary()
