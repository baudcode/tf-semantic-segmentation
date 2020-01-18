from tensorflow.keras import backend
SMOOTH = 1e-5

# ----------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------


def to1d(t):
    entries = backend.prod(t.shape)
    return backend.reshape(t, (entries, ))


def to2d(t):
    s = backend.prod(backend.shape(t)[1:])
    t = backend.reshape(t, [-1, s])
    return t


def onehot2image(y):
    """
    Arguments:
    - y: Tensor BHWC (onehot)

    Scales input of shape BHWC to BHW1 image of range (0, 1) tf.float32
    """
    if y.shape[-1] == 1:
        # assume masks are scaled using sigmoid
        return y

    y = backend.argmax(y, axis=-1)
    y = backend.expand_dims(y, axis=-1)
    y = backend.cast(y, backend.floatx())
    y = backend.cast(y, backend.floatx()) / backend.cast(backend.max(y), backend.floatx())
    return y


def _gather_channels(x, indexes):
    """Slice tensor along channels axis by given indexes"""
    if backend.image_data_format() == 'channels_last':
        x = backend.permute_dimensions(x, (3, 0, 1, 2))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 2, 3, 0))
    else:
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
        x = backend.gather(x, indexes)
        x = backend.permute_dimensions(x, (1, 0, 2, 3))
    return x


def get_reduce_axes(per_image):
    axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    if not per_image:
        axes.insert(0, 0)
    return axes


def gather_channels(*xs, indexes=None):
    """Slice tensors along channels axis by given indexes"""
    if indexes is None:
        return xs
    elif isinstance(indexes, (int)):
        indexes = [indexes]
    xs = [_gather_channels(x, indexes=indexes) for x in xs]
    return xs


def round_if_needed(x, threshold):
    if threshold is not None:
        x = backend.greater(x, threshold)
        x = backend.cast(x, backend.floatx())
    return x


def average(x, per_image=False, class_weights=None):
    if per_image:
        x = backend.mean(x, axis=0)
    if class_weights is not None:
        x = x * class_weights
    return backend.mean(x)


def expand_binary(x):
    # remove last dim
    x = backend.squeeze(x, axis=-1)
    # scale to 0 or 1
    x = backend.round(x)
    x = backend.cast(x, 'int32')
    x = backend.one_hot(x, 2)
    return x
