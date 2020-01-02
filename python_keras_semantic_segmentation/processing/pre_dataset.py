import tensorflow as tf
from . import ColorMode


def get_preprocess_fn(size, color_mode, scale_labels=False, is_training=True):

    def map_fn(image, labels, num_classes):

        if color_mode == ColorMode.RGB and image.shape[2] == 1:
            image = tf.image.grayscale_to_rgb(image)

        elif color_mode == ColorMode.GRAY and image.shape[2] != 1:
            image = tf.image.rgb_to_grayscale(image)

        # scale between 0 and 1
        image = tf.image.convert_image_dtype(image, tf.float32)

        # augmentatations
        image = tf.image.resize(image, size)

        # onehot

        labels = tf.expand_dims(labels, axis=-1)  # make 3dim for tf.image.resize
        labels = tf.image.resize(labels, size, method='nearest')  # use nearest for no interpolation
        labels = tf.squeeze(labels, axis=-1)  # reduce dim added before

        if scale_labels:
            num_classes = tf.cast(num_classes, tf.float32)
            labels = tf.cast(labels, tf.float32)
            labels = labels / (num_classes - 1.0)

        else:
            num_classes = tf.cast(num_classes, tf.int32)  # cast for onehot to accept it
            labels = tf.cast(labels, tf.int64)
            labels = tf.one_hot(labels, num_classes)

        return image, labels

    return map_fn


def prepare_dataset(dataset, batch_size, num_threads=8, shuffle_buffer_size=4000, buffer_size=1000, repeat=0, take=0, skip=0, num_workers=1, worker_index=0, cache=True):

    if cache:
        dataset = dataset.cache()

    if num_workers > 1:
        dataset = dataset.shard(num_workers, worker_index)

    if skip > 0:
        dataset = dataset.skip(skip)

    if take > 0:
        dataset = dataset.take(take)

    # dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.shuffle(buffer_size=200)  # tf.data.experimental.AUTOTUNE)

    if repeat > 0:
        dataset = dataset.repeat(repeat)
    else:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
