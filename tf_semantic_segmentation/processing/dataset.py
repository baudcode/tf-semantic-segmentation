from . import ColorMode

import tensorflow as tf
import numpy as np
import multiprocessing


def resize_and_change_color(image, mask, size, color_mode, resize_method='resize_with_pad'):
    if color_mode == ColorMode.RGB and image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)

    elif color_mode == ColorMode.GRAY and image.shape[-1] != 1:
        image = tf.image.rgb_to_grayscale(image)

    if size is not None:
        # make 3dim for tf.image.resize
        mask = tf.expand_dims(mask, axis=-1)

        # augmentatations
        if resize_method == 'resize':
            image = tf.image.resize(image, size[::-1], antialias=True)
            mask = tf.image.resize(mask, size[::-1], method='nearest')  # use nearest for no interpolation

        elif resize_method == 'resize_with_pad':
            image = tf.image.resize_with_pad(image, size[1], size[0], antialias=True)
            mask = tf.image.resize_with_pad(mask, size[1], size[0], method='nearest')  # use nearest for no interpolation

        elif resize_method == 'resize_with_crop_or_pad':
            image = tf.image.resize_with_crop_or_pad(image, size[1], size[0])
            mask = tf.image.resize_with_crop_or_pad(mask, size[1], size[0])  # use nearest for no interpolation
        else:
            raise Exception("unknown resize method %s" % resize_method)

        # reduce dim added before
        mask = tf.squeeze(mask, axis=-1)

    return image, mask


def get_preprocess_fn(size, color_mode, resize_method, scale_mask=False, is_training=True):

    def map_fn(image, mask, num_classes):

        # scale between 0 and 1
        image = tf.image.convert_image_dtype(image, tf.float32)

        # resize method for image create float32 image anyway
        image, mask = resize_and_change_color(image, mask, size, color_mode, resize_method=resize_method)

        if scale_mask:
            num_classes = tf.cast(num_classes, tf.float32)
            mask = tf.cast(mask, tf.float32)
            mask = mask / (num_classes - 1.0)
        else:
            num_classes = tf.cast(num_classes, tf.int32)  # cast for onehot to accept it
            mask = tf.cast(mask, tf.int64)
            mask = tf.one_hot(mask, num_classes)

        return image, mask

    return map_fn


def prepare_dataset(dataset, batch_size, num_threads=8, buffer_size=200, repeat=0, take=0, skip=0, num_workers=1, worker_index=0, cache=False, shuffle=True, prefetch=True, augment_fn=None):

    if num_workers > 1:
        dataset = dataset.shard(num_workers, worker_index)

    if skip > 0:
        dataset = dataset.skip(skip)

    if take > 0:
        dataset = dataset.take(take)

    if cache:
        dataset = dataset.cache()

    if repeat > 0:
        dataset = dataset.repeat(repeat)
    else:
        dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size)

    if augment_fn:
        dataset = dataset.map(augment_fn, num_parallel_calls=multiprocessing.cpu_count())

    # dataset = dataset.prefetch(buffer_size=buffer_size)
    if prefetch:
        dataset = dataset.prefetch(buffer_size // batch_size)
    return dataset


def random_flip_lr(**args):
    bool_right_left = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5
    for k, x in args.items():
        x = tf.cond(bool_right_left, lambda: tf.image.flip_left_right(x), lambda: x)
        args[k] = x
    return args


def random_flip_ud(**args):
    bool_up_down = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5
    for k, x in args.items():
        x = tf.cond(bool_up_down, lambda: tf.image.flip_up_down(x), lambda: x)
        args[k] = x
    return args


def random_rot180(**args):

    angle = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32) * 2
    for k, x in args.items():
        x = tf.image.rot90(x, angle)
        args[k] = x
    return args


def random_color(**args):

    bool_color = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) > 0.5

    def _apply_color_augmentations(x):
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return x

    for k, x in args.items():
        args[k] = tf.cond(bool_color, lambda: _apply_color_augmentations(x), lambda: x)
    return args


def random_zoom(size, batch_size, **args):
    """ TODO: Not working at the moment """
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img, idx):
        print(img.shape)
        crops = tf.image.crop_and_resize(img, boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=size, method='nearest')
        print(crops)
        return crops[idx]
    print(boxes)
    idxs = tf.random.uniform(shape=[batch_size], minval=0, maxval=len(scales), dtype=tf.int32)
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    for k, x in args.items():
        # Only apply cropping 50% of the time
        elems = [(args[k][i, :, :, :], idxs[i]) for i in range(batch_size)]
        args[k] = tf.cond(choice > 1.0, lambda: x, lambda: tf.map_fn(random_crop, elems=elems, dtype=args[k].dtype))

    return args


augmentation_methods = ['rot180', 'flip_lr', 'flip_ud', 'color']


def get_augment_fn(size, batch_size, methods=['rot180', 'flip_lr', 'flip_ud', 'color']):

    for method in methods:
        if method not in augmentation_methods:
            raise Exception("cannot find augmentation method %s, please choose one of %s" % (method, str(augmentation_methods)))

    def augment(images, masks):

        # random rotation
        if 'rot180' in methods:
            rot180 = random_rot180(images=images, masks=masks)
            images, masks = rot180['images'], rot180['masks']

        # random flipping
        if 'flip_lr' in methods:
            flip = random_flip_lr(images=images, masks=masks)
            images, masks = flip['images'], flip['masks']

        if 'flip_ud' in methods:
            flip = random_flip_ud(images=images, masks=masks)
            images, masks = flip['images'], flip['masks']

        # random zoom
        if 'zoom' in methods:
            zoom = random_zoom(size, batch_size, images=images, masks=masks)
            images, masks = zoom['images'], zoom['masks']

        if 'color' in methods:
            images = random_color(images=images)['images']
            # clip images

        images = tf.clip_by_value(images, 0, 1)
        return (images, masks)

    return augment


if __name__ == "__main__":
    from ..utils import get_random_image
    from ..visualizations import show

    width, height = 512, 512
    image = get_random_image(width=width, height=height, grayscale=False)
    mask = get_random_image(width=width, height=height, grayscale=True)

    images = tf.expand_dims(tf.image.convert_image_dtype(image, tf.float32), axis=0)
    masks = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=-1)

    print("images shape, dtype ", images.shape, images.dtype)
    print("masks shape, dtype ", masks.shape, masks.dtype)

    fn = get_augment_fn((width, height), 1, methods=['rot180', 'flip_lr', 'flip_ud', 'color'])

    for i in range(5):
        _images = tf.identity(images)
        _masks = tf.identity(masks)
        aimages, amasks = fn(_images, _masks)

        print("images shape, dtype ", aimages.shape, aimages.dtype)
        print("masks shape, dtype ", amasks.shape, amasks.dtype)

        show.show_images([mask, image, amasks[0].numpy(), aimages[0].numpy()], cols=2, titles=['mask', 'image', 'augmented mask', 'augmented image'])
