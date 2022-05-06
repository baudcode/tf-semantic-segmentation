from typing import Dict, List, Optional, Tuple

from tf_semantic_segmentation.datasets.utils import Color, logger
from . import ColorMode
from .. import utils

import tensorflow as tf
import numpy as np
import multiprocessing

resize_methods = ['resize', 'resize_with_pad', 'resize_with_crop_or_pad', 'patch']
THRESHOLD = 0.5


def resize_and_change_color(image, mask, size, color_mode, resize_method='resize_with_pad', mode='graph'):
    """
    Arguments:

    - image: 3d or 4d tensor
    - mask: 2d, 3d tensor or None
    - size: new_height, new_width
    - color_mode: ColorMode
    - resize_method: how to resize the images, options: 'resize', 'resize_with_pad', 'resize_with_crop_or_pad'

    Returns:
    tuple (image, mask)
    """
    if mode == 'graph':
        # len(tf.shape(image)) == 1 and tf.shape(image)[0] == 2 seems to be a wierd hack when width and height
        # are not defined
        is2d = False
        if len(tf.shape(image).shape) < 2:
            shape = image.shape
        else:
            shape = tf.shape(image)

        if len(shape) == 2 or (len(shape) == 1 and shape[0] == 2):
            image = tf.expand_dims(image, axis=-1)
            is2d = True

        if color_mode == ColorMode.RGB and (shape[-1] == 1 or is2d):
            image = tf.image.grayscale_to_rgb(image)

        elif color_mode == ColorMode.GRAY and shape[-1] != 1:
            image = tf.image.rgb_to_grayscale(image)

    elif mode == 'eager':
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)

        if color_mode == ColorMode.RGB and image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)

        elif color_mode == ColorMode.GRAY and image.shape[-1] != 1:
            image = tf.image.rgb_to_grayscale(image)
    else:
        raise Exception("unknown mode %s" % mode)

    if size is not None:
        # make 3dim for tf.image.resize
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)

        # augmentatations
        if resize_method == 'resize':
            image = tf.image.resize(image, size, antialias=True)
            if mask is not None:
                mask = tf.image.resize(mask, size, method='nearest')  # use nearest for no interpolation

        elif resize_method == 'resize_with_pad':
            image = tf.image.resize_with_pad(image, size[0], size[1], antialias=True)
            if mask is not None:
                mask = tf.image.resize_with_pad(mask, size[0], size[1], method='nearest')  # use nearest for no interpolation

        elif resize_method == 'resize_with_crop_or_pad':
            image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])
            if mask is not None:
                mask = tf.image.resize_with_crop_or_pad(mask, size[0], size[1])  # use nearest for no interpolation

        elif resize_method == 'patch':
            image, mask = select_patch(image, mask, size, color_mode)

        else:
            raise Exception("unknown resize method %s" % resize_method)

        # reduce dim added before
        if mask is not None:
            mask = tf.squeeze(mask, axis=-1)

    return image, mask


def resize(image, mask, size, resize_method):
    # make 3dim for tf.image.resize
    if mask is not None:
        mask = tf.expand_dims(mask, axis=-1)

    # augmentatations
    if resize_method == 'resize':
        image = tf.image.resize(image, size, antialias=True)
        if mask is not None:
            mask = tf.image.resize(mask, size, method='nearest')  # use nearest for no interpolation

    elif resize_method == 'resize_with_pad':
        image = tf.image.resize_with_pad(image, size[0], size[1], antialias=True)
        if mask is not None:
            mask = tf.image.resize_with_pad(mask, size[0], size[1], method='nearest')  # use nearest for no interpolation

    elif resize_method == 'resize_with_crop_or_pad':
        image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])
        if mask is not None:
            mask = tf.image.resize_with_crop_or_pad(mask, size[0], size[1])  # use nearest for no interpolation

    elif resize_method == 'patch':
        image, mask = select_patch(image, mask, size, None)

    else:
        raise Exception("unknown resize method %s" % resize_method)

    # reduce dim added before
    if mask is not None:
        mask = tf.squeeze(mask, axis=-1)

    return image, mask


def process_mask_v2(mask, scale_mask: bool, num_classes: int):
    print(f"using mask scaling={scale_mask} num_classes={num_classes}")
    if scale_mask:
        mask = tf.cast(mask, tf.float32)
        if num_classes > 2:
            mask = mask / (tf.convert_to_tensor(num_classes, tf.float32) - 1.0)
    else:
        mask = tf.cast(mask, tf.int64)
        mask = tf.one_hot(mask, tf.convert_to_tensor(num_classes, tf.int32))
    return mask


def get_preprocess_fn_v2(size: Optional[tuple], num_classes: int, resize_method: str,
                         scale_mask: bool = False,
                         multiscale: Dict[str, Tuple[int, int]] = {}):

    @tf.function
    def map_fn(image, mask, sample_weight=None):
        # expect image and mask to be dtype=uint8
        image = tf.image.convert_image_dtype(image, tf.float32)

        # resize method for image create float32 image anyway
        # image, mask = resize_and_change_color(image, mask, size, color_mode, resize_method=resize_method, mode=mode)
        if size != None:
            image, mask = resize(image, mask, size, resize_method)

        mask = process_mask_v2(mask, scale_mask, num_classes)

        if len(list(multiscale.keys())) > 0:
            logger.info("apply multiscaling to dataset output masks")

            outputs = {}

            for i, (name, multi_scale_size) in enumerate(multiscale.items()):
                logger.debug(f"[{name}[{i}]: {multi_scale_size}")

                mask_output = tf.identity(mask, name=f"mutliscale-mask-{i}")

                if num_classes == 1 or scale_mask:
                    mask_output = tf.expand_dims(mask_output, axis=-1)

                scaled_output = tf.image.resize(mask_output, multi_scale_size, antialias=False, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                if num_classes == 1 or scale_mask:
                    scaled_output = tf.squeeze(scaled_output, axis=-1)

                outputs[name] = scaled_output

            if sample_weight is None:
                return image, (outputs, sample_weight)
            else:
                return image, outputs
        else:
            if sample_weight is None:
                return image, mask
            else:
                return image, (mask, sample_weight)

    return map_fn


def get_preprocess_fn(size, color_mode, resize_method, scale_mask=False, mode='graph'):

    @tf.function
    def process_mask(mask, num_classes):
        if scale_mask:
            num_classes = tf.cast(num_classes, tf.float32)
            mask = tf.cast(mask, tf.float32)
            mask = mask / (num_classes - 1.0)
        else:
            num_classes = tf.cast(num_classes, tf.int32)  # cast for onehot to accept it
            mask = tf.cast(mask, tf.int64)
            mask = tf.one_hot(mask, num_classes)
        return mask

    @tf.function
    def map_fn(image, mask, num_classes):

        # scale between 0 and 1
        image = tf.image.convert_image_dtype(image, tf.float32)

        # resize method for image create float32 image anyway
        image, mask = resize_and_change_color(image, mask, size, color_mode, resize_method=resize_method, mode=mode)
        mask = process_mask(mask, num_classes)

        return image, mask

    return map_fn


def select_patch(image, mask, patch_size, color_mode):
    """
    Select a random patch on image, mask at the same location

    Args:
        image (tf.Tensor): Tensor for the input image of shape (h x w x (1 or 3)), dtype: float32
        mask (tf.Tensor): Tensor for the mask image of shape (h x w x 1), dtype: uint8
        patch_size (tuple): Size of patch (height, width)
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Tuple of tensors (image, mask) with shape (patch_size[0], patch_size[1], 3)
    """
    image = tf.image.convert_image_dtype(image, tf.uint8)
    if color_mode == None:
        color_mode = ColorMode.RGB if image.shape[2] == 3 else ColorMode.GRAY

    if color_mode == ColorMode.RGB:
        # use alpha channel for mask
        concat = tf.concat([image, mask], axis=-1)
        patches = tf.image.random_crop(concat, size=[patch_size[0], patch_size[1], 4])
        patch_image = tf.image.convert_image_dtype(patches[:, :, :3], tf.float32)
        patch_mask = tf.expand_dims(patches[:, :, 3], axis=-1)
    else:
        stack = tf.stack([image, mask], axis=0)
        patches = tf.image.random_crop(stack, size=[2, patch_size[0], patch_size[1], 1])
        patch_image = patches[0]
        patch_mask = patches[1]

    return (patch_image, patch_mask)


def prepare_dataset(dataset, batch_size, buffer_size=200, repeat=0, take=0, skip=0, num_workers=1, worker_index=0, cache=False, shuffle=False, prefetch=True, augment_fn=None):

    print(locals())
    if num_workers > 1:
        dataset = dataset.shard(num_workers, worker_index)

    if skip > 0:
        dataset = dataset.skip(skip)

    if take > 0:
        dataset = dataset.take(take)

    if shuffle:  # shuffle before repeat to have the maximum randomness
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    if repeat > 0:
        dataset = dataset.repeat(repeat)
    else:
        dataset = dataset.repeat()

    # `tf.data.Options()` object then setting `options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA`
    if num_workers > 1 and utils.tf_version_gt_eq('2.4'):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
    else:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)

    dataset = dataset.batch(batch_size)

    if cache:  # cache after batching
        dataset = dataset.cache()

    if augment_fn:  # memory intensive task
        dataset = dataset.map(augment_fn, num_parallel_calls=multiprocessing.cpu_count())

    if prefetch:  # prefetch at the end
        dataset = dataset.prefetch(buffer_size // batch_size)

    return dataset


def random_flip_lr(**args):
    bool_right_left = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) >= THRESHOLD
    for k, x in args.items():
        x = tf.cond(bool_right_left, lambda: tf.image.flip_left_right(x), lambda: x)
        args[k] = x
    return args


def random_bw(**args):

    def _invert_colors(x):
        gray = tf.image.rgb_to_grayscale(x)
        return tf.image.grayscale_to_rgb(gray)

    bool_right_left = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) >= THRESHOLD
    for k, x in args.items():
        x = tf.cond(bool_right_left, lambda: _invert_colors(x), lambda: x)
        args[k] = x

    return args


def random_rot90(**args):
    condition = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) >= THRESHOLD
    for k, x in args.items():
        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis=-1)
            x = tf.cond(condition, lambda: tf.image.rot90(x, 1), lambda: x)
            x = tf.squeeze(x, axis=-1)
        else:
            x = tf.cond(condition, lambda: tf.image.rot90(x, 1), lambda: x)

        args[k] = x
    return args


def random_flip_ud(**args):
    bool_up_down = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) >= THRESHOLD
    for k, x in args.items():
        x = tf.cond(bool_up_down, lambda: tf.image.flip_up_down(x), lambda: x)
        args[k] = x
    return args


def random_rot(**args):
    import tensorflow_addons as tfa

    cond = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) >= THRESHOLD

    angle = tf.random.uniform(shape=[], min_val=0, maxval=1.0, dtype=tf.float32)
    angle = tf.cast(angle * 360, tf.int32)

    for k, x in args.items():
        args[k] = tf.cond(cond, lambda: tfa.image.rotate(x, angle), lambda: x)

    return args


def random_rot180(**args):

    cond = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) >= THRESHOLD

    def _apply_rot180(x):
        return tf.image.rot90(x, 2)

    for k, x in args.items():
        args[k] = tf.cond(cond, lambda: _apply_rot180(x), lambda: x)

    return args


def random_color(**args):

    bool_color = tf.random.uniform(shape=[], minval=0.0, maxval=1.0, dtype=tf.float32) >= THRESHOLD

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
        crops = tf.image.crop_and_resize(img, boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=size, method='nearest')
        return crops[idx]

    idxs = tf.random.uniform(shape=[batch_size], minval=0, maxval=len(scales), dtype=tf.int32)
    cond = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32) >= THRESHOLD

    for k, x in args.items():
        # Only apply cropping 50% of the time
        elems = [(args[k][i, :, :, :], idxs[i]) for i in range(batch_size)]
        args[k] = tf.cond(cond, lambda: x, lambda: tf.map_fn(random_crop, elems=elems, dtype=args[k].dtype))

    return args


augmentation_methods = ['rot180', 'flip_lr', 'flip_ud', 'color', 'rot90', 'bw', 'rot']


def get_augment_fn(size, batch_size, methods=augmentation_methods):

    for method in methods:
        if method not in augmentation_methods:
            raise Exception("cannot find augmentation method %s, please choose one of %s" % (method, str(augmentation_methods)))

    def augment(images, masks):

        # random rotation
        if 'rot180' in methods:
            rot180 = random_rot180(images=images, masks=masks)
            images, masks = rot180['images'], rot180['masks']

        if 'rot90' in methods:
            rot90 = random_rot90(images=images, masks=masks)
            images, masks = rot90['images'], rot90['masks']

        # random flipping
        if 'flip_lr' in methods:
            flip = random_flip_lr(images=images, masks=masks)
            images, masks = flip['images'], flip['masks']

        if 'flip_ud' in methods:
            flip = random_flip_ud(images=images, masks=masks)
            images, masks = flip['images'], flip['masks']

        # if 'rot' in methods:
        #     rot = random_bw(images=images, masks=masks)
        #     images, masks = rot['images'], rot['masks']

        if 'bw' in methods:
            bw = random_bw(images=images)
            images = bw['images']

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
