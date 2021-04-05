import argparse

from ..datasets import tfrecord, DataType
from ..utils import get_size
import tensorflow as tf
import logging
import imageio
import numpy as np
import itertools

from ..settings import logger, logging
from ..visualizations import show, masks as mutils
from ..processing import ColorMode, dataset as preprocessing_ds

preprocessing_ds.THRESHOLD = 0.0


def str_list_type(x): return list(map(str, x.split(",")))


def float_list(x): return list(map(float, x.split(',')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', required=True)
    parser.add_argument('-s', '--size', type=float_list, default=[512, 512])
    parser.add_argument('-a', '--augmentations', default=preprocessing_ds.augmentation_methods, type=str_list_type)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    batch_size = 16
    print(args.augmentations)

    reader = tfrecord.TFReader(args.record_dir)
    num_classes = reader.num_classes
    ds = reader.get_dataset(DataType.TRAIN)
    train_preprocess_fn = preprocessing_ds.get_preprocess_fn(args.size, ColorMode.RGB, resize_method='resize', scale_mask=False)
    ds = ds.map(train_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(1)
    for images, masks in ds:
        break

    images = images.numpy()
    masks = masks.numpy()

    images_batch = []
    masks_batch = []

    images_batch.append(images[0])
    masks_batch.append(masks[0])

    for c in args.augmentations:
        print(c)
        augment_fn = preprocessing_ds.get_augment_fn(args.size, 1, methods=[c])
        aug_images, aug_masks = augment_fn(images, masks)
        images_batch.append(aug_images[0])
        masks_batch.append(aug_masks[0])

    images = mutils.get_colored_segmentation_mask(np.asarray(masks_batch), num_classes, images=np.asarray(images_batch))

    # mask_batch = np.argmax(mask_batch, axis=-1)
    # image_batch = (image_batch.numpy() * 255.).astype(np.uint8)
    # mask_batch = (mask_batch * 255. / (num_classes - 1)).astype(np.uint8)

    # images = [m for m in mask_batch]
    # images += [m for m in image_batch]
    show.show_images(images, cols=2, titles=['normal'] + args.augmentations)


if __name__ == "__main__":
    main()
