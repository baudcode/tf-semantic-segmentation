from ..datasets import datasets_by_name, get_dataset_by_name, get_cache_dir, DataType
from ..datasets.utils import convert2tfdataset
from ..processing.dataset import ColorMode, resize_and_change_color
from ..datasets.tfrecord import TFWriter
from ..settings import logger

import tensorflow as tf
import argparse
import os
import imageio
import tqdm
import multiprocessing


def export(ds, output_dir, size=None, resize_method="resize_with_pad", color_mode=ColorMode.NONE, overwrite=False, batch_size=4):

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'labels.txt'), 'w') as writer:
        for label in ds.labels:
            writer.write(label.strip() + "\n")

    def preprocess_fn(image, mask, num_classes):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image, mask = resize_and_change_color(image, mask, size, color_mode, resize_method)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        return image, mask

    for data_type in DataType.get():
        masks_dir = os.path.join(output_dir, data_type, 'masks')
        images_dir = os.path.join(output_dir, data_type, 'images')

        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        tfds = convert2tfdataset(ds, data_type)
        tfds = tfds.map(preprocess_fn, num_parallel_calls=multiprocessing.cpu_count())
        tfds = tfds.batch(batch_size)
        tfds = tfds.repeat(1)

        total_examples = ds.num_examples(data_type)
        for k, (images, masks) in tqdm.tqdm(enumerate(tfds), total=total_examples // batch_size, desc="exporting",
                                            postfix=dict(data_type=data_type, batch_size=batch_size, total=total_examples)):
            for g in range(len(images)):
                i = k * batch_size + g
                image = images[g]
                mask = masks[g]

                image_path = os.path.join(images_dir, '%d.png' % i)
                mask_path = os.path.join(masks_dir, '%d.png' % i)
                if not os.path.exists(image_path) or (os.path.exists(image_path) and overwrite):
                    imageio.imwrite(image_path, image)
                    imageio.imwrite(mask_path, mask)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=list(datasets_by_name.keys()), required=True)
    parser.add_argument('-c', '--data_dir', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-s', '--size', default=None, type=lambda x: list(map(int, x.split(','))))
    parser.add_argument('-rm', '--resize_method', default='resize_with_pad')
    parser.add_argument('-cm', '--color_mode', default=ColorMode.NONE, type=int)
    parser.add_argument('-overwrite', '--overwrite', action='store_true')

    args = parser.parse_args()

    cache_dir = get_cache_dir(args.data_dir, args.dataset.lower())
    ds = get_dataset_by_name(args.dataset, cache_dir)

    logger.info('wrting dataset to %s' % args.output_dir)
    export(ds, args.output_dir, args.size, args.resize_method, args.color_mode, args.overwrite, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
