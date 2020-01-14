from ..datasets import datasets_by_name, get_dataset_by_name, get_cache_dir
from ..processing.dataset import ColorMode, resize_and_change_color
from ..datasets.tfrecord import TFWriter
from ..settings import logger

import tensorflow as tf
import argparse
import os


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=list(datasets_by_name.keys()), required=True)
    parser.add_argument('-c', '--data_dir', required=True)
    parser.add_argument('-num', '--num_examples_per_record', default=100, type=int)
    parser.add_argument('-r', '--record_dir', default=None)
    parser.add_argument('-s', '--size', default=None, type=lambda x: list(map(int, x.split(','))))
    parser.add_argument('-rm', '--resize_method', default='resize_with_pad')
    parser.add_argument('-cm', '--color_mode', default=ColorMode.NONE, type=int)

    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    cache_dir = get_cache_dir(args.data_dir, args.dataset.lower())
    ds = get_dataset_by_name(args.dataset, cache_dir)

    # write records
    if args.record_dir:
        record_dir = args.record_dir
    else:
        record_dir = os.path.join(cache_dir, 'records', args.dataset.lower())

    def preprocess_fn(image, labels):
        image = tf.image.convert_image_dtype(image, tf.float32)
        return resize_and_change_color(image, labels, args.size, args.color_mode, args.resize_method)

    logger.info('wrting records to %s' % record_dir)
    writer = TFWriter(record_dir)
    writer.write(ds, overwrite=args.overwrite, num_examples_per_record=args.num_examples_per_record, preprocess_fn=preprocess_fn)

    # validate number of examples written
    writer.validate(ds)


if __name__ == "__main__":
    main()
