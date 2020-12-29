from ..datasets import datasets_by_name, get_dataset_by_name, get_cache_dir, DirectoryDataset
from ..processing.dataset import ColorMode, resize_and_change_color
from ..datasets.tfrecord import TFWriter
from ..settings import logger

import tensorflow as tf
import argparse
import os


def write_records_from_directory(directory, record_dir, size=None, color_mode=ColorMode.NONE,
                                 resize_method='resize', num_examples_per_record=100, overwrite=False):
    ds = DirectoryDataset(directory)

    if record_dir is None:
        raise AssertionError("record_dir cannot be None")

    write_records(ds, record_dir, size=size, color_mode=color_mode, resize_method=resize_method,
                  num_examples_per_record=num_examples_per_record, overwrite=overwrite)


def write_records_from_dataset_name(dataset, data_dir, record_dir=None, size=None, color_mode=ColorMode.NONE,
                                    resize_method='resize', num_examples_per_record=100, overwrite=False):

    cache_dir = get_cache_dir(data_dir, dataset.lower())
    ds = get_dataset_by_name(dataset, cache_dir)

    # write records
    if record_dir:
        record_dir = record_dir
    else:
        record_dir = os.path.join(cache_dir, 'records', dataset.lower())

    write_records(ds, record_dir, size=size, color_mode=color_mode, resize_method=resize_method,
                  num_examples_per_record=num_examples_per_record, overwrite=overwrite)

    return record_dir


def write_records(ds, record_dir, size=None, color_mode=ColorMode.NONE, resize_method='resize',
                  num_examples_per_record=100, overwrite=False):

    def preprocess_fn(image, mask):
        image = tf.image.convert_image_dtype(image, tf.float32)
        return resize_and_change_color(image, mask, size, color_mode, resize_method)

    logger.info('wrting records to %s' % record_dir)
    writer = TFWriter(record_dir)
    writer.write(ds, overwrite=overwrite, num_examples_per_record=num_examples_per_record, preprocess_fn=preprocess_fn)

    # validate number of examples written
    writer.validate(ds)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=None, choices=list(datasets_by_name.keys()))
    parser.add_argument('-r', '--record_dir', default=None)

    parser.add_argument('-dir', '--directory', default=None)
    parser.add_argument('-c', '--data_dir', default=None)
    parser.add_argument('-num', '--num_examples_per_record', default=100, type=int)
    parser.add_argument('-s', '--size', default=None, type=lambda x: list(map(int, x.split(','))), help='height,width')
    parser.add_argument('-rm', '--resize_method', default='resize')
    parser.add_argument('-cm', '--color_mode', default=ColorMode.NONE, type=int)

    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    if args.directory is None and args.dataset is None:
        raise AssertionError("please either supply a dataset or a directory containing your data")

    if args.dataset:
        assert(args.data_dir is not None), "data_dir argument is required"
        write_records_from_dataset_name(args.dataset, args.data_dir, args.record_dir, args.size, args.color_mode, args.resize_method,
                                        args.num_examples_per_record, args.overwrite)
    else:
        write_records_from_directory(args.directory, args.record_dir, args.size, args.color_mode, args.resize_method,
                                     args.num_examples_per_record, args.overwrite)


if __name__ == "__main__":
    main()
