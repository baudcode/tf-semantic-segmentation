import argparse

from ..datasets import tfrecord, DataType
from ..utils import get_size
import tensorflow as tf
import logging

from ..settings import logger, logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', required=True)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    reader = tfrecord.TFReader(args.record_dir)

    for image, mask, _ in reader.get_dataset(DataType.TRAIN):
        print("=" * 20)
        print("image/mask stats:")
        print("image: ", image.dtype, "(shape=", image.shape, ",max=", image.numpy().max(), ")")
        print("mask: ", mask.dtype, "(shape=", mask.shape, ",max=", mask.numpy().max(), ")")
        break

    print("=" * 20)
    print("num_classes: ", reader.num_classes)
    print("size: ", reader.size)
    print("input shape: ", reader.input_shape)
    print('Calculating entries...')
    sizes = []
    for data_type in [DataType.TRAIN, DataType.TEST, DataType.VAL]:
        n = reader.num_examples(data_type)
        print(data_type, ":", n)
        sizes.append(n)
    print("-> total: %d" % sum(sizes))
    print("=" * 20)
    print("size: %.2f GB" % (get_size(args.record_dir) / 1024. / 1024. / 1024.))


if __name__ == "__main__":
    main()
