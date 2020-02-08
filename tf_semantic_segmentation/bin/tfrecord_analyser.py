import argparse

from ..datasets import tfrecord, DataType
from ..utils import get_size
import tensorflow as tf
import logging
import imageio
import numpy as np

from ..settings import logger, logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', required=True)
    parser.add_argument('-d', '--dump_example', action='store_true')
    parser.add_argument('-m', '--mean', action='store_true')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)

    reader = tfrecord.TFReader(args.record_dir)
    for image, mask, num_classes in reader.get_dataset(DataType.TRAIN):
        if args.dump_example:
            imageio.imwrite('_example.png', (image.numpy() * 255.).astype(np.uint8))
            imageio.imwrite('_example_mask.png', (mask.numpy() * 255. / (num_classes.numpy() - 1)).astype(np.uint8))
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
        if args.mean:
            n, mean = reader.num_examples_and_mean(data_type)
            print("-> mean[%s(%d)] = %s" % (data_type, n, mean.tolist()))
        else:
            n = reader.num_examples(data_type)
            print(data_type, ":", n)
        sizes.append(n)
    print("-> total: %d" % sum(sizes))
    print("=" * 20)
    print("size: %.2f GB" % (get_size(args.record_dir) / 1024. / 1024. / 1024.))


if __name__ == "__main__":
    main()
