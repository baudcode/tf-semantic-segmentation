import argparse
from ..visualizations import show
from ..datasets.tfrecord import TFReader
from ..processing import dataset as ds_preprocessing
from ..datasets import DataType
import numpy as np
import tensorflow as tf


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir', required=True)
    parser.add_argument('-t', '--take', default=8, type=int)
    return parser.parse_args()


def main():

    args = get_args()
    reader = TFReader(args.record_dir)
    dataset = reader.get_dataset(DataType.TRAIN)
    dataset = dataset.shuffle(50)
    dataset = dataset.take(args.take)
    print("num_classes:", reader.num_classes)
    print("input shape: ", reader.input_shape)

    for image, _target, num_classes in dataset:
        show.show_images([image.numpy(), _target.numpy().astype(np.float32)], titles=['input', 'target'])


if __name__ == "__main__":
    main()
