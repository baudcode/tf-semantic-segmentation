import argparse
from ..visualizations import show
from ..datasets.tfrecord import TFReader
from ..processing import pre_dataset
from ..datasets import DataType
import numpy as np
import tensorflow as tf


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--record_dir')
    parser.add_argument('-t', '--take', default=8, type=int)
    return parser.parse_args()


def main():

    args = get_args()
    dataset = TFReader(args.record_dir).get_dataset(DataType.TRAIN)
    dataset = dataset.shuffle(50)
    dataset = dataset.take(args.take)

    for image, _target, num_classes in dataset:
        print(_target.numpy().max(), _target.numpy().min(), _target.numpy().mean())
        target = tf.image.resize(tf.expand_dims(_target, axis=-1), (128, 128), method='nearest')
        print("after resize:", target.numpy().max(), target.numpy().min(), target.numpy().mean())
        target = tf.squeeze(target, axis=-1)
        target = tf.cast(target, tf.int64)
        target = tf.one_hot(target, tf.cast(num_classes, tf.int32))
        target = tf.argmax(target, axis=-1)
        print('after argmax: ', target.numpy().max(), target.numpy().min(), target.numpy().mean())
        show.show_images([image.numpy(), target.numpy().astype(np.float32), _target.numpy().astype(np.float32)])


if __name__ == "__main__":
    main()
