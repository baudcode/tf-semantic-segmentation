from .utils import DataType
from ..utils import get_files
from ..settings import logger

import tensorflow as tf
import os
import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, labels, image_shape, num_classes):
    feature = {
        'image': _bytes_feature(image),
        'labels': _bytes_feature(labels),
        'num_classes': _int64_feature(num_classes),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'labels': tf.io.FixedLenFeature((), tf.string),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64),
        'num_classes': tf.io.FixedLenFeature((), tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
    image_shape = [example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)

    labels = tf.io.parse_tensor(example['labels'], out_type=tf.uint8)
    labels_shape = [example['height'], example['width']]
    labels = tf.reshape(labels, labels_shape)

    return image, labels, example['num_classes']


class TFReader:

    def __init__(self, record_dir):
        self.record_dir = record_dir

    def get_dataset(self, data_type):
        record_dir = os.path.join(self.record_dir, data_type)
        files = get_files(record_dir, extensions=['tfrecord'])
        logger.debug("TFReader found these (%d) files: %s" % (len(files), str(files)))
        tfrecord_dataset = tf.data.TFRecordDataset(files, num_parallel_reads=8)
        return tfrecord_dataset.map(read_tfrecord)

    def num_examples(self, data_type):
        return sum([1 for _ in self.get_dataset(data_type)])

    @property
    def num_classes(self):
        for _, _, num_classes in self.get_dataset(DataType.TRAIN):
            return num_classes.numpy()


class TFWriter:

    def __init__(self, record_dir):
        self.record_dir = record_dir
        self.record_dirs = {
            DataType.TRAIN: os.path.join(record_dir, DataType.TRAIN),
            DataType.VAL: os.path.join(record_dir, DataType.VAL),
            DataType.TEST: os.path.join(record_dir, DataType.TEST),
        }
        self.reset_written()

    def reset_written(self):
        self._written = {
            DataType.TRAIN: 0,
            DataType.TEST: 0,
            DataType.VAL: 0
        }

    def validate(self, ds):
        assert(self.num_written(DataType.TRAIN) == ds.num_examples(DataType.TRAIN))
        assert(self.num_written(DataType.TEST) == ds.num_examples(DataType.TEST))
        assert(self.num_written(DataType.VAL) == ds.num_examples(DataType.VAL))

    def num_written(self, data_type):
        return self._written[data_type]

    def write(self, ds, overwrite=False, num_examples_per_record=1000):
        self.reset_written()
        for data_type, record_dir in self.record_dirs.items():
            os.makedirs(record_dir, exist_ok=True)
            num_classes = ds.num_classes
            num_examples = ds.num_examples(data_type)

            # calculate the number of record files
            num_record_files = int(num_examples / (num_examples_per_record - 1)) + 1
            record_files = [os.path.join(record_dir, 'data-%d.tfrecord' % i) for i in range(num_record_files)]

            # check if all files exists (abort writing)
            if not overwrite and all(map(lambda x: os.path.exists(x), record_files)):
                logger.info('tfrecord for data_type=%s in %s does already exist' % (data_type, record_dir))
                self._written[data_type] = TFReader(self.record_dir).num_examples(data_type)
                continue

            with tqdm.tqdm(total=num_examples) as tq:
                # iterate through every element in generator
                gen = ds.get(data_type)()
                for record_file in record_files:
                    tq.set_postfix(record_file=record_file)
                    with tf.io.TFRecordWriter(record_file) as writer:

                        for image, labels in gen:
                            # images to bytes
                            image_bytes = tf.io.serialize_tensor(image)
                            labels_bytes = tf.io.serialize_tensor(labels)

                            example = serialize_example(image_bytes, labels_bytes, image.shape, num_classes)
                            writer.write(example)

                            # update counters
                            tq.update(1)
                            self._written[data_type] += 1

                            # go out of the generator and into another file
                            if self._written[data_type] % num_examples_per_record == 0:
                                break
