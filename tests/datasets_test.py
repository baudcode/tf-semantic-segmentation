from tf_semantic_segmentation import datasets
from tf_semantic_segmentation.datasets import tfrecord
from tf_semantic_segmentation.datasets.utils import DataType

import os
import tempfile
import numpy as np


def test_toy_dataset():
    ds = datasets.toy.Toy(os.path.join(tempfile.tempdir, 'TOY'))
    gen = ds.get()
    inputs, targets = next(gen())
    assert(len(targets.shape) == 2)
    assert(targets.shape[:2] == inputs.shape[:2])
    assert(targets.dtype == np.uint8)
    assert(inputs.dtype == np.uint8)

    assert(ds.num_examples(DataType.TRAIN) > 100)
    assert(ds.num_examples(DataType.TEST) > 10)
    assert(ds.num_examples(DataType.VAL) > 10)


def test_write_and_read_cmp_facade_tfrecord():
    cache_dir = os.path.join(tempfile.gettempdir(), 'TOY')
    ds = datasets.toy.Toy(cache_dir)
    record_dir = os.path.join(tempfile.gettempdir(), 'TOY', 'records')
    writer = tfrecord.TFWriter(record_dir)
    writer.write(ds)

    reader = tfrecord.TFReader(record_dir)
    dataset = reader.get_dataset(DataType.TRAIN)
    for image, labels, num_classes in dataset:
        print(image.shape, labels.shape, num_classes)
        break

    assert(writer.num_written(DataType.TRAIN) == ds.num_examples(DataType.TRAIN))
    assert(writer.num_written(DataType.TEST) == ds.num_examples(DataType.TEST))
    assert(writer.num_written(DataType.VAL) == ds.num_examples(DataType.VAL))

    assert(reader.num_examples(DataType.TRAIN) == writer.num_written(DataType.TRAIN))
    assert(reader.num_examples(DataType.TEST) == writer.num_written(DataType.TEST))
    assert(reader.num_examples(DataType.VAL) == writer.num_written(DataType.VAL))
