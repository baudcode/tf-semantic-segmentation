from tf_semantic_segmentation import datasets
from tf_semantic_segmentation.datasets import tfrecord
from tf_semantic_segmentation.datasets.utils import DataType

import os
import tempfile
import numpy as np
import pytest
import time


@pytest.fixture()
def ds():
    """ Returns the dataset fixture """
    return datasets.shapes.ShapesDS(os.path.join(tempfile.tempdir, 'SHAPES'), num_examples=100)


def test_toy_dataset(ds):

    gen = ds.get()
    inputs, targets = next(gen())
    assert(len(targets.shape) == 2)
    assert(targets.shape[:2] == inputs.shape[:2])
    assert(targets.dtype == np.uint8)
    assert(inputs.dtype == np.uint8)

    assert(ds.num_examples(DataType.TRAIN) == 80)
    assert(ds.num_examples(DataType.TEST) == 10)
    assert(ds.num_examples(DataType.VAL) == 10)


def write_and_read_records(ds, options):
    record_dir = os.path.join(tempfile.gettempdir(), str(time.time()), 'records')
    writer = tfrecord.TFWriter(record_dir, options=options)
    writer.write(ds)

    reader = tfrecord.TFReader(record_dir, options=options)
    dataset = reader.get_dataset(DataType.TRAIN)
    for image, mask, num_classes in dataset:
        print(image.shape, mask.shape, num_classes)
        break

    assert(writer.num_written(DataType.TRAIN) == ds.num_examples(DataType.TRAIN))
    assert(writer.num_written(DataType.TEST) == ds.num_examples(DataType.TEST))
    assert(writer.num_written(DataType.VAL) == ds.num_examples(DataType.VAL))

    assert(reader.num_examples(DataType.TRAIN) == writer.num_written(DataType.TRAIN))
    assert(reader.num_examples(DataType.TEST) == writer.num_written(DataType.TEST))
    assert(reader.num_examples(DataType.VAL) == writer.num_written(DataType.VAL))


def test_write_and_read_records_no_compression(ds):
    write_and_read_records(ds, "")


def test_write_and_read_records_gzip_compression(ds):
    write_and_read_records(ds, "GZIP")
