from tf_semantic_segmentation.datasets import tfrecord, DataType
from ..fixtures import dataset

import os
import tempfile
import numpy as np
import pytest
import time
import shutil


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
    shutil.rmtree(record_dir)


def test_write_and_read_records_no_compression(dataset):
    write_and_read_records(dataset, "")


def test_write_and_read_records_gzip_compression(dataset):
    write_and_read_records(dataset, "GZIP")
