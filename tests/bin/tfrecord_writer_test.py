from tf_semantic_segmentation.debug import dataset_export
from tf_semantic_segmentation.bin import tfrecord_writer
from tf_semantic_segmentation.datasets import TFReader, DataType
from tf_semantic_segmentation.processing import ColorMode
from ..fixtures import dataset
import tempfile
import shutil
import os


def test_export(dataset):
    output_dir = tempfile.mkdtemp()
    record_dir = os.path.join(output_dir, 'records')

    dataset_export.export(dataset, output_dir, size=None, color_mode=ColorMode.RGB, overwrite=True)
    tfrecord_writer.write_records_from_directory(output_dir, record_dir)
    reader = TFReader(record_dir)
    tfds = reader.get_dataset(DataType.TRAIN)

    for image, mask, num_classes in tfds:
        assert(image.numpy().shape == (64, 64, 3))
        assert(mask.numpy().shape == (64, 64))
        assert(num_classes.numpy() == dataset.num_classes)
        break

    dataset_export.export(dataset, output_dir, size=(32, 32), color_mode=ColorMode.RGB, overwrite=True)
    tfrecord_writer.write_records_from_directory(output_dir, record_dir, overwrite=True)
    reader = TFReader(record_dir)
    tfds = reader.get_dataset(DataType.TRAIN)

    for image, mask, num_classes in tfds:
        assert(image.numpy().shape == (32, 32, 3))
        assert(mask.numpy().shape == (32, 32))
        assert(num_classes.numpy() == dataset.num_classes)
        break

    dataset_export.export(dataset, output_dir, size=(31, 31), color_mode=ColorMode.GRAY, overwrite=True)
    tfrecord_writer.write_records_from_directory(output_dir, record_dir, overwrite=True)
    reader = TFReader(record_dir)
    tfds = reader.get_dataset(DataType.TRAIN)

    for image, mask, num_classes in tfds:
        assert(image.numpy().shape == (31, 31, 1))
        assert(mask.numpy().shape == (31, 31))
        assert(num_classes.numpy() == dataset.num_classes)
        break

    shutil.rmtree(output_dir)


def test_export_dataset_by_name():
    # test export dataset
    output_dir = tempfile.mkdtemp()
    record_dir = os.path.join(output_dir, 'records')
    name = 'shapesmini'
    num_classes = 2

    record_dir = tfrecord_writer.write_records_from_dataset_name(name, output_dir, size=None, overwrite=True)
    reader = TFReader(record_dir)
    tfds = reader.get_dataset(DataType.TRAIN)

    for image, mask, num_classes in tfds:
        assert(image.numpy().shape == (32, 32, 1))
        assert(mask.numpy().shape == (32, 32))
        assert(num_classes.numpy() == num_classes)
        break

    record_dir = tfrecord_writer.write_records_from_dataset_name(name, output_dir, size=(32, 32), overwrite=True)
    reader = TFReader(record_dir)
    tfds = reader.get_dataset(DataType.TRAIN)

    for image, mask, num_classes in tfds:
        assert(image.numpy().shape == (32, 32, 1))
        assert(mask.numpy().shape == (32, 32))
        assert(num_classes.numpy() == num_classes)
        break

    record_dir = tfrecord_writer.write_records_from_dataset_name(name, output_dir, size=(31, 31), color_mode=ColorMode.RGB, overwrite=True)
    reader = TFReader(record_dir)
    tfds = reader.get_dataset(DataType.TRAIN)

    for image, mask, num_classes in tfds:
        assert(image.numpy().shape == (31, 31, 3))
        assert(mask.numpy().shape == (31, 31))
        assert(num_classes.numpy() == num_classes)
        break

    shutil.rmtree(output_dir)
