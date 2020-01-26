from tf_semantic_segmentation.debug import dataset_export
from tf_semantic_segmentation.processing import ColorMode
from tf_semantic_segmentation.utils import get_files
from ..fixtures import dataset
import tempfile
import shutil
import os


def test_dataset_export(dataset):
    output_dir = tempfile.mkdtemp()
    dataset_export.export(dataset, output_dir, size=(64, 64), color_mode=ColorMode.RGB, overwrite=True)
    files = get_files(output_dir)
    assert(len(files) == (dataset.total_examples() * 2) + 1)

    masks = get_files(output_dir, ['png'])
    images = get_files(output_dir, ['jpg'])

    assert(len(masks) == dataset.total_examples())
    assert(len(images) == dataset.total_examples())
    labels_path = os.path.join(output_dir, 'labels.txt')

    assert(os.path.exists(labels_path))
    assert(dataset_export.read_labels(labels_path) == dataset.labels)

    shutil.rmtree(output_dir)
