from tf_semantic_segmentation.datasets import Dataset, DataType
import pytest
from ..fixtures import dataset
import types


@pytest.mark.usefixtures('dataset')
def test_dataset_structure(dataset):
    ds = dataset
    assert(ds.num_examples(DataType.TRAIN) == 80)
    assert(ds.num_examples(DataType.VAL) == 10)
    assert(ds.num_examples(DataType.TEST) == 10)
    assert(ds.total_examples() == 100)
    assert(ds.num_classes == 2)
    assert(all(map(lambda x: x in ds.raw().keys(), DataType.get())))

    image, mask = ds.get_random_item()
    assert(image.shape == (64, 64, 3) and image.dtype == 'uint8')
    assert(mask.shape == (64, 64) and mask.dtype == 'uint8')
    assert(mask.max() == 1)

    g = ds.get()()
    assert(isinstance(g, types.GeneratorType))
