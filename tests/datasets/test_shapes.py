from tf_semantic_segmentation.datasets.shapes import DataType
import os
import tempfile
import numpy as np
import pytest
from ..fixtures import shapes_ds


@pytest.mark.usefixtures('shapes_ds')
def test_shapes_dataset(shapes_ds):

    gen = shapes_ds.get()
    inputs, targets = next(gen())
    assert(len(targets.shape) == 2)
    assert(targets.shape[:2] == inputs.shape[:2])
    assert(targets.dtype == np.uint8)
    assert(inputs.dtype == np.uint8)

    assert(shapes_ds.num_examples(DataType.TRAIN) == 80)
    assert(shapes_ds.num_examples(DataType.TEST) == 10)
    assert(shapes_ds.num_examples(DataType.VAL) == 10)
