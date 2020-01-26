from tf_semantic_segmentation.processing import image
import numpy as np


def test_grid_vis():
    x = np.zeros((9, 16, 16))
    montage = image.grayscale_grid_vis(x, 3, 3)
    assert(montage.shape == (48, 48))


def test_fixed_resize():
    x = np.zeros((16, 16, 3))
    assert(image.fixed_resize(x, width=32).shape == (32, 32, 3))
    assert(image.fixed_resize(x, height=32).shape == (32, 32, 3))

    x = np.zeros((16, 16, 1))
    assert(image.fixed_resize(x, width=32).shape == (32, 32, 1))
    assert(image.fixed_resize(x, height=32).shape == (32, 32, 1))

    x = np.zeros((32, 16, 1))
    assert(image.fixed_resize(x, width=32).shape == (64, 32, 1))
    assert(image.fixed_resize(x, height=32).shape == (32, 16, 1))
