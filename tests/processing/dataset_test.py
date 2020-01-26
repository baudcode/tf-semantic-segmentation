from tf_semantic_segmentation.processing import dataset, ColorMode
from tf_semantic_segmentation.datasets.utils import convert2tfdataset, DataType
import numpy as np
from ..fixtures import dataset as ds


def test_preprocess_fn():
    size = (64, 64)
    color_mode = ColorMode.RGB
    resize_method = 'resize'

    f = dataset.get_preprocess_fn(size, color_mode, resize_method, scale_mask=False)

    num_classes = 3
    image = np.ones((32, 32, 1), np.uint8) * 255
    mask = np.ones((32, 32), np.uint8) * 2

    pimage, pmask = f(image, mask, num_classes)
    assert(pimage.shape == (64, 64, 3))
    assert(pmask.shape == (64, 64, num_classes))
    pmask_argmax = np.argmax(pmask.numpy(), axis=-1)
    np.testing.assert_array_equal(pmask_argmax, np.ones_like(pmask_argmax) * 2)

    # scale masks between 0 and 1.0
    f = dataset.get_preprocess_fn(size, color_mode, resize_method, scale_mask=True)
    pimage, pmask = f(image, mask, num_classes)

    assert(pmask.shape == (64, 64))
    assert(pmask.numpy().max() <= 1.0)


def test_prepare_dataset(ds):
    for dt in DataType.get():
        tfds = convert2tfdataset(ds, dt)
        tfds = tfds.map(dataset.get_preprocess_fn((64, 64), ColorMode.RGB, 'resize', False))
        tfds = dataset.prepare_dataset(tfds, 2)
        for image_batch, mask_batch in tfds:
            assert(image_batch.shape == (2, 64, 64, 3))
            assert(mask_batch.shape == (2, 64, 64, ds.num_classes))
            break

    tfds = convert2tfdataset(ds, DataType.TRAIN)
    tfds = tfds.map(dataset.get_preprocess_fn((64, 64), ColorMode.RGB, 'resize', True))
    tfds = dataset.prepare_dataset(tfds, 2)
    for image_batch, mask_batch in tfds:
        assert(image_batch.shape == (2, 64, 64, 3))
        assert(mask_batch.shape == (2, 64, 64))
        break

    batch_size, size = 2, (64, 64)
    augment_fn = dataset.get_augment_fn(size, batch_size)
    tfds = convert2tfdataset(ds, DataType.TRAIN)
    tfds = tfds.map(dataset.get_preprocess_fn(size, ColorMode.RGB, 'resize', False))
    tfds = dataset.prepare_dataset(tfds, batch_size, augment_fn=augment_fn)

    for image_batch, mask_batch in tfds:
        assert(image_batch.shape == (2, size[0], size[1], 3))
        assert(mask_batch.shape == (2, size[0], size[1], ds.num_classes))
        break
