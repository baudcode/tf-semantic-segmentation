from tf_semantic_segmentation.datasets import utils, DataType
from ..fixtures import dataset
import pytest
import copy
import tempfile
import numpy as np
import imageio
import os
import shutil


def test_all_data_types_exist():
    assert(utils.DataType.get() == ['train', 'test', 'val'])


@pytest.mark.usefictures('dataset')
def test_test_dataset(dataset):
    image, mask, num_classes = utils.test_dataset(dataset)

    assert(num_classes == dataset.num_classes)
    assert(image.shape == (64, 64, 3))
    assert(mask.shape == (64, 64))


@pytest.mark.usefictures('dataset')
def test_tfdataset(dataset):
    tfds = utils.convert2tfdataset(dataset, DataType.TRAIN)
    for image, mask, num_classes in tfds:
        assert(image.numpy().shape == (64, 64, 3) and image.numpy().dtype == 'uint8')
        assert(mask.numpy().shape == (64, 64) and mask.numpy().dtype == 'uint8')
        assert(dataset.num_classes == num_classes.numpy())
        break


def test_splits():
    # get_train_test_val_from_list
    l = list(range(100))
    train, test, val = utils.get_train_test_val_from_list(copy.deepcopy(l), train_split=0.8, val_split=0.5, shuffle=False)
    assert(len(train) == 80 and len(test) == 10 and len(val) == 10)
    assert(train == list(range(80)) and val == list(range(80, 90)) and test == list(range(90, 100)))

    train, test, val = utils.get_train_test_val_from_list(copy.deepcopy(l), train_split=0.8, val_split=0.5, shuffle=True)
    assert(list(sorted(train + test + val)) == l)
    train2, test2, val2 = utils.get_train_test_val_from_list(copy.deepcopy(l), train_split=0.8, val_split=0.5, shuffle=True, rand=lambda: 0.3)
    assert(train2 != train and test2 != 2 and val2 != val)

    train, val = utils.get_split_from_list(copy.copy(l), split=0.7)
    assert(train == list(range(70)) and val == list(range(70, 100)))

    images_dir = tempfile.mkdtemp()
    masks_dir = tempfile.mkdtemp()

    for i in range(10):
        imageio.imwrite(os.path.join(images_dir, '%d.png' % i), np.zeros((32, 32), np.uint8))
        imageio.imwrite(os.path.join(masks_dir, '%d.png' % i), np.zeros((32, 32), np.uint8))

    split = utils.get_split_from_dirs(images_dir, masks_dir, extensions=['png'], train_split=0.8, val_split=0.5)
    assert(all([dt in split.keys() for dt in DataType.get()]))
    assert(len(split[DataType.TRAIN]) == 8 and len(split[DataType.VAL]) == 1 and len(split[DataType.TEST]) == 1)

    shutil.rmtree(images_dir)
    shutil.rmtree(masks_dir)

    split = utils.get_split(copy.copy(l))
    assert(all([dt in split.keys() for dt in DataType.get()]))
    assert(len(split[DataType.TRAIN]) == 80 and len(split[DataType.VAL]) == 10 and len(split[DataType.TEST]) == 10)
