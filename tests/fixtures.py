from tf_semantic_segmentation.datasets import Dataset, DataType, utils, TFReader, TFWriter
from tf_semantic_segmentation.datasets.shapes import ShapesDS

import pytest
import os
import tqdm
import cv2
import numpy as np
import imageio
import tempfile
import pytest
import shutil


class TestDataset(Dataset):

    def __init__(self, cache_dir, num=100, size=(64, 64)):
        super(TestDataset, self).__init__(cache_dir)
        self._num = num
        self._size = size
        self.split = self._generate()

    def _generate(self):
        data = []
        for i in tqdm.trange(self._num, desc='creating test dataset'):
            image = np.zeros((self._size[0], self._size[1], 3), np.uint8)
            mask = np.zeros((self._size[0], self._size[1]), np.uint8)
            center = (self._size[1] // 2, self._size[0] // 2)
            image = cv2.circle(image, center, 5, (255, 0, 0), thickness=cv2.FILLED)
            mask = cv2.circle(mask, center, 5, 1, thickness=cv2.FILLED)
            mask_path = os.path.join(self.cache_dir, 'mask-%d.png')
            image_path = os.path.join(self.cache_dir, '%d.jpg')
            imageio.imwrite(image_path, image)
            imageio.imwrite(mask_path, mask)
            data.append((image_path, mask_path))

        return utils.get_split(data)

    def raw(self):
        return self.split

    def delete(self):
        shutil.rmtree(self.cache_dir)

    @property
    def labels(self):
        return ['bg', 'circle']


@pytest.fixture()
def dataset(request):
    cache_dir = os.path.join(tempfile.tempdir, 'testds')  # tempfile.mkdtemp()
    os.makedirs(cache_dir, exist_ok=True)
    ds = TestDataset(cache_dir, num=100, size=(64, 64))
    yield ds
    ds.delete()


@pytest.fixture()
def tfrecord_reader():
    cache_dir = os.path.join(tempfile.tempdir, 'testds')  # tempfile.mkdtemp()
    record_dir = os.path.join(cache_dir, 'records')

    os.makedirs(cache_dir, exist_ok=True)
    ds = TestDataset(cache_dir, num=100, size=(64, 64))
    TFWriter(record_dir).write(ds)
    reader = TFReader(record_dir)

    yield reader
    ds.delete()


@pytest.fixture()
def shapes_ds():
    """ Returns the dataset fixture """
    return ShapesDS(os.path.join(tempfile.tempdir, 'SHAPES'), num_examples=100)
