from .dataset import Dataset, DataType
from ..utils import download_and_extract, get_files
from .utils import get_split_from_list

import os
import csv
import imageio
import numpy as np


def read(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        lines = [row for row in reader]
    return lines


class CUB2002011(Dataset):
    IMAEGS_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    MASKS_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz"

    modes = ['binary', 'category']

    def __init__(self, cache_dir, mode='binary'):
        super(CUB2002011, self).__init__(cache_dir)
        assert(mode in self.modes)
        self.mode = mode

        images_dir = download_and_extract(self.IMAEGS_URL, os.path.join(self.cache_dir, 'images'))
        masks_dir = download_and_extract(self.MASKS_URL, os.path.join(self.cache_dir, 'masks'))
        info_dir = os.path.join(images_dir, 'CUB_200_2011')

        self.images_dir = os.path.join(info_dir, 'images')
        self.masks_dir = os.path.join(masks_dir, 'segmentations')

        self._labels = self._read_labels(info_dir)
        self.image_id_to_class_id = self._read_image_id_to_class_id(info_dir)
        self.bounding_boxes = self._read_bounding_boxes(info_dir)
        self.image_id_to_filename = self._read_id_to_filename(info_dir)
        self.traindata, self.testdata = self._read_train_test_split(info_dir)

    @property
    def labels(self):
        if self.mode == 'category':
            return self._labels
        else:
            return ['bg', 'bird']

    def _read_image_id_to_class_id(self, d, filename='image_class_labels.txt'):
        data = read(os.path.join(d, filename))
        return {d[0]: d[1] for d in data}

    def _read_bounding_boxes(self, d, filename='bounding_boxes.txt'):
        data = read(os.path.join(d, filename))
        return {
            d[0]: {
                "x": d[1],
                "y": d[2],
                "width": d[3],
                "height": d[4]
            } for d in data
        }

    def _read_labels(self, d, filename='classes.txt'):
        data = read(os.path.join(d, filename))
        labels = [d[1] for d in data]
        labels = ['bg'] + labels
        return labels

    def _read_id_to_filename(self, d, filename='images.txt'):
        data = read(os.path.join(d, filename))
        return {i[0]: i[1] for i in data}

    def _get_split(self, data, id_to_filename, id_to_class_id, images_dir, masks_dir):
        items = []
        for image_id in data:
            filename = id_to_filename[image_id]
            class_id = int(id_to_class_id[image_id])
            base, _ = os.path.splitext(filename)
            image_path = os.path.join(images_dir, filename)
            masks_path = os.path.join(masks_dir, "%s.png" % base)
            items.append((image_path, masks_path, class_id))
        return items

    def _read_train_test_split(self, d, filename='train_test_split.txt'):
        data = read(os.path.join(d, filename))
        trainset = [d[0] for d in data if int(d[1]) == 1]
        testset = [d[0] for d in data if int(d[1]) == 0]
        return trainset, testset

    def parse_example(self, example):
        image_path, mask_path, class_id = example
        image = imageio.imread(image_path)

        mask = imageio.imread(mask_path)
        if len(mask.shape) > 2:
            raise Exception("mask %s has invalid shape %s" % (mask_path, mask.shape))

        mask = (mask / 255.).astype(np.uint8)

        if self.mode == 'category':
            mask = mask * class_id

        return image, mask

    def raw(self):
        trainset = self._get_split(self.traindata, self.image_id_to_filename, self.image_id_to_class_id, self.images_dir, self.masks_dir)
        trainset, valset = get_split_from_list(trainset, split=0.95)
        testset = self._get_split(self.testdata, self.image_id_to_filename, self.image_id_to_class_id, self.images_dir, self.masks_dir)
        return {
            DataType.TRAIN: trainset,
            DataType.TEST: testset,
            DataType.VAL: valset
        }


class CUB2002011Binary(CUB2002011):

    def __init__(self, cache_dir):
        super(CUB2002011Binary, self).__init__(cache_dir, mode='binary')


class CUB2002011Category(CUB2002011):

    def __init__(self, cache_dir):
        super(CUB2002011Category, self).__init__(cache_dir, mode='category')


if __name__ == "__main__":
    from .utils import test_dataset

    ds = CUB2002011Binary('/hdd/datasets/CUB2002011'.lower())
    test_dataset(ds)

    ds = CUB2002011Category('/hdd/datasets/CUB2002011'.lower())
    test_dataset(ds)
