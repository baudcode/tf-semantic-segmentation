from ... import utils
from ..utils import DataType, get_split
from ..dataset import Dataset

from functools import reduce
from os.path import join
import numpy as np
import sys
import imageio


class CMP(Dataset):

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.base_dataset_url = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip"
        self.extended_dataset_url = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip"

    def raw(self):
        extended = utils.download_and_extract(self.extended_dataset_url, join(self.cache_dir, "extended"))
        base = utils.download_and_extract(self.base_dataset_url, join(self.cache_dir, "base"))

        train_in = utils.get_files(extended, extensions=['jpg']) + utils.get_files(base, extensions=['jpg'])
        train_out = utils.get_files(extended, extensions=['png']) + utils.get_files(base, extensions=['png'])
        xml = utils.get_files(extended, extensions=["xml"]) + utils.get_files(base, extensions=["xml"])

        labels = map(lambda x: {"xml": x[0], "image": x[1]}, zip(xml, train_out))
        data = list(zip(train_in, labels))
        return get_split(data)

    @property
    def labels(self):
        return ["background", "facade", "molding", "cornice", "pillar", "window", "door", "sill", "blind", "balcony", "shop", "deco"]

    @property
    def colormap2index(self):
        colors = [1, 2, 7, 11, 12, 10, 5, 3, 6, 4, 9, 8]
        color_by_label = {}

        for k, label in enumerate(self.labels):
            color_by_label[label] = int(colors[k] * (255. / float(max(colors))))

        return color_by_label

    def get(self, data_type=DataType.TRAIN):
        """ Returns a generator object """
        split = self.raw()

        def generator():
            for input_path, targets in split[data_type]:
                inputs = imageio.imread(input_path)
                labels = imageio.imread(targets['image'])[:, :, 0]
                labels = labels.astype(np.float32) / (255. / 12.)
                labels = labels.astype(np.uint8)

                yield inputs, labels

        return generator


if __name__ == "__main__":
    cmp = CMP('/hdd/datasets/CMP/')
    gen = cmp.get()
    inputs, targets = next(gen())
    print(inputs.shape, targets.shape)
