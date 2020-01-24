from .dataset import Dataset
from ..utils import get_files, download_and_extract, download_file
from .utils import get_split, DataType, Color

import imageio
import os
import numpy as np


class CamSeq01(Dataset):
    """
    Image Segmentation DataSet of Road Scenes

    Dataset url: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip
    """

    DATA_URL = "http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip"
    LABEL_COLORS_URL = "http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/label_colors.txt"

    def __init__(self, cache_dir):
        super(CamSeq01, self).__init__(cache_dir)
        self._labels = self.labels
        self._colormap = self.colormap

    def raw(self):
        dataset_dir = os.path.join(self.cache_dir, 'dataset')
        extracted = download_and_extract(self.DATA_URL, dataset_dir)
        imgs = get_files(extracted, extensions=["png"])
        images = list(filter(lambda x: not x.endswith("_L.png"), imgs))
        labels = list(filter(lambda x: x.endswith("_L.png"), imgs))
        trainset = list(zip(images, labels))
        return get_split(trainset)

    @property
    def colormap(self):
        file_path = download_file(self.LABEL_COLORS_URL, self.cache_dir)

        color_label_mapping = {}
        with open(file_path, "r") as handler:
            for line in handler.readlines():
                args = line.split("\t")
                color = list(map(lambda x: int(x), args[0].split(" ")))
                color = Color(*color)
                label = args[-1].replace("\n", "")
                color_label_mapping[color] = label

        return color_label_mapping

    @property
    def labels(self):
        file_path = download_file(self.LABEL_COLORS_URL, self.cache_dir)

        labels = []

        with open(file_path, "r") as handler:
            for line in handler.readlines():
                args = line.split("\t")
                label = args[-1].replace("\n", "")
                labels.append(label)

        return labels

    def parse_example(self, example):
        image_path, target_path = example
        i = imageio.imread(image_path)
        t = imageio.imread(target_path)
        mask = np.zeros((i.shape[0], i.shape[1]), np.uint8)

        for color, label in self._colormap.items():
            color = [color.r, color.g, color.b]
            idxs = np.where(np.all(t == color, axis=-1))
            mask[idxs] = self._labels.index(label)

        return i, mask


if __name__ == "__main__":
    from .utils import test_dataset

    ds = CamSeq01('/hdd/datasets/camvid')
    test_dataset(ds)
