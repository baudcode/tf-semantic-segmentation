from .dataset import Dataset
from ..utils import get_files, download_and_extract, download_file
from .utils import get_split, DataType, Color

import imageio
import os


class CamSeq01(Dataset):
    """
    Image Segmentation DataSet of Road Scenes

    Dataset url: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip
    """

    DATA_URL = "http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip"
    LABEL_COLORS_URL = "http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/label_colors.txt"

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

    def get(self, data_type=DataType.TRAIN):
        import imageio
        import numpy as np

        data = self.raw()[data_type]
        colormap = self.colormap
        labels = self.labels

        print(colormap)

        def gen():
            for image_path, target_path in data:
                i = imageio.imread(image_path)
                t = imageio.imread(target_path)
                mask = np.zeros((i.shape[0], i.shape[1]), np.uint8)

                for color, label in colormap.items():
                    color = [color.r, color.g, color.b]
                    idxs = np.where(np.all(t == color, axis=-1))
                    mask[idxs] = labels.index(label)

                print(i.shape, mask.shape, mask.max(), mask.min())
                yield i, mask
        return gen


if __name__ == "__main__":
    ds = CamSeq01('/hdd/datasets/camvid')
    print(ds.colormap)
    print(ds.labels)
    print(ds.num_examples(DataType.TRAIN))
    print(ds.num_examples(DataType.VAL))
    print(ds.num_examples(DataType.TEST))
