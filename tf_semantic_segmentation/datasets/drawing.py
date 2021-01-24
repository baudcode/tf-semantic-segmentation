#
from .dataset import Dataset, DataType
from ..utils import download_and_extract, get_files
from .utils import get_split_from_list
import os
import cv2
import imageio
import numpy as np


class APDrawing(Dataset):

    url = "https://cg.cs.tsinghua.edu.cn/people/~Yongjin/APDrawingDB.zip"

    def __init__(self, cache_dir: str, train_val_split: float = 0.9):
        super(APDrawing, self).__init__(cache_dir)
        self.train_val_split = train_val_split

    def raw(self):
        extract_dir = download_and_extract(self.url, self.cache_dir)
        trainset = get_files(os.path.join(extract_dir, 'APDrawingDB/data/train/'), extensions=['png'])
        testset = get_files(os.path.join(extract_dir, 'APDrawingDB/data/test/'), extensions=['png'])
        trainset, valset = get_split_from_list(trainset, split=self.train_val_split)

        return {
            DataType.TRAIN: trainset,
            DataType.TEST: testset,
            DataType.VAL: valset,
        }

    def parse_example(self, path):
        img = imageio.imread(path)

        inputs = img[:, :512, :]
        mask = img[:, 512:, :]
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) / 255
        mask = mask.astype(np.uint8)
        return inputs, mask


if __name__ == "__main__":
    ds = APDrawing('/hdd/datasets/APDrawing/')
    # image, mask = ds.get_random_item(data_type=DataType.TRAIN)
    # print(image.shape, mask.max(), mask.dtype)
    ds.show_random_item()
