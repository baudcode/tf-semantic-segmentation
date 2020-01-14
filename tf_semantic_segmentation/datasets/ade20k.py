import os
import imageio
import csv
import numpy as np
import csv

from .utils import get_split_from_list
from ..utils import get_files, download_file, download_and_extract
from .dataset import Dataset, DataType
# TODO: check it, why is it not working?
# https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.txt
# https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
# download: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/download_ADE20K.sh


class Ade20k(Dataset):

    DATA_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    LABELS_URL = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv"

    @property
    def labels(self):
        download_path = download_file(self.LABELS_URL, self.cache_dir, file_name="objectInfo150.csv")
        with open(download_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            labels = [dict(row)['Name'] for row in reader]
        return ['bg'] + labels  # add background

    def raw(self):
        extract_dir = download_and_extract(self.DATA_URL, self.cache_dir)
        extract_dir = os.path.join(extract_dir, "ADEChallengeData2016")
        print(extract_dir)

        val_images_dir = os.path.join(extract_dir, 'images', 'validation')
        train_images_dir = os.path.join(extract_dir, 'images', 'training')

        val_annotations_dir = os.path.join(extract_dir, 'annotations', 'validation')
        train_annotations_dir = os.path.join(extract_dir, 'annotations', 'training')

        val_images = get_files(val_images_dir, extensions=['jpg'])
        val_annotations = get_files(val_annotations_dir, extensions=['png'])

        train_images = get_files(train_images_dir, extensions=['jpg'])
        train_annotations = get_files(train_annotations_dir, extensions=['png'])

        return {
            DataType.TRAIN: list(zip(train_images, train_annotations)),
            DataType.VAL: list(zip(val_images, val_annotations)),
            DataType.TEST: []
        }


if __name__ == "__main__":
    ade20k = Ade20k('/hdd/datasets/ade20k')
    for image, target in ade20k.get()():
        print(image.shape, target.shape, target.max())
        pass
