from .dataset import DataType, Dataset
from ..utils import extract_zip, get_files
from ..settings import logger

import os
import imageio
import numpy as np


class ISIC2018(Dataset):

    DATASET_URL = "https://challenge.kitware.com/#challenge/5aab46f156357d5e82b00fe5"

    def __init__(self, cache_dir):
        super(ISIC2018, self).__init__(cache_dir)
        self.extract()

    @property
    def labels(self):
        return ['bg', 'melona']

    def extract(self):
        filenames = [
            "ISIC2018_Task1-2_Training_Input.zip",
            "ISIC2018_Task1_Training_GroundTruth.zip",
            "ISIC2018_Task1-2_Validation_Input.zip",
            "ISIC2018_Task1-2_Test_Input.zip"
        ]
        files = [os.path.join(self.cache_dir, f) for f in filenames]

        if not all(map(os.path.exists, files)):
            raise FileNotFoundError("Please download the following files %s to %s from %s" % (str(filenames), self.cache_dir, self.DATASET_URL))

        for f in files:
            logger.info("extracting %s, this may take a while" % f)
            output_dirname = os.path.splitext(os.path.basename(f))[0]
            destination = os.path.join(self.cache_dir, output_dirname)
            if not os.path.exists(destination):
                extract_zip(f, destination)
            else:
                logger.info("skipping extraction, because directory already exist")

    def raw(self):

        train_images = get_files(os.path.join(self.cache_dir, 'ISIC2018_Task1-2_Training_Input'), extensions=['jpg'])
        train_masks = get_files(os.path.join(self.cache_dir, 'ISIC2018_Task1_Training_GroundTruth'), extensions=['png'])
        val_images = get_files(os.path.join(self.cache_dir, 'ISIC2018_Task1-2_Validation_Input'), extensions=['jpg'])
        test_images = get_files(os.path.join(self.cache_dir, 'ISIC2018_Task1-2_Test_Input'), extensions=['jpg'])

        return {
            DataType.TRAIN: list(zip(train_images, train_masks)),
            DataType.VAL: list(zip(val_images, [None] * len(val_images))),
            DataType.TEST: list(zip(test_images, [None] * len(test_images))),
        }

    def parse_example(self, example):
        image_fn, mask_fn = example
        image = imageio.imread(image_fn)

        if mask_fn is None:
            logger.warning("mask of dataset is None, returning ZEROS")
            mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        else:
            mask = (imageio.imread(mask_fn) / 255.).astype(np.uint8)

        return image, mask


if __name__ == "__main__":
    from .utils import test_dataset

    ds = ISIC2018('/hdd/datasets/isic/')
    test_dataset(ds)
