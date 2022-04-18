from .dataset import DataType, Dataset
from ..utils import download_and_extract, extract_tar, get_files
from .utils import get_split_from_list
from ..settings import logger

import os
import json
import base64
import numpy as np
import cv2
from PIL import Image
import io
import zlib
import tqdm

class Duts(Dataset):
    """
    Uses last 10% of training set as validation set
    
    More Information here:
    http://saliencydetection.net/duts/
    """

    TRAIN_URL = "http://saliencydetection.net/duts/download/DUTS-TR.zip"
    TEST_URL = "http://saliencydetection.net/duts/download/DUTS-TE.zip"
    
    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        self.test_extracted = download_and_extract(self.TEST_URL, os.path.join(cache_dir, 'test'))
        self.train_extracted = download_and_extract(self.TRAIN_URL, os.path.join(cache_dir, 'train'))
    
    @property
    def labels(self):
        return ['bg', 'fg']
    
    def get_images_masks(self, image_dir: str, prefix: str):
        test_dir = os.path.join(image_dir, prefix)
        images_dir = os.path.join(test_dir, f"{prefix}-Image")
        masks_dir = os.path.join(test_dir, f"{prefix}-Mask")

        for image in get_files(images_dir, extensions=['jpg']):
            mask_path = os.path.join(masks_dir, os.path.basename(image).split(".")[0] + ".png")
            if os.path.exists(mask_path):
                yield image, mask_path
            else:
                logger.error(f"could not find mask {mask_path}")

    def raw(self):
        test = list(self.get_images_masks(self.test_extracted, 'DUTS-TE'))
        train = list(self.get_images_masks(self.train_extracted, 'DUTS-TR'))
        train, val = get_split_from_list(train, 0.9)
        
        return {
            DataType.TRAIN: train,
            DataType.VAL: val,
            DataType.TEST: test,
        }



if __name__ == '__main__':
    from .utils import test_dataset

    ds = Duts('/datasets/tf-semantic-segmentation/duts/')
    ds.summary()
    test_dataset(ds)