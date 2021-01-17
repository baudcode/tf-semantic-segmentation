from .dataset import Dataset, DataType
from .utils import get_split_from_list
from ..utils import download_and_extract, get_files
import os
import numpy as np
import imageio


class BioimageBenchmark(Dataset):
    """
    Kaggle 2018 Data Science Bowl
    https://data.broadinstitute.org/bbbc/BBBC038/

    Broad Bioimage Benchmark Collection

    This image data set contains a large number of segmented nuclei images and was created for the Kaggle 2018 Data Science Bowl sponsored by 
    Booz Allen Hamilton with cash prizes. The image set was a testing ground for the application of novel and cutting edge approaches 
    in computer vision and machine learning to the segmentation of the nuclei belonging to cells from a breadth of biological contexts.
    """

    TRAIN_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
    TEST_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip"
    TEST_URL_STAGE_2 = "https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip"

    def __init__(self, cache_dir, use_stage2_testing=False):
        super(BioimageBenchmark, self).__init__(cache_dir)
        self.use_stage2_testing = use_stage2_testing

    def raw(self):
        train_dir = download_and_extract(self.TRAIN_URL, os.path.join(self.cache_dir, 'train'))
        test_url = self.TEST_URL_STAGE_2 if self.use_stage2_testing else self.TEST_URL
        test_dir = download_and_extract(test_url, os.path.join(self.cache_dir, 'test'))
        trainset = []

        # make train dataset
        for dir in os.listdir(train_dir):
            abs_dir = os.path.join(train_dir, dir)
            images = get_files(os.path.join(abs_dir, 'images'), extensions=['png'])
            masks = get_files(os.path.join(abs_dir, 'masks'), extensions=['png'])
            if len(images) == 1:
                trainset.append((images[0], masks))

        # use part of train dataset as validation set
        trainset, valset = get_split_from_list(trainset, split=0.9)

        # make test dataset
        testset = []
        for dir in os.listdir(train_dir):
            abs_dir = os.path.join(train_dir, dir)
            images = get_files(os.path.join(abs_dir, 'images'), extensions=['png'])
            if len(images) == 1:
                testset.append((images[0], None))
        return {
            DataType.TRAIN: trainset,
            DataType.VAL: valset,
            DataType.TEST: testset
        }

    @property
    def labels(self):
        return ['bg', 'nucleus']

    def parse_example(self, example):
        image_path, masks = example

        image = imageio.imread(image_path)[:, :, :3]
        mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        # testset does not contain any masks, return zeros for masks
        if masks is None:
            return image, mask

        for mask_path in masks:
            nucleus = imageio.imread(mask_path)
            mask |= nucleus

        mask = np.divide(mask, 255)
        mask = mask.astype(np.uint8)
        return image, mask


if __name__ == "__main__":
    from .utils import test_dataset

    ds = BioimageBenchmark('/hdd/datasets/BioimageBenchmark'.lower())
    test_dataset(ds)
    ds.summary()
