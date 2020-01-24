from .dataset import Dataset, DataType
from ..utils import download_and_extract, get_files, ExtractException
from .utils import get_split

import os
import shutil
import imageio
import numpy as np


class CVCClinicDB(Dataset):
    """
    This dataset contains segmentations for polyps.

    https://polyp.grand-challenge.org/CVCClinicDB/
    """
    DATA_URL = "https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=1"

    @property
    def labels(self):
        return ['bg', 'polyps']

    def raw(self):
        output_path = os.path.join(self.cache_dir, 'data')
        try:
            extracted = download_and_extract(self.DATA_URL, output_path, file_name='CVC-ClinicDB.rar')
        except ExtractException as e:
            print(str(e))
            shutil.rmtree(output_path)

        images = get_files(os.path.join(output_path, 'CVC-ClinicDB/Original'), extensions=['tif'])
        masks = get_files(os.path.join(output_path, 'CVC-ClinicDB/Ground Truth/'), extensions=['tif'])
        dataset = list(zip(images, masks))
        return get_split(dataset)

    def parse_example(self, example):
        image_fn, mask_fn = example
        image = imageio.imread(image_fn)
        mask = (imageio.imread(mask_fn) / 255.).astype(np.uint8)

        return image, mask


if __name__ == "__main__":
    from .utils import test_dataset
    ds = CVCClinicDB('/hdd/datasets/cvc_clinic_db')
    test_dataset(ds)
