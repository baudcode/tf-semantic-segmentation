from .dataset import DataType, Dataset
from ..utils import download_and_extract, extract_tar
from .utils import get_train_test_val_from_list
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


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


class SuperviselyPerson(Dataset):

    DATA_URL = "https://cloud.enterprise.deepsystems.io/s/TK2z5TLYoAPl1w6/download"
    SAMPLE_DATA_URL = "https://cloud.enterprise.deepsystems.io/s/7VyML7ynZ9L1KEK/download"

    def __init__(self, cache_dir, sample: bool = False):
        super(SuperviselyPerson, self).__init__(cache_dir)
        self.url = self.SAMPLE_DATA_URL if sample else self.DATA_URL
        logger.info(f"using url {self.url}")
        self.extracted = download_and_extract(self.url, self.cache_dir, file_name='super_sample.tar' if sample else "super_full.tar")
        self.generated = self.generate_masks()

    @property
    def labels(self):
        return ['bg', 'person']

    def create_folder_masks(self, folder: str):
        img_folder = os.path.join(folder, 'img')
        ann_folder = os.path.join(folder, 'ann')

        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder, img_name)
            ann_path = os.path.join(ann_folder, img_name + ".json")

            # create a mask
            mask_path = os.path.join(ann_folder, img_name + ".mask.png")
            if os.path.exists(mask_path):
                yield img_path, mask_path
                continue

            if not os.path.exists(ann_path):
                logger.error(f"error: could not find annotation {ann_path}")
                continue

            json_data = json.load(open(ann_path, 'r'))
            img_size = (json_data['size']['width'], json_data['size']['height'])

            mask = np.zeros(img_size[::-1], np.uint8)

            for obj in json_data['objects']:
                geo_type = obj['geometryType']
                if geo_type == 'bitmap':
                    bitmap = obj['bitmap']['data']
                    bitmap_mask = base64_2_mask(bitmap)
                    origin = obj['bitmap']['origin']
                    x, y = origin
                    mask[y:(y + bitmap_mask.shape[0]), x:(x + bitmap_mask.shape[1])] = bitmap_mask

                elif geo_type == "polygon":
                    points = obj['points']['exterior']
                    points = np.array(points, np.int32).reshape((-1, 2))
                    cv2.drawContours(mask, [points], 0, 1, cv2.FILLED)
                else:
                    logger.error(f"skipping geo type {geo_type}")
                    continue

            Image.fromarray(mask, "L").save(mask_path)

            yield img_path, mask_path

    def generate_masks(self):
        logger.info("generating masks")
        m = {
            DataType.TRAIN: "train",
            DataType.TEST: "test",
            DataType.VAL: "validate",
        }

        main_dir = os.path.join(self.extracted, os.listdir(self.extracted)[0])
        logger.debug(f"main dir: {main_dir}")
        folders = [os.path.join(main_dir, o) for o in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, o))]
        logger.debug(f"folders: {folders}")

        entries = []
        for folder in tqdm.tqdm(folders, 'creating annotations for folder', unit='folder'):
            for img_path, mask_path in self.create_folder_masks(folder):
                entries.append((img_path, mask_path))

        train, test, val = get_train_test_val_from_list(entries, train_split=0.8, val_split=0.5, shuffle=True, rand=lambda: 0.3)

        return {
            DataType.TRAIN: train,
            DataType.VAL: val,
            DataType.TEST: test
        }

    def raw(self):
        return self.generated


class SuperviselyPersonSample(SuperviselyPerson):
    def __init__(self, cache_dir):
        super().__init__(cache_dir, True)


if __name__ == "__main__":
    from .utils import test_dataset

    ds = SuperviselyPerson('/datasets/tf-semantic-segmentation/supervisely_person/')
    ds.summary()
    test_dataset(ds)
