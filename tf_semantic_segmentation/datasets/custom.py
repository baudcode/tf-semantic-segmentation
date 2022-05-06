from .dataset import DataType, Dataset
from ..settings import logger

import os
import json


class SemSeg(Dataset):

    supports_v2 = True

    def __init__(self, cache_dir):
        super(SemSeg, self).__init__(cache_dir)
        self._parsed = self._parse_dataset_json(os.path.join(cache_dir, 'test', 'dataset.json'))

    @property
    def labels(self):
        categories = self._parsed['categories']
        categories = sorted(categories, key=lambda x: x['id'])
        return list(map(lambda x: x['name'], categories))

    def _parse_dataset_json(self, path: str):
        data = json.load(open(path, 'r'))
        # categories: (id: int, name: str)
        # images: (width: int, height: int, id: int, file_name: str, sem_seg_file_name: str)
        return {"images": data['images'], 'categories': data['categories']}

    def raw(self):
        m = {
            DataType.TRAIN: "train",
            DataType.TEST: "test",
            DataType.VAL: "validate",
        }

        def get_examples(name: str):
            d = os.path.join(self.cache_dir, name)
            json_path = os.path.join(d, 'dataset.json')
            logger.debug("SemSeg -> trying to read json file at %s" % json_path)

            parsed = self._parse_dataset_json(json_path)
            images = [os.path.join(d, item['file_name']) for item in parsed['images']]
            masks = [os.path.join(d, 'sem_seg', item['sem_seg_file_name']) for item in parsed['images']]
            logger.debug("SemSeg -> found %d images for name %s" % (len(images), name))

            return list(zip(images, masks))

        return {
            DataType.TRAIN: get_examples('train'),
            DataType.VAL: get_examples('validate'),
            DataType.TEST: get_examples('test')
        }


if __name__ == "__main__":
    from .utils import test_dataset

    ds = SemSeg('/home/baudcode/Notebooks/tf-semantic-segmentation/ceiling_sem.tgz/ceiling_sem')
    test_dataset(ds)
