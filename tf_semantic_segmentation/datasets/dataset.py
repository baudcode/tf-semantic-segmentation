from ..settings import logger
from ..visualizations import show, masks

import imageio
import random
import numpy as np


class DataType:
    TRAIN, TEST, VAL = 'train', 'test', 'val'

    @staticmethod
    def get():
        return list(map(lambda x: DataType.__dict__[x], list(filter(lambda k: not k.startswith("__") and type(DataType.__dict__[k]) == str, DataType.__dict__))))


class Dataset(object):

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def summary(self):
        logger.info("======================%s========================" % self.__class__.__name__)
        logger.info("dataset has %d classes" % self.num_classes)
        logger.info("labels: %s" % self.labels)
        examples = [(data_type, self.num_examples(data_type)) for data_type in [DataType.TRAIN, DataType.VAL, DataType.TEST]]
        logger.info("examples: %s" % str(examples))
        logger.info("total: %d" % sum([l for _, l in examples]))
        logger.info("================================================")

    @property
    def labels(self):
        return []

    @property
    def num_classes(self):
        return len(self.labels)

    def raw(self):
        return {
            DataType.TRAIN: [],
            DataType.VAL: [],
            DataType.TEST: []
        }

    def parse_example(self, example):
        return list(map(imageio.imread, example))

    def num_examples(self, data_type):
        return len(self.raw()[data_type])

    def total_examples(self):
        return sum([self.num_examples(dt) for dt in DataType.get()])

    def get_random_item(self, data_type=DataType.TRAIN):
        data = self.raw()[data_type]
        example = random.choice(data)
        return self.parse_example(example)

    def show_random_item(self, data_type=DataType.TRAIN):
        example = self.get_random_item(data_type=data_type)
        show.show_images([example[0], example[1].astype(np.float32)])

    def save_random_item(self, data_type=DataType.TRAIN, image_path='image.png', mask_path='mask.png', mask_mode='rgb', alpha=0.5):
        assert(mask_mode in ['gray', 'rgb', 'overlay']), 'mask mode must be in %s' % str(['gray', 'rgb', 'overlay'])
        item = self.get_random_item(data_type=data_type)
        imageio.imwrite(image_path, item[0])
        
        mask = item[1]
        if mask_mode == 'rgb':
            image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colors = masks.get_colors(self.num_classes)
            mask = masks.overlay_classes(image, mask, colors, self.num_classes, alpha=1.0)

        elif mask_mode == 'overlay':
            colors = masks.get_colors(self.num_classes)
            mask = masks.overlay_classes(item[0], mask, colors, self.num_classes, alpha=alpha)

        elif mask_mode == 'gray': # scale mask to 0 to 255
            mask = (mask * (255. / (self.num_classes - 1))).astype(np.uint8)

        imageio.imwrite(mask_path, mask)

    def get(self, data_type=DataType.TRAIN):
        data = self.raw()[data_type]

        def gen():
            for example in data:
                try:
                    yield self.parse_example(example)
                except:
                    logger.error("could not read either one of these files %s" % str(example))
        return gen
