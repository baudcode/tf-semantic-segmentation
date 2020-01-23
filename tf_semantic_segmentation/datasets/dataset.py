from ..settings import logger

import imageio


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

    def get(self, data_type=DataType.TRAIN):
        data = self.raw()[data_type]

        def gen():
            for image_path, mask_path in data:
                try:
                    yield imageio.imread(image_path), imageio.imread(mask_path)
                except:
                    logger.error("could not read either %s or %s" % (image_path, mask_path))
        return gen
