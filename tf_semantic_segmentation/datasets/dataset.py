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
        print("======================%s========================" % self.__class__.__name__)
        print("dataset has %d classes" % self.num_classes)
        print("labels: %s" % self.labels)
        examples = [(data_type, self.num_examples(data_type)) for data_type in [DataType.TRAIN, DataType.VAL, DataType.TEST]]
        print("examples: ", examples)
        print("total: ", sum([l for _, l in examples]))
        print("================================================")

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
                yield imageio.imread(image_path), imageio.imread(mask_path)

        return gen
