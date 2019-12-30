import random
import imageio
import collections
from ..utils import get_files

Color = collections.namedtuple('Color', ['r', 'g', 'b'])


class DataType:
    TRAIN, TEST, VAL = 'train', 'test', 'val'

    @staticmethod
    def get():
        return list(map(lambda x: DataType.__dict__[x], list(filter(lambda k: not k.startswith("__") and type(DataType.__dict__[k]) == str, DataType.__dict__))))


def get_train_test_val_from_list(l, train_split=0.8, val_split=0.5, shuffle=True, rand=lambda: 0.2):
    if shuffle:
        random.shuffle(l, random=rand)

    trainset = l[:int(round(train_split * len(l)))]
    valtestset = l[int(round(train_split * len(l))):]
    testset = valtestset[int(round(val_split * len(valtestset))):]
    valset = valtestset[:int(round(val_split * len(valtestset)))]
    return trainset, testset, valset


def get_split(l, train_split=0.8, val_split=0.5, shuffle=True, rand=lambda: 0.2):
    trainset, testset, valset = get_train_test_val_from_list(
        l, train_split=train_split, val_split=val_split, shuffle=shuffle, rand=rand)
    return {
        DataType.TRAIN: trainset,
        DataType.VAL: valset,
        DataType.TEST: testset
    }


def get_split_from_list(l, split=0.9):
    trainset = l[:int(round(split * len(l)))]
    valset = l[int(round(split * len(l))):]
    return trainset, valset


def get_split_from_dirs(images_dir, labels_dir, extensions=['png'], train_split=0.8, val_split=0.5, shuffle=True, rand=lambda: 0.2):
    images = get_files(images_dir, extensions=extensions)
    labels = get_files(labels_dir, extensions=extensions)

    trainset = list(zip(images, labels))
    return get_split(trainset, train_split=train_split, val_split=val_split, shuffle=shuffle, rand=rand)


def image_generator(data, color_map=None):
    import numpy as np

    def gen():
        for image_path, label_path in data:
            labels = imageio.imread(label_path)[:, :, :3]
            labels_idx = np.array(labels.shape, np.uint8)
            for color, value in color_map.items():
                labels_idx[labels == np.asarray(
                    [color.r, color.g, color.b])] = [value, value, value]
            labels_idx = labels_idx.mean(axis=-1)
            # print(labels_idx.max(), labels_idx.min())
            yield imageio.imread(image_path), labels_idx
    return gen
