from ..utils import get_files, download_from_google_drive, download_and_extract

import random
import imageio
import collections
import tensorflow as tf
import numpy as np
import multiprocessing

Color = collections.namedtuple('Color', ['r', 'g', 'b'])

# save google drive records here for now
google_drive_records_by_tag = {
    'camvid': "https://drive.google.com/file/d/12ma5YOdoCc0K-cY3meYMEU9sxgT25eeT/view?usp=sharing",
    'cityscapes-512x256': "https://drive.google.com/file/d/1VLzs6ttsFl7XRO6MF7b84F8-sqFeSDEr/view?usp=sharing",
    'shapes-10k-256x256-resize': 'https://drive.google.com/file/d/1LI06-7UmauMleY5LI2Pdo6_tOwBiVMMa/view?usp=sharing'
}


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


def get_split_from_dirs(images_dir, masks_dir, extensions=['png'], train_split=0.8, val_split=0.5, shuffle=True, rand=lambda: 0.2):
    images = get_files(images_dir, extensions=extensions)
    masks = get_files(masks_dir, extensions=extensions)

    trainset = list(zip(images, masks))
    return get_split(trainset, train_split=train_split, val_split=val_split, shuffle=shuffle, rand=rand)


def image_generator(data, color_map=None):
    import numpy as np

    def gen():
        for image_path, mask_path in data:
            mask = imageio.imread(mask_path)[:, :, :3]
            mask_idx = np.array(mask.shape, np.uint8)
            for color, value in color_map.items():
                mask_idx[mask == np.asarray(
                    [color.r, color.g, color.b])] = [value, value, value]

            mask_idx = mask_idx.mean(axis=-1)
            yield imageio.imread(image_path), mask_idx
    return gen


def convert2tfdataset(dataset, data_type, randomize=True):

    def gen():
        indexes = np.arange(dataset.num_examples(data_type))
        if randomize:
            indexes = np.random.permutation(indexes)

        data = dataset.raw()[data_type]
        for idx in indexes:
            example = data[idx]
            image, mask = dataset.parse_example(example)

            shape = image.shape

            if len(shape) == 2:
                shape = [shape[0], shape[1], 1]

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            yield image, mask, dataset.num_classes, shape

    def map_fn(image, mask, num_classes, shape):
        image = tf.reshape(image, shape)
        mask = tf.reshape(mask, (shape[0], shape[1]))
        return image, mask, num_classes

    ds = tf.data.Dataset.from_generator(
        gen, (tf.uint8, tf.uint8, tf.int64, tf.int64), ([None, None, None], [None, None], [], [3]))
    ds = ds.map(map_fn, num_parallel_calls=multiprocessing.cpu_count())

    #ds[0] = tf.reshape(ds[0], ds[2])
    #ds[1] = tf.reshape(ds[1], [ds[2][0], ds[2][1], tf.convert_to_tensor(dataset.num_classes, tf.int64)])
    return ds


def download_records(tag, destination_dir):
    if tag in google_drive_records_by_tag:
        drive_url = google_drive_records_by_tag[tag]
        drive_id = drive_url.split("/")[-2]
        print("download and extract ", drive_id, tag, destination_dir)
        download_and_extract(('%s.zip' % tag, drive_id), destination_dir, chk_exists=False)
    else:
        raise Exception("cannot download records of tag %s, please use one of %s" % (tag, str(google_drive_records_by_tag.keys())))
