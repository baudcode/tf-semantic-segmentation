from .. import utils
from ..settings import logger

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
    images = utils.get_files(images_dir, extensions=extensions)
    masks = utils.get_files(masks_dir, extensions=extensions)

    trainset = list(zip(images, masks))
    return get_split(trainset, train_split=train_split, val_split=val_split, shuffle=shuffle, rand=rand)


def convert2tfdataset(dataset, data_type, randomize=True):
    """
    Converts a tf_semantic_segmentation.datasets.dataset.Dataset class to tf.data.Dataset which returns a tuple
    (image, mask, num_classes)

    Arguments:
        dataset (tf_semantic_segmentation.datasets.dataset.Dataset): dataset to convert
        data_type (tf_semantic_segmentation.datasets.dataset.DataType): one of (train, val, test)
        randomize (bool): randomize the order of the paths, otherwise every epoch it yields the same
    Returns:
        tf.data.Dataset
    """

    def gen():
        indexes = np.arange(dataset.num_examples(data_type))
        if randomize:
            indexes = np.random.permutation(indexes)

        data = dataset.raw()[data_type]
        for idx in indexes:
            example = data[idx]
            try:
                image, mask = dataset.parse_example(example)
            except Exception as e:
                logger.warning("could not parse example %s, error: %s, skipping" % (str(example), str(e)))
                continue

            shape = image.shape

            if len(shape) == 2:
                shape = [shape[0], shape[1], 1]

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            if len(mask.shape) > 2:
                logger.warning("mask of example %s has invalid shape: %s, skipping" % (str(example), str(mask.shape)))
                continue

            yield image, mask, dataset.num_classes, shape

    def map_fn(image, mask, num_classes, shape):
        image = tf.reshape(image, shape)
        mask = tf.reshape(mask, (shape[0], shape[1]))
        return image, mask, num_classes

    if hasattr(dataset, 'tfdataset'):
        logger.info("convert2tfdataset is using tfdataset optimized wrapper function")
        ds = dataset.tfdataset(data_type, randomize=randomize)
    else:
        ds = tf.data.Dataset.from_generator(
            gen, (tf.uint8, tf.uint8, tf.int64, tf.int64), ([None, None, None], [None, None], [], [3]))
        ds = ds.map(map_fn, num_parallel_calls=multiprocessing.cpu_count())

    #ds[0] = tf.reshape(ds[0], ds[2])
    #ds[1] = tf.reshape(ds[1], [ds[2][0], ds[2][1], tf.convert_to_tensor(dataset.num_classes, tf.int64)])
    return ds


def load_image(image_path, dtype, squeeze=True, channels=3):
    image_string = tf.io.read_file(image_path)
    image = tf.cond(
      tf.image.is_jpeg(image_string),
      lambda: tf.image.decode_jpeg(image_string, channels=channels),
      lambda: tf.image.decode_png(image_string, channels=channels))

    image = tf.image.convert_image_dtype(image, dtype)
    
    if squeeze and channels == 1:
        image = tf.squeeze(image, axis=-1)
    
    return image


def download_records(tag, destination_dir):
    if tag in google_drive_records_by_tag:
        drive_url = google_drive_records_by_tag[tag]
        drive_id = drive_url.split("/")[-2]
        print("download and extract ", drive_id, tag, destination_dir)
        utils.download_and_extract(('%s.zip' % tag, drive_id), destination_dir, chk_exists=False)
    else:
        raise Exception("cannot download records of tag %s, please use one of %s" % (tag, str(google_drive_records_by_tag.keys())))


def test_dataset(ds):
    logger.debug("testing dataset %s" % ds.__class__.__name__)
    for data_type in DataType.get():
        logger.debug("using data_type %s" % data_type)
        tfds = convert2tfdataset(ds, data_type)

        for image, mask, num_classes in tfds:
            logger.debug("image shape: %s, %s |  mask shape: %s, %s, %d (max) | num_classes: %d" % (image.shape, image.dtype, mask.shape, mask.dtype, mask.numpy().max(), num_classes))
            break
    return image.numpy(), mask.numpy(), num_classes.numpy()
