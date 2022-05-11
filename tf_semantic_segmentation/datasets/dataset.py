from typing import Dict, Iterable, List, Tuple
from warnings import warn

from tf_semantic_segmentation.datasets.utils import load_image
from tf_semantic_segmentation.processing import ColorMode
from ..settings import logger
from ..visualizations import show, masks

try:
    import imageio.v2 as imageio
except:
    import imageio

import random
import math
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import cv2
import tqdm


class DataType:
    TRAIN, TEST, VAL = 'train', 'test', 'val'

    @staticmethod
    def get():
        return list(map(lambda x: DataType.__dict__[x], list(filter(lambda k: not k.startswith("__") and type(DataType.__dict__[k]) == str, DataType.__dict__))))


class Dataset(object):

    supports_v2 = False

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

    def raw(self) -> Dict[str, List[Tuple[str, str]]]:
        return {
            DataType.TRAIN: [],
            DataType.VAL: [],
            DataType.TEST: []
        }

    def parse_example(self, example):
        return list(map(imageio.imread, example))

    def parse_example_dict(self, example):
        data = {
            "image": imageio.imread(example[0]),
            "mask": imageio.imread(example[1])
        }
        if len(example) > 2:
            data['difficulty'] = example[2]

        return data

    def num_examples(self, data_type):
        return len(self.raw()[data_type])

    def total_examples(self):
        return sum([self.num_examples(dt) for dt in DataType.get()])

    def iter(self, data_type=DataType.TRAIN, use_tq: bool = True):
        data = self.raw()[data_type]

        if use_tq:
            for d in tqdm.tqdm(data):
                yield self.parse_example(d)
        else:
            for d in data:
                yield self.parse_example(d)

    def has_one_shape(self) -> bool:
        it = self.iter()
        e = next(it)
        shape = e[0].shape
        logger.info(f"checking all images of shape {shape} (first image)")
        for e in it:
            if e[0].shape != shape:
                return False

        return True

    @property
    def input_shape(self) -> List[int]:
        """ return the input shape of the first image """
        it = self.iter()
        e = next(it)
        return e[0].shape

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

        elif mask_mode == 'gray':  # scale mask to 0 to 255
            mask = (mask * (255. / (self.num_classes - 1))).astype(np.uint8)

        imageio.imwrite(mask_path, mask)

    def tfdataset_v2(self, data_type: str, color_mode: ColorMode = ColorMode.RGB, shuffle=False, reshuffle_each_iteration=True, buffer_size=-1) -> tf.data.Dataset:
        """ 
        Dataset().raw() as to return a dict for data_type to X
        X is a list of tuple [image_path, mask_path, optional: difficulty]

        Loads images and masks from image and mask paths.
        """
        channels = 3 if color_mode == ColorMode.RGB or color_mode == ColorMode.NONE else 1

        if hasattr(self, "supports_v2") and self.supports_v2:
            logger.info("using v2 load_image approach")

            data = self.raw()[data_type]

            image_paths = [d[0] for d in data]
            mask_paths = [d[1] for d in data]

            if len(data[0]) == 3:
                difficulty = [d[2] for d in data]
            else:
                difficulty = None

            assert(len(image_paths) == len(mask_paths)), "len of images does not equal len of masks"

            images_ds = tf.data.Dataset.from_tensor_slices(image_paths)
            masks_ds = tf.data.Dataset.from_tensor_slices(mask_paths)

            @tf.function
            def load(image_path: str, mask_path: str, difficulty: int = None):
                image = load_image(image_path, tf.uint8, squeeze=False, channels=channels)
                mask = load_image(mask_path, tf.uint8, squeeze=True, channels=1)

                if difficulty != None:
                    return image, mask, difficulty
                else:
                    return image, mask

            if difficulty is None:
                dataset = tf.data.Dataset.zip((images_ds, masks_ds))
            else:
                difficulty_ds = tf.data.Dataset.from_tensor_slices(difficulty)
                dataset = tf.data.Dataset.zip((images_ds, masks_ds, difficulty_ds))

            if shuffle:
                buffer_size = buffer_size if buffer_size != -1 else len(data)
                dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=reshuffle_each_iteration)

            # load images
            dataset = dataset.map(load, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            # use generator
            logger.info("using v1 generator approach")

            self.indexes = np.arange(self.num_examples(data_type))

            if shuffle:
                self.indexes = np.random.permutation(self.indexes)

            def gen():
                if shuffle and reshuffle_each_iteration:
                    self.indexes = np.random.permutation(self.indexes)

                data = self.raw()[data_type]

                for idx in self.indexes:
                    example = data[idx]

                    try:
                        e = self.parse_example(example)

                        image = e[0]

                        if len(image.shape) == 2:
                            image = np.expand_dims(image, axis=-1)

                        if image.shape[-1] == 1 and channels == 3:
                            image = cv2.cvtColor(np.squeeze(image, axis=-1), cv2.COLOR_GRAY2RGB)

                        elif image.shape[-1] == 3 and channels == 1:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                            image = np.expand_dims(image, axis=-1)

                        # remove alpha
                        image = image[:, :, :3]
                        mask = e[1]

                        if len(mask.shape) == 3 and mask.shape[-1] != 1:
                            raise Exception(f"invalid mask shape: {mask.shape}")

                        elif len(mask.shape) == 3:
                            mask = np.squeeze(mask, axis=-1)

                        yield image, mask
                    except Exception as e:
                        raise e
                        logger.error(f"could not read either one of these files {example} - {e}")

            dataset = tf.data.Dataset.from_generator(
                gen, (tf.uint8, tf.uint8), ([None, None, None], [None, None]))

        return dataset

    def get(self, data_type=DataType.TRAIN):
        warn('using get() is deprecated. Use tfdataset instead.', DeprecationWarning, stacklevel=2)
        data = self.raw()[data_type]

        def gen():
            for example in data:
                try:
                    yield self.parse_example(example)
                except:
                    logger.error("could not read either one of these files %s" % str(example))
        return gen

    def dict(self, data_type=DataType.TRAIN):
        data = self.raw()[data_type]

        def gen():
            for example in data:
                try:
                    yield self.parse_example_dict(example)
                except:
                    logger.error("could not read either one of these files %s" % str(example))
        return gen

    def test(self):
        """ Testing the functionality of the dataset"""
        logger.debug("testing dataset %s" % self.__class__.__name__)
        for data_type in DataType.get():
            logger.debug("using data_type %s" % data_type)
            tfds = self.tfdataset_v2(data_type)

            for image, mask in tfds:
                logger.debug("image shape: %s, %s |  mask shape: %s, %s, %d (max) | num_classes: %d" % (image.shape, image.dtype, mask.shape, mask.dtype, mask.numpy().max(), self.num_classes))

        return image.numpy(), mask.numpy()


class SequenceDataset(Sequence):

    def __init__(self, ds: Dataset, data_type: str, batch_size: int):
        self.ds = ds
        self.data_type = data_type
        self.x = ds.raw()[data_type]
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        array = list(map(self.ds.parse_example, batch))
        return np.asrray(array[:, 0]) / 255., np.asarray(array[:, 1]) / 255.
