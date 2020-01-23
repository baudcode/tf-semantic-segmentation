# combine cifar10 and tinyimagenet

from ..visualizations import show
import pickle
import tqdm
import imageio
import os
import numpy as np
import json
import cv2
from array import array as normal_array
import struct
import random

from .dataset import Dataset
from ..utils import download_and_extract, get_files
from .utils import DataType, get_split_from_list
from ..settings import logger


class Toy(Dataset):

    TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    MNIST_URLS = {
        "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    }

    def __init__(self, cache_dir):
        super(Toy, self).__init__(cache_dir)
        self.create()

    @property
    def labels(self):
        return ['background', 'foreground']

    def get_mnist(self):
        """
        Some is taken from https://github.com/sorki/python-mnist/blob/master/mnist/loader.py
        """
        mnist_dir = os.path.join(self.cache_dir, 'mnist')
        download_and_extract(self.MNIST_URLS['train_images'], os.path.join(mnist_dir, 'dtrain'))
        download_and_extract(self.MNIST_URLS['test_images'], os.path.join(mnist_dir, 'dtest'))

        train_file = os.path.join(mnist_dir, 'dtrain', 'train-images-idx3-ubyte')
        test_file = os.path.join(mnist_dir, 'dtest', 't10k-images-idx3-ubyte')

        data = {
            "train": [],
            "test": []
        }

        for path, data_type in zip([train_file, test_file], [DataType.TRAIN, DataType.TEST]):

            output_dir = os.path.join(mnist_dir, data_type)
            os.makedirs(output_dir, exist_ok=True)

            with open(path, 'rb') as file:
                magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
                if magic != 2051:
                    raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
                image_data = normal_array("B", file.read())

            images = []
            for i in range(size):
                images.append([0] * rows * cols)

            for i in range(size):
                images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

            image_paths = []
            for i, image in tqdm.tqdm(enumerate(images), desc='saving mnist images for %s' % data_type):
                image_path = os.path.join(output_dir, "%d.jpg" % (i))
                image_paths.append(image_path)
                if not os.path.exists(image_path):
                    imageio.imwrite(image_path, np.array(image, dtype=np.uint8).reshape([28, 28]))

            data[data_type] = image_paths

        return data[DataType.TRAIN], data[DataType.TEST]

    def get_tiny_imagenet(self):
        tiny_dir = download_and_extract(self.TINY_IMAGENET_URL, os.path.join(self.cache_dir, 'tinyimagenet'))
        tiny_train_dir = os.path.join(tiny_dir, "tiny-imagenet-200/train/")
        tiny_val_dir = os.path.join(tiny_dir, "tiny-imagenet-200/val/")

        print(tiny_train_dir, tiny_val_dir)
        train_files = get_files(tiny_train_dir, extensions=['jpeg'])
        val_files = get_files(tiny_val_dir, extensions=['jpeg'])
        return train_files, val_files

    def raw(self):

        data = {}

        for data_type in [DataType.TRAIN, DataType.TEST, DataType.VAL]:

            masks_dir = os.path.join(self.cache_dir, data_type, 'masks')
            inputs_dir = os.path.join(self.cache_dir, data_type, 'inputs')
            input_files = get_files(inputs_dir, extensions=['png'])
            mask_files = get_files(masks_dir, extensions=['png'])

            data[data_type] = list(zip(input_files, mask_files))

        return data

    def create(self):
        tiny_train_files, tiny_val_files = self.get_tiny_imagenet()
        print('tiny: ', len(tiny_train_files), len(tiny_val_files))
        blend_train_files, blend_val_files = self.get_mnist()
        print('blend: ', len(blend_train_files), len(blend_val_files))

        def make_mask(blend_path, tiny_path):
            blend = imageio.imread(blend_path)
            blend = cv2.resize(blend, (64, 64), interpolation=cv2.INTER_AREA)

            _, blend = cv2.threshold(blend, 127, 255, cv2.THRESH_BINARY)

            blend_1d = blend.copy()
            # blend_1d = 255 - blend.astype(np.uint8)
            blend_1d = np.expand_dims(blend_1d, axis=-1)

            cn = random.randint(0, 4)
            if cn == 0:
                blend_3d = np.concatenate([blend_1d, blend_1d, blend_1d], axis=-1)
            elif cn == 1:
                blend_3d = np.concatenate([blend_1d, np.zeros_like(blend_1d), np.zeros_like(blend_1d)], axis=-1)
            elif cn == 2:
                blend_3d = np.concatenate([np.zeros_like(blend_1d), blend_1d, np.zeros_like(blend_1d)], axis=-1)
            else:
                blend_3d = np.concatenate([np.zeros_like(blend_1d), np.zeros_like(blend_1d), blend_1d], axis=-1)

            # blend_3d = np.where(blend_3d == [0, 0, 0], np.ones_like(blend_3d) * 255, blend_3d)
            tiny = imageio.imread(tiny_path)
            if len(tiny.shape) == 2:
                tiny = cv2.cvtColor(tiny, cv2.COLOR_GRAY2RGB)

            masked = tiny.copy()
            masked = masked | blend_3d
            # show.show_images([masked, blend.astype(np.float32), blend_3d, tiny])
            return masked, (blend / 255).astype(np.uint8)

        trainset = list(zip(blend_train_files, tiny_train_files[:len(blend_train_files)]))
        valset = list(zip(blend_val_files, tiny_val_files[:len(blend_val_files)]))
        trainset, testset = get_split_from_list(trainset, split=0.9)
        sets = {
            DataType.TRAIN: trainset,
            DataType.VAL: valset,
            DataType.TEST: testset
        }
        toy_sets = {
            DataType.TRAIN: [],
            DataType.VAL: [],
            DataType.TEST: []
        }

        for data_type, ds in sets.items():

            masks_dir = os.path.join(self.cache_dir, data_type, 'masks')
            inputs_dir = os.path.join(self.cache_dir, data_type, 'inputs')
            os.makedirs(masks_dir, exist_ok=True)
            os.makedirs(inputs_dir, exist_ok=True)

            for k, (blend_path, tiny_path) in tqdm.tqdm(enumerate(ds), desc='creating %s set' % data_type, total=len(ds)):
                mask_path = os.path.join(masks_dir, "%d.png" % k)
                inputs_path = os.path.join(inputs_dir, "%d.png" % k)

                if os.path.exists(mask_path) and os.path.exists(inputs_path):
                    toy_sets[data_type].append((inputs_path, mask_path))
                    continue

                # get mask and image
                inputs, mask = make_mask(blend_path, tiny_path)
                assert(inputs.shape == (64, 64, 3))
                assert(mask.shape == (64, 64))
                # write images
                imageio.imwrite(mask_path, mask)
                imageio.imwrite(inputs_path, inputs)

                toy_sets[data_type].append((inputs_path, mask_path))

        return toy_sets

    def get_cifar10(self):
        train_batches = ["data_batch_%d" % b for b in range(1, 6)]
        test_batch = "test_batch"

        data_output_dir = os.path.join(self.cache_dir, 'cifar10')
        cifar_dir = download_and_extract(self.CIFAR_URL, data_output_dir)
        extracted = os.path.join(cifar_dir, "cifar-10-batches-py")

        images_path = os.path.join(data_output_dir, 'images')
        images_train_path = os.path.join(images_path, 'train')
        images_test_path = os.path.join(images_path, 'test')

        os.makedirs(images_path, exist_ok=True)
        os.makedirs(images_test_path, exist_ok=True)
        os.makedirs(images_train_path, exist_ok=True)

        logger.info("[*] cifar10 is extracting data")

        def _get_data(batches, out_dir):
            all_labels = []
            all_filenames = []
            for k in tqdm.trange(len(batches)):
                b = batches[k]
                with open(os.path.join(extracted, b), 'rb') as fo:
                    d = pickle.load(fo, encoding='bytes')
                    filenames = [os.path.join(out_dir, f.decode("utf-8")) for f in d[b'filenames']]
                    labels = d[b'labels']
                    all_labels += labels
                    all_filenames += list(map(str, filenames))

                    data = d[b'data']
                    assert(len(data) == len(filenames))
                    for i in range(len(data)):
                        imageio.imwrite(filenames[i], data[i].reshape((3, 32, 32)).transpose((1, 2, 0)))

            return all_filenames, all_labels

        cache_path = os.path.join(cifar_dir, 'cached.json')
        if os.path.exists(cache_path):
            paths = json.load(open(cache_path, 'r'))
            return paths['train'], paths['test']
        else:
            train_images, _ = _get_data(train_batches, images_train_path)
            test_images, _ = _get_data([test_batch], images_test_path)
            json.dump({"train": train_images, "test": test_images}, open(cache_path, 'w'))
            return train_images, test_images


if __name__ == "__main__":

    ds = Toy('/hdd/dataset/toy')
    for image_path, mask_path in ds.raw()[DataType.TRAIN]:
        print(image_path, mask_path)
        show.show_images([imageio.imread(image_path), imageio.imread(mask_path)])
        break
