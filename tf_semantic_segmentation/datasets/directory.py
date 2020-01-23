from .dataset import Dataset, DataType
from ..utils import get_files
from .utils import get_split
from ..settings import logger

import os

# TODO: Test this


class DirectoryDataset(Dataset):

    def __init__(self, directory, rand=0.2, extensions=['png', 'jpg', 'jpeg']):
        super(DirectoryDataset, self).__init__(directory)

        labels_path = os.path.join(directory, 'labels.txt')

        if not os.path.exists(labels_path):
            raise FileNotFoundError("Please provide a file containing the labels. Cannot find file %s" % labels_path)

        with open(labels_path, 'r') as reader:
            self._labels = list(map(lambda x: x.replace("\n", "").strip(), reader.readlines()))

        print("labels: ", self._labels)

        if len(self._labels) < 2:
            raise AttributeError("Please provide more than 1 label, only found %s in file %s" % (str(self._labels), labels_path))

        masks_dir = os.path.join(directory, 'masks')
        images_dir = os.path.join(directory, 'images')

        if os.path.exists(os.path.join(directory, 'train')):
            logger.info("using train, val, test found in directory")

            self.split = {}
            for data_type in [DataType.TRAIN, DataType.TEST, DataType.VAL]:
                d_masks_dir = os.path.join(directory, data_type, 'masks')
                d_images_dir = os.path.join(directory, data_type, 'images')

                if not os.path.exists(d_masks_dir) or not os.path.exists(d_images_dir):
                    logger.warning("either %s or %s does not exist, getting 0 examples for data_type %s" % (d_masks_dir, d_images_dir, data_type))
                    self.split[data_type] = []
                    continue

                masks = get_files(d_masks_dir, extensions=extensions)
                images = get_files(d_images_dir, extensions=extensions)

                if len(images) != len(masks):
                    raise Exception("len(images)=%d (%s) does not equal len(masks)=%d (%s)" % (len(images), d_images_dir, len(masks), d_masks_dir))

                self.split[data_type] = list(zip(images, masks))

        elif os.path.exists(masks_dir) and os.path.exists(images_dir):
            masks = get_files(masks_dir, extensions=extensions)
            if len(masks) == 0:
                raise Exception("cannot find any images in masks directory %s" % masks_dir)

            images = get_files(images_dir, extensions=extensions)
            if len(images) == 0:
                raise Exception("cannot find any pictures in images directory %s" % images_dir)

            if len(images) != len(masks):
                raise Exception("len(images)=%d does not equal len(masks)=%d" % (len(images), len(masks)))

            trainset = list(zip(images, masks))
            self.split = get_split(trainset, rand=lambda: rand)

    @property
    def labels(self):
        return self._labels

    def raw(self):
        return self.split
