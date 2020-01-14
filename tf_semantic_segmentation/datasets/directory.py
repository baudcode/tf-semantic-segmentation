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
            labels = reader.readlines(labels_path).replace("\n", '')
            self.labels = list(map(lambda x: x.strip(), labels))

        if len(self.labels) < 2:
            raise AttributeError("Please provide more than 1 label, only found %s in file %s" % (str(self.labels), labels_path))

        masks_dir = os.path.join(directory, 'masks')
        images_dir = os.path.join(directory, 'images')

        if not os.path.exists(masks_dir):
            raise FileNotFoundError("cannot find directory containing the masks %s" % masks_dir)

        if not os.path.exists(images_dir):
            raise FileNotFoundError("cannot find directory containing the images %s" % images_dir)

        if os.path.exists(os.path.join(images_dir, 'train')):
            logger.info("using train, val, test found in directory")

            self.split = {}
            for data_type in [DataType.TRAIN, DataType.TEST, DataType.VAL]:
                d_masks_dir = os.path.join(masks_dir, data_type)
                masks = get_files(d_masks_dir, extensions=extensions)

                d_images_dir = os.path.join(images_dir, data_type)
                images = get_files(d_images_dir, extensions=extensions)

                if len(images) != len(masks):
                    raise Exception("len(images)=%d (%s) does not equal len(masks)=%d (%s)" % (len(images), d_images_dir, len(masks), d_masks_dir))

                self.split[data_type] = list(zip(images, masks))

        else:
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

    def raw(self):
        return self.split
