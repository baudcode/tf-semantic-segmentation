import tqdm
import imageio
import os
import numpy as np
import cv2
import random
import shutil

from .dataset import Dataset
from .utils import DataType, get_split_from_list, get_split

from ..visualizations import show
from ..utils import download_and_extract, get_files
from ..processing import ColorMode
from ..settings import logger
from ..threading import parallize_v3


class ShapesDS(Dataset):

    SHAPES = ['rectangle', 'triangle', 'circle']

    def __init__(self, cache_dir, num_examples=10000, size=(512, 512), overwrite=False, color_mode=ColorMode.RGB, max_shapes_per_example=3):
        super(ShapesDS, self).__init__(cache_dir)
        self._num_examples = num_examples
        self.images_dir = os.path.join(self.cache_dir, 'images')
        self.masks_dir = os.path.join(self.cache_dir, 'masks')
        self.max_shapes_per_example = max_shapes_per_example
        self.overwrite = overwrite
        self.color_mode = color_mode
        self.size = size

        os.makedirs(self.masks_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.trainset = parallize_v3(self.create_example, list(range(self._num_examples)), desc='creating shapes dataset')

    @property
    def labels(self):
        return self.SHAPES

    def draw_shape(self, image, shape, x, y, w, h, color):
        assert(shape in self.SHAPES)
        if shape == "rectangle":
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=cv2.FILLED)
        elif shape == "circle":
            cv2.circle(image, (x, y), w, color, thickness=cv2.FILLED)
        elif shape == 'triangle':
            triangle_cnt = np.asarray([[x, y], [x + w, y], [x + w // 2, y + h]], np.int32)
            cv2.drawContours(image, [triangle_cnt], 0, color, thickness=cv2.FILLED)
        return image

    def get_random_color(self):
        if self.color_mode == ColorMode.RGB:
            return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif self.color_mode == ColorMode.GRAY:
            return (random.randint(0, 255))

    def draw_shapes(self, image, shapes):
        mask = np.zeros((self.size[1], self.size[0]), np.uint8)
        for shape in shapes:
            color = self.get_random_color()
            qx = self.size[0] // 4
            qy = self.size[1] // 4
            x = random.randint(qx, self.size[0] - qx)
            y = random.randint(qy, self.size[1] - qy)
            w = random.randint(0, self.size[0] // 2)
            h = random.randint(0, self.size[1] // 2)
            image = self.draw_shape(image, shape, x, y, w, h, color)
            mask = self.draw_shape(mask, shape, x, y, w, h, self.SHAPES.index(shape) + 1)
        return image, mask

    def create_example(self, i):
        mask_path = os.path.join(self.masks_dir, "%d.png" % i)
        image_path = os.path.join(self.images_dir, "%d.png" % i)

        if not self.overwrite and os.path.exists(mask_path) and os.path.exists(image_path):
            return image_path, mask_path

        num_shapes = random.randint(0, self.max_shapes_per_example)
        random_shapes = [random.choice(self.SHAPES) for i in range(num_shapes)]

        if self.color_mode == ColorMode.RGB:
            image = np.zeros((self.size[1], self.size[0], 3), np.uint8)

        elif self.color_mode == ColorMode.GRAY:
            image = np.zeros((self.size[1], self.size[0], 1), np.uint8)
        else:
            raise Exception("unknown color mode %s" % self.color_mode.name)

        image, mask = self.draw_shapes(image, random_shapes)
        imageio.imwrite(image_path, image)
        imageio.imwrite(mask_path, mask)
        return image_path, mask_path

    def raw(self):

        print(len(self.trainset), self.trainset[0])
        return get_split(self.trainset)


if __name__ == "__main__":

    ds = ShapesDS('/hdd/datasets/shapes', 1000)
    for image_path, mask_path in ds.raw()[DataType.TRAIN]:
        show.show_images([imageio.imread(image_path), imageio.imread(mask_path).astype(np.float32)])
