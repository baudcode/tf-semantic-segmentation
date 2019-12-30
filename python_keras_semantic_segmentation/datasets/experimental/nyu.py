# https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
from ..utils import download_file
from scipy.io import loadmat
import os
import h5py
import imageio
import tqdm
import numpy as np
from PIL import Image

# TODO: Implement


class NYU_V2:
    LABELED_DATASET_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    @property
    def mean(self):
        return np.array((92.318, 116.190, 97.203), dtype=np.float32)

    def raw(self):
        dst = download_file(self.LABELED_DATASET_URL,
                            destination_dir=self.cache_dir)

        images_dir = os.path.join(self.cache_dir, 'images')
        labels_dir = os.path.join(self.cache_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        path = os.path.join(self.cache_dir, "nyu_depth_v2_labeled.mat")
        with h5py.File(path, 'r') as f:
            images = f['images']
            labels = f['labels']
            for i, image in tqdm.tqdm(enumerate(images)):
                image = np.transpose(image, [1, 2, 0])
                print(np.unique(labels[i]))
                imageio.imwrite(os.path.join(images_dir, "%d.png" % i), image)
                imageio.imwrite(os.path.join(
                    labels_dir, "%d.png" % i), np.expand_dims(labels[i].astype(np.uint8), axis=-1))

    def get(self):

        def gen():
            print()


if __name__ == "__main__":
    NYU_V2('/hdd/datasets/nyu_v2').raw()
