from ..utils import download_and_extract, extract_zip
from .utils import get_split_from_dirs
from .dataset import Dataset, DataType
import os
import imageio


class MotsChallenge(Dataset):
    """ https://www.vision.rwth-aachen.de/page/mots """
    IMAGE_URL = "https://motchallenge.net/data/MOT17.zip"
    ANNOTATIONS_URL = "https://www.vision.rwth-aachen.de/media/resource_files/instances_motschallenge.zip"

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    @property
    def labels(self):
        return ['bg', 'car', 'pedestrian']

    def raw(self):
        images_dir = download_and_extract(
            self.IMAGE_URL, os.path.join(self.cache_dir, "images"))

        annotations_dir = download_and_extract(
            self.ANNOTATIONS_URL, os.path.join(self.cache_dir, 'annotations'))

        return get_split_from_dirs(images_dir, annotations_dir)

    def get(self, data_type=DataType.TRAIN):

        data = self.raw()[data_type]
        labels = self.labels

        def gen():
            for image_path, target_path in data:
                i = imageio.imread(image_path)
                t = imageio.imread(target_path)

                obj_ids = np.unique(t)
                # to correctly interpret the id of a single object

                mask = np.zeros(i.shape[:2], np.uint8)
                for obj_id in obj_ids:
                    class_id = obj_id // 1000

                    idxs = np.where(np.all(t == obj_id, axis=-1))
                    mask[idxs] = class_id

                yield i, mask
        return gen


if __name__ == "__main__":
    from ..visualizations import show
    import numpy as np

    mots = MotsChallenge('/hdd/datasets/mots/')
    gen = mots.get()
    for image, target in gen():
        print(np.unique(target))
        show.show_images([image, target.astype(np.float32)])
        break
