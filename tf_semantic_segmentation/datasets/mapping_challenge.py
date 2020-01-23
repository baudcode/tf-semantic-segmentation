from .dataset import Dataset, DataType
from .utils import get_files
from ..utils import download_and_extract
from ..visualizations import masks, show

import os
import tqdm
import json
import numpy as np
import cv2
import imageio


class MappingChallenge(Dataset):
    """ https://www.crowdai.org/challenges/mapping-challenge/dataset_files """

    TRAIN_URL = "https://crowdai-prd.s3.eu-central-1.amazonaws.com/dataset_files/challenge_25/8e089a94-555c-4d7b-8f2f-4d733aebb058_train.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAILFF3ZEGG7Y4HXEQ%2F20191218%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20191218T022310Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=5d49a86529cc2078a6ab4da22e7fe1b9e3b5dff2187fd106190799cb034eaa5c"
    VAL_URL = "https://crowdai-prd.s3.eu-central-1.amazonaws.com/dataset_files/challenge_25/0a5c561f-e361-4e9b-a3e2-94f42a003a2b_val.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAILFF3ZEGG7Y4HXEQ%2F20191218%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20191218T022310Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=76d8cbe79f17cdd673fe44135c3f0c456ae25f89756854b67958441a09d15716"

    def __init__(self, cache_dir):
        super(MappingChallenge, self).__init__(cache_dir)
        self.already_extracted = False

    @property
    def labels(self):
        return ['bg', 'building']

    def read_annotations_extract_masks(self, path, output_dir):

        print("loading annotation data %s" % path)
        data = json.load(open(path, 'r'))

        segmentation_data = {}

        categories = {c['id']: c for c in data['categories']}

        # label must start at 1
        for i, c in enumerate(data['categories']):
            categories[c['id']]['label'] = i + 1

        print(categories)

        for image_info in tqdm.tqdm(data['images'], desc='building image info (annotations)'):
            image_id = image_info['id']
            segmentation_data[image_id] = image_info

        for seg_info in tqdm.tqdm(data['annotations'], desc='writing masks'):
            info = segmentation_data[seg_info['image_id']]
            file_name = info['file_name']
            file_name, ext = os.path.splitext(file_name)
            output_path = os.path.join(output_dir, '%s.png' % file_name)

            if os.path.exists(output_path):
                continue

            category = categories[seg_info["category_id"]]

            mask = np.zeros((info['height'], info['width']), np.uint8)
            for poly in seg_info['segmentation']:
                print(poly)
                cnt = np.asarray(poly).reshape((-1, 2)).astype(np.int32)
                mask = cv2.drawContours(mask, [cnt], 0, int(category['label']), cv2.FILLED, cv2.LINE_AA)

            image = imageio.imread(output_path)
            show.show_images([mask.astype(np.float32), image])
            imageio.imwrite(output_path, mask)

    def raw(self):

        train_dir = os.path.join(self.cache_dir, 'train')
        val_dir = os.path.join(self.cache_dir, 'val')

        val_masks_dir = os.path.join(val_dir, 'masks')
        train_masks_dir = os.path.join(train_dir, 'masks')

        if not self.already_extracted:

            train_dir = download_and_extract(self.TRAIN_URL, train_dir, file_name='train.tar.gz')
            val_dir = download_and_extract(self.VAL_URL, val_dir, file_name='val.tar.gz')

            os.makedirs(train_masks_dir, exist_ok=True)
            os.makedirs(val_masks_dir, exist_ok=True)

            train_annotations_path = os.path.join(train_dir, 'train', 'annotation.json')
            val_annotations_path = os.path.join(val_dir, 'val', 'annotation.json')

            self.read_annotations_extract_masks(train_annotations_path, train_masks_dir)
            self.read_annotations_extract_masks(val_annotations_path, val_masks_dir)
            self.already_extracted = True

        train_images = get_files(train_dir, extensions=['jpg'])
        val_images = get_files(val_dir, extensions=['jpg'])

        train_masks = get_files(train_masks_dir, extensions=['png'])
        val_masks = get_files(train_masks_dir, extensions=['png'])

        return {
            DataType.TRAIN: list(zip(train_images, train_masks)),
            DataType.VAL: list(zip(val_images, val_masks)),
            DataType.TEST: []
        }


if __name__ == "__main__":
    from ..visualizations import show
    for image, mask in MappingChallenge('/hdd/datasets/mappingchallenge').get(DataType.TRAIN)():
        show.show_images([image, mask.astype(np.float32)])
