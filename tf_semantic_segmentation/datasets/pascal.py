import numpy as np

from .utils import DataType, get_split_from_list, image_generator, Color
from ..utils import download_and_extract, extract_tar
from .dataset import Dataset

import os
import imageio
import numpy as np
import xmltodict


class PascalVOC2012(Dataset):
    """
        Pascal VOC 2012 training and validation set
        It contains objects like aeroplane, bicycle or bottle. Image can have very different sizes,
        so they must be cropped.

        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    """

    TRAIN_VAL_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    TEST_URL = "http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar"

    def __init__(self, cache_dir, auth={}):
        super(PascalVOC2012, self).__init__(cache_dir)
        self.auth = auth

    @property
    def labels(self):

        return [
            "bg",
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor',
        ]

    @property
    def colormap(self):
        cmap = {}
        colors = PascalVOC2012.get_colors(len(self.labels))
        for i, label in enumerate(self.labels):
            cmap[colors[i]] = label
        return cmap

    @property
    def colormap2index(self):
        cmap = {}
        num_labels = len(self.labels)
        colors = PascalVOC2012.get_colors(num_labels)
        for i in range(num_labels):
            cmap[colors[i]] = i

        return cmap

    @staticmethod
    def _bitget(num, bit):
        return (num >> bit) & 1

    @staticmethod
    def get_colors(num):

        cmap = []
        for i in range(1, num + 1):
            _id = i - 1
            r, g, b = 0, 0, 0
            for j in range(8):
                r = r | (PascalVOC2012._bitget(_id, 0) << (7 - j))
                g = g | (PascalVOC2012._bitget(_id, 1) << (7 - j))
                b = b | (PascalVOC2012._bitget(_id, 2) << (7 - j))
                _id = (_id >> 3)
            cmap.append(Color(r, g, b))
        return cmap

    @staticmethod
    def _get_set(orig_img_dir, seg_img_dir, xml_img_dir, image_set_file):
        """
        Get Original and Classified images from txt file containing list of images

        :param orig_img_dir: str, typically: VOCdevkit/VOC2012/JPEGImages/
        :param seg_img_dir: str, typically: VOCdevkit/VOC2012/SegmentationObjects
        :param image_set_file: str, typically: VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt or val.txt
        :return: tuple, (orig_images, classified_images)
        """

        with open(image_set_file, "r") as handler:
            images = list(map(lambda x: x[:-1], handler.readlines()))
            orig_images = list(
                map(lambda x: os.path.join(orig_img_dir, x + ".jpg"), images))

            segmented_images = list(
                map(lambda x: os.path.join(seg_img_dir, x + ".png"), images))

            xmls = list(
                map(lambda x: os.path.join(xml_img_dir, x + ".xml"), images))

        return list(zip(orig_images, segmented_images, xmls))

    def get_sets(self, d, txts):
        img_set_seg_dir = os.path.join(d, "VOCdevkit/VOC2012/ImageSets/Segmentation")
        orig_images_dir = os.path.join(d, "VOCdevkit/VOC2012/JPEGImages/")
        img_seg_dir = os.path.join(d, "VOCdevkit/VOC2012/SegmentationObject/")
        img_ann_dir = os.path.join(d, "VOCdevkit/VOC2012/Annotations/")

        return [PascalVOC2012._get_set(orig_images_dir, img_seg_dir, img_ann_dir, os.path.join(img_set_seg_dir, txt)) for txt in txts]

    def get_train_val_set(self):
        extracted = download_and_extract(self.TRAIN_VAL_URL, os.path.join(self.cache_dir, 'trainval'))
        trainset, valset = self.get_sets(extracted, ['train.txt', 'val.txt'])

        return trainset, valset

    def get_test_set(self):
        output_dir = os.path.join(self.cache_dir, 'test')
        if self.auth:
            extracted = download_and_extract(self.TEST_URL, output_dir, auth=self.auth)
        else:
            archive = os.path.join(self.cache_dir, 'download.tar')
            if not os.path.exists(archive):
                raise Exception("either supply the auth for downloading the testset or download the file to %s" % archive)

            if not os.path.exists(output_dir):
                extracted = extract_tar(archive, output_dir)
            else:
                extracted = output_dir

        return self.get_sets(extracted, ['test.txt'])[0]

    def raw(self):

        trainset, valset = self.get_train_val_set()

        # do not using test set right now, because the logic for adding empty images
        # to the record is missing
        # TODO
        # testset = self.get_test_set()

        return {
            DataType.TRAIN: trainset,
            DataType.VAL: valset,
            DataType.TEST: []
        }

    def parse_xml(self, xml_path):
        objs = []
        with open(xml_path) as f:
            anno = xmltodict.parse(f.read())
            xml_objs = anno['annotation']['object']

            if not isinstance(xml_objs, list):
                xml_objs = [xml_objs]

            for o in xml_objs:
                bbox = [
                    int(o['bndbox']['xmin']),
                    int(o['bndbox']['ymin']),
                    int(o['bndbox']['xmax']),
                    int(o['bndbox']['ymax'])
                ]
                cx = (bbox[2] - bbox[0]) // 2
                cy = (bbox[3] - bbox[1]) // 2
                name = o['name']
                objs.append({
                    "name": name,
                    "bbox": bbox,
                    "center": (cx, cy)
                })

        return objs

    def parse_example(self, example):
        image_path, mask_path, xml_path = example

        # for the testset the labels path does not exist
        if not os.path.exists(mask_path):
            return imageio.imread(image_path), None

        mask = imageio.imread(mask_path)[:, :, :3]
        unique_colors = np.unique(mask.reshape((-1, 3)), axis=0)
        mask_idx = np.zeros(mask.shape[:2], np.uint8)

        xml_data = self.parse_xml(xml_path)
        # print('labels: ', set([o['name'] for o in xml_data]))
        for obj in xml_data:

            crop = mask[
                obj['bbox'][1]:obj['bbox'][3],
                obj['bbox'][0]:obj['bbox'][2],
                :
            ]
            unique_colors_cropped, counts = np.unique(crop.reshape((-1, 3)), axis=0, return_counts=True)

            # get color with most pixels in cropped area
            sorted_colors = sorted(list(zip(counts, unique_colors_cropped)), key=lambda x: x[0])
            sorted_colors = list(filter(lambda x: x[1].tolist() != [224, 224, 192] and x[1].tolist() != [0, 0, 0], sorted_colors))
            if len(sorted_colors) == 0:
                continue

            label_color = sorted_colors[-1][1]

            idxs = np.where(np.all(mask == label_color, axis=-1))
            class_idx = self.labels.index(obj['name'])
            mask_idx[idxs] = class_idx

        return imageio.imread(image_path), mask_idx

    def get(self, data_type=DataType.TRAIN):

        data = self.raw()[data_type]

        def gen():
            for image_path, mask_path, xml_path in data:
                yield self.parse_example((image_path, mask_path, xml_path))

        return gen


if __name__ == "__main__":
    from ..visualizations import show

    ds = PascalVOC2012('/hdd/datasets/pascalvoc2012')
    ds.summary()

    k = 1
    for image, target in ds.get()():
        print(np.unique(target))
        print(image.shape)
        k += 1
        if k == 100:
            break
        #show.show_images([image, target.astype(np.float32)])
