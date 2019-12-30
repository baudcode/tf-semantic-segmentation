from python_keras_semantic_segmentation import utils
import os

import imageio
import csv
import numpy as np
from scipy.io import loadmat
from ..utils import DataType, get_split_from_list

# TODO: check it, why is it not working?


class Ade20k:

    def __init__(self, data_output_dir):
        self.data_output_dir = data_output_dir
        self.data_url = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"

    @property
    def num_classes(self):
        return 151

    def raw(self):
        extract_dir = utils.download_and_extract(
            self.data_url, self.data_output_dir)

        extract_dir = os.path.join(extract_dir, "ADE20K_2016_07_26", "images")
        val_dir = os.path.join(extract_dir, 'validation')
        train_dir = os.path.join(extract_dir, 'training')

        train_inputs = utils.get_files(train_dir, ['jpg'])
        val_inputs = utils.get_files(val_dir, ['jpg'])

        train_seg = list(filter(lambda x: x.endswith(
            '_seg.png'), utils.get_files(train_dir, ['png'])))
        val_seg = list(filter(lambda x: x.endswith('_seg.png'),
                              utils.get_files(val_dir, ['png'])))

        train_txt_files = utils.get_files(train_dir, extensions=['txt'])
        val_txt_files = utils.get_files(val_dir, extensions=['txt'])

        assert(len(train_inputs) == len(train_seg))
        assert(len(train_seg) == len(train_txt_files))

        # classes are coded in the RG channels and instances in the B channel
        train_outputs = zip(train_seg, train_txt_files)

        train_data = list(zip(train_inputs, list(
            map(lambda x: {"image": x[0], "txt": x[1]}, train_outputs))))
        train_data, test_data = get_split_from_list(train_data, split=0.8)

        val_outputs = zip(val_seg, val_txt_files)
        val_outputs = list(
            map(lambda x: {"image": x[0], "txt": x[1]}, val_outputs))

        data = {
            DataType.TRAIN: train_data,
            DataType.TEST: test_data,
            DataType.VAL: list(zip(val_inputs, val_outputs))
        }

        return data

    def num_examples(self, data_type):
        return len(self.raw()[data_type])

    def parse_txt(self, txt):
        class_names = []
        with open(txt) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='#')
            for row in csv_reader:
                instance_number = int(row[0].strip())
                class_name = row[3].strip()
                orig_raw_name = row[4].strip()
                class_names.append(class_name)
        return class_names

    def get_labels(self):
        # returns 16 categories
        return loadmat(os.path.join(self.data_output_dir, self.data_output_dir, "ADE20K_2016_07_26", "index_ade20k.mat"))['index'][0][0]

    def get(self, data_type=DataType.TRAIN):

        split = self.raw()

        def generator():
            for input_path, targets in split[data_type]:
                inp = imageio.imread(input_path)
                labels = imageio.imread(targets['image'])
                r = labels[:, :, 0].astype(np.int32)
                g = labels[:, :, 1].astype(np.int32)
                labels = r / 10 + g
                # labels = (r / 10) * 256 + g
                print(np.unique(labels))
                # labels = ((labels.astype(np.float32) / 65535)
                #          * 255).astype(np.uint8)

                # print(np.unique(labels))
                # imageio.imwrite("labels.png", labels.astype(np.float32))
                labels = labels.astype(np.uint8)

                yield inp, labels

        return generator


if __name__ == "__main__":
    ade20k = Ade20k('/hdd/datasets/ade20k')
    for image, target in ade20k.get()():
        pass
