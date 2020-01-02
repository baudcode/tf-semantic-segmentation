from tensorflow import keras
import numpy as np
import cv2
from .pre import preprocess, ColorMode
from ..datasets import DataType


class DataLoader(keras.utils.Sequence):

    def __init__(self, ds, data_type, size, color_mode=ColorMode.RGB, batch_size=1):
        self.batch_size = batch_size
        self.data_type = data_type
        self.color_mode = color_mode
        self.size = size

        self.is_training = data_type == DataType.TRAIN
        self.shuffle = data_type == DataType.TRAIN
        self.indexes = np.arange(ds.num_examples(data_type))
        print('num_examples: ', data_type, ds.num_examples(data_type))
        print("len: ", len(self.indexes) // batch_size)
        self.ds = ds
        self.on_epoch_end()
        self.raw_data = self.ds.raw()

    def __getitem__(self, i):
        print("get item:", i, len(self))
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        images = []
        masks = []
        for j in range(start, stop):
            print('j: ', j)
            example = self.raw_data[self.data_type][j]
            image, mask = self.ds.parse_example(example)
            image, mask = preprocess(image, self.size, self.color_mode, num_classes=self.ds.num_classes, target=mask, is_training=self.is_training)

            images.append(image)
            masks.append(mask)

        # transpose list of lists
        batch = [np.stack(images, axis=0), np.stack(masks, axis=0)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


if __name__ == "__main__":
    from ..datasets.pascal import PascalVOC2012, DataType
    ds = PascalVOC2012('/hdd/datasets/pascal')
    loader = DataLoader(ds, DataType.TRAIN, (512, 512), batch_size=5)

    for data in loader:
        print(data[0].shape, data[1].shape)
        break
