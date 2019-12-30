from tensorflow import keras
import numpy as np
import cv2
# TODO: DataLoader


class Dataloader(keras.utils.Sequence):

    def __init__(self, ds, data_type, batch_size=1, shuffle=False):
        self.batch_size = batch_size
        self.data_type = data_type
        self.shuffle = shuffle
        self.indexes = np.arange(ds.num_examples(data_type))
        self.ds = ds
        self.on_epoch_end()
        self.raw_data = self.ds.raw()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        images = []
        masks = []
        for j in range(start, stop):
            example = self.raw_data[self.data_type][j]
            image, mask = self.ds.parse_example(example)

            image = cv2.resize(image, (500, 375))
            image = image.asteype(np.float32) / 255.

            mask = cv2.resize(mask, (500, 375))

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
    from .pascal import PascalVOC2012, DataType
    ds = PascalVOC2012('/hdd/datasets/pascal')
    loader = Dataloader(ds, DataType.TRAIN, batch_size=5)

    for data in loader:
        break
