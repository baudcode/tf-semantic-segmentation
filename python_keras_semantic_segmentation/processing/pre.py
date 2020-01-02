from . import ColorMode

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical


def preprocess(image, size, color_mode, num_classes=None, target=None, is_training=True):
    image = cv2.resize(image, size, cv2.INTER_NEAREST)
    if target is not None:
        target = cv2.resize(target, size, interpolation=cv2.INTER_NEAREST)

    is_square = size[0] == size[1]

    if is_training:

        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            if target is not None:
                target = cv2.flip(target, 0)

        if is_square and np.random.random() > 0.5:
            for i in range(int(np.random.randint(0, 3))):
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                if target is not None:
                    target = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)

        elif not is_square and np.random.random() > 0.5:
            image = cv2.rotate(image, cv2.ROTATE_180)
            if target is not None:
                target = cv2.rotate(target, cv2.ROTATE_180)

        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            if target is not None:
                target = cv2.flip(target, 1)

    image = (image.astype(np.float32) / 255.0)

    # limit to 3 channels
    if len(image.shape) == 3:
        image = image[:, :, :3]

    if len(image.shape) == 2 and color_mode == ColorMode.RGB:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 2 and color_mode == ColorMode.GRAY:
        image = np.expand_dims(image, axis=-1)
    elif len(image.shape) == 3 and color_mode == ColorMode.GRAY:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif (len(image.shape) == 3 and color_mode == ColorMode.RGB) or (len(image.shape) == 2 and color_mode == ColorMode.GRAY):
        pass
    else:
        raise Exception("cannot handle imaege of shape %s" % str(image.shape))

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if target is not None:
        target = np.reshape(target, (-1))
        target = to_categorical(target, num_classes)
        target = np.reshape(target, (size[1], size[0], num_classes))
        return image, target
    else:
        return image


def gen(g, size, batch_size, num_classes, color_mode, is_training=True):
    while 1:
        image_batch = []
        target_batch = []

        for image, target in g():
            image, target = preprocess(
                image, size, color_mode, num_classes=num_classes, target=target, is_training=is_training)
            image_batch.append(image)
            target_batch.append(target)
            if len(image_batch) == batch_size:
                yield np.asarray(image_batch), np.asarray(target_batch)
                image_batch, target_batch = [], []
