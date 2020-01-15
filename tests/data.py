import imageio
import numpy as np
import tensorflow as tf
from numpy import random

NUM_CLASSES = 5
TEST_IMG = ((random.random((32, 32)) - 0.001) * NUM_CLASSES).astype(np.uint8)
TEST_IMG_BINARY = TEST_IMG.astype(np.float32) / float(NUM_CLASSES - 1)
TEST_IMG_BINARY = np.where(TEST_IMG_BINARY > 0.5, 1.0, 0.0)
TEST_IMG_ONEHOT = tf.one_hot(TEST_IMG, NUM_CLASSES)
TEST_BATCH = np.expand_dims(TEST_IMG_ONEHOT, axis=0)
TEST_BATCH = TEST_BATCH.astype(np.float32)
TEST_BATCH_BINARY = np.expand_dims(np.expand_dims(TEST_IMG_BINARY, axis=0), axis=-1)


def load_large_mask_batch():
    mask = imageio.imread('tests/test.png')
    print(np.unique(mask))
    print(mask.max(), mask.shape)
    mask_onehot = tf.one_hot(mask, 35)
    mask_batch = tf.expand_dims(mask_onehot, axis=0)
    return mask_batch
