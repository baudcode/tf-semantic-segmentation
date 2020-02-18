from tf_semantic_segmentation.losses import utils
import numpy as np
import tensorflow as tf


def test_to_1d_2d():
    arr = np.zeros((64, 64, 3), np.uint8)

    arr1d = utils.to1d(arr)
    assert(arr1d.shape[0] == np.prod(arr.shape))

    arr2d = utils.to2d(arr)
    assert(arr2d.shape[0] == arr.shape[0] and arr2d.shape[1] == np.prod(arr.shape[1:]))


def test_round_if_needed():
    arr = np.ones((64, 64)) * 0.7
    t = utils.round_if_needed(arr, 0.5).numpy()
    np.testing.assert_equal(t, np.ones((64, 64)))


def test_gather_channels():
    arr = np.ones((1, 32, 32, 3))
    t = utils.gather_channels(arr, indexes=[1, 2])
    assert(t[0].shape == (1, 32, 32, 2))
    np.testing.assert_array_equal(t[0], arr[:, :, :, 1:])


def test_image2onehot():
    num_classes = 5
    idx = 2
    onehot = np.zeros((1, 64, 64, num_classes), np.float32)
    onehot[:, :, :, idx] = 1.0

    image = utils.onehot2image(onehot)
    arr = np.ones((1, 64, 64, 1), np.uint8) / (num_classes - 1.0) * idx
    np.testing.assert_array_equal(arr, image)


def test_expand_binary():
    binary = np.ones((1, 64, 64, 1), np.float32)
    onehot = utils.expand_binary(binary)
    assert(onehot.shape == (1, 64, 64, 2))
    assert(onehot.numpy().dtype == 'float32')

    onehot = onehot.numpy()
    np.testing.assert_array_equal(onehot[:, :, :, 1], np.squeeze(binary, axis=-1))
    np.testing.assert_array_equal(onehot[:, :, :, 0], np.squeeze(np.zeros_like(binary), axis=-1))


def test_average():
    batch = np.ones((5, 64, 64, 1), np.float32) * 5
    assert(utils.average(tf.convert_to_tensor(batch)).numpy() == 5.0)
    batch[0, :, :, :] = 0.0
    assert(utils.average(tf.convert_to_tensor(batch)).numpy() == 4.0)
    assert(utils.average(tf.convert_to_tensor(batch), per_image=True).numpy() == 4.0)
