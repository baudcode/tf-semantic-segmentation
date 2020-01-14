from tf_semantic_segmentation import losses
import numpy as np
from .data import TEST_BATCH
import tensorflow as tf

# x = np.array([-2.2, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6])
# y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

"""
def test_losses():
    for name, loss in losses.losses_by_name.items():
        print("testing loss %s" % name)
        print(loss(TEST_BATCH, TEST_BATCH))
"""


def test_ssim():
    with tf.device("/gpu:0"):
        np.testing.assert_almost_equal(losses.ssim_loss()(TEST_BATCH, TEST_BATCH), 0.0)


def test_ce():
    with tf.device("/gpu:0"):
        np.testing.assert_almost_equal(losses.ce_label_smoothing_loss(smoothing=0.05)(TEST_BATCH, TEST_BATCH), 0.0, decimal=2)
        np.testing.assert_almost_equal(losses.categorical_crossentropy_loss()(TEST_BATCH, TEST_BATCH), 0.0, decimal=6)
        # assert(losses.categorical_focal_loss()(TEST_BATCH, TEST_BATCH) < 1e-6)


def test_dice():
    with tf.device("/gpu:0"):
        np.testing.assert_almost_equal(losses.dice_loss()(TEST_BATCH, TEST_BATCH), 0.0)


def tests_tversky_loss():
    with tf.device("/gpu:0"):
        np.testing.assert_almost_equal(losses.tversky_loss()(TEST_BATCH, TEST_BATCH), 0.0)


def tests_tversky_focal_loss():
    with tf.device("/gpu:0"):
        np.testing.assert_almost_equal(losses.focal_tversky_loss()(TEST_BATCH, TEST_BATCH), 0.0)


def test_focal():
    with tf.device("/gpu:0"):
        np.testing.assert_almost_equal(losses.categorical_focal_loss()(TEST_BATCH, TEST_BATCH), 0.0)


def test_binary_losses():
    TEST_BATCH2 = losses.labels2image(TEST_BATCH)
    np.testing.assert_almost_equal(losses.binary_focal_loss()(TEST_BATCH2, TEST_BATCH2), 0.0)
    np.testing.assert_almost_equal(losses.binary_crossentropy_loss()(TEST_BATCH2, TEST_BATCH2), 0.0)
    np.testing.assert_almost_equal(losses.ssim_loss()(TEST_BATCH2, TEST_BATCH2), 0.0)
