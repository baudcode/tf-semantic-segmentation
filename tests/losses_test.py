from tf_semantic_segmentation import losses
import numpy as np
from .data import TEST_BATCH_BINARY, TEST_BATCH
import tensorflow as tf

# x = np.array([-2.2, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6])
# y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_losses():
    print("testing...")
    np.testing.assert_allclose(TEST_BATCH_BINARY, TEST_BATCH_BINARY)
    for name, loss in losses.losses_by_name.items():
        assert(losses.get_loss_by_name(name) == loss)
        print("testing loss %s" % name)
        with tf.device("/cpu:0"):
            if "binary" in name:
                l = loss(tf.convert_to_tensor(TEST_BATCH_BINARY), tf.convert_to_tensor(TEST_BATCH_BINARY))
            else:
                l = loss(tf.convert_to_tensor(TEST_BATCH), tf.convert_to_tensor(TEST_BATCH))
            print(l)

            if name == "ce_label_smoothing":
                np.testing.assert_almost_equal(l, 0.2, decimal=2)
            elif name == "categorical_crossentropy":
                np.testing.assert_almost_equal(l, 0.0, decimal=6)
            else:
                np.testing.assert_almost_equal(l, 0.0)
