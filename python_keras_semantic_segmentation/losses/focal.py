from tensorflow.keras import backend as K
import tensorflow as tf


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def smooth_l1(sigma=3.0):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        The smooth L1 loss of y_pred w.r.t. y_true.
    """
    def smooth_l1(y_true, y_pred):

        sigma_squared = sigma ** 2
        # separate target and state
        regression = y_pred
        regression_target = y_true

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = tf.abs(regression_diff)
        regression_loss = tf.where(
            tf.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * tf.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        return tf.reduce_mean(regression_loss)
    return smooth_l1
