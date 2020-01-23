from tensorflow.keras import backend as K
import tensorflow as tf
from .utils import gather_channels


def binary_focal_loss(gamma=2.0, alpha=0.25, **kwargs):
    r"""Implementation of Focal Loss from the paper in binary classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr) \
               - (1 - gt) * alpha * (pr^gamma) * log(1 - pr)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0

    """

    def binary_focal(gt, pr):
        # clip to prevent NaN's and Inf's
        pr = K.clip(pr, K.epsilon(), 1.0 - K.epsilon())

        loss_1 = - gt * (alpha * K.pow((1 - pr), gamma) * K.log(pr))
        loss_0 = - (1 - gt) * ((1 - alpha) * K.pow((pr), gamma) * K.log(1 - pr))
        loss = K.mean(loss_0 + loss_1)
        return loss
    return binary_focal


def categorical_focal_loss(gamma=2.0, alpha=0.25, class_indexes=None, **kwargs):
    r"""Implementation of Focal Loss from the paper in multiclass classification

    Formula:
        loss = - gt * alpha * ((1 - pr)^gamma) * log(pr)

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        alpha: the same as weighting factor in balanced cross entropy, default 0.25
        gamma: focusing parameter for modulating factor (1-p), default 2.0
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

    """

    def categorical_focal(gt, pr):
        gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)

        # clip to prevent NaN's and Inf's
        pr = K.clip(pr, K.epsilon(), 1.0 - K.epsilon())

        # Calculate focal loss
        loss = - gt * (alpha * K.pow((1 - pr), gamma) * K.log(pr))

        return K.mean(loss)
    return categorical_focal


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
