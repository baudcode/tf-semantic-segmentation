import tensorflow as tf
from ..losses import onehot2image


def ssim(filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03):
    def ssim(y_true, y_pred):
        # find label max
        y_true = onehot2image(y_true)
        y_pred = onehot2image(y_pred)

        ssim_batch = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
        return tf.reduce_mean(ssim_batch, axis=-1)
    return ssim
