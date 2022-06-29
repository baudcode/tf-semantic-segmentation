import tensorflow as tf
from .utils import onehot2image


def ssim_loss():
    def ssim(y_true, y_pred):
        num_channels = y_true.shape[-1]

        y_true = onehot2image(y_true)
        y_pred = onehot2image(y_pred)

        ssim_batch = tf.image.ssim(y_true, y_pred, max_val=float(num_channels), filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03)
        ssim = tf.reduce_mean(ssim_batch, axis=-1)

        return 1.0 - ssim
    return ssim


def ssim_ms_loss():
    def ssim_ms(y_true, y_pred):
        num_classes = y_true.shape[-1]
        y_true = onehot2image(y_true)
        y_pred = onehot2image(y_pred)

        ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=float(num_classes))
        return 1.0 - ssim
    return ssim_ms
