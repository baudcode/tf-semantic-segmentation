import tensorflow as tf
from tensorflow.keras import backend as K


def ssim2():
    """Computes a differentiable structured image similarity measure."""
    def ssim2(x, y):
        c1 = 0.01**2
        c2 = 0.03**2

        def pool(x):
            return tf.nn.max_pool2d(x, (3, 3), strides=(1, 1), padding="VALID")

        mu_x = pool(x)
        mu_y = pool(y)
        sigma_x = pool(x**2) - mu_x**2
        sigma_y = pool(y**2) - mu_y**2
        sigma_xy = pool(x * y) - mu_x * mu_y
        ssim_n = (2. * mu_x * mu_y + c1) * (2. * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        ssim = ssim_n / ssim_d
        return tf.reduce_mean(tf.clip_by_value((1.0 - ssim), 0, 1))
    return ssim2


def ssim():
    # source: https://gist.github.com/Dref360/a48feaecfdb9e0609c6a02590fd1f91b
    def ssim(y_true, y_pred):
        y_true = tf.expand_dims(y_true, -1)
        y_pred = tf.expand_dims(y_pred, -1)
        y_true = tf.transpose(y_true, [0, 2, 3, 1])
        y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

        patches_true = tf.image.extract_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
        patches_pred = tf.image.extract_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

        u_true = K.mean(patches_true, axis=3)
        u_pred = K.mean(patches_pred, axis=3)
        var_true = K.var(patches_true, axis=3)
        var_pred = K.var(patches_pred, axis=3)
        std_true = K.sqrt(var_true)
        std_pred = K.sqrt(var_pred)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
        denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
        ssim /= denom
        ssim = tf.where(tf.math.is_nan(ssim), K.zeros_like(ssim), ssim)
        return ssim
    return ssim
