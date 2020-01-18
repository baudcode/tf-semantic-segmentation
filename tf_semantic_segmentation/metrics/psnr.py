from tensorflow.keras import backend as K
import numpy as np
from ..losses import onehot2image


def psnr(SMOOTH=1e-9):
    def psnr(y_true, y_pred):
        # scale between 0 and 255
        y_true = onehot2image(y_true) * 255.
        y_pred = onehot2image(y_pred) * 255.

        # mean squared error and scale
        mse = K.mean(K.square(y_pred - y_true)) + SMOOTH
        k1 = 20 * K.log(255.0) / K.log(10.0)
        k2 = np.float32(10.0 / np.log(10)) * K.log(mse)
        return k1 - k2
    return psnr
