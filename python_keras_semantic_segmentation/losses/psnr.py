from tensorflow.keras import backend as K
import numpy as np


def psnr():
    def psnr(y_true, y_pred):
        mse = K.mean(K.square(y_pred - y_true))
        k1 = 20 * K.log(255.0) / K.log(10.0)
        k2 = np.float32(10.0 / np.log(10)) * K.log(mse)
        return k1 - k2
    return psnr
