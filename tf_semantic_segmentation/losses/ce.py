from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy, Reduction
from tensorflow.keras import backend as K
import tensorflow as tf
from .utils import to2d, to1d


def ce_label_smoothing_loss(smoothing=0.1):
    def ce_label_smoothing_fixed(y_true, y_pred):
        # y_true, y_pred = to2d(y_true), to2d(y_pred)
        return K.mean(CategoricalCrossentropy(label_smoothing=smoothing, reduction=Reduction.NONE)(y_true, y_pred))
    return ce_label_smoothing_fixed


def categorical_crossentropy_loss():
    def categorical_crossentropy(y_true, y_pred):
        # y_true, y_pred = to2d(y_true, y_pred)
        # y_true, y_pred = to2d(y_true), to2d(y_pred)
        return K.mean(CategoricalCrossentropy(reduction=Reduction.NONE)(y_true, y_pred))
    return categorical_crossentropy


def binary_crossentropy_loss():
    def binary_crossentropy(y_true, y_pred):
        #y_true, y_pred = to1d(y_true), to1d(y_pred)
        r = K.mean(BinaryCrossentropy(reduction=Reduction.NONE)(y_true, y_pred))
        return tf.cast(r, tf.float32)

    return binary_crossentropy
