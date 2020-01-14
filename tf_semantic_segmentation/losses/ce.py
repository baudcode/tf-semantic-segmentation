from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy, Reduction
from tensorflow.keras import backend as K


def ce_label_smoothing_loss(smoothing=0.1):
    def ce_label_smoothing_fixed(y_true, y_pred):
        return K.mean(CategoricalCrossentropy(label_smoothing=smoothing, reduction=Reduction.NONE)(y_true, y_pred))
    return ce_label_smoothing_fixed


def categorical_crossentropy_loss():
    def categorical_crossentropy(y_true, y_pred):
        # y_true, y_pred = to2d(y_true, y_pred)
        return K.mean(CategoricalCrossentropy(reduction=Reduction.NONE)(y_true, y_pred))
    return categorical_crossentropy


def binary_crossentropy_loss():
    def binary_crossentropy(y_true, y_pred):
        return K.mean(BinaryCrossentropy(reduction=Reduction.NONE)(y_true, y_pred))
    return binary_crossentropy
