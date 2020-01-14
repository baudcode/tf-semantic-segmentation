
from tensorflow.keras import backend as K
from tensorflow.keras import metrics


def mae(y_true, y_pred): return K.mean(metrics.mae(y_true, y_pred))


def binary_accuracy(y_true, y_pred): return K.mean(metrics.binary_accuracy(y_true, y_pred))


def categorical_accuracy(y_true, y_pred): return K.mean(metrics.categorical_accuracy(y_true, y_pred))


def top_3_categorical_accuracy(y_true, y_pred):
    s = K.prod(K.shape(y_true)[1:])
    y_true = K.reshape(y_true, [-1, s])
    y_pred = K.reshape(y_pred, [-1, s])
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_5_categorical_accuracy(y_true, y_pred):
    s = K.prod(K.shape(y_true)[1:])
    y_true = K.reshape(y_true, [-1, s])
    y_pred = K.reshape(y_pred, [-1, s])
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


def top_1_categorical_accuracy(y_true, y_pred):
    s = K.prod(K.shape(y_true)[1:])
    y_true = K.reshape(y_true, [-1, s])
    y_pred = K.reshape(y_pred, [-1, s])
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)
