
from tensorflow.keras import backend as K
from tensorflow.keras import metrics


def mae(y_true, y_pred): return K.mean(metrics.mae(y_true, y_pred))


def binary_accuracy(y_true, y_pred): return K.mean(metrics.binary_accuracy(y_true, y_pred))


def categorical_accuracy(y_true, y_pred): return K.mean(metrics.categorical_accuracy(y_true, y_pred))
