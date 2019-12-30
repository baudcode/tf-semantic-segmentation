
import numpy as np


def iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)


def iou_classwise(gt, pr, n_classes, EPS=1e-12):
    """ https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/metrics.py """
    class_wise = np.zeros(n_classes)
    for cl in range(n_classes):
        intersection = np.sum((gt == cl) * (pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection) / (union + EPS)
        class_wise[cl] = iou
    return class_wise


def tp(y_true, y_pred):
    return np.sum(np.logical_and(y_true, y_pred))


def tn(y_true, y_pred):
    t_true = 1 - y_true
    y_pred = 1 - y_pred
    return np.sum(np.logical_and(y_true, y_pred))


def fp(y_true, y_pred):
    return np.sum(y_pred) - np.sum(np.logical_and(y_true, y_pred))


def fn(y_true, y_pred):
    t_true = 1 - y_true
    y_pred = 1 - y_pred
    return np.sum(y_pred) - np.sum(np.logical_and(y_true, y_pred))


def accuracy(y_true, y_pred):
    _tp = tp(y_true, y_pred)
    _tn = tn(y_true, y_pred)
    _fp = fp(y_true, y_pred)
    _fn = fn(y_true, y_pred)

    return sum([_tp + _tn]) / sum([_tp, _tn, _fp, _fn])
