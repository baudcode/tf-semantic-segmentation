import tensorflow as tf


def dice_loss(y_true, y_pred):
    """ F1 Score """
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)


def tversky_loss(beta):
    """ Tversky index (TI) is a generalization of Dice’s coefficient. TI adds a weight to FP (false positives) and FN (false negatives). """
    def loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis=-1) + 1)

    return loss
