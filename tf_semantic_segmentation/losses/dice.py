import tensorflow as tf


def dice_loss():
    def dice_loss(y_true, y_pred):
        """ F1 Score """
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        r = 1 - (numerator + 1) / (denominator + 1)
        return tf.cast(r, tf.float32)

    return dice_loss


def tversky_loss(beta=0.7):
    """ Tversky index (TI) is a generalization of Dice’s coefficient. TI adds a weight to FP (false positives) and FN (false negatives). """
    def tversky_loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)

        r = 1 - (numerator + 1) / (tf.reduce_sum(denominator) + 1)
        return tf.cast(r, tf.float32)

    return tversky_loss


def focal_tversky_loss(beta=0.7, gamma=0.75):
    def focal_tversky(y_true, y_pred):
        loss = tversky_loss(beta)(y_true, y_pred)
        return tf.pow(loss, gamma)

    return focal_tversky
