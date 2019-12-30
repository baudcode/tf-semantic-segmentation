import tensorflow as tf

def total_variation():
    def total_variation_fixed(t_true, y_pred):
        return tf.reduce_mean(tf.image.total_variation(y_pred))
    return total_variation_fixed