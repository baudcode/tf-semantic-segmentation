from tensorflow.keras.losses import CategoricalCrossentropy


def ce_label_smoothing(smoothing=0.1):
    def ce_label_smoothing_fixed(y_true, y_pred):
        return CategoricalCrossentropy(label_smoothing=smoothing)(y_true, y_pred)
    return ce_label_smoothing_fixed
