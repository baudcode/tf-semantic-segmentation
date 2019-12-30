from tensorflow.keras import backend as K


def _gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]


def depth_smoothness():
    """Computes image-aware depth smoothness loss."""
    def depth_smoothness_fixed(y_true, y_pred):
        depth_dx = _gradient_x(y_pred)
        depth_dy = _gradient_y(y_pred)
        image_dx = _gradient_x(y_true)
        image_dy = _gradient_y(y_true)
        weights_x = K.exp(-K.mean(K.abs(image_dx), 3, keepdims=True))
        weights_y = K.exp(-K.mean(K.abs(image_dy), 3, keepdims=True))
        smoothness_x = depth_dx * weights_x
        smoothness_y = depth_dy * weights_y

        return K.mean(K.abs(smoothness_x)) + K.mean(K.abs(smoothness_y))

    return depth_smoothness_fixed
