from tensorflow.keras import backend as K


def iou_loss(smooth=1):
    def iou_loss_fixed(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou
    return iou_loss_fixed
