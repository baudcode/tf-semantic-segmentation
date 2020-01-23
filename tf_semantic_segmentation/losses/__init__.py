from tensorflow.keras import backend as K
from .utils import gather_channels, get_reduce_axes, round_if_needed, SMOOTH, average, onehot2image, expand_binary
from .focal import binary_focal_loss, categorical_focal_loss
from .ssim import ssim_loss
from .ce import ce_label_smoothing_loss, categorical_crossentropy_loss, binary_crossentropy_loss
from .dice import dice_loss, tversky_loss, focal_tversky_loss
from .combined import categorical_crossentropy_ssim_loss, binary_crossentropy_ssim_loss, \
    dice_binary_crossentropy_loss, dice_categorical_crossentropy_loss, dice_ssim_loss, \
    dice_ssim_binary_crossentropy_loss, dice_ssim_categorical_crossentropy_loss


losses_by_name = {
    "categorical_crossentropy": categorical_crossentropy_loss(),
    "ce_label_smoothing": ce_label_smoothing_loss(smoothing=0.1),
    "binary_crossentropy": binary_crossentropy_loss(),
    "categorical_focal": categorical_focal_loss(),
    "binary_focal": binary_focal_loss(),
    "ssim": ssim_loss(),
    "dice": dice_loss(),
    "tversky": tversky_loss(),
    "focal_tversky": focal_tversky_loss(),
    # combined losses
    "binary_crossentropy_ssim": binary_crossentropy_ssim_loss(),
    "categorical_crossentropy_ssim": categorical_crossentropy_ssim_loss(),
    "dice_binary_crossentropy": dice_binary_crossentropy_loss(),
    "dice_categorical_crossentropy": dice_categorical_crossentropy_loss(),
    "dice_ssim": dice_ssim_loss(),
    "dice_ssim_binary_crossentropy": dice_ssim_binary_crossentropy_loss(),
    "dice_ssim_categorical_crossentropy": dice_ssim_categorical_crossentropy_loss()
}


def get_loss_by_name(name):
    if name in losses_by_name:
        return losses_by_name[name]
    else:
        raise Exception("cannot find loss %s" % name)


__all__ = ["categorical_focal_loss", "binary_crossentropy_loss", "binary_focal_loss",
           "focal_tversky_loss", "tversky_loss", "dice_loss", "ssim_loss",
           "binary_crossentropy_ssim_loss", "categorical_crossentropy_ssim_loss", "dice_binary_crossentropy_loss",
           "dice_categorical_crossentropy_loss", "dice_ssim_loss", "dice_ssim_binary_crossentropy_loss", "dice_ssim_categorical_crossentropy_loss",
           "get_loss_by_name", "losses_by_name",
           "SMOOTH", "gather_channels", "get_reduce_axes", "round_if_needed", "average", "expand_binary"]
