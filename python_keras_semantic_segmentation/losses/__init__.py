from .iou import iou_loss
from .psnr import psnr
from .depth_smoothness import depth_smoothness
from .focal import focal_loss, smooth_l1
from .sharpen import total_variation
from .ssim import ssim, ssim2
from .ce import ce_label_smoothing
from .dice import dice_loss

from keras.losses import categorical_crossentropy, binary_crossentropy

losses_by_name = {
    "categorical_crossentropy": categorical_crossentropy,
    "ce_label_smoothing": ce_label_smoothing(smoothing=0.1),
    "binary_crossentropy": binary_crossentropy,
    "focal": focal_loss(),
    "smooth_l1": smooth_l1(),
    "ssim": ssim2(),
    "psnr": psnr(),
    "dice": dice_loss,
    "total_variation": total_variation(),
    "depth_smoothness": depth_smoothness(),
    "iou": iou_loss()
}


def get_loss_by_name(name):
    if name in losses_by_name:
        return losses_by_name[name]
    else:
        raise Exception("cannot find loss %s" % name)


__all__ = ['iou_loss', "psnr", "depth_smoothness", "focal_loss", "smooth_l1", "total_variation", "ssim",
           "ssim2", "categorical_crossentropy", "binary_crossentropy", "get_loss_by_name", "losses_by_name"]
