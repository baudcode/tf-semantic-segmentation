from .ssim import ssim_loss
from .ce import categorical_crossentropy_loss, binary_crossentropy_loss
from .dice import dice_loss
import tensorflow as tf


def categorical_crossentropy_ssim_loss(loss_weight_ce=1.0, loss_weight_ssim=1.0):
    def categorical_crossentropy_ssim(y_true, y_pred):
        ce = categorical_crossentropy_loss()(y_true, y_pred)
        ssim = ssim_loss()(y_true, y_pred)
        return ce * loss_weight_ce + loss_weight_ssim * ssim

    return categorical_crossentropy_ssim


def binary_crossentropy_ssim_loss(loss_weight_ce=1.0, loss_weight_ssim=1.0):
    def binary_crossentropy_ssim(y_true, y_pred):
        ce = binary_crossentropy_loss()(y_true, y_pred)
        ssim = ssim_loss()(y_true, y_pred)
        return ce * loss_weight_ce + loss_weight_ssim * ssim

    return binary_crossentropy_ssim


def dice_ssim_loss(loss_weight_dice=1.0, loss_weight_ssim=1.0):
    def dice_ssim(y_true, y_pred):
        dice = dice_loss()(y_true, y_pred)
        ssim = ssim_loss()(y_true, y_pred)
        return dice * loss_weight_dice + loss_weight_ssim * ssim

    return dice_ssim


def dice_binary_crossentropy_loss(loss_weight_dice=1.0, loss_weight_ce=1.0):
    def dice_binary_crossentropy(y_true, y_pred):
        dice = dice_loss()(y_true, y_pred)
        ce = binary_crossentropy_loss()(y_true, y_pred)
        return dice * loss_weight_dice + ce * loss_weight_ce

    return dice_binary_crossentropy


def dice_categorical_crossentropy_loss(loss_weight_dice=1.0, loss_weight_ce=1.0):
    def dice_categorical_crossentropy(y_true, y_pred):
        dice = dice_loss()(y_true, y_pred)
        ce = categorical_crossentropy_loss()(y_true, y_pred)
        return dice * loss_weight_dice + ce * loss_weight_ce

    return dice_categorical_crossentropy


def dice_ssim_binary_crossentropy_loss(loss_weight_dice=1.0, loss_weight_ce=1.0, loss_weight_ssim=1.0):
    def dice_ssim_binary_crossentropy(y_true, y_pred):
        dice = dice_loss()(y_true, y_pred)
        dice = tf.cast(dice, tf.float32)

        ce = binary_crossentropy_loss()(y_true, y_pred)
        ce = tf.cast(ce, tf.float32)
        ssim = ssim_loss()(y_true, y_pred)
        return dice * loss_weight_dice + ce * loss_weight_ce + loss_weight_ssim * ssim

    return dice_ssim_binary_crossentropy


def dice_ssim_categorical_crossentropy_loss(loss_weight_dice=1.0, loss_weight_ce=1.0, loss_weight_ssim=1.0):
    def dice_ssim_categorical_crossentropy(y_true, y_pred):
        dice = dice_loss()(y_true, y_pred)
        ce = categorical_crossentropy_loss()(y_true, y_pred)
        ssim = ssim_loss()(y_true, y_pred)
        tf.print(dice.dtype, ce.dtype, ssim.dtype)
        return dice * loss_weight_dice + ce * loss_weight_ce + loss_weight_ssim * ssim

    return dice_ssim_categorical_crossentropy
