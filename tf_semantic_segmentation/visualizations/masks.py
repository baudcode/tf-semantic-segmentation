import numpy as np
from PIL import ImageColor
import colorsys
import random


def random_colors(N, shuffle=False, bright=True):
    """
    https://github.com/pedropro/TACO/blob/master/detector/visualize.py

    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / (N - 1), 1, brightness) for i in range(N - 1)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors.insert(0, [0, 0, 0])
    if shuffle:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def overlay_classes(image, target, colors, num_classes, alpha=0.5):
    assert(len(colors) == num_classes)
    for k, color in enumerate(colors):
        mask = np.where(target == k, 1, 0)
        image = apply_mask(image, mask, color, alpha=alpha)
    return image


def draw_segmentation_masks(image, masks, color=(255, 255, 255)):
    if type(color) == str:
        color = ImageColor.getrgb(color)

    for mask in masks:
        mask = np.asarray(mask, np.uint8) * 255
        mask = np.expand_dims(mask, axis=2)
        mask_3d = np.concatenate(
            [mask, mask, mask], axis=2) * color / max(list(color))
        mask_3d = np.asarray(mask_3d, np.uint8)
        inverted = mask == 0
        mask_3d_inverted = np.concatenate(
            [inverted, inverted, inverted], axis=2)
        image = image * mask_3d_inverted
        image = image | mask_3d

    return image


def draw_segmentation_masks_2d(image, masks, color=1):
    image = image.astype(np.uint8)
    if type(color) == str:
        color = ImageColor.getcolor(color)[0]
    assert(len(image.shape) == 2)

    for mask in masks:
        mask = np.asarray(mask, np.uint8) * color
        inverted = mask == 0
        image = image * inverted
        image = image | mask
    return image
