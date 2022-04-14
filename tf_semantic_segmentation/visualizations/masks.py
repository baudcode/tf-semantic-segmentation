import numpy as np
from PIL import ImageColor
import colorsys
import random
import cv2


def get_colors(N, shuffle=False, bright=True):
    """
    https://github.com/pedropro/TACO/blob/master/detector/visualize.py

    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    colors = [[0, 0, 0]]
    if N == 1:
        return colors

    brightness = 1.0 if bright else 0.7
    hsv = [(i / (N - 1), 1, brightness) for i in range(N - 1)]
    colors.extend(list(map(lambda c: list(map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*c))), hsv)))

    if shuffle:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def apply_mask_v2(image, mask, color, alpha=0.5):
    from PIL import Image

    base_layer = Image.fromarray(image)

    # create color and alpha layer
    color_layer = Image.new('RGBA', base_layer.size, tuple(color))
    mask = mask.astype(np.uint8) * int(255 * alpha)
    alpha_mask = Image.fromarray(mask, mode='L')

    # composite
    base_layer = Image.composite(color_layer, base_layer, alpha_mask)
    return np.asarray(base_layer)


def overlay_classes(image, target, colors, num_classes, alpha=0.5):
    assert(len(colors) == (num_classes - 1))
    for k in range(1, num_classes):
        mask = np.where(target == k, 1, 0)
        image = apply_mask_v2(image, mask, colors[k - 1], alpha=alpha)
    return image


def get_colored_segmentation_mask(predictions, num_classes, images=None, binary_threshold=0.5, alpha=0.5):
    """
    Arguments:

    predictions: ndarray - BHWC (if C=1 - np.float32 else np.uint8) probabilities
    num_classes: int - number of classes
    images: ndarray - float32 (0..1) or uint8 (0..255)
    binary_threshold: float - when predicting only 1 value, threshold to set label to 1
    alpha: float - overlay percentage
    """

    predictions = predictions.copy()
    colors = get_colors(num_classes)[1:]

    if images is None:
        if isinstance(predictions, list):
            shape = (predictions[-1].shape[0], predictions[-1].shape[1], predictions[-1].shape[2], 3)
        else:
            shape = (predictions.shape[0], predictions.shape[1], predictions.shape[2], 3)
        images = np.zeros(shape, np.uint8)
    else:
        images = images.copy()

        if images.dtype == "float32" or images.dtype == 'float64':
            images = (images * 255).astype(np.uint8)

        if images.shape[-1] == 1:
            images = [cv2.cvtColor(i.copy(), cv2.COLOR_GRAY2RGB) for i in images]
            images = np.asarray(images)

    if not isinstance(predictions, list):
        predictions = [predictions]

    outputs = {}

    # for every resolution
    for p in predictions:

        overlays = []

        if p.shape[-1] == 1:
            # remove channel dimension
            p = np.squeeze(p, axis=-1)

        if len(p.shape) == 3:
            # set either zero or one
            p[p > binary_threshold] = 1.0
            p[p <= binary_threshold] = 0.0
        else:
            # find the argmax channel from all channels
            p = np.argmax(p, axis=-1)

        p = p.astype(np.uint8)

        for i in range(len(p)):
            size = p[i].shape[:2][::-1]
            image_resized = cv2.resize(images[i, :, :, :].copy(), tuple(size), interpolation=cv2.INTER_CUBIC)

            o = overlay_classes(image_resized, p[i], colors, num_classes, alpha=alpha)
            overlays.append(o)

        shape = p.shape[:2][::-1]
        shape = "x".join(map(str, shape))
        outputs[shape] = overlays

    return outputs


def get_rgb(mask):
    if mask.dtype == np.float32:
        mask = (mask * 255).astype(np.uint8)
    else:
        if mask.max() < 50:
            mask = mask * 255

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    if mask.shape[-1] == 1:
        rgb = np.concatenate([mask, mask, mask], axis=-1)
    else:
        rgb = mask.copy()

    return rgb
