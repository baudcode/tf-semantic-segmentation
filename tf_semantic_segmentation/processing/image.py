import numpy as np
import cv2
import imageio


def fixed_resize(image, width=None, height=None, interpolation=cv2.INTER_NEAREST):
    assert(width is not None or height is not None)
    size = image.shape[:2][::-1]
    depth = image.shape[-1] if len(image.shape) == 3 else 0
    if width:
        f = width / size[0]
        height = int(size[1] * f)
    else:
        f = height / size[1]
        width = int(size[0] * f)

    if image.shape[0] != height or image.shape[1] != width:
        image = cv2.resize(image, (width, height), 0, 0, interpolation=interpolation)

    if depth and len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return image


def grayscale_grid_vis(X, nh, nw, save_path=None):
    """ https://github.com/Newmu/dcgan_code/blob/master/lib/vis.py """
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw), np.uint8)
    for n, x in enumerate(X):
        j = int(n / nw)
        i = n % nw
        img[int(j * h):int(j * h + h), int(i * w):int(i * w + w)] = x

    if save_path is not None:
        imageio.imwrite(save_path, img)
    return img
