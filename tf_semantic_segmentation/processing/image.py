import numpy as np
import cv2


def fixed_resize(image, width=None, height=None):
    assert(width is not None or height is not None)
    size = image.shape[:2][::-1]
    if width:
        f = width / size[0]
        height = int(size[1] * f)
    else:
        f = height / size[0]
        width = int(size[0] * f)

    return cv2.resize(image, (width, height), 0, 0, interpolation=cv2.INTER_NEAREST)


def grayscale_grid_vis(X, nh, nw, save_path=None):
    """ https://github.com/Newmu/dcgan_code/blob/master/lib/vis.py """
    h, w = X[0].shape[:2]
    print(h, w)
    img = np.zeros((h * nh, w * nw), np.uint8)
    print(img.shape)
    print('--------')
    for n, x in enumerate(X):
        j = int(n / nw)
        i = n % nw
        print(int(j * h), int(j * h + h))
        print(int(i * w), int(i * w + w))
        img[int(j * h):int(j * h + h), int(i * w):int(i * w + w)] = x
    if save_path is not None:
        cv2.imwrite(save_path, img)
    return img
