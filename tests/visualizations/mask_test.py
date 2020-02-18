from tf_semantic_segmentation.visualizations import masks
import numpy as np


def test_get_colors():
    colors = masks.get_colors(10)
    assert(len(colors) == 10 and all([len(c) == 3 for c in colors]))

    shuffled = masks.get_colors(10, shuffle=True)
    assert(len(shuffled) == 10 and all([len(c) == 3 for c in shuffled]))
    assert(colors != shuffled)


def test_get_colored_segmentation():
    num_classes = 5
    predictions = np.ones((1, 32, 32, num_classes), np.float32)
    predictions[:, :, :, 2] = 1.0
    col = masks.get_colored_segmentation_mask(predictions, num_classes)
    assert(col.shape == (1, 32, 32, 3) and col.dtype == 'uint8')

    predictions = np.ones((1, 32, 32, 1), np.float32)
    predictions = predictions * 0.6
    col = masks.get_colored_segmentation_mask(predictions, num_classes, binary_threshold=0.5)
    assert(col.shape == (1, 32, 32, 3) and col.dtype == 'uint8')

    seg_test = np.zeros((1, 32, 32, 3), np.uint8)
    seg_test[:, :, :, 0] = 127
    np.testing.assert_array_equal(seg_test, col)

    predictions = np.ones((1, 32, 32, 1), np.float32)
    predictions = predictions * 0.4
    col = masks.get_colored_segmentation_mask(predictions, num_classes, binary_threshold=0.5)
    assert(col.shape == (1, 32, 32, 3) and col.dtype == 'uint8')

    seg_test = np.zeros((1, 32, 32, 3), np.uint8)
    np.testing.assert_array_equal(seg_test, col)


def test_overlay_classes():
    image = np.zeros((32, 32, 3), np.uint8)
    num_classes = 5
    colors = masks.get_colors(num_classes)
    mask = np.ones((32, 32), np.uint8) * 4
    overlay = masks.overlay_classes(image.copy(), mask, colors, num_classes, alpha=1.0)
    assert(all(np.mean(overlay.mean(axis=0), axis=0) == list(map(float, colors[4]))))
