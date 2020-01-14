from tf_semantic_segmentation import metrics
import imageio
import numpy as np
import tensorflow as tf
from .data import TEST_BATCH


def test_metrics_assert_1_0():
    """
    assert(metrics.precision(TEST_BATCH, TEST_BATCH) == 1.0)
    assert(metrics.f2_score(TEST_BATCH, TEST_BATCH) == 1.0)
    assert(metrics.recall(TEST_BATCH, TEST_BATCH) == 1.0)
    assert(metrics.recall(TEST_BATCH, TEST_BATCH) == 1.0)
    assert(metrics.iou_score(TEST_BATCH, TEST_BATCH) == 1.0)
    """
    for name, metric in metrics.metrics_by_name.items():
        print("metrics: %s" % name)
        if name == "psnr":
            assert(metric()(TEST_BATCH, TEST_BATCH) > 100)
        else:
            assert(metric()(TEST_BATCH, TEST_BATCH) == 1.0)
