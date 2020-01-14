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
    for name, metric in reversed(list(metrics.metrics_by_name.items())):
        assert(metrics.get_metric_by_name(name) == metric)
        print("metrics: %s" % name)
        if name == 'mae':
            assert(metric(TEST_BATCH, TEST_BATCH).numpy() == 0.0)
        elif name == "psnr":
            assert(metric(TEST_BATCH, TEST_BATCH).numpy() > 100)
        else:
            assert(metric(TEST_BATCH, TEST_BATCH).numpy() == 1.0)
