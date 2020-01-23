from ..losses import gather_channels, round_if_needed, get_reduce_axes, SMOOTH, average, expand_binary
from tensorflow.keras import backend as K


""" Taken from https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/metrics.py """


def precision(class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None, **kwargs):
    r"""Calculate precision between the ground truth (gt) and the prediction (pr).

    .. math:: F_\beta(tp, fp) = \frac{tp} {(tp + fp)}

    where:
         - tp - true positives;
         - fp - false positives;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or ``np.array`` of class weights (``len(weights) = num_classes``)
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: Float value to avoid division by zero.
        per_image: If ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch.
        threshold: Float value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round.
        name: Optional string, if ``None`` default ``precision`` name is used.

    Returns:
        float: precision score
    """
    def precision(gt, pr):
        if gt.shape[-1] == 1:
            # assuming binary
            gt, pr = expand_binary(gt), expand_binary(pr)

        gt, pr = gather_channels(gt, pr, indexes=class_indexes, **kwargs)
        pr = round_if_needed(pr, threshold, **kwargs)
        axes = get_reduce_axes(per_image, **kwargs)

        # score calculation
        tp = K.cast(K.sum(gt * pr, axis=axes), 'float64')
        fp = K.cast(K.sum(pr, axis=axes), 'float64') - tp

        score = (tp + smooth) / (tp + fp + smooth)
        score = average(score, per_image, class_weights, **kwargs)

        return score
    return precision
