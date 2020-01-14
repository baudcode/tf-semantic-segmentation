from ..losses import SMOOTH, gather_channels, round_if_needed, get_reduce_axes, average, expand_binary
from tensorflow.keras import backend as K

""" Taken from https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/metrics.py """


def _f_score(gt, pr, beta=1, class_weights=1, class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:

    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor (B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        threshold: value to round predictions (use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        F-score in range [0, 1]

    """
    if gt.shape[-1] == 1:
        # assuming binary
        gt, pr = expand_binary(gt), expand_binary(pr)

    gt, pr = gather_channels(gt, pr, indexes=class_indexes)
    pr = round_if_needed(pr, threshold)
    axes = get_reduce_axes(per_image)

    # calculate score
    tp = K.cast(K.sum(gt * pr, axis=axes), "float64")
    fp = K.cast(K.sum(pr, axis=axes), 'float64') - tp
    fn = K.cast(K.sum(gt, axis=axes), 'float64') - tp

    score = ((1 + beta ** 2) * tp + smooth) \
        / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = average(score, per_image, class_weights)

    return score


def f1_score(class_weights=1):
    def f1_score(gt, pr):
        return _f_score(gt, pr, beta=1, class_weights=class_weights)
    return f1_score


def f2_score(class_weights=1):
    def f2_score(gt, pr):
        return _f_score(gt, pr, beta=2, class_weights=class_weights)
    return f2_score
