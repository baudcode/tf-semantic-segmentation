from tensorflow.keras import backend as K
from ..losses import SMOOTH, gather_channels, round_if_needed, get_reduce_axes, average, expand_binary

""" Taken from https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/metrics.py """


def iou_score(class_weights=1., class_indexes=None, smooth=SMOOTH, per_image=False, threshold=None):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor(B, H, W, C) or (B, C, H, W)
        pr: prediction 4D keras tensor(B, H, W, C) or (B, C, H, W)
        class_weights: 1. or list of class weights, len(weights) = C
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch(B),
            else over whole batch
        threshold: value to round predictions(use ``>`` comparison), if ``None`` prediction will not be round

    Returns:
        IoU / Jaccard score in range[0, 1]

    .. _`Jaccard index`: https: // en.wikipedia.org / wiki / Jaccard_index

    """
    def iou_score(gt, pr):

        if gt.shape[-1] == 1:
            # assuming binary
            gt, pr = expand_binary(gt), expand_binary(pr)

        gt, pr = gather_channels(gt, pr, indexes=class_indexes)
        pr = round_if_needed(pr, threshold)
        axes = get_reduce_axes(per_image)

        # score calculation
        intersection = K.sum(gt * pr, axis=axes)
        union = K.sum(gt + pr, axis=axes) - intersection

        score = (intersection + smooth) / (union + smooth)
        score = average(score, per_image, class_weights)

        return score

    return iou_score
