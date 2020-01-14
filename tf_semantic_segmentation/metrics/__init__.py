from .f_scores import f1_score, f2_score
from .iou_score import iou_score
from .precision import precision
from .recall import recall
from .psnr import psnr
from .ssim import ssim
from . import kmetrics


metrics_by_name = {
    "f1_score": f1_score(),
    "f2_score": f2_score(),
    "precision": precision(),
    "recall": recall(),
    "iou_score": iou_score(),
    "psnr": psnr(),
    "ssim": ssim(),
    "mae": kmetrics.mae,
    "binary_accuracy": kmetrics.binary_accuracy,
    "categorical_accuracy": kmetrics.categorical_accuracy,
    "top_5_categorical_accuracy": kmetrics.top_5_categorical_accuracy,
    "top_3_categorical_accuracy": kmetrics.top_3_categorical_accuracy,
    "top_1_categorical_accuracy": kmetrics.top_1_categorical_accuracy
}


def get_metric_by_name(name):
    if name in metrics_by_name:
        return metrics_by_name[name]
    else:
        raise Exception("cannot find metric %s" % name)


__all__ = list(metrics_by_name.keys())
