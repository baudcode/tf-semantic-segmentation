from ..datasets import get_dataset_by_name, DataType
from ..datasets.utils import convert2tfdataset
from ..processing import dataset as ds_preprocessing
from ..processing import ColorMode
from ..visualizations.show import show_images
import numpy as np

if __name__ == "__main__":
    ds = 'bioimagebenchmark'
    ds = get_dataset_by_name(ds, '/hdd/datasets/%s' % ds)
    tfds = convert2tfdataset(ds, DataType.TRAIN)

    preprocess_fn = ds_preprocessing.get_preprocess_fn((256, 256), ColorMode.RGB, 'patch', scale_mask=True)
    tdds = tfds.map(preprocess_fn)
    for inputs, targets in tdds:
        show_images([inputs, targets])
        break
