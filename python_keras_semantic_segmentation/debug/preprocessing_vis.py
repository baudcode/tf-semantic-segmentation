from ..datasets import get_dataset_by_name
from ..processing import pre
from ..visualizations.show import show_images
from ..visualizations.masks import draw_segmentation_masks, draw_segmentation_masks_2d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ds = 'ade20k'
    ds = get_dataset_by_name(ds, '/hdd/datasets/%s' % ds)
    
    for inputs, targets in ds.get()():
        show_images([inputs, targets.astype(np.float32)])
