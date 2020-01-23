from ..datasets import datasets_by_name, get_dataset_by_name, DataType, get_cache_dir
from ..visualizations import masks, show
from ..processing.image import fixed_resize
import os
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=list(datasets_by_name.keys()), required=True)
    parser.add_argument('-data_dir', '--data_dir', default='/hdd/datasets/')
    parser.add_argument('-t', '--data_type', default=DataType.TRAIN, choices=DataType.get())
    args = parser.parse_args()

    cache_dir = get_cache_dir(args.data_dir, args.dataset.lower())
    ds = get_dataset_by_name(args.dataset, cache_dir)

    # download and cache data
    ds.raw()

    ds.summary()

    colors = masks.get_colors(ds.num_classes)
    print("using colors: %s" % str(colors))

    for image, target in ds.get(DataType.TRAIN)():

        image = fixed_resize(image, width=500)
        labels = np.unique(target)
        print("labels: ", labels)
        target = fixed_resize(target, width=500)

        labels = np.unique(target)
        print("labels resize: ", labels)
        print('============================')
        # print("counts: ", counts)
        # print('target:', target.dtype, target.max(), target.min())
        # target_3d = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        overlay_on_black = masks.overlay_classes(np.ones_like(image) * 255, target, colors, ds.num_classes, alpha=1.0)
        overlay = masks.overlay_classes(image.copy(), target, colors, ds.num_classes)
        # target = masks.apply_mask()
        show.show_images([image, target.astype(np.float32), overlay, overlay_on_black])
