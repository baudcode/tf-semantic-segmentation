import argparse
import numpy as np

from tf_semantic_segmentation.processing.dataset import get_preprocess_fn_v2
from ..datasets import DirectoryDataset, DataType
from ..visualizations import show

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True)
    parser.add_argument('-rm', '--resize_method', default='resize')
    parser.add_argument('-s', '--size', default=(256, 256), type=lambda x: list(map(int, x.split(","))))
    args = parser.parse_args()

    ds = DirectoryDataset(args.directory)
    ds.summary()

    tfds = ds.tfdataset(DataType.TRAIN, randomize=True)
    fn = get_preprocess_fn_v2(args.size, ds.num_classes, args.resize_method, True)
    tfds = tfds.map(fn)

    for i, (image, mask) in enumerate(tfds):
        print(image.numpy().max(), image.numpy().min(), image.numpy().dtype)
        mask = (mask.numpy() * 255. / (len(ds.labels) - 1)).astype(np.uint8)
        image = (image.numpy() * 255).astype(np.uint8)
        show.show_images([image, mask])
        print(image.shape, mask.shape)
