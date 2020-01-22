from tensorflow.keras.models import load_model
from .. import optimizers
import argparse
import imageio
import numpy as np
import cv2
from ..visualizations import show
from ..datasets.tfrecord import TFReader
from ..datasets import DataType
from ..processing import dataset as pre_dataset

"""
python -m tf_semantic_segmentation.eval.predict -i "/hdd/datasets/sun/SUNRGBD/kv2/kinect2data/000010_2014-05-26_14-32-36_260595134347_rgbf000020-resize/image/0000020.jpg" -m logs/model-best.h5
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True, help='path to the serialized model')
    parser.add_argument('-i', '--input', help='input file', default=None)
    parser.add_argument('-t', '--target', help='target file', default=None)
    parser.add_argument('-r', '--record_dir', help='record dir', default=None)
    parser.add_argument('-c', '--color_mode', help='color mode (0=RGB, 1=GRAY, 2=NONE), default: NONE', type=int, default=2)
    parser.add_argument('-rm', '--resize_method', help='default: resize', type=str, default='resize')
    parser.add_argument('-sm', '--scale_mask', help='scale mask between 0 and 1', action='store_true')
    return parser.parse_args()


def main():

    args = get_args()

    model = load_model(args.model_path)
    size = model.input.shape[1:3][::-1]
    size = (size[0], size[1])
    depth = model.input.shape[-1]
    # size = (128, 128)

    if args.record_dir:
        dataset = TFReader(args.record_dir).get_dataset(DataType.VAL)
        dataset = dataset.map(pre_dataset.get_preprocess_fn(size, args.color_mode, args.resize_method,
                                                            scale_mask=args.scale_mask, is_training=False))

        for image, target in dataset:
            target = np.argmax(target, axis=-1)

            show.show_images([image, target.astype(np.float32)])
            print(image.shape, target.shape)
            print(image.numpy().shape)
            image = np.expand_dims(image.numpy(), axis=0)
            p = model.predict_on_batch(image)
            p = p[0].numpy()
            p = np.argmax(p, axis=-1)
            show.show_images([p.astype(np.float32), image[0], target.astype(np.float32)])

    if args.image:
        image = imageio.imread(args.input)

        # prepare image

        if depth == 3:
            print('rgb input')
        else:
            print("gray input")

        image = cv2.resize(image, (size[0], size[1]))
        image = image.astype(np.float32) / 255.

        # make batch dimension
        batch = np.expand_dims(image, axis=0)

        p = model.predict_on_batch(batch)

        # get numpy vector
        p = p[0].numpy()  # h x w x num_classes
        p = np.argmax(p, axis=-1)  # h x w

        show.show_images([image, p.astype(np.float32)])


if __name__ == "__main__":
    main()
