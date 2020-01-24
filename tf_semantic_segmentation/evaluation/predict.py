from tensorflow.keras.models import load_model
from .. import optimizers
import argparse
import imageio
import numpy as np
import cv2
from ..visualizations import show, masks
from ..datasets.tfrecord import TFReader
from ..datasets import DataType
from ..processing import dataset as pre_dataset
from ..settings import logger


"""
python -m tf_semantic_segmentation.eval.predict -i "/hdd/datasets/sun/SUNRGBD/kv2/kinect2data/000010_2014-05-26_14-32-36_260595134347_rgbf000020-resize/image/0000020.jpg" -m logs/model-best.h5
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True, help='path to the serialized model')
    parser.add_argument('-i', '--image', help='input file', default=None)
    parser.add_argument('-t', '--target', help='target file', default=None)
    parser.add_argument('-r', '--record_dir', help='record dir', default=None)
    parser.add_argument('-rdt', '--record_data_type', help='data type (train, test, val)', default=DataType.VAL,
                        choices=[DataType.TRAIN, DataType.TEST, DataType.VAL])
    parser.add_argument('-rm', '--resize_method', help='default: resize', type=str, default='resize')
    parser.add_argument('-sm', '--scale_mask', help='scale mask between 0 and 1', action='store_true')
    return parser.parse_args()


def main():

    args = get_args()
    if args.record_dir is None and args.image is None:
        raise AssertionError("Please either specify an `image` or a `record_dir`")

    logger.info("loading model")
    model = load_model(args.model_path, compile=False)
    logger.info("model loaded, expects input shape of %s" % str(model.input.shape))

    size = tuple(model.input.shape[1:3][::-1])
    depth = model.input.shape[-1]
    color_mode = pre_dataset.ColorMode.GRAY if depth == 1 else pre_dataset.ColorMode.RGB

    # size = (128, 128)

    if args.record_dir:
        dataset = TFReader(args.record_dir).get_dataset(args.record_data_type)
        dataset = dataset.map(pre_dataset.get_preprocess_fn(size, color_mode, args.resize_method,
                                                            scale_mask=args.scale_mask, is_training=False))

        for image, target in dataset:
            image_batch = np.expand_dims(image.numpy(), axis=0)

            p = model.predict_on_batch(image_batch)
            num_classes = p.shape[-1]
            logger.info("model has %d classes" % num_classes)

            target_rgb = masks.get_colored_segmentation_mask(np.expand_dims(target, axis=0),
                                                             num_classes,
                                                             images=image_batch.copy(),
                                                             binary_threshold=0.5)[0]

            predictions_rgb = masks.get_colored_segmentation_mask(p,
                                                                  num_classes,
                                                                  images=image_batch.copy(),
                                                                  binary_threshold=0.5)[0]
            show.show_images([image_batch[0], predictions_rgb, target_rgb], titles=['input', 'predictions', 'target'])

    if args.image:
        image = imageio.imread(args.image)
        image = image.astype(np.float32) / 255.
        # prepare image
        image, _ = pre_dataset.resize_and_change_color(image, None, size, color_mode, resize_method=args.resize_method)

        print(image.shape)
        # prepare image
        batch = np.expand_dims(image, axis=0)

        p = model.predict_on_batch(batch)
        num_classes = p.shape[-1]
        logger.info("model has %d classes" % num_classes)

        predictions_on_inputs = masks.get_colored_segmentation_mask(p,
                                                                    num_classes,
                                                                    images=batch,
                                                                    binary_threshold=0.5)

        show.show_images([predictions_on_inputs[0]], titles=['predictions on input'])


if __name__ == "__main__":
    main()
