from tensorflow.keras.models import load_model
from .. import optimizers
import argparse
import imageio
import numpy as np
import cv2
import os

from ..visualizations import show, masks
from ..datasets.tfrecord import TFReader
from ..datasets import DataType
from ..processing import dataset as pre_dataset
from ..settings import logger
from .video import predict_video

"""
python -m tf_semantic_segmentation.eval.predict -i "/hdd/datasets/sun/SUNRGBD/kv2/kinect2data/000010_2014-05-26_14-32-36_260595134347_rgbf000020-resize/image/0000020.jpg" -m logs/model-best.h5
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', required=True, help='path to the serialized model')
    parser.add_argument('-i', '--image', help='input file', default=None)
    parser.add_argument('-t', '--target', help='target file', default=None)
    parser.add_argument('-v', '--video', help='video file', default=None)
    parser.add_argument('-r', '--record_dir', help='record dir', default=None)
    parser.add_argument('-o', '--output_dir', help='export predictions', default=None)
    parser.add_argument('-ns', '--no_stream', help='show predictions as continous stream of images using opencv', action='store_true')
    parser.add_argument('-rdt', '--record_data_type', help='data type (train, test, val)', default=DataType.VAL,
                        choices=[DataType.TRAIN, DataType.TEST, DataType.VAL])
    parser.add_argument('-rm', '--resize_method', help='default: resize', type=str, default='resize')
    parser.add_argument('-sm', '--scale_mask', help='scale mask between 0 and 1', action='store_true')
    return parser.parse_args()


def main():
    try:
        # for exporting in exr format
        imageio.plugins.freeimage.download()
    except Exception:
        pass

    args = get_args()
    if args.record_dir is None and args.image is None and args.video is None:
        raise AssertionError("Please either specify an `image`, `video` or a `record_dir`")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info("loading model")
    model = load_model(args.model_path, compile=False)
    logger.info("model loaded, expects input shape of %s" % str(model.input.shape))

    size = tuple(model.input.shape[1:3])
    depth = model.input.shape[-1]
    color_mode = pre_dataset.ColorMode.GRAY if depth == 1 else pre_dataset.ColorMode.RGB

    if args.record_dir:
        dataset = TFReader(args.record_dir).get_dataset(args.record_data_type)
        dataset = dataset.map(pre_dataset.get_preprocess_fn(size, color_mode, args.resize_method,
                                                            scale_mask=args.scale_mask))

        for k, (image, target) in enumerate(dataset):
            image_batch = np.expand_dims(image.numpy(), axis=0)

            p = model.predict_on_batch(image_batch)
            num_classes = p.shape[-1] if p.shape[-1] > 1 else 2
            logger.debug("model has %d classes" % num_classes)

            target_rgb = masks.get_colored_segmentation_mask(np.expand_dims(target, axis=0),
                                                             num_classes,
                                                             images=image_batch.copy(),
                                                             binary_threshold=0.5)[0]

            predictions_rgb = masks.get_colored_segmentation_mask(p,
                                                                  num_classes,
                                                                  images=image_batch.copy(),
                                                                  binary_threshold=0.5)[0]
            if args.no_stream:
                show.show_images([image_batch[0], predictions_rgb, target_rgb], titles=['input', 'predictions', 'target'])
            else:
                result = np.concatenate([(image_batch[0] * 255.).astype(np.uint8), predictions_rgb, target_rgb], axis=1)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.output_dir:
                imageio.imwrite(os.path.join(args.output_dir, '%d-input.png' % k), (image_batch[0] * 255.).astype(np.uint8))
                imageio.imwrite(os.path.join(args.output_dir, '%d-prediction.exr' % k), p[0])
                imageio.imwrite(os.path.join(args.output_dir, '%d-prediction-rgb.png' % k), predictions_rgb)

    elif args.image:
        image = imageio.imread(args.image)
        image = image.astype(np.float32) / 255.
        # prepare image
        image, _ = pre_dataset.resize_and_change_color(image, None, size, color_mode, resize_method=args.resize_method)

        print(image.shape)
        # prepare image
        image_batch = np.expand_dims(image, axis=0)

        p = model.predict_on_batch(image_batch)
        num_classes = p.shape[-1] if p.shape[-1] > 1 else 2
        logger.info("model has %d classes" % num_classes)

        predictions_rgb = masks.get_colored_segmentation_mask(p,
                                                              num_classes,
                                                              images=image_batch,
                                                              binary_threshold=0.5)[0]

        show.show_images([predictions_rgb], titles=['predictions on input'])

        if args.output_dir:
            imageio.imwrite(os.path.join(args.output_dir, '%d-input.png' % k), (image_batch[0] * 255.).astype(np.uint8))
            imageio.imwrite(os.path.join(args.output_dir, '%d-prediction.exr' % k), p[0])
            imageio.imwrite(os.path.join(args.output_dir, '%d-prediction-rgb.png' % k), predictions_rgb)
    elif args.video:
        output_path = None if not args.output_dir else os.path.join(args.output_dir, "p-%s" % os.path.basename(args.video))
        logger.info("saving predictions to %s" % output_path)
        predict_video(model, args.video, not args.no_stream, output_path, args.resize_method)


if __name__ == "__main__":
    main()
