import os
import numpy as np
from pprint import pprint
import argparse
import tensorflow as tf

from ..settings import logger
from ..metrics import get_metric_by_name
from ..datasets import get_dataset_by_name, get_cache_dir
from ..datasets.utils import convert2tfdataset, DataType
from ..processing.dataset import get_preprocess_fn, ColorMode
from ..visualizations import show
from ..serving import predict, predict_on_batch, get_models_from_directory, ensemble_inference, retrieve_metadata


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--models_dir', required=True, help='path to dir containing multiple trained models')
    parser.add_argument('-data', '--data_dir', required=True, help='data directory')
    parser.add_argument('-rand', '--randomize', action='store_true', help='randomize the dataset examples order')
    parser.add_argument('-c', '--contains', default=None, help='model name must contain this value')
    parser.add_argument('-d', '--dataset', required=True, help='dataset name')
    parser.add_argument('-dt', '--data_type', default=DataType.VAL, choices=DataType.get())
    parser.add_argument("-rm", '--resize_method', default='resize', help='resize method')
    parser.add_argument("-host", '--host', default='localhost', help='tf model server host')
    parser.add_argument("-p", '--port', default=8501, type=int, help='tf model server port')
    parser.add_argument("-npr", '--num_per_row', default=4, type=int, help='number of images per row')
    parser.add_argument("-t", '--threshold', default=0.5, type=float, help='binary threshold')
    parser.add_argument("-m", '--metric', default='iou_score', type=str, help='metric to evaluate the models with')
    return parser.parse_args()


def compare_models(models, image, mask, num_classes, host='localhost', port=8501, threshold=0.5, metric='iou_score', num_per_row=4):

    metric = get_metric_by_name(metric)

    def get_score(mask, p):
        p = tf.cast(p, tf.int32)
        p = tf.expand_dims(p, axis=0)

        mask = tf.expand_dims(mask, axis=0)
        mask = tf.expand_dims(mask, axis=-1)
        return metric(mask, p)

    ensemble, predictions = ensemble_inference(models, image, host=host, port=port, threshold=threshold)
    model_scores = [get_score(mask, p) for p in predictions]
    model_titles = ["%s (IoU: %.3f)" % (m['name'], model_scores[k]) for k, m in enumerate(models)]

    ensemble_score = get_score(mask, ensemble.astype(np.uint8))

    titles = ['input'] + model_titles + ['ensemble (IoU: %.3f)' % ensemble_score] + ['target']
    images = [image] + predictions + [ensemble] + [mask.astype(np.float32)]
    show.show_images(images, titles=titles, cols=len(titles) // num_per_row)


def main():

    args = get_args()
    models = get_models_from_directory(args.models_dir, contains=args.contains)

    logger.info("=============")
    logger.info("Found models:")
    pprint(models)

    try:
        meta = retrieve_metadata(models[0]['name'])
        pprint(meta)
    except:
        logger.info("Please start the tensorflow model server using `tensorflow_model_server - -model_config_file=models.yaml --rest_api_port=%d" % args.port)
        exit(0)

    input_shape = meta['inputs'][list(meta['inputs'].keys())[0]]['shape']
    output_shape = meta['outputs'][list(meta['outputs'].keys())[0]]['shape']

    logger.info("input shape: %s" % str(input_shape))
    logger.info("output shape: %s" % str(output_shape))

    # infer from retrieved meta data
    size = tuple(input_shape[1:3])
    scale_mask = output_shape[-1] == 1
    color_mode = 0 if input_shape[-1] == 3 else 1
    num_classes = 2 if output_shape[-1] == 1 else output_shape[-1]

    cache_dir = get_cache_dir(args.data_dir, args.dataset)
    ds = get_dataset_by_name(args.dataset, cache_dir)

    ds = convert2tfdataset(ds, args.data_type, randomize=args.randomize)
    ds = ds.map(get_preprocess_fn(size, color_mode, args.resize_method, scale_mask=scale_mask))

    for image, mask in ds:
        compare_models(models, image.numpy(), mask.numpy(), num_classes, host=args.host, port=args.port, threshold=args.threshold,
                       metric=args.metric, num_per_row=args.num_per_row)


if __name__ == "__main__":
    main()
