import matplotlib.pyplot as plt

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
from ..serving import predict, predict_on_batch, get_models_from_directory, ensemble_prediction, retrieve_metadata, threshold_predictions


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
    parser.add_argument("-bs", '--batch_size', default=1, type=int, help='batch size for every entry in the curve')
    parser.add_argument("-t", '--threshold', default=0.5, type=float, help='default binary threshold')
    parser.add_argument("-mode", '--mode', default='binary', type=str, help='mode, either binary or per class')
    parser.add_argument("-m", '--metric', default='iou_score', type=str, help='metric to evaluate the models with')
    return parser.parse_args()


def get_score_by_class():
    pass


def compare_models(models, image, mask, num_classes, host='localhost', thresholds=[0.1 * i for i in range(1, 10)], port=8501, metric='iou_score', default_binary_threshold=0.5):

    metric = get_metric_by_name(metric)

    def get_score(mask, p):
        p = tf.cast(p, tf.int32)
        p = tf.expand_dims(p, axis=0)

        mask = tf.expand_dims(mask, axis=0)
        mask = tf.expand_dims(mask, axis=-1)
        return metric(mask, p)

    # returns propabilities
    ensemble, predictions = ensemble_prediction(models, image, host=host, port=port)

    scores = {"ensemble": {}}
    for model in models:
        scores[model['name']] = {}

    if ensemble.shape[-1] == 1:
        for threshold in thresholds:
            en_thresh = threshold_predictions(ensemble.copy(), threshold=threshold)
            pr_thresh = [threshold_predictions(p.copy(), threshold=threshold) for p in predictions]

            for k, model in enumerate(models):
                scores[model['name']][threshold] = get_score(mask, pr_thresh[k])

            scores['ensemble'][threshold] = get_score(mask, en_thresh.astype(np.uint8))
    else:
        ensemble = np.expand_dims(np.argmax(ensemble, axis=-1), axis=-1)

    return scores


def main():
    """ Multiple models -> 

    What model is best for what class? - meanIOU per class
    What threshold is best for binary classifcation? - mIOU of threshold 0.1 .. 0.9 for every model 

    """
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
    ds = ds.batch(args.batch_size)

    ious = {"ensemble": {}}
    ious.update({model['name']: {} for model in models})

    for image_batch, mask_batch in ds:
        ious_batch = {name: {} for name in ious.keys()}
        for image, mask in zip(image_batch, mask_batch):
            print("comparing model...")
            scores = compare_models(models, image.numpy(), mask.numpy(), num_classes, host=args.host, port=args.port, default_binary_threshold=args.threshold,
                                    metric=args.metric)

            for name in scores.keys():
                for it, value in scores[name].items():
                    if it not in ious_batch[name]:
                        ious_batch[name][it] = []

                    ious_batch[name][it].append(value.numpy())
        print(ious_batch)
        for name in ious_batch.keys():
            for it, values in ious_batch[name].items():
                ious[name][it] = np.mean(values)

        break

    for name, it in ious.items():
        print("name: %s" % name)
        for iname, ivalue in it.items():
            print("-> %s %.2f" % (iname, ivalue))


if __name__ == "__main__":
    main()
