import os


from ..models import models_by_name
from ..settings import logger

from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import logging
import argparse


def counts(model):
    trainable_count = int(
        np.sum([K.count_params(p) for p in model.trainable_weights]))

    non_trainable_count = int(
        np.sum([K.count_params(p) for p in model.non_trainable_weights]))

    return (trainable_count, non_trainable_count)


if __name__ == "__main__":

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shape', default=[256, 256, 3], help='input shape default: (256, 256, 3)', type=lambda x: list(map(int, x.split(","))))
    parser.add_argument('-nc', '--num_classes', default=10, help='number of classes default: 10')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    num_classes = args.num_classes
    input_shape = tuple(args.shape)

    logger.info("using input shape %s" % str(input_shape))
    logger.info("using %d classes" % num_classes)
    logger.info("==================")

    models = {

    }

    for name, model_fn in list(models_by_name.items()):
        model, _ = model_fn(input_shape=input_shape, num_classes=num_classes)
        models[name] = counts(model)
        logger.info("%s has %d trainable and %d non trainable parameters" % (name, models[name][0], models[name][1]))

        K.clear_session()

    print("=" * 20, "summary", "=" * 20)
    max_length = max([len(name) for name in models.keys()])

    row_format = "{:>" + str(max_length) + "} {:>20} {:>20}"
    print(row_format.format("name", "trainable params", "non trainable params"))

    row_format = "{:>" + str(max_length) + "} {:>20,} {:>20,}"
    for name, (params, non_params) in models.items():
        print(row_format.format(name, params, non_params))
