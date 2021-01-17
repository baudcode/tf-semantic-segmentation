from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from ..bin.train import train_test_model, get_args
from ..models import models_by_name
from ..utils import logger
from ..datasets import get_dataset_by_name, datasets_by_name, DataType, get_cache_dir
from ..datasets import TFWriter
from tensorflow.keras import backend as K
import os
import json
import time


def _create_dataset(name, overwrite=False, data_dir='/tmp/data'):

    cache_dir = get_cache_dir(data_dir, name.lower())
    if os.path.exists(cache_dir) and not overwrite:
        return cache_dir

    ds = get_dataset_by_name(name, cache_dir)

    # print labels and classes
    print(ds.labels)
    print(ds.num_classes)

    # print number of training examples
    print(ds.num_examples(DataType.TRAIN))

    # or simply print the summary
    ds.summary()

    writer = TFWriter(cache_dir)
    writer.write(ds)
    writer.validate(ds)
    return cache_dir


def train_all(dataset, project='all_models', loss="dice", batch_size=4, epochs=1, steps_per_epoch=1, validation_steps=1, size=[512, 512], overwrite=False, data_dir='/tmp/data'):

    logger.info("cerating dataset")
    record_dir = _create_dataset(dataset, overwrite=overwrite, data_dir=data_dir)

    results_dir = "/tmp/%s/results/" % project
    os.makedirs(results_dir, exist_ok=True)

    for model_name in models_by_name.keys():
        # get the default args
        args = get_args({})

        # change some parameters
        # !rm -r logs/
        args.model = model_name
        args.batch_size = batch_size
        args.size = size  # resize input dataset to this size
        args.epochs = epochs
        args.steps_per_epoch = steps_per_epoch
        args.validation_steps = validation_steps
        args.learning_rate = 1e-4
        args.optimizer = 'adam'  # ['adam', 'radam', 'ranger']
        args.loss = loss
        args.logdir = '/tmp/%s/logs/%s' % (project, model_name)
        args.record_dir = record_dir
        args.final_activation = 'softmax'
        args.wandb_name = "%s-%s-%d" % (dataset, model_name, time.time())
        args.wandb_project = project
        # train and test
        results, model = train_test_model(args)
        json.dump(results, open(os.path.join(results_dir, "%s.json" % model_name), 'w'))

        K.clear_session()


def main():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--dataset', required=True, help='dataset name')
    parser.add_argument("-p", '--project', default='all_models', help="wandb project name")
    parser.add_argument("-steps", '--steps_per_epoch', default=-1, type=int, help='number of steps')
    parser.add_argument("-e", '--epochs', default=1, type=int, help='number of epochs')
    parser.add_argument("-val_steps", '--validation_steps', default=-1, type=int, help='number of val steos')
    parser.add_argument("-size", '--size', default=[512, 512], type=lambda x: list(map(float, x.split(","))), help='input size')
    parser.add_argument("-dd", '--data_dir', default="/tmp/data", help='dataset data dir')
    parser.add_argument("-l", '--loss', default="dice", help='loss function')
    parser.add_argument("-bs", '--batch_size', default=4, type=int, help='batch_size')

    args = parser.parse_args()

    train_all(args.dataset, args.project, args.loss, args.batch_size, args.epochs, args.steps_per_epoch, args.validation_steps, args.size)


if __name__ == "__main__":
    main()
