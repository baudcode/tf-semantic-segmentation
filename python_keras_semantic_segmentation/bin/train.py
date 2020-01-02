from ..models import get_model_by_name, models_by_name
from ..datasets import get_dataset_by_name, DataType, datasets_by_name
from ..datasets.tfrecord import TFWriter, TFReader
from ..losses import get_loss_by_name, losses_by_name
from ..processing import pre as preprocessing
from ..processing import pre_dataset as preprocessing_ds
from ..processing import loader
from ..settings import logger
from ..optimizers import get_optimizer_by_name, names as optimizer_choices
from ..utils import get_now_timestamp


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras import callbacks as kcallbacks

import os
import argparse


def get_args():

    color_modes = [int(cm) for cm in preprocessing.ColorMode]

    def str_list_type(x): return list(map(str, x.split(",")))
    def float_list_type(x): return list(map(float, x.split(",")))
    def tuple_type(x): return tuple(list(map(int, x.split(","))))

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', default='128,128', type=tuple_type)
    parser.add_argument('-m', '--model', default='erfnet', choices=list(models_by_name.keys()))
    parser.add_argument('-cm', '--color_mode', default=0, type=int, choices=color_modes, help='0 (RGB), 1 (GRAY)')
    parser.add_argument('-o', '--optimizer', default='adam', choices=optimizer_choices)
    parser.add_argument('-p', '--project', default=None, help='project name (logdir name), if None current data will be used')
    parser.add_argument('-wandb', '--enable_wandb', action='store_true')
    parser.add_argument('-bs', '--batch_size', default=1, type=int)
    parser.add_argument('-l', '--losses', default=['categorical_crossentropy'], type=str_list_type, help='choices: %s' % str(list(losses_by_name.keys())))
    parser.add_argument('-lm', '--metrics', default=['iou', 'dice', 'psnr'], type=str_list_type, help='choices: %s' % (list(losses_by_name.keys())))
    parser.add_argument('-lw', '--loss_weights', default=[1.0, 1.0, 500.0], type=float_list_type)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-logdir', '--logdir', default='logs/')
    parser.add_argument('-e', '--epochs', default=10, type=int)

    # data
    parser.add_argument('-data_dir', '--data_dir', default='/hdd/datasets')
    parser.add_argument('-rd', '--record_dir', default=None, help='if none, will be auto detected')
    parser.add_argument('-tog', '--train_on_generator', action='store_true')
    parser.add_argument('-ds', '--dataset', default='shapes', choices=list(datasets_by_name.keys()))
    parser.add_argument('-steps', '--steps_per_epoch', default=-1, help='if not set, will be calculated based on the number of examples in the record')
    parser.add_argument('-valsteps', '--validation_steps', default=-1, help='if not set, will be calculated based on the number of examples in the record')

    # model
    parser.add_argument("-fa", '--final_activation', default='softmax')
    parser.add_argument("-a", '--activation', default='relu')
    parser.add_argument('-sum', '--summary', action='store_true')

    # ray tune
    parser.add_argument('-no-ip-address', '--node-ip-address', default=None)
    parser.add_argument('-redis-address', '--redis-address', default=None)
    parser.add_argument('-config-list', '--config-list', default=None)
    parser.add_argument('-temp-dir', '--temp-dir', default=None)
    parser.add_argument('--use-pickle', '--use-pickle', action='store_true')
    parser.add_argument('-node-manager-port', '--node-manager-port', default=None)
    parser.add_argument('-object-store-name', '--object-store-name', default=None)
    parser.add_argument('-raylet-name', '--raylet-name', default=None)

    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    if args.project is None:
        args.project = get_now_timestamp()

    return args


def train_test_model(args, hparams=None, reporter=None):

    callbacks = []
    callbacks.append(kcallbacks.TensorBoard(log_dir=args.logdir, histogram_freq=0, write_graph=False,
                                            write_images=False, write_grads=True, update_freq='batch'))
    callbacks.append(kcallbacks.TerminateOnNaN())
    callbacks.append(kcallbacks.ModelCheckpoint(os.path.join(args.logdir, "model-best.h5"), monitor='val_loss', verbose=1, save_best_only=True, period=1))
    callbacks.append(kcallbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=20, verbose=1))

    if hparams:
        from tensorboard.plugins.hparams import api as hp
        callbacks.append(hp.KerasCallback(args.logdir, hparams))

    if reporter:
        from ray.tune.integration.keras import TuneReporterCallback
        callbacks.append(TuneReporterCallback(reporter))

    # callbacks.append(kcallbacks.LambdaCallback(on_epoch_end=on_epoch_end))

    # ds = get_dataset_by_name('cmp', '/tmp/cmp/')

    if not args.record_dir:
        cache_dir = os.path.join(args.data_dir, args.dataset)
        ds = get_dataset_by_name(args.dataset, cache_dir)
        assert(ds.num_classes > 0), "The dataset must have at least 1 class"
        logger.info("using dataset %s with %d classes" % (ds.__class__.__name__, ds.num_classes))

        if not args.train_on_generator:
            logger.info("writing records")

            record_dir = os.path.join(cache_dir, 'records')
            logger.info("using record dir %s" % record_dir)

            writer = TFWriter(record_dir)
            writer.write(ds)
            writer.validate(ds)

        num_classes = ds.num_classes
    else:
        record_dir = args.record_dir
        num_classes = TFReader(record_dir).num_classes

    losses = [get_loss_by_name(name) for name in args.losses]
    metrics = [get_loss_by_name(name) for name in args.metrics]

    logger.debug("metrics: %s" % str(metrics))
    logger.debug("losses: %s" % str(losses))

    def total_loss(y_true, y_pred):
        total = 0
        for i, l in enumerate(losses):
            total += l(y_true, y_pred) / args.loss_weights[i]
        return total

    input_shape = (args.size[1], args.size[0], 3 if args.color_mode ==
                   preprocessing.ColorMode.RGB else 1)

    logger.info("input shape: %s" % str(input_shape))

    # set scale labels based on sigmoid activation
    scale_labels = args.final_activation == 'sigmoid'

    if args.final_activation == 'sigmoid':
        logger.warn('using only 1 class for sigmoid activation function to work')
        num_classes = 1

    model = get_model_by_name(
        args.model, {'input_shape': input_shape, "num_classes": num_classes})

    model = Model(model.input, Activation(args.final_activation)(model.output))
    logger.info("output shape: %s" % model.output.shape)
    logger.info("input shape: %s" % model.input.shape)

    opt = get_optimizer_by_name(args.optimizer, args.learning_rate)

    # metrics = [tf.keras.metrics.MeanIoU(num_classes=num_classes)]
    # metrics += [tf.keras.metrics.Precision()]
    # metrics += [tf.keras.metrics.CategoricalAccuracy()]
    from python_keras_semantic_segmentation.losses.all import f1_score_keras, f2_score_keras, iou_score_keras, precision_keras, recall_keras  # , iou_score, precision, recall
    from python_keras_semantic_segmentation.losses import categorical_crossentropy

    metrics = [f1_score_keras, f2_score_keras, iou_score_keras, precision_keras, recall_keras, categorical_crossentropy]
    model.compile(optimizer=opt, loss=total_loss, metrics=metrics)  # metrics=losses

    if args.summary:
        model.summary()

    if args.enable_wandb:
        import wandb
        wandb.init(project=args.project)

        callbacks.append(wandb.keras.WandbCallback())

    if args.train_on_generator:
        """
        train_gen = ds.get(DataType.TRAIN)
        train_gen = preprocessing.gen(
            train_gen, args.size, args.batch_size, ds.num_classes, args.color_mode, is_training=True)
        val_gen = ds.get(DataType.VAL)
        val_gen = preprocessing.gen(
            val_gen, args.size, args.batch_size, ds.num_classes, args.color_mode, is_training=False)
        samples_per_epoch = int(ds.num_examples(DataType.TRAIN) / args.batch_size)
        val_steps = int(ds.num_examples(DataType.VAL) / args.batch_size)

        model.fit_generator(
            train_gen, samples_per_epoch=samples_per_epoch, epochs=args.epochs, callbacks=callbacks, validation_data=val_gen,
            validation_steps=val_steps, max_queue_size=10)
        return model.evaluate_generator(val_gen)
        """

        train_gen = loader.DataLoader(ds, DataType.TRAIN, args.size, args.color_mode, args.batch_size)
        val_gen = loader.DataLoader(ds, DataType.VAL, args.size, args.color_mode, args.batch_size)

        model.fit_generator(train_gen, epochs=args.epochs, callbacks=callbacks, validation_data=val_gen, max_queue_size=10)
        return model.evaluate_generator(val_gen)
    else:
        reader = TFReader(record_dir)
        train_ds = reader.get_dataset(DataType.TRAIN)

        train_preprocess_fn = preprocessing_ds.get_preprocess_fn(args.size, args.color_mode, scale_labels=scale_labels, is_training=True)
        train_ds = train_ds.map(train_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_ds = preprocessing_ds.prepare_dataset(train_ds, args.batch_size, shuffle_buffer_size=1000, buffer_size=500)

        val_ds = reader.get_dataset(DataType.VAL)
        val_preprocess_fn = preprocessing_ds.get_preprocess_fn(args.size, args.color_mode, scale_labels=scale_labels, is_training=False)
        val_ds = val_ds.map(val_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = preprocessing_ds.prepare_dataset(val_ds, args.batch_size, shuffle_buffer_size=100, buffer_size=50)

        if args.steps_per_epoch != -1:
            steps_per_epoch = args.steps_per_epoch
        else:
            steps_per_epoch = reader.num_examples(DataType.TRAIN) // args.batch_size

        if args.validation_steps != -1:
            validation_steps = args.validation_steps
        else:
            validation_steps = reader.num_examples(DataType.VAL) // args.batch_size

        model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps,
                  callbacks=callbacks, epochs=args.epochs)
        results = model.evaluate(val_ds, steps=validation_steps)
        return results


def main():
    args = get_args()
    print(train_test_model(args))


if __name__ == "__main__":
    args = get_args()
    results = train_test_model(args)
    print(results)
