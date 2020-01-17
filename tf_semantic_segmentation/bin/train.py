from ..models import get_model_by_name, models_by_name
from ..datasets import get_dataset_by_name, DataType, datasets_by_name, get_cache_dir, google_drive_records_by_tag, \
    download_records, DirectoryDataset, TFWriter, TFReader
from ..datasets.utils import convert2tfdataset
from ..losses import get_loss_by_name, losses_by_name
from ..metrics import metrics_by_name, get_metric_by_name
from ..processing import dataset as preprocessing_ds
from ..processing import ColorMode
from ..settings import logger
from ..optimizers import get_optimizer_by_name, names as optimizer_choices
from ..utils import get_now_timestamp


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras import callbacks as kcallbacks
from .. import callbacks as custom_callbacks
import contextlib
import os
import argparse
import shutil
import ast
import inspect


def get_args(args=None):

    color_modes = [int(cm) for cm in ColorMode]

    def str_list_type(x): return list(map(str, x.split(",")))
    def dict_type(x): return ast.literal_eval(x)
    def float_list_type(x): return list(map(float, x.split(",")))
    def int_list_type(x): return [] if x.strip() == "" else list(map(int, x.split(",")))
    def tuple_type(x): return tuple(list(map(int, x.split(","))))

    def any_of(args):
        def any_of_type(x):
            x_list = list(map(str, x.split(",")))
            for t in x_list:
                assert(t in args), "%s is not in list %s" % (t, str(args))

            return x_list

        return any_of_type

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--size', default=None, type=tuple_type)
    parser.add_argument('-m', '--model', default='erfnet', choices=list(models_by_name.keys()))
    parser.add_argument('-gpu', '--gpus', default=[0], type=int_list_type)
    parser.add_argument('-cm', '--color_mode', default=0, type=int, choices=color_modes, help='0 (RGB), 1 (GRAY), 2 (NONE)')
    parser.add_argument('-o', '--optimizer', default='adam', choices=optimizer_choices)
    parser.add_argument('-bs', '--batch_size', default=1, type=int)
    parser.add_argument('-l', '--loss', default='categorical_crossentropy', type=str, choices=list(losses_by_name.keys()))
    parser.add_argument('-lm', '--metrics', default=['iou_score', 'f1_score', 'categorical_accuracy'],
                        type=any_of(metrics_by_name.keys()),
                        help='choices: %s' % (list(metrics_by_name.keys())))
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('-logdir', '--logdir', default='logs/')
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-bufsize', '--buffer_size', default=50, type=int)
    parser.add_argument('-valbufsize', '--val_buffer_size', default=25, type=int)
    parser.add_argument('-log', '--log_level', default='INFO', type=str)
    parser.add_argument('-rm', '--resize_method', default='resize', type=str, choices=['resize', 'resize_with_pad', 'resize_with_crop_or_pad'])
    parser.add_argument('-args', '--model_args', default={}, type=dict_type)
    parser.add_argument('-tpu', '--tpu_strategy', action='store_true')

    # wandb
    parser.add_argument('-p', '--project', default=None, help='project name (logdir name), if None current data will be used')
    parser.add_argument('-wandb', '--enable_wandb', action='store_true')

    # weights
    parser.add_argument('-base_weights', '--base_weights', default=None, type=str)
    parser.add_argument('-weights', '--model_weights', default=None, type=str)
    parser.add_argument('-smv', '--saved_model_version', default=0, type=int)
    parser.add_argument('--no_save_base_model_weights', action="store_true")
    parser.add_argument('--no_save_model_weights', action="store_true")
    parser.add_argument('--no_export_saved_model', action="store_true")

    # data
    parser.add_argument('-data_dir', '--data_dir', default='/hdd/datasets')
    parser.add_argument('-rd', '--record_dir', default=None, help='if none, will be auto detected')
    parser.add_argument('-ds', '--dataset', default=None, choices=list(datasets_by_name.keys()))
    parser.add_argument('-rtag', '--record_tag', default=None, choices=list(google_drive_records_by_tag.keys()))
    parser.add_argument('-dir', '--directory', default=None)  # train a model from directory
    parser.add_argument('-aug', '--augmentations', default=[], type=any_of(preprocessing_ds.augmentation_methods))

    parser.add_argument('-tog', '--train_on_generator', action='store_true')
    parser.add_argument('-steps', '--steps_per_epoch', default=-1, type=int, help='if not set, will be calculated based on the number of examples in the record')
    parser.add_argument('-valsteps', '--validation_steps', default=-1, type=int, help='if not set, will be calculated based on the number of examples in the record')

    # model
    parser.add_argument("-fa", '--final_activation', default='softmax', type=str, choices=['softmax', 'sigmoid'])
    parser.add_argument("-a", '--activation', default='relu', choices=['relu', 'relu6', 'mish', 'swich'])
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

    # callbacks
    parser.add_argument('--no_terminate_on_nan', action='store_true')

    # model checkpoints
    parser.add_argument('--no_model_checkpoint', action='store_true')
    parser.add_argument('-mc-monitor', '--model_checkpoint_monitor', default='val_loss', type=str)
    parser.add_argument('-mc-no-sbo', '--no_save_best_only', action='store_true')

    # tensorboard
    parser.add_argument('--no_tensorboard', action='store_true')
    parser.add_argument('--uf', '--update_freq', default='batch', type=str, choices=['batch', 'epoch'])

    # early stopping
    parser.add_argument('--no_early_stopping', action='store_true')
    parser.add_argument('-es-patience', '--early_stopping_patience', default=20, type=int)
    parser.add_argument('-es-mode', '--early_stopping_mode', default='min', type=str)
    parser.add_argument('-es-monitor', '--early_stopping_monitor', default='val_loss', type=str)

    args = parser.parse_args(args=args)

    os.makedirs(args.logdir, exist_ok=True)
    if args.project is None:
        args.project = get_now_timestamp()

    # tf.get_logger().setLevel(args.log_level)
    logger.setLevel(args.log_level)

    return args


def setup_devices():
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def train_test_model(args, hparams=None, reporter=None):

    logger.info("setting up devices")
    # allow growth to precent memory errors
    setup_devices()

    logger.info("setting up callbacks")
    callbacks = []

    if not args.no_tensorboard:
        callbacks.append(kcallbacks.TensorBoard(log_dir=args.logdir, histogram_freq=0, write_graph=False,
                                                write_images=False, write_grads=True, update_freq=args.update_freq))

    if not args.no_terminate_on_nan:
        callbacks.append(kcallbacks.TerminateOnNaN())

    if not args.no_model_checkpoint:
        callbacks.append(kcallbacks.ModelCheckpoint(os.path.join(args.logdir, "model-best.h5"),
                                                    monitor=args.model_checkpoint_monitor,  # val_loss default
                                                    verbose=1,
                                                    save_best_only=not args.no_save_best_only,
                                                    period=1))

    if not args.no_early_stopping:
        callbacks.append(kcallbacks.EarlyStopping(monitor=args.early_stopping_monitor,  # default: val_loss
                                                  mode=args.early_stopping_mode,  # default: min
                                                  min_delta=0,
                                                  patience=args.early_stopping_patience,  # default: 20
                                                  verbose=1))

    if hparams:
        from tensorboard.plugins.hparams import api as hp
        callbacks.append(hp.KerasCallback(args.logdir, hparams))

    if reporter:
        from ray.tune.integration.keras import TuneReporterCallback
        callbacks.append(TuneReporterCallback(reporter))

    if args.tpu_strategy:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    elif len(args.gpus) == 0:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    elif len(args.gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:%d" % args.gpus[0])
    else:
        strategy = tf.distribute.MirroredStrategy(devices=['/gpu:%d' % gpu for gpu in args.gpus])

    global_batch_size = args.batch_size * (len(args.gpus) if len(args.gpus) > 0 else 1)

    # callbacks.append(kcallbacks.LambdaCallback(on_epoch_end=on_epoch_end))

    assert(args.record_dir is not None or args.dataset is not None or args.record_tag is not None or args.directory is not None)

    logger.info("setting up dataset")
    if args.dataset or args.directory:
        if args.dataset:
            cache_dir = get_cache_dir(args.data_dir, args.dataset)
            ds = get_dataset_by_name(args.dataset, cache_dir)
        else:
            ds = DirectoryDataset(args.directory)
            cache_dir = args.directory

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
    elif args.record_dir:
        if not os.path.exists(args.record_dir):
            raise Exception("cannot find record dir %s" % args.record_dir)
        record_dir = args.record_dir
        num_classes = TFReader(record_dir).num_classes
    elif args.record_tag:
        record_tag = args.record_tag
        record_dir = os.path.join(args.data_dir, 'downloaded', record_tag)
        download_records(record_tag, record_dir)

    if args.size and args.color_mode != ColorMode.NONE:
        input_shape = (args.size[1], args.size[0], 3 if args.color_mode == ColorMode.RGB else 1)

    elif args.train_on_generator:
        raise Exception("please specify the 'size' and 'color_mode' argument when training using the generator")
    else:
        input_shape = TFReader(record_dir).input_shape

    logger.info("input shape: %s" % str(input_shape))

    # set scale labels based on sigmoid activation
    scale_labels = args.final_activation == 'sigmoid'

    if num_classes != 2 and args.final_activation == 'sigmoid':
        logger.error('do not choose sigmoid as the final activation when the dataset has more than 2 classes')
        exit(0)

    if args.final_activation == 'sigmoid':
        logger.warning('using only 1 class for sigmoid activation function to work')
        num_classes = 1

    logger.debug('strategy: %s' % str(strategy))

    # check valid model args
    if args.model in models_by_name:
        valid_model_args = list(inspect.signature(models_by_name[args.model]).parameters.keys())

        for key in args.model_args.keys():
            if key not in valid_model_args:
                raise Exception("invalid model args; cannot find key %s in %s for model of name %s" % (key, str(valid_model_args), args.model))

    with strategy.scope():
        model_args = {'input_shape': input_shape, "num_classes": num_classes}
        model_args.update(args.model_args)

        model, base_model = get_model_by_name(
            args.model, model_args)

        if not args.no_save_base_model_weights:
            callbacks.append(custom_callbacks.SaveBestWeights(base_model, os.path.join(args.logdir, 'best-base-weights.h5')))

        if not args.no_save_model_weights:
            callbacks.append(custom_callbacks.SaveBestWeights(model, os.path.join(args.logdir, 'best-weights.h5')))

        if args.model_weights:
            logger.info("restoring model weights from %s" % args.model_weights)
            model.load_weights(args.model_weights)

        if args.base_weights:
            logger.info("restoring base model weights from %s" % args.base_weights)
            base_model.load_weights(args.base_weights)

        model = Model(model.input, Activation(args.final_activation)(model.output))
        logger.info("output shape: %s" % model.output.shape)
        logger.info("input shape: %s" % model.input.shape)

        # loss and metrics
        loss = get_loss_by_name(args.loss)
        metrics = [get_metric_by_name(name) for name in args.metrics]

        logger.info("metrics: %s" % str(metrics))
        logger.info("loss: %s" % str(loss))

        opt = get_optimizer_by_name(args.optimizer, args.learning_rate)
        model.compile(optimizer=opt, loss=loss, metrics=metrics)  # metrics=losses

    if args.summary:
        model.summary()

    if args.enable_wandb:
        import wandb
        wandb.init(project=args.project)
        callbacks.append(wandb.keras.WandbCallback())

    if args.train_on_generator:
        train_ds = convert2tfdataset(ds, DataType.TRAIN)
        val_ds = convert2tfdataset(ds, DataType.VAL)
    else:
        logger.info("using tfreader to read record dir %s" % record_dir)
        reader = TFReader(record_dir)
        train_ds = reader.get_dataset(DataType.TRAIN)
        val_ds = reader.get_dataset(DataType.VAL)

    logger.info("building input pipeline")
    # train preprocessing
    train_preprocess_fn = preprocessing_ds.get_preprocess_fn(args.size, args.color_mode, args.resize_method, scale_labels=scale_labels, is_training=True)
    train_ds = train_ds.map(train_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    augment_fn = None if len(args.augmentations) == 0 else preprocessing_ds.get_augment_fn(args.size, args.batch_size, methods=args.augmentations)
    train_ds = preprocessing_ds.prepare_dataset(train_ds, global_batch_size, buffer_size=args.buffer_size, augment_fn=augment_fn)

    # val preprocessing
    val_preprocess_fn = preprocessing_ds.get_preprocess_fn(args.size, args.color_mode, args.resize_method, scale_labels=scale_labels, is_training=False)
    val_ds = val_ds.map(val_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = preprocessing_ds.prepare_dataset(val_ds, global_batch_size, buffer_size=args.val_buffer_size)

    if args.steps_per_epoch != -1:
        steps_per_epoch = args.steps_per_epoch
    elif args.train_on_generator:
        steps_per_epoch = ds.num_examples(DataType.TRAIN) // global_batch_size
    else:
        steps_per_epoch = reader.num_examples(DataType.TRAIN) // global_batch_size

    if args.validation_steps != -1:
        validation_steps = args.validation_steps
    elif args.train_on_generator:
        validation_steps = ds.num_examples(DataType.VAL) // global_batch_size
    else:
        validation_steps = reader.num_examples(DataType.VAL) // global_batch_size

    model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps,
              callbacks=callbacks, epochs=args.epochs)

    results = model.evaluate(val_ds, steps=validation_steps)

    # saved model export
    saved_model_path = os.path.join(args.logdir, 'saved_model', str(args.saved_model_version))

    if os.path.exists(saved_model_path):
        shutil.rmtree(saved_model_path)

    if not args.no_export_saved_model:
        logger.info("exporting saved model to %s" % saved_model_path)
        model.save(saved_model_path, save_format='tf')

    return results


def main():
    args = get_args()
    print(train_test_model(args))


if __name__ == "__main__":
    args = get_args()
    results = train_test_model(args)
    print(results)
