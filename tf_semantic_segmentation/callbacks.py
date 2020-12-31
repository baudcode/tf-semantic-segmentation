from .visualizations import masks
from .settings import logger

import numpy as np
import tensorflow as tf
import wandb
import imageio
import time
from tensorflow.keras import backend as K


def get_time_diff_str(start, end, period=1):
    diff = int((end - start) / period) if period != 0 else int((end - start))
    s = diff % 60
    m = diff // 60
    h = diff // 3600
    return "%dh %dm %ds" % (h, m, s)

class TimingCallback(tf.keras.callbacks.Callback):
    """ Calculates the average time per batch """

    def __init__(self, batch_size: int, total: int, log_interval: int = 10):
        super(TimingCallback, self).__init__()
        self.start_time = time.time()
        self.times = []
        self.batch_size = batch_size
        self.total = total
        self.log_interval = log_interval
    
    def on_batch_end(self, batch, logs={}):
        self.times.append(time.time() - self.batch_start_time)

        if len(self.times) % self.log_interval == 0:
            log_times = self.times[-self.log_interval:]
            sec_per_batch = np.mean(log_times)
            images_per_sec = (self.batch_size * self.log_interval) / sum(log_times)
            logger.info("[TimingCallback] Batch %d: %.3f s/batch %.1f images/s" % (batch, sec_per_batch, images_per_sec))
    
    def on_epoch_end(self, epoch, logs={}):
        total_time = sum(self.times)
        sec_per_batch = np.mean(self.times)
        images_per_sec = self.total / total_time

        logger.info("[TimingCallback] Epoch %d: %.3f s/batch %.1f images/s" % (epoch, sec_per_batch, images_per_sec))
        self.times = []
        self.start_time = time.time()

    def on_batch_begin(self, batch, logs={}):
        self.batch_start_time = time.time()
    


class PredictionCallback(tf.keras.callbacks.Callback):
    """Predictions logged using tensorflow summary writer"""

    def __init__(self, model, logdir, generator, scaled_mask, binary_threshold=0.5, update_freq=1):
        super(PredictionCallback, self).__init__()
        self.generator = generator
        self.summary_writer = tf.summary.create_file_writer(logdir)
        self.scaled_mask = scaled_mask
        self.binary_threshold = binary_threshold
        self._model = model
        self.num_classes = self._model.output.shape.as_list()[-1] if not scaled_mask else 2
        self.update_freq = update_freq
        self.start_time = time.time()

    def _log(self, input_batch, target_batch, step):

        if self.scaled_mask:
            target_batch = np.expand_dims(target_batch, axis=-1)

        batch_size = input_batch.shape[0]

        # predict
        pred_batch = self._model.predict_on_batch(input_batch)
        pred_batch = _get_numpy(pred_batch)

        with self.summary_writer.as_default():

            def add_images(name, images):
                images = tf.split(images, num_or_size_splits=batch_size, axis=0)
                image = tf.concat(images, axis=2)
                tf.summary.image(name, image, step=step, max_outputs=batch_size)

            add_images('inputs', input_batch)
            # colored

            predictions_on_inputs = masks.get_colored_segmentation_mask(pred_batch, self.num_classes, images=input_batch, binary_threshold=self.binary_threshold)
            add_images('inputs/with_predictions', predictions_on_inputs)

            targets_on_inputs = masks.get_colored_segmentation_mask(target_batch, self.num_classes, images=input_batch, binary_threshold=self.binary_threshold)
            add_images('inputs/with_targets', targets_on_inputs)

            targets_rgb = masks.get_colored_segmentation_mask(target_batch, self.num_classes, binary_threshold=self.binary_threshold, alpha=1.0)
            add_images('targets/rgb', targets_rgb)

            pred_rgb = masks.get_colored_segmentation_mask(pred_batch, self.num_classes, binary_threshold=self.binary_threshold, alpha=1.0)
            add_images('predictions/rgb', pred_rgb)

            if not self.scaled_mask:
                pred_batch = np.argmax(pred_batch, axis=-1).astype(np.float32)
                target_batch = np.argmax(target_batch, axis=-1).astype(np.float32)

                # reshape, that add_images works
                pred_batch = np.expand_dims(pred_batch, axis=-1)
                target_batch = np.expand_dims(target_batch, axis=-1)
            else:
                pred_batch[pred_batch > self.binary_threshold] = 1.0
                pred_batch[pred_batch <= self.binary_threshold] = 0.0

            # simple
            # intersection = np.logical_or(target_batch, pred_batch).astype(np.float32)
            # union = np.logical_and(target_batch, pred_batch).astype(np.float32)

            # add_images('metrics/intersection', intersection)
            # add_images('metrics/union', union)

            # scale
            target_batch = (target_batch * 255. / (self.num_classes - 1)).astype(np.uint8)
            add_images('targets', target_batch)
            add_images('predictions', pred_batch)


def _get_numpy(tensor):
    try:
        return tensor.numpy()
    except:
        return tensor


class EpochPredictionCallback(PredictionCallback):

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        et = time.time()

        if epoch == -1:
            epoch = 0
        else:
            epoch = epoch + 1

            # log time for performance tests
            logger.info("time per epoch: %s, avg time per epoch: %s" %
                        (get_time_diff_str(self.epoch_start_time, et), get_time_diff_str(self.start_time, et, period=epoch)))

        if epoch % self.update_freq == 0:
            logger.debug("logging images to tensorboard, epoch=%d" % epoch)

            for input_batch, target_batch in self.generator:
                self._log(_get_numpy(input_batch), _get_numpy(target_batch), epoch)
                break


class BatchPredictionCallback(PredictionCallback):

    def on_batch_end(self, batch, logs={}):
        if batch == -1:
            self._batch = 0
        else:
            self._batch += 1

        if self._batch % self.update_freq == 0:
            logger.info("logging images to tensorboard, batch=%d" % self._batch)

            for input_batch, target_batch in self.generator:
                self._log(_get_numpy(input_batch), _get_numpy(target_batch), self._batch)
                break


class SaveBestWeights(tf.keras.callbacks.Callback):
    """Save the best weights, based on the current loss

    Arguments:
        base_model: (any keras model)
        weihghts_path: str, path to save the weights to
        monitor: str, what metric watch for the minimum
        verbose: bool, whether to log
    """

    def __init__(self, base_model, weights_path, monitor='loss', verbose=0):
        super(SaveBestWeights, self).__init__()
        self.monitor = monitor
        self.base_model = base_model
        self.weights_path = weights_path
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is not None and np.less(current, self.best):
            self.best = current
            self.base_model.save_weights(self.weights_path)
            if self.verbose:
                print("[Epoch %d] saving base model weights to %s" % (epoch, self.weights_path))
