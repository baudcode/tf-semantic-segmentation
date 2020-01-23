from .visualizations import masks
from .settings import logger

import numpy as np
import tensorflow as tf
import wandb
from tensorboardX import SummaryWriter


class PredictionCallback(tf.keras.callbacks.Callback):
    """Predictions logged using tensorboardX"""

    def __init__(self, model, logdir, val_generator, scaled_mask, binary_threshold=0.5, update_freq=1):
        super(PredictionCallback, self).__init__()
        self.val_generator = val_generator
        self.writer = SummaryWriter(logdir=logdir)
        self.scaled_mask = scaled_mask
        self.binary_threshold = binary_threshold
        self._model = model
        self.num_classes = self._model.output.shape.as_list()[-1]
        self.update_freq = update_freq

    def on_epoch_end(self, epoch, logs={}):

        if epoch == -1:
            epoch = 0
        else:
            epoch = epoch + 1

        if epoch % self.update_freq == 0:
            logger.debug("logging images to tensorboard, epoch=%d" % epoch)

            for input_batch, target_batch in self.val_generator:
                input_batch = input_batch.numpy()
                target_batch = target_batch.numpy()
                break

            # input_batch, target_batch = next(iter(self.val_generator.as_numpy_iterator()))

            if self.scaled_mask:
                target_batch = np.expand_dims(target_batch, axis=-1)

            # predict
            pred_batch = self._model.predict_on_batch(input_batch).numpy()

            predictions_on_inputs = masks.get_colored_segmentation_mask(pred_batch, self.num_classes, images=input_batch, binary_threshold=self.binary_threshold)
            self.writer.add_images('inputs/with_predictions', predictions_on_inputs, dataformats='NHWC', global_step=epoch)

            targets_on_inputs = masks.get_colored_segmentation_mask(target_batch, self.num_classes, images=input_batch, binary_threshold=self.binary_threshold)
            self.writer.add_images('inputs/with_targets', targets_on_inputs, dataformats='NHWC', global_step=epoch)

            targets_rgb = masks.get_colored_segmentation_mask(target_batch, self.num_classes, binary_threshold=self.binary_threshold, alpha=1.0)
            self.writer.add_images('targets/rgb', targets_rgb, dataformats='NHWC', global_step=epoch)

            pred_rgb = masks.get_colored_segmentation_mask(pred_batch, self.num_classes, binary_threshold=self.binary_threshold, alpha=1.0)
            self.writer.add_images('predictions/rgb', pred_rgb, dataformats='NHWC', global_step=epoch)

            if not self.scaled_mask:
                pred_batch = np.argmax(pred_batch, axis=-1).astype(np.float32)
                target_batch = np.argmax(target_batch, axis=-1).astype(np.float32)

                # reshape, that add_images works
                pred_batch = np.expand_dims(pred_batch, axis=-1)
                target_batch = np.expand_dims(target_batch, axis=-1)
            else:
                pred_batch[pred_batch > self.binary_threshold] = 1.0
                pred_batch[pred_batch <= self.binary_threshold] = 0.0

            self.writer.add_images('inputs', input_batch, dataformats='NHWC', global_step=epoch)
            self.writer.add_images('targets', target_batch, dataformats='NHWC', global_step=epoch)
            self.writer.add_images('predictions', pred_batch, dataformats='NHWC', global_step=epoch)

            # wandb_logs.update({"targets": [wandb.Image(i) for i in target_batch]})
            # wandb_logs.update({"inputs": [wandb.Image(i) for i in input_batch]})
            # wandb_logs.update({"predictions": [wandb.Image(i) for i in pred_batch]})
            # wandb.log(wandb_logs, step=epoch)


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
