import numpy as np
import tensorflow as tf


class PredictionCallback(tf.keras.callbacks.Callback):
    """ Very simple epoch metrics printer 
    TODO: enhance to log predictions to wandb or harddrive
    """

    def on_epoch_end(self, epoch, logs={}):
        metrics = self.model.evaluate(self.validation_data[0])
        print(metrics)
        print('evaluate at epoch: {}'.format(epoch))


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
