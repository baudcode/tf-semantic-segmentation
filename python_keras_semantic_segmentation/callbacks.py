import tensorflow as tf


class PredictionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        metrics = self.model.evaluate(self.validation_data[0])
        print(metrics)
        print('evaluate at epoch: {}'.format(epoch))
