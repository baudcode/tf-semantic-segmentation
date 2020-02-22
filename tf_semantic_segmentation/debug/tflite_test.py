import numpy as np
import tensorflow as tf
import time
from ..settings import logger


class TFLiteInterpreter():

    def __init__(self, tflite_model_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        self.interpreter.allocate_tensors()

        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    @property
    def output_shape(self):
        return self.interpreter.get_output_details()[0]['shape']

    @property
    def input_shape(self):
        return self.interpreter.get_input_details()[0]['shape']

    def predict(self, image):

        test_image = np.expand_dims(image, axis=0)

        start = time.time()
        self.interpreter.set_tensor(self.input_index, test_image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)[0]
        logger.debug("Inference took %.3f seconds" % (time.time() - start))
        return output

    def predict_on_batch(self, batch):

        start = time.time()
        self.interpreter.set_tensor(self.input_index, batch)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_index)
        logger.debug("Inference took %.3f seconds" % (time.time() - start))
        return output


if __name__ == "__main__":

    import argparse
    import imageio

    from ..processing.dataset import resize_and_change_color
    from ..visualizations import show, masks

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='input image path', required=True)
    parser.add_argument('-m', '--tflite_model_path', help='path to the tflite model', required=True)
    parser.add_argument('-rm', '--resize_method', default='resize', help='method for resizing inputs')
    parser.add_argument('-thresh', '--binary_threshold', default=0.5, help='threshold')
    args = parser.parse_args()

    interpreter = TFLiteInterpreter(args.tflite_model_path)
    logger.info("input shape: %s, output shape: %s" % (str(interpreter.input_shape), str(interpreter.output_shape)))

    # define model parameters
    size = interpreter.input_shape[1:3]
    color_mode = 0 if interpreter.input_shape[-1] == 3 else 1
    resize_method = args.resize_method
    scale_mask = interpreter.output_shape[-1] == 1
    num_classes = 2 if interpreter.output_shape[-1] == 1 else interpreter.output_shape[-1]

    image = imageio.imread(args.image)

    # scale between 0 and 1
    image = tf.image.convert_image_dtype(image, tf.float32)

    # resize method for image create float32 image anyway
    image, _ = resize_and_change_color(image, None, size, color_mode, resize_method=resize_method)

    batch = tf.expand_dims(image, axis=0)
    predictions = interpreter.predict_on_batch(batch)
    seg_masks = masks.get_colored_segmentation_mask(predictions, num_classes, images=batch.numpy(), binary_threshold=args.binary_threshold)

    show.show_images(seg_masks)
