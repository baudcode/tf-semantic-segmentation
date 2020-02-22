import tensorflow as tf
import argparse
from ..settings import logger


def convert(saved_model_dir, output_path, optimize_for_size=True):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE if optimize_for_size else tf.lite.Optimize.DEFAULT]
    logger.info('optimizations: %s', str(converter.optimizations))
    tflite_quant_model = converter.convert()
    open(output_path, "wb").write(tflite_quant_model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--saved_model_dir', required=True, help='path to the saved model dir')
    parser.add_argument('-o', '--output_path', default='output_model.tflite', help='output model path')
    parser.add_argument('--no_size_optimization', action='store_true')
    args = parser.parse_args()

    convert(args.saved_model_dir, args.output_path, optimize_for_size=not args.no_size_optimization)
