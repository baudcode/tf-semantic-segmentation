from tensorflow.keras.models import load_model
# import necessary modules
from ..bin.train import main
import argparse
import os


def export_saved_model(logdir, model_name='model-best.h5'):
    model_path = os.path.join(logdir, model_name)
    model = load_model(model_path, compile=False)

    saved_model_path = os.path.join(args.logdir, 'saved_model', '0')
    model.save(saved_model_path, save_format='tf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logdir', required=True)
    parser.add_argument('-m', '--model_name', default='model-best.h5')
    args = parser.parse_args()

    export_saved_model(args.logdir, args.model_name)
