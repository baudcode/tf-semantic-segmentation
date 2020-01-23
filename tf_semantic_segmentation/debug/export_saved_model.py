from tensorflow.keras.models import load_model
# import necessary modules
from ..bin.train import main
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logdir', required=True)
    args = parser.parse_args()

    model_path = os.path.join(args.logdir, 'model-best.h5')
    model = load_model(model_path, compile=False)

    saved_model_path = os.path.join(args.logdir, 'saved_model', '0')
    model.save(saved_model_path, save_format='tf')
