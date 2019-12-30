from .processing.pre import preprocess
from .visualizations import show
from keras.models import load_model
import argparse
import imageio
from tensorflow.keras import backend as K


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("-m", '--model_path', type=str, required=True)
    parser.add_argument("-i", "--inputs", type=str, help='input path (file or directory)', required=True)
    parser.add_argument("-o", "--outputs", type=str, help='output path (file or directory)', required=True)
    parser.add_argument("-v", "--vis", action='store_true')

    args = parser.parse_args()

    model = load_model(args.model_path)

    size = K.shape(model.input)[1:3]
    print(size)
    image = imageio.imread(args.inputs)
    image = preprocess(image, size, 1, is_training=False)
    predicted = model.predict(image)
    show.show(predicted)
