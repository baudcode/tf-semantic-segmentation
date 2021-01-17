from __future__ import print_function

from .utils import get_random_image
from .settings import logger

import requests
import numpy as np
import json
import os
from pprint import pprint


def _parse_int(i):
    try:
        return int(i)
    except:
        return i


def _parse_shape(shape):
    return list(map(_parse_int, shape))


def retrieve_metadata(model_name, host='localhost', port=8501, signature_def='serving_default'):
    server_url = "http://%s:%d/v1/models/%s/metadata" % (host, port, model_name)
    data = requests.get(server_url).json()
    inputs_metadata = data['metadata']['signature_def']['signature_def'][signature_def]['inputs']
    outputs_metadata = data['metadata']['signature_def']['signature_def'][signature_def]['outputs']

    inputs = {}
    for key, input_data in inputs_metadata.items():
        shape = [dim['size'] for dim in input_data['tensor_shape']['dim']]
        shape = _parse_shape(shape)
        inputs[key] = {}
        inputs[key]['shape'] = shape
        inputs[key]['dtype'] = input_data['dtype']

    outputs = {}
    for key, output_data in outputs_metadata.items():
        shape = [dim['size'] for dim in output_data['tensor_shape']['dim']]
        shape = _parse_shape(shape)
        outputs[key] = {}
        outputs[key]['shape'] = shape
        outputs[key]['dtype'] = output_data['dtype']

    return {"inputs": inputs, "outputs": outputs}


def predict(image, host='localhost', model_name='default', input_name='inputs', version=0, port=8501):
    if version > 0:
        server_url = "http://%s:%d/v1/models/%s/version/%d:predict" % (host, port, model_name, version)
    else:
        server_url = "http://%s:%d/v1/models/%s:predict" % (host, port, model_name)

    # Compose a JSON Predict request (send JPEG image in base64).

    image_as_list = image.tolist()
    instances = [{input_name: image_as_list}]
    payload = {
        "instances": instances
    }
    response = requests.post(server_url, json=payload)
    response.raise_for_status()
    return response.json()['predictions'][0]


def predict_on_batch(images, host='localhost', model_name='default', input_name='inputs', port=8501):
    server_url = "http://%s:%d/v1/models/%s:predict" % (host, port, model_name)

    # Compose a JSON Predict request (send JPEG image in base64).

    instances = [{input_name: image.tolist()} for image in images]
    payload = {
        "instances": instances
    }
    response = requests.post(server_url, json=payload)
    response.raise_for_status()
    return response.json()['predictions']


def calculate_average_latency(size=(256, 256), host='localhost', port=8501, input_name='inputs', model_name='default'):
    image = get_random_image(width=size[0], height=size[1])
    server_url = "http://%s:%d/v1/models/%s:predict" % (host, port, model_name)

    image_as_list = image.tolist()
    instances = [{input_name: image_as_list}]
    payload = {
        "instances": instances
    }

    print("warming up...")
    # Send few requests to warm-up the model.
    for _ in range(3):
        response = requests.post(server_url, json=payload)
        response.raise_for_status()

    # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(server_url, json=payload)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions'][0]

    print('Avg latency: {} ms'.format((total_time * 1000) / num_requests))


def write_model_config(models, model_config_path):
    """
        Schema:
        model_config_list {
        config {
            name: 'my_first_model'
            base_path: '/tmp/my_first_model/'
            model_platform: 'tensorflow'
        }
        config {
            name: 'my_second_model'
            base_path: '/tmp/my_second_model/'
            model_platform: 'tensorflow'
        }
        }
    """
    config = "model_config_list {\n"
    for model in models:
        config += "config {\n\tname: '%s'\n\tbase_path: '%s'\n\tmodel_platform: 'tensorflow'\n}\n" % (model["name"], os.path.abspath(model['path']))
    config += "}"
    with open(model_config_path, 'w') as writer:
        writer.write(config)


def get_models_from_directory(directory, contains=None):
    models = []

    for logdir in os.listdir(directory):
        if not contains or contains in logdir:
            model_path = os.path.join(directory, logdir, 'saved_model')
            if os.path.exists(model_path):
                logger.info("adding model %s" % logdir)
                models.append({
                    "path": model_path,
                    "name": str(len(models))
                })
            else:
                logger.warning("skipping model %s, because saved model does not exist" % logdir)
    return models


def write_model_config_from_models_dir(models_dir, contains=None, config_path='models.yaml'):

    models = get_models_from_directory(models_dir, contains=contains)
    logger.info("found %d models" % len(models))
    logger.info("written tensorflow model server config to %s" % config_path)
    write_model_config(models, config_path)
    return models


def threshold_predictions(p, threshold=0.5):
    p[p >= threshold] = 1.0
    p[p < threshold] = 0.0
    return p


def ensemble_prediction(models, image, host="localhost", port=8501):

    predictions = []

    for model in models:
        version = model['version'] if "version" in model else 0
        input_name = model["input_name"] if "input_name" in model else "input_1"
        p = predict(image, host=host, model_name=model['name'], port=port, input_name=input_name, version=version)
        predictions.append(np.asarray(p))
    ensemble = np.mean(predictions, axis=0)
    return ensemble, predictions


def ensemble_inference(models, image, host="localhost", port=8501, threshold=0.5):

    ensemble, predictions = ensemble_prediction(models, image, host=host, port=port)

    # for ensemble
    if ensemble.shape[-1] == 1 and threshold > 0:
        ensemble = threshold_predictions(ensemble, threshold=threshold)
    elif ensemble.shape[-1] > 1:
        ensemble = np.expand_dims(np.argmax(ensemble, axis=-1), axis=-1)

    # for predictions
    if ensemble.shape[-1] == 1 and threshold > 0:
        predictions = [threshold_predictions(p, threshold=threshold) for p in predictions]
    elif ensemble.shape[-1] > 1:
        predictions = [np.expand_dims(np.argmax(p, axis=-1), axis=-1) for p in predictions]

    return ensemble, predictions


if __name__ == '__main__':
    # calculate_average_latency()
    print(retrieve_metadata('0'))
