from __future__ import print_function

from .utils import get_random_image
import requests
import numpy as np
import json


def predict(image, host='localhost', model_name='default', input_name='inputs', port=8501):
    server_url = "http://%s:%d/v0/models/%s:predict" % (host, port, model_name)

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


if __name__ == '__main__':
    calculate_average_latency()
