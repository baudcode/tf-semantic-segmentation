# TF Semantic Segmentation

[![Build Status](https://travis-ci.org/baudcode/tf-semantic-segmentation.svg?branch=master)](https://travis-ci.org/baudcode/tf-semantic-segmentation)
[![PyPI Status Badge](https://badge.fury.io/py/tf-semantic-segmentation.svg)](https://pypi.org/project/tf-semantic-segmentation/)
[![codecov](https://codecov.io/gh/baudcode/tf-semantic-segmentation/branch/dev/graph/badge.svg)](https://codecov.io/gh/baudcode/tf-semantic-segmentation)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xBH4WxhJ7TlnC7pck7ifLjo3NrdYmug-)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://tf-semantic-segmentation.readthedocs.io/?badge=latest)

### Quick Start
See [GETTING_STARTED](#getting-started), or the [Colab Notebook](https://colab.research.google.com/drive/1xBH4WxhJ7TlnC7pck7ifLjo3NrdYmug-).

Learn more at our [documentation](https://tf-semantic-segmentation.readthedocs.io/en/latest/).
See upcoming features on our [roadmap](ROADMAP.md).

## Features

- Distributed Training on Multiple GPUs
- Hyper Parameter Optimization using WandB
- WandB Integration
- Easily create TFRecord from Directory
- Tensorboard visualizations
- Ensemble inference

#### Datasets

  - Ade20k
  - Camvid
  - Cityscapes
  - MappingChallenge
  - MotsChallenge
  - Coco
  - PascalVoc2012
  - Taco
  - Shapes (randomly creating triangles, rectangles and circles)
  - Toy (Overlaying TinyImageNet with MNIST)
  - ISIC2018
  - CVC-ClinicDB



#### Models

  
  - [Unet](https://arxiv.org/abs/1505.04597)
  - [Erfnet](https://arxiv.org/abs/1806.08522)
  - [MultiResUnet](https://arxiv.org/abs/1902.04049)
  - [PSP](https://arxiv.org/abs/1612.01105) (experimental)
  - [FCN](https://arxiv.org/abs/1411.4038) (experimental)
  - [NestedUnet (Unet++)](https://arxiv.org/abs/1807.10165) (experimental)
  - [U2Net / U2NetP](https://arxiv.org/abs/2005.09007) (experimental)
  - SatelliteUnet
  - MobilenetUnet (unet with mobilenet encoder pre-trained on imagenet)
  - InceptionResnetV2Unet (unet with inception-resnet v2 encoder pre-trained on imagenet)
  - ResnetUnet (unet with resnet50 encoder pre-trained on imagenet)
  - AttentionUnet

#### Losses

  - Catagorical Crossentropy
  - Binary Crossentropy
  - Crossentropy + SSIM
  - Dice
  - Crossentropy + Dice
  - Tversky
  - Focal
  - Focal + Tversky

#### Activations

  - mish
  - swish
  - relu6

#### Optimizers

  - Ranger
  - RAdam

#### Normalization

  - Instance Norm
  - Batch Norm

#### On the fly Augmentations

  - flip left/right
  - flip up/down
  - rot 180
  - color

## [Getting Started](#getting-started)

### Requirements

```shell
sudo apt-get install libsm6 libxext6 libxrender-dev libyaml-dev libpython3-dev
```

#### Tensorflow (2.x) & Tensorflow Addons (optional)

```shell
pip install tensorflow-gpu==2.4.0 --upgrade
pip install tensorflow-addons==0.12.0 --upgrade
```

### Installation

```shell
pip install tf-semantic-segmentation
```

### Run tensorboard

- Hint: To see train/test/val images you have to start tensorboard like this

```bash
tensorboard --logdir=logs/ --reload_multifile=true
```

### Train on inbuild datasets (generator)

```bash
python -m tf_semantic_segmentation.bin.train -ds 'tacobinary' -bs 8 -e 100 \
    -logdir 'logs/taco-binary-test' -o 'adam' -lr 5e-3 --size 256,256 \
    -l 'binary_crossentropy' -fa 'sigmoid' \
    --train_on_generator --gpus='0' \
    --tensorboard_train_images --tensorboard_val_images
```

### Create a tfrecord from a dataset
```bash

# create a tfrecord from the toy dataset and resize to 128x128
tf-semantic-segmentation-tfrecord-writer -d 'toy' -c /hdd/datasets/ -s '128,128'
```

### Train using a fixed record path

```bash
python -m tf_semantic_segmentation.bin.train --record_dir=records/cityscapes-512x256-rgb/ \
    -bs 4 -e 100 -logdir 'logs/cityscapes-bs8-e100-512x256' -o 'adam' -lr 1e-4 -l 'categorical_crossentropy' \
    -fa 'softmax' -bufsize 50 --metrics='iou_score,f1_score' -m 'erfnet' --gpus='0' -a 'mish' \
    --tensorboard_train_images --tensorboard_val_images
```

### Multi GPU training

```bash
python -m tf_semantic_segmentation.bin.train --record_dir=records/cityscapes-512x256-rgb/ \
    -bs 4 -e 100 -logdir 'logs/cityscapes-bs8-e100-512x256' -o 'adam' -lr 1e-4 -l 'categorical_crossentropy' \
    -fa 'softmax' -bufsize 50 --metrics='iou_score,f1_score' -m 'erfnet' --gpus='0,1,2,3' -a 'mish'
```

## Using Code

```python
from tf_semantic_segmentation.bin.train import train_test_model, get_args

# get the default args
args = get_args({})

# change some parameters
# !rm -r logs/
args.model = 'erfnet'
# args['color_mode'] = 0
args.batch_size = 8
args.size = [128, 128] # resize input dataset to this size
args.epochs = 10
args.learning_rate = 1e-4
args.optimizer = 'adam' # ['adam', 'radam', 'ranger']
args.loss = 'dice'
args.logdir = 'logs'
args.record_dir = "datasets/shapes/records"
args.final_activation = 'softmax'

# train and test
results, model = train_test_model(args)
```

## Models

- Erfnet
- Unet

```python
from tf_semantic_segmentation import models

# print all available models
print(list(modes.models_by_name.keys()))

# returns a model (without the final activation function)
model = models.get_model_by_name('erfnet', {"input_shape": (128, 128, 3), "num_classes": 5})

# call models directly
model = models.erfnet(input_shape=(128, 128), num_classes=5)
```

## Use your own dataset

- Accepted file types are: jpg(jpeg) and png

If you already have a train/test/val split then use the following data structure:

```text
dataset/
    labels.txt
    test/
        images/
        masks/
    train/
        images/
        masks/
    val/
        images/
        masks/
```

or use

```text
dataset/
    labels.txt
    images/
    masks/
```

The labels.txt should contain a list of labels separated by newline [/n]. For instance it looks like this:

```text
background
car
pedestrian
```

- To create a tfrecord using the original image size and color use the script like this:

```shell
INPUT_DIR = ...
tf-semantic-segmentation-tfrecord-writer -dir $INPUT_DIR -r $INPUT_DIR/records
```

There are the following addition arguments:

- -s [--size] '$width,$height' (f.e. "512,512")
- -rm [--resize_method] ('resize', 'resize_with_pad', 'resize_with_crop_or_pad)
- cm [--color_mode] (0=RGB, 1=GRAY, 2=NONE (default))



## Datasets

```python
from tf_semantic_sementation.datasets import get_dataset by name, datasets_by_name, DataType, get_cache_dir

# print availiable dataset names
print(list(datasets_by_name.keys()))

# get the binary (waste or not) dataset
data_dir = '/hdd/data/'
name = 'tacobinary'
cache_dir = get_cache_dir(data_dir, name.lower())
ds = get_dataset_by_name(name, cache_dir)

# print labels and classes
print(ds.labels)
print(ds.num_classes)

# print number of training examples
print(ds.num_examples(DataType.TRAIN))

# or simply print the summary
ds.summary()
```

Debug datasets

```bash
python -m tf_semantic_segmentation.debug.dataset_vis -d ade20k
```

## TFRecords

#### This library simplicifies the process of creating a tfrecord dataset for faster training.

Write tfrecords:

```python
from tf_semantic_segmentation.datasets import TFWriter
ds = ...
writer = TFWriter(record_dir)
writer.write(ds)
writer.validate(ds)
```

or use simple with this script (will be save with size 128 x 128 (width x height)):

```bash
tf-semantic-segmentation-tfrecord-writer -d 'toy' -c /hdd/datasets/ -s '128,128'
```

Analyse already written tfrecord (with mean)

```bash
python -m tf_semantic_segmentation.bin.tfrecord_analyser -r records/ --mean
```

## Docker

```shell
docker build -t tf_semantic_segmentation -f docker/Dockerfile ./
```

or pull the latest release

```shell
docker pull baudcode/tf_semantic_segmentation:latest
```

## Prediction

```shell
pip install matplotlib
```

#### Using Code

```python
from tensorflow.keras.models import load_model
import numpy as np
from tf_semantic_segmentation.processing import dataset
from tf_semantic_segmentation.visualizations import show, masks


model = load_model('logs/model-best.h5', compile=False)

# model parameters
size = tuple(model.input.shape[1:3])
depth = model.input.shape[-1]
color_mode = dataset.ColorMode.GRAY if depth == 1 else dataset.ColorMode.RGB

# define an image
image = np.zeros((256, 256, 3), np.uint8)

# preprocessing
image = image.astype(np.float32) / 255.
image, _ = dataset.resize_and_change_color(image, None, size, color_mode, resize_method='resize')

image_batch = np.expand_dims(image, axis=0)

# predict (returns probabilities)
p = model.predict(image_batch)

# draw segmentation map
num_classes = p.shape[-1] if p.shape[-1] > 1 else 2
predictions_rgb = masks.get_colored_segmentation_mask(p, num_classes, images=image_batch, binary_threshold=0.5)

# show images using matplotlib
show.show_images([predictions_rgb[0], image_batch[0]])
```

#### Using scripts

- On image

```shell
python -m tf_semantic_segmentation.evaluation.predict -m model-best.h5  -i image.png
```

- On TFRecord (data type 'val' is default)

```shell
python -m tf_semantic_segmentation.evaluation.predict -m model-best.h5 -r records/camvid/
```

- On TFRecord (with export to directory)

```shell
python -m tf_semantic_segmentation.evaluation.predict -m model-best.h5 -r records/cubbinary/ -o out/ -rm 'resize_with_pad'
```

- On Video

```shell
python -m tf_semantic_segmentation.evaluation.predict -m model-best.h5 -v video.mp4
```

- On Video (with export to out/p-video.mp4)

```shell
python -m tf_semantic_segmentation.evaluation.predict -m model-best.h5 -v video.mp4 -o out/
```

## Prediction using Tensorflow Model Server

- Installation

```bash
# install
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && apt-get install tensorflow-model-server
```

- Start Model Server

```bash
### using a single model
tensorflow_model_server --rest_api_port=8501 --model_base_path=/home/user/models/mymodel/saved_model

### or using an ensamble of multiple models

# helper to write the ensamble config yaml file (models/ contains multiple logdirs/, logdir must contain the name 'unet')
python -m tf_semantic_segmentation.bin.model_server_config_writer -d models/ -c 'unet'
# start model server with written models.yaml
tensorflow_model_server --model_config_file=models.yaml --rest_api_port=8501
```

### **Compare models and ensemnble**

```bash
python -m tf_semantic_segmentation.evaluation.compare_models -i logs/ -c 'taco' -data /hdd/datasets/ -d 'tacobinary'
```

Parameters:

- _-i_ (directory containing models)
- _-c_ (model name (directory name) must contain this value)
- _-data_ (data directory)
- _-d_ (dataset name)

Use **--help** to get more help

#### Using Code

```python
from tf_semantic_segmentation.serving import predict, predict_on_batch, ensamble_prediction, get_models_from_directory
from tf_semantic_segmentation.processing.dataset import resize_and_change_color

image = np.zeros((128, 128, 3))
image_size = (256, 256)
color_mode = 0  # 0=RGB, 1=GRAY
resize_method = 'resize'
scale_mask = False # only scale mask when model output is scaled using sigmoid activation
num_classes = 3

# preprocess image
image = image.astype(np.float32) / 255.
image, _ = resize_and_change_color(image, None, image_size, color_mode, resize_method='resize')

# prediction on 1 image
p = predict(image.numpy(), host='localhost', port=8501, input_name='input_1', model_name='0')

#############################################################################################################
# if the image size should not match, the color mode does not match or the model_name does not match
# you'll most likely get a `400 Client Error: Bad Request for url: http://localhost:8501/v1/models/0:predict`
# hint: if you only started 1 model try using model_name 'default'
#############################################################################################################

# prediction on batch (for faster prediction of multiple images)
p = predict_on_batch([image], host='localhost', port=8501, input_name='input_1', model_name='0')

# ensamble prediction (average the predictions of multiple models)

# either specify models like this:
models = [
    {
        "name": "0",
        "path": "/home/user/models/mymodel/saved_model/",
        "version": 0, # optional
        "input_name": "input_1"
    },
    {
        "name": "1",
        "path": "/home/user/models/mymodel2/saved_model/",
        "input_name": "input_1"
    }
]


# or load from models in directory (models/) that contain the name 'unet'
models = get_models_from_directory('models/', contains='unet')

# returns the ensamble and all predictions made
ensamble, predictions = ensamble_prediction(models, image.numpy(), host='localhost', port=8501)
```

## TFLite support

#### Convert the model

```shell
python -m tf_semantic_segmentation.bin.convert_tflite -i logs/mymodel/saved_model/0/ -o model.tflite
```

#### Test inference on the model

```shell
python -m tf_semantic_segmentation.debug.tflite_test -m model.tflite -i Harris_Sparrow_0001_116398.jpg
```
