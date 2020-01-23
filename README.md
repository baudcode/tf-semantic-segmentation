# TF Semantic Segmentation

[![Build Status](https://travis-ci.com/baudcode/tf-semantic-segmentation.svg?branch=master)](https://travis-ci.com/baudcode/tf-semantic-segmentation)
[![PyPI Status Badge](https://badge.fury.io/py/tf-semantic-segmentation.svg)](https://pypi.org/project/tf-semantic-segmentation/)
[![codecov](https://codecov.io/gh/baudcode/tf-semantic-segmentation/branch/dev/graph/badge.svg)](https://codecov.io/gh/baudcode/tf-semantic-segmentation)
[![latest tag](https://img.shields.io/github/v/tag/baudcode/tf-semantic-segmentation)]()

## Features

- Datasets

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

- Distributed Training on Multiple GPUs
- Hyper Parameter Optimization using WandB
- WandB Integration
- Easily create TFRecord from Directory
- Tensorboard visualizations

- Models:

  - Unet
  - Erfnet

- Losses:

  - Catagorical Crossentropy
  - Binary Crossentropy
  - Crossentropy + SSIM
  - Dice
  - Crossentropy + Dice
  - Tversky
  - Focal
  - Focal + Tversky

- Activations:

  - mish
  - swish
  - relu6

- Optimizers:

  - Ranger
  - RAdam

- Normalization

  - Instance Norm
  - Batch Norm

- On the fly Augmentations

  - flip left/right
  - flip up/down
  - rot 180
  - color

## Requirements

```shell
sudo apt-get install libsm6 libxext6 libxrender-dev libyaml-dev libpython3-dev
```

#### Tensorflow (2.x) & Tensorflow Addons (optional)

```shell
pip install tensorflow-gpu==2.1.0 --upgrade
pip install tensorflow-addons==0.7.0 --upgrade
```

## Training

### Using on the the inbuild datasets (generator)

```bash
python -m tf_semantic_segmentation.bin.train -ds 'tacobinary' -bs 8 -e 100 \
    -logdir 'logs/taco-binary-test' -o 'adam' -lr 5e-3 --size 256,256 \
    -l 'binary_crossentropy' -fa 'sigmoid' \
    --train_on_generator --gpus='0'
```

### Using a fixed record path

```bash
python -m tf_semantic_segmentation.bin.train --record_dir=/hdd/datasets/cityscapes/records/cityscapes-512x256-rgb/ \
    -bs 4 -e 100 -logdir 'logs/cityscapes-bs8-e100-512x256' -o 'adam' -lr 1e-4 -l 'categorical_crossentropy' \
    -fa 'softmax' -bufsize 50 --metrics='iou_score,f1_score' -m 'erfnet' --gpus='0' -a 'mish'
```

### Multi GPU training

```bash
python -m tf_semantic_segmentation.bin.train --record_dir=/hdd/datasets/cityscapes/records/cityscapes-512x256-rgb/ \
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

# returns a model without the final activation function
# because the activation function depends on the loss function
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

## Docker

```shell
docker build -t tf_semantic_segmentation -f docker/Dockerfile ./
```

## Prediction UI

```
# install
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && apt-get install tensorflow-model-server

# start
tensorflow_model_server --rest_api_port=8501 --model_base_path=/home/user/logs/taco_binary_erfnet_256x256_bs_8_rgb_ranger_lr_5e-3-e100-ce_label_smoothing/saved_model/

# start
pip install streamlit
python setup.py install && streamlit run tf_semantic_segmentation/eval/viewer.py
```
