[![Build Status](https://travis-ci.org/baudcode/tf-semantic-segmentation.svg?branch=master)](https://travis-ci.org/baudcode/tf-semantic-segmentation)
[![PyPI Status Badge](https://badge.fury.io/py/tf-semantic-segmentation.svg)](https://pypi.org/project/tf-semantic-segmentation/)
[![codecov](https://codecov.io/gh/baudcode/tf-semantic-segmentation/branch/dev/graph/badge.svg)](https://codecov.io/gh/baudcode/tf-semantic-segmentation)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xBH4WxhJ7TlnC7pck7ifLjo3NrdYmug-)

## Requirements

```shell
sudo apt-get install libsm6 libxext6 libxrender-dev libyaml-dev libpython3-dev
```

#### Tensorflow (2.x) & Tensorflow Addons (optional)

```shell
pip install tensorflow-gpu==2.4.0 --upgrade
pip install tensorflow-addons==0.12.0 --upgrade
```

## Installation

```shell
pip install tf-semantic-segmentation
```

## Features

- Fast and easy training/prediction on multiple datasets
- Distributed Training on Multiple GPUs
- Hyper Parameter Optimization using WandB
- WandB Integration
- Easily create TFRecord from Directory
- Tensorboard visualizations
- Ensemble inference


### Datasets

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



### Models

- [U2Net / U2NetP](https://arxiv.org/abs/2005.09007)
- [Unet](https://arxiv.org/abs/1505.04597)
- [PSP](https://arxiv.org/abs/1612.01105)
- [FCN](https://arxiv.org/abs/1411.4038)
- [Erfnet](https://arxiv.org/abs/1806.08522)
- [MultiResUnet](https://arxiv.org/abs/1902.04049)
- [NestedUnet (Unet++)](https://arxiv.org/abs/1807.10165)
- SatelliteUnet
- MobilenetUnet (unet with mobilenet encoder pre-trained on imagenet)
- InceptionResnetV2Unet (unet with inception-resnet v2 encoder pre-trained on imagenet)
- ResnetUnet (unet with resnet50 encoder pre-trained on imagenet)
- AttentionUnet

### Losses

- Catagorical Crossentropy
- Binary Crossentropy
- Crossentropy + SSIM
- Dice
- Crossentropy + Dice
- Tversky
- Focal
- Focal + Tversky

### Metrics

- f1
- f2
- iou
- precision
- recall
- psnr
- ssim

### Activations:

  - mish
  - swish
  - relu6

### Optimizers:

  - Ranger
  - RAdam

### Normalization

  - Instance Norm
  - Batch Norm

### Augmentations

  - flip left/right
  - flip up/down
  - rot 180
  - color