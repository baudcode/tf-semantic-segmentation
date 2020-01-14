# Requirements

```
sudo apt-get install libsm6 libxext6 libxrender-dev
```

# Training

### Using on the the inbuild datasets (generator)

```bash
python -m python_keras_semantic_segmentation.bin.train -ds 'tacobinary' -bs 8 -e 100 \
    -logdir 'logs/taco-binary-test' -o 'ranger' -lr 5e-3 --size 256,256 \
    -l 'binary_crossentropy' -fa 'sigmoid' \
    --train_on_generator
```

### Using a fixed record path

```bash
python -m tf_semantic_segmentation.bin.train --record_dir=/hdd/datasets/cityscapes/records/cityscapes-512x256-rgb/ \
    -bs 4 -e 100 -logdir 'logs/cityscapes-bs8-e100-512x256' -o 'ranger' -lr 1e-4 -l 'categorical_crossentropy' \
    -fa 'softmax' -bufsize 50 --metrics='iou_score,f1_score' -m 'erfnet' --gpus='0' -a 'mish'
```

# Models

- Erfnet
- Unet

```python
from tf_semantic_segmentation import models

# print all available models
print(list(modes.models_by_name.keys()))

# returns a model without the final activation function
# because the activation function depends on the loss function
model = models.get_model_by_name('erfnet')
```

# Datasets

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

# TFRecords

This library simplicifies the process of creating a tfrecord dataset for faster training.

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

### Prediction UI

```
# install
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update && apt-get install tensorflow-model-server

# start
tensorflow_model_server --rest_api_port=8501 --model_base_path=/home/baudcode/Code/python-keras-semantic-segmentation/logs/taco_binary_erfnet_256x256_bs_8_rgb_ranger_lr_5e-3-e100-ce_label_smoothing/saved_model/

# start
pip install streamlit
python setup.py install && streamlit run tf_semantic_segmentation/eval/viewer.py
```
