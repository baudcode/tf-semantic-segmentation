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

### Using Dataset (tacobinary)

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

### Debug Datasets

#### Visualize
```bash
python -m tf_semantic_segmentation.debug.dataset_vis -d ade20k -c '/tmp/data'
```

#### Analyse

```bash
python -m tf_semantic_segmentation.bin.tfrecord_analyser -r records/ --mean
```

### Create TFRecord

#### Using Code

```python
from tf_semantic_segmentation.datasets import TFWriter
ds = ...
writer = TFWriter(record_dir)
writer.write(ds)
writer.validate(ds)
```

#### Inbuild dataset

```shell
INPUT_DIR = ...
tf-semantic-segmentation-tfrecord-writer -dir $INPUT_DIR -r $INPUT_DIR/records
```


#### From Custom Dataset

```shell
INPUT_DIR = ...
tf-semantic-segmentation-tfrecord-writer -dir $INPUT_DIR -r $INPUT_DIR/records
```

There are the following addition arguments:

- -s [--size] '$width,$height' (f.e. "512,512")
- -rm [--resize_method] ('resize', 'resize_with_pad', 'resize_with_crop_or_pad)
- cm [--color_mode] (0=RGB, 1=GRAY, 2=NONE (default))



### Use your own dataset

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

#### Create TFRecord

```shell
INPUT_DIR = ...
tf-semantic-segmentation-tfrecord-writer -dir $INPUT_DIR -r $INPUT_DIR/records
```

There are the following addition arguments:

- -s [--size] '$width,$height' (f.e. "512,512")
- -rm [--resize_method] ('resize', 'resize_with_pad', 'resize_with_crop_or_pad)
- cm [--color_mode] (0=RGB, 1=GRAY, 2=NONE (default))