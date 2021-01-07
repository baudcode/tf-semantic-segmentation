## Create Dataset for fast training

- The following command will create the dataset `cvc_clinicdb` in `/tmp/data` and the records
in `/tmp/data/cvc_clinicdb/records`.
```shell
python -m tf_semantic_segmentation.bin.tfrecord_writer -d cvc_clinicdb -c '/tmp/data'
```
- For using your own dataset please refer to the [datasets](/datasets) section.


## Training

- Hint: To see train/test/val images you have to start tensorboard like this

```bash
tensorboard --logdir=logs/ --reload_multifile=true
```

#### On inbuild datasets - generator (slow)

```bash
python -m tf_semantic_segmentation.bin.train -ds 'cvc_clinicdb' -bs 8 -e 100 \
    -logdir 'logs/taco-binary-test' -o 'adam' -lr 5e-3 --size 256,256 \
    -l 'binary_crossentropy' -fa 'sigmoid' \
    --train_on_generator --gpus='0' \
    --tensorboard_train_images --tensorboard_val_images
```

#### Using a fixed record path (fast)

```bash
python -m tf_semantic_segmentation.bin.train --record_dir="/tmp/data/cvc_clinicdb/records" \
    -bs 4 -e 100 -logdir 'logs/cvc-adam-1e-4-mish-bs4' -o 'adam' -lr 1e-4 -l 'categorical_crossentropy' \
    -fa 'softmax' -bufsize 50 --metrics='iou_score,f1_score' -m 'unet' --gpus='0' -a 'mish' \
    --tensorboard_train_images --tensorboard_val_images
```

#### Multi GPU training

```bash
python -m tf_semantic_segmentation.bin.train --record_dir="/tmp/data/cvc_clinicdb/records" \
    -bs 4 -e 100 -logdir 'logs/cvc-adam-1e-4-mish-bs4' -o 'adam' -lr 1e-4 -l 'categorical_crossentropy' \
    -fa 'softmax' -bufsize 50 --metrics='iou_score,f1_score' -m 'unet' --gpus='0,1,2,3' -a 'mish'
```

## Using Code

```python
from tf_semantic_segmentation.bin.train import train_test_model, get_args

# get the default args
args = get_args({})

# change some parameters
# !rm -r logs/
args.model = 'unet'
# args['color_mode'] = 0
args.batch_size = 8
args.size = [128, 128] # resize input dataset to this size
args.epochs = 10
args.learning_rate = 1e-4
args.optimizer = 'adam' # ['adam', 'radam', 'ranger']
args.loss = 'dice'
args.logdir = 'logs'
args.record_dir = "/tmp/data/cvc_clinicdb/records"
args.final_activation = 'softmax'

# train and test
results, model = train_test_model(args)
results['evaluate'] # returns last evaluated results using val dataset
results['history'] # returns the history object from model.fit()
```