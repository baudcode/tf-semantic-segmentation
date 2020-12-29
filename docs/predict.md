#### Preparation

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

#### Scripts

- On image

```shell
python -m tf_semantic_segmentation.evaluation.predict -m model-best.h5  -i image.png
```

- On TFRecord (data type 'val' is default)

```python
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

## Tensorflow Model Server

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

### Compare Models and ensemble


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