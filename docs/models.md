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

```python
from tf_semantic_segmentation import models

# print all available models
print(list(modes.models_by_name.keys()))

# returns a model (without the final activation function)
model = models.get_model_by_name('erfnet', {"input_shape": (128, 128, 3), "num_classes": 5})

# call models directly
model = models.erfnet(input_shape=(128, 128), num_classes=5)
```