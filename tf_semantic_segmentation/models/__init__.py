from .erfnet import erfnet
from .unet import unet
from .imagenet_unet import unet_mobilenet, unet_inception_resnet_v2, unet_resnet
from .satellite_unet import satellite_unet
from .multires_unet import multires_unet
from .attention_unet import attention_unet
from .nested_unet import nested_unet
from .psp import psp
from .fcn import fcn
from .u2net import u2net, u2netp

from tensorflow.keras.models import Model
import inspect

models_by_name = {
    "erfnet": erfnet,
    "unet": unet,
    "unet_mobilenet": unet_mobilenet,
    "unet_inception_resnet_v2": unet_inception_resnet_v2,
    "unet_resnet": unet_resnet,
    "satellite_unet": satellite_unet,
    "multires_unet": multires_unet,
    "attention_unet": attention_unet,
    "nested_unet": nested_unet,
    "psp": psp,
    "fcn": fcn,
    "u2net": u2net,
    "u2netp": u2netp,
}


def get_model_description(name):
    return inspect.getdoc(models_by_name[name])


def get_model_by_name(name, args) -> Model:
    if name in models_by_name.keys():
        return models_by_name[name](**args)
    else:
        raise Exception("cannot find model %s" % name)


__all__ = ['erfnet', 'unet', 'multires_unet', "unet_mobilenet", "unet_inception_resnet_v2", "unet_resnet", "satellite_unet",
           "attention_unet", "nested_unet",
           'get_model_by_name', 'models_by_name']
