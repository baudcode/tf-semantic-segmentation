from .erfnet import erfnet
from .unet import unet, unet_v2
from .multires_unet import multires_unet

from tensorflow.keras.models import Model
import inspect

models_by_name = {
    "erfnet": erfnet,
    "unet": unet,
    "unet_v2": unet_v2,
    "multires_unet": multires_unet
}


def get_model_description(name):
    return inspect.getdoc(models_by_name[name])


def get_model_by_name(name, args) -> (Model, Model):
    if name in models_by_name.keys():
        return models_by_name[name](**args)
    else:
        raise Exception("cannot find model %s" % name)


__all__ = ['erfnet', 'unet', "unet_v2", 'multires_unet', 'get_model_by_name', 'models_by_name']
