from .erfnet import erfnet
from .unet import unet, unet_v2

from tensorflow.keras.models import Model

models_by_name = {
    "erfnet": erfnet,
    "unet": unet,
    "unet_v2": unet_v2
}


def get_model_by_name(name, args) -> (Model, Model):
    if name in models_by_name.keys():
        return models_by_name[name](**args)
    else:
        raise Exception("cannot find model %s" % name)


__all__ = ['erfnet', 'unet', "unet_v2", 'get_model_by_name', 'models_by_name']
