from .erfnet import erfnet
from .unet import unet

models_by_name = {
    "erfnet": erfnet,
    "unet": unet
}


def get_model_by_name(name, args):
    if name in models_by_name:
        return models_by_name[name](**args)
    else:
        raise Exception("cannot find model %s" % name)


__all__ = ['erfnet', 'unet', 'get_model_by_name', 'models_by_name']
