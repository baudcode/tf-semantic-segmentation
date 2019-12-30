from .experimental import cmp_facade
from .experimental import ade20k
from . import mots_challenge
from . import pascal
from . import sun
from . import camvid
from . import shapes
from . import toy
from . import mapping_challenge
from . import ms_coco
from . import plastic

from .utils import DataType, get_split, get_split_from_list
from .dataset import Dataset
import os

# kitty, sun, pascal-voc, ade20k

datasets_by_name = {
    # "cmp": cmp_facade.CMP,
    # "ade20k": ade20k.Ade20k,
    "sun": sun.Kinect2Data,
    "camvid": camvid.CamSeq01,
    "coco2014": ms_coco.Coco2014,
    "coco2017": ms_coco.Coco2017,
    "tacobinary": plastic.TacoBinary,
    "tacocategory": plastic.TacoCategory,
    "tacosupercategory": plastic.TacoSuperCategory,
    "mots": mots_challenge.MotsChallenge,
    "pascal": pascal.PascalVOC2012,
    # dummy datasets for testing
    "shapes": shapes.ShapesDS,
    "toy": toy.Toy,
    "mappingchallenge": mapping_challenge.MappingChallenge
}


def get_cache_dir(data_dir, name):
    if "taco" in name.lower():
        cache_dir = os.path.join(data_dir, 'taco')
    elif 'coco' in name.lower():
        cache_dir = os.path.join(data_dir, 'coco')
    else:
        cache_dir = os.path.join(data_dir, name.lower())

    return cache_dir


def get_dataset_by_name(name, cache_dir) -> Dataset:
    if name in datasets_by_name.keys():
        return datasets_by_name[name](cache_dir)
    else:
        raise Exception("could not find dataset %s" % name)


__all__ = ["get_dataset_by_name", "DataType",
           "get_split", "get_split_from_list", "datasets_by_name"]
