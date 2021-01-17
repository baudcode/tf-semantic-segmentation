# datasets
from . import camvid
from . import cityscapes
from . import mots_challenge
from . import mapping_challenge
from . import ms_coco
from . import pascal
from . import taco
from . import shapes
from . import sun
from . import toy
from . import ade20k
from . import isic
from . import cvc_clinicdb
from . import cub
from . import bioimage

from .directory import DirectoryDataset
from .utils import DataType, get_split, get_split_from_list, download_records, google_drive_records_by_tag
from .tfrecord import TFReader, TFWriter
from .dataset import Dataset
import os

# kitty, sun, pascal-voc, ade20k

datasets_by_name = {
    # "ade20k": ade20k.Ade20k,
    "sun": sun.Kinect2Data,
    "camvid": camvid.CamSeq01,
    "coco2014": ms_coco.Coco2014,
    "coco2017": ms_coco.Coco2017,
    "cityscapes": cityscapes.Cityscapes,
    "tacobinary": taco.TacoBinary,
    "tacocategory": taco.TacoCategory,
    "tacosupercategory": taco.TacoSuperCategory,
    "mots": mots_challenge.MotsChallenge,
    "pascalvoc2012": pascal.PascalVOC2012,
    # dummy datasets for testing
    "shapes": shapes.ShapesDS,
    "shapesmini": shapes.ShapesDSMini,
    "toy": toy.Toy,
    "mappingchallenge": mapping_challenge.MappingChallenge,
    "ade20k": ade20k.Ade20k,
    "isic2018": isic.ISIC2018,
    "cvc_clinicdb": cvc_clinicdb.CVCClinicDB,
    'cub2002011binary': cub.CUB2002011Binary,
    'cub2002011category': cub.CUB2002011Category,
    'bioimagebenchmark': bioimage.BioimageBenchmark
}


def get_cache_dir(data_dir: str, name: str) -> str:
    if "taco" in name.lower():
        cache_dir = os.path.join(data_dir, 'taco')
    elif "cub2002011" in name.lower():
        cache_dir = os.path.join(data_dir, 'cub2002011')
    elif 'isic' in name.lower():
        cache_dir = os.path.join(data_dir, 'isic')
    elif 'coco' in name.lower():
        cache_dir = os.path.join(data_dir, 'coco')
    else:
        cache_dir = os.path.join(data_dir, name.lower())

    return cache_dir


def get_dataset_by_name(name: str, cache_dir: str, args: dict = {}) -> Dataset:
    if name in datasets_by_name.keys():
        return datasets_by_name[name](cache_dir, **args)
    else:
        raise Exception("could not find dataset %s" % name)


__all__ = ["get_dataset_by_name", "DataType", "download_records", "google_drive_records_by_tag", "TFWriter", "TFReader",
           "get_split", "get_split_from_list", "get_cache_dir", "datasets_by_name", 'DirectoryDataset']
