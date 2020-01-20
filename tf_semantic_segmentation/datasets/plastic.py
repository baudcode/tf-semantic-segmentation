from .dataset import Dataset, DataType
from .utils import get_split
from ..utils import download_file, get_files
from .ms_coco import CocoAnnotationReader
import json
import os
import tqdm


class Taco(Dataset):

    ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"

    """ Super Categories:
    
    {'Scrap metal', 'Plastic utensils', 'Other plastic', 'Lid', 'Shoe', 'Plastic container', 'Bottle cap', 'Blister pack',
     'Aluminium foil', 'Broken glass', 'Carton', 'Can', 'Styrofoam piece', 'Plastic glooves', 'Glass jar', 'Paper bag', 
     'Squeezable tube', 'Paper', 'Rope & strings', 'Bottle', 'Cup', 'Food waste', 
     'Pop tab', 'Straw', 'Cigarette', 'Unlabeled litter', 'Plastic bag & wrapper', 'Battery'}
    """
    MODES = ['supercategory', 'category', 'binary']

    def __init__(self, cache_dir, mode='supercategory'):
        super(Taco, self).__init__(cache_dir)
        self.annotations_file = download_file(self.ANNOTATIONS_URL, self.cache_dir)
        self.ann_reader = CocoAnnotationReader(self.annotations_file)
        self.mode = mode

    @property
    def labels(self):
        return self.ann_reader.get_labels(self.mode)

    def download_files(self, path, dataset_dir):
        with open(path, 'r') as f:
            annotations = json.loads(f.read())

            nr_images = len(annotations['images'])
            for i in tqdm.trange(nr_images):

                image = annotations['images'][i]

                file_name = image['file_name']
                url_original = image['flickr_url']
                url_resized = image['flickr_640_url']

                file_path = os.path.join(dataset_dir, file_name)

                # Create subdir if necessary
                subdir = os.path.dirname(file_path)

                if not os.path.isdir(subdir):
                    os.makedirs(subdir)

                if not os.path.isfile(file_path):
                    # Load and Save Image
                    download_file(url_original, os.path.dirname(file_path), file_name=os.path.basename(file_path))

    def raw(self):

        data_dir = os.path.join(self.cache_dir, 'dataset')
        masks_dir = os.path.join(self.cache_dir, 'masks', self.mode)
        self.download_files(self.annotations_file, data_dir)

        anns = self.ann_reader.read_annotations()

        output_paths = []
        input_paths = get_files(data_dir, extensions=['jpg'])
        for image_path in tqdm.tqdm(input_paths):
            output_paths.append(self.ann_reader.generate_masks(image_path, anns, masks_dir, data_dir=data_dir, mode=self.mode))

        trainset = list(zip(input_paths, output_paths))
        assert(len(trainset) != 0)

        return get_split(trainset)


class TacoBinary(Taco):

    def __init__(self, cache_dir):
        super(TacoBinary, self).__init__(cache_dir, mode='binary')


class TacoCategory(Taco):

    def __init__(self, cache_dir):
        super(TacoCategory, self).__init__(cache_dir, mode='category')


class TacoSuperCategory(Taco):

    def __init__(self, cache_dir):
        super(TacoSuperCategory, self).__init__(cache_dir, mode='supercategory')


if __name__ == "__main__":
    import imageio
    t = Taco('/hdd/datasets/taco', mode='supercategory')
    t.summary()
    #for image, labels in t.get()():
    #    imageio.imwrite("test.png", labels)
    #    break
