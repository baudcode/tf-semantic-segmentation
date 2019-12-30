# https://groups.csail.mit.edu/vision/SUN/
# https://bitbucket.org/vrnowgit/ai/src/master/ai/dataset/sunrgbd.py

# TODO: Add sun rgbd


from .dataset import Dataset
from .utils import get_split, image_generator, DataType
from ..utils import download_and_extract
from ..settings import logger
from .pascal import PascalVOC2012
from ..threading import parallize_v2

import numpy as np
import cv2
import sys
import tqdm
import imageio
import multiprocessing
import os
import json
import copy

MAX_PARALLEL_THREADS = multiprocessing.cpu_count() * 2 + 1


def paint_image_poly(image, labeled_data, colors, with_text=True, overlay=False):

    if overlay:
        img = np.array(image, copy=True, dtype=np.uint8)
    else:
        img = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

    for i, label in enumerate(labeled_data):
        points = labeled_data[label]
        for poly in points:
            cv2.fillConvexPoly(img, poly.reshape(-1, 1, 2), colors[i])
            if with_text:
                centroid = np.mean(poly, axis=0)
                centroid = [centroid, centroid] if type(
                    centroid) is np.float32 or type(centroid) is np.float64 else centroid
                cv2.putText(img, label, (int(centroid[0]), int(centroid[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 0, 0], 2)

    return img


class SunRGBD(Dataset):

    """
    Indoor Scene Dataset from http://rgbd.cs.princeton.edu/
    """

    DATA_URL = "http://rgbd.cs.princeton.edu/data/SUNRGBD.zip"

    def __init__(self, cache_dir, data_path="kv2/kinect2data/"):
        super(SunRGBD, self).__init__(cache_dir)
        self.data_path = data_path

        self.extracted = download_and_extract(self.DATA_URL, self.cache_dir)
        self.frames_path = os.path.join(self.extracted, "SUNRGBD", self.data_path)

        # generate all frames
        self.generate_frames()

    @property
    def frames(self):
        return list(map(lambda x: os.path.join(self.frames_path, x), list(filter(
            lambda x: x != ".DS_Store" and x != "mapping.txt", os.listdir(self.frames_path)))))

    def generate_frames(self):
        mapping_path = os.path.join(self.frames_path, "mapping.txt")
        # Loading / Saving colors of labels
        if os.path.exists(mapping_path):
            logger.debug("Loading mapping from " + mapping_path)
            with open(mapping_path, "r") as handler:
                mapping = json.loads(handler.readline())
        else:
            all_labels = SunRGBD.get_all_labels(self.frames)
            colors = PascalVOC2012.get_colors(len(all_labels))
            mapping = dict(zip(all_labels, colors))
            logger.debug("Saving random mapping to :" + mapping_path)
            with open(mapping_path, "w") as handler:
                handler.write(json.dumps(mapping))

        SunRGBD.generate_images(self.frames, mapping, self.labels, overwrite=False)

    @property
    def labels(self):
        return ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture",
                "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floor_mat", "clothes",
                "ceiling", "books", "fridge", "tv", "paper", "towel", "shower_curtain", "box", "whiteboard", "person",
                "night_stand", "toilet", "sink", "lamp", "bathtub", "bag", "other-structure", "other-furniture",
                "other-prop"]

    @property
    def reduce_labels(self):
        return self.labels

    @staticmethod
    def generate_images(frames, mapping, valid_labels, overwrite=False):

        logger.info("generating labeled images")
        data = list(
            zip(frames, [mapping] * len(frames), [valid_labels] * len(frames), [overwrite] * len(frames)))
        parallize_v2(SunRGBD.save_labeled_frame, data)

    @staticmethod
    def save_labeled_frame(path, mapping, valid_labels, overwrite=True):
        """
        Save labeled frame to basename(path).split(".")[0] + "_L" + basename(path).split(".")[1]
        :param path: path to the original frame
        :param mapping: dict, {label: color} mapping
        :return: None
        """
        rel_image_path = os.path.join(path, "image/")
        image_name = list(filter(lambda x: "_L" not in x,
                                 os.listdir(rel_image_path)))[0]
        image_path = os.path.join(rel_image_path, image_name)
        filename = os.path.basename(image_path).split(".")[0]
        ext = os.path.basename(image_path).split(".")[1]
        labeled_path = os.path.join(os.path.dirname(image_path), str(
            filename) + "_L." + str(ext))

        if overwrite or not os.path.exists(labeled_path):
            image, labeled_data = SunRGBD._get_frame_data(path)

            if image is not None:
                labeled_data = {label: points for label, points in labeled_data.items() if label in valid_labels}
                # colors = [mapping[label] for label in labeled_data]
                colors = [i for i in range(len(labeled_data))]
                labeled_image_data = paint_image_poly(
                    image, labeled_data, colors, with_text=False)
                imageio.imwrite(labeled_path, labeled_image_data)

    @staticmethod
    def get_all_labels(frames):
        labels = set()
        for i, frame in enumerate(frames):
            image, labeled_data = SunRGBD._get_frame_data(frame)
            if labeled_data is not None:
                for key in labeled_data.keys():
                    labels.add(key)

            sys.stdout.write("All Labels - Done [%d of %d] \r" % (i, len(frames)))
            sys.stdout.flush()

        return labels

    def raw(self):

        trainset = []
        frames = copy.deepcopy(self.frames)
        while len(frames) > 0:
            image_dir = os.path.join(frames.pop(), "image")
            images = os.listdir(image_dir)
            if len(images) != 2:
                continue

            if "_L" in images[0]:
                image_labeled = os.path.join(image_dir, images[0])
                image = os.path.join(image_dir, images[1])
            else:
                image_labeled = os.path.join(image_dir, images[1])
                image = os.path.join(image_dir, images[0])

            trainset.append((image, image_labeled))

        return get_split(trainset)

    @staticmethod
    def _get_frame_data(frame_path):

        try:
            annotation_path = os.path.join(
                frame_path, "annotation2Dfinal/index.json")
            with open(annotation_path) as data_file:
                annotation_data = json.load(data_file)

            annotation_range = len(annotation_data["frames"][0]["polygon"])
            labeled_points = {}
            for i in range(annotation_range):
                x = annotation_data["frames"][0]["polygon"][i]["x"]
                y = annotation_data["frames"][0]["polygon"][i]["y"]
                idx_obj = annotation_data["frames"][0]["polygon"][i]["object"]
                if idx_obj >= len(annotation_data['objects']):
                    continue
                label = (annotation_data['objects'][idx_obj]["name"]).lower()
                if type(x) == float or type(x) == int or len(x) == 0:
                    continue
                points = np.transpose(np.array([x, y], dtype=np.int32))

                if label in labeled_points:
                    labeled_points[label].append(points)
                else:
                    labeled_points[label] = [points]

            rgb_path = os.path.join(frame_path, "image/")
            return imageio.imread(os.path.join(rgb_path, os.listdir(rgb_path)[0])), labeled_points

        except ValueError:
            return None, None
        except IndexError:
            print(x, y, idx_obj)
            print("polygon:", annotation_data["frames"][0]["polygon"][i])
            print('objects: ', len(annotation_data['objects']))
        except IOError:
            return None, None


class Kinect2Data(SunRGBD):

    def __init__(self, cache_dir):
        super(Kinect2Data, self).__init__(cache_dir, data_path="kv2/kinect2data/")


if __name__ == "__main__":
    from ..visualizations import show
    import imageio
    import random

    data = Kinect2Data('/hdd/datasets/sun')

    print(data.labels)
    for image, labels in data.get()():
        print(labels.max(), labels.mean())
        show.show_images([image, labels.astype(np.float32)], cols=2)
        break
