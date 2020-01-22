import argparse
import ast

from ..utils import get_files
from ..threading import parallize_v3
import imageio
import numpy as np
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-cm', '--color_map', type=lambda x: ast.literal_eval(x), required=True)
    args = parser.parse_args()

    files = get_files(args.input_dir, extensions=['jpg', 'png'])
    print("got %d files" % len(files))

    def convert(i, o):
        img = imageio.imread(i)[:, :, :3]
        for k, color in enumerate(args.color_map):
            img[np.asarray(color)] = np.asarray([k, k, k])

        img = img.mean(axis=-1, dtype=np.uint8)
        img[img >= len(args.color_map)] = 0
        imageio.imwrite(o, img)

    for f in files:
        output_path = os.path.join(args.output_dir, os.path.basename(f))
        convert(f, output_path)
        print("done")
