from ..datasets import datasets_by_name, get_dataset_by_name, get_cache_dir
from ..datasets.tfrecord import TFWriter

import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=list(datasets_by_name.keys()), required=True)
    parser.add_argument('-c', '--data_dir', required=True)
    parser.add_argument('-r', '--record_dir', default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    cache_dir = get_cache_dir(args.data_dir, args.dataset.lower())
    ds = get_dataset_by_name(args.dataset, cache_dir)

    # write records
    if args.record_dir:
        record_dir = args.record_dir
    else:
        record_dir = os.path.join(cache_dir, 'records', args.dataset.lower())

    print('wrting records to %s' % record_dir)
    writer = TFWriter(record_dir)
    writer.write(ds, overwrite=args.overwrite)

    # validate number of examples written
    writer.validate(ds)
