import argparse

from ..datasets import download_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', required=True)
    parser.add_argument('-r', '--record_dir', required=True)
    args = parser.parse_args()

    download_records(args.tag, args.record_dir)


if __name__ == "__main__":
    main()
