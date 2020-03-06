from ..serving import write_model_config_from_models_dir
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--models_dir', help='path to the directory containing the logdirs/models')
    parser.add_argument('-c', '--contains', help='logdir/model_name must contain the given sequence', default=None)
    parser.add_argument('-o', '--config_path', default='models.yaml')
    args = parser.parse_args()

    write_model_config_from_models_dir(args.models_dir, args.contains, args.config_path)


if __name__ == "__main__":
    main()
