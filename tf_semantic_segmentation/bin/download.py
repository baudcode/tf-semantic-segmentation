import argparse
from ..utils import download_and_extract, download_file, download_from_google_drive

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--destination', help='destintation directory', required=True)
    parser.add_argument('-u', '--url', help='url to download', default=None)
    parser.add_argument('-id', '--google_drive_id', help='google drive id to download')
    parser.add_argument('-fn', '--filename', help='google drive name to download', default=None)
    parser.add_argument('-e', '--extract', action='store_true')
    parser.add_argument('-remove', '--remove_archive_on_success', action='store_true')
    args = parser.parse_args()

    if args.url:
        url = args.url
    elif args.google_drive_id and args.filename:
        url = (args.google_drive_id, args.filename)
    else:
        raise Exception("invalid arguments")

    if args.extract:
        download_and_extract(url, args.destination, remove_archive_on_success=args.remove_archive_on_success)
    else:
        if type(url) == tuple:
            download_from_google_drive(url[0], args.destination, url[1])
        else:
            download_file(url, destination_dir=args.destination)
