import pytz
from datetime import datetime
import os
import sys
import shutil
import subprocess
import urllib.request as urllib
from io import BytesIO
import numpy as np
import PIL
import requests
import tqdm
import multiprocessing

from functools import reduce
from .settings import logger


class ExtractException(Exception):
    pass


def get_image_from_url(url, auth=None):
    req = requests.get(url, auth=auth)
    image = PIL.Image.open(BytesIO(req.content))
    return np.asarray(image)


def get_random_image(width=640, height=480, grayscale=False):
    # https://picsum.photos/g/200/300

    urls = [
        "https://picsum.photos/%s%d/%d/?random" % (
            "g/" if grayscale else "", width, height),
        "https://loremflickr.com/%s%d/%d" % (
            "g/" if grayscale else "", width, height),
        "http://lorempixel.com/%s%d/%d" % (
            "g/" if grayscale else "", width, height)
    ]
    for url in urls:
        try:
            image = get_image_from_url(url)
            return image
        except Exception:
            pass

    raise Exception("No service could be reached")


def get_files(directory, extensions=None):
    files = []
    extensions = [ext.lower() for ext in extensions]
    for root, _, filenames in sorted(os.walk(directory)):
        if extensions:
            files += [os.path.join(root, name) for name in sorted(filenames)
                      if any(name.lower().endswith("." + ext) for ext in extensions)]
        else:
            files += list(map(lambda filename: os.path.join(root,
                                                            filename), filenames))

    return sorted(files)


def extract(archive, destination, silent=False, remove_archive_on_success=True):
    import zipfile
    import gzip

    if os.path.exists(archive):
        if not silent:
            logger.info("[*] extracting " + archive + " to " + destination)
        ret = 0
        if archive.endswith(".zip"):
            ret = extract_zip(archive, destination)

        elif archive.endswith(".rar"):
            ret = unrar(archive, destination)
            if ret != 0:
                raise ExtractException(
                    "[*] could not extract rar, please install unrar")
        elif archive.endswith(".gz") and not archive.endswith("tar.gz"):
            with gzip.open(archive, 'rb') as f_in, open(os.path.join(destination, os.path.basename(archive)[:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            ret = extract_tar(archive, destination, silent=silent)

        if ret != 0:
            raise Exception("[*] could not extract tar %s" % archive)

        if not silent:
            logger.info("[*] removing archive %s" % archive)

        if remove_archive_on_success:
            os.remove(archive)
    else:
        logger.info("[*] already extracted")


def extract_tar(archive, destination, silent=False):
    import tarfile
    try:
        tarfile.TarFile(archive).extractall(destination)
        return 0

    except tarfile.ReadError:
        if not silent:
            logger.info(
                "[*] could extract tar %s via python, trying system call" % archive)

    return call_for_ret_code(["tar", "xfv", archive, "-C", destination], silent=silent)


def extract_zip(archive, destination, silent=False):
    import zipfile
    try:
        zipfile.ZipFile(archive).extractall(destination)
        return 0
    except zipfile.BadZipFile:
        if not silent:
            logger.info("[*] could not extract zip %s via python, trying system call" % archive)

    return call_for_ret_code(['unzip', archive, '-d', destination], silent=silent)


def unrar(archive, destination):

    if call_for_ret_code(["unrar"]) < 0:
        raise ExtractException("unrar not found. Please install unrar.")

    args = ["unrar", "x", archive, "-o", destination]
    return call_for_ret_code(args, silent=False)


def call_for_ret_code(args, silent=False):
    """
        Calls the subprocess and returns the return code
        :param args: list, arguments to fed into subprocess.call
        :param silent: bool, whether to display the ouput of the call
                       in stdout
        :returns int, 1 for failure and 0 for success, -1 for not found
    """
    if not silent:
        print("[+] " + reduce(lambda x, y: str(x) + " " + str(y), args))
    try:
        if silent:
            return subprocess.call(args, stdout=open(os.devnull, 'w'),
                                   stderr=open(os.devnull, 'w'))
        else:
            return subprocess.call(args)
    except IOError:
        return -1
    except OSError:
        return -1


def download_file(url, destination_dir=None, file_name=None, silent=False, auth=None):
    """url lib downloader
    Downloads content of url to destination dir with or without a given file name
    Supports basic authentication and report hooking

    Args:
        url: str, download path
        destination_dir: str, optional if specified file will be downloaded to destination
        file_name : str, optional: file name of the downloaded data
        auth: dict, user authentication {"username": your_user_name, "password": your_password}

    Returns:
        str: file path or None
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if not file_name:
        file_name = url.split('/')[-1]

    if not destination_dir:
        destination = file_name
    else:
        destination = os.path.join(destination_dir, file_name)

    if os.path.exists(destination):
        return destination

    if not silent:
        print('[*] downloading ' + url + " to " + destination)

    if auth and "username" in auth and "password" in auth:
        pass_manager = urllib.HTTPPasswordMgrWithDefaultRealm()
        pass_manager.add_password(
            None, url, auth["username"], auth["password"])
        urllib.install_opener(urllib.build_opener(
            urllib.HTTPBasicAuthHandler(pass_manager)))

    try:
        response = urllib.urlopen(url)
        _chunk_read(response, destination,
                    reporthook=None if silent else _download_reporthook)
    except urllib.URLError as error:
        if not silent:
            print("[*] error downloading", url, ":", error.reason)
        return None

    return destination


def _download_reporthook(bytes_count, block_size, total_size):
    if (bytes_count // block_size) % 10 == 0:
        sys.stdout.write('[*] downloaded %02.02f/%02.02f MB \r' % (
            bytes_count / 1000.0 / 1000.0,
            total_size / 1000.0 / 1000.0))
        sys.stdout.flush()
    if bytes_count >= total_size and total_size > 0.0:
        sys.stdout.write('[*] download finished \n')
        sys.stdout.flush()


def _chunk_read(response, filename, chunk_size=8192, reporthook=None):
    try:
        content_length = response.info().getheader('Content-Length')
    except AttributeError:
        content_length = response.headers['Content-Length']  # python3 fix
    if content_length is not None:
        total_size = content_length.strip()
        total_size = int(total_size)
    else:
        total_size = -1
    bytes_so_far = 0

    with open(filename, 'wb') as handler:
        with tqdm.tqdm(total=total_size, unit='B', mininterval=1, unit_scale=True) as tq:
            while 1:
                chunk = response.read(chunk_size)
                handler.write(chunk)
                bytes_so_far += len(chunk)
                if not chunk:
                    break

                tq.update(len(chunk))

                # if reporthook:
                #    reporthook(bytes_so_far, chunk_size, total_size)

    return bytes_so_far


def download_from_google_drive(drive_id, destination_dir, filename, block_size=32 * 1024, silent=False):
    import requests
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': drive_id}, stream=True)
    token = _google_drive_get_confirm_token(response)

    if token:
        params = {'id': drive_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    destination = os.path.join(destination_dir, filename)
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for count, chunk in enumerate(response.iter_content(block_size)):
            if not silent:
                _download_reporthook(count * block_size,
                                     block_size, total_size)
            if chunk:
                f.write(chunk)
    return destination


def _google_drive_get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def download_and_extract(url, destination, chk_exists=True, overwrite=False, silent=False, auth=None, file_name=None, cache_dir='/tmp/extracted'):
    """ Download and extract rar, zip and tar

    Args:
        url: str, download dataset url or {name, drive_id} for google drive download
        destination: str, directory path to extract the dataset to
        chk_exists: bool, check whether destination exists and download and extract if it does not
        silent: bool, do not print anything during downloading if true
        file_name: file name of the downloaded file (f.e. filename too long)
        auth: dict, user authentification {"username": your_user_name, "password": your_password}

    Returns:
        str: folder where the dataset is extracted to
    """
    if not os.path.exists(destination) or not chk_exists or overwrite:
        if not os.path.exists(destination):
            os.makedirs(destination)

        if type(url) == str:
            archive = download_file(
                url, destination_dir=cache_dir, file_name=file_name, silent=silent, auth=auth)
        else:
            name, drive_id = url
            archive = download_from_google_drive(
                drive_id=drive_id, destination_dir=cache_dir, filename=name, silent=silent)

        extract(archive, destination)

    return destination


def get_now_timestamp(mode="postgis"):
    now = datetime.now()
    return format_datetime(now, mode=mode)


def get_now_datetime():
    return datetime.now().replace(tzinfo=pytz.utc)


def format_datetime(d, mode="postgis"):
    if mode == 'postgis':
        return d.replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    if mode == 'filename':
        return d.replace(tzinfo=pytz.utc).strftime("%Y-%m-%d_%H-%M-%S")


class DownloadQueue:

    def __init__(self):
        self.files = []

    def add(self, url, local_path):
        self.files.append((url, os.path.dirname(local_path), os.path.basename(local_path)))
        return self

    def run(self):
        n_processes = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=n_processes) as pool:
            results = [r for r in tqdm.tqdm(pool.imap(download_file, self.files), desc='downloading', total=len(self.files))]
        return results
