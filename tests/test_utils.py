from tf_semantic_segmentation import utils
import tempfile
import shutil
import os
import pytest
import datetime
import numpy as np
import time

test_zip_url = "https://www.7-zip.org/a/7za920.zip"
test_tar_url = "http://deb.debian.org/debian/pool/non-free/u/unrar-nonfree/unrar-nonfree_5.2.7.orig.tar.gz"
test_rar_url = "https://www.philippwinterberg.com/download/example.rar"
test_image_url = 'https://homepages.cae.wisc.edu/~ece533/images/airplane.png'


def test_call_for_ret_code():
    code = utils.call_for_ret_code(['date'])
    assert(code == 0)

    code = utils.call_for_ret_code(['date --ver'])
    assert(code == -1)


def test_download_and_extract():
    cache_dir = tempfile.mkdtemp()
    destination = tempfile.mkdtemp()

    utils.download_and_extract(test_zip_url, destination, chk_exists=False, cache_dir=cache_dir)
    assert(os.path.exists(os.path.join(destination, 'license.txt')))
    assert(not os.path.exists(os.path.join(cache_dir, '7za920.zip')))

    utils.download_and_extract(test_zip_url, destination, file_name='a.zip', overwrite=True,
                               remove_archive_on_success=False, cache_dir=cache_dir)
    assert(os.path.exists(os.path.join(cache_dir, 'a.zip')))
    assert(os.path.exists(os.path.join(destination, 'license.txt')))

    shutil.rmtree(cache_dir)
    shutil.rmtree(destination)


@pytest.fixture()
def zip_file():
    cache_dir = tempfile.mkdtemp()
    yield utils.download_file(test_zip_url, cache_dir)
    shutil.rmtree(cache_dir)


@pytest.fixture()
def tar_file():
    cache_dir = tempfile.mkdtemp()
    yield utils.download_file(test_tar_url, cache_dir)
    shutil.rmtree(cache_dir)


@pytest.fixture()
def rar_file():
    cache_dir = tempfile.mkdtemp()
    yield utils.download_file(test_rar_url, cache_dir)
    shutil.rmtree(cache_dir)


def test_extract_zip(zip_file):
    destination = tempfile.mkdtemp()
    utils.extract(zip_file, destination, remove_archive_on_success=False)
    assert(os.path.exists(os.path.join(destination, 'license.txt')))
    shutil.rmtree(destination)

    destination = tempfile.mkdtemp()
    utils.extract_zip(zip_file, destination)
    assert(os.path.exists(os.path.join(destination, 'license.txt')))
    shutil.rmtree(destination)


def test_extract_tar(tar_file):
    print(tar_file)
    print(os.path.exists(tar_file))
    destination = tempfile.mkdtemp()
    utils.extract(tar_file, destination, remove_archive_on_success=False)
    assert(os.path.exists(os.path.join(destination, 'unrar', 'license.txt')))
    shutil.rmtree(destination)

    destination = tempfile.mkdtemp()
    utils.extract_tar(tar_file, destination)
    assert(os.path.exists(os.path.join(destination, 'unrar', 'license.txt')))
    shutil.rmtree(destination)


def test_extract_rar(rar_file):
    destination = tempfile.mkdtemp()
    utils.extract(rar_file, destination, remove_archive_on_success=False)
    assert(os.path.exists(os.path.join(destination, 'Fifteen_Feet_of_Time.pdf')))
    shutil.rmtree(destination)

    destination = tempfile.mkdtemp()
    utils.unrar(rar_file, destination)
    assert(os.path.exists(os.path.join(destination, 'Fifteen_Feet_of_Time.pdf')))
    shutil.rmtree(destination)


def test_download_from_google_drive():
    destination = tempfile.mkdtemp()
    url = 'https://drive.google.com/file/d/1f0tlM7oRZu5AsK52mQnAKESPugQA7iCQ/view?usp=sharing'
    drive_id = url.split("/")[-2]
    dst = utils.download_from_google_drive(drive_id, destination, 'LICENSE')
    assert(os.path.exists(dst) and os.path.getsize(dst) > 0)


def test_datetime():
    now = utils.get_now_datetime()
    ts = utils.get_now_timestamp()
    ts2 = utils.format_datetime(now)
    assert(datetime.datetime.strptime(ts2, "%Y-%m-%d %H:%M:%S").today == now.today)
    assert(isinstance(datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"), datetime.datetime))


def test_get_files(zip_file):
    cache_dir = tempfile.mkdtemp()
    utils.extract(zip_file, cache_dir, remove_archive_on_success=False)
    files = utils.get_files(cache_dir)
    basenames = list(map(os.path.basename, files))
    assert(len(files) == 4)
    assert(all([bn in basenames for bn in ['7-zip.chm', '7za.exe', 'license.txt', 'readme.txt']]))

    files = utils.get_files(cache_dir, extensions=['txt'])
    basenames = list(map(os.path.basename, files))
    assert(len(files) == 2)
    assert(all([bn in basenames for bn in ['license.txt', 'readme.txt']]))

    files = utils.get_files(cache_dir, extensions=['txt', 'exe'])
    basenames = list(map(os.path.basename, files))
    assert(len(files) == 3)
    assert(all([bn in basenames for bn in ['7za.exe', 'license.txt', 'readme.txt']]))

    shutil.rmtree(cache_dir)


def test_get_image_from_url():
    image = utils.get_image_from_url(test_image_url)
    assert(image.shape == (512, 512, 3))


def test_get_random_image():
    image = utils.get_random_image(width=200, height=300)
    assert(image.shape == (300, 200, 3))

    image = utils.get_random_image(width=200, height=300, grayscale=True)
    assert(image.shape == (300, 200))


def test_kill_and_start_tensorboard():
    print("kill start...")
    thread = utils.kill_start_tensorboard('./', port=6006)
    print(thread.getName())
    time.sleep(5.0)
    print("kill process by port")
    utils.kill(6006)
    thread.join()

    print("kill start")
    thread = utils.kill_start_tensorboard('./', port=6006)
    time.sleep(5.0)
    utils.kill(6006)
    thread.join()


def test_ndarray_to_base64():
    arr = np.zeros((64, 64, 3), np.uint8)
    b64 = utils.ndarray_to_base64(arr)
    assert(type(b64) == str)

    arr = np.zeros((64, 64, 3), np.float32)
    try:
        b64 = utils.ndarray_to_base64(arr)
    except TypeError:
        pass
