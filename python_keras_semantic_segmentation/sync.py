import argparse
import os
import tqdm

from .utils import get_files
import zipfile

def zip_app(zip_fn, d):
    with zipfile.ZipFile(zip_fn, 'w') as z:
        for fn in get_files(d):
            print('writing %s' % fn)
            z.write(fn, arcname=os.path.relpath(fn, d))

def create_dir(fname):
    drive.CreateFile({'title': fname, "mimeType": "application/vnd.google-apps.folder"})


def upload_file(drive, fn):
    file_drive = drive.CreateFile({'title': fn})
    file_drive.SetContentFile(fn)
    file_drive.Upload()

def upload_file_v2(title, fn):
    import json
    import requests

    key = "AIzaSyBolKpn_sKn-N4OwPt4f6m0nbJbiQWP7lg"
    headers = {
        "Authorization": "Bearer " + key,
        # "Content-Type": "application/zip",
        # "Content-Length": "%d" % os.path.getsize(fn)
    }
    para = {
        "name": "app.zip",
        "parents": ["1Sdk4MHoZTUrFeQL2WM6o-ViZdCQ3i12P"]
    }
    files = {
        'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
        'file': ('application/zip', open(fn, "rb"))
    }
    response = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart", headers=headers, files=files)
    print(response.text)
    assert(response.status_code == 200), "status code ist %d" % response.status_code
    return response

if __name__ == "__main__":

    zip_app("app.zip", "/home/baudcode/Code/python-keras-semantic-segmentation")
    if True:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        #Login to Google Drive and create drive object
        g_login = GoogleAuth()
        g_login.LocalWebserverAuth()
        drive = GoogleDrive(g_login)

        print("zipping app...")
        zip_app("app.zip", "/home/baudcode/Code/python-keras-semantic-segmentation")
        upload_file(drive, 'app.zip')
    else:
        upload_file_v2('app.zip', 'app.zip')
    """
  
    """