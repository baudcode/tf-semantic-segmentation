from tensorflow.python.client import device_lib
from pprint import pprint


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [(x.name, x.memory_limit / 1000. / 1000.) for x in local_device_protos if x.device_type == 'GPU']


if __name__ == "__main__":
    pprint(get_available_gpus())
