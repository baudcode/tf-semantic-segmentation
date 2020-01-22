from .vast import parse_query
from types import SimpleNamespace
import json
import requests
import os

try:
    from urllib import quote_plus  # Python 2.X
except ImportError:
    from urllib.parse import quote_plus


def display_table(rows, fields):
    header = [name for _, name, _, _, _ in fields]
    out_rows = [header]
    lengths = [len(x) for x in header]
    for instance in rows:
        row = []
        out_rows.append(row)
        for key, name, fmt, conv, _ in fields:
            conv = conv or (lambda x: x)
            val = instance.get(key, None)
            if val is None:
                s = "-"
            else:
                val = conv(val)
                s = fmt.format(val)
            idx = len(row)
            lengths[idx] = max(len(s), lengths[idx])
            row.append(s)
    for row in out_rows:
        out = []
        for l, s, f in zip(lengths, row, fields):
            _, _, _, _, ljust = f
            if ljust:
                s = s.ljust(l)
            else:
                s = s.rjust(l)
            out.append(s)
        print("  ".join(out))


displayable_fields = (
    ("id", "ID", "{}", None, True),
    ("machine_id", "Machine ID", "{}", None, True),
    ("cuda_max_good", "CUDA", "{:0.1f}", None, True),
    ("gpu_ram", "VRAM", "{:0.1f}", None, True),
    ("num_gpus", "Num", "{} x", None, False),
    ("gpu_name", "Model", "{}", None, True),
    ("pcie_bw", "PCIE BW", "{:0.1f}", None, True),
    ("cpu_cores_effective", "vCPUs", "{:0.1f}", None, True),
    ("cpu_ram", "RAM", "{:0.1f}", lambda x: x / 1000, False),
    ("disk_space", "Storage", "{:.0f}", None, True),
    ("dph_total", "$/hr", "{:0.4f}", None, True),
    ("dlperf", "DLPerf", "{:0.1f}", None, True),
    ("dlperf_per_dphtotal", "DLP/$", "{:0.1f}", None, True),
    ("inet_up", "Net up", "{:0.1f}", None, True),
    ("inet_down", "Net down", "{:0.1f}", None, True),
    ("reliability2", "R", "{:0.1f}", lambda x: x * 100, True),
    ("duration", "Max Days", "{:0.1f}", lambda x: x / (24.0 * 60.0 * 60.0), True),
)

instance_fields = (
    ("id", "ID", "{}", None, True),
    ("machine_id", "Machine", "{}", None, True),
    ("actual_status", "Status", "{}", None, True),
    ("num_gpus", "Num", "{} x", None, False),
    ("gpu_name", "Model", "{}", None, True),
    ("gpu_util", "Util. %", "{:0.1f}", None, True),
    ("cpu_cores_effective", "vCPUs", "{:0.1f}", None, True),
    ("cpu_ram", "RAM", "{:0.1f}", lambda x: x / 1000, False),
    ("disk_space", "Storage", "{:.0f}", None, True),
    ("ssh_host", "SSH Addr", "{}", None, True),
    ("ssh_port", "SSH Port", "{}", None, True),
    ("dph_total", "$/hr", "{:0.4f}", None, True),
    ("image_uuid", "Image", "{}", None, True),

    #("dlperf",              "DLPerf",   "{:0.1f}",  None, True),
    #("dlperf_per_dphtotal", "DLP/$",    "{:0.1f}",  None, True),
    ("inet_up", "Net up", "{:0.1f}", None, True),
    ("inet_down", "Net down", "{:0.1f}", None, True),
    ("reliability2", "R", "{:0.1f}", lambda x: x * 100, True),
    #("duration",            "Max Days", "{:0.1f}",  lambda x: x/(24.0*60.0*60.0), True),
)


class RequestHandler:

    def __init__(self, api_key_path="~/.vast_api_key", server_url_default="https://vast.ai/api/v0"):
        self.api_key_path = os.path.expanduser(api_key_path)
        self.server_url_default = server_url_default

        if os.path.exists(self.api_key_path):
            with open(self.api_key_path, "r") as reader:
                self.api_key = reader.read().strip()
        else:
            self.api_key = None
            print("warning: api key not found in file %s" % self.api_key_path)

    def _build_url(self, subpath, query_args=None):
        if query_args is None:
            query_args = {}

        query_args["api_key"] = self.api_key
        if query_args:
            #a_list      = [<expression> for <l-expression> in <expression>]
            '''
            vector result;
            for (l_expression: expression) {
                result.push_back(expression);
            }
            '''
            #an_iterator = (<expression> for <l-expression> in <expression>)
            print(query_args.items())
            return self.server_url_default + subpath + "?" + "&".join(
                "{x}={y}".format(x=x, y=quote_plus(y if isinstance(y, str) else json.dumps(y))) for x, y in query_args.items())
        else:
            return self.server_url_default + subpath

    def put(self, subpath, data, query_args=None):
        return self.request(subpath, method='put', data=data, query_args=query_args)

    def delete(self, subpath, data={}, query_args=None):
        return self.request(subpath, method='delete', data=data, query_args=query_args)

    def get(self, subpath, query_args=None):
        return self.request(subpath, method='get', data={}, query_args=query_args)

    def _handle_request(self, r):
        r.raise_for_status()
        result = {'r': r, 'status_code': r.status_code, 'msg': None, 'success': False, 'json': None}

        if (r.status_code == 200):
            rj = r.json()
            result['json'] = rj
            if 'msg' in rj:
                result['msg'] = rj['msg']

            if 'success' in rj:
                result['success'] = rj['success']
            else:
                result['success'] = True

        return result

    def request(self, subpath, method='get', data={}, query_args=None):
        url = self._build_url(subpath, query_args)

        if method == 'put':
            r = requests.put(url, json=data)
            return self._handle_request(r)

        elif method == 'delete':
            r = requests.delete(url, json=data)
            return self._handle_request(r)
        elif method == 'get':
            r = requests.get(url)
            return self._handle_request(r)
        else:
            r = requests.post(url, json=data)
            return self._handle_request(r)


def start_instance(id):
    suburl = "/instances/{id}/".format(id=id)
    r = RequestHandler().put(suburl, {
        "state": "running"
    })
    if r['success']:
        print("starting instance {id}.".format(**(locals())))
    else:
        print('no success {msg}, status_code: {status_code}'.format(status_code=r['status_code'], msg=['msg']))


def search_offsers(query, disable_bundling=False, query_type='on-demand', query_order='score-', raw=False):
    """
    argument("-t", "--type", default="on-demand", help="whether to show `interruptible` or `on-demand` offers. default: on-demand"),
    argument("-i", "--interruptible", dest="type", const="interruptible", action="store_const", help="Alias for --type=interruptible"),
    argument("-d", "--on-demand", dest="type", const="on-demand", action="store_const", help="Alias for --type=on-demand"),
    argument("-n", "--no-default", action="store_true", help="Disable default query"),
    argument("--disable-bundling", action="store_true", help="Show identical offers. This request is more heavily rate limited."),
    argument("--storage", type=float, default=5.0, help="amount of storage to use for pricing, in GiB. default=5.0GiB"),
    argument("-o", "--order", type=str, help="comma-separated list of fields to sort on. postfix field with - to sort desc. ex: -o 'num_gpus,total_flops-'.  default='score-'", default='score-'),
    argument("query",            help="Query to search for. default: 'external=false rentable=true verified=true', pass -n to ignore default", nargs="*", default=None),
    """
    field_alias = {
        "cuda_vers": "cuda_max_good",
        "reliability": "reliability2",
        "dlperf_usd": "dlperf_per_dphtotal",
        "dph": "dph_total",
        "flops_usd": "flops_per_dphtotal"
    }

    default_query = {"verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True}}
    query = parse_query(query, default_query)

    order = []
    for name in query_order.split(","):
        name = name.strip()
        if not name:
            continue
        direction = "asc"
        if name.strip("-") != name:
            direction = "desc"
        field = name.strip("-")
        if field in field_alias:
            field = field_alias[field]
        order.append([field, direction])

    query["order"] = order
    query["type"] = query_type
    if disable_bundling:
        query["disable_bundling"] = True

    r = RequestHandler().get("/bundles", {"q": query})
    rows = r['json']["offers"]

    if raw:
        print(json.dumps(rows, indent=1, sort_keys=True))
    else:
        display_table(rows, displayable_fields)

    return r['json']["offers"]


def destroy_instance(id):
    suburl = "/instances/{id}/".format(id=id)
    r = RequestHandler().delete(suburl)

    if r['success']:
        print("deleting instance {id}.".format(**(locals())))
    else:
        print('no success {msg}, status_code: {status_code}'.format(status_code=r['status_code'], msg=['msg']))


def remove_defjob(id):

    suburl = "/machines/{machine_id}/defjob/".format(machine_id=id)
    success, msg, r = RequestHandler().delete(suburl)

    if success:
        print("default instance for machine {machine_id} removed.".format(machine_id=id))
    else:
        print('no success {msg}, status_code: {r.status_code}'.format(**locals()))


def show_instances(raw=False):
    r = RequestHandler().get('/instances', {'owner': 'me'})
    rows = r['json']['instances']
    if raw:
        print(json.dumps(rows, indent=1, sort_keys=True))
    else:
        display_table(rows, instance_fields)
    return rows


def show_machines(raw=False, quiet=False):

    r = RequestHandler().get('/machines', {'owner': 'me'})
    rows = r['json']['machines']

    if raw:
        print(json.dumps(rows, indent=1, sort_keys=True))
    else:
        for machine in rows:
            if quiet:
                print("{id}".format(id=machine["id"]))
            else:
                print("{N} machines: ".format(N=len(rows)))
                print("{id}: {json}".format(id=machine["id"], json=json.dumps(machine, indent=4, sort_keys=True)))


def unlist_machine(id):
    r = RequestHandler().delete("/machines/{machine_id}/asks/".format(machine_id=id))

    if r['success']:
        print("all offers for machine {machine_id} removed, machine delisted.".format(machine_id=id))
    else:
        print("failed {msg} with error {status_code}".format(msg=r['msg'], status_code=r['status_code']))


def create_bid(id, price_gpu, price_inetu, price_inetd, image, args={}):
    """
        argument("id",            help="id of machine to launch default instance on", type=int),
        argument("--price_gpu",   help="per gpu rental price in $/hour", type=float),
        argument("--price_inetu", help="price for internet upload bandwidth in $/GB", type=float),
        argument("--price_inetd", help="price for internet download bandwidth in $/GB", type=float),
        argument("--image",       help="docker container image to launch", type=str),
        args: arguments passed to container
    """
    #req_url = args.url + "/machines/create_asks/?user_id=" + str(args.user);
    suburl = "/machines/create_bids/"

    #print("PUT " + req_url);
    payload = {'machine': id, 'price_gpu': price_gpu,
               'price_inetu': price_inetu, 'price_inetd': price_inetd,
               'image': image, 'args': args}

    r = RequestHandler().put(suburl, payload)

    if r["success"]:
        print("bids created for machine {id},  @ ${price_gpu}/gpu/day, ${price_inetu}/GB up, ${price_inetd}/GB down".format(**locals()))
    else:
        print("failed {msg} with error {status_code}".format(msg=r['msg'], status_code=r['status_code']))


def list_machine(machine_id, price_gpu, price_inetu, price_inetd, price_disk, args={}):
    suburl = "/machines/create_asks/"
    payload = {'machine': machine_id, 'price_gpu': price_gpu, 'price_disk': price_disk, 'price_inetu': price_inetu, 'price_inetd': price_inetd}
    #print("PUT " + req_url);
    r = RequestHandler().put(suburl, payload)
    if r['success']:
        price_gpu_ = str(price_gpu) if price_gpu is not None else "def"
        price_inetu_ = str(price_inetu)
        price_inetd_ = str(price_inetd)
        print("offers created for machine {machine_id},  @ ${price_gpu_}/gpu/day, ${price_inetu_}/GB up, ${price_inetd_}/GB down".format(**locals()))
    else:
        print('failed msg: {msg}, status_code: {status_code}'.format(status_code=r['status_code'], msg=r['msg']))


def label_instance(id, label):
    suburl = "/instances/{id}/".format(id=id)
    r = RequestHandler().put(suburl, {"label": label})

    if r["success"]:
        print("label for {id} set to {label}.".format(**(locals())))
    else:
        print(r["msg"])


def stop_instance(id):
    suburl = "/instances/{id}/".format(id=id)
    r = RequestHandler().put(suburl, {
        "state": "stopped"
    })

    if r['success']:
        print("stopped instance with id {id}".format(id=id))
    else:
        print('no success {msg}, status_code: {status_code}'.format(status_code=r['status_code'], msg=r['msg']))


def print_search_options():

    for field in displayable_fields:
        print(field[0], field[1])

    for field in instance_fields:
        print(field[0], field[1])


def create_instance(id, price, image, label=None, disk=20, onstart_cmd=None, runtype='ssh', args=None, extra='==SUPPRESS==', lang_utf8=False,
                    python_utf8=False, jupyter=False, jupyter_dir=None, jupyter_lab=False, create_from=None, force=False, raw=False):
    """
    argument("id",            help="id of instance type to launch", type=int),
    argument("--price",       help="per machine bid price in $/hour", type=float),
    argument("--disk",        help="size of local disk partition in GB", type=float, default=10),
    argument("--image",       help="docker container image to launch", type=str),
    argument("--label",       help="label to set on the instance", type=str),
    argument("--onstart",     help="filename to use as onstart script", type=str),
    argument("--onstart-cmd", help="contents of onstart script as single argument", type=str),
    argument("--jupyter",     help="Launch as a jupyter instance instead of an ssh instance.", action="store_true"),
    argument("--jupyter-dir", help="For runtype 'jupyter', directory in instance to use to launch jupyter. Defaults to image's working directory.", type=str),
    argument("--jupyter-lab", help="For runtype 'jupyter', directory in instance to use to launch jupyter. Defaults to image's working directory.", action="store_true"),
    argument("--lang-utf8",   help="Workaround for images with locale problems: install and generate locales before instance launch, and set locale to C.UTF-8.", action="store_true"),
    argument("--python-utf8", help="Workaround for images with locale problems: set python's locale to C.UTF-8.", action="store_true"),
    argument("--extra",       help=argparse.SUPPRESS),
    argument("--args",        nargs=argparse.REMAINDER, help="DEPRECATED: list of arguments passed to container launch. Onstart is recommended for this purpose."),
    argument("--create-from", help="Existing instance id to use as basis for new instance. Instance configuration should usually be identical, as only the difference from the base image is copied.", type=str),
    argument("--force",       help="Skip sanity checks when creating from an existing instance", action="store_true"),
    usage = "vast create instance id [OPTIONS] [--args ...]",
    """
    if jupyter_lab or jupyter_dir:
        jupyter = True

    if jupyter and runtype == 'args':
        raise ValueError("Error: Can't use jupyter and args together")

    if jupyter:
        runtype = 'jupyter'
    suburl = "/asks/{id}/".format(id=id)
    payload = {
        "client_id": "me",
        "image": image,
        "args": args,
        "price": price,
        "disk": disk,
        "label": label,
        "extra": extra,
        "onstart": onstart_cmd,
        "runtype": runtype,
        "python_utf8": python_utf8,
        "lang_utf8": lang_utf8,
        "use_jupyter_lab": jupyter_lab,
        "jupyter_dir": jupyter_dir,
        "create_from": create_from,
        "force": force
    }
    r = RequestHandler().put(suburl, payload)

    if raw:
        print(json.dumps(r['json'], indent=1))
    else:
        print("Started. {}".format(r['json']))

    return r['json']


if __name__ == "__main__":
    # print_search_options()

    if False:
        offsers = search_offsers('cpu_cores > 10, num_gpus = 4, cuda_max_good = 10.0, cpu_ram >= 32, gpu_ram >= 8, dph_total < 0.30', raw=False)
        offser_id = offsers[0]['id']
        price = offsers[0]['dph_total'] + 0.2
        image = 'baudcode/tf_semantic_segmentation:nightly-tf-2.0'
        cmd = 'python3 -m tf_semantic_segmentation.bin.tfrecord_download -t camvid -r /hdd/datasets/downloaded/camvid'
        disk = 20

        """
        print("machines:")
        show_machines()
        print("instances: ")
        show_instances()
        """
        instance = create_instance(offser_id, price, image, label='camvid', disk=disk, onstart_cmd=cmd)
        instance_id = instance['new_contract']

        instances = show_instances()
        for instance in instances:
            print("ssh -p {ssh_port} root@{ssh_host}".format(**instance))
    else:
        instances = show_instances(raw=True)
        for instance in instances:
            # destroy_instance(instance['id'])
            print("ssh -p {ssh_port} root@{ssh_host}".format(**instance))

    # stop_instance(453691)
    # destroy_instance(453691)
