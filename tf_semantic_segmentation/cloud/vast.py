# taken from https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py

from __future__ import unicode_literals, print_function

import re
import json
import sys
import argparse
import os
import requests
import getpass



try:
    from urllib import quote_plus  # Python 2.X
except ImportError:
    from urllib.parse import quote_plus  # Python 3+


try:
    JSONDecodeError = json.JSONDecodeError
except AttributeError:
    JSONDecodeError = ValueError

try:
    input = raw_input
except NameError:
    pass

server_url_default = "https://vast.ai/api/v0"
api_key_file_base = "~/.vast_api_key"
api_key_file = os.path.expanduser(api_key_file_base)
api_key_guard = object()

class argument(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class hidden_aliases(object):
    # just a bit of a hack
    def __init__(self, l):
        self.l = l
    def __iter__(self):
        return iter(self.l)
    def __bool__(self):
        return False
    def __nonzero__(self):
        return False
    def append(self,x):
        self.l.append(x)

class apwrap(object):
    def __init__(self, *args, **kwargs):
        kwargs["formatter_class"] = argparse.RawDescriptionHelpFormatter
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.parser.set_defaults(func=self.fail_with_help)
        self.subparsers_ = None
        self.subparser_objs = []
        self.added_help_cmd = False
        self.post_setup = []
        self.verbs = set()
        self.objs = set()

    def fail_with_help(self, *a, **kw):
        self.parser.print_help(sys.stderr)
        raise SystemExit

    def add_argument(self, *a, **kw):
        if not kw.get("parent_only"):
            for x in self.subparser_objs:
                try:
                    x.add_argument(*a, **kw)
                except argparse.ArgumentError:
                    # duplicate - or maybe other things, hopefully not
                    pass
        return self.parser.add_argument(*a, **kw)

    def subparsers(self, *a, **kw):
        if self.subparsers_ is None:
            kw["metavar"] = "command"
            kw["help"] = "command to run. one of:"
            self.subparsers_ = self.parser.add_subparsers(*a, **kw)
        return self.subparsers_

    def get_name(self, verb, obj):
        if obj:
            self.verbs.add(verb)
            self.objs.add(obj)
            name = verb + ' ' + obj
        else:
            self.objs.add(verb)
            name = verb
        return name

    def command(self, *arguments, aliases=(), help=None, **kwargs):
        help_ = help
        if not self.added_help_cmd:
            self.added_help_cmd = True

            @self.command(argument("subcommand", default=None, nargs="?"), help="print this help message")
            def help(*a, **kw):
                self.fail_with_help()

        def inner(func):
            dashed_name = func.__name__.replace("_", "-")
            verb, _, obj = dashed_name.partition("--")
            name = self.get_name(verb, obj)
            aliases_transformed = [] if aliases else hidden_aliases([])
            for x in aliases:
                verb, _, obj = x.partition(" ")
                aliases_transformed.append(self.get_name(verb, obj))
            kwargs["formatter_class"] =argparse.RawDescriptionHelpFormatter
            sp = self.subparsers().add_parser(name, aliases=aliases_transformed, help=help_, **kwargs)
            self.subparser_objs.append(sp)
            for arg in arguments:
                sp.add_argument(*arg.args, **arg.kwargs)
            sp.set_defaults(func=func)
            return func
        if len(arguments) == 1 and type(arguments[0]) != argument:
            func = arguments[0]
            arguments = []
            return inner(func)
        return inner

    def parse_args(self, argv=None, *a, **kw):
        if argv is None:
            argv = sys.argv[1:]
        argv_ = []
        for x in argv:
            if argv_ and argv_[-1] in self.verbs:
                argv_[-1] += " " + x
            else:
                argv_.append(x)
        args = self.parser.parse_args(argv_, *a, **kw)
        for func in self.post_setup:
            func(args)
        return args

parser = apwrap()

def apiurl(args, subpath, query_args=None):
    if query_args is None:
        query_args = {}
    if args.api_key is not None:
        query_args["api_key"] = args.api_key
    if query_args:
        #a_list      = [<expression> for <l-expression> in <expression>]
        '''
        vector result;
        for (l_expression: expression) {
            result.push_back(expression);
        }
        '''
        #an_iterator = (<expression> for <l-expression> in <expression>)
        return args.url + subpath + "?" + "&".join(
                "{x}={y}".format(x=x, y=quote_plus(y if isinstance(y, str) else json.dumps(y))) for x, y in query_args.items())
    else:
        return args.url + subpath

def deindent(message):
    """
    deindent a quoted string
    """
    message = re.sub(r" *$", "", message, flags=re.MULTILINE)
    indents = [len(x) for x in re.findall("^ *(?=[^ ])", message, re.MULTILINE) if len(x)]
    a = min(indents)
    message = re.sub(r"^ {," + str(a) + "}", "", message, flags=re.MULTILINE)
    return message.strip()

displayable_fields = (
    ("id",                  "ID",       "{}",       None, True),
    ("cuda_max_good",       "CUDA",     "{:0.1f}",  None, True),
    ("num_gpus",            "Num",      "{} x",     None, False),
    ("gpu_name",            "Model",    "{}",       None, True),
    ("pcie_bw",             "PCIE BW",  "{:0.1f}",  None, True),
    ("cpu_cores_effective", "vCPUs",    "{:0.1f}",  None, True),
    ("cpu_ram",             "RAM",      "{:0.1f}",  lambda x: x/1000, False),
    ("disk_space",          "Storage",  "{:.0f}",     None, True),
    ("dph_total",           "$/hr",     "{:0.4f}",  None, True),
    ("dlperf",              "DLPerf",   "{:0.1f}",   None, True),
    ("dlperf_per_dphtotal", "DLP/$",    "{:0.1f}",   None, True),
    ("inet_up",             "Net up",   "{:0.1f}",   None, True),
    ("inet_down",           "Net down", "{:0.1f}",   None, True),
    ("reliability2",        "R",        "{:0.1f}",   lambda x: x * 100, True),
    ("duration",            "Max Days", "{:0.1f}",   lambda x: x/(24.0*60.0*60.0), True),
)

instance_fields = (
    ("id",                  "ID",       "{}",       None, True),
    ("machine_id",          "Machine",  "{}",       None, True),
    ("actual_status",       "Status",   "{}",       None, True),
    ("num_gpus",            "Num",      "{} x",     None, False),
    ("gpu_name",            "Model",    "{}",       None, True),
    ("gpu_util",            "Util. %",  "{:0.1f}",  None, True),
    ("cpu_cores_effective", "vCPUs",    "{:0.1f}",  None, True),
    ("cpu_ram",             "RAM",      "{:0.1f}",  lambda x: x/1000, False),
    ("disk_space",          "Storage",  "{:.0f}",     None, True),
    ("ssh_host",            "SSH Addr", "{}",       None, True),
    ("ssh_port",            "SSH Port", "{}",       None, True),
    ("dph_total",           "$/hr",     "{:0.4f}",  None, True),
    ("image_uuid",          "Image",    "{}",       None, True),

    #("dlperf",              "DLPerf",   "{:0.1f}",  None, True),
    #("dlperf_per_dphtotal", "DLP/$",    "{:0.1f}",  None, True),
    ("inet_up",             "Net up",   "{:0.1f}",  None, True),
    ("inet_down",           "Net down", "{:0.1f}",  None, True),
    ("reliability2",        "R",        "{:0.1f}",  lambda x: x * 100, True),
    #("duration",            "Max Days", "{:0.1f}",  lambda x: x/(24.0*60.0*60.0), True),
)

def parse_query(query_str, res=None):
    if res is None: res = {}
    if type(query_str) == list:
        query_str = " ".join(query_str)
        
    query_str = query_str.strip()
    opts = re.findall("([a-zA-Z0-9_]+)( *[=><!]+| +(?:[lg]te?|nin|neq|eq|not ?eq|not ?in|in) )?( *)(\[[^\]]+\]|[^ ]+)?( *)", query_str)
    #res = {}
    op_names = {
        ">=": "gte",
        ">": "gt",
        "gt": "gt",
        "gte": "gte",
        "<=": "lte",
        "<": "lt",
        "lt": "lt",
        "lte": "lte",
        "!=": "neq",
        "==": "eq",
        "=": "eq",
        "eq": "eq",
        "neq": "neq",
        "noteq": "neq",
        "not eq": "neq",
        "notin": "notin",
        "not in": "notin",
        "nin": "notin",
        "in": "in",
    }
    
    field_alias = {
        "cuda_vers":        "cuda_max_good",
        "display_active":   "gpu_display_active",
        "reliability":      "reliability2",
        "dlperf_usd":       "dlperf_per_dphtotal",
        "dph":              "dph_total",
        "flops_usd":        "flops_per_dphtotal",
    }
    
    field_multiplier = {
        "cpu_ram"   : 1000,
        "duration"  : 1.0 / (24.0*60.0*60.0),
    }
    
    fields = {
        "compute_cap",
        "cpu_cores",
        "cpu_cores_effective",
        "cpu_ram",
        "cuda_max_good",
        "disk_bw",
        "disk_space",
        "dlperf",
        "dlperf_per_dphtotal"
        "dph_total",
        "duration",
        "external",
        "flops_per_dphtotal",
        "gpu_display_active",
        #"gpu_ram_free_min",
        "gpu_mem_bw",
        "gpu_name",
        "gpu_ram",
        "has_avx",
        "host_id",
        "id",
        "inet_down",
        "inet_down_cost",
        "inet_up",
        "inet_up_cost",
        "min_bid",
        "mobo_name",
        "num_gpus",
        "pci_gen",
        "pcie_bw",
        "reliability2",
        "rentable",
        "rented",
        "storage_cost",
        "total_flops",
        "verified"
    }
    
    joined = "".join("".join(x) for x in opts)
    if joined != query_str:
        raise ValueError("Unconsumed text. Did you forget to quote your query? " + repr(joined) + " != " + repr(query_str))
    for field, op, _, value, _ in opts:
        value = value.strip(",[]")
        v = res.setdefault(field, {})
        op = op.strip()
        op_name = op_names.get(op)
        
        if field in field_alias:
            field = field_alias[field];
            
        
        if not field in fields:
            print("Warning: Unrecognized field: {}, see list of recognized fields.".format(field), file=sys.stderr);
        if not op_name:
            raise ValueError("Unknown operator. Did you forget to quote your query? " + repr(op).strip("u"))
        if op_name in ["in", "notin"]:
            value = [x.strip() for x in value.split(",") if x.strip()]
        if not value:
            raise ValueError("Value cannot be blank. Did you forget to quote your query? " + repr((field, op, value)))
        if not field:
            raise ValueError("Field cannot be blank. Did you forget to quote your query? " + repr((field, op, value)))
        if value in ["?", "*", "any"]:
            if op_name != "eq":
                raise ValueError("Wildcard only makes sense with equals.")
            if field in v:
                del v[field]
            if field in res:
                del res[field]
            continue

        if field in field_multiplier:
            value = str(float(value) * field_multiplier[field]);

        v[op_name] = value
        res[field] = v;
    return res

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

@parser.command(
    argument("-t", "--type", default="on-demand", help="whether to show `interruptible` or `on-demand` offers. default: on-demand"),
    argument("-i", "--interruptible", dest="type", const="interruptible", action="store_const", help="Alias for --type=interruptible"),
    argument("-d", "--on-demand", dest="type", const="on-demand", action="store_const", help="Alias for --type=on-demand"),
    argument("-n", "--no-default", action="store_true", help="Disable default query"),
    argument("--disable-bundling", action="store_true", help="Show identical offers. This request is more heavily rate limited."),
    argument("--storage", type=float, default=5.0, help="amount of storage to use for pricing, in GiB. default=5.0GiB"),
    argument("-o", "--order", type=str, help="comma-separated list of fields to sort on. postfix field with - to sort desc. ex: -o 'num_gpus,total_flops-'.  default='score-'", default='score-'),
    argument("query",            help="Query to search for. default: 'external=false rentable=true verified=true', pass -n to ignore default", nargs="*", default=None),
    usage="vast search offers [--help] [--api-key API_KEY] [--raw] <query>",
    epilog=deindent("""
        Query syntax:
        
            query = comparison comparison...
            comparison = field op value
            field = <name of a field>
            op = one of: <, <=, ==, !=, >=, >, in, notin
            value = <bool, int, float, etc> | 'any'
        
        note: to pass '>' and '<' on the command line, make sure to use quotes

           
        Examples:
        
            ./vast search offers 'compute_cap > 610 total_flops < 5'
            ./vast search offers 'reliability > 0.99  num_gpus>=4' -o 'num_gpus-'
            ./vast search offers 'rentable = any'
       
        Available fields:
            
              Name                  Type       Description

            compute_cap:            int       cuda compute capability*100  (ie:  650 for 6.5, 700 for 7.0)
            cpu_cores:              int       # virtual cpus
            cpu_cores_effective:    float     # virtual cpus you get
            cpu_ram:                float     system RAM in gigabytes
            cuda_vers:              float     cuda version
            disk_bw:                float     disk read bandwidth, in MB/s
            disk_space:             float     disk storage space, in GB
            dlperf:                 float     DL-perf score  (see FAQ for explanation)
            dlperf_usd:             float     DL-perf/$
            dph:                    float     $/hour rental cost
            duration:               float     max rental duration in days
            external:               bool      show external offers
            flops_usd:              float     TFLOPs/$
            gpu_mem_bw:             float     GPU memory bandwidth in GB/s
            gpu_ram:                float     GPU RAM in GB
            gpu_frac:               float     Ratio of GPUs in the offer to gpus in the system
            has_avx:                bool      CPU supports AVX instruction set.
            id:                     int       instance unique ID
            inet_down:              float     internet download speed in Mb/s
            inet_down_cost:         float     internet download bandwidth cost in $/GB
            inet_up:                float     internet upload speed in Mb/s
            inet_up_cost:           float     internet upload bandwidth cost in $/GB
            min_bid:                float     current minimum bid price in $/hr for interruptible
            num_gpus:               int       # of GPUs
            pci_gen:                float     PCIE generation
            pcie_bw:                float     PCIE bandwidth (CPU to GPU)
            reliability:            float     machine reliability score (see FAQ for explanation)
            rentable:               bool      is the instance currently rentable
            rented:                 bool      is the instance currently rented
            storage_cost:           float     storage cost in $/GB/month
            total_flops:            float     total TFLOPs from all GPUs
            verified:               bool      is the machine verified
    """),
    aliases=hidden_aliases(["search instances"]),
)
def search__offers(args):
    field_alias = {
        "cuda_vers"     : "cuda_max_good",
        "reliability"   : "reliability2",
        "dlperf_usd"    : "dlperf_per_dphtotal",
        "dph"           : "dph_total",
        "flops_usd"     : "flops_per_dphtotal",
    };

    try:

        if args.no_default:
            query = {}
        else:
            query = { "verified":{"eq":True}, "external":{"eq":False}, "rentable":{"eq":True} }
    
        if args.query is not None:
            query = parse_query(args.query, query)
        #print("query length: {}".format(len(query)));
        #for k,q in query.items():
            #print("{} {}".format(k, q));
        order = []
        for name in args.order.split(","):
            name = name.strip()
            if not name: continue
            direction = "asc"
            if name.strip("-") != name:
                direction = "desc"
            field = name.strip("-");
            if field in field_alias:
                field = field_alias[field];
            order.append([field, direction])

        query["order"] = order
        query["type"]  = args.type
        if args.disable_bundling:
            query["disable_bundling"] = True
    except ValueError as e:
        print("Error: ", e)
        return 1
    
    url = apiurl(args, "/bundles", {"q":query});
    #url = apiurl(args, "/bundles") + "?q=" + quote_plus(json.dumps(query));
    r = requests.get(url);
    r.raise_for_status()
    rows = r.json()["offers"]
    if args.raw:
        print(json.dumps(rows, indent=1, sort_keys=True))
    else:
        #print(url);
        #print("{N} instances types: ".format(N=len(rows)) );
        display_table(rows, displayable_fields)


@parser.command(
    usage="vast show instances [--api-key API_KEY] [--raw]",
)
def show__instances(args):
    req_url = apiurl(args, "/instances", {"owner": "me"});
    r = requests.get(req_url);
    r.raise_for_status()
    rows = r.json()["instances"]
    if args.raw:
        print(json.dumps(rows, indent=1, sort_keys=True))
    else:
        display_table(rows, instance_fields)
        #print("{N} instances: ".format(N=len(rows)) );
        #print("%-10s%-10s%-12s%-5s%-14s%-7s%-7s%-8s%-10s%-14s%-10s%-8s%-12s" % ("Instance", "Machine", "Status", "#", "GPUs", "util%", "vCPUs", "RAM", "Storage", "SSH Addr", "SSH Port", "$/hr", "Image"));
        #for instance in rows:
        #    gpu_util = 0;
        #    if (instance["gpu_util"] is not None): 
        #        gpu_util = int(instance["gpu_util"]);
        #    cost = str(float(instance["dph_total"]));
        #    print("%-10s%-10s%-12s%-2s x %-16s%-6i%-5i%3iGB   %5iGB   %-16s%-9s%-8s%-12s" % (instance["id"], instance["machine_id"], instance["actual_status"], 1*instance["num_gpus"], instance["gpu_name"], gpu_util, int(instance["cpu_cores"]), int(instance["cpu_ram"])/1000, int(instance["disk_space"]), instance["ssh_host"], instance["ssh_port"], cost[0:5], instance["image_uuid"]));
        #    #print("{id}: {json}".format(id=instance["id"], json=json.dumps(instance, indent=4, sort_keys=True)))
    

@parser.command(
    argument("-q", "--quiet", action="store_true", help="only display numeric ids"),
    usage = "vast show machines [OPTIONS]",
)
def show__machines(args):
    req_url = apiurl(args, "/machines", {"owner": "me"});
    r = requests.get(req_url);
    r.raise_for_status()
    rows = r.json()["machines"]
    if args.raw:
        print(json.dumps(rows, indent=1, sort_keys=True))
    else:
        for machine in rows:
            if args.quiet:
                print("{id}".format(id=machine["id"]))
            else:
                print("{N} machines: ".format(N=len(rows)) );
                print("{id}: {json}".format(id=machine["id"], json=json.dumps(machine, indent=4, sort_keys=True)))


@parser.command(
    argument("id",                  help="id of machine to list", type=int),
    argument("-g", "--price_gpu",   help="per gpu rental price in $/hour  (price for active instances)", type=float),
    argument("-s", "--price_disk",  help="storage price in $/GB/month (price for inactive instances), default: $0.15/GB/month", type=float),
    argument("-u", "--price_inetu", help="price for internet upload bandwidth in $/GB", type=float),
    argument("-d", "--price_inetd", help="price for internet download bandwidth in $/GB", type=float),
    usage = "vast list machine id [--price_gpu PRICE_GPU] [--price_inetu PRICE_INETU] [--price_inetd PRICE_INETD] [--api-key API_KEY]",
)
def list__machine(args):
    #req_url = args.url + "/machines/create_asks/?user_id=" + str(args.user);
    req_url = apiurl(args, "/machines/create_asks/");

    #print("PUT " + req_url);
    r = requests.put(req_url, json = {'machine':args.id, 'price_gpu':args.price_gpu, 'price_disk':args.price_disk, 'price_inetu':args.price_inetu, 'price_inetd':args.price_inetd } );
    
    if (r.status_code == 200) :
        #print(r.text);
        rj = r.json();
        if (rj["success"]) :
            price_gpu_   = str(args.price_gpu) if args.price_gpu is not None else "def";
            price_inetu_ = str(args.price_inetu);
            price_inetd_ = str(args.price_inetd);
            print("offers created for machine {args.id},  @ ${price_gpu_}/gpu/day, ${price_inetu_}/GB up, ${price_inetd_}/GB down".format(**locals()));
        else :
            print(rj["msg"]);
    else :
        print(r.text);
        print("failed with error {r.status_code}".format(**locals()));


@parser.command(
    argument("id",          help="id of machine to list", type=int),
    usage = "vast unlist machine <id>",
)
def unlist__machine(args):
    req_url = apiurl(args, "/machines/{machine_id}/asks/".format(machine_id = args.id));
    #req_url = args.url + "/machines/{machine_id}/asks/".format(machine_id = args.id);
    #print(req_url);
    r = requests.delete(req_url);
    
    if (r.status_code == 200) :
        #print(r.text);
        rj = r.json();
        if (rj["success"]) :
            print("all offers for machine {machine_id} removed, machine delisted.".format(machine_id = args.id));
        else :
            print(rj["msg"]);
    else :
        print(r.text);
        print("failed with error {r.status_code}".format(**locals()));


@parser.command(
    argument("id",          help="id of machine to remove default instance from", type=int),
)
def remove__defjob(args):

    req_url = apiurl(args, "/machines/{machine_id}/defjob/".format(machine_id = args.id));
    #print(req_url);
    r = requests.delete(req_url);
    
    if (r.status_code == 200) :
        #print(r.text);
        rj = r.json();
        if (rj["success"]) :
            print("default instance for machine {machine_id} removed.".format(machine_id = args.id));
        else :
            print(rj["msg"]);
    else :
        print(r.text);
        print("failed with error {r.status_code}".format(**locals()));





def set_ask(args):
    print("set asks!\n");


@parser.command(
    argument("id",            help="id of instance to start/restart", type=int),
    usage = "vast start instance <id> [--raw]",
)
def start__instance(args):
    url = apiurl(args, "/instances/{id}/".format(id=args.id))
    r = requests.put(url, json={
        "state": "running"
    })
    r.raise_for_status()

    if (r.status_code == 200) :
        rj = r.json();
        if (rj["success"]) :
            print("starting instance {args.id}.".format(**(locals())) );
        else :
            print(rj["msg"]);
    else :
        print(r.text);
        print("failed with error {r.status_code}".format(**locals()));


@parser.command(
    argument("id",            help="id of instance to stop", type=int),
    usage = "vast stop instance [--raw] <id>",
)
def stop__instance(args):
    url = apiurl(args, "/instances/{id}/".format(id=args.id))
    r = requests.put(url, json={
        "state": "stopped"
    })
    r.raise_for_status()

    if (r.status_code == 200) :
        rj = r.json();
        if (rj["success"]) :
            print("stopping instance {args.id}.".format(**(locals())) );
        else :
            print(rj["msg"]);
    else :
        print(r.text);
        print("failed with error {r.status_code}".format(**locals()));

@parser.command(
    argument("id",            help="id of instance to label", type=int),
    argument("label",         help="label to set", type=str),
    usage = "vast label instance <id> <label>",
)
def label__instance(args):
    url = apiurl(args, "/instances/{id}/".format(id=args.id))
    r = requests.put(url, json={
        "label": args.label
    })
    r.raise_for_status()

    rj = r.json();
    if rj["success"]:
        print("label for {args.id} set to {args.label}.".format(**(locals())) );
    else :
        print(rj["msg"]);


@parser.command(
    argument("id",            help="id of instance to delete", type=int),
    usage="vast destroy instance id [-h] [--api-key API_KEY] [--raw]"
)
def destroy__instance(args):
    url = apiurl(args, "/instances/{id}/".format(id=args.id))
    r = requests.delete(url, json={})
    r.raise_for_status()
    

    if (r.status_code == 200) :
        rj = r.json();
        if (rj["success"]) :
            print("destroying instance {args.id}.".format(**(locals())) );
        else :
            print(rj["msg"]);
    else :
        print(r.text);
        print("failed with error {r.status_code}".format(**locals()));


        
@parser.command(
    argument("id",            help="id of machine to launch default instance on", type=int),
    argument("--price_gpu",   help="per gpu rental price in $/hour", type=float),
    argument("--price_inetu", help="price for internet upload bandwidth in $/GB", type=float),
    argument("--price_inetd", help="price for internet download bandwidth in $/GB", type=float),
    argument("--image",       help="docker container image to launch", type=str),
    argument("--args",        nargs=argparse.REMAINDER, help="list of arguments passed to container launch"),
    usage="vast set defjob id [--api-key API_KEY] [--price_gpu PRICE_GPU] [--price_inetu PRICE_INETU] [--price_inetd PRICE_INETD] [--image IMAGE] [--args ...]"
)
def set__defjob(args):
    #req_url = args.url + "/machines/create_asks/?user_id=" + str(args.user);
    req_url    = apiurl(args, "/machines/create_bids/");

    #print("PUT " + req_url);
    r = requests.put(req_url, json = 
        {'machine':args.id, 'price_gpu':args.price_gpu, 'price_inetu':args.price_inetu, 'price_inetd':args.price_inetd,
         'image':args.image, 'args':args.args } );
    
    if (r.status_code == 200) :
        #print(r.text);
        rj = r.json();
        if (rj["success"]) :
            print("bids created for machine {args.id},  @ ${args.price_gpu}/gpu/day, ${args.price_inetu}/GB up, ${args.price_inetd}/GB down".format(**locals()));
        else :
            print(rj["msg"]);
    else :
        print(r.text);
        print("failed with error {r.status_code}".format(**locals()));








#def delete_bid(args):
#    print("delete bids!\n");
#
#def set_bid(args):
#    print("set bids!\n");
#
#
#def accept_bid(args):
#    print("accept bid!\n");

@parser.command(
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
)
def create__instance(args):
    if args.onstart:
        with open(args.onstart, "r") as reader:
            args.onstart_cmd = reader.read()
    runtype = 'ssh'
    if args.args:
        runtype = 'args'
    if args.jupyter_dir or args.jupyter_lab:
        args.jupyter = True
    if args.jupyter and runtype == 'args':
        print("Error: Can't use --jupyter and --args together. Try --onstart or --onstart-cmd instead of --args.", file=sys.stderr)
        return 1
    if args.jupyter:
        runtype = 'jupyter'

    url = apiurl(args, "/asks/{id}/".format(id=args.id))
    r = requests.put(url, json={
        "client_id": "me",
        "image": args.image,
        "args":  args.args,
        "price": args.price,
        "disk":  args.disk,
        "label": args.label,
        "extra": args.extra,
        "onstart": args.onstart_cmd,
        "runtype": runtype,
        "python_utf8": args.python_utf8,
        "lang_utf8": args.lang_utf8,
        "use_jupyter_lab": args.jupyter_lab,
        "jupyter_dir": args.jupyter_dir,
        "create_from": args.create_from,
        "force": args.force
    })
    r.raise_for_status()
    if args.raw:
        print(json.dumps(r.json(), indent=1))
    else:
        print("Started. {}".format(r.json()))

@parser.command(
    argument("id",            help="id of instance type to launch", type=int),
    argument("--price",       help="per machine bid price in $/hour", type=float),
    usage = "vast change bid id [--price PRICE]",
    epilog = deindent("""
        Change the current bid price of instance id to PRICE.
        If PRICE is not specified, then a winning bid price is used as the default.
    """),
)
def change__bid(args):
    url = apiurl(args, "/instances/bid_price/{id}/".format(id=args.id))
    r = requests.put(url, json={
        "client_id": "me",
        "price" : args.price,
    })
    r.raise_for_status()
    print("Per gpu bid price changed".format(r.json()))


@parser.command(
    argument("id",            help="id of machine to set min bid price for", type=int),
    argument("--price",       help="per gpu min bid price in $/hour", type=float),
    usage = "vast set min_bid id [--price PRICE]",
    epilog = deindent("""
        Change the current min bid price of machine id to PRICE.
    """),
)
def set__min_bid(args):
    url = apiurl(args, "/machines/{id}/minbid/".format(id=args.id))
    r = requests.put(url, json={
        "client_id": "me",
        "price" : args.price,
    })
    r.raise_for_status()
    print("Per gpu min bid price changed".format(r.json()))




@parser.command(
    argument("api-key",    help="Api key to set as currently logged in user"),
    usage = "vast set api-key APIKEY",
)
def set__api_key(args):
    with open(api_key_file, "w") as writer:
        writer.write(args.api_key)
    print("Your api key has been saved in {}".format(api_key_file_base))

#def _load_sshkey(arg):
#    if arg is not None and os.path.exists(arg):
#        with open(arg, "r") as reader:
#            return reader.read()
#    return arg

@parser.command(
    argument("email",    help="Email"),
    argument("password",    help="Password"),
    #argument("--ssh-key",     help="The SSH Pubkey you'd like to use to connect to containers"),
    usage = "vast create account [--api-key API_KEY] [--ssh-key SSH_KEY] USERNAME PASSWORD",
)
def create__account(args):
    if args.username is None:
        args.username = input("Email: ");
    if args.password is None:
        args.password = getpass.getpass("Password: ");
    # TODO: do this?
    #if args.ssh_key is None:
    #    args.ssh_key = input("Ssh key: ");

    url = apiurl(args, "/users/");
    #msg = 'ssh_key': _load_sshkey(args.ssh_key)
    
    r = requests.post(url,
            json={'username':args.username, 'password':  args.password, } );
    r.raise_for_status()
    resp = r.json()
    print("You are user {}! Your new api key: {}".format(resp["id"], resp["api_key"]))
    args.api_key = resp["api_key"]
    set_api_key(args)

@parser.command(
    argument("username",    help="Username or Email", nargs="?", default=None),
    argument("password",    help="Password", nargs="?", default=None),
    argument("--ssh-key",     help="The SSH Pubkey you'd like to use to connect to containers"),
    usage = "vast login [--username USERNAME] [--password PASSWORD] [--api-key API_KEY] [--ssh-key SSH_KEY]",
)
def login(args):
    if args.username is None:
        args.username = input("Username or Email: ");
    if args.password is None:
        try:
            # weird try/except is because windows gives a typeerror on this line
            args.password = getpass.getpass("Password: ");
        except TypeError:
            try:
                args.password = getpass.getpass("Password: ".encode("utf-8"))
            except TypeError:
                args.password = raw_input("Password: ")

    url = apiurl(args, "/users/current/");
    print(url)
    
    r = requests.put(url,
            json={'username': args.username, 'password': args.password} );
    r.raise_for_status()
    resp = r.json()
    print("You are user {}! Your existing api key: {}".format(resp["id"], resp["api_key"]))
    args.api_key = resp["api_key"]
    set__api_key(args)

def main():
    parser.add_argument("--url", help="server REST api url", default=server_url_default)
    parser.add_argument("--raw", action="store_true", help="output machine-readable json");
    parser.add_argument("--api-key",     help="api key. defaults to using the one stored in {}".format(api_key_file_base), type=str, required=False, default=api_key_guard)

    #func_dict = {
    #    "set defjob":               set_defjob,
    #    "remove defjob":            remove_defjob,
    #    #"accept ask":              accept_ask,
    #    #"exec bid":                create_bid,
    #    #"set ask":                 set_ask,
    #    #"list asks":               list_asks,
    #    "create account":           create_account,
    #    "login":                    login,
    #    "create instance":          create_instance,
    #    "change bid":               change_bid,
    #    "destroy instance":         destroy_instance,
    #    "start instance":           start_instance,
    #    "stop instance":            stop_instance,
    #    "label instance":           label_instance,
    #    "list machine":             list_machine,
    #    "unlist machine":           unlist_machine,
    #    #"search offers":         search_instances,
    #    "show instances":           show_instances,
    #    "show host-instances":      host_instances,
    #    "show machines":            show_machines,
    #    "set min-bid":              set_min_bid,
    #    "set api-key":              set_api_key_cmd,
    #}


    func_help = [
        "General:",
        "create account          Create a new account with a username and password",
        "login                   Login to an account and or switch accounts",
        "set api-key             Set account via api key",
        "",
        "Client:",
        "create instance         Accept an offer and launch a new container instance",
        "destroy instance        Stop and destroy an existing container instance, deleting local storage",
        "change bid              Set a new bid price for an interruptible instance",
        "label instance          Set label for existing instance",
        "start instance          Start/restart an existing stopped container instance",
        "stop instance           Stop and hibernate an existing container instance; local storage persists",
        "search offers           Search for available instances to rent that match specific criteria",
        "show instances          Show all your current rental instances",
        "",
        "Host:",
        "show machines           Show all your physical machines connected to vast.ai",
        "set defjob              Change/configure the default low-priority instance to run on a machine",
        "remove defjob           remove any default instance from a machine",
        "list machine            List a machine for rental: creates and registers one or more instance offers with pricing",
        "unlist machine          Unlist a machine: destroys and unregisters instance offers (but doesn't effect any active instances)",
        "set min-bid             Set minimum per gpu bid price",
    ]
    
    #cmd0 = ""; cmd1 = "";
    #
    #if len(sys.argv) > 1:
    #    if (sys.argv[1] == "help"):
    #        sys.argv[1:2] = []
    #        sys.argv.append("--help")
    #if len(sys.argv) > 1: cmd0 = sys.argv[1];

    #command_type = cmd0;
        
    #if (cmd0 not in func_dict):
    #    if len(sys.argv) > 2: cmd1 = sys.argv[2];
    #    command_type = cmd0 + ' ' + cmd1;
    
    #print("command: {}".format(command_type));

    #parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter);

    args = parser.parse_args()
    if args.api_key is api_key_guard:
        if os.path.exists(api_key_file):
            with open(api_key_file, "r") as reader:
                args.api_key = reader.read().strip()
        else:
            args.api_key = None
    try:
        sys.exit(args.func(args) or 0)
    except requests.exceptions.HTTPError as e :
        try:
            errmsg = e.response.json().get("msg");
        except JSONDecodeError:
            if e.response.status_code == 401:
                errmsg = "Please log in or sign up"
            else:
                errmsg = "(no detail message supplied)"
        print("failed with error {e.response.status_code}: {errmsg}".format(**locals()));            
    #else:
    #    if cmd0 != "--help" : print("Unrecognized command '" + command_type.strip() + "'. Use vast --help for list of commands.")
    #    parser = argparse.ArgumentParser(
    #            description="Available Commands:\n\n{}".format("\n".join((func_help))),
    #            formatter_class=argparse.RawDescriptionHelpFormatter)
    #    parser.parse_args()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
