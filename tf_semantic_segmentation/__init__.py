__import__('pkg_resources').declare_namespace(__name__)  # noqa

from . import activations
from . import datasets
from . import debug
from . import evaluation
from . import layers
from . import losses
from . import metrics
from . import models
from . import optimizers
from . import processing
from . import visualizations
from . import callbacks
from . import serving
from . import settings
from . import threading
from . import utils
from . import version

from .version import __version__
