import os
import datetime
from .utils import fix_wkb_column

__version__ = "0.4.0"

HOME = os.path.expanduser("~")

info_service_endpoint = "https://www.dynamicannotationframework.com/info"
annotation_endpoint = "https://www.dynamicannotationframework.com/annotation"
materialization_endpoint = "https://www.dynamicannotationframework.com/materialize"