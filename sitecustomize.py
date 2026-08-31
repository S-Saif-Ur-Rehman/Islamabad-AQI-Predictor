"""Silence noisy warnings during local runs.uploading it if
 you are new user and try to supress anything or test it in your 
 local machine. you can use this file."""
from __future__ import annotations

import warnings

try:
    from urllib3.exceptions import InsecureRequestWarning
except Exception:  # pragma: no cover
    InsecureRequestWarning = None

if InsecureRequestWarning is not None:
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"google\._upb\._message.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"google\.protobuf.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*custom tp_new.*",
)
