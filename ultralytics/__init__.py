# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "6.8.3.48"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download
from ultralytics.zzz_ultralytics import step1_train, step2_Constraint_train, step3_pruning, step4_finetune, zzz_train

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "step1_train",
    "step2_Constraint_train",
    "step3_pruning",
    "step4_finetune",
    "zzz_train"
)
