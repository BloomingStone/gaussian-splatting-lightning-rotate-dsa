"""Lightning callbacks for Gaussian Splatting training.

Modules
-------
callbacks
    General callbacks: ProgressBar, ValidateOnTrainEnd, etc.
save_image
    ``SaveImage`` – validation / test image saving.
scene_saver
    ``SceneSaver`` – wraps Saver logic as a callback.
"""

from .callbacks import (
    KeepRunningIfWebViewerEnabled,
    ProgressBar,
    ValidateOnTrainEnd,
)
from .save_image import SaveImage
from .scene_saver import SceneSaver

__all__ = [
    "KeepRunningIfWebViewerEnabled",
    "ProgressBar",
    "SaveImage",
    "SceneSaver",
    "ValidateOnTrainEnd",
]
