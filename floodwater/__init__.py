"""Compact helpers for Prithvi-Tiny floodwater segmentation on Sen1Floods11-style chips."""

from .constants import (
    CHIP_SIZE,
    CROP_SIZE,
    IGNORE_INDEX,
    LABEL_INVALID,
    LABEL_NOWATER,
    LABEL_WATER,
    PRITHVI_BANDS,
    PRITHVI_BAND_LABELS,
    REFLECTANCE_SCALE,
)
from .metrics import (
    add,
    calibration_accumulator,
    calibration_summary,
    confusion,
    entropy_binary,
    scores,
    update_calibration,
)

__version__ = "1.0.0"
__all__ = [
    "CHIP_SIZE",
    "CROP_SIZE",
    "IGNORE_INDEX",
    "LABEL_INVALID",
    "LABEL_NOWATER",
    "LABEL_WATER",
    "PRITHVI_BANDS",
    "PRITHVI_BAND_LABELS",
    "REFLECTANCE_SCALE",
    "add",
    "calibration_accumulator",
    "calibration_summary",
    "confusion",
    "entropy_binary",
    "scores",
    "update_calibration",
]
