"""Shared between data prep, training, and the submitted policy. Keep it dependency-light
(numpy + opencv) so the same preprocessing runs at train and eval time."""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

CAMERAS = ("head", "left_wrist", "right_wrist")

FAMILIES = (
    "LiftPot",
    "LiftTray",
    "StackTwoBlocks",
    "StackSingleBookShelf",
    "CubeHandover",
    "PackBox",
    "PickSingleBookFromTable",
    "RotateValve",
)


def family_of(variant: str) -> str:
    """'CubeHandoverPositionAndOrientation' -> 'CubeHandover'."""
    for f in sorted(FAMILIES, key=len, reverse=True):
        if variant.startswith(f):
            return f
    raise KeyError(variant)


def resize_image(hwc: np.ndarray, size: int) -> np.ndarray:
    """uint8 HxWx3 -> uint8 size x size x 3, area interpolation."""
    if hwc.shape[0] == size and hwc.shape[1] == size:
        return hwc
    if cv2 is not None:
        return cv2.resize(hwc, (size, size), interpolation=cv2.INTER_AREA)
    from PIL import Image
    return np.asarray(Image.fromarray(hwc).resize((size, size), Image.BOX))
