"""Approximate, GT-blind assignment of frozen OCR crops to independent text slots."""

from __future__ import annotations

import math

import numpy as np

class EmptyDetectorBoundaryError(ValueError):
    """A finite boundary has an empty envelope after clipping to the image."""

    def __init__(self, unclipped_bbox: list[float], clipped_bbox: list[int]):
        super().__init__("detector polygon has no in-image area")
        self.unclipped_bbox = unclipped_bbox
        self.clipped_bbox = clipped_bbox

def _boundary_array(points, image_wh: tuple[int, int]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != 2 or len(array) < 3
        or not np.isfinite(array).all() or len(image_wh) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in image_wh)):
        raise ValueError("invalid detector boundary polygon")
    return array

def detector_boundary_in_original(points, image_wh: tuple[int, int],
                                  exif_orientation: int = 1) -> list[list[float]]:
    """Invert OpenCV's EXIF orientation in polygon pixel coordinates.

    PIL crop/annotation coordinates refer to the stored, untransposed raster.
    Orientations 0 (unspecified) and 1 both leave OpenCV's raster unchanged.
    """
    array = _boundary_array(points, image_wh)
    if (not isinstance(exif_orientation, int) or isinstance(exif_orientation, bool)
        or exif_orientation not in range(9)):
        raise ValueError("unsupported detector EXIF orientation")
    width, height = image_wh
    x, y = array[:, 0], array[:, 1]
    inverse = {
        0: (x, y), 1: (x, y), 2: (width - 1 - x, y),
        3: (width - 1 - x, height - 1 - y), 4: (x, height - 1 - y),
        5: (y, x), 6: (y, height - 1 - x),
        7: (width - 1 - y, height - 1 - x), 8: (width - 1 - y, x),
    }
    return np.column_stack(inverse[exif_orientation]).tolist()

def boundary_bbox(points, image_wh: tuple[int, int]) -> list[int]:
    """Envelope a detector polygon in original-image, exclusive xyxy coordinates."""
    array = _boundary_array(points, image_wh)
    width, height = image_wh
    box = [max(0, min(width, math.floor(array[:, 0].min()))),
           max(0, min(height, math.floor(array[:, 1].min()))),
           max(0, min(width, math.ceil(array[:, 0].max()))),
           max(0, min(height, math.ceil(array[:, 1].max())))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise EmptyDetectorBoundaryError(
            [float(array[:, 0].min()), float(array[:, 1].min()),
             float(array[:, 0].max()), float(array[:, 1].max())], box)
    return box
