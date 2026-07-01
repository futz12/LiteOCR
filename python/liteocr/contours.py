"""Contour processing helpers wrapping the LiteOCR C API."""

from __future__ import annotations

import ctypes
from typing import List, Sequence, Tuple, Union

from ._native import CContour, CIntRect, CPoint, CPoint2f, lib
from .image import Image

CHAIN_APPROX_NONE = 1
CHAIN_APPROX_SIMPLE = 2

Contour = List[Tuple[int, int]]


def _to_cpoints(contour: Sequence[Tuple[int, int]]) -> Any:
    """Convert a list of integer points to a C array."""
    n = len(contour)
    arr = (CPoint * n)()
    for i, (x, y) in enumerate(contour):
        arr[i].x = int(x)
        arr[i].y = int(y)
    return arr


def find_contours(
    mask: Union[Image, Any],
    approx_mode: int = CHAIN_APPROX_SIMPLE,
) -> List[Contour]:
    """Find contours in a binary mask.

    Returns a list of contours, each contour being a list of ``(x, y)`` points.
    """
    import numpy as np
    if isinstance(mask, Image):
        arr = mask.to_numpy()
    else:
        arr = np.ascontiguousarray(mask, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("Mask must be a 2D array")

    contours_ptr = ctypes.POINTER(CContour)()
    count = ctypes.c_int()
    rc = lib.liteocr_find_contours(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        arr.shape[1], arr.shape[0], arr.strides[0],
        ctypes.byref(contours_ptr), ctypes.byref(count),
        approx_mode,
    )
    if rc != 0:
        raise RuntimeError(f"find_contours failed (error code {rc})")

    contours: List[Contour] = []
    for i in range(count.value):
        cc = contours_ptr[i]
        contour: Contour = []
        for j in range(cc.point_count):
            pt = cc.points[j]
            contour.append((pt.x, pt.y))
        contours.append(contour)

    lib.liteocr_free_contours(contours_ptr, count.value)
    return contours


def min_area_rect(contour: Sequence[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """Return the four corner points (tl, tr, br, bl) of the minimum area rect."""
    c_pts = _to_cpoints(contour)
    out = (CPoint2f * 4)()
    lib.liteocr_min_area_rect(c_pts, len(contour), out)
    return [(out[i].x, out[i].y) for i in range(4)]


def bounding_rect(contour: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Return ``(x, y, width, height)`` of the bounding rectangle."""
    c_pts = _to_cpoints(contour)
    r: CIntRect = lib.liteocr_bounding_rect(c_pts, len(contour))
    return (r.x, r.y, r.width, r.height)


def contour_area(contour: Sequence[Tuple[int, int]]) -> float:
    """Compute the contour area."""
    c_pts = _to_cpoints(contour)
    return float(lib.liteocr_contour_area(c_pts, len(contour)))


def arc_length(contour: Sequence[Tuple[int, int]], closed: bool = False) -> float:
    """Compute the contour perimeter."""
    c_pts = _to_cpoints(contour)
    return float(lib.liteocr_arc_length(c_pts, len(contour), int(closed)))


def fill_poly(
    shape: Tuple[int, int],
    polygons: Sequence[Sequence[Tuple[int, int]]],
    value: int = 255,
) -> Any:
    """Fill polygons on a blank mask and return it."""
    import numpy as np
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    n = len(polygons)
    c_polys = (CContour * n)()
    refs = []
    for i, poly in enumerate(polygons):
        pts = _to_cpoints(poly)
        refs.append(pts)
        c_polys[i].points = pts
        c_polys[i].point_count = len(poly)

    lib.liteocr_fill_poly(
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        w, h, mask.strides[0],
        c_polys, n,
        value,
    )
    return mask
