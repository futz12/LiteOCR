"""Image processing helpers wrapping the LiteOCR C API."""

from __future__ import annotations

import ctypes
from typing import List, Sequence, Tuple, Union

from ._native import CImage, lib
from .image import Image

# Color format constants (matching liteocr_imgproc.h).
COLOR_GRAY = 1
COLOR_RGB = 2
COLOR_BGR = 3
COLOR_RGBA = 4
COLOR_BGRA = 5


def _ensure_numpy(arr: Any) -> Any:
    """Return a NumPy array; raise if NumPy is not available."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError("NumPy is required for image processing helpers") from exc
    if isinstance(arr, Image):
        return arr.to_numpy()
    return np.asarray(arr)


def _to_c_image(arr: Any, ref_holder: List[Any]) -> CImage:
    """Build a CImage descriptor for an Image or NumPy array.

    ``ref_holder`` keeps Python objects alive while the C image is in use.
    """
    import numpy as np
    if isinstance(arr, Image):
        return arr._img
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint8:
        raise TypeError("Input array must be uint8")
    if arr.ndim == 2:
        h, w = arr.shape
        c = 1
    elif arr.ndim == 3:
        h, w, c = arr.shape
    else:
        raise ValueError("Array must have 2 or 3 dimensions")
    ref_holder.append(arr)
    return CImage(
        data=arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        width=w,
        height=h,
        channels=c,
        stride=w * c,
    )


def _allocate_output(shape: Tuple[int, ...]) -> Any:
    import numpy as np
    return np.empty(shape, dtype=np.uint8)


def cvt_color(
    src: Union[Image, Any],
    dst_fmt: int,
    src_fmt: int,
) -> Any:
    """Convert image color format."""
    import numpy as np
    src_arr = _ensure_numpy(src)
    src_img = _to_c_image(src_arr, [])
    dst_channels = 1 if dst_fmt == COLOR_GRAY else (
        4 if dst_fmt in (COLOR_RGBA, COLOR_BGRA) else 3
    )
    dst = _allocate_output((src_img.height, src_img.width, dst_channels))
    dst_img = _to_c_image(dst, [])
    lib.liteocr_cvt_color(
        src_img.data, src_img.width, src_img.height, src_img.stride, src_fmt,
        dst_img.data, dst_img.stride, dst_fmt,
    )
    return dst


def threshold(
    src: Union[Image, Any],
    thresh: float,
    maxval: int = 255,
) -> Any:
    """Apply threshold to a grayscale float image, returning a uint8 mask."""
    import numpy as np
    src_arr = _ensure_numpy(src)
    if src_arr.ndim != 2:
        raise ValueError("threshold expects a 2D array")
    src_f = np.ascontiguousarray(src_arr, dtype=np.float32)
    dst = _allocate_output(src_f.shape)
    dst_img = _to_c_image(dst, [])
    lib.liteocr_threshold(
        src_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        src_f.shape[1], src_f.shape[0], src_f.strides[0] // src_f.itemsize,
        dst_img.data, dst_img.stride,
        float(thresh), np.uint8(maxval),
    )
    return dst


def mean_masked(
    src: Union[Image, Any],
    mask: Union[Image, Any],
) -> float:
    """Compute mean of ``src`` where ``mask`` is non-zero."""
    import numpy as np
    src_arr = np.ascontiguousarray(_ensure_numpy(src), dtype=np.float32)
    mask_arr = np.ascontiguousarray(_ensure_numpy(mask), dtype=np.uint8)
    if src_arr.shape[:2] != mask_arr.shape[:2]:
        raise ValueError("src and mask must have the same height and width")
    return float(lib.liteocr_mean_masked(
        src_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        src_arr.shape[1], src_arr.shape[0], src_arr.strides[0] // src_arr.itemsize,
        mask_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        mask_arr.strides[0] // mask_arr.itemsize,
    ))


def resize(
    src: Union[Image, Any],
    dst_width: int,
    dst_height: int,
) -> Any:
    """Resize an image."""
    src_arr = _ensure_numpy(src)
    src_img = _to_c_image(src_arr, [])
    dst = _allocate_output((dst_height, dst_width, src_img.channels))
    dst_img = _to_c_image(dst, [])
    lib.liteocr_resize(
        src_img.data, src_img.width, src_img.height, src_img.stride, src_img.channels,
        dst_img.data, dst_width, dst_height, dst_img.stride,
    )
    return dst


def rotate90(
    src: Union[Image, Any],
    counter_clockwise: bool = False,
) -> Any:
    """Rotate an image by 90 degrees."""
    src_arr = _ensure_numpy(src)
    src_img = _to_c_image(src_arr, [])
    dst = _allocate_output((src_img.width, src_img.height, src_img.channels))
    dst_img = _to_c_image(dst, [])
    lib.liteocr_rotate90(
        src_img.data, src_img.width, src_img.height, src_img.stride, src_img.channels,
        dst_img.data, dst_img.stride, int(counter_clockwise),
    )
    return dst


def rotate180(src: Union[Image, Any]) -> Any:
    """Rotate an image by 180 degrees."""
    src_arr = _ensure_numpy(src)
    src_img = _to_c_image(src_arr, [])
    dst = _allocate_output((src_img.height, src_img.width, src_img.channels))
    dst_img = _to_c_image(dst, [])
    lib.liteocr_rotate180(
        src_img.data, src_img.width, src_img.height, src_img.stride, src_img.channels,
        dst_img.data, dst_img.stride,
    )
    return dst


def copy_make_border(
    src: Union[Image, Any],
    top: int,
    bottom: int,
    left: int,
    right: int,
    fill_value: int = 0,
) -> Any:
    """Pad an image with a constant border."""
    src_arr = _ensure_numpy(src)
    src_img = _to_c_image(src_arr, [])
    dst_h = src_img.height + top + bottom
    dst_w = src_img.width + left + right
    dst = _allocate_output((dst_h, dst_w, src_img.channels))
    dst_img = _to_c_image(dst, [])
    lib.liteocr_copy_make_border(
        src_img.data, src_img.width, src_img.height, src_img.stride, src_img.channels,
        dst_img.data, dst_w, dst_h, dst_img.stride,
        top, bottom, left, right, fill_value,
    )
    return dst


def get_perspective_transform(
    src_pts: Sequence[Tuple[float, float]],
    dst_pts: Sequence[Tuple[float, float]],
) -> Any:
    """Compute a 3x3 perspective transform matrix."""
    import numpy as np
    if len(src_pts) != 4 or len(dst_pts) != 4:
        raise ValueError("Exactly 4 points are required")
    src_arr = np.array([c for p in src_pts for c in p], dtype=np.float32)
    dst_arr = np.array([c for p in dst_pts for c in p], dtype=np.float32)
    M = np.empty(9, dtype=np.float32)
    lib.liteocr_get_perspective_transform(
        src_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return M.reshape(3, 3)


def warp_perspective(
    src: Union[Image, Any],
    M: Any,
    dst_width: int,
    dst_height: int,
) -> Any:
    """Apply a perspective transform to an image."""
    import numpy as np
    src_arr = _ensure_numpy(src)
    src_img = _to_c_image(src_arr, [])
    M_arr = np.ascontiguousarray(M, dtype=np.float32).reshape(9)
    dst = _allocate_output((dst_height, dst_width, src_img.channels))
    dst_img = _to_c_image(dst, [])
    lib.liteocr_warp_perspective(
        src_img.data, src_img.width, src_img.height, src_img.stride, src_img.channels,
        dst_img.data, dst_width, dst_height, dst_img.stride,
        M_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return dst
