"""Image wrapper and I/O helpers."""

from __future__ import annotations

import ctypes
import os
from typing import Any, Tuple, Union

from ._native import CImage, lib


class Image:
    """Wrapper around a LiteOCR image buffer.

    The wrapper can be constructed from a file path, a NumPy array, or from the
    raw ``CImage`` struct returned by the C library.
    """

    def __init__(self, c_image: CImage, owns_data: bool = False, numpy_ref: Any = None):
        self._img = c_image
        self._owns = owns_data
        self._numpy_ref = numpy_ref

    @classmethod
    def from_file(cls, path: Union[str, os.PathLike], desired_channels: int = 3) -> "Image":
        """Load an image file (PNG/JPG/etc.) using LiteOCR's bundled decoder."""
        c_img = lib.liteocr_imread(os.fspath(path).encode("utf-8"), desired_channels)
        if c_img.data is None:
            raise RuntimeError(f"Failed to load image: {path}")
        return cls(c_img, owns_data=True)

    @classmethod
    def from_numpy(cls, array: Any) -> "Image":
        """Wrap a NumPy array as a LiteOCR image.

        The array must be contiguous ``uint8`` with shape ``(H, W)`` or
        ``(H, W, C)`` where ``C`` is 1, 3 or 4.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError("NumPy is required for from_numpy") from exc

        if not isinstance(array, np.ndarray):
            raise TypeError("Expected a NumPy ndarray")
        if array.dtype != np.uint8:
            raise TypeError("Array must have dtype uint8")
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        if array.ndim == 2:
            height, width = array.shape
            channels = 1
        elif array.ndim == 3:
            height, width, channels = array.shape
        else:
            raise ValueError("Array must have 2 or 3 dimensions")

        stride = width * channels
        c_img = CImage(
            data=array.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            width=width,
            height=height,
            channels=channels,
            stride=stride,
        )
        return cls(c_img, owns_data=False, numpy_ref=array)

    @property
    def width(self) -> int:
        return self._img.width

    @property
    def height(self) -> int:
        return self._img.height

    @property
    def channels(self) -> int:
        return self._img.channels

    @property
    def stride(self) -> int:
        return self._img.stride

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self._img.height, self._img.width, self._img.channels)

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Write the image to a file."""
        rc = lib.liteocr_imwrite(os.fspath(path).encode("utf-8"), ctypes.byref(self._img))
        if rc != 0:
            raise RuntimeError(f"Failed to write image: {path}")

    def to_numpy(self) -> Any:
        """Convert the image to a NumPy array (returns a copy)."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError("NumPy is required for to_numpy") from exc

        size = self._img.height * self._img.stride
        buf = ctypes.string_at(self._img.data, size)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(
            (self._img.height, self._img.stride)
        )
        if self._img.channels == 1:
            return arr[:, : self._img.width].copy()
        return arr[:, : self._img.width * self._img.channels].reshape(
            (self._img.height, self._img.width, self._img.channels)
        ).copy()

    def __del__(self):
        if self._owns and self._img.data:
            lib.liteocr_free_image(ctypes.byref(self._img))
            self._img.data = None


def load_image(path: Union[str, os.PathLike], desired_channels: int = 3) -> Image:
    """Convenience function: load an image file."""
    return Image.from_file(path, desired_channels)


def imwrite(path: Union[str, os.PathLike], image: Image) -> None:
    """Convenience function: write an image file."""
    image.save(path)
