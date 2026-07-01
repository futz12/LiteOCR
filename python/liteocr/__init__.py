"""LiteOCR Python wrapper.

This package wraps the LiteOCR C library via ``ctypes``.  The native shared
library is bundled inside the package by the build system, so a normal
``pip install .`` or wheel install works out of the box.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any, List, Optional, Union

from . import components, contours, imgproc, presets, table
from ._native import CImage, CTextBox, CTextLine, c_str, lib
from ._types import (
    InferOption,
    OCRResult,
    TableCell,
    TableResult,
    TextBox,
    TextLine,
)
from .components import (
    Detector,
    DocOrientation,
    Recognizer,
    SLANet,
    TextlineOrientation,
    UVDoc,
)
from .contours import (
    CHAIN_APPROX_NONE,
    CHAIN_APPROX_SIMPLE,
    Contour,
    arc_length,
    bounding_rect,
    contour_area,
    fill_poly,
    find_contours,
    min_area_rect,
)
from .image import Image, imwrite, load_image
from .imgproc import (
    COLOR_BGR,
    COLOR_BGRA,
    COLOR_GRAY,
    COLOR_RGB,
    COLOR_RGBA,
    copy_make_border,
    cvt_color,
    get_perspective_transform,
    mean_masked,
    resize,
    rotate180,
    rotate90,
    threshold,
    warp_perspective,
)
from .presets import (
    download_orientation_preset,
    download_preset,
    download_table_preset,
    ensure_preset,
    list_orientation_presets,
    list_presets,
    list_table_presets,
)
from .table import TableEngine

__version__ = "0.1.0"

__all__ = [
    # Core
    "Image",
    "InferOption",
    "TextBox",
    "TextLine",
    "OCRResult",
    "Engine",
    "load_image",
    "imwrite",
    # Table
    "TableEngine",
    "TableCell",
    "TableResult",
    # Low-level components
    "Detector",
    "Recognizer",
    "TextlineOrientation",
    "DocOrientation",
    "UVDoc",
    "SLANet",
    # Image processing
    "COLOR_GRAY",
    "COLOR_RGB",
    "COLOR_BGR",
    "COLOR_RGBA",
    "COLOR_BGRA",
    "cvt_color",
    "threshold",
    "mean_masked",
    "resize",
    "rotate90",
    "rotate180",
    "copy_make_border",
    "get_perspective_transform",
    "warp_perspective",
    # Contours
    "CHAIN_APPROX_NONE",
    "CHAIN_APPROX_SIMPLE",
    "Contour",
    "find_contours",
    "min_area_rect",
    "bounding_rect",
    "contour_area",
    "arc_length",
    "fill_poly",
    # Presets / utilities
    "presets",
    "download_preset",
    "download_orientation_preset",
    "download_table_preset",
    "ensure_preset",
    "list_presets",
    "list_orientation_presets",
    "list_table_presets",
    "merge_text_boxes",
    "ctc_decode",
]


# ---------------------------------------------------------------------------
# OCR Engine wrapper
# ---------------------------------------------------------------------------

class Engine:
    """High-level wrapper around ``liteocr_engine_t``."""

    def __init__(self, opt: Optional[InferOption] = None):
        self._handle = lib.liteocr_engine_create()
        if not self._handle:
            raise RuntimeError("Failed to create LiteOCR engine")
        self._opt = opt if opt is not None else InferOption()
        self._c_opt = self._opt.to_c()

    def load_model(
        self,
        det_param: Union[str, os.PathLike],
        det_bin: Union[str, os.PathLike],
        rec_param: Union[str, os.PathLike],
        rec_bin: Union[str, os.PathLike],
        vocab: Union[str, os.PathLike],
        ori_param: Optional[Union[str, os.PathLike]] = None,
        ori_bin: Optional[Union[str, os.PathLike]] = None,
    ) -> "Engine":
        """Load model files from disk.

        Returns ``self`` so calls can be chained.
        """
        from ._native import COCRModelPaths
        paths = COCRModelPaths(
            det_param=c_str(det_param),
            det_bin=c_str(det_bin),
            rec_param=c_str(rec_param),
            rec_bin=c_str(rec_bin),
            vocab=c_str(vocab),
            ori_param=c_str(ori_param),
            ori_bin=c_str(ori_bin),
        )
        rc = lib.liteocr_engine_load_model(
            self._handle, ctypes.byref(paths), ctypes.byref(self._c_opt)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load OCR model (error code {rc})")
        return self

    def load_preset(
        self,
        name: str,
        model_dir: str = "models",
        orientation: Optional[str] = None,
        download: bool = True,
    ) -> "Engine":
        """Load a named model preset, optionally downloading missing files."""
        if download:
            paths = ensure_preset(name, model_dir, orientation)
        else:
            paths = {k: Path(model_dir) / v for k, v in presets._OCR_PRESETS[name].items()}
            if orientation:
                paths.update(
                    {k: Path(model_dir) / v for k, v in presets._ORIENTATION_PRESETS[orientation].items()}
                )
        return self.load_model(
            det_param=paths["det_param"],
            det_bin=paths["det_bin"],
            rec_param=paths["rec_param"],
            rec_bin=paths["rec_bin"],
            vocab=paths["vocab"],
            ori_param=paths.get("ori_param"),
            ori_bin=paths.get("ori_bin"),
        )

    def recognize(self, image: Union[Image, Any, bytes, str, os.PathLike]) -> OCRResult:
        """Run OCR on an image.

        ``image`` may be:
        * an :class:`Image` instance,
        * a file path string,
        * encoded image bytes (PNG/JPG/etc.),
        * a NumPy array.
        """
        if isinstance(image, Image):
            return self._recognize_image(image)
        if isinstance(image, (str, os.PathLike)):
            return self._recognize_image(Image.from_file(image))
        if isinstance(image, bytes):
            return self._recognize_buffer(image)

        img = Image.from_numpy(image)
        return self._recognize_raw(img)

    def _recognize_image(self, image: Image) -> OCRResult:
        boxes_ptr = ctypes.POINTER(CTextBox)()
        box_count = ctypes.c_int()
        lines_ptr = ctypes.POINTER(CTextLine)()
        line_count = ctypes.c_int()
        rc = lib.liteocr_engine_recognize_image(
            self._handle,
            ctypes.byref(image._img),
            ctypes.byref(boxes_ptr),
            ctypes.byref(box_count),
            ctypes.byref(lines_ptr),
            ctypes.byref(line_count),
        )
        if rc != 0:
            raise RuntimeError(f"OCR recognition failed (error code {rc})")
        return _parse_ocr_result(boxes_ptr, box_count.value, lines_ptr, line_count.value)

    def _recognize_raw(self, image: Image) -> OCRResult:
        boxes_ptr = ctypes.POINTER(CTextBox)()
        box_count = ctypes.c_int()
        lines_ptr = ctypes.POINTER(CTextLine)()
        line_count = ctypes.c_int()
        rc = lib.liteocr_engine_recognize_raw(
            self._handle,
            image._img.data,
            image._img.width,
            image._img.height,
            image._img.channels,
            image._img.stride,
            ctypes.byref(boxes_ptr),
            ctypes.byref(box_count),
            ctypes.byref(lines_ptr),
            ctypes.byref(line_count),
        )
        if rc != 0:
            raise RuntimeError(f"OCR recognition failed (error code {rc})")
        return _parse_ocr_result(boxes_ptr, box_count.value, lines_ptr, line_count.value)

    def _recognize_buffer(self, buffer: bytes) -> OCRResult:
        boxes_ptr = ctypes.POINTER(CTextBox)()
        box_count = ctypes.c_int()
        lines_ptr = ctypes.POINTER(CTextLine)()
        line_count = ctypes.c_int()
        buf = (ctypes.c_ubyte * len(buffer)).from_buffer_copy(buffer)
        rc = lib.liteocr_engine_recognize_buffer(
            self._handle,
            buf,
            len(buffer),
            ctypes.byref(boxes_ptr),
            ctypes.byref(box_count),
            ctypes.byref(lines_ptr),
            ctypes.byref(line_count),
        )
        if rc != 0:
            raise RuntimeError(f"OCR recognition failed (error code {rc})")
        return _parse_ocr_result(boxes_ptr, box_count.value, lines_ptr, line_count.value)

    def __del__(self):
        if getattr(self, "_handle", None):
            lib.liteocr_engine_destroy(self._handle)
            self._handle = None


def _parse_ocr_result(
    boxes_ptr: ctypes.POINTER(CTextBox),
    box_count: int,
    lines_ptr: ctypes.POINTER(CTextLine),
    line_count: int,
) -> OCRResult:
    boxes: List[TextBox] = []
    for i in range(box_count):
        cb = boxes_ptr[i]
        pts = [(cb.points[j], cb.points[j + 1]) for j in range(0, 8, 2)]
        boxes.append(TextBox(points=pts, is_vertical=bool(cb.is_vertical), score=cb.score))

    lines: List[TextLine] = []
    for i in range(line_count):
        cl = lines_ptr[i]
        text = cl.text.decode("utf-8") if cl.text else ""
        anchors = []
        if cl.anchors and cl.anchor_count:
            for j in range(cl.anchor_count):
                anchors.append((cl.anchors[j * 2], cl.anchors[j * 2 + 1]))
        lines.append(TextLine(text=text, anchors=anchors))

    lib.liteocr_free_text_boxes(boxes_ptr, box_count)
    lib.liteocr_free_text_lines(lines_ptr, line_count)
    return OCRResult(boxes=boxes, lines=lines)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def merge_text_boxes(boxes: List[TextBox], lines: List[TextLine]) -> str:
    """Merge all text boxes into a single string."""
    c_boxes = (CTextBox * len(boxes))()
    for i, b in enumerate(boxes):
        for j, (x, y) in enumerate(b.points):
            c_boxes[i].points[j * 2] = x
            c_boxes[i].points[j * 2 + 1] = y
        c_boxes[i].is_vertical = int(b.is_vertical)
        c_boxes[i].score = b.score

    c_lines = (CTextLine * len(lines))()
    refs = []
    for i, line in enumerate(lines):
        c_lines[i].text = line.text.encode("utf-8")
        count = len(line.anchors)
        c_lines[i].anchor_count = count
        if count:
            arr = (ctypes.c_float * (count * 2))()
            for j, (x, y) in enumerate(line.anchors):
                arr[j * 2] = x
                arr[j * 2 + 1] = y
            c_lines[i].anchors = arr
            refs.append(arr)
        else:
            c_lines[i].anchors = None

    c_str_ptr = lib.liteocr_merge_text_boxes(c_boxes, len(boxes), c_lines, len(lines))
    result = c_str_ptr.decode("utf-8") if c_str_ptr else ""
    lib.liteocr_free_string(c_str_ptr)
    return result


def ctc_decode(
    probs: Any,
    blank_index: int,
) -> Tuple[List[int], List[float], List[int]]:
    """Run the CTC decoder on a probability image.

    ``probs`` should be a NumPy array with shape ``(T, C, 1)`` and dtype
    ``float32``.
    """
    import numpy as np
    arr = np.ascontiguousarray(probs, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("probs must be a 3D array (T, C, 1)")
    t, c, one = arr.shape
    if one != 1:
        raise ValueError("probs must have shape (T, C, 1)")

    c_img = CImage(
        data=arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        width=c,
        height=t,
        channels=1,
        stride=c * arr.itemsize,
    )

    tokens_ptr = ctypes.POINTER(ctypes.c_int)()
    scores_ptr = ctypes.POINTER(ctypes.c_float)()
    indices_ptr = ctypes.POINTER(ctypes.c_int)()
    count = ctypes.c_int()

    rc = lib.liteocr_ctc_decode(
        ctypes.byref(c_img),
        blank_index,
        ctypes.byref(tokens_ptr),
        ctypes.byref(scores_ptr),
        ctypes.byref(indices_ptr),
        ctypes.byref(count),
    )
    if rc != 0:
        raise RuntimeError(f"ctc_decode failed (error code {rc})")

    n = count.value
    tokens = [tokens_ptr[i] for i in range(n)]
    scores = [scores_ptr[i] for i in range(n)]
    indices = [indices_ptr[i] for i in range(n)]

    lib.liteocr_free(tokens_ptr)
    lib.liteocr_free(scores_ptr)
    lib.liteocr_free(indices_ptr)
    return tokens, scores, indices
