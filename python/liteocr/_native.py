"""Low-level ctypes bindings for the LiteOCR C API.

Most users should import from ``liteocr`` instead of using this module directly.
"""

from __future__ import annotations

import ctypes
import os
import platform
from typing import Any


def _load_native_lib() -> ctypes.CDLL:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    system = platform.system()
    if system == "Windows":
        names = ["liteocr.dll"]
    elif system == "Darwin":
        names = ["libliteocr.dylib"]
    else:
        names = ["libliteocr.so", "libliteocr.so.0"]

    for name in names:
        candidate = os.path.join(pkg_dir, name)
        if os.path.exists(candidate):
            return ctypes.CDLL(candidate)

    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue

    raise RuntimeError(
        f"Could not find the LiteOCR native library in {pkg_dir!r}. "
        "If you installed from source, please rebuild with pip install ."
    )


lib = _load_native_lib()


# ---------------------------------------------------------------------------
# C structures
# ---------------------------------------------------------------------------

class CImage(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("stride", ctypes.c_int),
    ]


class CPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int)]


class CPoint2f(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class CIntRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]


class CRect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
    ]


class CTextBox(ctypes.Structure):
    _fields_ = [
        ("points", ctypes.c_float * 8),
        ("is_vertical", ctypes.c_int),
        ("score", ctypes.c_float),
    ]


class CTextLine(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("anchors", ctypes.POINTER(ctypes.c_float)),
        ("anchor_count", ctypes.c_int),
    ]


class CInferOption(ctypes.Structure):
    _fields_ = [
        ("num_threads", ctypes.c_int),
        ("gpu_device_id", ctypes.c_int),
        ("use_fp16", ctypes.c_int),
        ("use_int8", ctypes.c_int),
        ("use_int8_det", ctypes.c_int),
        ("use_int8_rec", ctypes.c_int),
        ("use_bf16", ctypes.c_int),
        ("textline_ori_model_type", ctypes.c_int),
    ]


class CContour(ctypes.Structure):
    _fields_ = [
        ("points", ctypes.POINTER(CPoint)),
        ("point_count", ctypes.c_int),
    ]


class CTableCell(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_char_p),
        ("box", ctypes.c_float * 8),
    ]


class COCRModelPaths(ctypes.Structure):
    _fields_ = [
        ("det_param", ctypes.c_char_p),
        ("det_bin", ctypes.c_char_p),
        ("rec_param", ctypes.c_char_p),
        ("rec_bin", ctypes.c_char_p),
        ("vocab", ctypes.c_char_p),
        ("ori_param", ctypes.c_char_p),
        ("ori_bin", ctypes.c_char_p),
    ]


class CTableModelPaths(ctypes.Structure):
    _fields_ = [
        ("cnn_param", ctypes.c_char_p),
        ("cnn_bin", ctypes.c_char_p),
        ("slahead_param", ctypes.c_char_p),
        ("slahead_bin", ctypes.c_char_p),
        ("vocab", ctypes.c_char_p),
    ]


class CTableModelBuffers(ctypes.Structure):
    _fields_ = [
        ("cnn_param", ctypes.c_char_p),
        ("cnn_bin", ctypes.POINTER(ctypes.c_ubyte)),
        ("slahead_param", ctypes.c_char_p),
        ("slahead_bin", ctypes.POINTER(ctypes.c_ubyte)),
        ("vocab", ctypes.c_char_p),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def c_str(s: Any) -> Any:
    """Convert a path/string to bytes suitable for ``c_char_p``."""
    if s is None:
        return None
    return os.fspath(s).encode("utf-8")


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

lib.liteocr_free.argtypes = [ctypes.c_void_p]
lib.liteocr_free.restype = None

lib.liteocr_free_string.argtypes = [ctypes.c_char_p]
lib.liteocr_free_string.restype = None

lib.liteocr_free_image.argtypes = [ctypes.POINTER(CImage)]
lib.liteocr_free_image.restype = None

lib.liteocr_free_text_lines.argtypes = [ctypes.POINTER(CTextLine), ctypes.c_int]
lib.liteocr_free_text_lines.restype = None

lib.liteocr_free_text_boxes.argtypes = [ctypes.POINTER(CTextBox), ctypes.c_int]
lib.liteocr_free_text_boxes.restype = None

lib.liteocr_free_contours.argtypes = [ctypes.POINTER(CContour), ctypes.c_int]
lib.liteocr_free_contours.restype = None

lib.liteocr_free_table_cells.argtypes = [ctypes.POINTER(CTableCell), ctypes.c_int]
lib.liteocr_free_table_cells.restype = None


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

lib.liteocr_imread.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.liteocr_imread.restype = CImage

lib.liteocr_imwrite.argtypes = [ctypes.c_char_p, ctypes.POINTER(CImage)]
lib.liteocr_imwrite.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

lib.liteocr_cvt_color.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
]
lib.liteocr_cvt_color.restype = None

lib.liteocr_threshold.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
    ctypes.c_float, ctypes.c_ubyte,
]
lib.liteocr_threshold.restype = None

lib.liteocr_mean_masked.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
]
lib.liteocr_mean_masked.restype = ctypes.c_double

lib.liteocr_resize.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
lib.liteocr_resize.restype = None

lib.liteocr_rotate90.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
]
lib.liteocr_rotate90.restype = None

lib.liteocr_rotate180.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
]
lib.liteocr_rotate180.restype = None

lib.liteocr_copy_make_border.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_ubyte,
]
lib.liteocr_copy_make_border.restype = None

lib.liteocr_get_perspective_transform.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
lib.liteocr_get_perspective_transform.restype = None

lib.liteocr_warp_perspective.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
lib.liteocr_warp_perspective.restype = None


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------

lib.liteocr_find_contours.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(CContour)), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
lib.liteocr_find_contours.restype = ctypes.c_int

lib.liteocr_min_area_rect.argtypes = [
    ctypes.POINTER(CPoint), ctypes.c_int, ctypes.POINTER(CPoint2f),
]
lib.liteocr_min_area_rect.restype = None

lib.liteocr_bounding_rect.argtypes = [ctypes.POINTER(CPoint), ctypes.c_int]
lib.liteocr_bounding_rect.restype = CIntRect

lib.liteocr_contour_area.argtypes = [ctypes.POINTER(CPoint), ctypes.c_int]
lib.liteocr_contour_area.restype = ctypes.c_double

lib.liteocr_arc_length.argtypes = [ctypes.POINTER(CPoint), ctypes.c_int, ctypes.c_int]
lib.liteocr_arc_length.restype = ctypes.c_double

lib.liteocr_fill_poly.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(CContour), ctypes.c_int,
    ctypes.c_ubyte,
]
lib.liteocr_fill_poly.restype = None


# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------

lib.liteocr_engine_create.argtypes = []
lib.liteocr_engine_create.restype = ctypes.c_void_p

lib.liteocr_engine_destroy.argtypes = [ctypes.c_void_p]
lib.liteocr_engine_destroy.restype = None

lib.liteocr_engine_load_model.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(COCRModelPaths), ctypes.POINTER(CInferOption),
]
lib.liteocr_engine_load_model.restype = ctypes.c_int

lib.liteocr_engine_recognize_image.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(CImage),
    ctypes.POINTER(ctypes.POINTER(CTextBox)), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.POINTER(CTextLine)), ctypes.POINTER(ctypes.c_int),
]
lib.liteocr_engine_recognize_image.restype = ctypes.c_int

lib.liteocr_engine_recognize_raw.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(CTextBox)), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.POINTER(CTextLine)), ctypes.POINTER(ctypes.c_int),
]
lib.liteocr_engine_recognize_raw.restype = ctypes.c_int

lib.liteocr_engine_recognize_buffer.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(CTextBox)), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.POINTER(CTextLine)), ctypes.POINTER(ctypes.c_int),
]
lib.liteocr_engine_recognize_buffer.restype = ctypes.c_int

lib.liteocr_merge_text_boxes.argtypes = [
    ctypes.POINTER(CTextBox), ctypes.c_int,
    ctypes.POINTER(CTextLine), ctypes.c_int,
]
lib.liteocr_merge_text_boxes.restype = ctypes.c_char_p


# ---------------------------------------------------------------------------
# Table engine
# ---------------------------------------------------------------------------

lib.liteocr_table_engine_create.argtypes = []
lib.liteocr_table_engine_create.restype = ctypes.c_void_p

lib.liteocr_table_engine_destroy.argtypes = [ctypes.c_void_p]
lib.liteocr_table_engine_destroy.restype = None

lib.liteocr_table_engine_load_model.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(CTableModelPaths), ctypes.POINTER(CInferOption),
]
lib.liteocr_table_engine_load_model.restype = ctypes.c_int

lib.liteocr_table_engine_load_model_from_buffer.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(CTableModelBuffers), ctypes.POINTER(CInferOption),
]
lib.liteocr_table_engine_load_model_from_buffer.restype = ctypes.c_int

_table_recognize_args = [
    ctypes.c_void_p, ctypes.POINTER(CImage),
    ctypes.POINTER(CTextBox), ctypes.c_int,
    ctypes.POINTER(CTextLine), ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.POINTER(CRect)), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.POINTER(CTableCell)), ctypes.POINTER(ctypes.c_int),
]

lib.liteocr_table_engine_recognize_image.argtypes = _table_recognize_args
lib.liteocr_table_engine_recognize_image.restype = ctypes.c_int

lib.liteocr_table_engine_recognize_raw.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(CTextBox), ctypes.c_int,
    ctypes.POINTER(CTextLine), ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.POINTER(CRect)), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.POINTER(CTableCell)), ctypes.POINTER(ctypes.c_int),
]
lib.liteocr_table_engine_recognize_raw.restype = ctypes.c_int

lib.liteocr_table_engine_recognize_buffer.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
    ctypes.POINTER(CTextBox), ctypes.c_int,
    ctypes.POINTER(CTextLine), ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.POINTER(CRect)), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.POINTER(CTableCell)), ctypes.POINTER(ctypes.c_int),
]
lib.liteocr_table_engine_recognize_buffer.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Low-level model components
# ---------------------------------------------------------------------------

# Detector
lib.liteocr_detector_create.argtypes = []
lib.liteocr_detector_create.restype = ctypes.c_void_p

lib.liteocr_detector_destroy.argtypes = [ctypes.c_void_p]
lib.liteocr_detector_destroy.restype = None

lib.liteocr_detector_load_model.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(CInferOption),
]
lib.liteocr_detector_load_model.restype = ctypes.c_int

lib.liteocr_detector_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(CImage)]
lib.liteocr_detector_forward.restype = CImage

# Recognizer
lib.liteocr_recognizer_create.argtypes = []
lib.liteocr_recognizer_create.restype = ctypes.c_void_p

lib.liteocr_recognizer_destroy.argtypes = [ctypes.c_void_p]
lib.liteocr_recognizer_destroy.restype = None

lib.liteocr_recognizer_load_model.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(CInferOption),
]
lib.liteocr_recognizer_load_model.restype = ctypes.c_int

lib.liteocr_recognizer_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(CImage)]
lib.liteocr_recognizer_forward.restype = CImage

# Textline orientation
lib.liteocr_textline_ori_create.argtypes = []
lib.liteocr_textline_ori_create.restype = ctypes.c_void_p

lib.liteocr_textline_ori_destroy.argtypes = [ctypes.c_void_p]
lib.liteocr_textline_ori_destroy.restype = None

lib.liteocr_textline_ori_load_model.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(CInferOption),
]
lib.liteocr_textline_ori_load_model.restype = ctypes.c_int

lib.liteocr_textline_ori_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(CImage)]
lib.liteocr_textline_ori_forward.restype = ctypes.c_int

# Doc orientation
lib.liteocr_doc_ori_create.argtypes = []
lib.liteocr_doc_ori_create.restype = ctypes.c_void_p

lib.liteocr_doc_ori_destroy.argtypes = [ctypes.c_void_p]
lib.liteocr_doc_ori_destroy.restype = None

lib.liteocr_doc_ori_load_model.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(CInferOption),
]
lib.liteocr_doc_ori_load_model.restype = ctypes.c_int

lib.liteocr_doc_ori_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(CImage)]
lib.liteocr_doc_ori_forward.restype = ctypes.c_int

# UVDoc
lib.liteocr_uvdoc_create.argtypes = []
lib.liteocr_uvdoc_create.restype = ctypes.c_void_p

lib.liteocr_uvdoc_destroy.argtypes = [ctypes.c_void_p]
lib.liteocr_uvdoc_destroy.restype = None

lib.liteocr_uvdoc_load_model.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(CInferOption),
]
lib.liteocr_uvdoc_load_model.restype = ctypes.c_int

lib.liteocr_uvdoc_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(CImage)]
lib.liteocr_uvdoc_forward.restype = CImage

# SLANet
lib.liteocr_slanet_create.argtypes = []
lib.liteocr_slanet_create.restype = ctypes.c_void_p

lib.liteocr_slanet_destroy.argtypes = [ctypes.c_void_p]
lib.liteocr_slanet_destroy.restype = None

lib.liteocr_slanet_load_model.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.POINTER(CInferOption),
]
lib.liteocr_slanet_load_model.restype = ctypes.c_int

lib.liteocr_slanet_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(CImage),
    ctypes.POINTER(ctypes.POINTER(CTableCell)), ctypes.POINTER(ctypes.c_int),
]
lib.liteocr_slanet_forward.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# CTC decoder
# ---------------------------------------------------------------------------

lib.liteocr_ctc_decode.argtypes = [
    ctypes.POINTER(CImage), ctypes.c_int,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_int),
]
lib.liteocr_ctc_decode.restype = ctypes.c_int
