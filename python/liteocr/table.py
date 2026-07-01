"""Table recognition engine wrapper."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List, Optional, Tuple, Union

from . import presets
from ._native import (
    CImage,
    CRect,
    CTableCell,
    CTableModelPaths,
    CTextBox,
    CTextLine,
    c_str,
    lib,
)
from ._types import OCRResult, TableCell, TableResult, InferOption
from .image import Image


def _to_c_boxes_lines(boxes, lines):
    """Convert Python TextBox/TextLine lists to C arrays."""
    c_boxes = (CTextBox * len(boxes))()
    for i, b in enumerate(boxes):
        for j, (x, y) in enumerate(b.points):
            c_boxes[i].points[j * 2] = x
            c_boxes[i].points[j * 2 + 1] = y
        c_boxes[i].is_vertical = int(b.is_vertical)
        c_boxes[i].score = b.score

    c_lines = (CTextLine * len(lines))()
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
        else:
            c_lines[i].anchors = None
    return c_boxes, c_lines


class TableEngine:
    """High-level wrapper around ``liteocr_table_engine_t``."""

    def __init__(self, opt: Optional[InferOption] = None):
        self._opt = opt if opt is not None else InferOption()
        self._c_opt = self._opt.to_c()
        self._handle = lib.liteocr_table_engine_create()
        if not self._handle:
            raise RuntimeError("Failed to create table engine")

    def load_model(
        self,
        cnn_param: Union[str, Path],
        cnn_bin: Union[str, Path],
        slahead_param: Union[str, Path],
        slahead_bin: Union[str, Path],
        vocab: Union[str, Path],
    ) -> "TableEngine":
        paths = CTableModelPaths(
            cnn_param=c_str(cnn_param),
            cnn_bin=c_str(cnn_bin),
            slahead_param=c_str(slahead_param),
            slahead_bin=c_str(slahead_bin),
            vocab=c_str(vocab),
        )
        rc = lib.liteocr_table_engine_load_model(
            self._handle, ctypes.byref(paths), ctypes.byref(self._c_opt)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load table model (error code {rc})")
        return self

    def load_preset(
        self,
        name: str,
        model_dir: str = "models",
        download: bool = True,
    ) -> "TableEngine":
        """Load a named table model preset, optionally downloading files."""
        if download:
            paths = presets.download_table_preset(name, model_dir)
        else:
            from pathlib import Path
            paths = {k: Path(model_dir) / v for k, v in presets._TABLE_PRESETS[name].items()}
        return self.load_model(
            cnn_param=paths["cnn_param"],
            cnn_bin=paths["cnn_bin"],
            slahead_param=paths["slahead_param"],
            slahead_bin=paths["slahead_bin"],
            vocab=paths["vocab"],
        )

    def _parse_table_result(
        self,
        html_ptr: ctypes.POINTER(ctypes.c_char),
        cells_ptr: ctypes.POINTER(CRect),
        cell_count: int,
        structure_ptr: ctypes.POINTER(CTableCell),
        structure_count: int,
    ) -> TableResult:
        html = html_ptr.value.decode("utf-8") if html_ptr and html_ptr.value else ""
        lib.liteocr_free_string(html_ptr)

        cells: List[Tuple[float, float, float, float]] = []
        for i in range(cell_count):
            cr = cells_ptr[i]
            cells.append((cr.x, cr.y, cr.width, cr.height))

        structure: List[TableCell] = []
        for i in range(structure_count):
            cc = structure_ptr[i]
            tag = cc.tag.decode("utf-8") if cc.tag else ""
            box = [(cc.box[j], cc.box[j + 1]) for j in range(0, 8, 2)]
            structure.append(TableCell(tag=tag, box=box))

        lib.liteocr_free(cells_ptr)
        lib.liteocr_free_table_cells(structure_ptr, structure_count)
        return TableResult(html=html, cells=cells, structure=structure)

    def recognize(
        self,
        image: Union[Image, str, Path],
        ocr_result: OCRResult,
    ) -> TableResult:
        """Run table recognition on an image using existing OCR boxes/lines."""
        if isinstance(image, (str, Path)):
            image = Image.from_file(image)
        return self._recognize_image(image, ocr_result.boxes, ocr_result.lines)

    def _recognize_image(
        self,
        image: Image,
        boxes: List["TextBox"],
        lines: List["TextLine"],
    ) -> TableResult:
        c_boxes, c_lines = _to_c_boxes_lines(boxes, lines)

        html_ptr = ctypes.c_char_p()
        cells_ptr = ctypes.POINTER(CRect)()
        cell_count = ctypes.c_int()
        structure_ptr = ctypes.POINTER(CTableCell)()
        structure_count = ctypes.c_int()

        rc = lib.liteocr_table_engine_recognize_image(
            self._handle,
            ctypes.byref(image._img),
            c_boxes, len(boxes),
            c_lines, len(lines),
            ctypes.byref(html_ptr),
            ctypes.byref(cells_ptr), ctypes.byref(cell_count),
            ctypes.byref(structure_ptr), ctypes.byref(structure_count),
        )

        # Free the text strings/anchors we allocated for lines.
        for cl in c_lines:
            if cl.text:
                lib.liteocr_free_string(cl.text)
            if cl.anchors:
                lib.liteocr_free(cl.anchors)

        if rc != 0:
            raise RuntimeError(f"Table recognition failed (error code {rc})")

        return self._parse_table_result(
            html_ptr, cells_ptr, cell_count.value, structure_ptr, structure_count.value
        )

    def __del__(self):
        if getattr(self, "_handle", None):
            lib.liteocr_table_engine_destroy(self._handle)
            self._handle = None
