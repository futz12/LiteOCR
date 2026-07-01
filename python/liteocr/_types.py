"""Common data classes used by the LiteOCR Python wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ._native import CInferOption


@dataclass
class InferOption:
    """Inference options passed to engines and low-level components."""

    num_threads: int = 4
    gpu_device_id: int = -1
    use_fp16: bool = False
    use_int8: bool = False
    use_int8_det: bool = False
    use_int8_rec: bool = False
    use_bf16: bool = False
    textline_ori_model_type: int = 0

    def to_c(self) -> CInferOption:
        return CInferOption(
            num_threads=self.num_threads,
            gpu_device_id=self.gpu_device_id,
            use_fp16=int(self.use_fp16),
            use_int8=int(self.use_int8),
            use_int8_det=int(self.use_int8_det),
            use_int8_rec=int(self.use_int8_rec),
            use_bf16=int(self.use_bf16),
            textline_ori_model_type=self.textline_ori_model_type,
        )


@dataclass
class TextBox:
    """A detected text box with four corner points."""

    points: List[Tuple[float, float]]
    is_vertical: bool
    score: float


@dataclass
class TextLine:
    """A recognized text line with optional anchor points."""

    text: str
    anchors: List[Tuple[float, float]]


@dataclass
class OCRResult:
    """Result returned by ``Engine.recognize``."""

    boxes: List[TextBox]
    lines: List[TextLine]


@dataclass
class TableCell:
    """A table cell parsed by the table engine."""

    tag: str
    box: List[Tuple[float, float]]


@dataclass
class TableResult:
    """Result returned by ``TableEngine.recognize``."""

    html: str
    cells: List[Tuple[float, float, float, float]]
    structure: List[TableCell]
