"""Low-level LiteOCR model components (detector, recognizer, orientation, etc.)."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Union

from ._native import CImage, CTableCell, CLatexOCRModelPaths, c_str, lib
from ._types import InferOption, TableCell
from .image import Image
from . import presets


class _BaseComponent:
    """Base class for components backed by a C handle."""

    _create_fn = None
    _destroy_fn = None

    def __init__(self, opt: Optional[InferOption] = None):
        self._opt = opt if opt is not None else InferOption()
        self._c_opt = self._opt.to_c()
        self._handle = self._create_fn()
        if not self._handle:
            raise RuntimeError(f"Failed to create {self.__class__.__name__}")

    def __del__(self):
        if getattr(self, "_handle", None):
            self._destroy_fn(self._handle)
            self._handle = None


class Detector(_BaseComponent):
    """Standalone text detector."""

    _create_fn = lib.liteocr_detector_create
    _destroy_fn = lib.liteocr_detector_destroy

    def load_model(
        self,
        param: Union[str, Path],
        bin: Union[str, Path],
    ) -> "Detector":
        rc = lib.liteocr_detector_load_model(
            self._handle, c_str(param), c_str(bin), ctypes.byref(self._c_opt)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load detector model (error code {rc})")
        return self

    def forward(self, image: Image) -> Image:
        c_out = lib.liteocr_detector_forward(self._handle, ctypes.byref(image._img))
        if c_out.data is None:
            raise RuntimeError("Detector forward failed")
        return Image(c_out, owns_data=True)


class Recognizer(_BaseComponent):
    """Standalone text recognizer."""

    _create_fn = lib.liteocr_recognizer_create
    _destroy_fn = lib.liteocr_recognizer_destroy

    def load_model(
        self,
        param: Union[str, Path],
        bin: Union[str, Path],
    ) -> "Recognizer":
        rc = lib.liteocr_recognizer_load_model(
            self._handle, c_str(param), c_str(bin), ctypes.byref(self._c_opt)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load recognizer model (error code {rc})")
        return self

    def forward(self, image: Image) -> Image:
        c_out = lib.liteocr_recognizer_forward(self._handle, ctypes.byref(image._img))
        if c_out.data is None:
            raise RuntimeError("Recognizer forward failed")
        return Image(c_out, owns_data=True)


class TextlineOrientation(_BaseComponent):
    """Text-line orientation classifier."""

    _create_fn = lib.liteocr_textline_ori_create
    _destroy_fn = lib.liteocr_textline_ori_destroy

    def load_model(
        self,
        param: Union[str, Path],
        bin: Union[str, Path],
    ) -> "TextlineOrientation":
        rc = lib.liteocr_textline_ori_load_model(
            self._handle, c_str(param), c_str(bin), ctypes.byref(self._c_opt)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load textline orientation model (error code {rc})")
        return self

    def forward(self, image: Image) -> int:
        return int(lib.liteocr_textline_ori_forward(self._handle, ctypes.byref(image._img)))


class DocOrientation(_BaseComponent):
    """Document orientation classifier."""

    _create_fn = lib.liteocr_doc_ori_create
    _destroy_fn = lib.liteocr_doc_ori_destroy

    def load_model(
        self,
        param: Union[str, Path],
        bin: Union[str, Path],
    ) -> "DocOrientation":
        rc = lib.liteocr_doc_ori_load_model(
            self._handle, c_str(param), c_str(bin), ctypes.byref(self._c_opt)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load doc orientation model (error code {rc})")
        return self

    def forward(self, image: Image) -> int:
        return int(lib.liteocr_doc_ori_forward(self._handle, ctypes.byref(image._img)))


class UVDoc(_BaseComponent):
    """UVDoc document unwarping model."""

    _create_fn = lib.liteocr_uvdoc_create
    _destroy_fn = lib.liteocr_uvdoc_destroy

    def load_model(
        self,
        param: Union[str, Path],
        bin: Union[str, Path],
    ) -> "UVDoc":
        rc = lib.liteocr_uvdoc_load_model(
            self._handle, c_str(param), c_str(bin), ctypes.byref(self._c_opt)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load UVDoc model (error code {rc})")
        return self

    def forward(self, image: Image) -> Image:
        c_out = lib.liteocr_uvdoc_forward(self._handle, ctypes.byref(image._img))
        if c_out.data is None:
            raise RuntimeError("UVDoc forward failed")
        return Image(c_out, owns_data=True)


class SLANet(_BaseComponent):
    """SLANet table structure parser."""

    _create_fn = lib.liteocr_slanet_create
    _destroy_fn = lib.liteocr_slanet_destroy

    def load_model(
        self,
        cnn_param: Union[str, Path],
        cnn_bin: Union[str, Path],
        slahead_param: Union[str, Path],
        slahead_bin: Union[str, Path],
        vocab: Union[str, Path],
    ) -> "SLANet":
        rc = lib.liteocr_slanet_load_model(
            self._handle,
            c_str(cnn_param), c_str(cnn_bin),
            c_str(slahead_param), c_str(slahead_bin),
            c_str(vocab),
            ctypes.byref(self._c_opt),
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load SLANet model (error code {rc})")
        return self

    def forward(self, image: Image) -> List[TableCell]:
        cells_ptr = ctypes.POINTER(CTableCell)()
        count = ctypes.c_int()
        rc = lib.liteocr_slanet_forward(
            self._handle, ctypes.byref(image._img),
            ctypes.byref(cells_ptr), ctypes.byref(count),
        )
        if rc != 0:
            raise RuntimeError(f"SLANet forward failed (error code {rc})")

        cells: List[TableCell] = []
        for i in range(count.value):
            cc = cells_ptr[i]
            tag = cc.tag.decode("utf-8") if cc.tag else ""
            box = [(cc.box[j], cc.box[j + 1]) for j in range(0, 8, 2)]
            cells.append(TableCell(tag=tag, box=box))

        lib.liteocr_free_table_cells(cells_ptr, count.value)
        return cells


class LatexOCR:
    """PP-FormulaNet formula recognizer: image -> LaTeX string."""

    def __init__(self, opt: Optional[InferOption] = None):
        self._opt = opt if opt is not None else InferOption()
        self._c_opt = self._opt.to_c()
        self._handle = lib.liteocr_latexocr_create()
        if not self._handle:
            raise RuntimeError("Failed to create LatexOCR engine")

    def load_model(
        self,
        encoder_param: Union[str, Path],
        encoder_bin: Union[str, Path],
        embed_param: Union[str, Path],
        embed_bin: Union[str, Path],
        decoder_param: Union[str, Path],
        decoder_bin: Union[str, Path],
        vocab: Union[str, Path],
    ) -> "LatexOCR":
        """Load encoder/embed/decoder model files and the tokenizer vocab."""
        paths = CLatexOCRModelPaths(
            encoder_param=c_str(encoder_param),
            encoder_bin=c_str(encoder_bin),
            embed_param=c_str(embed_param),
            embed_bin=c_str(embed_bin),
            decoder_param=c_str(decoder_param),
            decoder_bin=c_str(decoder_bin),
            vocab=c_str(vocab),
        )
        rc = lib.liteocr_latexocr_load_model(
            self._handle, ctypes.byref(paths), ctypes.byref(self._c_opt)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load LatexOCR model (error code {rc})")
        return self

    def load_preset(
        self,
        name: str,
        model_dir: str = "models",
        download: bool = True,
    ) -> "LatexOCR":
        """Load a named LaTeX formula recognition preset, optionally downloading missing files."""
        if download:
            paths = presets.download_latexocr_preset(name, model_dir)
        else:
            paths = {k: Path(model_dir) / v for k, v in presets._LATEXOCR_PRESETS[name].items()}
        return self.load_model(
            encoder_param=paths["encoder_param"],
            encoder_bin=paths["encoder_bin"],
            embed_param=paths["embed_param"],
            embed_bin=paths["embed_bin"],
            decoder_param=paths["decoder_param"],
            decoder_bin=paths["decoder_bin"],
            vocab=paths["vocab"],
        )

    def recognize(self, image: Union[Image, Any, str, os.PathLike]) -> str:
        """Recognize a formula image and return a LaTeX string.

        ``image`` may be:
        * an :class:`Image` instance,
        * a file path string,
        * a NumPy array (H, W) or (H, W, C) uint8.
        """
        if isinstance(image, Image):
            return self._recognize_image(image)
        if isinstance(image, (str, os.PathLike)):
            return self._recognize_image(Image.from_file(image))

        img = Image.from_numpy(image)
        return self._recognize_image(img)

    def _recognize_image(self, image: Image) -> str:
        text_ptr = lib.liteocr_latexocr_recognize_image(
            self._handle, ctypes.byref(image._img)
        )
        if text_ptr is None:
            raise RuntimeError("LatexOCR recognition failed")
        text = ctypes.cast(text_ptr, ctypes.c_char_p).value.decode("utf-8")
        lib.liteocr_free_string(ctypes.c_char_p(text_ptr))
        return text

    def _recognize_raw(self, image: Image) -> str:
        text_ptr = lib.liteocr_latexocr_recognize_raw(
            self._handle,
            image._img.data,
            image._img.width,
            image._img.height,
            image._img.channels,
            image._img.stride,
        )
        if text_ptr is None:
            raise RuntimeError("LatexOCR recognition failed")
        text = ctypes.cast(text_ptr, ctypes.c_char_p).value.decode("utf-8")
        lib.liteocr_free_string(ctypes.c_char_p(text_ptr))
        return text

    def __del__(self):
        if getattr(self, "_handle", None):
            lib.liteocr_latexocr_destroy(self._handle)
            self._handle = None
