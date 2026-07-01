"""Model presets and automatic download helper.

LiteOCR model files are hosted at
``https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/``.  This module lets you
refer to a commonly used model set by a short name and automatically download
missing files.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

MIRROR_BASE = "https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/"

_OCR_PRESETS: Dict[str, Dict[str, str]] = {
    "PP-OCRv5_mobile": {
        "det_param": "PP-OCRv5_mobile_det.param",
        "det_bin": "PP-OCRv5_mobile_det.bin",
        "rec_param": "PP-OCRv5_mobile_rec.param",
        "rec_bin": "PP-OCRv5_mobile_rec.bin",
        "vocab": "PP-OCRv5_vocab.txt",
    },
    "PP-OCRv5_server": {
        "det_param": "PP-OCRv5_server_det.param",
        "det_bin": "PP-OCRv5_server_det.bin",
        "rec_param": "PP-OCRv5_server_rec.param",
        "rec_bin": "PP-OCRv5_server_rec.bin",
        "vocab": "PP-OCRv5_vocab.txt",
    },
    "PP-OCRv6_tiny": {
        "det_param": "PP-OCRv6_tiny_det.param",
        "det_bin": "PP-OCRv6_tiny_det.bin",
        "rec_param": "PP-OCRv6_tiny_rec.param",
        "rec_bin": "PP-OCRv6_tiny_rec.bin",
        "vocab": "PP-OCRv6_vocab.txt",
    },
    "PP-OCRv6_small": {
        "det_param": "PP-OCRv6_small_det.param",
        "det_bin": "PP-OCRv6_small_det.bin",
        "rec_param": "PP-OCRv6_small_rec.param",
        "rec_bin": "PP-OCRv6_small_rec.bin",
        "vocab": "PP-OCRv6_vocab.txt",
    },
    "PP-OCRv6_medium": {
        "det_param": "PP-OCRv6_medium_det.param",
        "det_bin": "PP-OCRv6_medium_det.bin",
        "rec_param": "PP-OCRv6_medium_rec.param",
        "rec_bin": "PP-OCRv6_medium_rec.bin",
        "vocab": "PP-OCRv6_vocab.txt",
    },
}

_ORIENTATION_PRESETS: Dict[str, Dict[str, str]] = {
    "PP-LCNet_textline_ori": {
        "ori_param": "PP-LCNet_x1_0_textline_ori.param",
        "ori_bin": "PP-LCNet_x1_0_textline_ori.bin",
    },
    "PP-LCNet_doc_ori": {
        "ori_param": "PP-LCNet_x1_0_doc_ori.param",
        "ori_bin": "PP-LCNet_x1_0_doc_ori.bin",
    },
    "Chineseocr_AngleNet": {
        "ori_param": "Chineseocr_Lite_AngleNet.param",
        "ori_bin": "Chineseocr_Lite_AngleNet.bin",
    },
}

_TABLE_PRESETS: Dict[str, Dict[str, str]] = {
    "PP-StructureV2_SLANet_plus": {
        "cnn_param": "PP-StructrureV2_SLANet_plus_cnn.param",
        "cnn_bin": "PP-StructrureV2_SLANet_plus_cnn.bin",
        "slahead_param": "PP-StructrureV2_SLANet_plus_slahead.param",
        "slahead_bin": "PP-StructrureV2_SLANet_plus_slahead.bin",
        "vocab": "table_structure_dict_ch.txt",
    },
}


def list_presets() -> List[str]:
    """Return all supported OCR preset names."""
    return list(_OCR_PRESETS.keys())


def list_orientation_presets() -> List[str]:
    """Return all supported orientation preset names."""
    return list(_ORIENTATION_PRESETS.keys())


def list_table_presets() -> List[str]:
    """Return all supported table preset names."""
    return list(_TABLE_PRESETS.keys())


def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download ``url`` to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out_file:
        shutil.copyfileobj(response, out_file, chunk_size)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")


def _resolve_paths(
    preset_files: Dict[str, str],
    model_dir: Path,
    overwrite: bool,
) -> Dict[str, Path]:
    """Download missing files and return local paths keyed by role."""
    local: Dict[str, Path] = {}
    for role, filename in preset_files.items():
        dest = model_dir / filename
        if overwrite or not dest.exists():
            _download_file(MIRROR_BASE + filename, dest)
        local[role] = dest
    return local


def download_preset(
    name: str,
    model_dir: str = "models",
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Download the files for an OCR preset.

    Args:
        name: Preset name, e.g. ``"PP-OCRv5_mobile"``.
        model_dir: Directory in which to store the downloaded models.
        overwrite: Re-download files even if they already exist.

    Returns:
        Mapping from role (``det_param``, ``det_bin``, ...) to local path.
    """
    if name not in _OCR_PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. "
            f"Available presets: {', '.join(list_presets())}"
        )
    return _resolve_paths(_OCR_PRESETS[name], Path(model_dir), overwrite)


def download_orientation_preset(
    name: str,
    model_dir: str = "models",
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Download the files for an orientation-classification preset."""
    if name not in _ORIENTATION_PRESETS:
        raise ValueError(
            f"Unknown orientation preset {name!r}. "
            f"Available presets: {', '.join(list_orientation_presets())}"
        )
    return _resolve_paths(_ORIENTATION_PRESETS[name], Path(model_dir), overwrite)


def download_table_preset(
    name: str,
    model_dir: str = "models",
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Download the files for a table recognition preset."""
    if name not in _TABLE_PRESETS:
        raise ValueError(
            f"Unknown table preset {name!r}. "
            f"Available presets: {', '.join(list_table_presets())}"
        )
    return _resolve_paths(_TABLE_PRESETS[name], Path(model_dir), overwrite)


def ensure_preset(
    name: str,
    model_dir: str = "models",
    orientation: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Download an OCR preset and optionally an orientation preset.

    Returns a single dictionary that can be passed directly to
    ``Engine.load_model(**paths)``.
    """
    paths = download_preset(name, model_dir, overwrite)
    if orientation:
        ori_paths = download_orientation_preset(orientation, model_dir, overwrite)
        paths.update(ori_paths)
    return paths
