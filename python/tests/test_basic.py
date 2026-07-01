"""Basic import and API smoke tests for the liteocr package."""

import os

import liteocr


def test_package_version():
    assert isinstance(liteocr.__version__, str)


def test_native_library_loaded():
    assert liteocr.lib is not None
    assert os.path.exists(liteocr.lib._name)


def test_engine_create():
    engine = liteocr.Engine()
    assert engine._handle is not None
    del engine


def test_infer_option_defaults():
    opt = liteocr.InferOption()
    assert opt.num_threads == 4
    assert opt.gpu_device_id == -1


def test_presets_listed():
    assert "PP-OCRv5_mobile" in liteocr.list_presets()
    assert "PP-LCNet_textline_ori" in liteocr.list_orientation_presets()
    assert "PP-StructureV2_SLANet_plus" in liteocr.list_table_presets()


def test_components_importable():
    assert liteocr.Detector is not None
    assert liteocr.Recognizer is not None
    assert liteocr.TextlineOrientation is not None
    assert liteocr.DocOrientation is not None
    assert liteocr.UVDoc is not None
    assert liteocr.SLANet is not None
    assert liteocr.TableEngine is not None


def test_imgproc_constants():
    assert liteocr.COLOR_GRAY == 1
    assert liteocr.COLOR_RGB == 2
    assert liteocr.COLOR_BGR == 3


def test_contour_constants():
    assert liteocr.CHAIN_APPROX_NONE == 1
    assert liteocr.CHAIN_APPROX_SIMPLE == 2
