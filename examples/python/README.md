# LiteOCR Python 示例

本目录包含可直接运行的 Python 示例脚本，展示 LiteOCR Python 封装的常见用法。

## 运行前准备

```bash
pip install .
```

部分示例会使用预设自动下载模型到 `models/` 目录；首次运行需要联网。

## 示例列表

| 脚本 | 说明 |
|------|------|
| `example_ocr_file.py` | 最基础的图片 OCR 识别 |
| `example_ocr_numpy.py` | 从 NumPy 数组进行识别 |
| `example_manual_model.py` | 手动指定模型路径加载 |
| `example_table.py` | 表格识别（OCR + TableEngine） |
| `example_detector_recognizer.py` | 单独使用 Detector / Recognizer 组件 |
| `example_doc_orientation.py` | 文档方向分类 |
| `example_uvdoc.py` | UVDoc 文档畸变校正 |
| `example_batch.py` | 批量识别目录下的图片 |

## 快速体验

```bash
python examples/python/example_ocr_file.py test.png
```

默认使用 `PP-OCRv5_mobile` 预设。也可以指定其它预设：

```bash
python examples/python/example_ocr_file.py test.png --preset PP-OCRv6_small
```
