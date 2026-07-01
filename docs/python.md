# LiteOCR Python 封装

LiteOCR 现在提供了完整的 Python 封装，基于 `ctypes` 调用底层 C API。支持：

- 从源码安装：`pip install .`
- 打包成 wheel：`pip wheel . -w dist`

## 安装

```bash
# 从源码安装
pip install .

# 打包成 wheel
pip wheel . -w dist
```

编译需要 CMake（>=3.18）和 C++ 编译器。默认关闭 ncnn Vulkan，以获得更好的跨平台可移植性；如需 GPU 推理，可设置环境变量：

```bash
LITEOCR_ENABLE_VULKAN=1 pip install .
```

## 模型

模型文件不随 wheel 一起分发，默认托管在镜像：

```
https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/
```

Python 封装提供了**模型预设**，可以按名称自动下载并加载常用模型组合。

### 支持的 OCR 预设

| 预设名 | 说明 |
|--------|------|
| `PP-OCRv5_mobile` | 移动端 PP-OCRv5 |
| `PP-OCRv5_server` | 服务端 PP-OCRv5 |
| `PP-OCRv6_tiny` | PP-OCRv6 tiny |
| `PP-OCRv6_small` | PP-OCRv6 small |
| `PP-OCRv6_medium` | PP-OCRv6 medium |

### 支持的方向分类预设

| 预设名 | 说明 |
|--------|------|
| `PP-LCNet_textline_ori` | 文本行方向分类 |
| `PP-LCNet_doc_ori` | 文档方向分类 |
| `Chineseocr_AngleNet` | Chineseocr Lite AngleNet |

### 支持的表格预设

| 预设名 | 说明 |
|--------|------|
| `PP-StructureV2_SLANet_plus` | 表格结构解析 |

## 使用示例

### 使用预设自动下载并识别

```python
import liteocr

engine = liteocr.Engine()
engine.load_preset("PP-OCRv5_mobile", model_dir="models")

result = engine.recognize("test.png")
for line in result.lines:
    print(line.text)
```

首次调用会自动从镜像下载模型到 `models/` 目录，后续复用本地缓存。

### 手动指定模型路径

```python
import liteocr

engine = liteocr.Engine()
engine.load_model(
    det_param="models/PP-OCRv5_mobile_det.param",
    det_bin="models/PP-OCRv5_mobile_det.bin",
    rec_param="models/PP-OCRv5_mobile_rec.param",
    rec_bin="models/PP-OCRv5_mobile_rec.bin",
    vocab="models/PP-OCRv5_vocab.txt",
)

result = engine.recognize("test.png")
for line in result.lines:
    print(line.text)
```

### 从 NumPy 数组识别

```python
import liteocr
import numpy as np

engine = liteocr.Engine()
engine.load_preset("PP-OCRv5_mobile", model_dir="models")

arr = np.fromfile("test.raw", dtype=np.uint8).reshape((480, 640, 3))
result = engine.recognize(arr)
```

### 单独下载模型

```python
import liteocr

# 下载 OCR 预设
liteocr.download_preset("PP-OCRv5_mobile", model_dir="models")

# 下载方向分类预设
liteocr.download_orientation_preset("PP-LCNet_textline_ori", model_dir="models")

# 下载表格预设
liteocr.download_table_preset("PP-StructureV2_SLANet_plus", model_dir="models")

# 查看所有预设
print(liteocr.list_presets())
print(liteocr.list_orientation_presets())
print(liteocr.list_table_presets())
```

## 主要 API

### OCR 引擎

- `liteocr.Engine(opt=None)`：OCR 引擎。
  - `load_model(det_param, det_bin, rec_param, rec_bin, vocab, ori_param=None, ori_bin=None)`：加载模型。
  - `load_preset(name, model_dir="models", orientation=None, download=True)`：按预设名加载模型。
  - `recognize(image)`：识别图片，支持文件路径、`bytes`、NumPy 数组或 `liteocr.Image`。
- `liteocr.merge_text_boxes(boxes, lines)`：合并文本框为一段文本。

### 表格引擎

- `liteocr.TableEngine(opt=None)`：表格识别引擎。
  - `load_model(cnn_param, cnn_bin, slahead_param, slahead_bin, vocab)`：加载表格模型。
  - `load_preset(name, model_dir="models", download=True)`：按表格预设加载。
  - `recognize(image, ocr_result)`：根据 OCR 结果解析表格，返回 HTML / cells / structure。

### 底层模型组件

- `liteocr.Detector`：文本检测器。
- `liteocr.Recognizer`：文本识别器。
- `liteocr.TextlineOrientation`：文本行方向分类器。
- `liteocr.DocOrientation`：文档方向分类器。
- `liteocr.UVDoc`：文档扭曲校正。
- `liteocr.SLANet`：表格结构解析（独立组件）。

每个组件都提供 `load_model(param, bin)` 和 `forward(image)` 方法。

### 图像处理

- `liteocr.Image.from_file(path, desired_channels=3)`：加载图片。
- `liteocr.Image.from_numpy(array)`：将 NumPy 数组包装为图片。
- `liteocr.load_image(path)` / `liteocr.imwrite(path, image)`。
- `liteocr.cvt_color(src, dst_fmt, src_fmt)`：颜色空间转换。
- `liteocr.resize(src, dst_width, dst_height)`：缩放。
- `liteocr.rotate90(src, counter_clockwise=False)` / `liteocr.rotate180(src)`：旋转。
- `liteocr.copy_make_border(src, top, bottom, left, right, fill_value=0)`：加边框。
- `liteocr.threshold(src, thresh, maxval=255)`：阈值分割。
- `liteocr.mean_masked(src, mask)`：掩码均值。
- `liteocr.get_perspective_transform(src_pts, dst_pts)`：计算透视变换矩阵。
- `liteocr.warp_perspective(src, M, dst_width, dst_height)`：透视变换。

颜色格式常量：`COLOR_GRAY=1`、`COLOR_RGB=2`、`COLOR_BGR=3`、`COLOR_RGBA=4`、`COLOR_BGRA=5`。

### 轮廓处理

- `liteocr.find_contours(mask, approx_mode)`：查找轮廓。
- `liteocr.min_area_rect(contour)`：最小外接矩形。
- `liteocr.bounding_rect(contour)`：正向外接矩形。
- `liteocr.contour_area(contour)`：轮廓面积。
- `liteocr.arc_length(contour, closed=False)`：轮廓周长。
- `liteocr.fill_poly(shape, polygons, value=255)`：填充多边形。

近似模式常量：`CHAIN_APPROX_NONE=1`、`CHAIN_APPROX_SIMPLE=2`。

### 推理选项与工具

- `liteocr.InferOption`：推理选项，包括 `num_threads`、`gpu_device_id`、`use_fp16` 等。
- `liteocr.ctc_decode(probs, blank_index)`：CTC 解码。
- `liteocr.download_preset` / `liteocr.download_orientation_preset` / `liteocr.download_table_preset`：下载模型。
- `liteocr.list_presets` / `liteocr.list_orientation_presets` / `liteocr.list_table_presets`：列出预设。

## 运行测试

```bash
pip install .[dev]
pytest python/tests/test_basic.py
```

## CI

项目已配置 Python CI：`.github/workflows/python.yml`。

CI 会在 Windows 和 Ubuntu 上：

- 安装构建依赖；
- 执行 `pip install .[dev]`；
- 运行 `python -m liteocr` 冒烟测试；
- 运行 `pytest python/tests/test_basic.py`；
- 构建 wheel 并上传 artifact。

手动触发 CI 只需推送代码或提交 Pull Request。
