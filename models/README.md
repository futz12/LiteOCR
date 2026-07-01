# 模型文件下载

模型文件**不随 git 仓库分发**，需要从镜像单独下载后放到本目录（`models/`）：

```
https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/
```

每个文件的下载地址为镜像根地址拼接文件名，例如：

```
https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/PP-OCRv5_mobile_det.param
```

## 方式一：Python 自动下载（推荐）

Python 封装会在首次加载预设时按需下载缺失的模型到 `models/` 目录：

```python
import liteocr

engine = liteocr.Engine()
engine.load_preset("PP-OCRv5_mobile", model_dir="models")  # 自动下载
```

也可以只下载不加载：

```python
import liteocr

liteocr.download_preset("PP-OCRv5_mobile", model_dir="models")
liteocr.download_orientation_preset("PP-LCNet_textline_ori", model_dir="models")
liteocr.download_table_preset("PP-StructureV2_SLANet_plus", model_dir="models")
```

## 方式二：手动下载（C/C++ 用户）

C/C++ 示例通过 `./models/<文件名>` 直接读取模型文件，需要提前手动下载。
下载你需要的文件即可，例如 `example_baseocr` 用到的一组：

```bash
BASE="https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/"
cd models
for f in \
  PP-OCRv6_small_det.param PP-OCRv6_small_det.bin \
  PP-OCRv6_small_rec.param PP-OCRv6_small_rec.bin \
  PP-OCRv6_vocab.txt \
  PP-LCNet_x1_0_textline_ori.param PP-LCNet_x1_0_textline_ori.bin
do
  curl -fL -O "${BASE}${f}"
done
```

## 可用文件清单

镜像上提供以下模型（`.param` + `.bin` 成对，`int8` 为量化版本）：

- **OCR 检测/识别**
  - `PP-OCRv5_mobile_det` / `PP-OCRv5_mobile_rec`（含 `_int8`）
  - `PP-OCRv5_server_det` / `PP-OCRv5_server_rec`
  - `PP-OCRv6_tiny_det` / `PP-OCRv6_tiny_rec`（含 `_int8`）
  - `PP-OCRv6_small_det` / `PP-OCRv6_small_rec`（含 `_int8`）
  - `PP-OCRv6_medium_det` / `PP-OCRv6_medium_rec`（含 `_int8`）
- **文本行方向**：`PP-LCNet_x1_0_textline_ori`（含 `_int8`）、`Chineseocr_Lite_AngleNet`
- **文档方向**：`PP-LCNet_x1_0_doc_ori`（含 `_int8`）
- **表格结构**：`PP-StructrureV2_SLANet_plus_cnn`、`PP-StructrureV2_SLANet_plus_slahead`
- **文档畸变校正**：`PP-UVDoc`
- **词表 / 字典**：`PP-OCRv5_vocab.txt`、`PP-OCRv6_vocab.txt`、`PP-OCRv6_vocab_tiny.txt`、`table_structure_dict_ch.txt`
