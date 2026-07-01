# 使用示例

示例代码以 C API 为主，接口定义见 `include/liteocr.h`。

> 示例中的 `./models/<文件名>` 需提前从镜像 `https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/` 下载到 `models/` 目录，详见 [`models/README.md`](../models/README.md)。

## OCR 图片识别

```c
#include "liteocr.h"
#include <stdio.h>

int main(void) {
    liteocr_engine_t engine = liteocr_engine_create();

    liteocr_ocr_model_paths_t paths = {
        "./models/PP-OCRv6_small_det.param",
        "./models/PP-OCRv6_small_det.bin",
        "./models/PP-OCRv6_small_rec.param",
        "./models/PP-OCRv6_small_rec.bin",
        "./models/PP-OCRv6_vocab.txt",
        "./models/PP-LCNet_x1_0_textline_ori.param",
        "./models/PP-LCNet_x1_0_textline_ori.bin"
    };

    liteocr_infer_option_t opt = {};
    opt.num_threads = 8;
    opt.gpu_device_id = -1;

    if (liteocr_engine_load_model(engine, &paths, &opt) != 0) {
        liteocr_engine_destroy(engine);
        return 1;
    }

    liteocr_image_t img = liteocr_imread("test.png", 3);
    if (!img.data) {
        liteocr_engine_destroy(engine);
        return 1;
    }

    liteocr_text_box_t* boxes = NULL;
    liteocr_text_line_t* lines = NULL;
    int box_count = 0;
    int line_count = 0;

    int ret = liteocr_engine_recognize_image(
        engine, &img,
        &boxes, &box_count,
        &lines, &line_count);

    if (ret == 0) {
        int n = box_count < line_count ? box_count : line_count;
        for (int i = 0; i < n; ++i) {
            printf("%s\n", lines[i].text ? lines[i].text : "");
        }
    }

    liteocr_free_text_boxes(boxes, box_count);
    liteocr_free_text_lines(lines, line_count);
    liteocr_free_image(&img);
    liteocr_engine_destroy(engine);
    return ret == 0 ? 0 : 1;
}
```

## 使用图片文件 buffer 识别

`liteocr_engine_recognize_buffer` 接收的是完整图片文件字节，不是裸像素数据。

```c
liteocr_text_box_t* boxes = NULL;
liteocr_text_line_t* lines = NULL;
int box_count = 0;
int line_count = 0;

liteocr_engine_recognize_buffer(
    engine,
    file_bytes,
    file_size,
    &boxes,
    &box_count,
    &lines,
    &line_count);
```

如果你已经有解码后的裸像素数据，使用：

```c
liteocr_engine_recognize_raw(
    engine,
    data,
    width,
    height,
    channels,
    stride,
    &boxes,
    &box_count,
    &lines,
    &line_count);
```

## 设置线程数和精度选项

```c
liteocr_infer_option_t opt = {};
opt.num_threads = 4;
opt.gpu_device_id = -1;
opt.use_fp16 = 0;
opt.use_int8 = 0;
opt.use_bf16 = 1;
```

说明：

- `num_threads` 控制 CPU 推理线程数。
- `gpu_device_id = -1` 使用 CPU。
- `use_bf16` 会启用 ncnn 的 bf16 storage/packed 选项。

## 表格识别

表格识别通常分两步：

1. 先用 OCR 引擎得到 `boxes` 和 `lines`。
2. 再把 OCR 结果传给表格引擎生成 HTML。

```c
liteocr_table_engine_t table_engine = liteocr_table_engine_create();

liteocr_table_model_paths_t table_paths = {
    "./models/PP-StructrureV2_SLANet_plus_cnn.param",
    "./models/PP-StructrureV2_SLANet_plus_cnn.bin",
    "./models/PP-StructrureV2_SLANet_plus_slahead.param",
    "./models/PP-StructrureV2_SLANet_plus_slahead.bin",
    "./models/table_structure_dict_ch.txt"
};

liteocr_table_engine_load_model(table_engine, &table_paths, NULL);

char* html = NULL;
liteocr_rect_t* cells = NULL;
int cell_count = 0;
liteocr_table_cell_t* structure = NULL;
int structure_count = 0;

liteocr_table_engine_recognize_image(
    table_engine,
    &img,
    boxes,
    box_count,
    lines,
    line_count,
    &html,
    &cells,
    &cell_count,
    &structure,
    &structure_count);

if (html) {
    printf("%s\n", html);
}

liteocr_free_string(html);
liteocr_free(cells);
liteocr_free_table_cells(structure, structure_count);
liteocr_table_engine_destroy(table_engine);
```

## 单独调用检测器

```c
liteocr_detector_t det = liteocr_detector_create();
liteocr_detector_load_model(
    det,
    "./models/PP-OCRv5_mobile_det.param",
    "./models/PP-OCRv5_mobile_det.bin",
    NULL);

liteocr_image_t prob = liteocr_detector_forward(det, &img);

liteocr_free_image(&prob);
liteocr_detector_destroy(det);
```

## 单独调用识别器和 CTC 解码

```c
liteocr_recognizer_t rec = liteocr_recognizer_create();
liteocr_recognizer_load_model(
    rec,
    "./models/PP-OCRv5_mobile_rec.param",
    "./models/PP-OCRv5_mobile_rec.bin",
    NULL);

liteocr_image_t probs = liteocr_recognizer_forward(rec, &line_img);

int* tokens = NULL;
float* token_probs = NULL;
int* indices = NULL;
int count = 0;

liteocr_ctc_decode(&probs, 0, &tokens, &token_probs, &indices, &count);

liteocr_free(tokens);
liteocr_free(token_probs);
liteocr_free(indices);
liteocr_free_image(&probs);
liteocr_recognizer_destroy(rec);
```

## 常见清理模板

```c
liteocr_free_text_boxes(boxes, box_count);
liteocr_free_text_lines(lines, line_count);
liteocr_free_image(&img);
liteocr_engine_destroy(engine);
```

空指针可以传给释放函数，释放函数会直接返回。

---

## Python 示例

完整可运行脚本位于 [`examples/python/`](../examples/python/)。

### 运行前准备

```bash
pip install .
```

### 基础图片识别

```python
import liteocr

engine = liteocr.Engine()
engine.load_preset("PP-OCRv5_mobile", model_dir="models")

result = engine.recognize("test.png")
for line in result.lines:
    print(line.text)
```

运行示例脚本：

```bash
python examples/python/example_ocr_file.py test.png
```

### 表格识别

```python
import liteocr

ocr_engine = liteocr.Engine()
ocr_engine.load_preset("PP-OCRv5_mobile", model_dir="models")
ocr_result = ocr_engine.recognize("table.png")

table_engine = liteocr.TableEngine()
table_engine.load_preset("PP-StructureV2_SLANet_plus", model_dir="models")
table_result = table_engine.recognize("table.png", ocr_result)

print(table_result.html)
```

运行示例脚本：

```bash
python examples/python/example_table.py table.png
```

### 批量识别

```bash
python examples/python/example_batch.py ./images --ext png,jpg
```

### 更多示例

- `example_ocr_numpy.py`：从 NumPy 数组识别。
- `example_manual_model.py`：手动指定模型路径。
- `example_detector_recognizer.py`：单独调用 Detector / Recognizer。
- `example_doc_orientation.py`：文档方向分类。
- `example_uvdoc.py`：UVDoc 文档畸变校正。

详细信息请参考 [`examples/python/README.md`](../examples/python/README.md) 中的说明。

