# C API 参考

公共 C API 定义在 `include/liteocr.h`。

## 基础类型

### `liteocr_image_t`

```c
typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
    int stride;
} liteocr_image_t;
```

字段说明：

- `data`：图像数据指针。
- `width` / `height`：图像宽高。
- `channels`：`1=Gray`、`3=BGR`、`4=BGRA`。
- `stride`：每行字节数。

由 LiteOCR 返回的图像需要调用：

```c
liteocr_free_image(&img);
```

### `liteocr_infer_option_t`

```c
typedef struct {
    int num_threads;
    int gpu_device_id;
    int use_fp16;
    int use_int8;
    int use_bf16;
    int textline_ori_model_type;
} liteocr_infer_option_t;
```

字段说明：

- `num_threads`：CPU 推理线程数。大于 0 时生效。
- `gpu_device_id`：GPU 设备 id，`-1` 表示 CPU。
- `use_fp16`：启用 fp16 相关推理选项。
- `use_int8`：启用 int8 相关推理选项。
- `use_bf16`：启用 bf16 storage/packed 相关推理选项。
- `textline_ori_model_type`：textline 方向分类器类型。
  - `LITEOCR_TEXTLINE_ORI_PADDLE`（默认，如 `PP-LCNet_x1_0_textline_ori`）
  - `LITEOCR_TEXTLINE_ORI_ANGLENET`（如 `Chineseocr_Lite_AngleNet`）

建议用零初始化：

```c
liteocr_infer_option_t opt = {};
opt.num_threads = 8;
opt.gpu_device_id = -1;
opt.use_bf16 = 1;
```

## 内存释放

| 返回内容 | 释放函数 |
| --- | --- |
| 普通 `malloc` 返回指针 | `liteocr_free` |
| 字符串 | `liteocr_free_string` |
| 图像 | `liteocr_free_image` |
| 文本行数组 | `liteocr_free_text_lines` |
| 文本框数组 | `liteocr_free_text_boxes` |
| 轮廓数组 | `liteocr_free_contours` |
| 表格结构数组 | `liteocr_free_table_cells` |

## 图像 I/O

```c
liteocr_image_t liteocr_imread(const char* filename, int desired_channels);
int liteocr_imwrite(const char* filename, const liteocr_image_t* img);
```

示例：

```c
liteocr_image_t img = liteocr_imread("test.png", 3);
if (!img.data) {
    return 1;
}

liteocr_imwrite("out.png", &img);
liteocr_free_image(&img);
```

## OCR 整体引擎

### 创建和销毁

```c
liteocr_engine_t liteocr_engine_create(void);
void liteocr_engine_destroy(liteocr_engine_t engine);
```

### 模型路径

```c
typedef struct {
    const char* det_param;
    const char* det_bin;
    const char* rec_param;
    const char* rec_bin;
    const char* vocab;
    const char* ori_param;
    const char* ori_bin;
} liteocr_ocr_model_paths_t;
```

`ori_param` 和 `ori_bin` 可传 `NULL`，表示不加载文本行方向模型。

### 加载模型

```c
int liteocr_engine_load_model(
    liteocr_engine_t engine,
    const liteocr_ocr_model_paths_t* paths,
    const liteocr_infer_option_t* opt);
```

### 识别图片

```c
int liteocr_engine_recognize_image(
    liteocr_engine_t engine,
    const liteocr_image_t* img,
    liteocr_text_box_t** out_boxes,
    int* out_box_count,
    liteocr_text_line_t** out_lines,
    int* out_line_count);
```

### 识别原始像素

```c
int liteocr_engine_recognize_raw(
    liteocr_engine_t engine,
    const unsigned char* data,
    int width,
    int height,
    int channels,
    int stride,
    liteocr_text_box_t** out_boxes,
    int* out_box_count,
    liteocr_text_line_t** out_lines,
    int* out_line_count);
```

### 识别图片文件缓冲区

```c
int liteocr_engine_recognize_buffer(
    liteocr_engine_t engine,
    const unsigned char* buffer,
    int size,
    liteocr_text_box_t** out_boxes,
    int* out_box_count,
    liteocr_text_line_t** out_lines,
    int* out_line_count);
```

`buffer` 应是完整图片文件内容，例如 PNG/JPEG 文件字节。

### 合并文本

```c
char* liteocr_merge_text_boxes(
    const liteocr_text_box_t* boxes,
    int box_count,
    const liteocr_text_line_t* lines,
    int line_count);
```

返回字符串需要用 `liteocr_free_string` 释放。

## 表格引擎

### 创建和销毁

```c
liteocr_table_engine_t liteocr_table_engine_create(void);
void liteocr_table_engine_destroy(liteocr_table_engine_t engine);
```

### 模型路径

```c
typedef struct {
    const char* cnn_param;
    const char* cnn_bin;
    const char* slahead_param;
    const char* slahead_bin;
    const char* vocab;
} liteocr_table_model_paths_t;
```

### 加载模型

```c
int liteocr_table_engine_load_model(
    liteocr_table_engine_t engine,
    const liteocr_table_model_paths_t* paths,
    const liteocr_infer_option_t* opt);
```

### 表格识别

```c
int liteocr_table_engine_recognize_buffer(
    liteocr_table_engine_t engine,
    const unsigned char* buffer,
    int size,
    const liteocr_text_box_t* boxes,
    int box_count,
    const liteocr_text_line_t* lines,
    int line_count,
    char** out_html,
    liteocr_rect_t** out_cells,
    int* out_cell_count,
    liteocr_table_cell_t** out_structure,
    int* out_structure_count);
```

输出说明：

- `out_html`：生成的 HTML 字符串，使用 `liteocr_free_string` 释放。
- `out_cells`：单元格矩形数组，使用 `liteocr_free` 释放。
- `out_structure`：表格结构数组，使用 `liteocr_free_table_cells` 释放。

当前高层表格接口依赖 OCR 结果作为输入，因此通常先调用 OCR 引擎获得 `boxes` 和 `lines`。

## 底层模型组件

LiteOCR 也允许单独调用底层模型。

### Detector

```c
liteocr_detector_t liteocr_detector_create(void);
void liteocr_detector_destroy(liteocr_detector_t det);
int liteocr_detector_load_model(liteocr_detector_t det, const char* param, const char* bin, const liteocr_infer_option_t* opt);
liteocr_image_t liteocr_detector_forward(liteocr_detector_t det, const liteocr_image_t* input);
```

`liteocr_detector_forward` 返回概率图图像，需要 `liteocr_free_image`。

### Recognizer

```c
liteocr_recognizer_t liteocr_recognizer_create(void);
void liteocr_recognizer_destroy(liteocr_recognizer_t rec);
int liteocr_recognizer_load_model(liteocr_recognizer_t rec, const char* param, const char* bin, const liteocr_infer_option_t* opt);
liteocr_image_t liteocr_recognizer_forward(liteocr_recognizer_t rec, const liteocr_image_t* input);
```

识别器输出可配合 `liteocr_ctc_decode` 解码。

### Textline Orientation

```c
liteocr_textline_ori_t liteocr_textline_ori_create(void);
void liteocr_textline_ori_destroy(liteocr_textline_ori_t ori);
int liteocr_textline_ori_load_model(liteocr_textline_ori_t ori, const char* param, const char* bin, const liteocr_infer_option_t* opt);
int liteocr_textline_ori_forward(liteocr_textline_ori_t ori, const liteocr_image_t* input);
```

### Doc Orientation

```c
liteocr_doc_ori_t liteocr_doc_ori_create(void);
void liteocr_doc_ori_destroy(liteocr_doc_ori_t ori);
int liteocr_doc_ori_load_model(liteocr_doc_ori_t ori, const char* param, const char* bin, const liteocr_infer_option_t* opt);
int liteocr_doc_ori_forward(liteocr_doc_ori_t ori, const liteocr_image_t* input);
```

### UVDoc

```c
liteocr_uvdoc_t liteocr_uvdoc_create(void);
void liteocr_uvdoc_destroy(liteocr_uvdoc_t uv);
int liteocr_uvdoc_load_model(liteocr_uvdoc_t uv, const char* param, const char* bin, const liteocr_infer_option_t* opt);
liteocr_image_t liteocr_uvdoc_forward(liteocr_uvdoc_t uv, const liteocr_image_t* input);
```

### SLANet

```c
liteocr_slanet_t liteocr_slanet_create(void);
void liteocr_slanet_destroy(liteocr_slanet_t sla);
int liteocr_slanet_load_model(
    liteocr_slanet_t sla,
    const char* cnn_param,
    const char* cnn_bin,
    const char* slahead_param,
    const char* slahead_bin,
    const char* vocab,
    const liteocr_infer_option_t* opt);
int liteocr_slanet_forward(
    liteocr_slanet_t sla,
    const liteocr_image_t* input,
    liteocr_table_cell_t** out_cells,
    int* out_count);
```

`out_cells` 需要用 `liteocr_free_table_cells` 释放。

## CTC 解码

```c
int liteocr_ctc_decode(
    const liteocr_image_t* probs,
    int blank_index,
    int** out_tokens,
    float** out_probs,
    int** out_indices,
    int* out_count);
```

输出数组分别用 `liteocr_free` 释放。

## 图像处理工具

常用函数：

- `liteocr_cvt_color`
- `liteocr_threshold`
- `liteocr_mean_masked`
- `liteocr_resize`
- `liteocr_rotate90`
- `liteocr_rotate180`
- `liteocr_copy_make_border`
- `liteocr_get_perspective_transform`
- `liteocr_warp_perspective`

## 轮廓工具

```c
int liteocr_find_contours(...);
void liteocr_min_area_rect(...);
liteocr_intrect_t liteocr_bounding_rect(...);
double liteocr_contour_area(...);
double liteocr_arc_length(...);
void liteocr_fill_poly(...);
```

`liteocr_find_contours` 的 `approx_mode`：

- `1`：`CHAIN_APPROX_NONE`
- `2`：`CHAIN_APPROX_SIMPLE`

