#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 *  LiteOCR - Pure C API
 * ============================================================================ */

/* ---------- 基础类型 ---------- */

typedef struct {
    unsigned char* data;   /* 库返回的图像需调用 liteocr_free_image 释放 */
    int width;
    int height;
    int channels;          /* 1=Gray, 3=BGR, 4=BGRA */
    int stride;            /* bytes per row */
} liteocr_image_t;

typedef struct {
    int x;
    int y;
} liteocr_point_t;

typedef struct {
    float x;
    float y;
} liteocr_point2f_t;

typedef struct {
    int x;
    int y;
    int width;
    int height;
} liteocr_intrect_t;

/* TextBox: 用4个角点表示（tl, tr, br, bl） */
typedef struct {
    float points[8];       /* x0,y0, x1,y1, x2,y2, x3,y3 */
    int is_vertical;
    float score;
} liteocr_text_box_t;

typedef struct {
    char* text;            /* 需调用 liteocr_free_string 释放 */
    float* anchors;        /* 需调用 liteocr_free 释放 */
    int anchor_count;
} liteocr_text_line_t;

typedef struct {
    float x;
    float y;
    float width;
    float height;
} liteocr_rect_t;

typedef enum {
    LITEOCR_TEXTLINE_ORI_PADDLE = 0,    /* PP-LCNet 系列，默认 */
    LITEOCR_TEXTLINE_ORI_ANGLENET = 1   /* Chineseocr_Lite_AngleNet */
} liteocr_textline_ori_model_type_t;

typedef struct {
    int num_threads;
    int gpu_device_id;     /* -1 = CPU */
    int use_fp16;
    int use_int8;          /* 全局开关，为1时det和rec都使用INT8 */
    int use_int8_det;      /* 仅检测器使用INT8，use_int8为0时生效 */
    int use_int8_rec;      /* 仅识别器使用INT8，use_int8为0时生效 */
    int use_bf16;
    int textline_ori_model_type; /* liteocr_textline_ori_model_type_t */
} liteocr_infer_option_t;

typedef struct {
    liteocr_point_t* points;
    int point_count;
} liteocr_contour_t;

typedef struct {
    char* tag;             /* 如 "<td>", "<td></td>" 等 */
    float box[8];          /* x0,y0,x1,y1,x2,y2,x3,y3 */
} liteocr_table_cell_t;

/* ---------- Opaque Handles ---------- */

typedef struct liteocr_engine*        liteocr_engine_t;
typedef struct liteocr_table_engine*  liteocr_table_engine_t;
typedef struct liteocr_detector*      liteocr_detector_t;
typedef struct liteocr_recognizer*    liteocr_recognizer_t;
typedef struct liteocr_textline_ori*  liteocr_textline_ori_t;
typedef struct liteocr_doc_ori*       liteocr_doc_ori_t;
typedef struct liteocr_uvdoc*         liteocr_uvdoc_t;
typedef struct liteocr_slanet*        liteocr_slanet_t;

/* ---------- 模型路径结构体 ---------- */

typedef struct {
    const char* det_param;
    const char* det_bin;
    const char* rec_param;
    const char* rec_bin;
    const char* vocab;
    const char* ori_param;   /* 可选，NULL表示不加载方向模型 */
    const char* ori_bin;     /* 可选，NULL表示不加载方向模型 */
} liteocr_ocr_model_paths_t;

typedef struct {
    const char* det_param;
    const unsigned char* det_bin;
    const char* rec_param;
    const unsigned char* rec_bin;
    const char* vocab;
    const char* ori_param;          /* 可选，NULL表示不加载 */
    const unsigned char* ori_bin;   /* 可选，NULL表示不加载 */
} liteocr_ocr_model_buffers_t;

typedef struct {
    const char* cnn_param;
    const char* cnn_bin;
    const char* slahead_param;
    const char* slahead_bin;
    const char* vocab;
} liteocr_table_model_paths_t;

typedef struct {
    const char* cnn_param;
    const unsigned char* cnn_bin;
    const char* slahead_param;
    const unsigned char* slahead_bin;
    const char* vocab;
} liteocr_table_model_buffers_t;

/* ---------- 内存管理 ---------- */

void liteocr_free(void* ptr);
void liteocr_free_string(char* str);
void liteocr_free_image(liteocr_image_t* img);       /* 释放库创建的图像数据 */
void liteocr_free_text_lines(liteocr_text_line_t* lines, int count);
void liteocr_free_text_boxes(liteocr_text_box_t* boxes, int count);
void liteocr_free_contours(liteocr_contour_t* contours, int contour_count);
void liteocr_free_table_cells(liteocr_table_cell_t* cells, int count);

/* ---------- 图像 I/O ---------- */

liteocr_image_t liteocr_imread(const char* filename, int desired_channels);
int liteocr_imwrite(const char* filename, const liteocr_image_t* img);

/* ---------- 图像处理 ---------- */

/* cvtColor: src_fmt/dst_fmt: 1=GRAY, 2=RGB, 3=BGR, 4=RGBA, 5=BGRA */
void liteocr_cvt_color(const unsigned char* src, int src_w, int src_h, int src_step, int src_fmt,
                       unsigned char* dst, int dst_step, int dst_fmt);

void liteocr_threshold(const float* src, int w, int h, int src_step,
                       unsigned char* dst, int dst_step,
                       float thresh, unsigned char maxval);

double liteocr_mean_masked(const float* src, int w, int h, int src_step,
                           const unsigned char* mask, int mask_step);

void liteocr_resize(const unsigned char* src, int src_w, int src_h, int src_step, int channels,
                    unsigned char* dst, int dst_w, int dst_h, int dst_step);

void liteocr_rotate90(const unsigned char* src, int src_w, int src_h, int src_step, int channels,
                      unsigned char* dst, int dst_step, int counter_clockwise);
void liteocr_rotate180(const unsigned char* src, int src_w, int src_h, int src_step, int channels,
                       unsigned char* dst, int dst_step);

void liteocr_copy_make_border(const unsigned char* src, int src_w, int src_h, int src_step, int channels,
                              unsigned char* dst, int dst_w, int dst_h, int dst_step,
                              int top, int bottom, int left, int right,
                              unsigned char fill_value);

void liteocr_get_perspective_transform(const float src[8], const float dst[8], float M[9]);
void liteocr_warp_perspective(const unsigned char* src, int src_w, int src_h, int src_step, int channels,
                              unsigned char* dst, int dst_w, int dst_h, int dst_step,
                              const float M[9]);

/* ---------- 轮廓处理 ---------- */

/* approx_mode: 1=CHAIN_APPROX_NONE, 2=CHAIN_APPROX_SIMPLE */
int liteocr_find_contours(const unsigned char* data, int width, int height, int step,
                          liteocr_contour_t** out_contours, int* out_contour_count,
                          int approx_mode);

/* 返回4个角点（tl, tr, br, bl），用户预分配 liteocr_point2f_t pts[4] */
void liteocr_min_area_rect(const liteocr_point_t* contour, int point_count, liteocr_point2f_t out_pts[4]);

liteocr_intrect_t liteocr_bounding_rect(const liteocr_point_t* contour, int point_count);
double liteocr_contour_area(const liteocr_point_t* contour, int point_count);
double liteocr_arc_length(const liteocr_point_t* contour, int point_count, int closed);

void liteocr_fill_poly(unsigned char* data, int width, int height, int step,
                       const liteocr_contour_t* polygons, int polygon_count,
                       unsigned char value);

/* ---------- OCR 引擎 ---------- */

liteocr_engine_t liteocr_engine_create(void);
void liteocr_engine_destroy(liteocr_engine_t engine);

int liteocr_engine_load_model(liteocr_engine_t engine,
    const liteocr_ocr_model_paths_t* paths,
    const liteocr_infer_option_t* opt);

int liteocr_engine_load_model_from_buffer(liteocr_engine_t engine,
    const liteocr_ocr_model_buffers_t* buffers,
    const liteocr_infer_option_t* opt);

/* recognize系列：返回的boxes和lines需调用对应free函数释放 */
int liteocr_engine_recognize_image(liteocr_engine_t engine, const liteocr_image_t* img,
    liteocr_text_box_t** out_boxes, int* out_box_count,
    liteocr_text_line_t** out_lines, int* out_line_count);

int liteocr_engine_recognize_raw(liteocr_engine_t engine,
    const unsigned char* data, int width, int height, int channels, int stride,
    liteocr_text_box_t** out_boxes, int* out_box_count,
    liteocr_text_line_t** out_lines, int* out_line_count);

int liteocr_engine_recognize_buffer(liteocr_engine_t engine,
    const unsigned char* buffer, int size,
    liteocr_text_box_t** out_boxes, int* out_box_count,
    liteocr_text_line_t** out_lines, int* out_line_count);

/* 合并文本框为一个字符串，需调用 liteocr_free_string 释放 */
char* liteocr_merge_text_boxes(const liteocr_text_box_t* boxes, int box_count,
                                const liteocr_text_line_t* lines, int line_count);

/* ---------- 表格引擎 ---------- */

liteocr_table_engine_t liteocr_table_engine_create(void);
void liteocr_table_engine_destroy(liteocr_table_engine_t engine);

int liteocr_table_engine_load_model(liteocr_table_engine_t engine,
    const liteocr_table_model_paths_t* paths,
    const liteocr_infer_option_t* opt);

int liteocr_table_engine_load_model_from_buffer(liteocr_table_engine_t engine,
    const liteocr_table_model_buffers_t* buffers,
    const liteocr_infer_option_t* opt);

int liteocr_table_engine_recognize_image(liteocr_table_engine_t engine,
    const liteocr_image_t* img,
    const liteocr_text_box_t* boxes, int box_count,
    const liteocr_text_line_t* lines, int line_count,
    char** out_html,
    liteocr_rect_t** out_cells, int* out_cell_count,
    liteocr_table_cell_t** out_structure, int* out_structure_count);

int liteocr_table_engine_recognize_raw(liteocr_table_engine_t engine,
    const unsigned char* data, int width, int height, int channels, int stride,
    const liteocr_text_box_t* boxes, int box_count,
    const liteocr_text_line_t* lines, int line_count,
    char** out_html,
    liteocr_rect_t** out_cells, int* out_cell_count,
    liteocr_table_cell_t** out_structure, int* out_structure_count);

int liteocr_table_engine_recognize_buffer(liteocr_table_engine_t engine,
    const unsigned char* buffer, int size,
    const liteocr_text_box_t* boxes, int box_count,
    const liteocr_text_line_t* lines, int line_count,
    char** out_html,
    liteocr_rect_t** out_cells, int* out_cell_count,
    liteocr_table_cell_t** out_structure, int* out_structure_count);

/* ---------- 底层模型组件 ---------- */

/* Detector */
liteocr_detector_t liteocr_detector_create(void);
void liteocr_detector_destroy(liteocr_detector_t det);
int liteocr_detector_load_model(liteocr_detector_t det, const char* param, const char* bin, const liteocr_infer_option_t* opt);
int liteocr_detector_load_model_from_buffer(liteocr_detector_t det, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt);
/* forward 返回图像，需 liteocr_free_image */
liteocr_image_t liteocr_detector_forward(liteocr_detector_t det, const liteocr_image_t* input);

/* Recognizer */
liteocr_recognizer_t liteocr_recognizer_create(void);
void liteocr_recognizer_destroy(liteocr_recognizer_t rec);
int liteocr_recognizer_load_model(liteocr_recognizer_t rec, const char* param, const char* bin, const liteocr_infer_option_t* opt);
int liteocr_recognizer_load_model_from_buffer(liteocr_recognizer_t rec, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt);
liteocr_image_t liteocr_recognizer_forward(liteocr_recognizer_t rec, const liteocr_image_t* input);

/* Textline Orientation */
liteocr_textline_ori_t liteocr_textline_ori_create(void);
void liteocr_textline_ori_destroy(liteocr_textline_ori_t ori);
int liteocr_textline_ori_load_model(liteocr_textline_ori_t ori, const char* param, const char* bin, const liteocr_infer_option_t* opt);
int liteocr_textline_ori_load_model_from_buffer(liteocr_textline_ori_t ori, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt);
int liteocr_textline_ori_forward(liteocr_textline_ori_t ori, const liteocr_image_t* input);

/* Doc Orientation */
liteocr_doc_ori_t liteocr_doc_ori_create(void);
void liteocr_doc_ori_destroy(liteocr_doc_ori_t ori);
int liteocr_doc_ori_load_model(liteocr_doc_ori_t ori, const char* param, const char* bin, const liteocr_infer_option_t* opt);
int liteocr_doc_ori_load_model_from_buffer(liteocr_doc_ori_t ori, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt);
int liteocr_doc_ori_forward(liteocr_doc_ori_t ori, const liteocr_image_t* input);

/* UVDoc */
liteocr_uvdoc_t liteocr_uvdoc_create(void);
void liteocr_uvdoc_destroy(liteocr_uvdoc_t uv);
int liteocr_uvdoc_load_model(liteocr_uvdoc_t uv, const char* param, const char* bin, const liteocr_infer_option_t* opt);
int liteocr_uvdoc_load_model_from_buffer(liteocr_uvdoc_t uv, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt);
/* forward 返回图像，需 liteocr_free_image */
liteocr_image_t liteocr_uvdoc_forward(liteocr_uvdoc_t uv, const liteocr_image_t* input);

/* SLANet */
liteocr_slanet_t liteocr_slanet_create(void);
void liteocr_slanet_destroy(liteocr_slanet_t sla);
int liteocr_slanet_load_model(liteocr_slanet_t sla, const char* cnn_param, const char* cnn_bin,
                               const char* slahead_param, const char* slahead_bin,
                               const char* vocab,
                               const liteocr_infer_option_t* opt);
int liteocr_slanet_load_model_from_buffer(liteocr_slanet_t sla,
    const char* cnn_param_buf, const unsigned char* cnn_bin_buf,
    const char* slahead_param_buf, const unsigned char* slahead_bin_buf,
    const char* vocab_buf,
    const liteocr_infer_option_t* opt);
/* forward 返回cells需 liteocr_free_table_cells 释放 */
int liteocr_slanet_forward(liteocr_slanet_t sla, const liteocr_image_t* input,
    liteocr_table_cell_t** out_cells, int* out_count);

/* CTC Decoder */
int liteocr_ctc_decode(const liteocr_image_t* probs, int blank_index,
    int** out_tokens, float** out_probs, int** out_indices, int* out_count);

#ifdef __cplusplus
}
#endif
