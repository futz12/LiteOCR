#include "liteocr.h"

#include "liteocr_engine.h"
#include "liteocr_docinfer.h"
#include "liteocr_image.h"
#include "liteocr_imgproc.h"
#include "contours/liteocr_contours.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <array>

#ifdef _WIN32
#define LITEOCR_STRDUP _strdup
#else
#define LITEOCR_STRDUP strdup
#endif

/* ============================================================================
 *  Internal helper: C++ liteocr_image <-> C image
 * ============================================================================ */

static liteocr_image_t imageToC(const liteocr_image& img) {
    liteocr_image_t result = {};
    if (img.empty()) return result;
    if (img.width <= 0 || img.height <= 0 || img.channels <= 0 || img.stride <= 0) return result;

    size_t row_bytes = static_cast<size_t>(img.width) * static_cast<size_t>(img.elem_size());
    size_t stride = static_cast<size_t>(img.stride);
    if (row_bytes > stride) return result;
    if (static_cast<size_t>(img.height) > std::numeric_limits<size_t>::max() / stride) return result;

    result.width = img.width;
    result.height = img.height;
    result.channels = img.channels;
    result.stride = img.stride;
    size_t total = static_cast<size_t>(img.height) * stride;
    result.data = (unsigned char*)malloc(total);
    if (!result.data) {
        return {};
    }
    for (int y = 0; y < img.height; ++y) {
        std::memcpy(result.data + y * result.stride, img.data + y * img.stride, row_bytes);
    }
    return result;
}

static liteocr_image imageFromC(const liteocr_image_t* img) {
    if (!img || !img->data || img->width <= 0 || img->height <= 0) return liteocr_image();
    if (img->channels != 1 && img->channels != 3 && img->channels != 4) return liteocr_image();
    if (img->width > std::numeric_limits<int>::max() / img->channels) return liteocr_image();
    if (img->stride < img->width * img->channels) return liteocr_image();
    liteocr_image_type t = (img->channels == 1) ? liteocr_image_type::LITEOCR_IMAGE_U8C1 :
                           (img->channels == 3) ? liteocr_image_type::LITEOCR_IMAGE_U8C3 : liteocr_image_type::LITEOCR_IMAGE_U8C4;
    return liteocr_image(img->width, img->height, t, const_cast<unsigned char*>(img->data), img->stride);
}

/* ============================================================================
 *  Internal helper: liteocr_infer_option conversion
 * ============================================================================ */

static liteocr_infer_option optionFromC(const liteocr_infer_option_t* opt) {
    liteocr_infer_option result;
    if (opt) {
        if (opt->num_threads > 0) result.num_threads = opt->num_threads;
        if (opt->gpu_device_id >= 0) result.gpu_device_id = opt->gpu_device_id;
        result.use_fp16 = opt->use_fp16 != 0;
        result.use_int8 = opt->use_int8 != 0;
        result.use_int8_det = opt->use_int8_det != 0;
        result.use_int8_rec = opt->use_int8_rec != 0;
        result.use_bf16 = opt->use_bf16 != 0;
        result.textline_ori_model_type = opt->textline_ori_model_type;
    }
    return result;
}

/* ============================================================================
 *  Internal helper: liteocr_text_box / TextLine conversions
 * ============================================================================ */

static void convertTextBox(const liteocr_text_box& src, liteocr_text_box_t& dst) {
    for (int i = 0; i < 8; ++i) {
        dst.points[i] = src.points[i];
    }
    dst.is_vertical = src.is_vertical ? 1 : 0;
    dst.score = src.score;
}

static liteocr_text_box textBoxFromC(const liteocr_text_box_t* src) {
    liteocr_text_box dst;
    for (int i = 0; i < 8; ++i) {
        dst.points[i] = src->points[i];
    }
    dst.is_vertical = src->is_vertical != 0;
    dst.score = src->score;
    return dst;
}

static int copyOcrResultToC(const std::vector<liteocr_text_box>& boxes,
    const std::vector<liteocr_text_line>& lines,
    liteocr_text_box_t** out_boxes, int* out_box_count,
    liteocr_text_line_t** out_lines, int* out_line_count) {

    *out_boxes = nullptr;
    *out_box_count = 0;
    *out_lines = nullptr;
    *out_line_count = 0;

    if (boxes.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        lines.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return -1;
    }

    if (!boxes.empty()) {
        *out_boxes = (liteocr_text_box_t*)malloc(sizeof(liteocr_text_box_t) * boxes.size());
        if (!*out_boxes) return -1;
        for (size_t i = 0; i < boxes.size(); ++i) {
            convertTextBox(boxes[i], (*out_boxes)[i]);
        }
        *out_box_count = (int)boxes.size();
    }

    if (!lines.empty()) {
        *out_lines = (liteocr_text_line_t*)malloc(sizeof(liteocr_text_line_t) * lines.size());
        if (!*out_lines) {
            free(*out_boxes);
            *out_boxes = nullptr;
            *out_box_count = 0;
            return -1;
        }
        std::memset(*out_lines, 0, sizeof(liteocr_text_line_t) * lines.size());

        for (size_t i = 0; i < lines.size(); ++i) {
            (*out_lines)[i].text = LITEOCR_STRDUP(lines[i].text.c_str());
            if (!(*out_lines)[i].text) {
                liteocr_free_text_lines(*out_lines, (int)lines.size());
                free(*out_boxes);
                *out_lines = nullptr;
                *out_boxes = nullptr;
                *out_box_count = 0;
                return -1;
            }

            (*out_lines)[i].anchor_count = (int)lines[i].anchors.size();
            if (!lines[i].anchors.empty()) {
                size_t sz = sizeof(float) * lines[i].anchors.size();
                (*out_lines)[i].anchors = (float*)malloc(sz);
                if (!(*out_lines)[i].anchors) {
                    liteocr_free_text_lines(*out_lines, (int)lines.size());
                    free(*out_boxes);
                    *out_lines = nullptr;
                    *out_boxes = nullptr;
                    *out_box_count = 0;
                    return -1;
                }
                std::memcpy((*out_lines)[i].anchors, lines[i].anchors.data(), sz);
            }
        }
        *out_line_count = (int)lines.size();
    }

    return 0;
}

/* ============================================================================
 *  Memory management
 * ============================================================================ */

void liteocr_free(void* ptr) {
    free(ptr);
}

void liteocr_free_string(char* str) {
    free(str);
}

void liteocr_free_image(liteocr_image_t* img) {
    if (!img) return;
    free(img->data);
    img->data = nullptr;
    img->width = img->height = img->channels = img->stride = 0;
}

void liteocr_free_text_lines(liteocr_text_line_t* lines, int count) {
    if (!lines) return;
    for (int i = 0; i < count; ++i) {
        free(lines[i].text);
        free(lines[i].anchors);
    }
    free(lines);
}

void liteocr_free_text_boxes(liteocr_text_box_t* boxes, int count) {
    if (!boxes) return;
    free(boxes);
}

void liteocr_free_contours(liteocr_contour_t* contours, int contour_count) {
    if (!contours) return;
    for (int i = 0; i < contour_count; ++i) {
        free(contours[i].points);
    }
    free(contours);
}

void liteocr_free_table_cells(liteocr_table_cell_t* cells, int count) {
    if (!cells) return;
    for (int i = 0; i < count; ++i) {
        free(cells[i].tag);
    }
    free(cells);
}

/* ============================================================================
 *  liteocr_image I/O
 * ============================================================================ */

liteocr_image_t liteocr_imread(const char* filename, int desired_channels) {
    liteocr_image img = liteocr_imread_image(filename, desired_channels);
    return imageToC(img);
}

int liteocr_imwrite(const char* filename, const liteocr_image_t* img) {
    if (!filename || !img) return -1;
    liteocr_image cpp = imageFromC(img);
    return liteocr_imwrite_image(filename, cpp) ? 0 : -1;
}

/* ============================================================================
 *  liteocr_image processing
 * ============================================================================ */

void liteocr_cvt_color(const unsigned char* src, int src_w, int src_h, int src_step, int src_fmt,
                       unsigned char* dst, int dst_step, int dst_fmt) {
    liteocr_cvt_color(src, src_w, src_h, src_step, (liteocr_color_format)src_fmt,
                      dst, dst_step, (liteocr_color_format)dst_fmt);
}

void liteocr_rotate90(const unsigned char* src, int src_w, int src_h, int src_step, int channels,
                      unsigned char* dst, int dst_step, int counter_clockwise) {
    liteocr_rotate90(src, src_w, src_h, src_step, channels,
                      dst, dst_step, counter_clockwise != 0);
}

/* ============================================================================
 *  Contours
 * ============================================================================ */

int liteocr_find_contours(const unsigned char* data, int width, int height, int step,
                          liteocr_contour_t** out_contours, int* out_contour_count,
                          int approx_mode) {
    if (!data || !out_contours || !out_contour_count) return -1;
    *out_contours = nullptr;
    *out_contour_count = 0;
    if (width <= 0 || height <= 0 || step < width) return -1;

    std::vector<std::vector<liteocr_point>> contours;
    liteocr_contour_approx_mode mode = (approx_mode == 1)
        ? LITEOCR_CHAIN_APPROX_NONE : LITEOCR_CHAIN_APPROX_SIMPLE;
    liteocr_find_contours(data, width, height, step, contours, mode);

    *out_contour_count = (int)contours.size();
    if (contours.empty()) {
        *out_contours = nullptr;
        return 0;
    }

    *out_contours = (liteocr_contour_t*)malloc(sizeof(liteocr_contour_t) * contours.size());
    if (!*out_contours) {
        *out_contour_count = 0;
        return -1;
    }
    std::memset(*out_contours, 0, sizeof(liteocr_contour_t) * contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        (*out_contours)[i].point_count = (int)contours[i].size();
        if (contours[i].empty()) {
            (*out_contours)[i].points = nullptr;
            continue;
        }
        (*out_contours)[i].points = (liteocr_point_t*)malloc(sizeof(liteocr_point_t) * contours[i].size());
        if (!(*out_contours)[i].points) {
            liteocr_free_contours(*out_contours, (int)contours.size());
            *out_contours = nullptr;
            *out_contour_count = 0;
            return -1;
        }
        for (size_t j = 0; j < contours[i].size(); ++j) {
            (*out_contours)[i].points[j].x = contours[i][j].x;
            (*out_contours)[i].points[j].y = contours[i][j].y;
        }
    }
    return 0;
}

void liteocr_min_area_rect(const liteocr_point_t* contour, int point_count, liteocr_point2f_t out_pts[4]) {
    if (!contour || point_count <= 0 || !out_pts) return;
    std::vector<liteocr_point> pts;
    pts.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        pts.push_back({contour[i].x, contour[i].y});
    }
    auto rr = liteocr_min_area_rect(pts);
    auto rpts = liteocr_get_rotated_rect_points(rr);
    for (int i = 0; i < 4; ++i) {
        out_pts[i].x = rpts[i].x;
        out_pts[i].y = rpts[i].y;
    }
}

liteocr_intrect_t liteocr_bounding_rect(const liteocr_point_t* contour, int point_count) {
    if (!contour || point_count <= 0) return {};
    std::vector<liteocr_point> pts;
    pts.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        pts.push_back({contour[i].x, contour[i].y});
    }
    auto r = liteocr_bounding_rect(pts);
    return {r.x, r.y, r.width, r.height};
}

double liteocr_contour_area(const liteocr_point_t* contour, int point_count) {
    if (!contour || point_count <= 0) return 0.0;
    std::vector<liteocr_point> pts;
    pts.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        pts.push_back({contour[i].x, contour[i].y});
    }
    return liteocr_contour_area(pts);
}

double liteocr_arc_length(const liteocr_point_t* contour, int point_count, int closed) {
    if (!contour || point_count <= 0) return 0.0;
    std::vector<liteocr_point> pts;
    pts.reserve(point_count);
    for (int i = 0; i < point_count; ++i) {
        pts.push_back({contour[i].x, contour[i].y});
    }
    return liteocr_arc_length(pts, closed != 0);
}

void liteocr_fill_poly(unsigned char* data, int width, int height, int step,
                       const liteocr_contour_t* polygons, int polygon_count,
                       unsigned char value) {
    if (!data || !polygons || polygon_count <= 0) return;
    std::vector<std::vector<liteocr_point>> poly;
    poly.reserve(polygon_count);
    for (int i = 0; i < polygon_count; ++i) {
        std::vector<liteocr_point> pts;
        pts.reserve(polygons[i].point_count);
        for (int j = 0; j < polygons[i].point_count; ++j) {
            pts.push_back({polygons[i].points[j].x, polygons[i].points[j].y});
        }
        poly.push_back(std::move(pts));
    }
    liteocr_fill_poly(data, width, height, step, poly, value);
}

/* ============================================================================
 *  OCR Engine
 * ============================================================================ */

struct liteocr_engine {
    liteocr_ocr_engine impl;
};

liteocr_engine_t liteocr_engine_create(void) {
    auto e = new liteocr_engine();
    liteocr_ocr_engine_init(&e->impl);
    return e;
}

void liteocr_engine_destroy(liteocr_engine_t engine) {
    delete engine;
}

int liteocr_engine_load_model(liteocr_engine_t engine,
    const liteocr_ocr_model_paths_t* paths,
    const liteocr_infer_option_t* opt) {
    if (!engine || !paths) return -1;
    bool ret = liteocr_ocr_engine_load_model(&engine->impl,
        paths->det_param, paths->det_bin,
        paths->rec_param, paths->rec_bin,
        paths->vocab,
        paths->ori_param, paths->ori_bin,
        optionFromC(opt));
    return ret ? 0 : -1;
}

int liteocr_engine_load_model_from_buffer(liteocr_engine_t engine,
    const liteocr_ocr_model_buffers_t* buffers,
    const liteocr_infer_option_t* opt) {
    if (!engine || !buffers) return -1;
    bool ret = liteocr_ocr_engine_load_model_from_buffer(&engine->impl,
        buffers->det_param, buffers->det_bin,
        buffers->rec_param, buffers->rec_bin,
        buffers->vocab,
        buffers->ori_param, buffers->ori_bin,
        optionFromC(opt));
    return ret ? 0 : -1;
}

static int do_recognize(liteocr_engine_t engine, const liteocr_image& input,
    liteocr_text_box_t** out_boxes, int* out_box_count,
    liteocr_text_line_t** out_lines, int* out_line_count) {

    if (input.empty()) return -1;
    auto result = liteocr_ocr_engine_recognize(&engine->impl, input);
    const auto& boxes = result.first;
    const auto& lines = result.second;

    return copyOcrResultToC(boxes, lines, out_boxes, out_box_count, out_lines, out_line_count);
}

int liteocr_engine_recognize_image(liteocr_engine_t engine, const liteocr_image_t* img,
    liteocr_text_box_t** out_boxes, int* out_box_count,
    liteocr_text_line_t** out_lines, int* out_line_count) {
    if (!engine || !img || !out_boxes || !out_box_count || !out_lines || !out_line_count) return -1;
    *out_boxes = nullptr;
    *out_box_count = 0;
    *out_lines = nullptr;
    *out_line_count = 0;
    liteocr_image input = imageFromC(img);
    return do_recognize(engine, input, out_boxes, out_box_count, out_lines, out_line_count);
}

int liteocr_engine_recognize_raw(liteocr_engine_t engine,
    const unsigned char* data, int width, int height, int channels, int stride,
    liteocr_text_box_t** out_boxes, int* out_box_count,
    liteocr_text_line_t** out_lines, int* out_line_count) {
    if (!engine || !data || !out_boxes || !out_box_count || !out_lines || !out_line_count) return -1;
    *out_boxes = nullptr;
    *out_box_count = 0;
    *out_lines = nullptr;
    *out_line_count = 0;
    liteocr_image_t img = {const_cast<unsigned char*>(data), width, height, channels, stride};
    liteocr_image input = imageFromC(&img);
    return do_recognize(engine, input, out_boxes, out_box_count, out_lines, out_line_count);
}

int liteocr_engine_recognize_buffer(liteocr_engine_t engine,
    const unsigned char* buffer, int size,
    liteocr_text_box_t** out_boxes, int* out_box_count,
    liteocr_text_line_t** out_lines, int* out_line_count) {
    if (!engine || !buffer || size <= 0 || !out_boxes || !out_box_count || !out_lines || !out_line_count) return -1;
    *out_boxes = nullptr;
    *out_box_count = 0;
    *out_lines = nullptr;
    *out_line_count = 0;

    auto result = liteocr_ocr_engine_recognize_buffer(&engine->impl, buffer, size);
    const auto& boxes = result.first;
    const auto& lines = result.second;

    return copyOcrResultToC(boxes, lines, out_boxes, out_box_count, out_lines, out_line_count);
}

char* liteocr_merge_text_boxes(const liteocr_text_box_t* boxes, int box_count,
                                const liteocr_text_line_t* lines, int line_count) {
    if (!boxes || !lines || box_count <= 0 || line_count <= 0) return nullptr;
    std::string merged;
    int n = std::min(box_count, line_count);
    for (int i = 0; i < n; ++i) {
        if (lines[i].text) {
            merged += lines[i].text;
        }
        if (i + 1 < n) merged += "\n";
    }
    return LITEOCR_STRDUP(merged.c_str());
}

/* ============================================================================
 *  Table Engine
 * ============================================================================ */

liteocr_table_engine_t liteocr_table_engine_create(void) {
    auto e = new liteocr_table_engine();
    liteocr_table_engine_init(e);
    return e;
}

void liteocr_table_engine_destroy(liteocr_table_engine_t engine) {
    delete engine;
}

int liteocr_table_engine_load_model(liteocr_table_engine_t engine,
    const liteocr_table_model_paths_t* paths,
    const liteocr_infer_option_t* opt) {
    if (!engine || !paths) return -1;
    bool ret = liteocr_table_engine_load_model(engine,
        paths->cnn_param, paths->cnn_bin,
        paths->slahead_param, paths->slahead_bin,
        paths->vocab,
        optionFromC(opt));
    return ret ? 0 : -1;
}

int liteocr_table_engine_load_model_from_buffer(liteocr_table_engine_t engine,
    const liteocr_table_model_buffers_t* buffers,
    const liteocr_infer_option_t* opt) {
    if (!engine || !buffers) return -1;
    bool ret = liteocr_table_engine_load_model_from_buffer(engine,
        buffers->cnn_param, buffers->cnn_bin,
        buffers->slahead_param, buffers->slahead_bin,
        buffers->vocab,
        optionFromC(opt));
    return ret ? 0 : -1;
}

static int do_table_recognize(liteocr_table_engine_t engine, const liteocr_image& input,
    const liteocr_text_box_t* boxes, int box_count,
    const liteocr_text_line_t* lines, int line_count,
    char** out_html,
    liteocr_rect_t** out_cells, int* out_cell_count,
    liteocr_table_cell_t** out_structure, int* out_structure_count) {

    if (input.empty() || box_count < 0 || line_count < 0) return -1;
    std::vector<liteocr_text_box> cpp_boxes;
    cpp_boxes.reserve(box_count);
    for (int i = 0; i < box_count; ++i) {
        cpp_boxes.push_back(textBoxFromC(&boxes[i]));
    }

    std::vector<liteocr_text_line> cpp_lines;
    cpp_lines.reserve(line_count);
    for (int i = 0; i < line_count; ++i) {
        std::string text = lines[i].text ? lines[i].text : "";
        std::vector<float> anchors;
        if (lines[i].anchors && lines[i].anchor_count > 0) {
            anchors.assign(lines[i].anchors, lines[i].anchors + lines[i].anchor_count);
        }
        cpp_lines.push_back({text, anchors});
    }

    auto result = liteocr_table_engine_recognize(engine, input, {cpp_boxes, cpp_lines});
    const auto& html = result.first;
    const auto& rects = result.second;

    if (out_html) {
        *out_html = LITEOCR_STRDUP(html.c_str());
        if (!*out_html) return -1;
    }
    if (out_cells && out_cell_count) {
        *out_cell_count = (int)rects.size();
        if (!rects.empty()) {
            *out_cells = (liteocr_rect_t*)malloc(sizeof(liteocr_rect_t) * rects.size());
            if (!*out_cells) {
                if (out_html) {
                    free(*out_html);
                    *out_html = nullptr;
                }
                *out_cell_count = 0;
                return -1;
            }
            for (size_t i = 0; i < rects.size(); ++i) {
                (*out_cells)[i] = {rects[i].x, rects[i].y, rects[i].width, rects[i].height};
            }
        } else {
            *out_cells = nullptr;
        }
    }
    if (out_structure && out_structure_count) {
        *out_structure = nullptr;
        *out_structure_count = 0;
    }
    return 0;
}

int liteocr_table_engine_recognize_image(liteocr_table_engine_t engine,
    const liteocr_image_t* img,
    const liteocr_text_box_t* boxes, int box_count,
    const liteocr_text_line_t* lines, int line_count,
    char** out_html,
    liteocr_rect_t** out_cells, int* out_cell_count,
    liteocr_table_cell_t** out_structure, int* out_structure_count) {
    if (!engine || !img || box_count < 0 || line_count < 0 || (box_count > 0 && !boxes) || (line_count > 0 && !lines)) return -1;
    if (out_html) *out_html = nullptr;
    if (out_cells) *out_cells = nullptr;
    if (out_cell_count) *out_cell_count = 0;
    if (out_structure) *out_structure = nullptr;
    if (out_structure_count) *out_structure_count = 0;
    liteocr_image input = imageFromC(img);
    return do_table_recognize(engine, input, boxes, box_count, lines, line_count,
        out_html, out_cells, out_cell_count, out_structure, out_structure_count);
}

int liteocr_table_engine_recognize_raw(liteocr_table_engine_t engine,
    const unsigned char* data, int width, int height, int channels, int stride,
    const liteocr_text_box_t* boxes, int box_count,
    const liteocr_text_line_t* lines, int line_count,
    char** out_html,
    liteocr_rect_t** out_cells, int* out_cell_count,
    liteocr_table_cell_t** out_structure, int* out_structure_count) {
    if (!engine || !data || box_count < 0 || line_count < 0 || (box_count > 0 && !boxes) || (line_count > 0 && !lines)) return -1;
    if (out_html) *out_html = nullptr;
    if (out_cells) *out_cells = nullptr;
    if (out_cell_count) *out_cell_count = 0;
    if (out_structure) *out_structure = nullptr;
    if (out_structure_count) *out_structure_count = 0;
    liteocr_image_t img = {const_cast<unsigned char*>(data), width, height, channels, stride};
    liteocr_image input = imageFromC(&img);
    return do_table_recognize(engine, input, boxes, box_count, lines, line_count,
        out_html, out_cells, out_cell_count, out_structure, out_structure_count);
}

int liteocr_table_engine_recognize_buffer(liteocr_table_engine_t engine,
    const unsigned char* buffer, int size,
    const liteocr_text_box_t* boxes, int box_count,
    const liteocr_text_line_t* lines, int line_count,
    char** out_html,
    liteocr_rect_t** out_cells, int* out_cell_count,
    liteocr_table_cell_t** out_structure, int* out_structure_count) {
    if (!engine || !buffer || size <= 0 || box_count < 0 || line_count < 0 ||
        (box_count > 0 && !boxes) || (line_count > 0 && !lines)) return -1;
    if (out_html) *out_html = nullptr;
    if (out_cells) *out_cells = nullptr;
    if (out_cell_count) *out_cell_count = 0;
    if (out_structure) *out_structure = nullptr;
    if (out_structure_count) *out_structure_count = 0;

    std::vector<liteocr_text_box> cpp_boxes;
    cpp_boxes.reserve(box_count);
    for (int i = 0; i < box_count; ++i) {
        cpp_boxes.push_back(textBoxFromC(&boxes[i]));
    }

    std::vector<liteocr_text_line> cpp_lines;
    cpp_lines.reserve(line_count);
    for (int i = 0; i < line_count; ++i) {
        std::string text = lines[i].text ? lines[i].text : "";
        std::vector<float> anchors;
        if (lines[i].anchors && lines[i].anchor_count > 0) {
            anchors.assign(lines[i].anchors, lines[i].anchors + lines[i].anchor_count);
        }
        cpp_lines.push_back({text, anchors});
    }

    auto r = liteocr_table_engine_recognize_buffer(engine, buffer, size, {cpp_boxes, cpp_lines});
    const auto& html = r.first;
    const auto& rects = r.second;
    if (out_html) {
        *out_html = LITEOCR_STRDUP(html.c_str());
        if (!*out_html) return -1;
    }
    if (out_cells && out_cell_count) {
        *out_cell_count = (int)rects.size();
        if (!rects.empty()) {
            *out_cells = (liteocr_rect_t*)malloc(sizeof(liteocr_rect_t) * rects.size());
            if (!*out_cells) {
                if (out_html) {
                    free(*out_html);
                    *out_html = nullptr;
                }
                *out_cell_count = 0;
                return -1;
            }
            for (size_t i = 0; i < rects.size(); ++i) {
                (*out_cells)[i] = {rects[i].x, rects[i].y, rects[i].width, rects[i].height};
            }
        } else {
            *out_cells = nullptr;
        }
    }
    if (out_structure && out_structure_count) {
        *out_structure = nullptr;
        *out_structure_count = 0;
    }
    return 0;
}

/* ============================================================================
 *  Low-level components
 * ============================================================================ */

/* liteocr_detector */
liteocr_detector_t liteocr_detector_create(void) {
    return new liteocr_detector();
}

void liteocr_detector_destroy(liteocr_detector_t det) {
    delete det;
}

int liteocr_detector_load_model(liteocr_detector_t det, const char* param, const char* bin, const liteocr_infer_option_t* opt) {
    if (!det) return -1;
    return liteocr_detector_load_model(det, param, bin, optionFromC(opt)) ? 0 : -1;
}

int liteocr_detector_load_model_from_buffer(liteocr_detector_t det, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt) {
    if (!det) return -1;
    return liteocr_detector_load_model_from_buffer(det, param_buf, bin_buf, optionFromC(opt)) ? 0 : -1;
}

liteocr_image_t liteocr_detector_forward(liteocr_detector_t det, const liteocr_image_t* input) {
    if (!det || !input) return {};
    liteocr_image img = imageFromC(input);
    if (img.empty()) return {};
    liteocr_image out = liteocr_detector_forward(det, img);
    return imageToC(out);
}

/* liteocr_recognizer */
liteocr_recognizer_t liteocr_recognizer_create(void) {
    return new liteocr_recognizer();
}

void liteocr_recognizer_destroy(liteocr_recognizer_t rec) {
    delete rec;
}

int liteocr_recognizer_load_model(liteocr_recognizer_t rec, const char* param, const char* bin, const liteocr_infer_option_t* opt) {
    if (!rec) return -1;
    return liteocr_recognizer_load_model(rec, param, bin, optionFromC(opt)) ? 0 : -1;
}

int liteocr_recognizer_load_model_from_buffer(liteocr_recognizer_t rec, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt) {
    if (!rec) return -1;
    return liteocr_recognizer_load_model_from_buffer(rec, param_buf, bin_buf, optionFromC(opt)) ? 0 : -1;
}

liteocr_image_t liteocr_recognizer_forward(liteocr_recognizer_t rec, const liteocr_image_t* input) {
    if (!rec || !input) return {};
    liteocr_image img = imageFromC(input);
    if (img.empty()) return {};
    liteocr_image out = liteocr_recognizer_forward(rec, img);
    return imageToC(out);
}

/* liteocr_text_line Orientation */
liteocr_textline_ori_t liteocr_textline_ori_create(void) {
    return new liteocr_textline_ori();
}

void liteocr_textline_ori_destroy(liteocr_textline_ori_t ori) {
    delete ori;
}

int liteocr_textline_ori_load_model(liteocr_textline_ori_t ori, const char* param, const char* bin, const liteocr_infer_option_t* opt) {
    if (!ori) return -1;
    return liteocr_textline_ori_load_model(ori, param, bin, optionFromC(opt)) ? 0 : -1;
}

int liteocr_textline_ori_load_model_from_buffer(liteocr_textline_ori_t ori, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt) {
    if (!ori) return -1;
    return liteocr_textline_ori_load_model_from_buffer(ori, param_buf, bin_buf, optionFromC(opt)) ? 0 : -1;
}

int liteocr_textline_ori_forward(liteocr_textline_ori_t ori, const liteocr_image_t* input) {
    if (!ori || !input) return -1;
    liteocr_image img = imageFromC(input);
    if (img.empty()) return -1;
    return liteocr_textline_ori_forward(ori, img);
}

/* Doc Orientation */
liteocr_doc_ori_t liteocr_doc_ori_create(void) {
    return new liteocr_doc_ori();
}

void liteocr_doc_ori_destroy(liteocr_doc_ori_t ori) {
    delete ori;
}

int liteocr_doc_ori_load_model(liteocr_doc_ori_t ori, const char* param, const char* bin, const liteocr_infer_option_t* opt) {
    if (!ori) return -1;
    return liteocr_doc_ori_load_model(ori, param, bin, optionFromC(opt)) ? 0 : -1;
}

int liteocr_doc_ori_load_model_from_buffer(liteocr_doc_ori_t ori, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt) {
    if (!ori) return -1;
    return liteocr_doc_ori_load_model_from_buffer(ori, param_buf, bin_buf, optionFromC(opt)) ? 0 : -1;
}

int liteocr_doc_ori_forward(liteocr_doc_ori_t ori, const liteocr_image_t* input) {
    if (!ori || !input) return -1;
    liteocr_image img = imageFromC(input);
    if (img.empty()) return -1;
    return liteocr_doc_ori_forward(ori, img);
}

/* liteocr_uvdoc */
liteocr_uvdoc_t liteocr_uvdoc_create(void) {
    return new liteocr_uvdoc();
}

void liteocr_uvdoc_destroy(liteocr_uvdoc_t uv) {
    delete uv;
}

int liteocr_uvdoc_load_model(liteocr_uvdoc_t uv, const char* param, const char* bin, const liteocr_infer_option_t* opt) {
    if (!uv) return -1;
    return liteocr_uvdoc_load_model(uv, param, bin, optionFromC(opt)) ? 0 : -1;
}

int liteocr_uvdoc_load_model_from_buffer(liteocr_uvdoc_t uv, const char* param_buf, const unsigned char* bin_buf, const liteocr_infer_option_t* opt) {
    if (!uv) return -1;
    return liteocr_uvdoc_load_model_from_buffer(uv, param_buf, bin_buf, optionFromC(opt)) ? 0 : -1;
}

liteocr_image_t liteocr_uvdoc_forward(liteocr_uvdoc_t uv, const liteocr_image_t* input) {
    if (!uv || !input) return {};
    liteocr_image img = imageFromC(input);
    if (img.empty()) return {};
    liteocr_image out = liteocr_uvdoc_forward(uv, img);
    return imageToC(out);
}

/* liteocr_slanet */
liteocr_slanet_t liteocr_slanet_create(void) {
    return new liteocr_slanet();
}

void liteocr_slanet_destroy(liteocr_slanet_t sla) {
    delete sla;
}

int liteocr_slanet_load_model(liteocr_slanet_t sla, const char* cnn_param, const char* cnn_bin,
                               const char* slahead_param, const char* slahead_bin,
                               const char* vocab,
                               const liteocr_infer_option_t* opt) {
    if (!sla) return -1;
    return liteocr_slanet_load_model(sla, cnn_param, cnn_bin, slahead_param, slahead_bin, vocab, optionFromC(opt)) ? 0 : -1;
}

int liteocr_slanet_load_model_from_buffer(liteocr_slanet_t sla,
    const char* cnn_param_buf, const unsigned char* cnn_bin_buf,
    const char* slahead_param_buf, const unsigned char* slahead_bin_buf,
    const char* vocab_buf,
    const liteocr_infer_option_t* opt) {
    if (!sla) return -1;
    return liteocr_slanet_load_model_from_buffer(sla, cnn_param_buf, cnn_bin_buf, slahead_param_buf, slahead_bin_buf, vocab_buf, optionFromC(opt)) ? 0 : -1;
}

int liteocr_slanet_forward(liteocr_slanet_t sla, const liteocr_image_t* input,
    liteocr_table_cell_t** out_cells, int* out_count) {
    if (!sla || !input || !out_cells || !out_count) return -1;
    *out_cells = nullptr;
    *out_count = 0;
    liteocr_image img = imageFromC(input);
    if (img.empty()) return -1;
    auto result = liteocr_slanet_forward(sla, img);
    *out_count = (int)result.size();
    if (result.empty()) {
        *out_cells = nullptr;
        return 0;
    }
    *out_cells = (liteocr_table_cell_t*)malloc(sizeof(liteocr_table_cell_t) * result.size());
    if (!*out_cells) {
        *out_count = 0;
        return -1;
    }
    std::memset(*out_cells, 0, sizeof(liteocr_table_cell_t) * result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        (*out_cells)[i].tag = LITEOCR_STRDUP(result[i].first.c_str());
        if (!(*out_cells)[i].tag) {
            liteocr_free_table_cells(*out_cells, (int)result.size());
            *out_cells = nullptr;
            *out_count = 0;
            return -1;
        }
        for (int j = 0; j < 8; ++j) {
            (*out_cells)[i].box[j] = result[i].second[j];
        }
    }
    return 0;
}

/* ============================================================================
 *  CTC Decoder
 * ============================================================================ */

int liteocr_ctc_decode(const liteocr_image_t* probs, int blank_index,
    int** out_tokens, float** out_probs, int** out_indices, int* out_count) {
    if (!probs || !probs->data || !out_tokens || !out_probs || !out_indices || !out_count) return -1;
    *out_tokens = nullptr;
    *out_probs = nullptr;
    *out_indices = nullptr;
    *out_count = 0;
    liteocr_image img = imageFromC(probs);
    if (img.empty()) return -1;
    auto result = liteocr_ctc_decode(img, blank_index);
    *out_count = (int)result.size();
    if (result.empty()) {
        *out_tokens = nullptr;
        *out_probs = nullptr;
        *out_indices = nullptr;
        return 0;
    }
    *out_tokens = (int*)malloc(sizeof(int) * result.size());
    *out_probs = (float*)malloc(sizeof(float) * result.size());
    *out_indices = (int*)malloc(sizeof(int) * result.size());
    if (!*out_tokens || !*out_probs || !*out_indices) {
        free(*out_tokens);
        free(*out_probs);
        free(*out_indices);
        *out_tokens = nullptr;
        *out_probs = nullptr;
        *out_indices = nullptr;
        *out_count = 0;
        return -1;
    }
    for (size_t i = 0; i < result.size(); ++i) {
        (*out_tokens)[i] = std::get<0>(result[i]);
        (*out_probs)[i] = std::get<1>(result[i]);
        (*out_indices)[i] = std::get<2>(result[i]);
    }
    return 0;
}
