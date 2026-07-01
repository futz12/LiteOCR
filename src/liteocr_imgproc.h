#pragma once
#include <cstdint>
#include <cmath>

enum liteocr_color_format {
    LITEOCR_FORMAT_GRAY = 1,
    LITEOCR_FORMAT_RGB  = 2,
    LITEOCR_FORMAT_BGR  = 3,
    LITEOCR_FORMAT_RGBA = 4,
    LITEOCR_FORMAT_BGRA = 5
};

void liteocr_cvt_color(const uint8_t* src, int src_w, int src_h, int src_step, liteocr_color_format src_fmt,
                       uint8_t* dst, int dst_step, liteocr_color_format dst_fmt);

extern "C" void liteocr_threshold(const float* src, int w, int h, int src_step_bytes,
                       uint8_t* dst, int dst_step_bytes,
                       float thresh, uint8_t maxval);

extern "C" double liteocr_mean_masked(const float* src, int w, int h, int src_step_bytes,
                           const uint8_t* mask, int mask_step_bytes);

extern "C" void liteocr_resize(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                    uint8_t* dst, int dst_w, int dst_h, int dst_step);

void liteocr_rotate90(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                      uint8_t* dst, int dst_step, bool counter_clockwise);
extern "C" void liteocr_rotate180(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                       uint8_t* dst, int dst_step);

extern "C" void liteocr_copy_make_border(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                              uint8_t* dst, int dst_w, int dst_h, int dst_step,
                              int top, int bottom, int left, int right,
                              uint8_t fill_value);

extern "C" void liteocr_get_perspective_transform(const float src[8], const float dst[8], float M[9]);
extern "C" void liteocr_warp_perspective(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                              uint8_t* dst, int dst_w, int dst_h, int dst_step,
                              const float M[9]);

inline float liteocr_norm(float dx, float dy) {
    return std::sqrt(dx * dx + dy * dy);
}
