#include "liteocr_imgproc.h"
#include <cstring>
#include <algorithm>
#include <mat.h>

static inline int liteocr_format_channels(liteocr_color_format fmt) {
    switch (fmt) {
        case LITEOCR_FORMAT_GRAY: return 1;
        case LITEOCR_FORMAT_RGB:
        case LITEOCR_FORMAT_BGR: return 3;
        case LITEOCR_FORMAT_RGBA:
        case LITEOCR_FORMAT_BGRA: return 4;
    }
    return 0;
}

void liteocr_cvt_color(const uint8_t* src, int src_w, int src_h, int src_step, liteocr_color_format src_fmt,
                       uint8_t* dst, int dst_step, liteocr_color_format dst_fmt)
{
    if (src_fmt == dst_fmt) {
        int ch = liteocr_format_channels(src_fmt);
        for (int y = 0; y < src_h; ++y)
            std::memcpy(dst + y * dst_step, src + y * src_step, src_w * ch);
        return;
    }

    for (int y = 0; y < src_h; ++y) {
        const uint8_t* s = src + y * src_step;
        uint8_t* d = dst + y * dst_step;
        for (int x = 0; x < src_w; ++x) {
            uint8_t r = 0, g = 0, b = 0, a = 255;
            if (src_fmt == LITEOCR_FORMAT_GRAY) {
                r = g = b = s[x];
            } else if (src_fmt == LITEOCR_FORMAT_RGB) {
                r = s[x * 3 + 0]; g = s[x * 3 + 1]; b = s[x * 3 + 2];
            } else if (src_fmt == LITEOCR_FORMAT_BGR) {
                b = s[x * 3 + 0]; g = s[x * 3 + 1]; r = s[x * 3 + 2];
            } else if (src_fmt == LITEOCR_FORMAT_RGBA) {
                r = s[x * 4 + 0]; g = s[x * 4 + 1]; b = s[x * 4 + 2]; a = s[x * 4 + 3];
            } else if (src_fmt == LITEOCR_FORMAT_BGRA) {
                b = s[x * 4 + 0]; g = s[x * 4 + 1]; r = s[x * 4 + 2]; a = s[x * 4 + 3];
            }

            if (dst_fmt == LITEOCR_FORMAT_GRAY) {
                d[x] = (uint8_t)((r * 76 + g * 150 + b * 29) >> 8);
            } else if (dst_fmt == LITEOCR_FORMAT_RGB) {
                d[x * 3 + 0] = r; d[x * 3 + 1] = g; d[x * 3 + 2] = b;
            } else if (dst_fmt == LITEOCR_FORMAT_BGR) {
                d[x * 3 + 0] = b; d[x * 3 + 1] = g; d[x * 3 + 2] = r;
            } else if (dst_fmt == LITEOCR_FORMAT_RGBA) {
                d[x * 4 + 0] = r; d[x * 4 + 1] = g; d[x * 4 + 2] = b; d[x * 4 + 3] = a;
            } else if (dst_fmt == LITEOCR_FORMAT_BGRA) {
                d[x * 4 + 0] = b; d[x * 4 + 1] = g; d[x * 4 + 2] = r; d[x * 4 + 3] = a;
            }
        }
    }
}

extern "C" void liteocr_threshold(const float* src, int w, int h, int src_step_bytes,
                       uint8_t* dst, int dst_step_bytes,
                       float thresh, uint8_t maxval)
{
    int src_stride = src_step_bytes / sizeof(float);
    for (int y = 0; y < h; ++y) {
        const float* s = src + y * src_stride;
        uint8_t* d = dst + y * dst_step_bytes;
        for (int x = 0; x < w; ++x)
            d[x] = s[x] > thresh ? maxval : 0;
    }
}

extern "C" double liteocr_mean_masked(const float* src, int w, int h, int src_step_bytes,
                           const uint8_t* mask, int mask_step_bytes)
{
    int src_stride = src_step_bytes / sizeof(float);
    double sum = 0.0;
    int count = 0;
    for (int y = 0; y < h; ++y) {
        const float* s = src + y * src_stride;
        const uint8_t* m = mask + y * mask_step_bytes;
        for (int x = 0; x < w; ++x) {
            if (m[x] > 0) {
                sum += s[x];
                ++count;
            }
        }
    }
    return count > 0 ? sum / count : 0.0;
}

extern "C" void liteocr_resize(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                    uint8_t* dst, int dst_w, int dst_h, int dst_step)
{
    if (channels == 1)
        ncnn::resize_bilinear_c1(src, src_w, src_h, src_step, dst, dst_w, dst_h, dst_step);
    else if (channels == 3)
        ncnn::resize_bilinear_c3(src, src_w, src_h, src_step, dst, dst_w, dst_h, dst_step);
    else if (channels == 4)
        ncnn::resize_bilinear_c4(src, src_w, src_h, src_step, dst, dst_w, dst_h, dst_step);
}

void liteocr_rotate90(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                      uint8_t* dst, int dst_step, bool counter_clockwise)
{
    int type = counter_clockwise ? 3 : 1;
    int dst_w = src_h;
    int dst_h = src_w;
    if (channels == 1)
        ncnn::kanna_rotate_c1(src, src_w, src_h, src_step, dst, dst_w, dst_h, dst_step, type);
    else if (channels == 3)
        ncnn::kanna_rotate_c3(src, src_w, src_h, src_step, dst, dst_w, dst_h, dst_step, type);
    else if (channels == 4)
        ncnn::kanna_rotate_c4(src, src_w, src_h, src_step, dst, dst_w, dst_h, dst_step, type);
}

extern "C" void liteocr_rotate180(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                       uint8_t* dst, int dst_step)
{
    if (channels == 1)
        ncnn::kanna_rotate_c1(src, src_w, src_h, src_step, dst, src_w, src_h, dst_step, 2);
    else if (channels == 3)
        ncnn::kanna_rotate_c3(src, src_w, src_h, src_step, dst, src_w, src_h, dst_step, 2);
    else if (channels == 4)
        ncnn::kanna_rotate_c4(src, src_w, src_h, src_step, dst, src_w, src_h, dst_step, 2);
}

extern "C" void liteocr_copy_make_border(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                              uint8_t* dst, int dst_w, int dst_h, int dst_step,
                              int top, int bottom, int left, int right,
                              uint8_t fill_value)
{
    for (int y = 0; y < dst_h; ++y) {
        uint8_t* d = dst + y * dst_step;
        for (int x = 0; x < dst_w * channels; ++x)
            d[x] = fill_value;
    }
    for (int y = 0; y < src_h; ++y) {
        const uint8_t* s = src + y * src_step;
        uint8_t* d = dst + (top + y) * dst_step + left * channels;
        std::memcpy(d, s, src_w * channels);
    }
}

extern "C" void liteocr_get_perspective_transform(const float src[8], const float dst[8], float M[9])
{
    float A[8][9];
    for (int i = 0; i < 4; ++i) {
        float x = src[i * 2], y = src[i * 2 + 1];
        float u = dst[i * 2], v = dst[i * 2 + 1];

        A[i * 2][0] = x;   A[i * 2][1] = y;   A[i * 2][2] = 1;
        A[i * 2][3] = 0;   A[i * 2][4] = 0;   A[i * 2][5] = 0;
        A[i * 2][6] = -x * u; A[i * 2][7] = -y * u; A[i * 2][8] = u;

        A[i * 2 + 1][0] = 0;   A[i * 2 + 1][1] = 0;   A[i * 2 + 1][2] = 0;
        A[i * 2 + 1][3] = x;   A[i * 2 + 1][4] = y;   A[i * 2 + 1][5] = 1;
        A[i * 2 + 1][6] = -x * v; A[i * 2 + 1][7] = -y * v; A[i * 2 + 1][8] = v;
    }

    for (int col = 0; col < 8; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 8; ++row) {
            if (std::abs(A[row][col]) > std::abs(A[pivot][col]))
                pivot = row;
        }
        std::swap(A[col], A[pivot]);

        float div = A[col][col];
        if (std::abs(div) < 1e-10f) div = 1e-10f;
        for (int j = col; j < 9; ++j)
            A[col][j] /= div;

        for (int row = 0; row < 8; ++row) {
            if (row == col) continue;
            float factor = A[row][col];
            for (int j = col; j < 9; ++j)
                A[row][j] -= factor * A[col][j];
        }
    }

    M[0] = A[0][8]; M[1] = A[1][8]; M[2] = A[2][8];
    M[3] = A[3][8]; M[4] = A[4][8]; M[5] = A[5][8];
    M[6] = A[6][8]; M[7] = A[7][8]; M[8] = 1.0f;
}

extern "C" void liteocr_warp_perspective(const uint8_t* src, int src_w, int src_h, int src_step, int channels,
                              uint8_t* dst, int dst_w, int dst_h, int dst_step,
                              const float M[9])
{
    float a = M[0], b = M[1], c = M[2];
    float d = M[3], e = M[4], f = M[5];
    float g = M[6], h = M[7], i = M[8];

    float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::abs(det) < 1e-10f) det = 1e-10f;

    float invM[9];
    invM[0] = (e * i - f * h) / det;
    invM[1] = (c * h - b * i) / det;
    invM[2] = (b * f - c * e) / det;
    invM[3] = (f * g - d * i) / det;
    invM[4] = (a * i - c * g) / det;
    invM[5] = (c * d - a * f) / det;
    invM[6] = (d * h - e * g) / det;
    invM[7] = (b * g - a * h) / det;
    invM[8] = (a * e - b * d) / det;

    for (int y = 0; y < dst_h; ++y) {
        uint8_t* d = dst + y * dst_step;
        for (int x = 0; x < dst_w; ++x) {
            float fx = invM[0] * x + invM[1] * y + invM[2];
            float fy = invM[3] * x + invM[4] * y + invM[5];
            float fw = invM[6] * x + invM[7] * y + invM[8];
            if (std::abs(fw) > 1e-10f) {
                fx /= fw;
                fy /= fw;
            }

            int sx = (int)fx;
            int sy = (int)fy;
            float dx = fx - sx;
            float dy = fy - sy;

            if (sx >= 0 && sx < src_w - 1 && sy >= 0 && sy < src_h - 1) {
                for (int c = 0; c < channels; ++c) {
                    const uint8_t* s = src + sy * src_step + sx * channels + c;
                    float v00 = s[0];
                    float v01 = s[channels];
                    float v10 = s[src_step];
                    float v11 = s[src_step + channels];
                    float v = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01
                            + (1 - dx) * dy * v10 + dx * dy * v11;
                    d[x * channels + c] = (uint8_t)(v + 0.5f);
                }
            } else if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h) {
                for (int c = 0; c < channels; ++c)
                    d[x * channels + c] = src[sy * src_step + sx * channels + c];
            } else {
                for (int c = 0; c < channels; ++c)
                    d[x * channels + c] = 0;
            }
        }
    }
}
