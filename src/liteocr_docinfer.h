#pragma once

#include "liteocr_image.h"
#include <net.h>
#include <vector>
#include <string>
#include <array>

struct liteocr_infer_option;

struct liteocr_rect {
    float x;
    float y;
    float width;
    float height;
};

struct liteocr_text_box {
    float points[8];
    bool is_vertical;
    float score;
};

struct liteocr_text_line {
    std::string text;
    std::vector<float> anchors;
};

std::pair<std::string, std::vector<liteocr_rect>> liteocr_merge_table_ocr(
    const std::vector<std::pair<std::string, std::array<float, 8>>>& table_structure,
    const std::vector<liteocr_text_box>& detected_text_objects,
    const std::vector<liteocr_text_line>& recognized_texts);

struct liteocr_slanet {
    ncnn::Net cnn_model;
    ncnn::Net slahead_model;
    std::vector<std::string> vocab;
    float mean_vals[3];
    float norm_vals[3];
    int target_size;
};

bool liteocr_slanet_load_model(liteocr_slanet* sla, const char* cnn_param_path, const char* cnn_bin_path,
                               const char* slahead_param_path, const char* slahead_bin_path,
                               const char* vocab_path,
                               const liteocr_infer_option& opt);
bool liteocr_slanet_load_model_from_buffer(liteocr_slanet* sla, const char* cnn_param_buffer, const unsigned char* cnn_bin_buffer,
                                           const char* slahead_param_buffer, const unsigned char* slahead_bin_buffer,
                                           const char* vocab_buffer,
                                           const liteocr_infer_option& opt);
std::vector<std::pair<std::string, std::array<float, 8>>> liteocr_slanet_forward(liteocr_slanet* sla, const liteocr_image& input);
