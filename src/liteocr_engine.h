#pragma once

#include <string>
#include <vector>
#include "liteocr_image.h"
#include <net.h>
#include <tuple>
#include "liteocr_docinfer.h"

struct liteocr_infer_option {
    int num_threads = 4;
    int gpu_device_id = -1;
    bool use_fp16 = false;
    bool use_int8 = false;
    bool use_int8_det = false;
    bool use_int8_rec = false;
    bool use_bf16 = false;
    int textline_ori_model_type = 0; /* 0 = Paddle (PP-LCNet), 1 = AngleNet */
};

void liteocr_apply_net_options(ncnn::Net& model, const liteocr_infer_option& opt);

struct liteocr_detector {
    ncnn::Net model;
    float mean_vals[3];
    float norm_vals[3];
    int stride;
};

struct liteocr_recognizer {
    ncnn::Net model;
    float mean_vals[3];
    float norm_vals[3];
    int target_height;
};

struct liteocr_textline_ori {
    ncnn::Net model;
    float mean_vals[3];
    float norm_vals[3];
    int target_width;
    int target_height;
    bool is_anglenet = false;
};

struct liteocr_doc_ori {
    ncnn::Net model;
    float mean_vals[3];
    float norm_vals[3];
    int target_width;
    int target_height;
};

struct liteocr_uvdoc {
    ncnn::Net model;
    float norm_vals[3];
};

bool liteocr_detector_load_model(liteocr_detector* det, const char* param_path, const char* bin_path, const liteocr_infer_option& opt);
bool liteocr_detector_load_model_from_buffer(liteocr_detector* det, const char* param_buffer, const unsigned char* bin_buffer, const liteocr_infer_option& opt);
liteocr_image liteocr_detector_forward(liteocr_detector* det, const liteocr_image& input);

bool liteocr_recognizer_load_model(liteocr_recognizer* rec, const char* param_path, const char* bin_path, const liteocr_infer_option& opt);
bool liteocr_recognizer_load_model_from_buffer(liteocr_recognizer* rec, const char* param_buffer, const unsigned char* bin_buffer, const liteocr_infer_option& opt);
liteocr_image liteocr_recognizer_forward(liteocr_recognizer* rec, const liteocr_image& input);

bool liteocr_textline_ori_load_model(liteocr_textline_ori* cls, const char* param_path, const char* bin_path, const liteocr_infer_option& opt);
bool liteocr_textline_ori_load_model_from_buffer(liteocr_textline_ori* cls, const char* param_buffer, const unsigned char* bin_buffer, const liteocr_infer_option& opt);
int liteocr_textline_ori_forward(liteocr_textline_ori* cls, const liteocr_image& input);

bool liteocr_doc_ori_load_model(liteocr_doc_ori* cls, const char* param_path, const char* bin_path, const liteocr_infer_option& opt);
bool liteocr_doc_ori_load_model_from_buffer(liteocr_doc_ori* cls, const char* param_buffer, const unsigned char* bin_buffer, const liteocr_infer_option& opt);
int liteocr_doc_ori_forward(liteocr_doc_ori* cls, const liteocr_image& input);

bool liteocr_uvdoc_load_model(liteocr_uvdoc* uv, const char* param_path, const char* bin_path, const liteocr_infer_option& opt);
bool liteocr_uvdoc_load_model_from_buffer(liteocr_uvdoc* uv, const char* param_buffer, const unsigned char* bin_buffer, const liteocr_infer_option& opt);
liteocr_image liteocr_uvdoc_forward(liteocr_uvdoc* uv, const liteocr_image& input);

std::vector<std::tuple<int, float, int>> liteocr_ctc_decode(const liteocr_image& probs, int blank_index = 0);

struct liteocr_ocr_engine {
    liteocr_detector detector;
    liteocr_recognizer recognizer;
    liteocr_textline_ori textline_ori;
    bool has_textline_ori;
    std::vector<std::string> vocab;
    float threshold;
    float box_threshold;
    int max_candidates;
    float unclip_ratio;
    int min_size;
    int target_height;
};

struct liteocr_table_engine {
    liteocr_slanet slanet;
};

void liteocr_ocr_engine_init(liteocr_ocr_engine* engine);
bool liteocr_ocr_engine_load_model(liteocr_ocr_engine* engine,
    const char* det_param_path, const char* det_bin_path,
    const char* rec_param_path, const char* rec_bin_path,
    const char* vocab_path,
    const char* ori_param_path, const char* ori_bin_path,
    const liteocr_infer_option& opt);
bool liteocr_ocr_engine_load_model_from_buffer(liteocr_ocr_engine* engine,
    const char* det_param_buffer, const unsigned char* det_bin_buffer,
    const char* rec_param_buffer, const unsigned char* rec_bin_buffer,
    const char* vocab_buffer,
    const char* ori_param_buffer, const unsigned char* ori_bin_buffer,
    const liteocr_infer_option& opt);
std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>> liteocr_ocr_engine_recognize(liteocr_ocr_engine* engine, const liteocr_image& img);
std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>> liteocr_ocr_engine_recognize_raw(liteocr_ocr_engine* engine, const unsigned char* img_data, int width, int height, int channels, int cstep);
std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>> liteocr_ocr_engine_recognize_buffer(liteocr_ocr_engine* engine, const unsigned char* img_data, int size);

void liteocr_table_engine_init(liteocr_table_engine* engine);
bool liteocr_table_engine_load_model(liteocr_table_engine* engine,
    const char* cnn_param_path, const char* cnn_bin_path,
    const char* slahead_param_path, const char* slahead_bin_path,
    const char* vocab_path,
    const liteocr_infer_option& opt);
bool liteocr_table_engine_load_model_from_buffer(liteocr_table_engine* engine,
    const char* cnn_param_buffer, const unsigned char* cnn_bin_buffer,
    const char* slahead_param_buffer, const unsigned char* slahead_bin_buffer,
    const char* vocab_buffer,
    const liteocr_infer_option& opt);
std::pair<std::string, std::vector<liteocr_rect>> liteocr_table_engine_recognize(liteocr_table_engine* engine, const liteocr_image& img, const std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>>& ocr_result);
std::pair<std::string, std::vector<liteocr_rect>> liteocr_table_engine_recognize_raw(liteocr_table_engine* engine, const unsigned char* img_data, int width, int height, int channels, int cstep, const std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>>& ocr_result);
std::pair<std::string, std::vector<liteocr_rect>> liteocr_table_engine_recognize_buffer(liteocr_table_engine* engine, const unsigned char* img_data, int size, const std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>>& ocr_result);
