#pragma once

#include "LiteOCREngine.h"

#include <ncnn/net.h>
#include <opencv2/core.hpp>


namespace LiteOCR {

    std::pair<std::string, std::vector<Rect>> merge_table_ocr(
        const std::vector<std::pair<std::string, std::array<float, 8>>> &table_structure,
        const std::vector<TextBox> &detected_text_objects,
        const std::vector<Textline> &recognized_texts);
    
    class PaddleSLANet
    {
    public:
        PaddleSLANet() = default;
        ~PaddleSLANet() = default;

        bool loadModel(const char* cnnParamPath, const char* cnnBinPath,
                       const char* slaheadParamPath, const char* slaheadBinPath,
                       const char* vocabPath,
                       const InferOption &opt);
        bool loadModelFromBuffer(const char* cnnParamBuffer, const unsigned char* cnnBinBuffer,
                                 const char* slaheadParamBuffer, const unsigned char* slaheadBinBuffer,
                                 const char* vocabBuffer,
                                 const InferOption &opt);
        std::vector<std::pair<std::string,std::array<float,8>>> forward(const cv::Mat& input);
    private:
        ncnn::Net cnnModel;
        ncnn::Net slaheadModel;
        std::vector<std::string> vocab;

        const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
        const float norm_vals[3] = { 1 / (0.229f * 255.f), 1 / (0.224f * 255.f), 1 / (0.225f * 255.f) };
        const int target_size = 488;
    };

}