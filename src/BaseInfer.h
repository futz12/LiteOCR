#pragma once

#include "LiteOCREngine.h"

#include <ncnn/net.h>
#include <opencv2/core.hpp>
#include <tuple>
#include <vector>

namespace LiteOCR {
    class BaseDetector {
    public:
        virtual ~BaseDetector() = default;
        virtual bool loadModel(const char* paramPath, const char* binPath, const InferOption &opt) = 0;
        virtual bool loadModelFromBuffer(const char *paramBuffer, const unsigned char *binBuffer, const InferOption &opt) = 0;
        virtual cv::Mat forward(const cv::Mat& input) = 0;

    };

    class BaseRecognizer {
    public:
        virtual ~BaseRecognizer() = default;
        virtual bool loadModel(const char* paramPath, const char* binPath, const InferOption &opt) = 0;
        virtual bool loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) = 0;

        virtual cv::Mat forward(const cv::Mat& input) = 0;
    };

    class BaseClassifier {
    public:
        virtual ~BaseClassifier() = default;
        virtual bool loadModel(const char* paramPath, const char* binPath, const InferOption &opt) = 0;
        virtual bool loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) = 0;

        virtual int forward(const cv::Mat& input) = 0;
    };

    class PaddleDetector : public BaseDetector {
    public:
        PaddleDetector() = default;
        ~PaddleDetector() override = default;

        bool loadModel(const char* paramPath, const char* binPath, const InferOption &opt) override;
        bool loadModelFromBuffer(const char *paramBuffer, const unsigned char *binBuffer, const InferOption &opt) override;
        cv::Mat forward(const cv::Mat& input) override;

    private:
        ncnn::Net model;

        const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
        const float norm_vals[3] = {1 / (0.229f * 255.f), 1 / (0.224f * 255.f), 1 / (0.225f * 255.f)};

        const int stride = 32;
    };

    class PaddleRecognizer : public BaseRecognizer {
    public:
        PaddleRecognizer() = default;
        ~PaddleRecognizer() override = default;

        bool loadModel(const char* paramPath, const char* binPath, const InferOption &opt) override;
        bool loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) override;
        cv::Mat forward(const cv::Mat& input) override;
    private:
        ncnn::Net model;

        const float mean_vals[3] = {0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f};
        const float norm_vals[3] = {1 / (0.5f * 255.f), 1 / (0.5f * 255.f), 1 / (0.5f * 255.f)};

        const int target_height = 48;
    };

    class PaddleTextlineORI : public BaseClassifier {
    public:
        PaddleTextlineORI() = default;
        ~PaddleTextlineORI() override = default;

        bool loadModel(const char* paramPath, const char* binPath, const InferOption &opt) override;
        bool loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) override;
        int forward(const cv::Mat& input) override;
    private:
        ncnn::Net model;

        const float mean_vals[3] = {0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f};
        const float norm_vals[3] = {1 / (0.5f * 255.f), 1 / (0.5f * 255.f), 1 / (0.5f * 255.f)};
        const int target_width = 160;
        const int target_height = 80;
    };

    class PaddleDocORI : public BaseClassifier {
    public:
        PaddleDocORI() = default;
        ~PaddleDocORI() override = default;

        bool loadModel(const char* paramPath, const char* binPath, const InferOption &opt) override;
        bool loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) override;
        int forward(const cv::Mat& input) override;
    private:
        ncnn::Net model;

        const float mean_vals[3] = {0.5 * 255.f, 0.5 * 255.f, 0.5 * 255.f};
        const float norm_vals[3] = {1 / (0.5 * 255.f), 1 / (0.5 * 255.f), 1 / (0.5 * 255.f)};
        const int target_width = 224;;
        const int target_height = 224;
    };


    class CTCDecoder {
    public:
        CTCDecoder() = default;
        ~CTCDecoder() = default;

        static std::vector<std::tuple<int, float, int>> decode(const cv::Mat& probs, int blank_index = 0); // return token, prob, index
    };
} // namespace LiteOCR