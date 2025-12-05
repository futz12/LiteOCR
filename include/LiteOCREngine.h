#pragma once

#include <string>
#include <vector>
#include <memory>

namespace LiteOCR {

    struct InferOption {
        int numThreads = 4;
        int gpuDeviceId = -1; // -1 means CPU
        bool useFp16 = false;
        bool useInt8 = false;
    };

    struct Textline {
        std::string text;
        std::vector<float> anchors; // position for each character in textline
    };

    struct TextBox {
        struct RotatedRect {
            struct Point {
                float x;
                float y;
            } center;
            struct Size {
                float width;
                float height;
            } size;
            float angle;
        } box;
        bool isVertical;
        float score;
    };

    class LiteOCREngineImpl;

    class LiteOCREngine {
    public:
        LiteOCREngine();
        ~LiteOCREngine();

        bool loadModel(const char* detParamPath, const char* detBinPath,
                       const char* recParamPath, const char* recBinPath,
                       const char* vocabPath,
                       const char* oriParamPath = nullptr, const char* oriBinPath = nullptr,
                       const InferOption &opt = InferOption());

        bool loadModelFromBuffer(const char* detParamBuffer, const unsigned char* detBinBuffer,
                                 const char* recParamBuffer, const unsigned char* recBinBuffer,
                                 const char* vocabBuffer,
                                 const char* oriParamBuffer = nullptr, const unsigned char* oriBinBuffer = nullptr,
                                 const InferOption &opt = InferOption());

        std::pair<std::vector<TextBox>, std::vector<Textline>> recognize(const void *cvMat);

        std::pair<std::vector<TextBox>, std::vector<Textline>> recognize(const unsigned char* imgData, int width, int height, int channels, int cstep);

        std::pair<std::vector<TextBox>, std::vector<Textline>> recognize(const unsigned char* imgData, int size);

        static std::string mergeTextBox(const std::vector<TextBox>& textBoxes);

    private:
        std::unique_ptr<LiteOCREngineImpl> impl; 
    };

} // namespace LiteOCR