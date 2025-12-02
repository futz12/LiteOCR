#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "BaseInfer.h"

int main()
{
    std::cout << "LiteOCR Recognizer Test" << std::endl;

    cv::Mat input = cv::imread("test_line.png", cv::IMREAD_COLOR);

    LiteOCR::PaddleRecognizer recognizer;
    recognizer.loadModel("./models/PP-OCRv5_mobile_rec.param", "./models/PP-OCRv5_mobile_rec.bin", LiteOCR::InferOption());

    cv::Mat output = recognizer.forward(input);

    std::cout << "Output size: " << output.cols << "x" << output.rows << std::endl;

    auto results = LiteOCR::CTCDecoder::decode(output, 0);

    std::cout << "Decoded results: " << std::endl;
    for (const auto& res : results) {
        int token;
        float prob;
        int index;
        std::tie(token, prob, index) = res;
        std::cout << "Index: " << index << ", Token: " << token << ", Prob: " << prob << std::endl;
    }

    return 0;
}