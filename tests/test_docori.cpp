#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "BaseInfer.h"

int main()
{
    std::cout << "LiteOCR Textline Orientation Classifier Test" << std::endl;
    cv::Mat input = cv::imread("test_line.png", cv::IMREAD_COLOR);

    LiteOCR::PaddleDocORI classifier;
    classifier.loadModel("./models/PP-LCNet_x1_0_doc_ori.param", "./models/PP-LCNet_x1_0_doc_ori.bin", LiteOCR::InferOption());

    int orientation = classifier.forward(input);
    std::cout << "Predicted orientation: " << orientation << std::endl;

    cv::rotate(input, input, cv::ROTATE_180);
    int orientation2 = classifier.forward(input);
    std::cout << "Predicted orientation after rotation: " << orientation2 << std::endl;

    if (orientation != orientation2) {
        std::cout << "Orientation classifier works correctly." << std::endl;
    } else {
        std::cout << "Orientation classifier failed." << std::endl;
        return -1;
    }

    return 0;
}