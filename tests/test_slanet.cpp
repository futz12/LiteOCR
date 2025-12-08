#include "DocInfer.h"
#include "LiteOCREngine.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>


int main()
{
    LiteOCR::PaddleSLANet infer;
    infer.loadModel(
        "./models/PP-StructrureV2_SLANet_plus_cnn.param",
        "./models/PP-StructrureV2_SLANet_plus_cnn.bin",
        "./models/PP-StructrureV2_SLANet_plus_slahead.param",
        "./models/PP-StructrureV2_SLANet_plus_slahead.bin",
        "./models/table_structure_dict_ch.txt",LiteOCR::InferOption());

    cv::Mat input = cv::imread("./table.jpg");

    auto results = infer.forward(input);

    for (int i = 0; i < results.size(); i++)
    {
        std::cout << results[i].first;

        if (results[i].first != "<td>" && results[i].first != "<td" && results[i].first != "<td></td>")
            continue;
        // Draw box
        for (int j = 0; j < 4; j++)
        {
            cv::line(input, cv::Point(results[i].second[j * 2], results[i].second[j * 2 + 1]),
                cv::Point(results[i].second[((j + 1) % 4) * 2], results[i].second[((j + 1) % 4) * 2 + 1]),
                cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imshow("Result", input);
    cv::waitKey(0);

    return 0;
}