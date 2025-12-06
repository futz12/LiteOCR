#include "BaseInfer.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int main()
{
    cv::Mat input = cv::imread("doc_test.jpg", cv::IMREAD_COLOR);
    LiteOCR::PaddleUVDoc uvdoc;
    uvdoc.loadModel("./models/PP-UVDoc.param", "./models/PP-UVDoc.bin", LiteOCR::InferOption());
    cv::Mat output = uvdoc.forward(input);
    cv::imwrite("uvdoc_output.jpg", output);
    return 0;
}