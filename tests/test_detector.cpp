#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "BaseInfer.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>
using namespace std;
using namespace LiteOCR;

static double contour_score(const cv::Mat& binary, const std::vector<cv::Point>& contour)
{
    cv::Rect rect = cv::boundingRect(contour);
    if (rect.x < 0)
        rect.x = 0;
    if (rect.y < 0)
        rect.y = 0;
    if (rect.x + rect.width > binary.cols)
        rect.width = binary.cols - rect.x;
    if (rect.y + rect.height > binary.rows)
        rect.height = binary.rows - rect.y;

    cv::Mat binROI = binary(rect);

    cv::Mat mask = cv::Mat::zeros(rect.height, rect.width, CV_8U);
    std::vector<cv::Point> roiContour;
    for (size_t i = 0; i < contour.size(); i++)
    {
        cv::Point pt = cv::Point(contour[i].x - rect.x, contour[i].y - rect.y);
        roiContour.push_back(pt);
    }

    std::vector<std::vector<cv::Point> > roiContours = {roiContour};
    cv::fillPoly(mask, roiContours, cv::Scalar(255));

    double score = cv::mean(binROI, mask).val[0];
    return score;
}

int main() {
    cout << "LiteOCR Detector Test" << std::endl;

    cv::Mat input = cv::imread("test.png", cv::IMREAD_COLOR);

    cout << "Image size: " << input.cols << "x" << input.rows << endl;

    PaddleDetector detector;

    detector.loadModel("./models/PP-OCRv5_mobile_det.param", "./models/PP-OCRv5_mobile_det.bin", InferOption());
    cv::Mat output = detector.forward(input);
    cout << "Output size: " << output.cols << "x" << output.rows << endl;

    const float threshold = 0.3f;
    const float box_threshold = 0.6f;
    const int max_candidates = 1000;
    const float unclip_ratio = 1.95f;

    cv::Mat binary_map;
    cv::threshold(output, binary_map, threshold, 1, cv::THRESH_BINARY);
    binary_map.convertTo(binary_map, CV_8U, 255);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_map, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    contours.resize(std::min(contours.size(), static_cast<size_t>(max_candidates)));

    for (size_t i = 0; i < contours.size(); i++) {
        const std::vector<cv::Point>& contour = contours[i];
        if (contour.size() <= 2) {
            continue;
        }

        double score = contour_score(output, contour);
        if (score < box_threshold) {
            continue;
        }

        cv::RotatedRect box = cv::minAreaRect(contour);

        int orientation = 0;
            if (box.angle >= -30 && box.angle <= 30 && box.size.height > box.size.width * 2.7)
            {
                // vertical text
                orientation = 1;
            }
            if ((box.angle <= -60 || box.angle >= 60) && box.size.width > box.size.height * 2.7)
            {
                // vertical text
                orientation = 1;
            }

            if (box.angle < -30)
            {
                // make orientation from -90 ~ -30 to 90 ~ 150
                box.angle += 180;
            }
            if (orientation == 0 && box.angle < 30)
            {
                // make it horizontal
                box.angle += 90;
                std::swap(box.size.width, box.size.height);
            }
            if (orientation == 1 && box.angle >= 60)
            {
                // make it vertical
                box.angle -= 90;
                std::swap(box.size.width, box.size.height);
            }


        // enlarge
        box.size.height += box.size.width * (unclip_ratio - 1);
        box.size.width *= unclip_ratio;

        // draw box
        cv::Point2f vertices[4];
        box.points(vertices);
        for (int j = 0; j < 4; j++) {
            cv::line(input, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imwrite("output.png", input);


    return 0;
}