#include "LiteOCREngine.h"
#include "BaseInfer.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/opencv.hpp>
#include <ncnn/net.h>
#include <fstream>
#include <string>
#include <vector>

namespace LiteOCR {

static float contour_score(const cv::Mat& binary, const std::vector<cv::Point>& contour)
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
    cv::fillPoly(mask, roiContours, cv::Scalar(1.0f));

    float score = cv::mean(binROI, mask).val[0];
    return score;
}

class LiteOCREngineImpl {
private:
    std::unique_ptr<LiteOCR::BaseDetector> detector;
    std::unique_ptr<LiteOCR::BaseRecognizer> recognizer;
    std::unique_ptr<LiteOCR::BaseClassifier> textlineORI;

    std::vector<std::string> vocab;

    const float threshold = 0.3f;
    const float box_threshold = 0.6f;
    const int max_candidates = 1000;
    const float unclip_ratio = 1.95f;
    const int target_height = 48;

public:
    LiteOCREngineImpl() {
        
    }

    bool loadModel(const char* detParamPath, const char* detBinPath,
                   const char* recParamPath, const char* recBinPath,
                   const char* vocabPath,
                   const char* oriParamPath,
                   const char* oriBinPath,
                   const LiteOCR::InferOption &opt) {
        detector = std::unique_ptr<LiteOCR::BaseDetector>(new LiteOCR::PaddleDetector());
        recognizer = std::unique_ptr<LiteOCR::BaseRecognizer>(new LiteOCR::PaddleRecognizer());

        bool ret = detector->loadModel(detParamPath, detBinPath, opt);
        if (!ret) {
            fprintf(stderr, "[LiteOCR]Failed to load detector model from %s and %s\n", detParamPath, detBinPath);
            detector.reset();
            recognizer.reset();
            return false;
        }
        ret = recognizer->loadModel(recParamPath, recBinPath, opt);
        if (!ret) {
            fprintf(stderr, "[LiteOCR]Failed to load recognizer model from %s and %s\n", recParamPath, recBinPath);
            detector.reset();
            recognizer.reset();
            return false;
        }
        if (oriParamPath && oriBinPath) {
            textlineORI = std::unique_ptr<LiteOCR::BaseClassifier>(new LiteOCR::PaddleTextlineORI());
            ret = textlineORI->loadModel(oriParamPath, oriBinPath, opt);
            if (!ret) {
                fprintf(stderr, "[LiteOCR]Failed to load textline orientation model from %s and %s\n", oriParamPath, oriBinPath);
                detector.reset();
                recognizer.reset();
                textlineORI.reset();
                return false;
            }
        }
        // load vocab
        vocab.clear();
        std::ifstream vocabFile(vocabPath);
        if (!vocabFile.is_open()) {
            fprintf(stderr, "[LiteOCR]Failed to open vocab file from %s\n", vocabPath);
            detector.reset();
            recognizer.reset();
            textlineORI.reset();
            return false;
        }
        std::string line;
        while (std::getline(vocabFile, line)) {
            vocab.push_back(line);
        }
        vocabFile.close();
        return true;
    }

    bool loadModelFromBuffer(const char* detParamBuffer, const unsigned char* detBinBuffer,
                             const char* recParamBuffer, const unsigned char* recBinBuffer,
                             const char* vocabBuffer,
                             const char* oriParamBuffer,
                             const unsigned char* oriBinBuffer,
                             const LiteOCR::InferOption &opt) {
        bool ret = detector->loadModelFromBuffer(detParamBuffer, detBinBuffer, opt);
        if (!ret) return false;
        ret = recognizer->loadModelFromBuffer(recParamBuffer, recBinBuffer, opt);
        if (!ret) return false;
        if (oriParamBuffer && oriBinBuffer) {
            ret = textlineORI->loadModelFromBuffer(oriParamBuffer, oriBinBuffer, opt);
            if (!ret) return false;
        }
        // load vocab from buffer
        vocab.clear();
        std::istringstream vocabStream(vocabBuffer);
        std::string line;
        while (std::getline(vocabStream, line)) {
            vocab.push_back(line);
        }
        return true;
    }

    std::vector<TextBox> detect(const cv::Mat &input)
    {
        auto pred = detector->forward(input);
        cv::Mat binary;
        cv::threshold(pred, binary, threshold, 1, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8U, 255);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        contours.resize(std::min(contours.size(), (size_t)max_candidates));

        std::vector<TextBox> textBoxes;

        for (const auto& contour : contours) {
            if (contour.size() < 4) continue;

            float score = contour_score(pred, contour);
            if (score < box_threshold) continue;

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
            
            textBoxes.push_back(TextBox{
                .box = TextBox::RotatedRect{
                    .center = TextBox::RotatedRect::Point{box.center.x, box.center.y},
                    .size = TextBox::RotatedRect::Size{box.size.width, box.size.height},
                    .angle = box.angle
                },
                .isVertical = (orientation == 1),
                .score = score
            });
        }

        return textBoxes;
    }

    std::vector<Textline> recognize(const cv::Mat &input, std::vector<TextBox> &textBoxes)
    {
        std::vector<Textline> results;
        std::vector<cv::Mat> rois;

        for (const auto &textBox : textBoxes) {
            cv::Point2f corners[4];
            cv::RotatedRect(
                cv::Point2f(textBox.box.center.x, textBox.box.center.y),
                cv::Size2f(textBox.box.size.width, textBox.box.size.height),
                textBox.box.angle
            ).points(corners);

            int target_width = static_cast<int>(textBox.box.size.height * target_height / textBox.box.size.width);

            cv::Mat dst;

            if (!textBox.isVertical)
            {
                // horizontal text
                // corner points order
                //  0--------1
                //  |        |rw  -> as angle=90
                //  3--------2
                //      rh

                std::vector<cv::Point2f> src_pts(3);
                src_pts[0] = corners[0];
                src_pts[1] = corners[1];
                src_pts[2] = corners[3];

                std::vector<cv::Point2f> dst_pts(3);
                dst_pts[0] = cv::Point2f(0, 0);
                dst_pts[1] = cv::Point2f(target_width, 0);
                dst_pts[2] = cv::Point2f(0, target_height);

                cv::Mat tm = cv::getAffineTransform(src_pts, dst_pts);

                cv::warpAffine(input, dst, tm, cv::Size(target_width, target_height), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            }
            else
            {
                // vertial text
                // corner points order
                //  1----2
                //  |    |
                //  |    |
                //  |    |rh  -> as angle=0
                //  |    |
                //  |    |
                //  0----3
                //    rw

                std::vector<cv::Point2f> src_pts(3);
                src_pts[0] = corners[2];
                src_pts[1] = corners[3];
                src_pts[2] = corners[1];

                std::vector<cv::Point2f> dst_pts(3);
                dst_pts[0] = cv::Point2f(0, 0);
                dst_pts[1] = cv::Point2f(target_width, 0);
                dst_pts[2] = cv::Point2f(0, target_height);

                cv::Mat tm = cv::getAffineTransform(src_pts, dst_pts);

                cv::warpAffine(input, dst, tm, cv::Size(target_width, target_height), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            }

            rois.push_back(dst);
        }

        for (int i = 0; i < rois.size(); i++) {
            cv::Mat roi = rois[i];
            if (roi.isContinuous() == false) {
                roi = roi.clone();
            }

            if (textlineORI) {
                int ori_label = textlineORI->forward(roi);
                if (ori_label == 1) {
                    // upside down
                    cv::rotate(roi, roi, cv::ROTATE_180);
                    textBoxes[i].box.angle += 180.0f;
                }
            }

            auto textline = recognizer->forward(roi);
            auto decoded = CTCDecoder::decode(textline);

            std::string text;
            std::vector<float> anchors;

            for (const auto& [token, prob, index] : decoded) {
                if (token > 0 && token <= vocab.size()) {
                    text += vocab[token - 1];
                    float pos = (index + 0.5f) / textline.cols * roi.cols;
                    anchors.push_back(pos);
                } else if (!text.empty() && text.back() != ' ') {
                    text += ' ';
                    float pos = (index + 0.5f) / textline.cols * roi.cols;
                    anchors.push_back(pos);
                }
            }
            results.push_back({text, anchors});
        }

        return results;
    }

    std::pair<std::vector<TextBox>, std::vector<Textline>> run(const cv::Mat &input_)
    {
        // must be BGR format
        if (input_.empty()) {
            return {{}, {}};
        }

        cv::Mat input;
        if (input_.channels() == 1) {
            cv::cvtColor(input_, input, cv::COLOR_GRAY2BGR);
        } else if (input_.channels() == 4) {
            cv::cvtColor(input_, input, cv::COLOR_BGRA2BGR);
        } else {
            input = input_;
        }
        
        auto textBoxes = detect(input);
        auto textlines = recognize(input, textBoxes);
        return {textBoxes, textlines};
    }
};

LiteOCREngine::LiteOCREngine() : impl(nullptr) {}

LiteOCREngine::~LiteOCREngine() = default;

bool LiteOCREngine::loadModel(const char* detParamPath, const char* detBinPath,
                             const char* recParamPath, const char* recBinPath,
                             const char* vocabPath,
                             const char* oriParamPath,
                             const char* oriBinPath,
                             const InferOption &opt) {
    impl = std::make_unique<LiteOCREngineImpl>();
    return impl->loadModel(detParamPath, detBinPath, recParamPath, recBinPath, vocabPath, oriParamPath, oriBinPath, opt);
}

bool LiteOCREngine::loadModelFromBuffer(const char* detParamBuffer, const unsigned char* detBinBuffer,
                                       const char* recParamBuffer, const unsigned char* recBinBuffer,
                                       const char* vocabBuffer,
                                       const char* oriParamBuffer,
                                       const unsigned char* oriBinBuffer,
                                       const InferOption &opt) {
    impl = std::make_unique<LiteOCREngineImpl>();
    return impl->loadModelFromBuffer(detParamBuffer, detBinBuffer, recParamBuffer, recBinBuffer,
                                    vocabBuffer, oriParamBuffer, oriBinBuffer, opt);
}

std::pair<std::vector<TextBox>, std::vector<Textline>> LiteOCREngine::recognize(const void *cvMat) {
    const cv::Mat* mat = static_cast<const cv::Mat*>(cvMat);
    return impl->run(*mat);
}

std::pair<std::vector<TextBox>, std::vector<Textline>> LiteOCREngine::recognize(const unsigned char* imgData, int width, int height, int channels, int cstep) {
    cv::Mat img(height, width, (channels == 1) ? CV_8UC1 : ((channels == 3) ? CV_8UC3 : CV_8UC4), (void*)imgData, cstep);
    return impl->run(img);
}

std::pair<std::vector<TextBox>, std::vector<Textline>> LiteOCREngine::recognize(const unsigned char* imgData, int size) {
    std::vector<unsigned char> data(imgData, imgData + size);
    cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
    return impl->run(img);
}

} // namespace LiteOCR