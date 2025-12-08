#include "DocInfer.h"
#include "LiteOCREngine.h"
#include "opencv2/core/types.hpp"

#include <fstream>
#include <sstream>

namespace LiteOCR {

    bool is_ocr_box_inside_cell(const TextBox &ocr_obj, const std::array<float, 8> &cell_coords) {
        // Convert cell_coords to a cv::Rect for easier intersection check
        // Assuming cell_coords are [x1, y1, x2, y2, x3, y3, x4, y4] representing a rotated rectangle
        // For simplicity, we'll use the bounding box of the cell_coords.
        // A more accurate approach would involve checking polygon intersection.

        float min_x = std::min({cell_coords[0], cell_coords[2], cell_coords[4], cell_coords[6]});
        float max_x = std::max({cell_coords[0], cell_coords[2], cell_coords[4], cell_coords[6]});
        float min_y = std::min({cell_coords[1], cell_coords[3], cell_coords[5], cell_coords[7]});
        float max_y = std::max({cell_coords[1], cell_coords[3], cell_coords[5], cell_coords[7]});

        cv::Rect cell_bbox(static_cast<int>(min_x), static_cast<int>(min_y),
                        static_cast<int>(max_x - min_x), static_cast<int>(max_y - min_y));

        cv::Rect ocr_bbox = cv::RotatedRect(
            cv::Point2f(ocr_obj.box.center.x, ocr_obj.box.center.y),
            cv::Size2f(ocr_obj.box.size.width, ocr_obj.box.size.height),
            ocr_obj.box.angle
        ).boundingRect();

        // Check IOU (Intersection over Union) or simple intersection
        cv::Rect intersection = cell_bbox & ocr_bbox;
        float intersection_area = static_cast<float>(intersection.area());
        float ocr_area = static_cast<float>(ocr_bbox.area());
        if (ocr_area <= 0) return false;
        float iou = intersection_area / ocr_area;
        return iou > 0.5f; // Consider inside if more than 50% of OCR box is inside cell
    }

    std::pair<std::string, std::vector<Rect>> merge_table_ocr(
        const std::vector<std::pair<std::string, std::array<float, 8>>> &table_structure,
        const std::vector<TextBox> &detected_text_objects,
        const std::vector<Textline> &recognized_texts) {
        std::string html_output = "<table>";
        std::string last_tag_content = ""; // To store content for tags like <td ...>
        std::vector<Rect> cell_rects;

        for (const auto &entry: table_structure) {
            const std::string &tag = entry.first;
            const std::array<float, 8> &coords = entry.second;

            if (tag.substr(0, 3) == "<td") {
                // This is a table data cell tag (e.g., <td> or <td colspan="2">)
                std::string cell_text = "";

                // Find OCR results that fall within this cell's coordinates
                for (size_t i = 0; i < detected_text_objects.size(); ++i) {
                    if (is_ocr_box_inside_cell(detected_text_objects[i], coords)) {
                        cell_text += recognized_texts[i].text; // Concatenate recognized text
                    }
                }

                if (tag == "<td></td>") {
                    // Simple <td> tag, just append the text
                    html_output += "<td>" + cell_text + "</td>";
                } else {
                    // Tag with attributes (e.g., <td colspan="2">)
                    // Store the text to be inserted when the '>' is encountered
                    html_output += tag; // Append the opening tag part (e.g., "<td colspan="2"")
                    last_tag_content = cell_text; // Store the text for later
                }

                // Store cell rectangle
                float min_x = std::min({coords[0], coords[2], coords[4], coords[6]});
                float max_x = std::max({coords[0], coords[2], coords[4], coords[6]});
                float min_y = std::min({coords[1], coords[3], coords[5], coords[7]});
                float max_y = std::max({coords[1], coords[3], coords[5], coords[7]});
                cell_rects.push_back(Rect{min_x, min_y, max_x - min_x, max_y - min_y});
            } else if (tag == ">") {
                // This signifies the end of an opening tag, and content should follow
                if (!last_tag_content.empty()) {
                    html_output += ">" + last_tag_content; // Append '>' and the stored text
                    last_tag_content = ""; // Clear the stored content
                } else {
                    html_output += tag; // Just append '>' if no content was stored (e.g., for <th> or other tags)
                }
            } else {
                // Other HTML tags (e.g., <tr>, <th>, </tr>, </table>)
                html_output += tag;
            }
        }

        html_output += "</table>";
        return {html_output, cell_rects};
    }

    bool PaddleSLANet::loadModel(const char* cnnParamPath, const char* cnnBinPath,
                                 const char* slaheadParamPath, const char* slaheadBinPath,
                                 const char* vocabPath,
                                 const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                cnnModel.set_vulkan_device(opt.gpuDeviceId);
                slaheadModel.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        cnnModel.opt.num_threads = opt.numThreads;
        slaheadModel.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            cnnModel.opt.use_fp16_arithmetic = true;
            cnnModel.opt.use_fp16_storage = true;
            cnnModel.opt.use_fp16_packed = true;

            slaheadModel.opt.use_fp16_arithmetic = true;
            slaheadModel.opt.use_fp16_storage = true;
            slaheadModel.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            cnnModel.opt.use_int8_arithmetic = true;
            cnnModel.opt.use_int8_storage = true;
            cnnModel.opt.use_int8_packed = true;

            slaheadModel.opt.use_int8_arithmetic = true;
            slaheadModel.opt.use_int8_storage = true;
            slaheadModel.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = cnnModel.load_param(cnnParamPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet CNN param file from %s\n", cnnParamPath);
            return false;
        }
        ret = cnnModel.load_model(cnnBinPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet CNN bin file from %s\n", cnnBinPath);
            return false;
        }
        ret = slaheadModel.load_param(slaheadParamPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet SLA-Head param file from %s\n", slaheadParamPath);
            return false;
        }
        ret = slaheadModel.load_model(slaheadBinPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet SLA-Head bin file from %s\n", slaheadBinPath);
            return false;
        }
        // load vocab
        vocab.clear();
        std::ifstream vocabFile(vocabPath);
        if (!vocabFile.is_open()) {
            fprintf(stderr, "[LiteOCR]Failed to open vocab file from %s\n", vocabPath);
            return false;
        }
        std::string line;
        while (std::getline(vocabFile, line)) {
            vocab.push_back(line);
        }

        return true;
    }

    bool PaddleSLANet::loadModelFromBuffer(const char* cnnParamBuffer, const unsigned char* cnnBinBuffer,
                                           const char* slaheadParamBuffer, const unsigned char* slaheadBinBuffer,
                                           const char* vocabBuffer,
                                           const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                cnnModel.set_vulkan_device(opt.gpuDeviceId);
                slaheadModel.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        cnnModel.opt.num_threads = opt.numThreads;
        slaheadModel.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            cnnModel.opt.use_fp16_arithmetic = true;
            cnnModel.opt.use_fp16_storage = true;
            cnnModel.opt.use_fp16_packed = true;

            slaheadModel.opt.use_fp16_arithmetic = true;
            slaheadModel.opt.use_fp16_storage = true;
            slaheadModel.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            cnnModel.opt.use_int8_arithmetic = true;
            cnnModel.opt.use_int8_storage = true;
            cnnModel.opt.use_int8_packed = true;

            slaheadModel.opt.use_int8_arithmetic = true;
            slaheadModel.opt.use_int8_storage = true;
            slaheadModel.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = cnnModel.load_param_mem(cnnParamBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet CNN param from buffer\n");
            return false;
        }
        ret = cnnModel.load_model(cnnBinBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet CNN bin from buffer\n");
            return false;
        }
        ret = slaheadModel.load_param_mem(slaheadParamBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet SLA-Head param from buffer\n");
            return false;
        }
        ret = slaheadModel.load_model(slaheadBinBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet SLA-Head bin from buffer\n");
            return false;
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

    std::vector<std::pair<std::string,std::array<float,8>>> PaddleSLANet::forward(const cv::Mat& input) {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(input.data, ncnn::Mat::PIXEL_BGR, input.cols, input.rows, target_size, target_size);
        in.substract_mean_normalize(mean_vals, norm_vals);

        auto ex = cnnModel.create_extractor();
        ex.input("in0", in);
        ncnn::Mat feat;
        ex.extract("out0", feat);

        feat = feat.reshape(96, 256);
        ncnn::Mat hidden(256, 1);
        ncnn::Mat one_hot_feat(50);

        hidden.fill(0.0f);
        one_hot_feat.fill(0.0f);
        one_hot_feat[0] = 1.0f;

        int step = 0;
        static const int max_step = 1024;
        static const int eos = 49;

        std::vector<std::pair<std::string, std::array<float, 8>>> result;

        while (step < max_step) {
            auto ex2 = slaheadModel.create_extractor();
            ex2.input("in0", hidden.clone());
            ex2.input("in1", feat.clone());
            ex2.input("in2", one_hot_feat.clone());

            ncnn::Mat hidden2, structure, loc;
            ex2.extract("out0", hidden2);
            ex2.extract("out1", structure);
            ex2.extract("out2", loc);

            hidden = hidden2.clone();

            int token = 0;
            float max_score = -1e30;
            for (int i = 0; i < 50; i++) {
                if (structure[i] > max_score) {
                    max_score = structure[i];
                    token = i;
                }
            }

            if (token == eos) break;

            std::string code = vocab[token - 1];
            std::array<float, 8> locs;
            for (int i = 0; i < 8; i += 2) {
                locs[i] = loc[i] * input.cols;
            }
            for (int i = 1; i < 8; i += 2) {
                locs[i] = loc[i] * input.rows;
            }
            result.push_back(std::make_pair(code, locs));

            one_hot_feat.fill(0.0f);
            one_hot_feat[token] = 1.0f;
            step++;
        }

        return result;
    }


}