#include "liteocr_docinfer.h"
#include "liteocr_engine.h"
#include "contours/liteocr_contours.h"

#include <array>
#include <algorithm>
#include <fstream>
#include <sstream>



    bool is_ocr_box_inside_cell(const liteocr_text_box &ocr_obj, const std::array<float, 8> &cell_coords) {
        float cmin_x = std::min({cell_coords[0], cell_coords[2], cell_coords[4], cell_coords[6]});
        float cmax_x = std::max({cell_coords[0], cell_coords[2], cell_coords[4], cell_coords[6]});
        float cmin_y = std::min({cell_coords[1], cell_coords[3], cell_coords[5], cell_coords[7]});
        float cmax_y = std::max({cell_coords[1], cell_coords[3], cell_coords[5], cell_coords[7]});

        int cx1 = static_cast<int>(cmin_x);
        int cy1 = static_cast<int>(cmin_y);
        int cx2 = static_cast<int>(cmax_x);
        int cy2 = static_cast<int>(cmax_y);

        float cx = 0, cy = 0;
        for (int i = 0; i < 4; ++i) {
            cx += ocr_obj.points[i * 2 + 0];
            cy += ocr_obj.points[i * 2 + 1];
        }
        cx /= 4.0f;
        cy /= 4.0f;
        float w = std::sqrt(
            (ocr_obj.points[2] - ocr_obj.points[0]) * (ocr_obj.points[2] - ocr_obj.points[0]) +
            (ocr_obj.points[3] - ocr_obj.points[1]) * (ocr_obj.points[3] - ocr_obj.points[1])
        );
        float h = std::sqrt(
            (ocr_obj.points[6] - ocr_obj.points[0]) * (ocr_obj.points[6] - ocr_obj.points[0]) +
            (ocr_obj.points[7] - ocr_obj.points[1]) * (ocr_obj.points[7] - ocr_obj.points[1])
        );
        float dx = ocr_obj.points[2] - ocr_obj.points[0];
        float dy = ocr_obj.points[3] - ocr_obj.points[1];
        float angle = std::atan2(dy, dx) * 180.0f / 3.14159265f;
        auto pts = liteocr_get_rotated_rect_points({
            {cx, cy},
            {w, h},
            angle
        });
        float omin_x = pts[0].x, omax_x = pts[0].x;
        float omin_y = pts[0].y, omax_y = pts[0].y;
        for (int i = 1; i < 4; ++i) {
            omin_x = std::min(omin_x, pts[i].x);
            omax_x = std::max(omax_x, pts[i].x);
            omin_y = std::min(omin_y, pts[i].y);
            omax_y = std::max(omax_y, pts[i].y);
        }
        int ox1 = static_cast<int>(omin_x);
        int oy1 = static_cast<int>(omin_y);
        int ox2 = static_cast<int>(omax_x);
        int oy2 = static_cast<int>(omax_y);

        int ix1 = std::max(cx1, ox1);
        int iy1 = std::max(cy1, oy1);
        int ix2 = std::min(cx2, ox2);
        int iy2 = std::min(cy2, oy2);
        float intersection_area = 0.0f;
        if (ix1 < ix2 && iy1 < iy2) {
            intersection_area = static_cast<float>((ix2 - ix1) * (iy2 - iy1));
        }
        float ocr_area = static_cast<float>((ox2 - ox1) * (oy2 - oy1));
        if (ocr_area <= 0) return false;
        float iou = intersection_area / ocr_area;
        return iou > 0.5f;
    }

    std::pair<std::string, std::vector<liteocr_rect>> liteocr_merge_table_ocr(
        const std::vector<std::pair<std::string, std::array<float, 8>>> &table_structure,
        const std::vector<liteocr_text_box> &detected_text_objects,
        const std::vector<liteocr_text_line> &recognized_texts) {
        std::string html_output = "<table>";
        std::string last_tag_content = "";
        std::vector<liteocr_rect> cell_rects;

        for (const auto &entry: table_structure) {
            const std::string &tag = entry.first;
            const std::array<float, 8> &coords = entry.second;

            if (tag.substr(0, 3) == "<td") {
                std::string cell_text = "";

                size_t ocr_count = std::min(detected_text_objects.size(), recognized_texts.size());
                for (size_t i = 0; i < ocr_count; ++i) {
                    if (is_ocr_box_inside_cell(detected_text_objects[i], coords)) {
                        cell_text += recognized_texts[i].text;
                    }
                }

                if (tag == "<td></td>") {
                    html_output += "<td>" + cell_text + "</td>";
                } else {
                    html_output += tag;
                    last_tag_content = cell_text;
                }

                float min_x = std::min({coords[0], coords[2], coords[4], coords[6]});
                float max_x = std::max({coords[0], coords[2], coords[4], coords[6]});
                float min_y = std::min({coords[1], coords[3], coords[5], coords[7]});
                float max_y = std::max({coords[1], coords[3], coords[5], coords[7]});
                cell_rects.push_back(liteocr_rect{min_x, min_y, max_x - min_x, max_y - min_y});
            } else if (tag == ">") {
                if (!last_tag_content.empty()) {
                    html_output += ">" + last_tag_content;
                    last_tag_content = "";
                } else {
                    html_output += tag;
                }
            } else {
                html_output += tag;
            }
        }

        html_output += "</table>";
        return {html_output, cell_rects};
    }

    bool liteocr_slanet_load_model(liteocr_slanet* sla, const char* cnnParamPath, const char* cnnBinPath,
                           const char* slaheadParamPath, const char* slaheadBinPath,
                           const char* vocabPath,
                           const liteocr_infer_option &opt) {
        if (!sla) return false;
        sla->mean_vals[0] = 0.485f * 255.f;
        sla->mean_vals[1] = 0.456f * 255.f;
        sla->mean_vals[2] = 0.406f * 255.f;
        sla->norm_vals[0] = 1 / (0.229f * 255.f);
        sla->norm_vals[1] = 1 / (0.224f * 255.f);
        sla->norm_vals[2] = 1 / (0.225f * 255.f);
        sla->target_size = 488;

        liteocr_apply_net_options(sla->cnn_model, opt);
        liteocr_apply_net_options(sla->slahead_model, opt);
        if (sla->cnn_model.load_param(cnnParamPath) == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet CNN param file from %s\n", cnnParamPath);
            return false;
        }
        if (sla->cnn_model.load_model(cnnBinPath) == 0) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet CNN bin file from %s\n", cnnBinPath);
            return false;
        }
        if (sla->slahead_model.load_param(slaheadParamPath) == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet SLA-Head param file from %s\n", slaheadParamPath);
            return false;
        }
        if (sla->slahead_model.load_model(slaheadBinPath) == 0) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet SLA-Head bin file from %s\n", slaheadBinPath);
            return false;
        }
        sla->vocab.clear();
        std::ifstream vocabFile(vocabPath);
        if (!vocabFile.is_open()) {
            fprintf(stderr, "[LiteOCR]Failed to open vocab file from %s\n", vocabPath);
            return false;
        }
        std::string line;
        while (std::getline(vocabFile, line)) {
            sla->vocab.push_back(line);
        }

        return true;
    }

    bool liteocr_slanet_load_model_from_buffer(liteocr_slanet* sla, const char* cnnParamBuffer, const unsigned char* cnnBinBuffer,
                                       const char* slaheadParamBuffer, const unsigned char* slaheadBinBuffer,
                                       const char* vocabBuffer,
                                       const liteocr_infer_option &opt) {
        if (!sla) return false;
        sla->mean_vals[0] = 0.485f * 255.f;
        sla->mean_vals[1] = 0.456f * 255.f;
        sla->mean_vals[2] = 0.406f * 255.f;
        sla->norm_vals[0] = 1 / (0.229f * 255.f);
        sla->norm_vals[1] = 1 / (0.224f * 255.f);
        sla->norm_vals[2] = 1 / (0.225f * 255.f);
        sla->target_size = 488;

        liteocr_apply_net_options(sla->cnn_model, opt);
        liteocr_apply_net_options(sla->slahead_model, opt);
        if (sla->cnn_model.load_param_mem(cnnParamBuffer) == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet CNN param from buffer\n");
            return false;
        }
        if (sla->cnn_model.load_model(cnnBinBuffer) == 0) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet CNN bin from buffer\n");
            return false;
        }
        if (sla->slahead_model.load_param_mem(slaheadParamBuffer) == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet SLA-Head param from buffer\n");
            return false;
        }
        if (sla->slahead_model.load_model(slaheadBinBuffer) == 0) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleSLANet SLA-Head bin from buffer\n");
            return false;
        }

        sla->vocab.clear();
        std::istringstream vocabStream(vocabBuffer);
        std::string line;
        while (std::getline(vocabStream, line)) {
            sla->vocab.push_back(line);
        }
        return true;
    }

    std::vector<std::pair<std::string, std::array<float, 8>>> liteocr_slanet_forward(liteocr_slanet* sla, const liteocr_image& input) {
        if (!sla || input.empty() || input.channels != 3) return {};
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(input.data, ncnn::Mat::PIXEL_BGR, input.width, input.height, sla->target_size, sla->target_size);
        in.substract_mean_normalize(sla->mean_vals, sla->norm_vals);

        auto ex = sla->cnn_model.create_extractor();
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
            auto ex2 = sla->slahead_model.create_extractor();
            ex2.input("in0", hidden.clone());
            ex2.input("in1", feat.clone());
            ex2.input("in2", one_hot_feat.clone());

            ncnn::Mat hidden2, structure, loc;
            ex2.extract("out0", hidden2);
            ex2.extract("out1", structure);
            ex2.extract("out2", loc);

            hidden = hidden2.clone();

            int token = 0;
            float max_score = -1e30f;
            for (int i = 0; i < 50; i++) {
                if (structure[i] > max_score) {
                    max_score = structure[i];
                    token = i;
                }
            }

            if (token == eos) break;
            if (token <= 0 || token > (int)sla->vocab.size()) {
                one_hot_feat.fill(0.0f);
                one_hot_feat[0] = 1.0f;
                step++;
                continue;
            }

            std::string code = sla->vocab[token - 1];
            std::array<float, 8> locs;
            for (int i = 0; i < 8; i += 2) {
                locs[i] = loc[i] * input.width;
            }
            for (int i = 1; i < 8; i += 2) {
                locs[i] = loc[i] * input.height;
            }
            result.push_back(std::make_pair(code, locs));

            one_hot_feat.fill(0.0f);
            one_hot_feat[token] = 1.0f;
            step++;
        }

        return result;
    }


