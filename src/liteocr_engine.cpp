#include "stb_image.h"

#include "liteocr_engine.h"
#include "liteocr_docinfer.h"
#include "clipper.hpp"
#include "contours/liteocr_contours.h"
#include "liteocr_imgproc.h"
#include "liteocr_image.h"

#include <net.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <chrono>

void liteocr_apply_net_options(ncnn::Net& model, const liteocr_infer_option& opt) {
#if NCNN_VULKAN
    if (opt.gpu_device_id != -1) {
        if (ncnn::get_gpu_count() <= 0) {
            fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
        } else if (ncnn::get_gpu_count() <= opt.gpu_device_id) {
            fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpu_device_id);
        } else {
            model.set_vulkan_device(opt.gpu_device_id);
            model.opt.use_vulkan_compute = true;
        }
    }
#else
    if (opt.gpu_device_id != -1) {
        fprintf(stderr,
            "[LiteOCR] This build was compiled without Vulkan support (LITEOCR_ENABLE_VULKAN=OFF). "
            "gpu_device_id=%d will be ignored; falling back to CPU.\n",
            opt.gpu_device_id);
    }
#endif
    model.opt.num_threads = opt.num_threads;
    if (opt.use_fp16 && !opt.use_bf16) {
        model.opt.use_fp16_arithmetic = true;
        model.opt.use_fp16_storage = true;
        model.opt.use_fp16_packed = true;
    }
    if (opt.use_int8) {
        model.opt.use_int8_arithmetic = true;
        model.opt.use_int8_storage = true;
        model.opt.use_int8_packed = true;
    }
    if (opt.use_bf16) {
        model.opt.use_fp16_arithmetic = false;
        model.opt.use_fp16_storage = false;
        model.opt.use_fp16_packed = false;
        model.opt.use_bf16_storage = true;
        model.opt.use_bf16_packed = true;
    }
}



static liteocr_image loadImageFromBuffer(const unsigned char* data, int size)
{
    int w, h, c;
    unsigned char* pixels = stbi_load_from_memory(data, size, &w, &h, &c, 0);
    if (!pixels) {
        return liteocr_image();
    }

    liteocr_image img(w, h, liteocr_image_type::LITEOCR_IMAGE_U8C3);
    if (c == 1) {
        liteocr_cvt_color(pixels, w, h, w, LITEOCR_FORMAT_GRAY, img.data, img.stride, LITEOCR_FORMAT_BGR);
    } else if (c == 2) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                uint8_t gray = pixels[y * w * 2 + x * 2 + 0];
                img.data[y * img.stride + x * 3 + 0] = gray;
                img.data[y * img.stride + x * 3 + 1] = gray;
                img.data[y * img.stride + x * 3 + 2] = gray;
            }
        }
    } else if (c == 3) {
        liteocr_cvt_color(pixels, w, h, w * 3, LITEOCR_FORMAT_RGB, img.data, img.stride, LITEOCR_FORMAT_BGR);
    } else if (c == 4) {
        liteocr_cvt_color(pixels, w, h, w * 4, LITEOCR_FORMAT_RGBA, img.data, img.stride, LITEOCR_FORMAT_BGR);
    } else {
        stbi_image_free(pixels);
        return liteocr_image();
    }

    liteocr_image result = img.clone();
    stbi_image_free(pixels);
    return result;
}

// Order box points: tl, tr, br, bl
static std::vector<liteocr_point2f> orderBoxPoints(const std::vector<liteocr_point2f>& pts)
{
    auto sorted = pts;
    std::sort(sorted.begin(), sorted.end(), [](const liteocr_point2f& a, const liteocr_point2f& b) {
        return a.x < b.x;
    });
    auto tl = (sorted[0].y < sorted[1].y) ? sorted[0] : sorted[1];
    auto bl = (sorted[0].y < sorted[1].y) ? sorted[1] : sorted[0];
    auto tr = (sorted[2].y < sorted[3].y) ? sorted[2] : sorted[3];
    auto br = (sorted[2].y < sorted[3].y) ? sorted[3] : sorted[2];
    return {tl, tr, br, bl};
}

static std::pair<std::vector<liteocr_point2f>, float> orderMinAreaBoxPoints(const std::vector<liteocr_point>& contour)
{
    auto rrect = liteocr_min_area_rect(contour);
    auto pts = liteocr_get_rotated_rect_points(rrect);
    auto ordered = orderBoxPoints(pts);
    float minSide = std::min(rrect.size.width, rrect.size.height);
    return {ordered, minSide};
}

static float boxScore(const liteocr_image& prob, const std::vector<liteocr_point2f>& box)
{
    int h = prob.height;
    int w = prob.width;

    float xmin = std::max(0.0f, std::floor(std::min({box[0].x, box[1].x, box[2].x, box[3].x})));
    float xmax = std::min((float)(w - 1), std::ceil(std::max({box[0].x, box[1].x, box[2].x, box[3].x})));
    float ymin = std::max(0.0f, std::floor(std::min({box[0].y, box[1].y, box[2].y, box[3].y})));
    float ymax = std::min((float)(h - 1), std::ceil(std::max({box[0].y, box[1].y, box[2].y, box[3].y})));

    if (xmin >= xmax || ymin >= ymax) return 0.0f;

    int roiW = (int)(xmax - xmin + 1);
    int roiH = (int)(ymax - ymin + 1);
    std::vector<uint8_t> mask(roiH * roiW);
    std::vector<liteocr_point> shifted;
    for (const auto& p : box) {
        shifted.push_back({(int)(p.x - xmin), (int)(p.y - ymin)});
    }
    std::vector<std::vector<liteocr_point>> contours = {shifted};
    liteocr_fill_poly(mask.data(), roiW, roiH, roiW, contours, 1);

    const float* probData = prob.ptr<float>((int)ymin) + (int)xmin;
    float score = (float)liteocr_mean_masked(probData, roiW, roiH, prob.stride, mask.data(), roiW);
    return score;
}

static std::vector<liteocr_point2f> unclipBox(const std::vector<liteocr_point2f>& box, float unclip_ratio)
{
    std::vector<liteocr_point> intBox;
    for (const auto& p : box) {
        intBox.push_back({(int)p.x, (int)p.y});
    }
    double area = liteocr_contour_area(intBox);
    double length = liteocr_arc_length(intBox, true);
    if (length < 1e-6) return box;
    double distance = area * unclip_ratio / length;

    const double SCALE = 1000.0;
    ClipperLib::Path path;
    for (const auto& p : box) {
        path.push_back(ClipperLib::IntPoint((ClipperLib::cInt)(p.x * SCALE), (ClipperLib::cInt)(p.y * SCALE)));
    }

    ClipperLib::ClipperOffset offset;
    offset.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    ClipperLib::Paths solution;
    offset.Execute(solution, distance * SCALE);

    if (solution.empty() || solution[0].empty()) {
        return box;
    }

    std::vector<liteocr_point2f> result;
    for (const auto& p : solution[0]) {
        result.push_back({(float)(p.X / SCALE), (float)(p.Y / SCALE)});
    }
    return result;
}

/* ============================================================================
 *  liteocr_ocr_engine
 * ============================================================================ */

void liteocr_ocr_engine_init(liteocr_ocr_engine* engine)
{
    engine->has_textline_ori = false;
    engine->threshold = 0.3f;
    engine->box_threshold = 0.6f;
    engine->max_candidates = 1000;
    engine->unclip_ratio = 1.5f;
    engine->min_size = 3;
    engine->target_height = 48;
}

bool liteocr_ocr_engine_load_model(liteocr_ocr_engine* engine,
    const char* detParamPath, const char* detBinPath,
    const char* recParamPath, const char* recBinPath,
    const char* vocabPath,
    const char* oriParamPath,
    const char* oriBinPath,
    const liteocr_infer_option &opt) {

    liteocr_infer_option det_opt = opt;
    liteocr_infer_option rec_opt = opt;
    if (!opt.use_int8) {
        det_opt.use_int8 = opt.use_int8_det;
        rec_opt.use_int8 = opt.use_int8_rec;
    }

    bool ret = liteocr_detector_load_model(&engine->detector, detParamPath, detBinPath, det_opt);
    if (!ret) {
        fprintf(stderr, "[LiteOCR]Failed to load detector model from %s and %s\n", detParamPath, detBinPath);
        return false;
    }
    ret = liteocr_recognizer_load_model(&engine->recognizer, recParamPath, recBinPath, rec_opt);
    if (!ret) {
        fprintf(stderr, "[LiteOCR]Failed to load recognizer model from %s and %s\n", recParamPath, recBinPath);
        return false;
    }
    if (oriParamPath && oriBinPath) {
        ret = liteocr_textline_ori_load_model(&engine->textline_ori, oriParamPath, oriBinPath, opt);
        if (!ret) {
            fprintf(stderr, "[LiteOCR]Failed to load textline orientation model from %s and %s\n", oriParamPath, oriBinPath);
            return false;
        }
        engine->has_textline_ori = true;
    }
    engine->vocab.clear();
    std::ifstream vocabFile(vocabPath);
    if (!vocabFile.is_open()) {
        fprintf(stderr, "[LiteOCR]Failed to open vocab file from %s\n", vocabPath);
        return false;
    }
    std::string line;
    while (std::getline(vocabFile, line)) {
        engine->vocab.push_back(line);
    }
    vocabFile.close();
    return true;
}

bool liteocr_ocr_engine_load_model_from_buffer(liteocr_ocr_engine* engine,
    const char* detParamBuffer, const unsigned char* detBinBuffer,
    const char* recParamBuffer, const unsigned char* recBinBuffer,
    const char* vocabBuffer,
    const char* oriParamBuffer,
    const unsigned char* oriBinBuffer,
    const liteocr_infer_option &opt) {

    liteocr_infer_option det_opt = opt;
    liteocr_infer_option rec_opt = opt;
    if (!opt.use_int8) {
        det_opt.use_int8 = opt.use_int8_det;
        rec_opt.use_int8 = opt.use_int8_rec;
    }

    bool ret = liteocr_detector_load_model_from_buffer(&engine->detector, detParamBuffer, detBinBuffer, det_opt);
    if (!ret) return false;
    ret = liteocr_recognizer_load_model_from_buffer(&engine->recognizer, recParamBuffer, recBinBuffer, rec_opt);
    if (!ret) return false;
    if (oriParamBuffer && oriBinBuffer) {
        ret = liteocr_textline_ori_load_model_from_buffer(&engine->textline_ori, oriParamBuffer, oriBinBuffer, opt);
        if (!ret) return false;
        engine->has_textline_ori = true;
    }
    engine->vocab.clear();
    std::istringstream vocabStream(vocabBuffer);
    std::string line;
    while (std::getline(vocabStream, line)) {
        engine->vocab.push_back(line);
    }
    return true;
}

static std::vector<liteocr_text_box> liteocr_ocr_engine_detect(liteocr_ocr_engine* engine, const liteocr_image &input)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    auto pred = liteocr_detector_forward(&engine->detector, input);
    auto t1 = std::chrono::high_resolution_clock::now();
    double det_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    liteocr_image binary(pred.width, pred.height, liteocr_image_type::LITEOCR_IMAGE_U8C1);
    liteocr_threshold(pred.ptr<float>(), pred.width, pred.height, pred.stride,
                       binary.ptr<uint8_t>(), binary.stride,
                       engine->threshold, 255);

    std::vector<std::vector<liteocr_point>> contours;
    liteocr_find_contours(binary.ptr<uint8_t>(), binary.width, binary.height, binary.stride,
                          contours, LITEOCR_CHAIN_APPROX_SIMPLE);
    auto t2 = std::chrono::high_resolution_clock::now();
    double post_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    contours.resize(std::min(contours.size(), (size_t)engine->max_candidates));

    std::vector<liteocr_text_box> textBoxes;
    int dst_w = input.width;
    int dst_h = input.height;
    float ws = (float)dst_w / pred.width;
    float hs = (float)dst_h / pred.height;

    auto t3 = std::chrono::high_resolution_clock::now();
    for (const auto& contour : contours) {
        if (contour.size() < 4) continue;

        auto boxResult = orderMinAreaBoxPoints(contour);
        const auto& pts = boxResult.first;
        float sside = boxResult.second;
        if (sside < engine->min_size) continue;

        float score = boxScore(pred, pts);
        if (score < engine->box_threshold) continue;

        auto expanded = unclipBox(pts, engine->unclip_ratio);
        if (expanded.size() < 4) continue;

        std::vector<liteocr_point> expandedContour;
        for (const auto& p : expanded) {
            expandedContour.push_back({(int)p.x, (int)p.y});
        }

        auto rrect = liteocr_min_area_rect(expandedContour);
        float sside2 = std::min(rrect.size.width, rrect.size.height);
        if (sside2 < engine->min_size + 2) continue;

        if (rrect.angle >= 90.0f) rrect.angle -= 180.0f;
        if (rrect.angle < -90.0f) rrect.angle += 180.0f;

        auto mappedPts = liteocr_get_rotated_rect_points(rrect);
        for (int i = 0; i < 4; ++i) {
            mappedPts[i].x = std::max(0.0f, std::min((float)std::round(mappedPts[i].x * ws), (float)dst_w));
            mappedPts[i].y = std::max(0.0f, std::min((float)std::round(mappedPts[i].y * hs), (float)dst_h));
        }

        float cx = 0, cy = 0;
        for (int i = 0; i < 4; ++i) {
            cx += mappedPts[i].x;
            cy += mappedPts[i].y;
        }
        cx /= 4.0f;
        cy /= 4.0f;

        auto _pts = liteocr_get_rotated_rect_points({
            {cx, cy},
            {rrect.size.width * ws, rrect.size.height * hs},
            rrect.angle
        });
        liteocr_text_box _tb;
        for (int i = 0; i < 4; ++i) {
            _tb.points[i * 2 + 0] = _pts[i].x;
            _tb.points[i * 2 + 1] = _pts[i].y;
        }
        _tb.is_vertical = false;
        _tb.score = score;
        textBoxes.push_back(_tb);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double boxproc_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    fprintf(stderr, "[PROFILE] detect: det_forward=%.2f ms, threshold/contours=%.2f ms, box_proc=%.2f ms, boxes=%zu\n",
            det_ms, post_ms, boxproc_ms, textBoxes.size());

    return textBoxes;
}

static std::vector<liteocr_text_line> liteocr_ocr_engine_recognize_internal(liteocr_ocr_engine* engine, const liteocr_image &input, std::vector<liteocr_text_box> &textBoxes)
{
    auto rec_t0 = std::chrono::high_resolution_clock::now();
    std::vector<liteocr_text_line> results;
    std::vector<liteocr_image> rois;
    double total_warp_ms = 0.0;
    double total_rec_ms = 0.0;
    double total_decode_ms = 0.0;
    double max_warp_ms = 0.0;
    size_t valid_boxes = 0;

    for (const auto &textBox : textBoxes) {
        auto warp_t0 = std::chrono::high_resolution_clock::now();
        float _cx = 0, _cy = 0;
        for (int i = 0; i < 4; ++i) {
            _cx += textBox.points[i * 2 + 0];
            _cy += textBox.points[i * 2 + 1];
        }
        _cx /= 4.0f;
        _cy /= 4.0f;
        float _w = std::sqrt(
            (textBox.points[2] - textBox.points[0]) * (textBox.points[2] - textBox.points[0]) +
            (textBox.points[3] - textBox.points[1]) * (textBox.points[3] - textBox.points[1])
        );
        float _h = std::sqrt(
            (textBox.points[6] - textBox.points[0]) * (textBox.points[6] - textBox.points[0]) +
            (textBox.points[7] - textBox.points[1]) * (textBox.points[7] - textBox.points[1])
        );
        float _angle = std::atan2(textBox.points[3] - textBox.points[1], textBox.points[2] - textBox.points[0]) * 180.0f / 3.14159265f;
        auto pts = liteocr_get_rotated_rect_points({
            {_cx, _cy},
            {_w, _h},
            _angle
        });
        liteocr_point2f corners[4];
        for (int i = 0; i < 4; ++i) {
            corners[i] = {pts[i].x, pts[i].y};
        }

        auto ordered = orderBoxPoints({
            {corners[0].x, corners[0].y},
            {corners[1].x, corners[1].y},
            {corners[2].x, corners[2].y},
            {corners[3].x, corners[3].y}
        });

        float crop_w = std::max(
            liteocr_norm(ordered[0].x - ordered[1].x, ordered[0].y - ordered[1].y),
            liteocr_norm(ordered[2].x - ordered[3].x, ordered[2].y - ordered[3].y)
        );
        float crop_h = std::max(
            liteocr_norm(ordered[0].x - ordered[3].x, ordered[0].y - ordered[3].y),
            liteocr_norm(ordered[1].x - ordered[2].x, ordered[1].y - ordered[2].y)
        );
        if (crop_w < 1 || crop_h < 1) {
            rois.push_back(liteocr_image());
            continue;
        }

        float src_pts[8] = {
            ordered[0].x, ordered[0].y,
            ordered[1].x, ordered[1].y,
            ordered[2].x, ordered[2].y,
            ordered[3].x, ordered[3].y
        };
        float dst_pts[8] = {
            0, 0,
            crop_w, 0,
            crop_w, crop_h,
            0, crop_h
        };

        float M[9];
        liteocr_get_perspective_transform(src_pts, dst_pts, M);
        liteocr_image dst((int)crop_w, (int)crop_h, input.type);
        liteocr_warp_perspective(input.data, input.width, input.height, input.stride, input.channels,
                                 dst.data, dst.width, dst.height, dst.stride, M);
        auto warp_t1 = std::chrono::high_resolution_clock::now();
        double warp_ms = std::chrono::duration<double, std::milli>(warp_t1 - warp_t0).count();
        total_warp_ms += warp_ms;
        max_warp_ms = std::max(max_warp_ms, warp_ms);
        valid_boxes++;

        if (dst.height / (float)dst.width >= 1.5f) {
            liteocr_image rot(dst.height, dst.width, dst.type);
            liteocr_rotate90(dst.data, dst.width, dst.height, dst.stride, dst.channels,
                              rot.data, rot.stride, true);
            dst = rot;
        }

        int rec_width = static_cast<int>(dst.width * engine->target_height / dst.height);
        liteocr_image resized(rec_width, engine->target_height, dst.type);
        liteocr_resize(dst.data, dst.width, dst.height, dst.stride, dst.channels,
                        resized.data, rec_width, engine->target_height, resized.stride);
        dst = resized;

        rois.push_back(dst);
    }

    for (size_t i = 0; i < rois.size(); i++) {
        liteocr_image roi = rois[i];
        if (roi.empty() || roi.width == 0 || roi.height == 0) {
            results.push_back({std::string(), std::vector<float>()});
            continue;
        }

        if (engine->has_textline_ori) {
            int ori_label = liteocr_textline_ori_forward(&engine->textline_ori, roi);
            if (ori_label == 1) {
                liteocr_image rot(roi.width, roi.height, roi.type);
                liteocr_rotate180(roi.data, roi.width, roi.height, roi.stride, roi.channels,
                                   rot.data, rot.stride);
                roi = rot;
                float _cx2 = 0, _cy2 = 0;
                for (int j = 0; j < 4; ++j) {
                    _cx2 += textBoxes[i].points[j * 2 + 0];
                    _cy2 += textBoxes[i].points[j * 2 + 1];
                }
                _cx2 /= 4.0f;
                _cy2 /= 4.0f;
                for (int j = 0; j < 4; ++j) {
                    textBoxes[i].points[j * 2 + 0] = 2 * _cx2 - textBoxes[i].points[j * 2 + 0];
                    textBoxes[i].points[j * 2 + 1] = 2 * _cy2 - textBoxes[i].points[j * 2 + 1];
                }
            }
        }

        auto rec_t0 = std::chrono::high_resolution_clock::now();
        auto textline = liteocr_recognizer_forward(&engine->recognizer, roi);
        auto rec_t1 = std::chrono::high_resolution_clock::now();
        auto decoded = liteocr_ctc_decode(textline);
        auto rec_t2 = std::chrono::high_resolution_clock::now();
        total_rec_ms += std::chrono::duration<double, std::milli>(rec_t1 - rec_t0).count();
        total_decode_ms += std::chrono::duration<double, std::milli>(rec_t2 - rec_t1).count();

        std::string text;
        std::vector<float> anchors;

        for (const auto& decoded_item : decoded) {
            int token = std::get<0>(decoded_item);
            float prob = std::get<1>(decoded_item);
            int index = std::get<2>(decoded_item);
            if (token > 0 && token <= (int)engine->vocab.size()) {
                text += engine->vocab[token - 1];
                float pos = (index + 0.5f) / textline.width * roi.width;
                anchors.push_back(pos);
            } else if (!text.empty() && text.back() != ' ') {
                text += ' ';
                float pos = (index + 0.5f) / textline.width * roi.width;
                anchors.push_back(pos);
            }
        }
        results.push_back({text, anchors});
    }

    auto rec_t3 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(rec_t3 - rec_t0).count();
    fprintf(stderr, "[PROFILE] recognize: total=%.2f ms, warp=%.2f ms (max_single=%.2f), rec_forward=%.2f ms, decode=%.2f ms, boxes=%zu\n",
            total_ms, total_warp_ms, max_warp_ms, total_rec_ms, total_decode_ms, valid_boxes);

    return results;
}

static std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>> liteocr_ocr_engine_run(liteocr_ocr_engine* engine, const liteocr_image &input_)
{
    if (input_.empty()) {
        return {{}, {}};
    }

    liteocr_image input;
    if (input_.channels == 1) {
        input = liteocr_image(input_.width, input_.height, liteocr_image_type::LITEOCR_IMAGE_U8C3);
        liteocr_cvt_color(input_.data, input_.width, input_.height, input_.stride,
                          LITEOCR_FORMAT_GRAY, input.data, input.stride, LITEOCR_FORMAT_BGR);
    } else if (input_.channels == 4) {
        input = liteocr_image(input_.width, input_.height, liteocr_image_type::LITEOCR_IMAGE_U8C3);
        liteocr_cvt_color(input_.data, input_.width, input_.height, input_.stride,
                          LITEOCR_FORMAT_BGRA, input.data, input.stride, LITEOCR_FORMAT_BGR);
    } else {
        input = input_;
    }
    
    auto run_t0 = std::chrono::high_resolution_clock::now();
    auto textBoxes = liteocr_ocr_engine_detect(engine, input);
    auto run_t1 = std::chrono::high_resolution_clock::now();
    auto getTopLeft = [](const liteocr_text_box& box) -> liteocr_point2f {
        float _cx = 0, _cy = 0;
        for (int i = 0; i < 4; ++i) {
            _cx += box.points[i * 2 + 0];
            _cy += box.points[i * 2 + 1];
        }
        _cx /= 4.0f;
        _cy /= 4.0f;
        float _w = std::sqrt(
            (box.points[2] - box.points[0]) * (box.points[2] - box.points[0]) +
            (box.points[3] - box.points[1]) * (box.points[3] - box.points[1])
        );
        float _h = std::sqrt(
            (box.points[6] - box.points[0]) * (box.points[6] - box.points[0]) +
            (box.points[7] - box.points[1]) * (box.points[7] - box.points[1])
        );
        float _angle = std::atan2(box.points[3] - box.points[1], box.points[2] - box.points[0]) * 180.0f / 3.14159265f;
        auto pts = liteocr_get_rotated_rect_points({
            {_cx, _cy},
            {_w, _h},
            _angle
        });
        auto ordered = orderBoxPoints(pts);
        return ordered[0];
    };
    std::vector<std::pair<liteocr_point2f, liteocr_text_box>> sortedBoxes;
    sortedBoxes.reserve(textBoxes.size());
    for (auto& box : textBoxes) {
        sortedBoxes.push_back({getTopLeft(box), box});
    }
    std::sort(sortedBoxes.begin(), sortedBoxes.end(), [](const std::pair<liteocr_point2f, liteocr_text_box>& a, const std::pair<liteocr_point2f, liteocr_text_box>& b) {
        return (a.first.y < b.first.y) || (a.first.y == b.first.y && a.first.x < b.first.x);
    });
    for (size_t i = 0; i + 1 < sortedBoxes.size(); ++i) {
        for (int j = (int)i; j >= 0; --j) {
            if (std::abs(sortedBoxes[j + 1].first.y - sortedBoxes[j].first.y) < 10.0f &&
                sortedBoxes[j + 1].first.x < sortedBoxes[j].first.x) {
                std::swap(sortedBoxes[j], sortedBoxes[j + 1]);
            } else {
                break;
            }
        }
    }
    textBoxes.clear();
    for (auto& p : sortedBoxes) {
        textBoxes.push_back(std::move(p.second));
    }

    auto run_t2 = std::chrono::high_resolution_clock::now();
    auto textlines = liteocr_ocr_engine_recognize_internal(engine, input, textBoxes);
    auto run_t3 = std::chrono::high_resolution_clock::now();
    double sort_ms = std::chrono::duration<double, std::milli>(run_t2 - run_t1).count();
    double total_ms = std::chrono::duration<double, std::milli>(run_t3 - run_t0).count();
    fprintf(stderr, "[PROFILE] pipeline total=%.2f ms (detect+recognize), sort=%.2f ms\n", total_ms, sort_ms);
    return {textBoxes, textlines};
}

std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>> liteocr_ocr_engine_recognize(liteocr_ocr_engine* engine, const liteocr_image& img) {
    return liteocr_ocr_engine_run(engine, img);
}

std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>> liteocr_ocr_engine_recognize_raw(liteocr_ocr_engine* engine, const unsigned char* imgData, int width, int height, int channels, int cstep) {
    liteocr_image_type t = (channels == 1) ? liteocr_image_type::LITEOCR_IMAGE_U8C1 :
                           (channels == 3) ? liteocr_image_type::LITEOCR_IMAGE_U8C3 : liteocr_image_type::LITEOCR_IMAGE_U8C4;
    liteocr_image img(width, height, t, const_cast<unsigned char*>(imgData), cstep);
    return liteocr_ocr_engine_run(engine, img);
}

std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>> liteocr_ocr_engine_recognize_buffer(liteocr_ocr_engine* engine, const unsigned char* imgData, int size) {
    liteocr_image img = loadImageFromBuffer(imgData, size);
    return liteocr_ocr_engine_run(engine, img);
}

/* ============================================================================
 *  liteocr_table_engine
 * ============================================================================ */

void liteocr_table_engine_init(liteocr_table_engine* engine)
{
}

bool liteocr_table_engine_load_model(liteocr_table_engine* engine,
    const char* cnnParamPath, const char* cnnBinPath,
    const char* slaheadParamPath, const char* slaheadBinPath,
    const char* vocabPath,
    const liteocr_infer_option &opt) {
    return liteocr_slanet_load_model(&engine->slanet, cnnParamPath, cnnBinPath, slaheadParamPath, slaheadBinPath, vocabPath, opt);
}

bool liteocr_table_engine_load_model_from_buffer(liteocr_table_engine* engine,
    const char* cnnParamBuffer, const unsigned char* cnnBinBuffer,
    const char* slaheadParamBuffer, const unsigned char* slaheadBinBuffer,
    const char* vocabBuffer,
    const liteocr_infer_option &opt) {
    return liteocr_slanet_load_model_from_buffer(&engine->slanet, cnnParamBuffer, cnnBinBuffer, slaheadParamBuffer, slaheadBinBuffer, vocabBuffer, opt);
}

static std::pair<std::string, std::vector<liteocr_rect>> liteocr_table_engine_run(liteocr_table_engine* engine, const liteocr_image &input, const std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>>& ocrResult) {
    auto table_structure = liteocr_slanet_forward(&engine->slanet, input);
    return liteocr_merge_table_ocr(table_structure, ocrResult.first, ocrResult.second);
}

std::pair<std::string, std::vector<liteocr_rect>> liteocr_table_engine_recognize(liteocr_table_engine* engine, const liteocr_image& img, const std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>>& ocrResult) {
    return liteocr_table_engine_run(engine, img, ocrResult);
}

std::pair<std::string, std::vector<liteocr_rect>> liteocr_table_engine_recognize_raw(liteocr_table_engine* engine, const unsigned char* imgData, int width, int height, int channels, int cstep, const std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>>& ocrResult) {
    liteocr_image_type t = (channels == 1) ? liteocr_image_type::LITEOCR_IMAGE_U8C1 :
                           (channels == 3) ? liteocr_image_type::LITEOCR_IMAGE_U8C3 : liteocr_image_type::LITEOCR_IMAGE_U8C4;
    liteocr_image img(width, height, t, const_cast<unsigned char*>(imgData), cstep);
    return liteocr_table_engine_run(engine, img, ocrResult);
}

std::pair<std::string, std::vector<liteocr_rect>> liteocr_table_engine_recognize_buffer(liteocr_table_engine* engine, const unsigned char* imgData, int size, const std::pair<std::vector<liteocr_text_box>, std::vector<liteocr_text_line>>& ocrResult) {
    liteocr_image img = loadImageFromBuffer(imgData, size);
    return liteocr_table_engine_run(engine, img, ocrResult);
}


