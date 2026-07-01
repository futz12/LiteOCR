#include <iostream>
#include <vector>
#include "liteocr.h"

int main() {
    std::cout << "LiteOCR Detector Example" << std::endl;

    liteocr_image_t input = liteocr_imread("test.png", 3);
    if (!input.data) {
        std::cerr << "Failed to load test.png" << std::endl;
        return 1;
    }

    std::cout << "Image size: " << input.width << "x" << input.height << std::endl;

    liteocr_detector_t detector = liteocr_detector_create();
    liteocr_infer_option_t opt = {};
    liteocr_detector_load_model(detector, "./models/PP-OCRv5_mobile_det.param", "./models/PP-OCRv5_mobile_det.bin", &opt);
    
    liteocr_image_t output = liteocr_detector_forward(detector, &input);
    std::cout << "Output size: " << output.width << "x" << output.height << std::endl;

    const float threshold = 0.3f;
    const float box_threshold = 0.6f;
    const int max_candidates = 1000;
    const float unclip_ratio = 1.95f;

    int w = output.width;
    int h = output.height;
    std::vector<unsigned char> binary_map(w * h);
    liteocr_threshold((const float*)output.data, w, h, output.stride,
                      binary_map.data(), w,
                      threshold, 255);

    liteocr_contour_t* contours = nullptr;
    int contour_count = 0;
    liteocr_find_contours(binary_map.data(), w, h, w,
                          &contours, &contour_count, 2);

    if (contour_count > max_candidates) {
        contour_count = max_candidates;
    }

    for (int i = 0; i < contour_count; i++) {
        if (contours[i].point_count <= 2) {
            continue;
        }

        liteocr_point2f_t pts[4];
        liteocr_min_area_rect(contours[i].points, contours[i].point_count, pts);

        // Draw box logic omitted (same as original)
        // The original test didn't actually draw boxes in the shown code
    }

    liteocr_free_contours(contours, contour_count);
    liteocr_free_image(&output);
    liteocr_free_image(&input);
    liteocr_detector_destroy(detector);

    return 0;
}
