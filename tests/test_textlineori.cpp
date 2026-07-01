#include <iostream>
#include <cstring>
#include <cstdlib>
#include "liteocr.h"

struct TestConfig {
    const char* param = "./models/PP-LCNet_x1_0_textline_ori.param";
    const char* bin = "./models/PP-LCNet_x1_0_textline_ori.bin";
    const char* image = "test_line.png";
    int gpu_device_id = -1;
    bool use_fp16 = false;
    bool use_int8 = false;
    bool use_bf16 = false;
    bool use_anglenet = false;
};

static bool parse_args(int argc, char** argv, TestConfig& cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((strcmp(arg, "-p") == 0 || strcmp(arg, "--param") == 0) && i + 1 < argc) {
            cfg.param = argv[++i];
        } else if ((strcmp(arg, "-b") == 0 || strcmp(arg, "--bin") == 0) && i + 1 < argc) {
            cfg.bin = argv[++i];
        } else if ((strcmp(arg, "-i") == 0 || strcmp(arg, "--image") == 0) && i + 1 < argc) {
            cfg.image = argv[++i];
        } else if ((strcmp(arg, "-g") == 0 || strcmp(arg, "--gpu") == 0) && i + 1 < argc) {
            cfg.gpu_device_id = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--fp16") == 0) {
            cfg.use_fp16 = true;
        } else if (strcmp(arg, "--int8") == 0) {
            cfg.use_int8 = true;
        } else if (strcmp(arg, "--bf16") == 0) {
            cfg.use_bf16 = true;
        } else if (strcmp(arg, "--anglenet") == 0) {
            cfg.use_anglenet = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    TestConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        std::cerr << "Usage: " << argv[0] << " [-p param] [-b bin] [-i image] [-g gpu] [--fp16] [--int8] [--bf16] [--anglenet]" << std::endl;
        return 2;
    }

    std::cout << "LiteOCR Textline Orientation Classifier Test" << std::endl;
    std::cout << "Model: " << cfg.param << std::endl;
    std::cout << "Image: " << cfg.image << std::endl;

    liteocr_image_t input = liteocr_imread(cfg.image, 3);
    if (!input.data) {
        std::cerr << "Missing test image: " << cfg.image << std::endl;
        return 77;
    }

    liteocr_textline_ori_t classifier = liteocr_textline_ori_create();
    liteocr_infer_option_t opt = {};
    opt.num_threads = 4;
    opt.gpu_device_id = cfg.gpu_device_id;
    opt.use_fp16 = cfg.use_fp16 ? 1 : 0;
    opt.use_int8 = cfg.use_int8 ? 1 : 0;
    opt.use_bf16 = cfg.use_bf16 ? 1 : 0;
    opt.textline_ori_model_type = cfg.use_anglenet ? LITEOCR_TEXTLINE_ORI_ANGLENET : LITEOCR_TEXTLINE_ORI_PADDLE;

    if (liteocr_textline_ori_load_model(classifier, cfg.param, cfg.bin, &opt) != 0) {
        std::cerr << "Failed to load textline orientation model." << std::endl;
        liteocr_free_image(&input);
        liteocr_textline_ori_destroy(classifier);
        return 1;
    }

    int orientation = liteocr_textline_ori_forward(classifier, &input);
    if (orientation < 0) {
        std::cerr << "Textline orientation forward failed." << std::endl;
        liteocr_free_image(&input);
        liteocr_textline_ori_destroy(classifier);
        return 1;
    }
    std::cout << "Predicted orientation: " << orientation << std::endl;

    liteocr_image_t rot = liteocr_image_t{nullptr, input.height, input.width, input.channels, input.channels * input.width};
    rot.data = (unsigned char*)malloc(rot.height * rot.stride);
    if (!rot.data) {
        std::cerr << "Failed to allocate rotated image." << std::endl;
        liteocr_free_image(&input);
        liteocr_textline_ori_destroy(classifier);
        return 1;
    }
    liteocr_rotate180(input.data, input.width, input.height, input.stride, input.channels,
                       rot.data, rot.stride);

    int orientation2 = liteocr_textline_ori_forward(classifier, &rot);
    if (orientation2 < 0) {
        std::cerr << "Textline orientation forward failed after rotation." << std::endl;
        free(rot.data);
        liteocr_free_image(&input);
        liteocr_textline_ori_destroy(classifier);
        return 1;
    }
    std::cout << "Predicted orientation after rotation: " << orientation2 << std::endl;

    int rc = 0;
    if (orientation != orientation2) {
        std::cout << "Orientation classifier works correctly." << std::endl;
    } else {
        std::cout << "Orientation classifier failed." << std::endl;
        rc = -1;
    }

    free(rot.data);
    liteocr_free_image(&input);
    liteocr_textline_ori_destroy(classifier);
    return rc;
}
