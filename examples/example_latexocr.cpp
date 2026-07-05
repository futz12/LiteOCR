#include <iostream>
#include <cstring>
#include <string>
#include "liteocr.h"

// 解析 --key=value 形式的命令行参数；找不到则返回 default_val
static const char* parse_arg(int argc, char** argv, const char* key, const char* default_val) {
    size_t key_len = std::strlen(key);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], key, key_len) == 0 && argv[i][key_len] == '=') {
            return argv[i] + key_len + 1;
        }
    }
    return default_val;
}

int main(int argc, char** argv) {
    std::cout << "LiteOCR LatexOCR Example (PP-FormulaNet)" << std::endl;

    const char* encoder_param = parse_arg(argc, argv, "--encoder", "./models/PP-FormulaNet_plus_S_encoder.param");
    const char* encoder_bin   = parse_arg(argc, argv, "--encoder-bin", "./models/PP-FormulaNet_plus_S_encoder.bin");
    const char* embed_param   = parse_arg(argc, argv, "--embed", "./models/PP-FormulaNet_plus_S_embed.param");
    const char* embed_bin     = parse_arg(argc, argv, "--embed-bin", "./models/PP-FormulaNet_plus_S_embed.bin");
    const char* decoder_param = parse_arg(argc, argv, "--decoder", "./models/PP-FormulaNet_plus_S_decoder.param");
    const char* decoder_bin   = parse_arg(argc, argv, "--decoder-bin", "./models/PP-FormulaNet_plus_S_decoder.bin");
    const char* vocab_path    = parse_arg(argc, argv, "--vocab", "./models/PP-FormulaNet_plus_S_vocab.txt");
    const char* image_path    = parse_arg(argc, argv, "--image", "./test_formula.png");

    // 加载图像
    liteocr_image_t input = liteocr_imread(image_path, 3);
    if (!input.data) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "Loaded image: " << image_path
              << " (" << input.width << "x" << input.height << ", " << input.channels << " channels)" << std::endl;

    // 创建 LatexOCR 句柄
    liteocr_latexocr_t latex = liteocr_latexocr_create();
    if (!latex) {
        std::cerr << "Failed to create latexocr handle" << std::endl;
        liteocr_free_image(&input);
        return -1;
    }

    // 加载模型
    liteocr_latexocr_model_paths_t paths = {};
    paths.encoder_param = encoder_param;
    paths.encoder_bin = encoder_bin;
    paths.embed_param = embed_param;
    paths.embed_bin = embed_bin;
    paths.decoder_param = decoder_param;
    paths.decoder_bin = decoder_bin;
    paths.vocab = vocab_path;

    liteocr_infer_option_t opt = {};
    if (liteocr_latexocr_load_model(latex, &paths, &opt) != 0) {
        std::cerr << "Failed to load latexocr model" << std::endl;
        liteocr_latexocr_destroy(latex);
        liteocr_free_image(&input);
        return -1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // 识别
    char* result = liteocr_latexocr_recognize_image(latex, &input);
    if (!result) {
        std::cerr << "Recognition failed" << std::endl;
        liteocr_latexocr_destroy(latex);
        liteocr_free_image(&input);
        return -1;
    }

    std::cout << "================ LaTeX Result ================" << std::endl;
    std::cout << result << std::endl;
    std::cout << "==============================================" << std::endl;

    liteocr_free_string(result);
    liteocr_latexocr_destroy(latex);
    liteocr_free_image(&input);
    return 0;
}
