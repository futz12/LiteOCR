#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "liteocr.h"

static std::vector<std::string> load_vocab(const char* path)
{
    std::vector<std::string> vocab;
    std::ifstream fs(path);
    if (!fs.is_open()) return vocab;
    std::string line;
    while (std::getline(fs, line)) {
        vocab.push_back(line);
    }
    return vocab;
}

int main(int argc, char** argv)
{
    std::cout << "LiteOCR Recognizer Example" << std::endl;

    const char* param_path = "./models/PP-OCRv5_mobile_rec.param";
    const char* bin_path = "./models/PP-OCRv5_mobile_rec.bin";
    const char* vocab_path = nullptr;
    if (argc >= 3)
    {
        param_path = argv[1];
        bin_path = argv[2];
    }
    if (argc >= 4)
    {
        vocab_path = argv[3];
    }

    liteocr_image_t input = liteocr_imread("test_line.png", 3);

    liteocr_recognizer_t recognizer = liteocr_recognizer_create();
    liteocr_infer_option_t opt = {};
    if (liteocr_recognizer_load_model(recognizer, param_path, bin_path, &opt) != 0) {
        std::cerr << "Failed to load model: " << param_path << ", " << bin_path << std::endl;
        liteocr_free_image(&input);
        liteocr_recognizer_destroy(recognizer);
        return -1;
    }

    liteocr_image_t output = liteocr_recognizer_forward(recognizer, &input);

    std::cout << "Output size: " << output.width << "x" << output.height << std::endl;

    int* tokens = nullptr;
    float* probs = nullptr;
    int* indices = nullptr;
    int count = 0;
    liteocr_ctc_decode(&output, 0, &tokens, &probs, &indices, &count);

    std::cout << "Decoded results: " << std::endl;
    for (int i = 0; i < count; i++) {
        std::cout << "Index: " << indices[i] << ", Token: " << tokens[i] << ", Prob: " << probs[i] << std::endl;
    }

    if (vocab_path)
    {
        std::vector<std::string> vocab = load_vocab(vocab_path);
        std::ostringstream oss;
        for (int i = 0; i < count; i++) {
            int idx = tokens[i] - 1;
            if (idx >= 0 && idx < (int)vocab.size()) {
                oss << vocab[idx];
            }
        }
        std::cout << "Decoded text: " << oss.str() << std::endl;
    }

    liteocr_free(tokens);
    liteocr_free(probs);
    liteocr_free(indices);
    liteocr_free_image(&output);
    liteocr_free_image(&input);
    liteocr_recognizer_destroy(recognizer);

    return 0;
}
