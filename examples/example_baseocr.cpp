#include "liteocr.h"
#include <iostream>
#include <vector>
#include <fstream>

int main() {
    const char* inputfile = "test2.png";
    liteocr_engine_t engine = liteocr_engine_create();
    
    liteocr_ocr_model_paths_t paths = {
        "./models/PP-OCRv6_small_det.param",
        "./models/PP-OCRv6_small_det.bin",
        "./models/PP-OCRv6_small_rec.param",
        "./models/PP-OCRv6_small_rec.bin",
        "./models/PP-OCRv6_vocab.txt",
        "./models/PP-LCNet_x1_0_textline_ori.param",
        "./models/PP-LCNet_x1_0_textline_ori.bin"
    };
    
    if (liteocr_engine_load_model(engine, &paths, nullptr) != 0) {
        std::cerr << "Failed to load model" << std::endl;
        liteocr_engine_destroy(engine);
        return -1;
    }
    
    std::vector<unsigned char> imgData;
    auto ifs = std::ifstream(inputfile, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open image file: " << inputfile << std::endl;
        liteocr_engine_destroy(engine);
        return -1;
    }
    
    ifs.seekg(0, std::ios::end);
    size_t fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    imgData.resize(fileSize);
    ifs.read(reinterpret_cast<char*>(imgData.data()), fileSize);
    ifs.close();
    
    liteocr_text_box_t* boxes = nullptr;
    int box_count = 0;
    liteocr_text_line_t* lines = nullptr;
    int line_count = 0;
    
    if (liteocr_engine_recognize_buffer(engine, imgData.data(), (int)imgData.size(),
        &boxes, &box_count, &lines, &line_count) != 0) {
        std::cerr << "Recognition failed" << std::endl;
        liteocr_engine_destroy(engine);
        return -1;
    }
    
    std::cout << "Detected " << box_count << " text boxes." << std::endl;
    for (int i = 0; i < box_count; i++) {
        std::cout << "Box " << i << ": points("
                  << boxes[i].points[0] << "," << boxes[i].points[1] << "; "
                  << boxes[i].points[2] << "," << boxes[i].points[3] << "; "
                  << boxes[i].points[4] << "," << boxes[i].points[5] << "; "
                  << boxes[i].points[6] << "," << boxes[i].points[7] << "), "
                  << "isVertical: " << boxes[i].is_vertical << ", "
                  << "score: " << boxes[i].score << std::endl;
        std::cout << "Recognized Text: " << lines[i].text << std::endl;
    }
    
    liteocr_free_text_boxes(boxes, box_count);
    liteocr_free_text_lines(lines, line_count);
    liteocr_engine_destroy(engine);
    return 0;
}
