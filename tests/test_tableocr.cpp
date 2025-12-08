#include "LiteOCREngine.h"
#include <iostream>
#include <vector>
#include <fstream>
int main() {
    const char* inputfile = "table.jpg";
    LiteOCR::LiteOCREngine engine;
    engine.loadModel(
        "./models/PP-OCRv5_mobile_det.param",
        "./models/PP-OCRv5_mobile_det.bin",
        "./models/PP-OCRv5_mobile_rec.param",
        "./models/PP-OCRv5_mobile_rec.bin",
        "./models/PP-OCRv5_vocab.txt",
        "./models/PP-LCNet_x0_25_textline_ori.param",
        "./models/PP-LCNet_x0_25_textline_ori.bin"
    );
    
    std::vector<unsigned char> imgData;
    auto ifs = std::ifstream(inputfile, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open image file: " << inputfile << std::endl;
        return -1;
    }

    ifs.seekg(0, std::ios::end);
    size_t fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    imgData.resize(fileSize);
    ifs.read(reinterpret_cast<char*>(imgData.data()), fileSize);
    ifs.close();

    auto result = engine.recognize(imgData.data(), imgData.size());
    const auto& textBoxes = result.first;
    const auto& textlines = result.second;

    std::cout << "Detected " << textBoxes.size() << " text boxes." << std::endl;
    for (size_t i = 0; i < textBoxes.size(); i++) {
        const auto& box = textBoxes[i];
        std::cout << "Box " << i << ": center(" << box.box.center.x << ", " << box.box.center.y << "), "
                  << "size(" << box.box.size.width << "x" << box.box.size.height << "), "
                  << "angle: " << box.box.angle << ", "
                  << "isVertical: " << box.isVertical << ", "
                  << "score: " << box.score << std::endl;
        std::cout << "Recognized Text: " << textlines[i].text << std::endl;
    }

    LiteOCR::LiteOCRTableEngine tableEngine;
    tableEngine.loadModel(
        "./models/PP-StructrureV2_SLANet_plus_cnn.param",
        "./models/PP-StructrureV2_SLANet_plus_cnn.bin",
        "./models/PP-StructrureV2_SLANet_plus_slahead.param",
        "./models/PP-StructrureV2_SLANet_plus_slahead.bin",
        "./models/table_structure_dict_ch.txt",
        LiteOCR::InferOption()
    );

    auto tableResult = tableEngine.recognize(imgData.data(), imgData.size(), result);
    const auto& html = tableResult.first;

    std::cout << "Generated HTML Table:" << std::endl;
    std::cout << html << std::endl;

    // draw cell rectangles
    for (const auto& rect : tableResult.second) {
        std::cout << "Cell Rect: x=" << rect.x << ", y=" << rect.y
                  << ", width=" << rect.width << ", height=" << rect.height << std::endl;
    }

    
    return 0;
}