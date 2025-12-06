#include "LiteOCREngine.h"
#include <iostream>
#include <vector>
#include <fstream>
int main() {
    const char* inputfile = "test2.png";
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

    
    return 0;
}