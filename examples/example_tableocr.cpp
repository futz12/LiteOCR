#include "liteocr.h"
#include <iostream>
#include <vector>
#include <fstream>

int main() {
    const char* inputfile = "table.jpg";
    liteocr_engine_t engine = liteocr_engine_create();
    
    liteocr_ocr_model_paths_t paths = {
        "./models/PP-OCRv5_mobile_det.param",
        "./models/PP-OCRv5_mobile_det.bin",
        "./models/PP-OCRv5_mobile_rec.param",
        "./models/PP-OCRv5_mobile_rec.bin",
        "./models/PP-OCRv5_vocab.txt",
        "./models/PP-LCNet_x1_0_textline_ori.param",
        "./models/PP-LCNet_x1_0_textline_ori.bin"
    };
    
    liteocr_engine_load_model(engine, &paths, nullptr);
    
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
    
    liteocr_text_box_t* boxes = nullptr;
    int box_count = 0;
    liteocr_text_line_t* lines = nullptr;
    int line_count = 0;
    
    liteocr_engine_recognize_buffer(engine, imgData.data(), (int)imgData.size(),
        &boxes, &box_count, &lines, &line_count);
    
    std::cout << "LiteOCR Table OCR Example" << std::endl;

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
    
    liteocr_table_engine_t tableEngine = liteocr_table_engine_create();
    liteocr_table_model_paths_t table_paths = {
        "./models/PP-StructrureV2_SLANet_plus_cnn.param",
        "./models/PP-StructrureV2_SLANet_plus_cnn.bin",
        "./models/PP-StructrureV2_SLANet_plus_slahead.param",
        "./models/PP-StructrureV2_SLANet_plus_slahead.bin",
        "./models/table_structure_dict_ch.txt"
    };
    
    liteocr_table_engine_load_model(tableEngine, &table_paths, nullptr);
    
    char* html = nullptr;
    liteocr_rect_t* cells = nullptr;
    int cell_count = 0;
    liteocr_table_cell_t* structure = nullptr;
    int structure_count = 0;
    
    liteocr_table_engine_recognize_buffer(tableEngine, imgData.data(), (int)imgData.size(),
        boxes, box_count, lines, line_count,
        &html, &cells, &cell_count, &structure, &structure_count);
    
    std::cout << "Generated HTML Table:" << std::endl;
    std::cout << html << std::endl;
    
    for (int i = 0; i < cell_count; i++) {
        std::cout << "Cell Rect: x=" << cells[i].x << ", y=" << cells[i].y
                  << ", width=" << cells[i].width << ", height=" << cells[i].height << std::endl;
    }
    
    liteocr_free_string(html);
    liteocr_free(cells);
    liteocr_free_table_cells(structure, structure_count);
    liteocr_free_text_boxes(boxes, box_count);
    liteocr_free_text_lines(lines, line_count);
    liteocr_engine_destroy(engine);
    liteocr_table_engine_destroy(tableEngine);
    return 0;
}
