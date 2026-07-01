#include "liteocr.h"
#include <iostream>

int main()
{
    liteocr_image_t input = liteocr_imread("doc_test.jpg", 3);
    
    liteocr_uvdoc_t uvdoc = liteocr_uvdoc_create();
    liteocr_infer_option_t opt = {};
    liteocr_uvdoc_load_model(uvdoc, "./models/PP-UVDoc.param", "./models/PP-UVDoc.bin", &opt);
    
    std::cout << "LiteOCR UVDoc Example" << std::endl;

    liteocr_image_t output = liteocr_uvdoc_forward(uvdoc, &input);
    liteocr_imwrite("uvdoc_output.jpg", &output);
    
    liteocr_free_image(&output);
    liteocr_free_image(&input);
    liteocr_uvdoc_destroy(uvdoc);
    return 0;
}
