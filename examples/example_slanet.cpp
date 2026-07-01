#include "liteocr.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>

static void drawLine(liteocr_image_t img, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b, int thickness)
{
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    int half = thickness / 2;

    while (true) {
        for (int ty = -half; ty <= half; ++ty) {
            for (int tx = -half; tx <= half; ++tx) {
                int px = x0 + tx;
                int py = y0 + ty;
                if (px >= 0 && px < img.width && py >= 0 && py < img.height) {
                    uint8_t* p = img.data + py * img.stride + px * 3;
                    p[0] = b; p[1] = g; p[2] = r;
                }
            }
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

int main()
{
    std::cout << "LiteOCR SLANet Example" << std::endl;

    liteocr_slanet_t infer = liteocr_slanet_create();
    liteocr_infer_option_t opt = {};
    liteocr_slanet_load_model(infer,
        "./models/PP-StructrureV2_SLANet_plus_cnn.param",
        "./models/PP-StructrureV2_SLANet_plus_cnn.bin",
        "./models/PP-StructrureV2_SLANet_plus_slahead.param",
        "./models/PP-StructrureV2_SLANet_plus_slahead.bin",
        "./models/table_structure_dict_ch.txt", &opt);

    liteocr_image_t input = liteocr_imread("./table.jpg", 3);

    liteocr_table_cell_t* cells = nullptr;
    int cell_count = 0;
    liteocr_slanet_forward(infer, &input, &cells, &cell_count);

    for (int i = 0; i < cell_count; i++)
    {
        std::cout << cells[i].tag;

        if (strcmp(cells[i].tag, "<td>") != 0 && strncmp(cells[i].tag, "<td", 3) != 0 && strcmp(cells[i].tag, "<td></td>") != 0)
            continue;
        // Draw box
        for (int j = 0; j < 4; j++)
        {
            int x0 = (int)cells[i].box[j * 2];
            int y0 = (int)cells[i].box[j * 2 + 1];
            int x1 = (int)cells[i].box[((j + 1) % 4) * 2];
            int y1 = (int)cells[i].box[((j + 1) % 4) * 2 + 1];
            drawLine(input, x0, y0, x1, y1, 0, 255, 0, 2);
        }
    }
    liteocr_imwrite("table_result.jpg", &input);
    std::cout << "Result saved to table_result.jpg" << std::endl;

    liteocr_free_table_cells(cells, cell_count);
    liteocr_free_image(&input);
    liteocr_slanet_destroy(infer);

    return 0;
}
