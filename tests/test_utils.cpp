#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cstdio>
#include "liteocr.h"

static bool nearEqual(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) < eps;
}

static bool nearEqual(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}

static void testThreshold() {
    float src[] = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
    unsigned char dst[5];
    liteocr_threshold(src, 5, 1, 5 * (int)sizeof(float), dst, 5, 0.4f, 255);

    assert(dst[0] == 0);
    assert(dst[1] == 0);
    assert(dst[2] == 255);
    assert(dst[3] == 255);
    assert(dst[4] == 255);
    std::cout << "[PASS] threshold\n";
}

static void testThresholdMultipleRows() {
    float src[] = {
        0.1f, 0.6f,
        0.4f, 0.8f,
        0.9f, 0.2f
    };
    unsigned char dst[6];
    liteocr_threshold(src, 2, 3, 2 * (int)sizeof(float), dst, 2, 0.4f, 255);

    assert(dst[0] == 0);
    assert(dst[1] == 255);
    assert(dst[2] == 0);
    assert(dst[3] == 255);
    assert(dst[4] == 255);
    assert(dst[5] == 0);
    std::cout << "[PASS] thresholdMultipleRows\n";
}

static void testCopyMakeBorder() {
    unsigned char src[] = {1, 2, 3, 4};
    unsigned char dst[36];
    memset(dst, 0, sizeof(dst));
    liteocr_copy_make_border(src, 2, 2, 2, 1, dst, 4, 4, 4, 1, 1, 1, 1, 0);

    assert(dst[0] == 0);
    assert(dst[1] == 0);
    assert(dst[2] == 0);
    assert(dst[3] == 0);
    assert(dst[5] == 1);
    assert(dst[6] == 2);
    assert(dst[7] == 0);
    assert(dst[9] == 3);
    assert(dst[10] == 4);
    assert(dst[11] == 0);
    assert(dst[12] == 0);
    assert(dst[13] == 0);
    assert(dst[14] == 0);
    assert(dst[15] == 0);
    std::cout << "[PASS] copyMakeBorder\n";
}

static void testCvtColorGrayToBGR() {
    unsigned char src[] = {100, 200};
    unsigned char dst[6];
    liteocr_cvt_color(src, 2, 1, 2, 1, dst, 6, 3);

    assert(dst[0] == 100);
    assert(dst[1] == 100);
    assert(dst[2] == 100);
    assert(dst[3] == 200);
    assert(dst[4] == 200);
    assert(dst[5] == 200);
    std::cout << "[PASS] cvtColorGrayToBGR\n";
}

static void testCvtColorBGRToGray() {
    unsigned char src[] = {100, 150, 200};
    unsigned char dst[1];
    liteocr_cvt_color(src, 1, 1, 3, 3, dst, 1, 1);

    // Expected: 0.114*100 + 0.587*150 + 0.299*200 = 11.4 + 88.05 + 59.8 = 159.25
    assert(std::abs((int)dst[0] - 159) <= 1);
    std::cout << "[PASS] cvtColorBGRToGray\n";
}

static void testWarpPerspectiveIdentity() {
    unsigned char src[] = {
        10, 20, 30,
        40, 50, 60
    };
    unsigned char dst[6];

    float M[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    liteocr_warp_perspective(src, 3, 2, 3, 1, dst, 3, 2, 3, M);

    assert(dst[0] == 10);
    assert(dst[1] == 20);
    assert(dst[2] == 30);
    assert(dst[3] == 40);
    assert(dst[4] == 50);
    assert(dst[5] == 60);
    std::cout << "[PASS] warpPerspectiveIdentity\n";
}

static void testMeanMasked() {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    unsigned char mask[] = {1, 0, 1, 1};
    double mean = liteocr_mean_masked(src, 2, 2, 2 * (int)sizeof(float), mask, 2);

    assert(nearEqual(mean, 8.0 / 3.0));
    std::cout << "[PASS] meanMasked\n";
}

static void testImageReadWrite() {
    unsigned char data[] = {10, 20, 30, 40, 50, 60};
    liteocr_image_t img = {data, 2, 1, 3, 6};

    const char* path = "test_rw_output.png";
    int ok = liteocr_imwrite(path, &img);
    assert(ok == 0);

    liteocr_image_t loaded = liteocr_imread(path, 3);
    assert(loaded.data != nullptr);
    assert(loaded.width == 2);
    assert(loaded.height == 1);
    assert(loaded.channels == 3);

    assert(std::abs((int)loaded.data[0] - 10) <= 2);
    assert(std::abs((int)loaded.data[3] - 40) <= 2);

    liteocr_free_image(&loaded);
    std::cout << "[PASS] imageReadWrite\n";
}

static void testImageReadBGRChannelOrder() {
    const char* path = "test_channel_order.ppm";
    unsigned char rgb[] = {
        255, 0, 0,
        0, 0, 255
    };
    {
        std::ofstream f(path, std::ios::binary);
        f << "P6\n2 1\n255\n";
        f.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
    }

    liteocr_image_t loaded = liteocr_imread(path, 3);
    assert(loaded.data != nullptr);
    assert(loaded.width == 2);
    assert(loaded.height == 1);
    assert(loaded.channels == 3);

    assert(loaded.data[0] == 0);
    assert(loaded.data[1] == 0);
    assert(loaded.data[2] == 255);
    assert(loaded.data[3] == 255);
    assert(loaded.data[4] == 0);
    assert(loaded.data[5] == 0);

    liteocr_free_image(&loaded);
    std::remove(path);
    std::cout << "[PASS] imageReadBGRChannelOrder\n";
}

static void testReadNonExistentFile() {
    liteocr_image_t img = liteocr_imread("nonexistent_file_12345.png", 3);
    assert(img.data == nullptr);
    assert(img.width == 0);
    assert(img.height == 0);
    std::cout << "[PASS] readNonExistentFile\n";
}

static char* my_strdup(const char* s) {
    size_t len = strlen(s) + 1;
    char* p = (char*)malloc(len);
    if (p) memcpy(p, s, len);
    return p;
}

static void testFreeFunctions() {
    liteocr_text_line_t* lines = (liteocr_text_line_t*)malloc(2 * sizeof(liteocr_text_line_t));
    lines[0].text = my_strdup("hello");
    lines[0].anchors = (float*)malloc(2 * sizeof(float));
    lines[0].anchor_count = 2;
    lines[1].text = my_strdup("world");
    lines[1].anchors = (float*)malloc(1 * sizeof(float));
    lines[1].anchor_count = 1;
    liteocr_free_text_lines(lines, 2);

    liteocr_text_box_t* boxes = (liteocr_text_box_t*)malloc(1 * sizeof(liteocr_text_box_t));
    boxes[0].points[0] = 1.0f;
    liteocr_free_text_boxes(boxes, 1);

    liteocr_contour_t* contour = (liteocr_contour_t*)malloc(1 * sizeof(liteocr_contour_t));
    contour[0].points = (liteocr_point_t*)malloc(4 * sizeof(liteocr_point_t));
    contour[0].point_count = 4;
    liteocr_free_contours(contour, 1);

    liteocr_table_cell_t* cells = (liteocr_table_cell_t*)malloc(1 * sizeof(liteocr_table_cell_t));
    cells[0].tag = my_strdup("<td>");
    liteocr_free_table_cells(cells, 1);

    std::cout << "[PASS] freeFunctions\n";
}

int main() {
    std::cout << "=== LiteOCR Utils Unit Tests ===" << std::endl;
    testThreshold();
    testThresholdMultipleRows();
    testCopyMakeBorder();
    testCvtColorGrayToBGR();
    testCvtColorBGRToGray();
    testWarpPerspectiveIdentity();
    testMeanMasked();
    testImageReadWrite();
    testImageReadBGRChannelOrder();
    testReadNonExistentFile();
    testFreeFunctions();
    std::cout << "=== All tests passed! ===" << std::endl;
    return 0;
}
