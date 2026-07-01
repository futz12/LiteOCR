#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "liteocr.h"

static bool nearEqual(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) < eps;
}

void testBoundingRect() {
    liteocr_point_t pts[] = {{0,0}, {10,0}, {10,5}, {0,5}};
    auto r = liteocr_bounding_rect(pts, 4);
    assert(r.x == 0);
    assert(r.y == 0);
    assert(r.width == 11);
    assert(r.height == 6);
    std::cout << "[PASS] boundingRect\n";
}

void testContourArea() {
    liteocr_point_t pts[] = {{0,0}, {10,0}, {10,10}, {0,10}};
    double area = liteocr_contour_area(pts, 4);
    assert(std::fabs(area - 100.0) < 1e-3);
    std::cout << "[PASS] contourArea\n";
}

void testArcLength() {
    liteocr_point_t pts[] = {{0,0}, {10,0}, {10,10}, {0,10}};
    double len = liteocr_arc_length(pts, 4, 1);
    assert(std::fabs(len - 40.0) < 1e-3);
    std::cout << "[PASS] arcLength\n";
}

void testFindContoursSquare() {
    int w = 10, h = 10;
    std::vector<uint8_t> img(w * h, 0);
    for (int y = 2; y < 6; ++y)
        for (int x = 2; x < 6; ++x)
            img[y * w + x] = 255;

    liteocr_contour_t* contours = nullptr;
    int contour_count = 0;
    liteocr_find_contours(img.data(), w, h, w, &contours, &contour_count, 2);
    assert(contour_count == 1);
    assert(contours[0].point_count == 4);
    std::cout << "[PASS] findContoursSquare (" << contours[0].point_count << " points)\n";
    liteocr_free_contours(contours, contour_count);
}

void testFindContoursDonut() {
    int w = 10, h = 10;
    std::vector<uint8_t> img(w * h, 0);
    for (int y = 2; y < 7; ++y) {
        for (int x = 2; x < 7; ++x) {
            if (x >= 4 && x <= 5 && y >= 4 && y <= 5)
                img[y * w + x] = 0;
            else
                img[y * w + x] = 255;
        }
    }

    liteocr_contour_t* contours = nullptr;
    int contour_count = 0;
    liteocr_find_contours(img.data(), w, h, w, &contours, &contour_count, 2);
    assert(contour_count == 2);
    std::cout << "[PASS] findContoursDonut (" << contour_count << " contours)\n";
    liteocr_free_contours(contours, contour_count);
}

void testFindContoursApproxModes() {
    int w = 10, h = 10;
    std::vector<uint8_t> img(w * h, 0);
    for (int y = 2; y < 6; ++y)
        for (int x = 2; x < 6; ++x)
            img[y * w + x] = 255;

    {
        liteocr_contour_t* contours = nullptr;
        int contour_count = 0;
        liteocr_find_contours(img.data(), w, h, w, &contours, &contour_count, 1);
        assert(contour_count == 1);
        assert(contours[0].point_count >= 12 && contours[0].point_count <= 20);
        std::cout << "[PASS] findContours CHAIN_APPROX_NONE (" << contours[0].point_count << " points)\n";
        liteocr_free_contours(contours, contour_count);
    }

    {
        liteocr_contour_t* contours = nullptr;
        int contour_count = 0;
        liteocr_find_contours(img.data(), w, h, w, &contours, &contour_count, 2);
        assert(contour_count == 1);
        assert(contours[0].point_count == 4);
        std::cout << "[PASS] findContours CHAIN_APPROX_SIMPLE (" << contours[0].point_count << " points)\n";
        liteocr_free_contours(contours, contour_count);
    }
}

void testFillPoly() {
    int w = 10, h = 10;
    std::vector<uint8_t> img(w * h, 0);
    liteocr_point_t poly_pts[] = {{2,2}, {7,2}, {7,7}, {2,7}};
    liteocr_contour_t poly = {poly_pts, 4};
    liteocr_fill_poly(img.data(), w, h, w, &poly, 1, 255);

    assert(img[3 * w + 3] == 255);
    assert(img[3 * w + 6] == 255);
    assert(img[6 * w + 3] == 255);
    assert(img[0] == 0);
    assert(img[9 * w + 9] == 0);
    std::cout << "[PASS] fillPoly\n";
}

void testMinAreaRect() {
    liteocr_point_t pts[] = {{0,0}, {10,0}, {10,5}, {0,5}};
    liteocr_point2f_t out_pts[4];
    liteocr_min_area_rect(pts, 4, out_pts);
    
    // Calculate center from 4 points
    float cx = 0, cy = 0;
    for (int i = 0; i < 4; ++i) {
        cx += out_pts[i].x;
        cy += out_pts[i].y;
    }
    cx /= 4.0f;
    cy /= 4.0f;
    
    assert(nearEqual(cx, 5.0f));
    assert(nearEqual(cy, 2.5f));
    std::cout << "[PASS] minAreaRect\n";
}

void testMinAreaRectRotated() {
    liteocr_point_t pts[] = {{5,0}, {10,5}, {5,10}, {0,5}};
    liteocr_point2f_t out_pts[4];
    liteocr_min_area_rect(pts, 4, out_pts);
    
    float cx = 0, cy = 0;
    for (int i = 0; i < 4; ++i) {
        cx += out_pts[i].x;
        cy += out_pts[i].y;
    }
    cx /= 4.0f;
    cy /= 4.0f;
    
    assert(nearEqual(cx, 5.0f));
    assert(nearEqual(cy, 5.0f));
    std::cout << "[PASS] minAreaRectRotated\n";
}

int main() {
    std::cout << "=== LiteOCR Contours Unit Tests ===" << std::endl;
    testBoundingRect();
    testContourArea();
    testArcLength();
    testFindContoursSquare();
    testFindContoursDonut();
    testFindContoursApproxModes();
    testFillPoly();
    testMinAreaRect();
    testMinAreaRectRotated();
    std::cout << "=== All tests passed! ===" << std::endl;
    return 0;
}
