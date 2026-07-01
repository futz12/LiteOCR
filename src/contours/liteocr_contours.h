#pragma once

#include <vector>
#include <cstdint>
#include <cmath>

struct liteocr_point {
    int x;
    int y;
};

inline bool liteocr_point_eq(const liteocr_point& a, const liteocr_point& b) {
    return a.x == b.x && a.y == b.y;
}
inline bool liteocr_point_ne(const liteocr_point& a, const liteocr_point& b) {
    return !(liteocr_point_eq(a, b));
}

struct liteocr_point2f {
    float x;
    float y;
};

struct liteocr_size2f {
    float width;
    float height;
};

struct liteocr_intrect {
    int x;
    int y;
    int width;
    int height;
};

struct liteocr_rotated_rect {
    liteocr_point2f center;
    liteocr_size2f  size;
    float angle;
};

enum liteocr_contour_approx_mode {
    LITEOCR_CHAIN_APPROX_NONE = 1,
    LITEOCR_CHAIN_APPROX_SIMPLE = 2
};

void liteocr_find_contours(const uint8_t* data, int width, int height, int step,
                           std::vector<std::vector<liteocr_point>>& contours,
                           liteocr_contour_approx_mode approx = LITEOCR_CHAIN_APPROX_SIMPLE);

liteocr_rotated_rect liteocr_min_area_rect(const std::vector<liteocr_point>& contour);
liteocr_intrect liteocr_bounding_rect(const std::vector<liteocr_point>& contour);
double liteocr_contour_area(const std::vector<liteocr_point>& contour);
double liteocr_arc_length(const std::vector<liteocr_point>& contour, bool closed);

void liteocr_fill_poly(uint8_t* data, int width, int height, int step,
                       const std::vector<std::vector<liteocr_point>>& polygons,
                       uint8_t value);

std::vector<liteocr_point2f> liteocr_get_rotated_rect_points(const liteocr_rotated_rect& rr);
