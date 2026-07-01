#include "liteocr_contours.h"
#include <algorithm>
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



// ------------------------------------------------------------------
// liteocr_bounding_rect
// ------------------------------------------------------------------
liteocr_intrect liteocr_bounding_rect(const std::vector<liteocr_point>& contour)
{
    if (contour.empty())
        return {0, 0, 0, 0};

    int xMin = contour[0].x, xMax = contour[0].x;
    int yMin = contour[0].y, yMax = contour[0].y;

    for (size_t i = 1; i < contour.size(); ++i) {
        xMin = std::min(xMin, contour[i].x);
        xMax = std::max(xMax, contour[i].x);
        yMin = std::min(yMin, contour[i].y);
        yMax = std::max(yMax, contour[i].y);
    }

    return {xMin, yMin, xMax - xMin + 1, yMax - yMin + 1};
}

// ------------------------------------------------------------------
// liteocr_contour_area (shoelace formula)
// ------------------------------------------------------------------
double liteocr_contour_area(const std::vector<liteocr_point>& contour)
{
    size_t n = contour.size();
    if (n < 3)
        return 0.0;

    double a = 0.0;
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        a += (double)contour[i].x * contour[j].y - (double)contour[j].x * contour[i].y;
    }
    return a * 0.5;
}

// ------------------------------------------------------------------
// liteocr_arc_length
// ------------------------------------------------------------------
double liteocr_arc_length(const std::vector<liteocr_point>& contour, bool closed)
{
    size_t n = contour.size();
    if (n < 2)
        return 0.0;

    double len = 0.0;
    size_t limit = closed ? n : n - 1;
    for (size_t i = 0; i < limit; ++i) {
        const liteocr_point& p1 = contour[i];
        const liteocr_point& p2 = contour[(i + 1) % n];
        double dx = (double)(p1.x - p2.x);
        double dy = (double)(p1.y - p2.y);
        len += std::sqrt(dx * dx + dy * dy);
    }
    return len;
}

// ------------------------------------------------------------------
// Convex hull (Monotone chain / Andrew's algorithm)
// ------------------------------------------------------------------
namespace {

static std::vector<liteocr_point> convexHull(std::vector<liteocr_point> pts)
{
    size_t n = pts.size();
    if (n <= 1)
        return pts;

    std::sort(pts.begin(), pts.end(), [](const liteocr_point& a, const liteocr_point& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });

    std::vector<liteocr_point> hull;
    hull.reserve(n * 2);

    auto cross = [](const liteocr_point& O, const liteocr_point& A, const liteocr_point& B) -> long long {
        return (long long)(A.x - O.x) * (B.y - O.y) - (long long)(A.y - O.y) * (B.x - O.x);
    };

    // Lower hull
    for (const liteocr_point& p : pts) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0)
            hull.pop_back();
        hull.push_back(p);
    }

    // Upper hull
    size_t lowerSize = hull.size();
    for (int i = (int)n - 2; i >= 0; --i) {
        const liteocr_point& p = pts[i];
        while (hull.size() > lowerSize && cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0)
            hull.pop_back();
        hull.push_back(p);
    }

    // Last point == first point
    if (!hull.empty())
        hull.pop_back();

    // Remove possible duplicate from lower/upper overlap
    auto last = std::unique(hull.begin(), hull.end(), [](const liteocr_point& a, const liteocr_point& b) {
        return a.x == b.x && a.y == b.y;
    });
    hull.erase(last, hull.end());

    return hull;
}

} // anonymous namespace

// ------------------------------------------------------------------
// liteocr_min_area_rect (brute-force rotating calipers on convex hull)
// ------------------------------------------------------------------
liteocr_rotated_rect liteocr_min_area_rect(const std::vector<liteocr_point>& contour)
{
    if (contour.empty())
        return {{0.0f, 0.0f}, {0.0f, 0.0f}, 0.0f};
    if (contour.size() == 1)
        return {{(float)contour[0].x, (float)contour[0].y}, {0.0f, 0.0f}, 0.0f};

    std::vector<liteocr_point> hull = convexHull(contour);
    int n = (int)hull.size();

    if (n == 2) {
        float cx = (hull[0].x + hull[1].x) * 0.5f;
        float cy = (hull[0].y + hull[1].y) * 0.5f;
        float dx = (float)(hull[0].x - hull[1].x);
        float dy = (float)(hull[0].y - hull[1].y);
        float len = std::sqrt(dx * dx + dy * dy);
        float angle = 0.0f;
        if (std::abs(dx) < 1e-6f) {
            angle = -90.0f;
        } else {
            angle = (float)(std::atan2(dy, dx) * 180.0 / M_PI);
        }
        if (angle >= 0.0f)
            angle -= 180.0f;
        return {{cx, cy}, {0.0f, len}, angle};
    }

    float minArea = FLT_MAX;
    float bestCx = 0.0f, bestCy = 0.0f;
    float bestW = 0.0f, bestH = 0.0f;
    float bestAngle = 0.0f;

    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        float ex = (float)(hull[j].x - hull[i].x);
        float ey = (float)(hull[j].y - hull[i].y);
        float edgeLen = std::sqrt(ex * ex + ey * ey);
        if (edgeLen < 1e-6f)
            continue;

        // Unit edge vector and its perpendicular (rotated CCW).
        float ux = ex / edgeLen;
        float uy = ey / edgeLen;
        float vx = -uy;
        float vy =  ux;

        float minU = 0.0f, maxU = 0.0f;
        float minV = 0.0f, maxV = 0.0f;

        for (int k = 0; k < n; ++k) {
            float px = (float)(hull[k].x - hull[i].x);
            float py = (float)(hull[k].y - hull[i].y);
            float pu = px * ux + py * uy;
            float pv = px * vx + py * vy;
            if (k == 0) {
                minU = maxU = pu;
                minV = maxV = pv;
            } else {
                minU = std::min(minU, pu);
                maxU = std::max(maxU, pu);
                minV = std::min(minV, pv);
                maxV = std::max(maxV, pv);
            }
        }

        float width  = maxU - minU;
        float height = maxV - minV;
        float area   = width * height;

        if (area < minArea) {
            minArea = area;
            float cu = (minU + maxU) * 0.5f;
            float cv = (minV + maxV) * 0.5f;
            bestCx = (float)hull[i].x + cu * ux + cv * vx;
            bestCy = (float)hull[i].y + cu * uy + cv * vy;
            bestW  = width;
            bestH  = height;
            bestAngle = (float)(std::atan2(uy, ux) * 180.0 / M_PI);
        }
    }

    // Normalize to OpenCV convention: angle in [-90, 0), width = long side.
    if (bestAngle >= 90.0f)
        bestAngle -= 180.0f;
    if (bestAngle < -90.0f)
        bestAngle += 180.0f;
    if (bestAngle >= 0.0f) {
        std::swap(bestW, bestH);
        bestAngle -= 90.0f;
    }
    if (bestAngle < -90.0f)
        bestAngle += 180.0f;

    return {{bestCx, bestCy}, {bestW, bestH}, bestAngle};
}

// ------------------------------------------------------------------
// liteocr_fill_poly (scanline active edge table)
// ------------------------------------------------------------------
void liteocr_fill_poly(uint8_t* data, int width, int height, int step,
              const std::vector<std::vector<liteocr_point>>& polygons,
              uint8_t value)
{
    if (!data || width <= 0 || height <= 0)
        return;

    // Reusable intersection buffer to avoid per-row allocations
    std::vector<float> xs;

    for (const auto& poly : polygons) {
        int n = (int)poly.size();
        if (n < 3)
            continue;

        int yMin = poly[0].y, yMax = poly[0].y;
        for (int i = 1; i < n; ++i) {
            yMin = std::min(yMin, poly[i].y);
            yMax = std::max(yMax, poly[i].y);
        }
        yMin = std::max(yMin, 0);
        yMax = std::min(yMax, height - 1);

        for (int y = yMin; y <= yMax; ++y) {
            xs.clear();
            xs.reserve(n);

            for (int i = 0; i < n; ++i) {
                int j = (i + 1) % n;
                int y1 = poly[i].y;
                int y2 = poly[j].y;

                if (y1 == y2)
                    continue; // horizontal edge

                // Ensure y1 < y2 for consistent half-open interval [y1, y2)
                int x1 = poly[i].x;
                int x2 = poly[j].x;
                if (y1 > y2) {
                    std::swap(y1, y2);
                    std::swap(x1, x2);
                }

                if (y < y1 || y >= y2)
                    continue;

                float t = (float)(y - y1) / (float)(y2 - y1);
                float x = x1 + t * (x2 - x1);
                xs.push_back(x);
            }

            std::sort(xs.begin(), xs.end());
            for (size_t k = 0; k + 1 < xs.size(); k += 2) {
                int x1 = (int)std::ceil(xs[k]);
                int x2 = (int)std::floor(xs[k + 1]);
                x1 = std::max(x1, 0);
                x2 = std::min(x2, width - 1);
                for (int x = x1; x <= x2; ++x) {
                    data[y * step + x] = value;
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// liteocr_get_rotated_rect_points
// ------------------------------------------------------------------
std::vector<liteocr_point2f> liteocr_get_rotated_rect_points(const liteocr_rotated_rect& rr)
{
    float cx = rr.center.x;
    float cy = rr.center.y;
    float w = rr.size.width * 0.5f;
    float h = rr.size.height * 0.5f;
    float angle = rr.angle * (float)M_PI / 180.0f;
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);

    std::vector<liteocr_point2f> pts(4);
    // OpenCV-compatible corner order
    pts[0] = {cx - w*cosA + h*sinA, cy - w*sinA - h*cosA};
    pts[1] = {cx + w*cosA + h*sinA, cy + w*sinA - h*cosA};
    pts[2] = {cx + w*cosA - h*sinA, cy + w*sinA + h*cosA};
    pts[3] = {cx - w*cosA - h*sinA, cy - w*sinA + h*cosA};
    return pts;
}


