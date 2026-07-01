#include "liteocr_contours.h"
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <climits>



namespace {

// Freeman chain code deltas: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
static const int dx[8] = { 1,  1,  0, -1, -1, -1,  0,  1 };
static const int dy[8] = { 0, -1, -1, -1,  0,  1,  1,  1 };

static inline void initDeltas(int delta[16], int w)
{
    delta[0]  =  1;      delta[1]  =  1 - w; delta[2]  = -w;     delta[3]  = -1 - w;
    delta[4]  = -1;      delta[5]  = -1 + w; delta[6]  =  w;     delta[7]  =  1 + w;
    delta[8]  =  1;      delta[9]  =  1 - w; delta[10] = -w;     delta[11] = -1 - w;
    delta[12] = -1;      delta[13] = -1 + w; delta[14] =  w;     delta[15] =  1 + w;
}

class ContourScanner {
public:
    int approxMethod;
    liteocr_point pt = {1, 1};
    int nbd = 1;
    std::vector<std::vector<liteocr_point>> contours;
    int w, h;
    int8_t* img;
    int delta[16];

    ContourScanner(int method, int _w, int _h, int8_t* _img)
        : approxMethod(method), w(_w), h(_h), img(_img)
    {
        initDeltas(delta, w);
    }

    bool traceContour(int x, int y, bool isHole);
};

// ------------------------------------------------------------------
// Trace a single border (Suzuki border following).
// ------------------------------------------------------------------
bool ContourScanner::traceContour(int x, int y, bool isHole)
{
    const bool isDirect = (approxMethod == LITEOCR_CHAIN_APPROX_NONE);
    int i0 = y * w + x;

    // Initial direction search
    int s_end = isHole ? 0 : 4;
    int s = s_end;
    int i1;
    do {
        s = (s - 1) & 7;
        i1 = i0 + delta[s];
    } while (img[i1] == 0 && s != s_end);

    std::vector<liteocr_point> pts;
    pts.reserve(64);  // conservative initial capacity, avoids early realloc
    liteocr_intrect bbox = {0, 0, 0, 0};

    if (s == s_end) {
        // Single-pixel contour
        img[i0] = -nbd;
        pts.push_back({x - 1, y - 1});
        bbox = {x - 1, y - 1, 1, 1};
        contours.push_back(std::move(pts));
        return true;
    }

    int i3 = i0;
    int prev_s = s ^ 4;
    liteocr_point pt = {x - 1, y - 1};
    bbox = {pt.x, pt.y, pt.x, pt.y};

    for (;;) {
        int s_end_loop = s;
        s = std::min(s, 15);
        int i4 = 0;
        while (s < 15) {
            ++s;
            i4 = i3 + delta[s];
            if (img[i4] != 0)
                break;
        }
        s &= 7;

        if (img[i4] == 0)
            break; // safety

        // Mark current pixel according to Suzuki rules
        if ((unsigned)(s - 1) < (unsigned)s_end_loop) {
            img[i3] = -nbd;
        } else if (img[i3] == 1) {
            img[i3] = nbd;
        }

        // Record point (SIMPLE skips straight-line pixels)
        if (s != prev_s || isDirect) {
            pts.push_back(pt);
        }

        // Update bbox
        if (s != prev_s) {
            if (pt.x < bbox.x) bbox.x = pt.x;
            else if (pt.x > bbox.width) bbox.width = pt.x;
            if (pt.y < bbox.y) bbox.y = pt.y;
            else if (pt.y > bbox.height) bbox.height = pt.y;
        }

        prev_s = s;
        pt.x += dx[s];
        pt.y += dy[s];

        if (i4 == i0 && i3 == i1)
            break;

        i3 = i4;
        s = (s + 4) & 7;
    }

    bbox.width  -= bbox.x - 1;
    bbox.height -= bbox.y - 1;
    contours.push_back(std::move(pts));
    return true;
}

// ------------------------------------------------------------------
// Check whether (x, y) is a valid contour start point.
// ------------------------------------------------------------------
static bool contourScan(ContourScanner& scanner, int prev, int p, int x, int y)
{
    bool isHole = false;
    int startX = x;

    if (prev == 0 && p == 1) {
        isHole = false;
        startX = x;
    } else if (prev >= 1 && p == 0) {
        isHole = true;
        startX = x - 1;
    } else {
        return false;
    }

    scanner.nbd++;
    scanner.pt.x = startX + 1;
    scanner.pt.y = y;
    return scanner.traceContour(startX, y, isHole);
}

// ------------------------------------------------------------------
// Advance x to the next pixel with a different value than prev.
// ------------------------------------------------------------------
static int findNextX(int x, int y, int& prev, int& p, int8_t* img, int w, int width)
{
    if (p = img[y * w + x], p != prev)
        return x;

    // Unrolled batch scan: process 8 pixels at a time
    for (; x <= width - 8; x += 8) {
        int8_t v0 = img[y * w + x];
        int8_t v1 = img[y * w + x + 1];
        int8_t v2 = img[y * w + x + 2];
        int8_t v3 = img[y * w + x + 3];
        int8_t v4 = img[y * w + x + 4];
        int8_t v5 = img[y * w + x + 5];
        int8_t v6 = img[y * w + x + 6];
        int8_t v7 = img[y * w + x + 7];
        if (v0 != prev || v1 != prev || v2 != prev || v3 != prev ||
            v4 != prev || v5 != prev || v6 != prev || v7 != prev) {
            if (v0 != prev) { p = v0; return x; }
            if (v1 != prev) { p = v1; return x + 1; }
            if (v2 != prev) { p = v2; return x + 2; }
            if (v3 != prev) { p = v3; return x + 3; }
            if (v4 != prev) { p = v4; return x + 4; }
            if (v5 != prev) { p = v5; return x + 5; }
            if (v6 != prev) { p = v6; return x + 6; }
            p = v7; return x + 7;
        }
    }

    // Remainder
    for (; x < width; ++x) {
        p = img[y * w + x];
        if (p != prev)
            return x;
    }
    return x;
}

// ------------------------------------------------------------------
// Main scan loop
// ------------------------------------------------------------------
static bool findNext(ContourScanner& scanner)
{
    int x = scanner.pt.x;
    int y = scanner.pt.y;
    int width = scanner.w - 1;
    int height = scanner.h - 1;
    int prev = scanner.img[y * scanner.w + (x - 1)];

    for (; y < height; y++) {
        int p = 0;
        for (; x < width; x++) {
            x = findNextX(x, y, prev, p, scanner.img, scanner.w, width);
            if (x >= width)
                break;
            if (contourScan(scanner, prev, p, x, y))
                return true;
            prev = p;
        }
        x = 1;
        prev = 0;
    }
    return false;
}

} // anonymous namespace

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------
void liteocr_find_contours(const uint8_t* data, int width, int height, int step,
                  std::vector<std::vector<liteocr_point>>& contours,
                  liteocr_contour_approx_mode approx)
{
    contours.clear();

    if (width <= 0 || height <= 0 || !data)
        return;

    const int w = width + 2;
    const int h = height + 2;

    // Allocate working image with 1-pixel zero padding.
    std::vector<int8_t> work(w * h, 0);
    for (int y = 0; y < height; ++y) {
        const uint8_t* srcRow = data + y * step;
        int8_t* dstRow = work.data() + (y + 1) * w + 1;
        for (int x = 0; x < width; ++x)
            dstRow[x] = srcRow[x] > 0 ? 1 : 0;
    }

    ContourScanner scanner(approx, w, h, work.data());
    while (findNext(scanner)) {
        // keep scanning
    }

    contours = std::move(scanner.contours);
}


