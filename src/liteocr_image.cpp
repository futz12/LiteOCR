#include "liteocr_image.h"
#include "liteocr_imgproc.h"
#include <vector>
#include <cstring>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static int liteocr_channels_from_type(liteocr_image_type t) {
    return (t == LITEOCR_IMAGE_F32C1) ? 1 : static_cast<int>(t);
}

static int liteocr_elem_size_from_type(liteocr_image_type t) {
    return (t == LITEOCR_IMAGE_F32C1) ? sizeof(float) : 1;
}

liteocr_image::liteocr_image(int w, int h, liteocr_image_type t)
    : width(w), height(h), type(t)
{
    channels = liteocr_channels_from_type(t);
    int es = liteocr_elem_size_from_type(t);
    stride = w * channels * es;
    size_t total = static_cast<size_t>(h) * stride;
    storage = std::shared_ptr<uint8_t[]>(new uint8_t[total]());
    data = storage.get();
}

liteocr_image::liteocr_image(int w, int h, liteocr_image_type t, void* external_data, int step)
    : width(w), height(h), type(t)
{
    channels = liteocr_channels_from_type(t);
    stride = step;
    data = reinterpret_cast<uint8_t*>(external_data);
}

liteocr_image liteocr_image::from_roi(const liteocr_image& src, int x, int y, int w, int h) {
    liteocr_image roi;
    roi.width = w;
    roi.height = h;
    roi.channels = src.channels;
    roi.type = src.type;
    roi.stride = src.stride;
    roi.storage = src.storage;
    int es = src.elem_size1();
    int offset = y * src.stride + x * src.channels * es;
    roi.data = src.data + offset;
    return roi;
}

liteocr_image liteocr_image::clone() const {
    if (empty()) return liteocr_image();
    liteocr_image c(width, height, type);
    int row_bytes = width * channels * elem_size1();
    for (int y = 0; y < height; ++y)
        std::memcpy(c.data + y * c.stride, data + y * stride, row_bytes);
    return c;
}

int liteocr_image::elem_size() const {
    return channels * elem_size1();
}

int liteocr_image::elem_size1() const {
    return liteocr_elem_size_from_type(type);
}

liteocr_image liteocr_imread_image(const char* filename, int desired_channels) {
    int w, h, c;
    unsigned char* pixels = stbi_load(filename, &w, &h, &c, desired_channels);
    if (!pixels) return liteocr_image();
    
    int actual_channels = desired_channels ? desired_channels : c;
    liteocr_image_type t;
    if (actual_channels == 1) t = LITEOCR_IMAGE_U8C1;
    else if (actual_channels == 3) t = LITEOCR_IMAGE_U8C3;
    else if (actual_channels == 4) t = LITEOCR_IMAGE_U8C4;
    else {
        stbi_image_free(pixels);
        return liteocr_image();
    }
    
    liteocr_image img(w, h, t);
    if (actual_channels == 1) {
        for (int y = 0; y < h; ++y)
            std::memcpy(img.data + y * img.stride, pixels + y * w, w);
    } else if (actual_channels == 3) {
        liteocr_cvt_color(pixels, w, h, w * 3, LITEOCR_FORMAT_RGB,
                          img.data, img.stride, LITEOCR_FORMAT_BGR);
    } else if (actual_channels == 4) {
        liteocr_cvt_color(pixels, w, h, w * 4, LITEOCR_FORMAT_RGBA,
                          img.data, img.stride, LITEOCR_FORMAT_BGRA);
    }
    stbi_image_free(pixels);
    return img;
}

bool liteocr_imwrite_image(const char* filename, const liteocr_image& img) {
    if (img.empty()) return false;
    
    const char* ext = std::strrchr(filename, '.');
    if (!ext) return false;
    
    int comp = img.channels;
    int packed_stride = img.width * comp;
    std::vector<uint8_t> pixels(static_cast<size_t>(img.height) * packed_stride);
    if (img.channels == 1) {
        for (int y = 0; y < img.height; ++y)
            std::memcpy(pixels.data() + y * packed_stride, img.data + y * img.stride, img.width);
    } else if (img.channels == 3) {
        liteocr_cvt_color(img.data, img.width, img.height, img.stride, LITEOCR_FORMAT_BGR,
                          pixels.data(), packed_stride, LITEOCR_FORMAT_RGB);
    } else if (img.channels == 4) {
        liteocr_cvt_color(img.data, img.width, img.height, img.stride, LITEOCR_FORMAT_BGRA,
                          pixels.data(), packed_stride, LITEOCR_FORMAT_RGBA);
    } else {
        return false;
    }

    if (std::strcmp(ext, ".png") == 0 || std::strcmp(ext, ".PNG") == 0) {
        return stbi_write_png(filename, img.width, img.height, comp, pixels.data(), packed_stride) != 0;
    } else if (std::strcmp(ext, ".jpg") == 0 || std::strcmp(ext, ".JPG") == 0 ||
               std::strcmp(ext, ".jpeg") == 0 || std::strcmp(ext, ".JPEG") == 0) {
        return stbi_write_jpg(filename, img.width, img.height, comp, pixels.data(), 95) != 0;
    } else if (std::strcmp(ext, ".bmp") == 0 || std::strcmp(ext, ".BMP") == 0) {
        return stbi_write_bmp(filename, img.width, img.height, comp, pixels.data()) != 0;
    }
    return false;
}
