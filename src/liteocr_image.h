#pragma once
#include <cstdint>
#include <memory>
#include <cstring>

enum liteocr_image_type {
    LITEOCR_IMAGE_U8C1 = 1,
    LITEOCR_IMAGE_U8C3 = 3,
    LITEOCR_IMAGE_U8C4 = 4,
    LITEOCR_IMAGE_F32C1 = 5
};

struct liteocr_image {
    uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int stride = 0;
    liteocr_image_type type = LITEOCR_IMAGE_U8C1;
    
    std::shared_ptr<uint8_t[]> storage;
    
    liteocr_image() = default;
    liteocr_image(int w, int h, liteocr_image_type t);
    liteocr_image(int w, int h, liteocr_image_type t, void* external_data, int step);
    
    static liteocr_image from_roi(const liteocr_image& src, int x, int y, int w, int h);
    
    bool empty() const { return data == nullptr || width == 0 || height == 0; }
    liteocr_image clone() const;
    
    int elem_size() const;
    int elem_size1() const;
    
    template<typename T>
    T* ptr(int y = 0) { return reinterpret_cast<T*>(data + y * stride); }
    
    template<typename T>
    const T* ptr(int y = 0) const { return reinterpret_cast<const T*>(data + y * stride); }
};

liteocr_image liteocr_imread_image(const char* filename, int desired_channels = 0);
bool liteocr_imwrite_image(const char* filename, const liteocr_image& img);
