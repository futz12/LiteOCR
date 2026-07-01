#include "liteocr_engine.h"
#include "mat.h"
#include "liteocr_image.h"



bool liteocr_uvdoc_load_model(liteocr_uvdoc* uv, const char* paramPath, const char* binPath, const liteocr_infer_option &opt) {
    if (!uv) return false;
    uv->norm_vals[0] = 1 / 255.f;
    uv->norm_vals[1] = 1 / 255.f;
    uv->norm_vals[2] = 1 / 255.f;

    liteocr_apply_net_options(uv->model, opt);
    if (uv->model.load_param(paramPath) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleUVDoc param file from %s\n", paramPath);
        return false;
    }
    if (uv->model.load_model(binPath) == 0) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleUVDoc bin file from %s\n", binPath);
        return false;
    }
    return true;
}

bool liteocr_uvdoc_load_model_from_buffer(liteocr_uvdoc* uv, const char* paramBuffer, const unsigned char* binBuffer, const liteocr_infer_option &opt) {
    if (!uv) return false;
    uv->norm_vals[0] = 1 / 255.f;
    uv->norm_vals[1] = 1 / 255.f;
    uv->norm_vals[2] = 1 / 255.f;

    liteocr_apply_net_options(uv->model, opt);
    if (uv->model.load_param_mem(paramBuffer) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleUVDoc param from buffer\n");
        return false;
    }
    if (uv->model.load_model(binBuffer) == 0) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleUVDoc bin from buffer\n");
        return false;
    }
    return true;
}

liteocr_image liteocr_uvdoc_forward(liteocr_uvdoc* uv, const liteocr_image& input) {
    if (!uv || input.empty() || input.channels != 3) return liteocr_image();
    ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_BGR2RGB, input.width, input.height);
    in.substract_mean_normalize(0, uv->norm_vals);
    ncnn::Extractor ex = uv->model.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);
    liteocr_image output(out.w, out.h, liteocr_image_type::LITEOCR_IMAGE_U8C3);

    for (int y = 0; y < out.h; y++) {
        uint8_t* dst = output.ptr<uint8_t>(y);
        for (int x = 0; x < out.w; x++) {
            dst[x * 3 + 0] = static_cast<unsigned char>(out.channel(0).row(y)[x] * 255.0f);
            dst[x * 3 + 1] = static_cast<unsigned char>(out.channel(1).row(y)[x] * 255.0f);
            dst[x * 3 + 2] = static_cast<unsigned char>(out.channel(2).row(y)[x] * 255.0f);
        }
    }

    return output;
}


