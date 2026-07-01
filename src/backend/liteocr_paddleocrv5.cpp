#include "liteocr_engine.h"
#include "liteocr_imgproc.h"

#include <gpu.h>
#include <mat.h>

/* ---------- liteocr_detector ---------- */
bool liteocr_detector_load_model(liteocr_detector* det, const char* paramPath, const char* binPath, const liteocr_infer_option &opt) {
    if (!det) return false;
    det->mean_vals[0] = 0.485f * 255.f;
    det->mean_vals[1] = 0.456f * 255.f;
    det->mean_vals[2] = 0.406f * 255.f;
    det->norm_vals[0] = 1 / (0.229f * 255.f);
    det->norm_vals[1] = 1 / (0.224f * 255.f);
    det->norm_vals[2] = 1 / (0.225f * 255.f);
    det->stride = 32;

    liteocr_apply_net_options(det->model, opt);
    if (det->model.load_param(paramPath) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 param file from %s\n", paramPath);
        return false;
    }
    if (det->model.load_model(binPath) != 0) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 bin file from %s\n", binPath);
        return false;
    }
    return true;
}

bool liteocr_detector_load_model_from_buffer(liteocr_detector* det, const char *paramBuffer, const unsigned char *binBuffer, const liteocr_infer_option &opt) {
    if (!det) return false;
    det->mean_vals[0] = 0.485f * 255.f;
    det->mean_vals[1] = 0.456f * 255.f;
    det->mean_vals[2] = 0.406f * 255.f;
    det->norm_vals[0] = 1 / (0.229f * 255.f);
    det->norm_vals[1] = 1 / (0.224f * 255.f);
    det->norm_vals[2] = 1 / (0.225f * 255.f);
    det->stride = 32;

    liteocr_apply_net_options(det->model, opt);
    if (det->model.load_param_mem(paramBuffer) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 param from buffer\n");
        return false;
    }
    if (det->model.load_model(binBuffer) != 0) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 bin from buffer\n");
        return false;
    }
    return true;
}

liteocr_image liteocr_detector_forward(liteocr_detector* det, const liteocr_image& input) {
    if (!det || input.empty() || input.channels != 3) return liteocr_image();
    ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_BGR, input.width, input.height);
    int w = in.w;
    int h = in.h;
    int wpad = (w + det->stride - 1) / det->stride * det->stride - w;
    int hpad = (h + det->stride - 1) / det->stride * det->stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    in_pad.substract_mean_normalize(det->mean_vals, det->norm_vals);

    auto ex = det->model.create_extractor();
    ex.input("in0", in_pad);
    ncnn::Mat out;
    ex.extract("out0", out);
    liteocr_image output(out.w, out.h, liteocr_image_type::LITEOCR_IMAGE_F32C1, out.data, out.w * sizeof(float));

    liteocr_image cropped = liteocr_image::from_roi(output, wpad / 2, hpad / 2, w, h).clone();
    return cropped;
}

/* ---------- liteocr_recognizer ---------- */
bool liteocr_recognizer_load_model(liteocr_recognizer* rec, const char* paramPath, const char* binPath, const liteocr_infer_option &opt) {
    if (!rec) return false;
    rec->mean_vals[0] = 0.5f * 255.f;
    rec->mean_vals[1] = 0.5f * 255.f;
    rec->mean_vals[2] = 0.5f * 255.f;
    rec->norm_vals[0] = 1 / (0.5f * 255.f);
    rec->norm_vals[1] = 1 / (0.5f * 255.f);
    rec->norm_vals[2] = 1 / (0.5f * 255.f);
    rec->target_height = 48;

    liteocr_apply_net_options(rec->model, opt);
    if (rec->model.load_param(paramPath) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 param file from %s\n", paramPath);
        return false;
    }
    if (rec->model.load_model(binPath) != 0) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 bin file from %s\n", binPath);
        return false;
    }
    return true;
}

bool liteocr_recognizer_load_model_from_buffer(liteocr_recognizer* rec, const char* paramBuffer, const unsigned char* binBuffer, const liteocr_infer_option &opt) {
    if (!rec) return false;
    rec->mean_vals[0] = 0.5f * 255.f;
    rec->mean_vals[1] = 0.5f * 255.f;
    rec->mean_vals[2] = 0.5f * 255.f;
    rec->norm_vals[0] = 1 / (0.5f * 255.f);
    rec->norm_vals[1] = 1 / (0.5f * 255.f);
    rec->norm_vals[2] = 1 / (0.5f * 255.f);
    rec->target_height = 48;

    liteocr_apply_net_options(rec->model, opt);
    if (rec->model.load_param_mem(paramBuffer) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 param from buffer\n");
        return false;
    }
    if (rec->model.load_model(binBuffer) != 0) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 bin from buffer\n");
        return false;
    }
    return true;
}

liteocr_image liteocr_recognizer_forward(liteocr_recognizer* rec, const liteocr_image& input) {
    if (!rec || input.empty() || input.channels != 3) return liteocr_image();
    int target_width = input.width * rec->target_height / input.height;
    if (target_width <= 0) return liteocr_image();
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(input.data, ncnn::Mat::PIXEL_BGR, input.width, input.height, target_width, rec->target_height);
    in.substract_mean_normalize(rec->mean_vals, rec->norm_vals);
    auto ex = rec->model.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);
    liteocr_image output(out.w, out.h, liteocr_image_type::LITEOCR_IMAGE_F32C1, out.data, out.w * sizeof(float));
    return output.clone();
}

/* ---------- Textline Orientation ---------- */
static void liteocr_textline_ori_init_paddle(liteocr_textline_ori* cls)
{
    cls->is_anglenet = false;
    cls->mean_vals[0] = 0.5f * 255.f;
    cls->mean_vals[1] = 0.5f * 255.f;
    cls->mean_vals[2] = 0.5f * 255.f;
    cls->norm_vals[0] = 1 / (0.5f * 255.f);
    cls->norm_vals[1] = 1 / (0.5f * 255.f);
    cls->norm_vals[2] = 1 / (0.5f * 255.f);
    cls->target_width = 160;
    cls->target_height = 80;
}

static void liteocr_textline_ori_init_anglenet(liteocr_textline_ori* cls)
{
    cls->is_anglenet = true;
    cls->mean_vals[0] = 127.5f;
    cls->mean_vals[1] = 127.5f;
    cls->mean_vals[2] = 127.5f;
    cls->norm_vals[0] = 1 / 127.5f;
    cls->norm_vals[1] = 1 / 127.5f;
    cls->norm_vals[2] = 1 / 127.5f;
    cls->target_width = 192;
    cls->target_height = 32;
}

bool liteocr_textline_ori_load_model(liteocr_textline_ori* cls, const char* paramPath, const char* binPath, const liteocr_infer_option &opt) {
    if (!cls) return false;
    if (opt.textline_ori_model_type == 1)
        liteocr_textline_ori_init_anglenet(cls);
    else
        liteocr_textline_ori_init_paddle(cls);

    liteocr_apply_net_options(cls->model, opt);
    if (cls->model.load_param(paramPath) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load textline orientation param file from %s\n", paramPath);
        return false;
    }
    if (cls->model.load_model(binPath) != 0) {
        fprintf(stderr, "[LiteOCR]Failed to load textline orientation bin file from %s\n", binPath);
        return false;
    }
    return true;
}

bool liteocr_textline_ori_load_model_from_buffer(liteocr_textline_ori* cls, const char* paramBuffer, const unsigned char* binBuffer, const liteocr_infer_option &opt) {
    if (!cls) return false;
    if (opt.textline_ori_model_type == 1)
        liteocr_textline_ori_init_anglenet(cls);
    else
        liteocr_textline_ori_init_paddle(cls);

    liteocr_apply_net_options(cls->model, opt);
    if (cls->model.load_param_mem(paramBuffer) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load textline orientation param from buffer\n");
        return false;
    }
    if (cls->model.load_model(binBuffer) != 0) {
        fprintf(stderr, "[LiteOCR]Failed to load textline orientation bin from buffer\n");
        return false;
    }
    return true;
}

int liteocr_textline_ori_forward(liteocr_textline_ori* cls, const liteocr_image& input) {
    if (!cls || input.empty() || input.channels != 3) return -1;

    ncnn::Mat in;

    if (cls->is_anglenet) {
        // ChineseOCR AngleNet preprocessing:
        // resize height to 32, keep aspect ratio, pad right with white or crop left to width 192.
        float ratio = static_cast<float>(cls->target_height) / static_cast<float>(input.height);
        int rsz_w = static_cast<int>(input.width * ratio);
        if (rsz_w <= 0) return -1;

        liteocr_image rsz_image(rsz_w, cls->target_height, input.type);
        liteocr_resize(input.data, input.width, input.height, input.stride, input.channels,
                       rsz_image.data, rsz_w, cls->target_height, rsz_image.stride);

        liteocr_image final_image;
        if (rsz_w < cls->target_width) {
            final_image = liteocr_image(cls->target_width, cls->target_height, rsz_image.type);
            liteocr_copy_make_border(rsz_image.data, rsz_image.width, rsz_image.height, rsz_image.stride, rsz_image.channels,
                                     final_image.data, final_image.width, final_image.height, final_image.stride,
                                     0, 0, 0, cls->target_width - rsz_w, 255);
        } else if (rsz_w > cls->target_width) {
            final_image = liteocr_image::from_roi(rsz_image, 0, 0, cls->target_width, cls->target_height).clone();
        } else {
            final_image = rsz_image.clone();
        }

        in = ncnn::Mat::from_pixels(final_image.data, ncnn::Mat::PIXEL_BGR2RGB,
                                    final_image.width, final_image.height);
    } else {
        constexpr float max_downscale = 3.0f;
        float ratio = static_cast<float>(cls->target_height) / static_cast<float>(input.height);
        int rsz_w = static_cast<int>(input.width * ratio);
        if (rsz_w <= 0) return -1;

        liteocr_image rsz_image;
        if (rsz_w < cls->target_width) {
            rsz_image = liteocr_image(rsz_w, cls->target_height, input.type);
            liteocr_resize(input.data, input.width, input.height, input.stride, input.channels,
                           rsz_image.data, rsz_w, cls->target_height, rsz_image.stride);
            int pad_width = cls->target_width - rsz_w;
            liteocr_image padded(cls->target_width, cls->target_height, rsz_image.type);
            liteocr_copy_make_border(rsz_image.data, rsz_image.width, rsz_image.height, rsz_image.stride, rsz_image.channels,
                                    padded.data, padded.width, padded.height, padded.stride,
                                    0, 0, 0, pad_width, 114);
            rsz_image = padded;
        } else if (rsz_w < static_cast<int>(cls->target_width * max_downscale)) {
            rsz_image = liteocr_image(cls->target_width, cls->target_height, input.type);
            liteocr_resize(input.data, input.width, input.height, input.stride, input.channels,
                            rsz_image.data, cls->target_width, cls->target_height, rsz_image.stride);
        } else {
            int crop_w = static_cast<int>(max_downscale * cls->target_width / ratio);
            crop_w = std::min(crop_w, input.width);
            liteocr_image crop_image = liteocr_image::from_roi(input, 0, 0, crop_w, input.height);
            rsz_image = liteocr_image(cls->target_width, cls->target_height, input.type);
            liteocr_resize(crop_image.data, crop_image.width, crop_image.height, crop_image.stride, crop_image.channels,
                            rsz_image.data, cls->target_width, cls->target_height, rsz_image.stride);
        }

        in = ncnn::Mat::from_pixels(rsz_image.data, ncnn::Mat::PIXEL_BGR, rsz_image.width, rsz_image.height);
    }

    in.substract_mean_normalize(cls->mean_vals, cls->norm_vals);
    auto ex = cls->model.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    int pred = out[0] > out[1] ? 0 : 1;
    // AngleNet: class 1 = upright, class 0 = rotated 180.
    // Paddle textline ori: class 1 = rotated 180.
    // liteocr_textline_ori_forward returns 1 when the image needs 180-degree rotation.
    return cls->is_anglenet ? (pred == 0 ? 1 : 0) : pred;
}

/* ---------- Doc Orientation ---------- */
bool liteocr_doc_ori_load_model(liteocr_doc_ori* cls, const char* paramPath, const char* binPath, const liteocr_infer_option &opt) {
    if (!cls) return false;
    cls->mean_vals[0] = 0.5f * 255.f;
    cls->mean_vals[1] = 0.5f * 255.f;
    cls->mean_vals[2] = 0.5f * 255.f;
    cls->norm_vals[0] = 1 / (0.5f * 255.f);
    cls->norm_vals[1] = 1 / (0.5f * 255.f);
    cls->norm_vals[2] = 1 / (0.5f * 255.f);
    cls->target_width = 224;
    cls->target_height = 224;

    liteocr_apply_net_options(cls->model, opt);
    if (cls->model.load_param(paramPath) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleDocORI param file from %s\n", paramPath);
        return false;
    }
    if (cls->model.load_model(binPath) != 0) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleDocORI bin file from %s\n", binPath);
        return false;
    }
    return true;
}

bool liteocr_doc_ori_load_model_from_buffer(liteocr_doc_ori* cls, const char* paramBuffer, const unsigned char* binBuffer, const liteocr_infer_option &opt) {
    if (!cls) return false;
    cls->mean_vals[0] = 0.5f * 255.f;
    cls->mean_vals[1] = 0.5f * 255.f;
    cls->mean_vals[2] = 0.5f * 255.f;
    cls->norm_vals[0] = 1 / (0.5f * 255.f);
    cls->norm_vals[1] = 1 / (0.5f * 255.f);
    cls->norm_vals[2] = 1 / (0.5f * 255.f);
    cls->target_width = 224;
    cls->target_height = 224;

    liteocr_apply_net_options(cls->model, opt);
    if (cls->model.load_param_mem(paramBuffer) == -1) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleDocORI param from buffer\n");
        return false;
    }
    if (cls->model.load_model(binBuffer) != 0) {
        fprintf(stderr, "[LiteOCR]Failed to load PaddleDocORI bin from buffer\n");
        return false;
    }
    return true;
}

int liteocr_doc_ori_forward(liteocr_doc_ori* cls, const liteocr_image& input) {
    if (!cls || input.empty() || input.channels != 3) return -1;
    int target_size = 256;
    int target_width_resize = 0;
    int target_height_resize = 0;

    if (input.width < input.height) {
        target_width_resize = target_size;
        target_height_resize = input.height * target_size / input.width;
    } else {
        target_height_resize = target_size;
        target_width_resize = input.width * target_size / input.height;
    }
    if (target_width_resize < cls->target_width || target_height_resize < cls->target_height) return -1;

    liteocr_image resized(target_width_resize, target_height_resize, input.type);
    liteocr_resize(input.data, input.width, input.height, input.stride, input.channels,
                    resized.data, target_width_resize, target_height_resize, resized.stride);

    int x_start = (target_width_resize - cls->target_width) / 2;
    int y_start = (target_height_resize - cls->target_height) / 2;

    ncnn::Mat in = ncnn::Mat::from_pixels_roi(resized.data, ncnn::Mat::PIXEL_BGR, resized.width, resized.height, x_start, y_start, cls->target_width, cls->target_height);
    in.substract_mean_normalize(cls->mean_vals, cls->norm_vals);
    auto ex = cls->model.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    int max_index = 0;
    float max_value = out[0];
    for (int i = 1; i < out.w; i++) {
        if (out[i] > max_value) {
            max_value = out[i];
            max_index = i;
        }
    }
    return max_index;
}

/* ---------- CTC Decoder ---------- */
std::vector<std::tuple<int, float, int>> liteocr_ctc_decode(const liteocr_image& probs, int blank_index) {
    std::vector<std::tuple<int, float, int>> result;
    int prev_index = -1;
    for (int i = 0; i < probs.height; i++) {
        float max_value = -1e10;
        int max_index = -1;
        const float* row = probs.ptr<float>(i);
        for (int j = 0; j < probs.width; j++) {
            float value = row[j];
            if (value > max_value) {
                max_value = value;
                max_index = j;
            }
        }
        if (max_index != blank_index && max_index != prev_index) {
            result.push_back(std::make_tuple(max_index, max_value, i));
        }
        prev_index = max_index;
    }
    return result;
}
