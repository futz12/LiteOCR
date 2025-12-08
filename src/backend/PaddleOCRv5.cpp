#include "BaseInfer.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"

#include <ncnn/gpu.h>
#include <ncnn/mat.h>

namespace LiteOCR {
    bool PaddleDetector::loadModel(const char* paramPath, const char* binPath, const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                model.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        model.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            model.opt.use_fp16_arithmetic = true;
            model.opt.use_fp16_storage = true;
            model.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            model.opt.use_int8_arithmetic = true;
            model.opt.use_int8_storage = true;
            model.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = model.load_param(paramPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 param file from %s\n", paramPath);
            return false;
        }
        ret = model.load_model(binPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 bin file from %s\n", binPath);
            return false;
        }
        return true;
    }

    bool PaddleDetector::loadModelFromBuffer(const char *paramBuffer, const unsigned char *binBuffer, const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                model.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        model.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            model.opt.use_fp16_arithmetic = true;
            model.opt.use_fp16_storage = true;
            model.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            model.opt.use_int8_arithmetic = true;
            model.opt.use_int8_storage = true;
            model.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = model.load_param_mem(paramBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 param from buffer\n");
            return false;
        }
        ret = model.load_model(paramBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 bin from buffer\n");
            return false;
        }
        return true;
    }

    cv::Mat PaddleDetector::forward(const cv::Mat& input) {
        ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_BGR, input.cols, input.rows);
        // pad to stride
        int w = in.w;;
        int h = in.h;
        int wpad = (w + stride - 1) / stride * stride - w;
        int hpad = (h + stride - 1) / stride * stride - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
        in_pad.substract_mean_normalize(mean_vals, norm_vals);

        auto ex = model.create_extractor();
        ex.input("in0", in_pad);
        ncnn::Mat out;
        ex.extract("out0", out);
        cv::Mat output(out.h, out.w, CV_32FC1, out.data);

        // crop to original size
        cv::Mat output_cropped = output(cv::Rect(wpad / 2, hpad / 2, w, h)).clone();
        return output_cropped;
    }

    bool PaddleRecognizer::loadModel(const char* paramPath, const char* binPath, const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                model.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        model.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            model.opt.use_fp16_arithmetic = true;
            model.opt.use_fp16_storage = true;
            model.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            model.opt.use_int8_arithmetic = true;
            model.opt.use_int8_storage = true;
            model.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = model.load_param(paramPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 param file from %s\n", paramPath);
            return false;
        }
        ret = model.load_model(binPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 bin file from %s\n", binPath);
            return false;
        }
        return true;
    }

    bool PaddleRecognizer::loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                model.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        model.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            model.opt.use_fp16_arithmetic = true;
            model.opt.use_fp16_storage = true;
            model.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            model.opt.use_int8_arithmetic = true;
            model.opt.use_int8_storage = true;
            model.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = model.load_param_mem(paramBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 param from buffer\n");
            return false;
        }
        ret = model.load_model(paramBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleOCRv5 bin from buffer\n");
            return false;
        }
        return true;
    }

    cv::Mat PaddleRecognizer::forward(const cv::Mat& input) {
        int target_width = input.cols * target_height / input.rows;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(input.data, ncnn::Mat::PIXEL_BGR, input.cols, input.rows, target_width, target_height);
        in.substract_mean_normalize(mean_vals, norm_vals);
        auto ex = model.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);
        cv::Mat output(out.h, out.w, CV_32FC1, out.data);
        return output.clone();
    }

    std::vector<std::tuple<int, float, int>> CTCDecoder::decode(const cv::Mat& probs, int blank_index) {
        std::vector<std::tuple<int, float, int>> result;
        int prev_index = -1;
        for (int i = 0; i < probs.rows; i++) {
            // find max index
            float max_value = -1e10;
            int max_index = -1;
            for (int j = 0; j < probs.cols; j++) {
                float value = probs.at<float>(i, j);
                if (value > max_value) {
                    max_value = value;
                    max_index = j;
                }
            }
            // skip if blank or same as previous
            if (max_index != blank_index && max_index != prev_index) {
                result.push_back(std::make_tuple(max_index, max_value, i));
            }
            prev_index = max_index;
        }
        return result;
    }

    bool PaddleTextlineORI::loadModel(const char* paramPath, const char* binPath, const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                model.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        model.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            model.opt.use_fp16_arithmetic = true;
            model.opt.use_fp16_storage = true;
            model.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            model.opt.use_int8_arithmetic = true;
            model.opt.use_int8_storage = true;
            model.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = model.load_param(paramPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleTextlineORI param file from %s\n", paramPath);
            return false;
        }
        ret = model.load_model(binPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleTextlineORI bin file from %s\n", binPath);
            return false;
        }
        return true;
    }

    bool PaddleTextlineORI::loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                model.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        model.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            model.opt.use_fp16_arithmetic = true;
            model.opt.use_fp16_storage = true;
            model.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            model.opt.use_int8_arithmetic = true;
            model.opt.use_int8_storage = true;
            model.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = model.load_param_mem(paramBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleTextlineORI param from buffer\n");
            return false;
        }
        ret = model.load_model(paramBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleTextlineORI bin from buffer\n");
            return false;
        }
        return true;
    }

    int PaddleTextlineORI::forward(const cv::Mat& input) {
        constexpr float max_downscale = 3.0f;

        float ratio = static_cast<float>(target_height) / static_cast<float>(input.rows);
        int rsz_w   = static_cast<int>(input.cols * ratio);

        cv::Mat rsz_image;

        if (rsz_w < target_width) {
            cv::resize(input, rsz_image, cv::Size(target_width, target_height));
            int pad_width = target_width - rsz_w;
            cv::copyMakeBorder(rsz_image, rsz_image, 0, 0, 0, pad_width,
                            cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0));
        } else if (rsz_w < static_cast<int>(target_width * max_downscale)) {
            cv::resize(input, rsz_image, cv::Size(target_width, target_height));
        } else {
            int crop_w = static_cast<int>(max_downscale * target_width / ratio);
            crop_w = std::min(crop_w, input.cols);
            cv::Mat crop_image = input(cv::Rect(0, 0, crop_w, input.rows));
            cv::resize(crop_image, rsz_image, cv::Size(target_width, target_height));
        }

        ncnn::Mat in = ncnn::Mat::from_pixels(rsz_image.data, ncnn::Mat::PIXEL_BGR, rsz_image.cols, rsz_image.rows);
    
        in.substract_mean_normalize(mean_vals, norm_vals);

        auto ex = model.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);

        return out[0] > out[1] ? 0 : 1;
    }
    
    bool PaddleDocORI::loadModel(const char* paramPath, const char* binPath, const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                model.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        model.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            model.opt.use_fp16_arithmetic = true;
            model.opt.use_fp16_storage = true;
            model.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            model.opt.use_int8_arithmetic = true;
            model.opt.use_int8_storage = true;
            model.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = model.load_param(paramPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleDocORI param file from %s\n", paramPath);
            return false;
        }
        ret = model.load_model(binPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleDocORI bin file from %s\n", binPath);
            return false;
        }
        return true;
    }

    bool PaddleDocORI::loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) {
        if (opt.gpuDeviceId != -1) {
            if (ncnn::get_gpu_count() <= 0) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have any vulkan device. Switch to cpu mode\n");
            } else if (ncnn::get_gpu_count() <= opt.gpuDeviceId) {
                fprintf(stderr, "[LiteOCR]Your Device don`t have gpu device %d. Switch to cpu mode\n", opt.gpuDeviceId);
            } else {
                model.set_vulkan_device(opt.gpuDeviceId);
            }
        }
        model.opt.num_threads = opt.numThreads;
        if (opt.useFp16) {
            model.opt.use_fp16_arithmetic = true;
            model.opt.use_fp16_storage = true;
            model.opt.use_fp16_packed = true;
        }
        if (opt.useInt8) {
            model.opt.use_int8_arithmetic = true;
            model.opt.use_int8_storage = true;
            model.opt.use_int8_packed = true;
        }
        int ret = 0;
        ret = model.load_param_mem(paramBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleDocORI param from buffer\n");
            return false;
        }
        ret = model.load_model(paramBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleDocORI bin from buffer\n");
            return false;
        }
        return true;
    }

    int PaddleDocORI::forward(const cv::Mat& input) {
        int target_size = 256; // short side resize to 256
        int target_width_resize = 0;
        int target_height_resize = 0;

        if (input.cols < input.rows) {
            target_width_resize = target_size;
            target_height_resize = input.rows * target_size / input.cols;
        } else {
            target_height_resize = target_size;
            target_width_resize = input.cols * target_size / input.rows;
        }

        cv::Mat resized;
        cv::resize(input, resized, cv::Size(target_width_resize, target_height_resize));
        
        // Crop center
        int x_start = (target_width_resize - target_width) / 2;
        int y_start = (target_height_resize - target_height) / 2;
        
        ncnn::Mat in = ncnn::Mat::from_pixels_roi(resized.data, ncnn::Mat::PIXEL_BGR, resized.cols, resized.rows, x_start, y_start, target_width, target_height);
        in.substract_mean_normalize(mean_vals, norm_vals);
        auto ex = model.create_extractor();
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
}