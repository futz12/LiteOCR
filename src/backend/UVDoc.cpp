#include "BaseInfer.h"
#include "ncnn/mat.h"
#include "opencv2/core/mat.hpp"

namespace LiteOCR {
    bool PaddleUVDoc::loadModel(const char* paramPath, const char* binPath, const InferOption &opt) {
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
            fprintf(stderr, "[LiteOCR]Failed to load PaddleUVDoc param file from %s\n", paramPath);
            return false;
        }
        ret = model.load_model(binPath);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleUVDoc bin file from %s\n", binPath);
            return false;
        }
        return true;
    }

    bool PaddleUVDoc::loadModelFromBuffer(const char* paramBuffer,const unsigned char* binBuffer, const InferOption &opt) {
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
            fprintf(stderr, "[LiteOCR]Failed to load PaddleUVDoc param from buffer\n");
            return false;
        }
        ret = model.load_model(binBuffer);
        if (ret == -1) {
            fprintf(stderr, "[LiteOCR]Failed to load PaddleUVDoc bin from buffer\n");
            return false;
        }
        return true;
    }

    cv::Mat PaddleUVDoc::forward(const cv::Mat& input) {
        ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_BGR2RGB, input.cols, input.rows);
        in.substract_mean_normalize(0, norm_vals);
        ncnn::Extractor ex = model.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);
        cv::Mat output(out.h, out.w, CV_8UC3);

        for (int y = 0; y < out.h; y++) {
            for (int x = 0; x < out.w; x++) {
                output.at<cv::Vec3b>(y, x)[0] = static_cast<unsigned char>(out.channel(0).row(y)[x] * 255.0f);
                output.at<cv::Vec3b>(y, x)[1] = static_cast<unsigned char>(out.channel(1).row(y)[x] * 255.0f);
                output.at<cv::Vec3b>(y, x)[2] = static_cast<unsigned char>(out.channel(2).row(y)[x] * 255.0f);
            }
        }

        return output;
    }
}