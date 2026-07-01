#include "liteocr.h"

#include <gpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct benchmark_config {
    std::string det_param;
    std::string det_bin;
    std::string rec_param;
    std::string rec_bin;
    std::string mode = "det";     // det / rec
    int num_threads = 4;
    int gpu_device_id = -1;         // -1 = CPU, >=0 = Vulkan GPU
    bool use_fp16 = false;
    bool use_int8 = false;
    bool use_bf16 = false;
    int warmup = 3;
    int iterations = 10;
    bool verbose = false;
    bool list_gpu = false;
};

static void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "\n"
              << "Benchmark LiteOCR detector or recognizer on synthetic random images.\n"
              << "\n"
              << "Options:\n"
              << "  -d, --det-param <path>     Detector param path\n"
              << "      --det-bin   <path>     Detector bin path (default: replace .param with .bin)\n"
              << "  -r, --rec-param <path>     Recognizer param path\n"
              << "      --rec-bin   <path>     Recognizer bin path (default: replace .param with .bin)\n"
              << "  -m, --mode      <mode>     Benchmark mode: det, rec (default: det)\n"
              << "  -t, --threads   <n>        Number of threads (default: 4)\n"
              << "  -g, --gpu       <id>       Vulkan GPU device id, -1 for CPU (default: -1)\n"
              << "      --list-gpu             List available Vulkan GPUs and exit\n"
              << "      --fp16                 Enable FP16 inference\n"
              << "      --int8                 Enable INT8 inference\n"
              << "      --bf16                 Enable BF16 inference\n"
              << "      --warmup      <n>      Warmup iterations (default: 3)\n"
              << "  -n, --iter      <n>        Benchmark iterations (default: 10)\n"
              << "      --verbose              Print per-iteration time\n"
              << "  -h, --help                 Show this help\n"
              << "\n"
              << "Examples:\n"
              << "  " << argv0 << " -d ./models/PP-OCRv5_mobile_det.param -m det -t 4 -g 1 --fp16 -n 20\n"
              << "  " << argv0 << " -r ./models/PP-OCRv5_mobile_rec.param -m rec -t 4 --int8 -n 20\n";
}

static std::string param_to_bin(const std::string& param_path) {
    if (param_path.size() > 6 && param_path.compare(param_path.size() - 6, 6, ".param") == 0) {
        return param_path.substr(0, param_path.size() - 6) + ".bin";
    }
    return param_path + ".bin";
}

static bool parse_args(int argc, char** argv, benchmark_config& cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return false;
        } else if ((strcmp(arg, "-d") == 0 || strcmp(arg, "--det-param") == 0) && i + 1 < argc) {
            cfg.det_param = argv[++i];
        } else if (strcmp(arg, "--det-bin") == 0 && i + 1 < argc) {
            cfg.det_bin = argv[++i];
        } else if ((strcmp(arg, "-r") == 0 || strcmp(arg, "--rec-param") == 0) && i + 1 < argc) {
            cfg.rec_param = argv[++i];
        } else if (strcmp(arg, "--rec-bin") == 0 && i + 1 < argc) {
            cfg.rec_bin = argv[++i];
        } else if ((strcmp(arg, "-m") == 0 || strcmp(arg, "--mode") == 0) && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else if ((strcmp(arg, "-t") == 0 || strcmp(arg, "--threads") == 0) && i + 1 < argc) {
            cfg.num_threads = std::atoi(argv[++i]);
        } else if ((strcmp(arg, "-g") == 0 || strcmp(arg, "--gpu") == 0) && i + 1 < argc) {
            cfg.gpu_device_id = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--list-gpu") == 0) {
            cfg.list_gpu = true;
        } else if (strcmp(arg, "--fp16") == 0) {
            cfg.use_fp16 = true;
        } else if (strcmp(arg, "--int8") == 0) {
            cfg.use_int8 = true;
        } else if (strcmp(arg, "--bf16") == 0) {
            cfg.use_bf16 = true;
        } else if (strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup = std::atoi(argv[++i]);
        } else if ((strcmp(arg, "-n") == 0 || strcmp(arg, "--iter") == 0) && i + 1 < argc) {
            cfg.iterations = std::atoi(argv[++i]);
        } else if (strcmp(arg, "--verbose") == 0) {
            cfg.verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return false;
        }
    }

    if (cfg.det_bin.empty() && !cfg.det_param.empty()) {
        cfg.det_bin = param_to_bin(cfg.det_param);
    }
    if (cfg.rec_bin.empty() && !cfg.rec_param.empty()) {
        cfg.rec_bin = param_to_bin(cfg.rec_param);
    }

    if (cfg.list_gpu) {
        return true;
    }

    if (cfg.mode != "det" && cfg.mode != "rec") {
        std::cerr << "Invalid mode: " << cfg.mode << ". Must be det or rec." << std::endl;
        return false;
    }

    if (cfg.mode == "det" && cfg.det_param.empty()) {
        std::cerr << "Detector model path is required for det mode." << std::endl;
        return false;
    }
    if (cfg.mode == "rec" && cfg.rec_param.empty()) {
        std::cerr << "Recognizer model path is required for rec mode." << std::endl;
        return false;
    }

    return true;
}

static liteocr_infer_option_t make_option(const benchmark_config& cfg) {
    liteocr_infer_option_t opt = {};
    opt.num_threads = cfg.num_threads;
    opt.gpu_device_id = cfg.gpu_device_id;
    opt.use_fp16 = cfg.use_fp16 ? 1 : 0;
    opt.use_int8 = cfg.use_int8 ? 1 : 0;
    opt.use_bf16 = cfg.use_bf16 ? 1 : 0;
    return opt;
}

static void list_vulkan_gpus() {
    int gpu_count = ncnn::get_gpu_count();
    if (gpu_count <= 0) {
        std::cout << "No Vulkan GPU device available." << std::endl;
        return;
    }
    std::cout << "Available Vulkan GPU devices:" << std::endl;
    for (int i = 0; i < gpu_count; ++i) {
        std::cout << "  [" << i << "] " << ncnn::get_gpu_info(i).device_name() << std::endl;
    }
}

static void print_config(const benchmark_config& cfg) {
    std::cout << "Benchmark Configuration:\n"
              << "  Mode:        " << cfg.mode << "\n"
              << "  Threads:     " << cfg.num_threads << "\n"
              << "  GPU device:  " << cfg.gpu_device_id << "\n"
              << "  FP16:        " << (cfg.use_fp16 ? "yes" : "no") << "\n"
              << "  INT8:        " << (cfg.use_int8 ? "yes" : "no") << "\n"
              << "  BF16:        " << (cfg.use_bf16 ? "yes" : "no") << "\n"
              << "  Warmup:      " << cfg.warmup << "\n"
              << "  Iterations:  " << cfg.iterations << "\n";
    if (!cfg.det_param.empty()) {
        std::cout << "  Detector:    " << cfg.det_param << "\n";
    }
    if (!cfg.rec_param.empty()) {
        std::cout << "  Recognizer:  " << cfg.rec_param << "\n";
    }
    std::cout << "  Input:       synthetic random images\n" << std::endl;
}

static void report_times(const std::vector<double>& times_ms, const char* label) {
    if (times_ms.empty()) return;

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());

    double sum = 0.0;
    for (double t : times_ms) sum += t;
    double avg = sum / times_ms.size();
    double min_t = sorted.front();
    double max_t = sorted.back();
    double median = sorted[sorted.size() / 2];

    std::cout << label << ":\n"
              << "  Iterations: " << times_ms.size() << "\n"
              << "  Average:    " << avg << " ms\n"
              << "  Median:     " << median << " ms\n"
              << "  Min:        " << min_t << " ms\n"
              << "  Max:        " << max_t << " ms\n"
              << "  Throughput: " << (1000.0 / avg) << " it/s\n";
}

static liteocr_image_t make_random_image(int w, int h, int channels, unsigned int seed) {
    liteocr_image_t img = {};
    img.width = w;
    img.height = h;
    img.channels = channels;
    img.stride = w * channels;
    img.data = (unsigned char*)std::malloc(w * h * channels);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) {
                seed = seed * 1103515245u + 12345u;
                img.data[y * img.stride + x * channels + c] = (unsigned char)((seed >> 16) & 0xFF);
            }
        }
    }
    return img;
}

static int benchmark_detector(liteocr_detector_t det, int w, int h, const benchmark_config& cfg) {
    liteocr_image_t img = make_random_image(w, h, 3, 12345u);

    for (int i = 0; i < cfg.warmup; ++i) {
        liteocr_image_t out = liteocr_detector_forward(det, &img);
        liteocr_free_image(&out);
    }

    std::vector<double> times_ms;
    times_ms.reserve(cfg.iterations);
    for (int i = 0; i < cfg.iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        liteocr_image_t out = liteocr_detector_forward(det, &img);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
        if (cfg.verbose) {
            std::cout << "  iter " << i << ": " << ms << " ms" << std::endl;
        }
        liteocr_free_image(&out);
    }

    char label[64];
    std::snprintf(label, sizeof(label), "Detector forward %dx%d", w, h);
    report_times(times_ms, label);
    liteocr_free_image(&img);
    return 0;
}

static int benchmark_recognizer(liteocr_recognizer_t rec, int w, int h, const benchmark_config& cfg) {
    liteocr_image_t img = make_random_image(w, h, 3, 12345u);

    for (int i = 0; i < cfg.warmup; ++i) {
        liteocr_image_t out = liteocr_recognizer_forward(rec, &img);
        liteocr_free_image(&out);
    }

    std::vector<double> times_ms;
    times_ms.reserve(cfg.iterations);
    for (int i = 0; i < cfg.iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        liteocr_image_t out = liteocr_recognizer_forward(rec, &img);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
        if (cfg.verbose) {
            std::cout << "  iter " << i << ": " << ms << " ms" << std::endl;
        }
        liteocr_free_image(&out);
    }

    char label[64];
    std::snprintf(label, sizeof(label), "Recognizer forward %dx%d", w, h);
    report_times(times_ms, label);
    liteocr_free_image(&img);
    return 0;
}

int main(int argc, char** argv) {
    benchmark_config cfg;
    if (!parse_args(argc, argv, cfg)) {
        return (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) ? 0 : -1;
    }

    if (cfg.list_gpu) {
        list_vulkan_gpus();
        return 0;
    }

    print_config(cfg);

    int rc = 0;
    if (cfg.mode == "det") {
        liteocr_detector_t det = liteocr_detector_create();
        liteocr_infer_option_t opt = make_option(cfg);
        if (liteocr_detector_load_model(det, cfg.det_param.c_str(), cfg.det_bin.c_str(), &opt) != 0) {
            std::cerr << "Failed to load detector model." << std::endl;
            liteocr_detector_destroy(det);
            return -1;
        }
        const int det_sizes[][2] = {{320, 320}, {640, 640}, {960, 960}};
        for (const auto& s : det_sizes) {
            rc = benchmark_detector(det, s[0], s[1], cfg);
            if (rc != 0) break;
        }
        liteocr_detector_destroy(det);
    } else if (cfg.mode == "rec") {
        liteocr_recognizer_t rec = liteocr_recognizer_create();
        liteocr_infer_option_t opt = make_option(cfg);
        if (liteocr_recognizer_load_model(rec, cfg.rec_param.c_str(), cfg.rec_bin.c_str(), &opt) != 0) {
            std::cerr << "Failed to load recognizer model." << std::endl;
            liteocr_recognizer_destroy(rec);
            return -1;
        }
        const int rec_sizes[][2] = {{128, 48}, {256, 48}, {512, 48}};
        for (const auto& s : rec_sizes) {
            rc = benchmark_recognizer(rec, s[0], s[1], cfg);
            if (rc != 0) break;
        }
        liteocr_recognizer_destroy(rec);
    }

    return rc;
}
