#pragma once

#include "liteocr_image.h"
#include <net.h>
#include <mat.h>
#include <vector>
#include <string>
#include <utility>

struct liteocr_infer_option;

// LatexOCR 后端：PP-FormulaNet_plus-S/M 公式识别（图像 -> LaTeX 字符串）
// 三段式 ncnn 子网：encoder（PPHGNetV2 视觉编码）+ embed（token/位置嵌入）
//                  + decoder（含 KV-cache 的 MBart 解码器，greedy 生成）
struct liteocr_latexocr {
    ncnn::Net encoder_net;
    ncnn::Net embed_net;
    ncnn::Net decoder_net;

    // 词表：第 N 行对应 id=N 的 token 字符串（ByteLevel BPE 编码后的 unicode 形式）
    std::vector<std::string> id_to_token;

    // 自注意力和交叉注意力 KV-cache：各 num_layers 对 (k, v)
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> self_kv_cache;
    std::vector<std::pair<ncnn::Mat, ncnn::Mat>> cross_kv_cache;

    bool loaded = false;

    // 生成 / 模型常量
    int decoder_start_token_id = 0;  // BOS (<s>)
    int eos_token_id = 2;            // </s>
    int max_new_tokens = 1024;
    int vocab_size = 50000;
    int num_layers = 2;              // 带 KV-cache 的自注意力层数
    int parallel_step = 3;           // PP-FormulaNet_plus-S 并行解码步数
    int input_size = 384;            // 编码器输入边长
    int vision_tokens = 144;         // 视觉 token 数
    int hidden_size = 2048;          // 视觉 token 维度
};

// 加载 encoder/embed/decoder 三个 ncnn 子网与 vocab.txt 词表
// 成功返回 true。任一文件加载失败返回 false。
bool liteocr_latexocr_load_model(liteocr_latexocr* m,
    const char* encoder_param, const char* encoder_bin,
    const char* embed_param, const char* embed_bin,
    const char* decoder_param, const char* decoder_bin,
    const char* vocab_path,
    const liteocr_infer_option& opt);

// 识别公式图像，返回 LaTeX 字符串（已做 ByteLevel 解码与 LaTeX 后处理）
std::string liteocr_latexocr_recognize(liteocr_latexocr* m, const liteocr_image& image);
