#include "liteocr_latexocr.h"
#include "liteocr_engine.h"
#include "liteocr_imgproc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

// ========== ByteLevel BPE 字节<->unicode 映射 ==========

// 构建 unicode code point -> byte 的反向映射表（-1 表示无映射）
// 对应 Python: _bytes_to_unicode() 后取 {v:k}
static const std::vector<int>& get_unicode_to_byte() {
    static const std::vector<int> uni_to_byte = []() {
        std::vector<int> v(512, -1);

        // byte -> code point 正向表
        int byte_to_cp[256];
        for (int i = 0; i < 256; ++i) byte_to_cp[i] = -1;
        std::vector<int> bs, cs;
        // 可打印范围：33..126, 161..172, 174..255（映射到自身）
        for (int b = 33; b <= 126; ++b) { bs.push_back(b); cs.push_back(b); }
        for (int b = 161; b <= 172; ++b) { bs.push_back(b); cs.push_back(b); }
        for (int b = 174; b <= 255; ++b) { bs.push_back(b); cs.push_back(b); }
        // 其余字节映射到 256+n
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool found = false;
            for (size_t i = 0; i < bs.size(); ++i) {
                if (bs[i] == b) { found = true; break; }
            }
            if (!found) { bs.push_back(b); cs.push_back(256 + n); n += 1; }
        }
        for (size_t i = 0; i < bs.size(); ++i) byte_to_cp[bs[i]] = cs[i];

        // 反向：cp -> byte
        for (int b = 0; b < 256; ++b) {
            int cp = byte_to_cp[b];
            if (cp >= 0 && cp < (int)v.size()) v[cp] = b;
        }
        return v;
    }();
    return uni_to_byte;
}

// 将 UTF-8 字符串解码为 unicode code point 序列
static std::vector<int> utf8_to_codepoints(const std::string& s) {
    std::vector<int> cps;
    cps.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        int cp = 0;
        int advance = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F; advance = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F; advance = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07; advance = 4;
        } else {
            cp = c; advance = 1;
        }
        for (int j = 1; j < advance && i + (size_t)j < s.size(); ++j) {
            unsigned char cc = (unsigned char)s[i + (size_t)j];
            if ((cc & 0xC0) == 0x80) cp = (cp << 6) | (cc & 0x3F);
            else { advance = j; break; }
        }
        cps.push_back(cp);
        i += (size_t)advance;
    }
    return cps;
}

// ByteLevel BPE 解码：token id 序列 -> LaTeX 字符串
// 跳过特殊 token（id 0..22），拼接 token 字符串后按 unicode->byte 反映射回原始字节，再 UTF-8 解码
static std::string bytelevel_decode(const liteocr_latexocr* m, const std::vector<int>& ids) {
    const std::vector<int>& uni_to_byte = get_unicode_to_byte();

    std::string concatenated;
    for (int id : ids) {
        if (id < 0 || id >= (int)m->id_to_token.size()) continue;
        if (id <= 22) continue;  // 特殊 token 全部跳过
        concatenated += m->id_to_token[id];
    }

    // 每个 code point 映射回单字节
    std::vector<unsigned char> bytes;
    std::vector<int> cps = utf8_to_codepoints(concatenated);
    bytes.reserve(cps.size());
    for (int cp : cps) {
        int b = -1;
        if (cp >= 0 && cp < (int)uni_to_byte.size()) b = uni_to_byte[cp];
        if (b < 0) b = cp;  // 回退（与 Python ord(ch) 一致，截断到字节）
        bytes.push_back((unsigned char)(b & 0xFF));
    }

    return std::string(bytes.begin(), bytes.end());
}

// ========== logits argmax ==========

static int argmax_logits(const ncnn::Mat& logits, int vocab_size) {
    const float* p = (const float*)logits.data;
    int n = vocab_size;
    if (n > logits.w) n = logits.w;
    if (n <= 0) return 0;
    int best = 0;
    float bestv = p[0];
    for (int i = 1; i < n; ++i) {
        if (p[i] > bestv) { bestv = p[i]; best = i; }
    }
    return best;
}

// 对 logits 的第 row 行做 argmax（用于并行解码多行 logits）
static int argmax_logits_row(const ncnn::Mat& logits, int row, int vocab_size) {
    int n = vocab_size;
    if (n > logits.w) n = logits.w;
    if (n <= 0) return 0;
    const float* p = (const float*)logits.data + (size_t)row * logits.w;
    int best = 0;
    float bestv = p[0];
    for (int i = 1; i < n; ++i) {
        if (p[i] > bestv) { bestv = p[i]; best = i; }
    }
    return best;
}

// ========== 图像预处理：任意 liteocr_image -> encoder 输入 Mat(384,384,1) ==========

// 取灰度 uint8 行缓冲（step 返回行步长，单位字节）
static std::vector<uint8_t> to_gray_u8(const liteocr_image& img, int& step) {
    step = img.width;
    std::vector<uint8_t> gray((size_t)img.width * (size_t)img.height, 0);
    if (img.type == LITEOCR_IMAGE_U8C1) {
        for (int y = 0; y < img.height; ++y) {
            const uint8_t* src = img.data + (size_t)y * img.stride;
            uint8_t* dst = gray.data() + (size_t)y * step;
            std::memcpy(dst, src, (size_t)img.width);
        }
    } else if (img.type == LITEOCR_IMAGE_U8C3) {
        // stbi_load 返回 RGB；Python 用 cv2.cvtColor(BGR2GRAY)。
        // OpenCV BGR2GRAY: Y = 0.114*B + 0.587*G + 0.299*R
        // RGB 像素 [R,G,B] 对应 OpenCV BGR 的 B=R/G=G/R=B，因此：
        // Y = 0.114*R + 0.587*G + 0.299*B
        for (int y = 0; y < img.height; ++y) {
            const uint8_t* src = img.data + (size_t)y * img.stride;
            uint8_t* dst = gray.data() + (size_t)y * step;
            for (int x = 0; x < img.width; ++x) {
                float r = src[x * 3 + 0];
                float g = src[x * 3 + 1];
                float b = src[x * 3 + 2];
                float y_val = 0.114f * r + 0.587f * g + 0.299f * b;
                if (y_val < 0.f) y_val = 0.f;
                if (y_val > 255.f) y_val = 255.f;
                dst[x] = (uint8_t)(y_val + 0.5f);
            }
        }
    } else if (img.type == LITEOCR_IMAGE_U8C4) {
        for (int y = 0; y < img.height; ++y) {
            const uint8_t* src = img.data + (size_t)y * img.stride;
            uint8_t* dst = gray.data() + (size_t)y * step;
            for (int x = 0; x < img.width; ++x) {
                float r = src[x * 4 + 0];
                float g = src[x * 4 + 1];
                float b = src[x * 4 + 2];
                float y_val = 0.114f * r + 0.587f * g + 0.299f * b;
                if (y_val < 0.f) y_val = 0.f;
                if (y_val > 255.f) y_val = 255.f;
                dst[x] = (uint8_t)(y_val + 0.5f);
            }
        }
    } else if (img.type == LITEOCR_IMAGE_F32C1) {
        const float* pf = (const float*)img.data;
        int fstep = img.stride / (int)sizeof(float);
        if (fstep <= 0) fstep = img.width;
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                float v = pf[(size_t)y * fstep + (size_t)x];
                if (v < 0.f) v = 0.f;
                if (v > 255.f) v = 255.f;
                gray[(size_t)y * step + (size_t)x] = (uint8_t)(v + 0.5f);
            }
        }
    }
    return gray;
}

// UniMERNet eval 风格预处理：
// 1. 转灰度；2. 裁剪白边；3. 保持长宽比 resize（短边 -> target_size）；
// 4. 白色背景 pad 到 target_size x target_size；5. mean=0.7931, std=0.1738 归一化
static ncnn::Mat preprocess_encoder_input(const liteocr_image& image, int target_size) {
    int src_step = 0;
    std::vector<uint8_t> gray = to_gray_u8(image, src_step);

    int w = image.width;
    int h = image.height;

    // 1. 裁剪白边（与 UniMERNet FormulaImageBaseProcessor.crop_margin 一致）
    int min_val = 255;
    int max_val = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = gray.data() + (size_t)y * src_step;
        for (int x = 0; x < w; ++x) {
            int v = row[x];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
    }
    if (max_val > min_val) {
        int min_x = w, min_y = h, max_x = -1, max_y = -1;
        float inv_range = 255.0f / (float)(max_val - min_val);
        for (int y = 0; y < h; ++y) {
            const uint8_t* row = gray.data() + (size_t)y * src_step;
            for (int x = 0; x < w; ++x) {
                float normalized = (float)(row[x] - min_val) * inv_range;
                if (normalized < 200.0f) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }
        if (max_x >= min_x && max_y >= min_y) {
            int cw = max_x - min_x + 1;
            int ch = max_y - min_y + 1;
            std::vector<uint8_t> cropped((size_t)cw * (size_t)ch);
            for (int y = 0; y < ch; ++y) {
                const uint8_t* src = gray.data() + (size_t)(min_y + y) * src_step + min_x;
                uint8_t* dst = cropped.data() + (size_t)y * cw;
                std::memcpy(dst, src, (size_t)cw);
            }
            gray.swap(cropped);
            w = cw;
            h = ch;
            src_step = cw;
        }
    }

    // 2. 保持长宽比 resize：短边缩放到 target_size
    float scale_h = (float)target_size / (float)h;
    float scale_w = (float)target_size / (float)w;
    float scale = scale_h < scale_w ? scale_h : scale_w;
    int new_h = (int)(h * scale + 0.5f);
    int new_w = (int)(w * scale + 0.5f);
    if (new_h < 1) new_h = 1;
    if (new_w < 1) new_w = 1;

    std::vector<uint8_t> resized((size_t)new_w * (size_t)new_h, 255);
    liteocr_resize(gray.data(), w, h, src_step, 1,
                   resized.data(), new_w, new_h, new_w);

    // 3. 白色背景 pad 到 target_size x target_size（居中）
    int pad_top = (target_size - new_h) / 2;
    int pad_bottom = target_size - new_h - pad_top;
    int pad_left = (target_size - new_w) / 2;
    int pad_right = target_size - new_w - pad_left;

    ncnn::Mat input(target_size, target_size, 1);
    float* pf = (float*)input.data;
    const float mean = 0.7931f;
    const float std = 0.1738f;
    const float inv_255 = 1.0f / 255.0f;

    for (int y = 0; y < target_size; ++y) {
        for (int x = 0; x < target_size; ++x) {
            uint8_t v;
            if (y < pad_top || y >= target_size - pad_bottom ||
                x < pad_left || x >= target_size - pad_right) {
                v = 255;  // 白色填充
            } else {
                int sy = y - pad_top;
                int sx = x - pad_left;
                v = resized[(size_t)sy * new_w + (size_t)sx];
            }
            pf[y * target_size + x] = ((float)v * inv_255 - mean) / std;
        }
    }
    return input;
}

// ========== int32 ncnn::Mat 构造（用于 embed 输入 token/位置 id） ==========

static ncnn::Mat make_int32_mat(const int* data, int n) {
    // ncnn::Mat(int w, int h, void* data, size_t elemsize=4u)
    // sizeof(int)==4，默认 elemsize=4 与 int32 一致；clone 后持有独立内存
    ncnn::Mat m(n, 1, (void*)data);
    return m.clone();
}

// ========== LaTeX 后处理（参考 RapidDoc / UniMERNet） ==========

static std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r' || s[a] == '\f' || s[a] == '\v')) ++a;
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\t' || s[b-1] == '\n' || s[b-1] == '\r' || s[b-1] == '\f' || s[b-1] == '\v')) --b;
    return std::string(s, a, b - a);
}

static std::string remove_all_substr(std::string s, const std::string& sub) {
    if (sub.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(sub, pos)) != std::string::npos) {
        s.erase(pos, sub.size());
    }
    return s;
}

// 所有 needle 子串替换为 repl
static std::string replace_all_substr(std::string s, const std::string& needle, const std::string& repl) {
    if (needle.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        s.replace(pos, needle.size(), repl);
        pos += repl.size();
    }
    return s;
}

// 检查位置 pos 的字符是否被反斜杠转义
static bool is_escaped(const std::string& s, size_t pos) {
    int backslash_count = 0;
    size_t j = pos;
    while (j > 0 && s[j - 1] == '\\') {
        ++backslash_count;
        --j;
    }
    return (backslash_count % 2) == 1;
}

// 修复无法配对的花括号（删除未匹配的 { 或 }）
static std::string fix_unbalanced_braces(std::string s) {
    std::vector<size_t> stack;
    std::vector<bool> unmatched(s.size(), false);
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '{' || s[i] == '}') {
            if (is_escaped(s, i)) continue;
            if (s[i] == '{') {
                stack.push_back(i);
            } else if (!stack.empty()) {
                stack.pop_back();
            } else {
                unmatched[i] = true;
            }
        }
    }
    for (size_t idx : stack) unmatched[idx] = true;
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (!unmatched[i]) out += s[i];
    }
    return out;
}

// 查找从 pos 开始、depth 层花括号组的结束位置（返回 '}' 的索引）
static size_t find_group_end(const std::string& s, size_t pos, int depth) {
    int current_depth = depth;
    for (size_t i = pos; i < s.size(); ++i) {
        if (s[i] == '{' && !is_escaped(s, i)) {
            ++current_depth;
        } else if (s[i] == '}' && !is_escaped(s, i)) {
            --current_depth;
            if (current_depth < depth) return i;
        }
    }
    return std::string::npos;
}

// 修复 \left/\right 不在同一组的情况
static std::string fix_left_right_pairs(std::string s) {
    std::vector<size_t> brace_stack;
    std::vector<std::pair<size_t, int>> left_stack;  // (位置, 花括号深度)
    std::vector<std::tuple<size_t, size_t, size_t>> adjustments;  // (start, end, target)

    size_t i = 0;
    while (i < s.size()) {
        if (i > 0 && s[i - 1] == '\\' && is_escaped(s, i)) {
            ++i;
            continue;
        }

        if (i + 5 < s.size() && s.compare(i, 5, "\\left") == 0) {
            left_stack.push_back(std::make_pair(i, (int)brace_stack.size()));
            i += 6;  // 跳过 \left 和分隔符
            continue;
        }
        if (i + 6 < s.size() && s.compare(i, 6, "\\right") == 0) {
            if (!left_stack.empty()) {
                size_t left_pos = left_stack.back().first;
                int left_depth = left_stack.back().second;
                left_stack.pop_back();
                if (left_depth != (int)brace_stack.size()) {
                    size_t target = find_group_end(s, left_pos, left_depth);
                    if (target != std::string::npos) {
                        adjustments.push_back(std::make_tuple(i, i + 7, target));
                    }
                }
            }
            i += 7;  // 跳过 \right 和分隔符
            continue;
        }

        if (s[i] == '{' && !is_escaped(s, i)) {
            brace_stack.push_back(i);
        } else if (s[i] == '}' && !is_escaped(s, i)) {
            if (!brace_stack.empty()) brace_stack.pop_back();
        }
        ++i;
    }

    if (adjustments.empty()) return s;

    // 从后向前处理，避免索引变化
    std::sort(adjustments.begin(), adjustments.end(),
        [](const std::tuple<size_t, size_t, size_t>& a, const std::tuple<size_t, size_t, size_t>& b) {
            return std::get<0>(a) > std::get<0>(b);
        });

    for (const auto& adj : adjustments) {
        size_t start = std::get<0>(adj);
        size_t end = std::get<1>(adj);
        size_t target = std::get<2>(adj);
        std::string right_part = s.substr(start, end - start);
        s.erase(start, end - start);
        if (target <= s.size()) s.insert(target, right_part);
    }
    return s;
}

// 检查 cmd 后是否跟着有效分隔符；否则返回 cmd + "."
static std::string fix_delim_suffix(const std::string& s, size_t pos,
                                    const std::vector<std::string>& valid_delims,
                                    const std::string& cmd) {
    if (pos >= s.size()) return cmd + ".";
    for (const std::string& d : valid_delims) {
        if (s.compare(pos, d.size(), d) == 0) return cmd + d;
    }
    return cmd + ".";
}

// 修复 \left/\right：补齐缺失分隔符、平衡数量、配对位置
static std::string fix_latex_left_right(std::string s) {
    static const std::vector<std::string> valid_delims = {
        "(", ")", "[", "]", "{", "}", "/", "|",
        "\\{", "\\}", "\\lceil", "\\rceil", "\\lfloor",
        "\\rfloor", "\\backslash", "\\uparrow", "\\downarrow",
        "\\Uparrow", "\\Downarrow", "\\|", "\\."
    };

    static const std::regex left_pattern(R"(\\left(\S?))");
    static const std::regex right_pattern(R"(\\right(\S?))");

    // 手动替换，因为 C++11 std::regex_replace 不支持 lambda 回调
    auto replace_with_callback = [&](const std::regex& re, bool is_left) -> void {
        std::string result;
        std::sregex_iterator it(s.begin(), s.end(), re);
        std::sregex_iterator end;
        size_t last = 0;
        for (; it != end; ++it) {
            const std::smatch& m = *it;
            size_t pos = (size_t)m.position();
            size_t len = (size_t)m.length();
            result.append(s, last, pos - last);
            std::string cmd = is_left ? "\\left" : "\\right";
            size_t rest_pos = pos + cmd.size();
            result += fix_delim_suffix(s, rest_pos, valid_delims, cmd);
            last = pos + len;
        }
        if (last < s.size()) result.append(s, last, std::string::npos);
        s.swap(result);
    };

    replace_with_callback(left_pattern, true);
    replace_with_callback(right_pattern, false);

    // 统计 \left/\right 数量（不匹配 \lefteqn、\rightarrow 等）
    static const std::regex left_count_re(R"(\\left(?![a-zA-Z]))");
    static const std::regex right_count_re(R"(\\right(?![a-zA-Z]))");
    static const std::regex lr_remove_re(R"(\\left\.?|\\right\.?)");

    size_t left_count = std::distance(std::sregex_iterator(s.begin(), s.end(), left_count_re), std::sregex_iterator());
    size_t right_count = std::distance(std::sregex_iterator(s.begin(), s.end(), right_count_re), std::sregex_iterator());

    if (left_count == right_count) {
        return fix_left_right_pairs(s);
    } else {
        return std::regex_replace(s, lr_remove_re, "");
    }
}

// 修复数学环境标签（\begin/\end 不匹配时补齐）
static std::string fix_latex_environments(std::string s) {
    static const char* env_types[] = {
        "array", "matrix", "pmatrix", "bmatrix", "vmatrix",
        "Bmatrix", "Vmatrix", "cases", "aligned", "gathered", "align", "align*"
    };
    for (const char* env : env_types) {
        std::string begin_str = std::string("\\begin{") + env + "}";
        std::string end_str = std::string("\\end{") + env + "}";
        std::string begin_re_str = std::string("\\\\begin\\{") + env + "\\}";
        std::regex begin_re(begin_re_str);
        std::regex end_re(std::string("\\\\end\\{") + env + "\\}");

        size_t begin_count = std::distance(std::sregex_iterator(s.begin(), s.end(), begin_re), std::sregex_iterator());
        size_t end_count = std::distance(std::sregex_iterator(s.begin(), s.end(), end_re), std::sregex_iterator());

        if (begin_count != end_count) {
            if (end_count > begin_count) {
                std::regex format_re(std::string("\\\\begin\\{") + env + "\\}\\{([^}]*)\\}");
                std::smatch fm;
                std::string format_str = "{c}";
                if (std::regex_search(s, fm, format_re)) {
                    format_str = std::string("{") + fm.str(1) + "}";
                } else if (std::string(env) != "array") {
                    format_str = "";
                }
                std::string begin_cmd = begin_str + format_str + " ";
                for (size_t k = 0; k < end_count - begin_count; ++k) {
                    s = begin_cmd + s;
                }
            } else {
                std::string end_cmd = std::string(" \\end{") + env + "}";
                for (size_t k = 0; k < begin_count - end_count; ++k) {
                    s += end_cmd;
                }
            }
        }
    }
    return s;
}

// 移除不必要的 \up 前缀（保留少数例外）
static std::string remove_up_commands(std::string s) {
    static const std::regex up_re(R"(\\up([a-zA-Z]+))");
    std::string result;
    std::sregex_iterator it(s.begin(), s.end(), up_re);
    std::sregex_iterator end;
    size_t last = 0;
    for (; it != end; ++it) {
        const std::smatch& m = *it;
        size_t pos = (size_t)m.position();
        size_t len = (size_t)m.length();
        result.append(s, last, pos - last);
        std::string word = m.str(1);
        if (word == "arrow" || word == "downarrow" || word == "lus" || word == "silon") {
            result.append(m.str(0));
        } else {
            result += '\\';
            result += word;
        }
        last = pos + len;
    }
    if (last < s.size()) result.append(s, last, std::string::npos);
    return result;
}

// 移除不支持的命令
static std::string remove_unsupported_commands(std::string s) {
    static const std::regex unsupported_re(R"(\\(?:lefteqn|boldmath|ensuremath|centering|textsubscript|sides|textsl|textcent|emph|protect|null))");
    return std::regex_replace(s, unsupported_re, "");
}

// 应用常见命令替换
static std::string apply_replacements(std::string s) {
    static const std::pair<std::regex, std::string> repls[] = {
        {std::regex(R"(\\underbar)"), "\\underline"},
        {std::regex(R"(\\Bar)"), "\\hat"},
        {std::regex(R"(\\Hat)"), "\\hat"},
        {std::regex(R"(\\Tilde)"), "\\tilde"},
        {std::regex(R"(\\slash)"), "/"},
        {std::regex(R"(\\textperthousand)"), "\xe2\x80\xb0"},   // ‰
        {std::regex(R"(\\sun)"), "\xe2\x98\x89"},              // ☉
        {std::regex(R"(\\textunderscore)"), "\\_"},
        {std::regex(R"(\\fint)"), "\xe2\xa8\x8f"},             // ⨏
        {std::regex(R"(\\up )"), "\\ "},
        {std::regex(R"(\\vline = )"), "\\models "},
        {std::regex(R"(\\vDash )"), "\\models "},
        {std::regex(R"(\\sq \\sqcup )"), "\\square "},
        {std::regex(R"(\\copyright)"), "\xc2\xa9"},           // ©
    };
    for (const auto& p : repls) {
        s = std::regex_replace(s, p.first, p.second);
    }
    return s;
}

// 处理反斜杠后空格：\x 在需要时变成 \ x
static std::string process_latex(std::string s) {
    static const std::regex backslash_re(R"(\\(.))");
    static const std::string special_chars = "#$%&~_^|\\{} \t\n\r\v\f";
    std::string result;
    std::sregex_iterator it(s.begin(), s.end(), backslash_re);
    std::sregex_iterator end;
    size_t last = 0;
    for (; it != end; ++it) {
        const std::smatch& m = *it;
        size_t pos = (size_t)m.position();
        size_t len = (size_t)m.length();
        result.append(s, last, pos - last);
        char next = m.str(1)[0];
        size_t after = pos + len;
        // 特殊字符或空格：保持原样
        if (special_chars.find(next) != std::string::npos) {
            result.append(m.str(0));
        }
        // \ 后接两个字母命令：保持原样
        else if (after < s.size() && ((s[after] >= 'a' && s[after] <= 'z') || (s[after] >= 'A' && s[after] <= 'Z'))) {
            result.append(m.str(0));
        }
        else {
            result += "\\ ";
            result += next;
        }
        last = pos + len;
    }
    if (last < s.size()) result.append(s, last, std::string::npos);
    return result;
}

// 检查 UTF-8 字符串 s 在位置 i 是否为中文字符（CJK Unified Ideographs）
static bool is_chinese_char(const std::string& s, size_t i) {
    if (i >= s.size()) return false;
    unsigned char c = (unsigned char)s[i];
    // CJK Unified Ideographs: U+4E00 - U+9FFF
    if (c >= 0xE4 && c <= 0xE9) {
        if (i + 2 < s.size()) {
            unsigned char c2 = (unsigned char)s[i + 1];
            unsigned char c3 = (unsigned char)s[i + 2];
            int codepoint = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
            return codepoint >= 0x4E00 && codepoint <= 0x9FFF;
        }
    }
    return false;
}

// 移除中文 \text{...} 包装
static std::string remove_chinese_text_wrapping(std::string s) {
    std::string result;
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, 5, "\\text") == 0 || s.compare(i, 6, "\\text ") == 0) {
            size_t cmd_end = i + 5;
            while (cmd_end < s.size() && (s[cmd_end] == ' ' || s[cmd_end] == '\t')) ++cmd_end;
            if (cmd_end < s.size() && s[cmd_end] == '{') {
                size_t brace_start = cmd_end;
                size_t j = brace_start + 1;
                int depth = 1;
                bool has_chinese = false;
                while (j < s.size() && depth > 0) {
                    if (s[j] == '{' && !is_escaped(s, j)) ++depth;
                    else if (s[j] == '}' && !is_escaped(s, j)) --depth;
                    if (depth > 0 && is_chinese_char(s, j)) has_chinese = true;
                    if (depth > 0) ++j;
                }
                if (depth == 0 && has_chinese) {
                    // 提取 { } 之间的内容，去除首尾空白
                    size_t content_start = brace_start + 1;
                    size_t content_end = j - 1;
                    while (content_start <= content_end &&
                           (s[content_start] == ' ' || s[content_start] == '\t' ||
                            s[content_start] == '\n' || s[content_start] == '\r')) ++content_start;
                    while (content_end > content_start &&
                           (s[content_end] == ' ' || s[content_end] == '\t' ||
                            s[content_end] == '\n' || s[content_end] == '\r')) --content_end;
                    result.append(s, content_start, content_end - content_start + 1);
                    i = j;
                    continue;
                }
            }
        }
        result += s[i];
        ++i;
    }
    result = remove_all_substr(result, "\"");
    return result;
}

// 综合 LaTeX 后处理（参考 RapidDoc）
static std::string latex_rm_whitespace(std::string s) {
    s = fix_unbalanced_braces(s);
    s = fix_latex_left_right(s);
    s = fix_latex_environments(s);
    s = remove_up_commands(s);
    s = remove_unsupported_commands(s);
    s = apply_replacements(s);
    s = process_latex(s);
    // \qquad 后补空格
    static const std::regex qquad_re(R"(\\qquad(?!\s))");
    s = std::regex_replace(s, qquad_re, "\\qquad ");
    // 去掉末尾反斜杠
    while (!s.empty() && s.back() == '\\') s.pop_back();
    return trim_ws(s);
}

// LaTeX 后处理入口
static std::string post_process_formula(std::string s) {
    // 1. 去除特殊 token 字符串
    s = remove_all_substr(s, "[BOS]");
    s = remove_all_substr(s, "[EOS]");
    s = remove_all_substr(s, "[PAD]");
    s = remove_all_substr(s, "<s>");
    s = remove_all_substr(s, "</s>");
    s = remove_all_substr(s, "<pad>");

    // 2. 移除中文 \text{} 包装
    s = remove_chinese_text_wrapping(s);

    // 3. RapidDoc 风格结构修复
    s = latex_rm_whitespace(s);

    return trim_ws(s);
}

static bool configure_decoder_variant(
    liteocr_latexocr* m, const char* decoder_param) {
    std::ifstream file(decoder_param);
    if (!file.is_open()) return false;

    int mha_count = 0;
    int masked_mha_count = 0;
    int cached_mha_count = 0;
    int decoder_hidden_size = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string type;
        std::string name;
        int bottom_count = 0;
        int top_count = 0;
        if (!(stream >> type >> name >> bottom_count >> top_count)) continue;
        if (type != "MultiHeadAttention") continue;

        ++mha_count;
        std::string field;
        for (int i = 0; i < bottom_count + top_count; ++i) {
            if (!(stream >> field)) return false;
        }
        while (stream >> field) {
            if (field.compare(0, 2, "0=") == 0) {
                decoder_hidden_size = std::atoi(field.c_str() + 2);
            } else if (field == "5=1") {
                ++masked_mha_count;
            } else if (field == "7=1") {
                ++cached_mha_count;
            }
        }
    }

    if (mha_count == 0 || masked_mha_count * 2 != mha_count ||
        cached_mha_count != mha_count) {
        return false;
    }

    m->num_layers = masked_mha_count;
    if (m->num_layers == 2 && decoder_hidden_size == 384) {
        m->parallel_step = 3;
        m->max_new_tokens = 1024;
    } else if (m->num_layers == 6 && decoder_hidden_size == 512) {
        m->parallel_step = 1;
        m->max_new_tokens = 2560;
    } else {
        return false;
    }
    return true;
}

}  // namespace

// ========== 模型加载 ==========

bool liteocr_latexocr_load_model(liteocr_latexocr* m,
    const char* encoder_param, const char* encoder_bin,
    const char* embed_param, const char* embed_bin,
    const char* decoder_param, const char* decoder_bin,
    const char* vocab_path,
    const liteocr_infer_option& opt) {
    if (!m) return false;
    m->loaded = false;

    if (!configure_decoder_variant(m, decoder_param)) return false;

    liteocr_apply_net_options(m->encoder_net, opt);
    liteocr_apply_net_options(m->embed_net, opt);
    liteocr_apply_net_options(m->decoder_net, opt);

    if (m->encoder_net.load_param(encoder_param) != 0) return false;
    if (m->encoder_net.load_model(encoder_bin) != 0) return false;
    if (m->embed_net.load_param(embed_param) != 0) return false;
    if (m->embed_net.load_model(embed_bin) != 0) return false;
    if (m->decoder_net.load_param(decoder_param) != 0) return false;
    if (m->decoder_net.load_model(decoder_bin) != 0) return false;

    // 加载词表：每行一个 token 字符串，行号 = id
    m->id_to_token.clear();
    std::ifstream vocabFile(vocab_path);
    if (!vocabFile.is_open()) return false;
    std::string line;
    while (std::getline(vocabFile, line)) {
        // 去除 Windows 行尾 \r
        if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
        m->id_to_token.push_back(line);
    }

    if ((int)m->id_to_token.size() != m->vocab_size) return false;
    m->self_kv_cache.clear();
    m->self_kv_cache.resize((size_t)m->num_layers);
    m->cross_kv_cache.clear();
    m->cross_kv_cache.resize((size_t)m->num_layers);
    m->loaded = true;
    return true;
}

// ========== 识别 ==========

std::string liteocr_latexocr_recognize(liteocr_latexocr* m, const liteocr_image& image) {
    if (!m || !m->loaded || image.empty()) return std::string();

    // 1. encoder：图像 -> 视觉 token（全程复用作为交叉注意力 K/V 源）
    ncnn::Mat enc_in = preprocess_encoder_input(image, m->input_size);
    ncnn::Mat encoder_out;
    {
        ncnn::Extractor ex = m->encoder_net.create_extractor();
        ex.input("in0", enc_in);
        ex.extract("out0", encoder_out);
    }
    if (encoder_out.empty()) return std::string();

    // 2. 重置 KV-cache
    m->self_kv_cache.assign(
        (size_t)m->num_layers, std::make_pair(ncnn::Mat(), ncnn::Mat()));
    m->cross_kv_cache.assign(
        (size_t)m->num_layers, std::make_pair(ncnn::Mat(), ncnn::Mat()));

    // 3. 并行 greedy 生成（PP-FormulaNet_plus-S 使用 parallel_step=3）
    //    - prefill：输入 P 个 BOS，位置 [0..P-1]，mask (P,P) 全 0（组内双向）
    //    - decode：输入上一步生成的 P 个 token，位置 [cur_len..cur_len+P-1]
    //              mask (kv_len+P, P) 全 0（past 全可见 + 当前组内双向）
    const int P = m->parallel_step;
    std::vector<int> generated;          // 生成的 token（不含 BOS、不含 EOS）
    std::vector<int> input_tokens(P, m->decoder_start_token_id);  // 当步输入 token
    int cur_len = 0;                     // 已缓存 token 数（= 下一个 token 的绝对位置）
    bool stop = false;

    int max_steps = (m->max_new_tokens + P - 1) / P + 1;

    for (int step = 0; step < max_steps && !stop; ++step) {
        bool is_prefill = (step == 0);
        int kv_len = cur_len;

        // 3a. embed：P 个 token id + P 个绝对位置 id
        //     原 param 中 BinaryOp add_0 (pos_id + 2.0) 会破坏 int32（ncnn BinaryOp 把
        //     int32 位模式当 float 读），已从 param 移除。offset=2 在 C++ 侧预加。
        int token_ids[8], pos_ids[8];
        for (int i = 0; i < P; ++i) {
            token_ids[i] = input_tokens[i];
            pos_ids[i] = cur_len + i + 2;  // +2 对应 MBart embed_positions.offset=2
        }
        ncnn::Mat ids_mat = make_int32_mat(token_ids, P);
        ncnn::Mat pos_mat = make_int32_mat(pos_ids, P);
        ncnn::Mat embeds;
        {
            ncnn::Extractor ex = m->embed_net.create_extractor();
            ex.input("in0", ids_mat);
            ex.input("in1", pos_mat);
            ex.extract("out0", embeds);
        }
        if (embeds.empty()) break;

        // 3b. 注意力掩码（加性：0=可看，大负数=屏蔽）
        //     参考 unimernet_head.py _make_causal_mask_parallel：
        //     当 tgt_len == parallel_step 时，mask 全 0（并行组内全双向可见）。
        //     组间因果由 past KV-cache 实现（past 全可见），当前组内 3 个 query 互相可见。
        //     SDPA mask 形状：w=key_len(含past), h=query_len(=P), c=1
        int key_len = kv_len + P;
        ncnn::Mat mask(key_len, P, 1);
        {
            float* pm = (float*)mask.data;
            for (int i = 0; i < key_len * P; ++i) {
                pm[i] = 0.0f;
            }
        }

        // 3c. decoder：输入 encoder_out / embeds / mask（+ 上一步 cache）
        std::vector<std::pair<ncnn::Mat, ncnn::Mat>> next_self_cache(
            (size_t)m->num_layers);
        std::vector<std::pair<ncnn::Mat, ncnn::Mat>> next_cross_cache(
            (size_t)m->num_layers);
        ncnn::Mat logits;
        {
            ncnn::Extractor ex = m->decoder_net.create_extractor();
            ex.input("in0", encoder_out);
            ex.input("in1", embeds);
            ex.input("in2", mask);
            if (!is_prefill) {
                for (int i = 0; i < m->num_layers; ++i) {
                    const std::string suffix = std::to_string(i);
                    ex.input(("self_cache_k" + suffix).c_str(),
                             m->self_kv_cache[(size_t)i].first);
                    ex.input(("self_cache_v" + suffix).c_str(),
                             m->self_kv_cache[(size_t)i].second);
                    ex.input(("cross_cache_k" + suffix).c_str(),
                             m->cross_kv_cache[(size_t)i].first);
                    ex.input(("cross_cache_v" + suffix).c_str(),
                             m->cross_kv_cache[(size_t)i].second);
                }
            }
            for (int i = 0; i < m->num_layers; ++i) {
                const std::string suffix = std::to_string(i);
                if (ex.extract(("out_self_cache_k" + suffix).c_str(),
                               next_self_cache[(size_t)i].first) != 0 ||
                    ex.extract(("out_self_cache_v" + suffix).c_str(),
                               next_self_cache[(size_t)i].second) != 0 ||
                    ex.extract(("out_cross_cache_k" + suffix).c_str(),
                               next_cross_cache[(size_t)i].first) != 0 ||
                    ex.extract(("out_cross_cache_v" + suffix).c_str(),
                               next_cross_cache[(size_t)i].second) != 0) {
                    return std::string();
                }
            }
            if (ex.extract("out0", logits) != 0) return std::string();
        }
        m->self_kv_cache.swap(next_self_cache);
        m->cross_kv_cache.swap(next_cross_cache);
        if (logits.empty()) break;

        // 3d. argmax 每行 logits -> P 个新 token
        cur_len += P;
        int next_tokens[8];
        for (int i = 0; i < P; ++i) {
            next_tokens[i] = argmax_logits_row(logits, i, m->vocab_size);
        }

        // 3e. 检查 EOS 并加入结果（命中 EOS 则停止，不加入 EOS 本身）
        for (int i = 0; i < P; ++i) {
            if (next_tokens[i] == m->eos_token_id) {
                stop = true;
                break;
            }
            generated.push_back(next_tokens[i]);
            if ((int)generated.size() >= m->max_new_tokens) {
                stop = true;
                break;
            }
        }

        // 下一步输入 = 本步生成的 P 个 token
        for (int i = 0; i < P; ++i) input_tokens[i] = next_tokens[i];
    }

    // 4. token id 序列 -> LaTeX 字符串
    std::string text = bytelevel_decode(m, generated);
    text = post_process_formula(text);
    return text;
}
