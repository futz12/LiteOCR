# 构建与运行

## 依赖

LiteOCR 使用 CMake 构建，并默认使用仓库中的 `ncnn` 子模块。

```bash
git submodule update --init --recursive
```

## 准备模型

模型文件不随仓库分发。运行示例或测试前，需从镜像下载所需模型到 `models/` 目录：

```
https://mirrors.sdu.edu.cn/ncnn_modelzoo/liteocr/
```

下载说明见 [`models/README.md`](../models/README.md)。仅构建库本身不需要模型文件。

## CMake 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Windows 使用 Visual Studio 生成器时，可执行文件通常输出到：

```text
build/Release/
```

## 测试

如果构建目录已生成 CTest 测试，可运行：

```bash
ctest --test-dir build -C Release --output-on-failure
```

部分模型测试依赖本地测试图片，例如 `test_line.png`。缺少测试图片时这类测试可能无法完整运行。

## GPU / Vulkan 支持

LiteOCR 默认关闭 ncnn 的 Vulkan/GPU 支持，以保持生成的 wheel 和 CI 产物不依赖 Vulkan loader/runtime。如需启用 GPU 推理，在配置时打开：

```bash
cmake -S . -B build -DLITEOCR_ENABLE_VULKAN=ON
cmake --build build --config Release
```

注意：
- 仅当系统存在可用的 Vulkan 设备时，`gpu_device_id` 才会生效。
- 如果编译时关闭了 `LITEOCR_ENABLE_VULKAN`，但运行时仍传入了 `gpu_device_id != -1`，LiteOCR 会打印警告并自动回退到 CPU 推理。

通过 `setup.py` / `pip` 构建时，可用环境变量控制：

```bash
# Windows
set LITEOCR_ENABLE_VULKAN=1
pip install .

# Linux / macOS
LITEOCR_ENABLE_VULKAN=1 pip install .
```

## 使用系统 ncnn

如果本机已安装 ncnn，可以在配置时启用系统 ncnn：

```bash
cmake -S . -B build -DUSE_SYSTEM_NCNN=ON
cmake --build build --config Release
```

当同时开启 `-DLITEOCR_ENABLE_VULKAN=ON` 时，若系统 ncnn 未携带 Vulkan 支持，LiteOCR 会自动回退到 bundled ncnn。

## 裁剪 ncnn 算子体积

LiteOCR 默认会在配置阶段分析 `models/*.param` 模型文件，并只编译当前模型以及 ncnn 核心操作所需的 ncnn 算子，从而减小 ncnn 静态库和最终可执行文件的体积。

> `models/` 目录为空（尚未下载模型）时构建不会失败：此时仅使用硬编码的默认算子白名单。

相关 CMake 选项：

- `LITEOCR_TRIM_NCNN_LAYERS`（默认 `ON`）：是否开启 ncnn 算子裁剪。
- `LITEOCR_TRIM_NCNN_LAYERS_FROM_MODELS`（默认 `ON`）：是否从 `models/*.param` 自动分析所需算子并合并到白名单。
- `LITEOCR_EXTRA_NCNN_LAYERS`：手动追加需要保留的算子，例如 `"foo;bar"`。

示例：

```bash
# 关闭裁剪，编译完整 ncnn
cmake -S . -B build -DLITEOCR_TRIM_NCNN_LAYERS=OFF

# 使用硬编码白名单，不自动扫描 models 目录
cmake -S . -B build -DLITEOCR_TRIM_NCNN_LAYERS_FROM_MODELS=OFF

# 在自动分析的基础上额外保留某些算子
cmake -S . -B build -DLITEOCR_EXTRA_NCNN_LAYERS="foo;bar"
```

当使用系统 ncnn（`USE_SYSTEM_NCNN=ON`）时，裁剪选项不会生效，因为无法重新编译已安装的系统库。

## 链接库

构建后主库目标为 `LiteOCR`。C/C++ 应用需要：

- 包含头文件目录：`include/`
- 链接 `LiteOCR`
- 链接或可访问 ncnn 依赖

