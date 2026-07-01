# LiteOCR 文档

本文档基于当前 `include/liteocr.h` 暴露的 C API 编写。

## 文档导航

- [构建与运行](build.md)
- [C API 参考](c-api.md)
- [使用示例](examples.md)
- [Python 封装](python.md)

## API 分层

LiteOCR 当前主要提供四类能力：

- OCR 整体引擎：检测文本框并识别文本。
- 表格引擎：结合 OCR 结果生成表格 HTML 和单元格区域。
- 底层模型组件：单独调用检测、识别、方向分类、UVDoc、SLANet。
- 图像处理和轮廓工具：图像读写、颜色转换、缩放、旋转、阈值、透视变换和轮廓处理。

公共接口都在 `include/liteocr.h` 中声明。

## 基本约定

- 返回 `0` 通常表示成功，返回非 `0` 表示失败。
- 由 LiteOCR 分配并返回给调用方的内存必须用对应的 `liteocr_free_*` 函数释放。
- 图像通道约定为 `1=Gray`、`3=BGR`、`4=BGRA`。
- `liteocr_infer_option_t.gpu_device_id = -1` 表示 CPU 推理。
- `liteocr_infer_option_t.num_threads` 可设置 CPU 推理线程数；小于等于 0 时使用库内默认值。

