"""文档方向分类示例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import liteocr


# 预设名到模型文件名的映射（示例使用 PP-LCNet_doc_ori）
DOC_ORI_PRESET = "PP-LCNet_doc_ori"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteOCR 文档方向分类示例")
    parser.add_argument(
        "image",
        nargs="?",
        default="test.png",
        help="待分类图片路径（默认：test.png）",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="模型文件存放目录（默认：models）",
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"图片不存在：{image_path}", file=sys.stderr)
        return 1

    # 自动下载方向分类模型
    paths = liteocr.download_orientation_preset(DOC_ORI_PRESET, model_dir=args.model_dir)

    cls = liteocr.DocOrientation()
    cls.load_model(paths["ori_param"], paths["ori_bin"])

    image = liteocr.load_image(str(image_path))
    label = cls.forward(image)

    # PP-LCNet_doc_ori 通常输出 0/1/2/3，分别对应 0°/90°/180°/270°
    angle = label * 90
    print(f"文档方向标签：{label}，对应旋转角度：{angle}°")

    return 0


if __name__ == "__main__":
    sys.exit(main())
