"""从 NumPy 数组进行 OCR 识别示例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import liteocr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteOCR NumPy 数组识别示例")
    parser.add_argument(
        "image",
        nargs="?",
        default="test.png",
        help="待识别图片路径（默认：test.png）",
    )
    parser.add_argument(
        "--preset",
        default="PP-OCRv5_mobile",
        help="使用的 OCR 预设（默认：PP-OCRv5_mobile）",
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

    # 用 Pillow 或 opencv 读取均可，这里用 LiteOCR 自带接口读取后再转 NumPy。
    img = liteocr.load_image(str(image_path))
    arr = img.to_numpy()
    print(f"图片数组 shape：{arr.shape}, dtype：{arr.dtype}")

    engine = liteocr.Engine()
    engine.load_preset(args.preset, model_dir=args.model_dir)

    print("从 NumPy 数组识别...")
    result = engine.recognize(arr)

    print(f"\n识别结果（共 {len(result.lines)} 行）：")
    for i, line in enumerate(result.lines, 1):
        print(f"{i}. {line.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
