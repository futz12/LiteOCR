"""最基础的 OCR 示例：加载预设并识别图片文件。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import liteocr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteOCR 基础图片识别示例")
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

    print(f"加载预设：{args.preset}")
    engine = liteocr.Engine()
    engine.load_preset(args.preset, model_dir=args.model_dir)

    print(f"识别图片：{image_path}")
    result = engine.recognize(str(image_path))

    print(f"检测到 {len(result.boxes)} 个文本框，识别出 {len(result.lines)} 行文本：\n")
    for i, line in enumerate(result.lines, 1):
        print(f"{i}. {line.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
