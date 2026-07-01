"""单独使用 Detector 和 Recognizer 底层组件示例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import liteocr
from liteocr import COLOR_BGR, cvt_color


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteOCR 检测器+识别器组件示例")
    parser.add_argument(
        "image",
        nargs="?",
        default="test.png",
        help="待识别图片路径（默认：test.png）",
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

    model_dir = Path(args.model_dir)

    print("加载图片...")
    image = liteocr.load_image(str(image_path))
    print(f"  尺寸：{image.width}x{image.height}, 通道：{image.channels}")

    print("\n加载检测器...")
    detector = liteocr.Detector()
    detector.load_model(
        model_dir / "PP-OCRv5_mobile_det.param",
        model_dir / "PP-OCRv5_mobile_det.bin",
    )
    prob = detector.forward(image)
    print(f"  检测输出尺寸：{prob.width}x{prob.height}")

    print("\n加载识别器...")
    recognizer = liteocr.Recognizer()
    recognizer.load_model(
        model_dir / "PP-OCRv5_mobile_rec.param",
        model_dir / "PP-OCRv5_mobile_rec.bin",
    )

    # 为演示，直接用整张图作为识别输入；真实场景需要把检测到的文本框crop出来。
    print("\n对整图进行识别（仅演示）...")
    probs = recognizer.forward(image)
    print(f"  识别输出尺寸：{probs.width}x{probs.height}x{probs.channels}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
