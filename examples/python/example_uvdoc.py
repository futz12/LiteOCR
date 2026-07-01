"""UVDoc 文档畸变校正示例。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import liteocr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteOCR UVDoc 文档畸变校正示例")
    parser.add_argument(
        "image",
        nargs="?",
        default="test.png",
        help="待校正图片路径（默认：test.png）",
    )
    parser.add_argument(
        "--output",
        default="uvdoc_output.png",
        help="校正后图片保存路径（默认：uvdoc_output.png）",
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

    # 请确保 models/ 目录下存在 UVDoc 模型文件；如未下载需先准备。
    uvdoc = liteocr.UVDoc()
    uvdoc.load_model(
        model_dir / "uvdoc.param",
        model_dir / "uvdoc.bin",
    )

    image = liteocr.load_image(str(image_path))
    corrected = uvdoc.forward(image)
    liteocr.imwrite(args.output, corrected)

    print(f"校正完成，已保存到：{args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
