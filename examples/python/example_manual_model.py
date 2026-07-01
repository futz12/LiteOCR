"""手动指定模型路径加载，而不是使用预设自动下载。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import liteocr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteOCR 手动加载模型示例")
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

    engine = liteocr.Engine()
    engine.load_model(
        det_param=model_dir / "PP-OCRv5_mobile_det.param",
        det_bin=model_dir / "PP-OCRv5_mobile_det.bin",
        rec_param=model_dir / "PP-OCRv5_mobile_rec.param",
        rec_bin=model_dir / "PP-OCRv5_mobile_rec.bin",
        vocab=model_dir / "PP-OCRv5_vocab.txt",
        # 如需方向分类模型，取消下面两行注释：
        # ori_param=model_dir / "PP-LCNet_x1_0_textline_ori.param",
        # ori_bin=model_dir / "PP-LCNet_x1_0_textline_ori.bin",
    )

    result = engine.recognize(str(image_path))
    print(f"识别到 {len(result.lines)} 行文本：")
    for i, line in enumerate(result.lines, 1):
        print(f"{i}. {line.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
