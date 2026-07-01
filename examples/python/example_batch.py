"""批量识别一个目录下的所有图片。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import liteocr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteOCR 批量图片识别示例")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="待识别图片所在目录（默认：当前目录）",
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
    parser.add_argument(
        "--ext",
        default="png,jpg,jpeg,bmp",
        help="要识别的图片扩展名，逗号分隔（默认：png,jpg,jpeg,bmp）",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"目录不存在：{input_dir}", file=sys.stderr)
        return 1

    exts = {f".{e.strip().lstrip('.').lower()}" for e in args.ext.split(",")}
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        print(f"在 {input_dir} 中未找到匹配的图片（扩展名：{exts}）")
        return 0

    print(f"加载预设：{args.preset}")
    engine = liteocr.Engine()
    engine.load_preset(args.preset, model_dir=args.model_dir)

    print(f"共发现 {len(images)} 张图片，开始批量识别...\n")
    total_time = 0.0
    for img_path in images:
        t0 = time.perf_counter()
        result = engine.recognize(str(img_path))
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        texts = [line.text for line in result.lines]
        print(f"[{elapsed:.3f}s] {img_path.name}")
        for text in texts:
            print(f"    - {text}")

    print(f"\n全部完成，总耗时：{total_time:.3f}s，平均每张：{total_time / len(images):.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
