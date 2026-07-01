"""表格识别示例：先 OCR 得到文本框和文本行，再调用 TableEngine。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import liteocr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiteOCR 表格识别示例")
    parser.add_argument(
        "image",
        nargs="?",
        default="test.png",
        help="待识别表格图片路径（默认：test.png）",
    )
    parser.add_argument(
        "--ocr-preset",
        default="PP-OCRv5_mobile",
        help="OCR 预设（默认：PP-OCRv5_mobile）",
    )
    parser.add_argument(
        "--table-preset",
        default="PP-StructureV2_SLANet_plus",
        help="表格结构预设（默认：PP-StructureV2_SLANet_plus）",
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

    print("步骤 1：OCR 识别文本...")
    ocr_engine = liteocr.Engine()
    ocr_engine.load_preset(args.ocr_preset, model_dir=args.model_dir)
    ocr_result = ocr_engine.recognize(str(image_path))
    print(f"  识别到 {len(ocr_result.lines)} 行文本")

    print("步骤 2：表格结构解析...")
    table_engine = liteocr.TableEngine()
    table_engine.load_preset(args.table_preset, model_dir=args.model_dir)
    table_result = table_engine.recognize(str(image_path), ocr_result)

    print(f"\n表格单元格数量：{len(table_result.cells)}")
    print(f"表格结构标记数量：{len(table_result.structure)}")
    print("\n生成的 HTML：")
    print(table_result.html)

    return 0


if __name__ == "__main__":
    sys.exit(main())
