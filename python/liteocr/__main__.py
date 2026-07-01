"""Simple CLI smoke test for the liteocr package."""

import sys

import liteocr


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    print(f"liteocr {liteocr.__version__}")
    print(f"native library loaded: {liteocr.lib._name}")
    if args:
        preset = args[0]
        engine = liteocr.Engine()
        engine.load_preset(preset, model_dir="models")
        print(f"loaded preset: {preset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
