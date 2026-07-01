"""Setuptools-based build for the LiteOCR Python wrapper.

The native LiteOCR shared library is built via CMake inside a custom
``build`` command, then copied into the ``liteocr`` package so that it is
included in wheels and source installs.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build import build as _build

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
    from setuptools.command.bdist_wheel import get_platform
except Exception:  # pragma: no cover
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
        from wheel._bdist_wheel import get_platform
    except Exception:
        _bdist_wheel = None


def _build_liteocr_shared(source_dir: Path, build_temp: Path, build_lib: Path) -> None:
    """Configure, build and install the LiteOCR shared library with CMake."""
    cmake_build_dir = build_temp / "cmake_build"
    cmake_install_dir = build_temp / "cmake_install"

    # Decide whether to enable ncnn Vulkan.  Default is OFF so that the
    # produced wheel does not depend on a Vulkan loader/runtime.
    enable_vulkan = os.environ.get("LITEOCR_ENABLE_VULKAN", "0") in ("1", "ON", "YES", "TRUE")

    cmake_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLITEOCR_BUILD_SHARED=ON",
        f"-DLITEOCR_ENABLE_VULKAN={'ON' if enable_vulkan else 'OFF'}",
        f"-DCMAKE_INSTALL_PREFIX={cmake_install_dir.as_posix()}",
    ]

    cmake = os.environ.get("CMAKE_EXECUTABLE", "cmake")

    # Configure.
    subprocess.check_call(
        [cmake, "-S", source_dir.as_posix(), "-B", cmake_build_dir.as_posix(), *cmake_args]
    )

    # Build only the shared library target to save time.
    build_cmd = [
        cmake,
        "--build",
        cmake_build_dir.as_posix(),
        "--config",
        "Release",
        "--target",
        "LiteOCRShared",
    ]
    if platform.system() != "Windows":
        build_cmd.extend(["--", f"-j{os.cpu_count() or 2}"])
    subprocess.check_call(build_cmd)

    # Install into a temporary prefix so we can reliably locate the artifact.
    subprocess.check_call(
        [
            cmake,
            "--install",
            cmake_build_dir.as_posix(),
            "--prefix",
            cmake_install_dir.as_posix(),
            "--config",
            "Release",
        ]
    )

    # Locate the shared library in the install prefix.
    lib_dir = cmake_install_dir / "lib"
    bin_dir = cmake_install_dir / "bin"
    candidates = []
    if platform.system() == "Windows":
        candidates.extend(bin_dir.glob("liteocr.dll"))
        candidates.extend(lib_dir.glob("liteocr.dll"))
    elif platform.system() == "Darwin":
        candidates.extend(lib_dir.glob("libliteocr.dylib"))
        candidates.extend(lib_dir.glob("libliteocr.*.dylib"))
    else:
        candidates.extend(lib_dir.glob("libliteocr.so"))
        candidates.extend(lib_dir.glob("libliteocr.so.*"))

    if not candidates:
        raise RuntimeError(
            f"Could not find the LiteOCR shared library in {cmake_install_dir}. "
            "Check the CMake build output above."
        )

    lib_file = candidates[0]
    package_dir = build_lib / "liteocr"
    package_dir.mkdir(parents=True, exist_ok=True)
    dest = package_dir / lib_file.name
    shutil.copy2(lib_file, dest)
    print(f"Copied native library to {dest}")

    # Also copy into the source package directory so editable installs work.
    src_package_dir = source_dir / "python" / "liteocr"
    if src_package_dir.exists():
        src_dest = src_package_dir / lib_file.name
        shutil.copy2(lib_file, src_dest)
        print(f"Copied native library to {src_dest} for editable installs")


class CMakeBuild(_build):
    """Run the CMake build before the normal Python package build."""

    def run(self) -> None:
        source_dir = Path(__file__).parent.resolve()
        build_temp = Path(self.build_temp).resolve()
        build_lib = Path(self.build_lib).resolve()
        _build_liteocr_shared(source_dir, build_temp, build_lib)
        super().run()


_cmdclass: dict[str, type] = {"build": CMakeBuild}

if _bdist_wheel is not None:
    class BdistWheel(_bdist_wheel):
        """Force a platform-specific wheel tag because we ship a native DLL."""

        def get_tag(self):
            python, abi, plat = super().get_tag()
            if plat == "any":
                raw_plat = self.plat_name or get_platform(self.bdist_dir)
                plat = raw_plat.replace("-", "_").replace(".", "_")
            return python, abi, plat

    _cmdclass["bdist_wheel"] = BdistWheel


setup(
    name="liteocr",
    version="0.1.0",
    author="LiteOCR Contributors",
    description="Python wrapper for LiteOCR",
    long_description=Path("Readme.md").read_text(encoding="utf-8") if Path("Readme.md").exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/wuye9036/LiteOCR",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={
        "liteocr": ["*.dll", "*.so", "*.dylib"],
    },
    cmdclass=_cmdclass,
    python_requires=">=3.8",
    install_requires=[
        # NumPy is optional at runtime but required for the numpy helper APIs.
        "numpy>=1.20",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
