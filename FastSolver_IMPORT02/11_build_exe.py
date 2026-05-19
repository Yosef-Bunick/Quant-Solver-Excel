#!/usr/bin/env python3
# build_exe.py — Build FastSolver.exe (bundled Python) using PyInstaller.
#
# Usage:
#   python patch_scipy_stats.py      # run ONCE first (fixes scipy.stats)
#   python 11_build_exe.py
#
# This version wraps everything in try/except and KEEPS THE WINDOW OPEN
# on any failure so the error is readable instead of the window vanishing.
#
# The scipy.stats `del obj` crash is fixed by patch_scipy_stats.py editing
# the scipy source that PyInstaller bundles. No runtime hook is needed.
#
# Output:
#   dist/FastSolver/FastSolver.exe   (Windows)

import platform
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).parent.resolve()
MIN_PYINSTALLER = (6, 6)


def _pause_and_exit(code):
    """Keep the console open so the user can read what happened."""
    print()
    print("=" * 60)
    print(f"  build_exe.py finished with exit code {code}")
    print("  (window held open so you can read the messages above)")
    print("  Press ENTER to close.")
    print("=" * 60)
    try:
        input()
    except Exception:
        pass
    sys.exit(code)


def _check_pyinstaller():
    try:
        import PyInstaller
    except Exception as e:
        print(f"PyInstaller is not installed / not importable: {e}")
        print('Fix: pip install --upgrade "pyinstaller>=6.6"')
        _pause_and_exit(1)
    try:
        ver = tuple(int(p) for p in PyInstaller.__version__.split(".")[:2])
    except Exception:
        ver = (0, 0)
    print(f"PyInstaller version: {PyInstaller.__version__}")
    if ver < MIN_PYINSTALLER:
        print(f"Too old. Need >= {'.'.join(map(str, MIN_PYINSTALLER))}.")
        print('Fix: pip install --upgrade "pyinstaller>=6.6"')
        _pause_and_exit(1)


def _scipy_patch_warning():
    """Warn (do not fail) if scipy source still has the unguarded del obj."""
    try:
        import scipy.stats._distn_infrastructure as m
        txt = Path(m.__file__).read_text(encoding="utf-8")
    except Exception as e:
        print(f"NOTE: could not verify scipy patch state ({e}); continuing.")
        return
    if "# [FASTSOLVER PATCH]" in txt:
        print("scipy.stats patch: DETECTED (good).")
    elif ("for obj in [s for s in dir() if s.startswith('_doc_')]:"
          in txt and "\ndel obj\n" in txt):
        print("=" * 60)
        print("  WARNING: scipy.stats is NOT patched.")
        print("  The built exe will likely crash at runtime with")
        print("  NameError: name 'obj' is not defined.")
        print("  Run:  python patch_scipy_stats.py   then rebuild.")
        print("  (Building anyway so you can see other errors too.)")
        print("=" * 60)
    else:
        print("scipy.stats patch state: unclear; continuing.")


def main():
    print("=" * 60)
    print("  FastSolver build")
    print("=" * 60)

    _check_pyinstaller()
    _scipy_patch_warning()

    entry = HERE / "07_fastsolver_bridge.py"
    if not entry.exists():
        print(f"MISSING entry script: {entry}")
        _pause_and_exit(1)

    # NOTE: 08_excel_functions.py and 09_excel_functions_80.py are NOT
    # bundled. Verified: 07_fastsolver_bridge.py (the only entry point)
    # never imports them (no static import, no importlib/exec/runpy), and
    # no VBA/other module invokes them. They only pull in jax/jaxlib, which
    # is the largest dead weight in the bundle. Excluding them + jax is the
    # single biggest size reduction available.
    extras = []
    for e in extras:
        if not e.exists():
            print(f"MISSING required source: {e}")
            _pause_and_exit(1)

    sep = ";" if platform.system() == "Windows" else ":"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "FastSolver",
        "--onedir",
        "--noconfirm",
        "--clean",
        "--console",
        "--collect-all", "scipy",
        "--collect-all", "pycel",
        "--collect-all", "openpyxl",
        "--collect-submodules", "numpy",
        # --- Slimming: exclude heavy packages the solver never imports.
        # Verified: none of 07/08/09 reference any of these. They were only
        # being pulled in because jax/scipy cross-reference them.
        # jax/jaxlib: 08/09 are the only jax users and are no longer
        # bundled (entry point never imports them). Exclude outright -
        # this is the biggest single size win.
        "--exclude-module", "jax",
        "--exclude-module", "jaxlib",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "tensorflow",
        "--exclude-module", "keras",
        "--exclude-module", "sklearn",
        "--exclude-module", "cv2",
        "--exclude-module", "pandas",
        "--exclude-module", "matplotlib",
        "--exclude-module", "numba",
        "--exclude-module", "llvmlite",
        "--exclude-module", "sympy",
        "--exclude-module", "h5py",
        "--exclude-module", "PIL",
        "--exclude-module", "IPython",
        "--exclude-module", "notebook",
        "--exclude-module", "dask",
        "--exclude-module", "sqlalchemy",
        # Test suites bundled by collect-all but never run in production:
        "--exclude-module", "scipy._lib.array_api_compat.dask",
        "--exclude-module", "pytest",
        "--exclude-module", "_pytest",
        "--exclude-module", "nose",
        "--exclude-module", "hypothesis",
        # Additional dead weight identified from _internal size audit:
        "--exclude-module", "grpc",
        "--exclude-module", "cryptography",
        "--exclude-module", "pydantic",
        "--exclude-module", "pydantic_core",
        "--exclude-module", "jedi",
        "--exclude-module", "parso",
        "--exclude-module", "aiohttp",
        "--exclude-module", "numexpr",
        "--exclude-module", "onnx",
        "--exclude-module", "hf_xet",
        "--exclude-module", "safetensors",
        "--exclude-module", "sentencepiece",
        "--exclude-module", "lightning",
        "--exclude-module", "ml_dtypes",
        "--exclude-module", "optree",
        "--exclude-module", "nbformat",
        "--exclude-module", "jsonschema",
        "--exclude-module", "tornado",
        "--exclude-module", "yaml",
        "--exclude-module", "google",
        "--hidden-import", "pycel",
        "--hidden-import", "pycel.excelcompiler",
        "--hidden-import", "pycel.excelwrapper",
        "--hidden-import", "pycel.excelutil",
        "--hidden-import", "pycel.excelformula",
        "--hidden-import", "pycel.lib.date_time",
        "--hidden-import", "pycel.lib.engineering",
        "--hidden-import", "pycel.lib.function_info",
        "--hidden-import", "pycel.lib.information",
        "--hidden-import", "pycel.lib.logical",
        "--hidden-import", "pycel.lib.lookup",
        "--hidden-import", "pycel.lib.stats",
        "--hidden-import", "pycel.lib.text",
        "--hidden-import", "scipy.optimize",
        "--hidden-import", "scipy.optimize._minimize",
        "--hidden-import", "scipy.optimize._differentialevolution",
        "--hidden-import", "scipy.stats",
    ]
    for e in extras:
        cmd += ["--add-data", f"{e}{sep}."]
    cmd += [str(entry)]

    print("\nRunning PyInstaller:")
    print(" ".join(cmd))
    print()

    try:
        res = subprocess.run(cmd)
    except FileNotFoundError as e:
        print(f"\nCould not launch PyInstaller: {e}")
        print('Fix: pip install --upgrade "pyinstaller>=6.6"')
        _pause_and_exit(1)
    except Exception:
        print("\nUnexpected error launching PyInstaller:")
        traceback.print_exc()
        _pause_and_exit(1)

    if res.returncode != 0:
        print(f"\nPyInstaller FAILED with exit code {res.returncode}.")
        print("Scroll up for the PyInstaller error lines.")
        _pause_and_exit(res.returncode)

    print("\nBuild succeeded: dist/FastSolver/FastSolver.exe")
    _pause_and_exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        print("\n" + "=" * 60)
        print("  UNHANDLED ERROR in build_exe.py")
        print("=" * 60)
        traceback.print_exc()
        try:
            input("\nPress ENTER to close.")
        except Exception:
            pass
        sys.exit(1)