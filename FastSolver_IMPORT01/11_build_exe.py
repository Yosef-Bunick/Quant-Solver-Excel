# build_exe.py — Build FastSolver.exe (bundled Python) using PyInstaller.
#
# Usage:
#   pip install pyinstaller
#   python build_exe.py
#
# Output:
#   dist/FastSolver/FastSolver.exe   (Windows)
#   dist/FastSolver/FastSolver       (Mac/Linux)
#
# The bundle directory contains all dependencies. Distribute the whole
# `FastSolver/` folder alongside the .xlam.

import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

def main():
    # Pull in every Python source we need at runtime
    entry = HERE / "07_fastsolver_bridge.py"
    if not entry.exists():
        print(f"Missing {entry}", file=sys.stderr)
        sys.exit(1)

    extras = [
        HERE / "08_excel_functions.py",
        HERE / "09_excel_functions_80.py",
        HERE / "10_optimizer.py",
    ]
    for e in extras:
        if not e.exists():
            print(f"Missing {e}", file=sys.stderr)
            sys.exit(1)

    sep = ";" if platform.system() == "Windows" else ":"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "FastSolver",
        "--onedir",          # folder-style (faster startup than --onefile)
        "--noconfirm",
        "--clean",
        "--console",         # keep console for now; switch to --noconsole later
        # JAX/SciPy ship many submodules that PyInstaller may miss
        "--collect-all", "jax",
        "--collect-all", "jaxlib",
        "--collect-all", "scipy",
        "--collect-all", "pycel",
        "--collect-all", "openpyxl",
        "--collect-all", "numpy",
        # Force-add pycel submodules that get missed via dynamic imports
        "--hidden-import", "pycel",
        "--hidden-import", "pycel.excelcompiler",
        "--hidden-import", "pycel.excelwrapper",
        "--hidden-import", "pycel.excelutil",
        "--hidden-import", "pycel.excelformula",
        "--hidden-import", "pycel.lib.binary",
        "--hidden-import", "pycel.lib.date_time",
        "--hidden-import", "pycel.lib.engineering",
        "--hidden-import", "pycel.lib.financial",
        "--hidden-import", "pycel.lib.function_info",
        "--hidden-import", "pycel.lib.information",
        "--hidden-import", "pycel.lib.logical",
        "--hidden-import", "pycel.lib.lookup",
        "--hidden-import", "pycel.lib.stats",
        "--hidden-import", "pycel.lib.text",
        # Bundle our extra source files
    ]
    for e in extras:
        cmd += ["--add-data", f"{e}{sep}."]
    cmd += [str(entry)]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()