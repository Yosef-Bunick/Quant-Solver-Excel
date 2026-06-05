# QuantSolver Engine

A custom optimization engine built in Excel/VBA + Python (Cython-compiled). Handles multi-objective, multi-variable, multi-constraint problems with parallel convergence. Built after the native Excel Solver fell short — tested and verified against a quant's reference benchmark, and beat it on accuracy.

Shipped as an `.xlam` Excel add-in so it drops into existing workflows with a single in-sheet button.

---

## Build instructions

### Dependencies

```
Python 3.12
pip install pyinstaller==6.20.0 cython==3.2.4 setuptools==80.9.0 scipy==1.16.3 numpy==2.2.6 openpyxl==3.1.5 pycel==1.0b30
```

### Build

```bash
# 1. Set flag before building
#    In bridge_07.py, ensure: FS_RUN_FROM_SOURCE = False

# 2. Compile Python to native binaries
python setup_cython.py build_ext --inplace

# 3. Patch scipy for PyInstaller compatibility
python patch_scipy_stats.py

# 4. Bundle into standalone .exe
python 11_build_exe.py
```

### Install in Excel

1. Extract the built `.exe` — the `_internal/` folder and `.xlam` must be in the same directory
2. Right-click the `.xlam`, check "Unblock" / "Trust"
3. Open Excel → Options → Add-ins → Browse → select the `.xlam`

### Version history (in archive/)

| Version | Performance |
|---------|------------|
| 3 | Working baseline |
| 4 | First C++ compile pass (partial) |
| 5 | 12 seconds |
| 6 | 8 seconds |
| 9 | Feature-complete UI |
| 12 | Best — ready to build |

