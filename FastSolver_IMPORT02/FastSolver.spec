# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['pycel', 'pycel.excelcompiler', 'pycel.excelwrapper', 'pycel.excelutil', 'pycel.excelformula', 'pycel.lib.date_time', 'pycel.lib.engineering', 'pycel.lib.function_info', 'pycel.lib.information', 'pycel.lib.logical', 'pycel.lib.lookup', 'pycel.lib.stats', 'pycel.lib.text', 'scipy.optimize', 'scipy.optimize._minimize', 'scipy.optimize._differentialevolution', 'scipy.stats']
hiddenimports += collect_submodules('numpy')
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pycel')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('openpyxl')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\YosefBunickFastSolver\\07_fastsolver_bridge.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['jax', 'jaxlib', 'torch', 'torchvision', 'torchaudio', 'tensorflow', 'keras', 'sklearn', 'cv2', 'pandas', 'matplotlib', 'numba', 'llvmlite', 'sympy', 'h5py', 'PIL', 'IPython', 'notebook', 'dask', 'sqlalchemy', 'scipy._lib.array_api_compat.dask', 'pytest', '_pytest', 'nose', 'hypothesis', 'grpc', 'cryptography', 'pydantic', 'pydantic_core', 'jedi', 'parso', 'aiohttp', 'numexpr', 'onnx', 'hf_xet', 'safetensors', 'sentencepiece', 'lightning', 'ml_dtypes', 'optree', 'nbformat', 'jsonschema', 'tornado', 'yaml', 'google'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FastSolver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FastSolver',
)
