
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import get_package_paths, collect_dynamic_libs

# --- Define paths and collect binaries ---
# XGBoost Binary
_, xgboost_path = get_package_paths('xgboost')
xgboost_binary = [(os.path.join(xgboost_path, 'lib', 'xgboost.dll'), 'xgboost/lib')]

# ONNX Runtime Binaries (optional, include only if needed)
onnx_binaries = collect_dynamic_libs('onnxruntime')  # Keep in case a model uses it

# MetaTrader5 Binaries (optional, include if needed)
_, mt5_path = get_package_paths('MetaTrader5')
mt5_binary = [(os.path.join(mt5_path, 'libmt5.dll'), 'MetaTrader5')] if os.path.exists(os.path.join(mt5_path, 'libmt5.dll')) else []

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['C:\\DGv5'],
    binaries=xgboost_binary + onnx_binaries + mt5_binary,
    datas=[
        ('frontend.html', '.'),
        ('meta_learner_optv3.pkl', '.'),
        ('xgboost_15m.json', '.'),
        ('xgboost_1h.json', '.'),
        ('xgboost_4h.json', '.'),
        (os.path.join(xgboost_path, 'VERSION'), 'xgboost'),
    ],
    hiddenimports=[
        'xgboost',
        'onnxruntime',  # Keep in case needed
        'onnxruntime.c_api',
        'MetaTrader5',
        'joblib',
        'pandas',
        'numpy',
        'fastapi',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.wsproto_impl',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Dylan Ai',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=r'C:\DGv5\Dylan log3.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Dylan Ai',
)
