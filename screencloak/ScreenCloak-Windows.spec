# ScreenCloak-Windows.spec — PyInstaller config for Windows builds
# -*- mode: python ; coding: utf-8 -*-

# NOTE: Tesseract is NOT bundled. Users must install separately from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Default install path: C:\Program Files\Tesseract-OCR\

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        ("data/bip39_wordlist.txt", "data"),
        ("data/api_patterns.json", "data"),
        ("config.yaml", "."),
    ],
    hiddenimports=[
        "obswebsocket",
        "obswebsocket.core",
        "obswebsocket.events",
        "obswebsocket.requests",
        "obswebsocket.exceptions",
        "cv2",
        "mss",
        "mss.windows",
        "rapidfuzz",
        "rapidfuzz.fuzz",
        "rapidfuzz.process",
        "pytesseract",
        "yaml",
        "pystray",
        "pystray._win32",
        "pywintypes",
        "pythoncom",
        "win32gui_struct",
        "win32api",
        "win32con",
        "win32gui",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "paddleocr",
        "paddlepaddle",
        "torch",
        "easyocr",
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ScreenCloak",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    target_arch="x86_64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="ScreenCloak",
)
