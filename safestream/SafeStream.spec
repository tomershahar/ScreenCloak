# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

# Detect Tesseract paths (Apple Silicon homebrew default)
TESS_BIN = "/opt/homebrew/bin/tesseract"
TESS_DATA = "/opt/homebrew/share/tessdata"

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[
        # NOTE: Do NOT bundle the Tesseract binary — it conflicts with cv2's
        # bundled libtesseract.5.dylib at runtime (SIGABRT on symbol mismatch).
        # bundle_paths.py points pytesseract to the system Homebrew Tesseract instead.
    ],
    datas=[
        (TESS_DATA, "tessdata"),
        ("data/bip39_wordlist.txt", "data"),
        ("data/api_patterns.json", "data"),
        ("config.yaml", "."),
    ],
    hiddenimports=[
        # obswebsocket submodules
        "obswebsocket",
        "obswebsocket.core",
        "obswebsocket.events",
        "obswebsocket.requests",
        "obswebsocket.exceptions",
        # cv2 / mss
        "cv2",
        "mss",
        "mss.darwin",
        # rapidfuzz
        "rapidfuzz",
        "rapidfuzz.fuzz",
        "rapidfuzz.process",
        # pytesseract
        "pytesseract",
        # yaml
        "yaml",
        # pystray
        "pystray",
        "pystray._darwin",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages
        "paddleocr",
        "paddlepaddle",
        "torch",
        "easyocr",
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
    ],
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
    name="SafeStream",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,   # CLI sidecar app — keep stdout/stderr visible in Terminal
    disable_windowed_traceback=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file="scripts/entitlements.plist",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SafeStream",
)

app = BUNDLE(
    coll,
    name="SafeStream.app",
    icon=None,
    bundle_identifier="com.safestream.app",
    version="1.0.0",
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "NSScreenCaptureUsageDescription":
            "SafeStream captures your screen to detect sensitive information before it reaches viewers.",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "LSUIElement": True,
    },
)
