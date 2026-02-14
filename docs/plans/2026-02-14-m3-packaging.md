# M3 macOS Packaging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Package SafeStream as a signed, notarized macOS `.app` distributed via a `.dmg` drag-to-install.

**Architecture:** PyInstaller bundles the Python interpreter + all deps + Tesseract binary into `SafeStream.app`. A new `core/bundle_paths.py` module detects frozen mode and redirects pytesseract and config/log paths to the correct locations. `scripts/build_mac.sh` runs the full pipeline: build → sign → dmg → notarize → staple.

**Tech Stack:** PyInstaller, create-dmg, codesign, xcrun notarytool/stapler

---

## Working directory

All commands run from `/Users/tomershahar/SafeSense/safestream/` unless stated otherwise.

---

### Task 1: `core/bundle_paths.py` — runtime path resolver

**Files:**
- Create: `core/bundle_paths.py`
- Create: `tests/test_bundle_paths.py`

**Step 1: Write the failing test**

```python
# tests/test_bundle_paths.py
import sys
from pathlib import Path
from unittest.mock import patch


def test_dev_mode_config_dir():
    """In dev mode (not frozen), config_dir is the repo root (safestream/)."""
    with patch.object(sys, 'frozen', False, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
    assert paths.config_dir == Path(__file__).parent.parent


def test_dev_mode_log_dir():
    with patch.object(sys, 'frozen', False, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
    assert paths.log_dir == Path(__file__).parent.parent / "logs"


def test_bundle_mode_config_dir():
    """In bundle mode (frozen), config_dir is ~/Library/Application Support/SafeStream."""
    fake_meipass = "/tmp/fake_bundle"
    with patch.object(sys, 'frozen', True, create=True), \
         patch.object(sys, '_MEIPASS', fake_meipass, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
    expected = Path.home() / "Library" / "Application Support" / "SafeStream"
    assert paths.config_dir == expected


def test_bundle_mode_tessdata_prefix():
    fake_meipass = "/tmp/fake_bundle"
    with patch.object(sys, 'frozen', True, create=True), \
         patch.object(sys, '_MEIPASS', fake_meipass, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
    assert paths.tessdata_prefix == Path("/tmp/fake_bundle/tessdata")


def test_bundle_mode_tesseract_cmd():
    fake_meipass = "/tmp/fake_bundle"
    with patch.object(sys, 'frozen', True, create=True), \
         patch.object(sys, '_MEIPASS', fake_meipass, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
    assert paths.tesseract_cmd == Path("/tmp/fake_bundle/bin/tesseract")
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_bundle_paths.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.bundle_paths'`

**Step 3: Implement `core/bundle_paths.py`**

```python
"""Bundle path resolver — detects PyInstaller frozen mode vs dev mode.

Call setup() once at startup (before importing pytesseract) to redirect
the Tesseract binary and configure config/log directories.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppPaths:
    config_dir: Path
    log_dir: Path
    data_dir: Path
    tesseract_cmd: Path | None      # None = use system PATH
    tessdata_prefix: Path | None    # None = use system default


def get_paths() -> AppPaths:
    """Return resolved paths for the current runtime environment."""
    if getattr(sys, "frozen", False):
        # Running inside PyInstaller .app bundle
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        return AppPaths(
            config_dir=Path.home() / "Library" / "Application Support" / "SafeStream",
            log_dir=Path.home() / "Library" / "Logs" / "SafeStream",
            data_dir=base / "data",
            tesseract_cmd=base / "bin" / "tesseract",
            tessdata_prefix=base / "tessdata",
        )
    else:
        # Dev mode — use repo directory
        repo = Path(__file__).parent.parent
        return AppPaths(
            config_dir=repo,
            log_dir=repo / "logs",
            data_dir=repo / "data",
            tesseract_cmd=None,     # use system Tesseract from PATH
            tessdata_prefix=None,   # use system tessdata
        )


def setup() -> AppPaths:
    """
    Resolve paths and apply environment patches.

    Must be called before importing pytesseract or core.config_manager.
    Idempotent — safe to call multiple times.
    """
    paths = get_paths()

    # Redirect pytesseract to bundled binary (bundle mode only)
    if paths.tesseract_cmd is not None:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = str(paths.tesseract_cmd)
        except ImportError:
            pass

    if paths.tessdata_prefix is not None:
        os.environ["TESSDATA_PREFIX"] = str(paths.tessdata_prefix)

    # Create writable directories if they don't exist
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    # On first launch in bundle mode, copy config template
    config_file = paths.config_dir / "config.yaml"
    if getattr(sys, "frozen", False) and not config_file.exists():
        template = Path(sys._MEIPASS) / "config.yaml"  # type: ignore[attr-defined]
        if template.exists():
            import shutil
            shutil.copy(template, config_file)

    return paths
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bundle_paths.py -v
```
Expected: 5 PASSED

**Step 5: Commit**

```bash
git add core/bundle_paths.py tests/test_bundle_paths.py
git commit -m "feat: add bundle_paths module for PyInstaller path resolution"
```

---

### Task 2: Patch `main.py` to call `bundle_paths.setup()` first

**Files:**
- Modify: `main.py` (top of file, before other safestream imports)

**Step 1: Find the import block in main.py**

The top of `main.py` has argument parsing before heavy imports. We need to call `bundle_paths.setup()` before `ConfigManager` or `OCREngineFactory` are imported, but after arg parsing is done.

**Step 2: Add the setup call**

In `main.py`, find the section where components are initialized (after arg parsing, before the main loop). Add at the very top of the initialization function, before any safestream core imports:

```python
# At the top of main.py, after the argparse section:
from core.bundle_paths import setup as _bundle_setup

# Then in the main() function, as the FIRST line before anything else:
_PATHS = _bundle_setup()
```

Then pass `_PATHS.config_dir / "config.yaml"` as the config path to `ConfigManager.load()`:

```python
# Find this line:
config = ConfigManager.load(args.config)

# Replace with:
config_path = str(_PATHS.config_dir / "config.yaml") if not args.config else args.config
config = ConfigManager.load(config_path)
```

**Step 3: Verify existing tests still pass**

```bash
pytest tests/test_detector.py tests/test_integration.py -v
```
Expected: 67 PASSED

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: call bundle_paths.setup() at startup for frozen app support"
```

---

### Task 3: `SafeStream.spec` — PyInstaller configuration

**Files:**
- Create: `SafeStream.spec` (in `safestream/` root)

**Step 1: Install PyInstaller**

```bash
pip3 install pyinstaller --quiet
pyinstaller --version
```
Expected: `6.x.x`

**Step 2: Create the spec file**

```python
# SafeStream.spec
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
        (TESS_BIN, "bin"),
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
    console=False,   # No terminal window — background app
    disable_windowed_traceback=False,
    target_arch="arm64",   # Apple Silicon; use "universal2" for Intel+ARM
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
    icon=None,             # Add icon.icns here later
    bundle_identifier="com.safestream.app",
    version="1.0.0",
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "NSScreenCaptureUsageDescription":
            "SafeStream captures your screen to detect sensitive information before it reaches viewers.",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "LSUIElement": True,   # Hide from Dock (background app)
    },
)
```

**Step 3: Create entitlements file**

```bash
mkdir -p scripts
```

Create `scripts/entitlements.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
</dict>
</plist>
```

**Step 4: Do a test build (no signing yet)**

```bash
pyinstaller SafeStream.spec --clean
```
Expected: `dist/SafeStream.app` created, no errors.

If you see `ModuleNotFoundError` for any import, add it to `hiddenimports` in the spec.

**Step 5: Smoke test the bundle**

```bash
dist/SafeStream.app/Contents/MacOS/SafeStream --help
```
Expected: SafeStream help text printed, no import errors.

**Step 6: Commit**

```bash
git add SafeStream.spec scripts/entitlements.plist
git commit -m "feat: add PyInstaller spec and entitlements for macOS .app bundle"
```

---

### Task 4: `scripts/build_mac.sh` — full build pipeline

**Files:**
- Create: `scripts/build_mac.sh`

**Step 1: Install create-dmg**

```bash
brew install create-dmg
```

**Step 2: Create the build script**

```bash
#!/bin/bash
# scripts/build_mac.sh — Build, sign, package, and notarize SafeStream.app
#
# Required environment variables for signing/notarizing:
#   APPLE_ID           — your Apple ID email
#   APPLE_TEAM_ID      — 10-char team ID from developer.apple.com
#   APPLE_APP_PASSWORD — app-specific password from appleid.apple.com
#   SIGN_IDENTITY      — "Developer ID Application: Name (TEAMID)"
#
# Usage:
#   ./scripts/build_mac.sh             # build + sign + dmg + notarize
#   SKIP_NOTARIZE=1 ./scripts/build_mac.sh  # skip notarization (faster dev builds)

set -euo pipefail

VERSION="1.0.0"
APP_NAME="SafeStream"
DMG_NAME="${APP_NAME}-${VERSION}.dmg"

echo "=== SafeStream macOS Build ==="
echo "Version: $VERSION"
echo ""

# ── 1. Clean previous build ──────────────────────────────────────────────────
echo "→ Cleaning previous build..."
rm -rf build/ dist/

# ── 2. PyInstaller ───────────────────────────────────────────────────────────
echo "→ Running PyInstaller..."
pyinstaller SafeStream.spec --clean --noconfirm
echo "✓ dist/${APP_NAME}.app created"

# ── 3. Code signing ──────────────────────────────────────────────────────────
if [ -n "${SIGN_IDENTITY:-}" ]; then
    echo "→ Signing .app..."
    codesign \
        --deep --force --verify --verbose \
        --sign "$SIGN_IDENTITY" \
        --options runtime \
        --entitlements scripts/entitlements.plist \
        "dist/${APP_NAME}.app"
    echo "✓ Signed"
else
    echo "⚠ SIGN_IDENTITY not set — skipping code signing"
fi

# ── 4. Create DMG ────────────────────────────────────────────────────────────
echo "→ Creating DMG..."
create-dmg \
    --volname "$APP_NAME" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "${APP_NAME}.app" 175 190 \
    --hide-extension "${APP_NAME}.app" \
    --app-drop-link 425 190 \
    "dist/${DMG_NAME}" \
    "dist/${APP_NAME}.app"
echo "✓ dist/${DMG_NAME} created"

# ── 5. Notarize ───────────────────────────────────────────────────────────────
if [ "${SKIP_NOTARIZE:-0}" = "1" ]; then
    echo "→ Skipping notarization (SKIP_NOTARIZE=1)"
elif [ -n "${SIGN_IDENTITY:-}" ]; then
    echo "→ Submitting for notarization (this takes ~2 minutes)..."
    xcrun notarytool submit "dist/${DMG_NAME}" \
        --apple-id "$APPLE_ID" \
        --team-id "$APPLE_TEAM_ID" \
        --password "$APPLE_APP_PASSWORD" \
        --wait
    echo "✓ Notarized"

    echo "→ Stapling notarization ticket..."
    xcrun stapler staple "dist/${DMG_NAME}"
    echo "✓ Stapled"
else
    echo "⚠ Skipping notarization — SIGN_IDENTITY not set"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Build complete ==="
echo "  App:  dist/${APP_NAME}.app"
echo "  DMG:  dist/${DMG_NAME}"
```

**Step 3: Make it executable and commit**

```bash
chmod +x scripts/build_mac.sh
git add scripts/build_mac.sh
git commit -m "feat: add macOS build script (PyInstaller + codesign + DMG + notarize)"
```

---

### Task 5: Test the full unsigned build

**Step 1: Run the build script without signing**

```bash
SKIP_NOTARIZE=1 ./scripts/build_mac.sh
```
Expected:
```
✓ dist/SafeStream.app created
⚠ SIGN_IDENTITY not set — skipping code signing
✓ dist/SafeStream-1.0.0.dmg created
⚠ Skipping notarization
=== Build complete ===
```

**Step 2: Mount the DMG and install**

```bash
open dist/SafeStream-1.0.0.dmg
# Drag SafeStream.app to Applications in the DMG window
```

**Step 3: Launch from Applications**

```bash
open /Applications/SafeStream.app
```

Check `~/Library/Logs/SafeStream/` for log output. Expected: SafeStream starts, config copied to `~/Library/Application Support/SafeStream/config.yaml`.

**Step 4: Smoke test — run with mock image**

```bash
/Applications/SafeStream.app/Contents/MacOS/SafeStream \
  --mock /Applications/SafeStream.app/Contents/MacOS/data/test_images/seed_phrase_12word.png \
  --once --no-obs
```
Expected: seed_phrase detection logged, clean exit.

**Step 5: Commit if all passes**

```bash
git add -A
git commit -m "build: verify unsigned DMG build passes smoke test"
```

---

### Task 6: Sign and notarize (run when ready to distribute)

**Prerequisites:**
- Xcode Command Line Tools installed
- Apple Developer account with `Developer ID Application` certificate in Keychain
- App-specific password generated at appleid.apple.com

**Step 1: Set environment variables**

```bash
export SIGN_IDENTITY="Developer ID Application: Your Name (XXXXXXXXXX)"
export APPLE_ID="you@example.com"
export APPLE_TEAM_ID="XXXXXXXXXX"
export APPLE_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"  # app-specific password
```

**Step 2: Run the full build**

```bash
./scripts/build_mac.sh
```
Expected: all steps complete including notarization and stapling.

**Step 3: Verify notarization**

```bash
spctl --assess --verbose dist/SafeStream.app
```
Expected: `dist/SafeStream.app: accepted` — Gatekeeper approves it.

**Step 4: Test on a clean Mac** (or via a new user account)
- Copy the `.dmg` to another machine or user
- Double-click `.dmg`, drag to Applications
- Launch — should open without any Gatekeeper warning

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` in bundle | Add the module to `hiddenimports` in `SafeStream.spec` |
| `tesseract not found` in bundle | Verify `/opt/homebrew/bin/tesseract` exists; check spec `binaries` path |
| `tessdata not found` | Verify `TESSDATA_PREFIX` set correctly in bundle_paths.py |
| Gatekeeper blocks app | Run `./scripts/build_mac.sh` with signing env vars set |
| Notarization rejected | Check notarytool log: `xcrun notarytool log <submission-id>` |
| App crashes on launch | Run from terminal to see stderr: `dist/SafeStream.app/Contents/MacOS/SafeStream` |

---

## Files Created/Modified Summary

| Action | File |
|--------|------|
| Create | `core/bundle_paths.py` |
| Create | `tests/test_bundle_paths.py` |
| Modify | `main.py` |
| Create | `SafeStream.spec` |
| Create | `scripts/entitlements.plist` |
| Create | `scripts/build_mac.sh` |
