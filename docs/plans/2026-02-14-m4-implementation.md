# M4 Implementation Plan — Menu Bar Icon, API Key Detection, Windows Packaging

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add API key detection, a cross-platform system tray icon, and Windows packaging before beta.

**Architecture:** API key detection plugs directly into the existing detector pipeline. The tray icon runs in a background thread via `pystray` and communicates with the scan loop via a shared `threading.Event` (pause) and integer counter (detections). Windows packaging mirrors the macOS pipeline: PyInstaller produces a folder, Inno Setup wraps it into a `.exe` installer.

**Tech Stack:** `pystray`, `Pillow` (already a dep), PyInstaller, Inno Setup 6 (Windows only)

---

## Working directory

All commands run from `/Users/tomershahar/SafeSense/safestream/` unless stated otherwise.

---

### Task 1: API Key Detection

**Files:**
- Modify: `detectors/api_keys.py`
- Modify: `config.yaml`
- Test: `tests/test_detector.py` (add new test cases)

#### Context

`detectors/api_keys.py` already has `_load_patterns()` which reads `data/api_patterns.json`. The `detect()` method is a stub returning `[]`. The patterns file has 14 services with regex + confidence per pattern. `DetectionResult` supports a `metadata` dict field (used in `credit_card.py`).

The `detect()` import chain to understand: `ocr_results` is `list[OCRResult]` where each `OCRResult` has `.text` (str) and `.bounding_box` (tuple `(x, y, w, h)`).

**Step 1: Write the failing tests**

Add to `tests/test_detector.py` (at the bottom of the file, before any `if __name__` block):

```python
# ---------------------------------------------------------------------------
# API key detection
# ---------------------------------------------------------------------------

class TestAPIKeyDetector:
    """Tests for APIKeysDetector."""

    def _make_config(self, enabled: bool = True) -> Any:
        from dataclasses import dataclass
        @dataclass
        class APIKeysConfig:
            enabled: bool = True
        return APIKeysConfig(enabled=enabled)

    def test_aws_key_detected(self) -> None:
        from detectors.api_keys import APIKeysDetector
        detector = APIKeysDetector(self._make_config())
        ocr = [OCRResult(text="AKIAIOSFODNN7EXAMPLE", bounding_box=(0, 0, 200, 20), confidence=0.9)]
        results = detector.detect(ocr)
        assert len(results) == 1
        assert results[0].type == "api_key"
        assert results[0].metadata["service"] == "AWS"
        assert results[0].action == "blur"

    def test_github_token_detected(self) -> None:
        from detectors.api_keys import APIKeysDetector
        detector = APIKeysDetector(self._make_config())
        token = "ghp_" + "a" * 36
        ocr = [OCRResult(text=token, bounding_box=(0, 0, 300, 20), confidence=0.9)]
        results = detector.detect(ocr)
        assert len(results) == 1
        assert results[0].metadata["service"] == "GitHub"

    def test_disabled_returns_empty(self) -> None:
        from detectors.api_keys import APIKeysDetector
        detector = APIKeysDetector(self._make_config(enabled=False))
        ocr = [OCRResult(text="AKIAIOSFODNN7EXAMPLE", bounding_box=(0, 0, 200, 20), confidence=0.9)]
        assert detector.detect(ocr) == []

    def test_no_false_positive_on_normal_text(self) -> None:
        from detectors.api_keys import APIKeysDetector
        detector = APIKeysDetector(self._make_config())
        ocr = [OCRResult(text="Hello world, no secrets here.", bounding_box=(0, 0, 300, 20), confidence=0.9)]
        assert detector.detect(ocr) == []
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/tomershahar/SafeSense/safestream && python -m pytest tests/test_detector.py::TestAPIKeyDetector -v
```
Expected: `FAILED` — `assert len(results) == 1` fails (stub returns `[]`).

**Step 3: Implement `detect()` in `detectors/api_keys.py`**

Replace the entire file content:

```python
"""API key detection module — detects leaked API keys and tokens."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import BaseDetector, DetectionResult, OCRResult


class APIKeysDetector(BaseDetector):
    """
    Detects API keys and tokens for common services via regex patterns.

    Supported: AWS, GitHub, Stripe, OpenAI, Anthropic, Google,
               Slack, Twilio, SendGrid, DigitalOcean, NPM, Discord,
               Cloudflare, Heroku (14 services total).
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.enabled: bool = getattr(config, "enabled", True)
        self._patterns: dict[str, list[dict[str, Any]]] = {}
        if self.enabled:
            self._patterns = self._load_patterns()

    def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
        if not self.enabled or not ocr_results:
            return []

        combined_text = " ".join(r.text for r in ocr_results)
        detections: list[DetectionResult] = []

        for service, patterns in self._patterns.items():
            for pattern_info in patterns:
                raw_pattern = pattern_info["pattern"]
                confidence = float(pattern_info.get("confidence", 0.90))
                description = pattern_info.get("description", service)

                try:
                    match = re.search(raw_pattern, combined_text)
                except re.error:
                    continue

                if not match:
                    continue

                matched_text = match.group()
                bbox = self._bbox_for_match(matched_text, ocr_results)

                detections.append(DetectionResult(
                    type="api_key",
                    confidence=confidence,
                    text_preview=f"[{service}] {description}",
                    bounding_box=bbox,
                    action="blur" if confidence >= 0.9 else "warn",
                    metadata={"service": service, "description": description},
                ))
                break  # one detection per service is enough

        return detections

    def _bbox_for_match(
        self, matched_text: str, ocr_results: list[OCRResult]
    ) -> tuple[int, int, int, int]:
        """Return bounding box of the OCR token containing the match prefix."""
        prefix = matched_text[:8]
        for result in ocr_results:
            if prefix in result.text:
                return result.bounding_box
        # Fallback: first OCR result
        return ocr_results[0].bounding_box

    def _load_patterns(self) -> dict[str, list[dict[str, Any]]]:
        patterns_path = Path(__file__).parent.parent / "data" / "api_patterns.json"
        if not patterns_path.exists():
            return {}
        with open(patterns_path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def get_supported_services(self) -> list[str]:
        return list(self._patterns.keys())
```

**Step 4: Update `config.yaml`** — change `api_keys` section:

```yaml
  api_keys:
    enabled: true
```

(Remove the `# Paid tier feature` comment too.)

**Step 5: Run tests to verify they pass**

```bash
cd /Users/tomershahar/SafeSense/safestream && python -m pytest tests/test_detector.py -v -q
```
Expected: all tests pass including the 4 new `TestAPIKeyDetector` tests.

**Step 6: Commit**

```bash
git -C /Users/tomershahar/SafeSense add safestream/detectors/api_keys.py safestream/config.yaml safestream/tests/test_detector.py
git -C /Users/tomershahar/SafeSense commit -m "feat: implement API key detection for 14 services"
```

---

### Task 2: System Tray Icon (`ui/tray.py`)

**Files:**
- Create: `ui/tray.py`
- Create: `tests/test_tray.py`

#### Context

`ui/` already exists (has `__init__.py` and `preview.py`). `pystray` needs to be installed: `pip3 install pystray`. On macOS, `pystray` uses `pyobjc-framework-AppKit` as its backend; on Windows it uses `pywin32`. Both are installed as transitive dependencies of `pystray` on each platform.

The tray is **optional** — if `pystray` is not installed or the `--no-tray` flag is passed, the app runs headlessly with no tray icon.

**Step 1: Install pystray**

```bash
pip3 install pystray --quiet
python3 -c "import pystray; print('pystray OK')"
```
Expected: `pystray OK`

**Step 2: Write the failing tests**

Create `tests/test_tray.py`:

```python
"""Tests for ui/tray.py — tests run without a real display by mocking pystray."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch


def test_set_state_updates_internal_state() -> None:
    """set_state() stores the new state and updates the icon if running."""
    with patch.dict("sys.modules", {"pystray": MagicMock(), "PIL": MagicMock(),
                                     "PIL.Image": MagicMock(), "PIL.ImageDraw": MagicMock()}):
        from ui import tray as tray_module
        tray_module._PYSTRAY_AVAILABLE = True
        t = tray_module.SystemTray(on_quit=lambda: None)
        t.set_state("alert")
        assert t._state == "alert"


def test_set_state_idle() -> None:
    with patch.dict("sys.modules", {"pystray": MagicMock(), "PIL": MagicMock(),
                                     "PIL.Image": MagicMock(), "PIL.ImageDraw": MagicMock()}):
        from ui import tray as tray_module
        tray_module._PYSTRAY_AVAILABLE = True
        t = tray_module.SystemTray(on_quit=lambda: None)
        t.set_state("idle")
        assert t._state == "idle"


def test_increment_detections() -> None:
    """increment_detections() bumps the counter each call."""
    with patch.dict("sys.modules", {"pystray": MagicMock(), "PIL": MagicMock(),
                                     "PIL.Image": MagicMock(), "PIL.ImageDraw": MagicMock()}):
        from ui import tray as tray_module
        tray_module._PYSTRAY_AVAILABLE = True
        t = tray_module.SystemTray(on_quit=lambda: None)
        assert t._detection_count == 0
        t.increment_detections()
        t.increment_detections()
        assert t._detection_count == 2


def test_toggle_pause_sets_event() -> None:
    """_toggle_pause() sets pause_event when pausing, clears it when resuming."""
    with patch.dict("sys.modules", {"pystray": MagicMock(), "PIL": MagicMock(),
                                     "PIL.Image": MagicMock(), "PIL.ImageDraw": MagicMock()}):
        from ui import tray as tray_module
        tray_module._PYSTRAY_AVAILABLE = True
        t = tray_module.SystemTray(on_quit=lambda: None)
        assert not t.pause_event.is_set()
        t._toggle_pause(None, None)   # pause
        assert t.pause_event.is_set()
        assert t._paused is True
        t._toggle_pause(None, None)   # resume
        assert not t.pause_event.is_set()
        assert t._paused is False


def test_no_tray_when_pystray_unavailable() -> None:
    """start() is a no-op when pystray is not installed."""
    with patch.dict("sys.modules", {"pystray": MagicMock(), "PIL": MagicMock(),
                                     "PIL.Image": MagicMock(), "PIL.ImageDraw": MagicMock()}):
        from ui import tray as tray_module
        tray_module._PYSTRAY_AVAILABLE = False
        t = tray_module.SystemTray(on_quit=lambda: None)
        t.start()   # should not raise
        assert t._icon is None
```

**Step 3: Run tests to verify they fail**

```bash
cd /Users/tomershahar/SafeSense/safestream && python -m pytest tests/test_tray.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` — `ui/tray.py` doesn't exist yet.

**Step 4: Implement `ui/tray.py`**

```python
"""System tray / menu bar icon for SafeStream (macOS + Windows via pystray)."""
from __future__ import annotations

import threading
from typing import Any, Callable, Literal

try:
    import pystray
    from PIL import Image, ImageDraw
    _PYSTRAY_AVAILABLE = True
except ImportError:
    _PYSTRAY_AVAILABLE = False

# Icon colours for each state
_COLORS: dict[str, tuple[int, int, int, int]] = {
    "idle":  (128, 128, 128, 255),  # grey
    "clean": (0,   200, 0,   255),  # green
    "alert": (220, 50,  50,  255),  # red
}


def _make_icon(state: str, size: int = 64) -> Any:
    """Draw a filled circle icon for the given state."""
    color = _COLORS.get(state, _COLORS["idle"])
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
    return img


class SystemTray:
    """
    Manages the system tray / menu bar icon and menu.

    Usage:
        tray = SystemTray(on_quit=app.shutdown)
        tray.start()                       # launches background thread
        tray.set_state("alert")            # icon turns red
        tray.increment_detections()        # bumps menu counter
        tray.stop()                        # removes icon
    """

    def __init__(self, on_quit: Callable[[], None]) -> None:
        self._on_quit = on_quit
        self._state: str = "idle"
        self._detection_count: int = 0
        self._paused: bool = False
        self._pause_event: threading.Event = threading.Event()
        self._icon: Any = None
        self._thread: threading.Thread | None = None

    @property
    def pause_event(self) -> threading.Event:
        """Set when the user pauses scanning via the tray menu."""
        return self._pause_event

    def set_state(self, state: Literal["idle", "clean", "alert"]) -> None:
        """Update icon colour. Call from any thread."""
        self._state = state
        if self._icon is not None:
            try:
                self._icon.icon = _make_icon(state)
            except Exception:
                pass

    def increment_detections(self) -> None:
        """Bump the detection counter shown in the menu."""
        self._detection_count += 1
        if self._icon is not None:
            try:
                self._icon.update_menu()
            except Exception:
                pass

    def start(self) -> None:
        """Start the tray icon in a daemon background thread. No-op if pystray not available."""
        if not _PYSTRAY_AVAILABLE:
            return

        menu = pystray.Menu(
            pystray.MenuItem("SafeStream", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: "⏸  Pause" if not self._paused else "▶  Resume",
                self._toggle_pause,
            ),
            pystray.MenuItem(
                lambda item: f"Detections: {self._detection_count}",
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )
        self._icon = pystray.Icon(
            "SafeStream",
            icon=_make_icon("idle"),
            title="SafeStream",
            menu=menu,
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True, name="tray")
        self._thread.start()

    def stop(self) -> None:
        """Remove the tray icon."""
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass

    def _toggle_pause(self, icon: Any, item: Any) -> None:
        self._paused = not self._paused
        if self._paused:
            self._pause_event.set()
            self.set_state("idle")
        else:
            self._pause_event.clear()

    def _quit(self, icon: Any, item: Any) -> None:
        self.stop()
        self._on_quit()
```

**Step 5: Run tests to verify they pass**

```bash
cd /Users/tomershahar/SafeSense/safestream && python -m pytest tests/test_tray.py -v
```
Expected: 5 PASSED.

**Step 6: Commit**

```bash
git -C /Users/tomershahar/SafeSense add safestream/ui/tray.py safestream/tests/test_tray.py
git -C /Users/tomershahar/SafeSense commit -m "feat: add system tray icon module (pystray, macOS + Windows)"
```

---

### Task 3: Integrate Tray into `main.py`

**Files:**
- Modify: `main.py`
- Modify: `SafeStream.spec` (add pystray to hiddenimports)
- Modify: `requirements.txt` (add pystray)

#### Context

`SafeStream` class in `main.py` has a `setup()` method that initialises components, and a `run()` method with the scan loop (`while self._running`). The scan loop calls `self._handle_detections(scan_result)` when something is found.

A `--no-tray` flag needs to be added so the app can run headlessly (tests, CI, servers).

**Step 1: Add `--no-tray` to the argument parser**

In `_build_parser()`, add after the `--once` argument:

```python
parser.add_argument(
    "--no-tray",
    action="store_true",
    help="Disable system tray icon (useful for headless/server environments)",
)
```

**Step 2: Add tray to `SafeStream.__init__`**

In `SafeStream.__init__`, add:
```python
self._tray: Any = None
```

**Step 3: Add tray initialisation to `SafeStream.setup()`**

At the END of `setup()`, after all other components are initialised, add:

```python
# 8. System tray (optional)
if not getattr(self.args, "no_tray", False):
    from ui.tray import SystemTray  # noqa: PLC0415
    self._tray = SystemTray(on_quit=self.shutdown)
    self._tray.start()
    self._logger.info("System tray icon started")
```

**Step 4: Check pause event at the top of the scan loop**

In `run()`, at the top of the `while self._running:` loop, add (right after `t_start = time.monotonic()`):

```python
# Respect pause from tray menu
if self._tray is not None and self._tray.pause_event.is_set():
    time.sleep(0.1)
    continue
```

**Step 5: Update tray state after each scan**

In `run()`, after `scan_result = self._pipeline.scan(ocr_results)`, add:

```python
# Update tray icon state
if self._tray is not None:
    if scan_result.should_blur or scan_result.should_warn:
        self._tray.set_state("alert")
    else:
        self._tray.set_state("clean")
```

**Step 6: Call `increment_detections()` on detection**

In `_handle_detections()`, after the existing logging, add:

```python
if self._tray is not None:
    for _ in scan_result.detections:
        self._tray.increment_detections()
```

**Step 7: Stop tray in `shutdown()`**

In `SafeStream.shutdown()`, add:

```python
if self._tray is not None:
    self._tray.stop()
```

**Step 8: Add pystray to `requirements.txt`**

Add this line to `requirements.txt`:
```
pystray>=0.19.0
```

**Step 9: Add pystray to `SafeStream.spec` hiddenimports**

In `SafeStream.spec`, add to the `hiddenimports` list:
```python
"pystray",
"pystray._darwin",
```

**Step 10: Run full test suite**

```bash
cd /Users/tomershahar/SafeSense/safestream && python -m pytest tests/ --ignore=tests/benchmark.py -q
```
Expected: all tests pass (76+ passed, 0 failed).

**Step 11: Quick smoke test**

```bash
cd /Users/tomershahar/SafeSense/safestream && timeout 5 python3 main.py --no-obs --verbose --once 2>&1 || true
```
Expected: starts cleanly, logs "System tray icon started" (or skips if `--no-tray`), exits after one frame.

**Step 12: Commit**

```bash
git -C /Users/tomershahar/SafeSense add safestream/main.py safestream/SafeStream.spec safestream/requirements.txt
git -C /Users/tomershahar/SafeSense commit -m "feat: integrate system tray icon into main scan loop"
```

---

### Task 4: Windows Path Support in `bundle_paths.py`

**Files:**
- Modify: `core/bundle_paths.py`
- Modify: `tests/test_bundle_paths.py`

#### Context

`bundle_paths.py` currently only handles macOS frozen paths (hardcoded `/opt/homebrew/bin/tesseract` and `~/Library/...`). On Windows, paths must use `%APPDATA%\SafeStream\` and the UB-Mannheim Tesseract default install path `C:\Program Files\Tesseract-OCR\`.

**Step 1: Write the failing tests**

Add to `tests/test_bundle_paths.py`:

```python
def test_windows_bundle_mode_config_dir():
    """In Windows bundle mode, config_dir is %APPDATA%\SafeStream."""
    import os
    fake_meipass = "/tmp/fake_bundle"
    fake_appdata = "/tmp/fake_appdata"
    with patch.object(sys, 'frozen', True, create=True), \
         patch.object(sys, '_MEIPASS', fake_meipass, create=True), \
         patch.object(sys, 'platform', 'win32'), \
         patch.dict(os.environ, {"APPDATA": fake_appdata}):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
        assert paths.config_dir == Path(fake_appdata) / "SafeStream"


def test_windows_bundle_mode_log_dir():
    """In Windows bundle mode, log_dir is %APPDATA%\SafeStream\logs."""
    import os
    fake_meipass = "/tmp/fake_bundle"
    fake_appdata = "/tmp/fake_appdata"
    with patch.object(sys, 'frozen', True, create=True), \
         patch.object(sys, '_MEIPASS', fake_meipass, create=True), \
         patch.object(sys, 'platform', 'win32'), \
         patch.dict(os.environ, {"APPDATA": fake_appdata}):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
        assert paths.log_dir == Path(fake_appdata) / "SafeStream" / "logs"
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/tomershahar/SafeSense/safestream && python -m pytest tests/test_bundle_paths.py::test_windows_bundle_mode_config_dir tests/test_bundle_paths.py::test_windows_bundle_mode_log_dir -v
```
Expected: FAIL — current code always returns macOS paths.

**Step 3: Update `get_paths()` in `core/bundle_paths.py`**

Replace the frozen branch with a platform-aware version:

```python
def get_paths() -> AppPaths:
    """Return resolved paths for the current runtime environment."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]

        if sys.platform == "win32":
            # Windows: use %APPDATA%\SafeStream
            appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            win_tess = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
            win_tessdata = Path("C:/Program Files/Tesseract-OCR/tessdata")
            return AppPaths(
                config_dir=appdata / "SafeStream",
                log_dir=appdata / "SafeStream" / "logs",
                data_dir=base / "data",
                tesseract_cmd=win_tess if win_tess.exists() else None,
                tessdata_prefix=win_tessdata if win_tessdata.exists() else None,
            )
        else:
            # macOS: use system Homebrew Tesseract
            system_tess = Path("/opt/homebrew/bin/tesseract")
            return AppPaths(
                config_dir=Path.home() / "Library" / "Application Support" / "SafeStream",
                log_dir=Path.home() / "Library" / "Logs" / "SafeStream",
                data_dir=base / "data",
                tesseract_cmd=system_tess if system_tess.exists() else None,
                tessdata_prefix=base / "tessdata",
            )
    else:
        # Dev mode — use repo directory
        repo = Path(__file__).parent.parent
        return AppPaths(
            config_dir=repo,
            log_dir=repo / "logs",
            data_dir=repo / "data",
            tesseract_cmd=None,
            tessdata_prefix=None,
        )
```

**Step 4: Run all bundle_paths tests**

```bash
cd /Users/tomershahar/SafeSense/safestream && python -m pytest tests/test_bundle_paths.py -v
```
Expected: all 11 tests pass.

**Step 5: Commit**

```bash
git -C /Users/tomershahar/SafeSense add safestream/core/bundle_paths.py safestream/tests/test_bundle_paths.py
git -C /Users/tomershahar/SafeSense commit -m "feat: add Windows path support to bundle_paths"
```

---

### Task 5: Windows Packaging Files

**Files:**
- Create: `SafeStream-Windows.spec`
- Create: `scripts/safestream.iss`
- Create: `scripts/build_windows.bat`

#### Context

These files are used on a **Windows machine** to build the installer. They cannot be run on macOS — this task creates the files only. A Windows machine (physical, VM, or CI) is needed to actually execute the build.

Inno Setup 6 must be installed on the Windows build machine: https://jrsoftware.org/isdl.php

**Step 1: Create `SafeStream-Windows.spec`**

```python
# SafeStream-Windows.spec — PyInstaller config for Windows builds
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
    name="SafeStream",
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
    name="SafeStream",
)
```

**Step 2: Create `scripts/safestream.iss`**

```ini
; Inno Setup 6 script for SafeStream Windows installer
[Setup]
AppName=SafeStream
AppVersion=1.0.0
AppPublisher=SafeStream
DefaultDirName={autopf}\SafeStream
DefaultGroupName=SafeStream
OutputDir=dist
OutputBaseFilename=SafeStream-1.0.0-Setup
Compression=lzma
SolidCompression=yes
; No code signing — user must right-click > Run anyway on first launch
; (or run from PowerShell to bypass SmartScreen)

[Files]
Source: "dist\SafeStream\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SafeStream"; Filename: "{app}\SafeStream.exe"
Name: "{commondesktop}\SafeStream"; Filename: "{app}\SafeStream.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\SafeStream.exe"; Description: "Launch SafeStream now"; Flags: nowait postinstall skipifsilent

[Code]
// Check that Tesseract is installed before completing setup
function InitializeSetup(): Boolean;
var
  TesseractPath: String;
begin
  TesseractPath := 'C:\Program Files\Tesseract-OCR\tesseract.exe';
  if not FileExists(TesseractPath) then
  begin
    MsgBox(
      'Tesseract OCR is not installed.' + #13#10 + #13#10 +
      'Please install Tesseract before running SafeStream:' + #13#10 +
      'https://github.com/UB-Mannheim/tesseract/wiki' + #13#10 + #13#10 +
      'After installing Tesseract, run this installer again.',
      mbInformation, MB_OK
    );
    Result := False;
  end else
    Result := True;
end;
```

**Step 3: Create `scripts/build_windows.bat`**

```batch
@echo off
setlocal enabledelayedexpansion

set VERSION=1.0.0
set APP_NAME=SafeStream
set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"

echo === SafeStream Windows Build ===
echo Version: %VERSION%
echo.

echo ^> Cleaning previous build...
if exist build rmdir /s /q build
if exist "dist\%APP_NAME%" rmdir /s /q "dist\%APP_NAME%"

echo ^> Running PyInstaller...
python -m PyInstaller SafeStream-Windows.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller failed
    exit /b 1
)
echo    dist\%APP_NAME%\ created

echo ^> Building installer with Inno Setup...
if not exist %ISCC% (
    echo ERROR: Inno Setup 6 not found at %ISCC%
    echo Install from: https://jrsoftware.org/isdl.php
    exit /b 1
)
%ISCC% scripts\safestream.iss
if errorlevel 1 (
    echo ERROR: Inno Setup failed
    exit /b 1
)

echo.
echo === Build complete ===
echo   Installer: dist\%APP_NAME%-%VERSION%-Setup.exe
```

**Step 4: Verify syntax (macOS)**

Check the `.bat` and `.iss` files are well-formed by inspecting them.

The `.spec` can be syntax-checked:
```bash
cd /Users/tomershahar/SafeSense/safestream && python3 -c "
with open('SafeStream-Windows.spec') as f:
    compile(f.read(), 'SafeStream-Windows.spec', 'exec')
print('SafeStream-Windows.spec syntax OK')
"
```
Expected: `SafeStream-Windows.spec syntax OK`

**Step 5: Commit**

```bash
git -C /Users/tomershahar/SafeSense add safestream/SafeStream-Windows.spec safestream/scripts/safestream.iss safestream/scripts/build_windows.bat
git -C /Users/tomershahar/SafeSense commit -m "feat: add Windows packaging (PyInstaller spec + Inno Setup installer)"
```

---

## Summary of Files Changed

| Action | File |
|--------|------|
| Modify | `detectors/api_keys.py` |
| Modify | `config.yaml` |
| Modify | `tests/test_detector.py` |
| Create | `ui/tray.py` |
| Create | `tests/test_tray.py` |
| Modify | `main.py` |
| Modify | `SafeStream.spec` |
| Modify | `requirements.txt` |
| Modify | `core/bundle_paths.py` |
| Modify | `tests/test_bundle_paths.py` |
| Create | `SafeStream-Windows.spec` |
| Create | `scripts/safestream.iss` |
| Create | `scripts/build_windows.bat` |

## Windows Build Instructions (for later, on a Windows machine)

1. Install Python 3.11+ and run `pip install -r requirements.txt`
2. Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki (default path)
3. Install Inno Setup 6 from https://jrsoftware.org/isdl.php
4. Clone the repo and `cd safestream`
5. Run: `scripts\build_windows.bat`
6. Distribute: `dist\SafeStream-1.0.0-Setup.exe`
