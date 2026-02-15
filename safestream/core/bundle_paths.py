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
            if getattr(sys, "frozen", False):
                raise RuntimeError(
                    "pytesseract is missing from the SafeStream bundle. "
                    "Re-run PyInstaller with pytesseract in hiddenimports."
                ) from None
            # dev mode: pytesseract may not be installed, that's acceptable

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
