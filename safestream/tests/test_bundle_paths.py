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
