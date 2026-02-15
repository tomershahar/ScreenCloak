# tests/test_bundle_paths.py
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_dev_mode_config_dir():
    """In dev mode (not frozen), config_dir is the repo root (screencloak/)."""
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


def test_dev_mode_tesseract_cmd_is_none():
    """In dev mode, tesseract_cmd must be None (use system PATH)."""
    with patch.object(sys, 'frozen', False, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
    assert paths.tesseract_cmd is None


def test_dev_mode_tessdata_prefix_is_none():
    """In dev mode, tessdata_prefix must be None (use system default)."""
    with patch.object(sys, 'frozen', False, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
    assert paths.tessdata_prefix is None


def test_bundle_mode_config_dir():
    """In bundle mode (frozen), config_dir is ~/Library/Application Support/ScreenCloak."""
    fake_meipass = "/tmp/fake_bundle"
    with patch.object(sys, 'frozen', True, create=True), \
         patch.object(sys, '_MEIPASS', fake_meipass, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
        expected = Path.home() / "Library" / "Application Support" / "ScreenCloak"
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
    # bundle_paths no longer bundles the Tesseract binary (it conflicts with cv2's
    # libtesseract). Instead it points to the system Homebrew binary if present, or None.
    fake_meipass = "/tmp/fake_bundle"
    with patch.object(sys, 'frozen', True, create=True), \
         patch.object(sys, '_MEIPASS', fake_meipass, create=True):
        from core import bundle_paths
        import importlib
        importlib.reload(bundle_paths)
        paths = bundle_paths.get_paths()
    system_tess = Path("/opt/homebrew/bin/tesseract")
    expected = system_tess if system_tess.exists() else None
    assert paths.tesseract_cmd == expected


def test_setup_dev_mode_creates_log_dir(tmp_path):
    """setup() in dev mode must create the log_dir directory."""
    import importlib
    from core import bundle_paths

    fake_log_dir = tmp_path / "logs"
    fake_config_dir = tmp_path

    fake_paths = bundle_paths.AppPaths(
        config_dir=fake_config_dir,
        log_dir=fake_log_dir,
        data_dir=tmp_path / "data",
        tesseract_cmd=None,
        tessdata_prefix=None,
    )

    with patch.object(sys, 'frozen', False, create=True):
        importlib.reload(bundle_paths)
        with patch.object(bundle_paths, 'get_paths', return_value=fake_paths):
            bundle_paths.setup()

    assert fake_log_dir.exists()


def test_setup_bundle_mode_sets_tessdata_prefix(tmp_path):
    """setup() in bundle mode must set os.environ['TESSDATA_PREFIX']."""
    import importlib
    from core import bundle_paths

    fake_tessdata = tmp_path / "tessdata"
    fake_tessdata.mkdir()

    fake_paths = bundle_paths.AppPaths(
        config_dir=tmp_path / "config",
        log_dir=tmp_path / "logs",
        data_dir=tmp_path / "data",
        tesseract_cmd=tmp_path / "bin" / "tesseract",
        tessdata_prefix=fake_tessdata,
    )

    mock_pytesseract = MagicMock()
    mock_pytesseract.pytesseract = MagicMock()

    original_tessdata = os.environ.pop("TESSDATA_PREFIX", None)
    try:
        with patch.object(sys, 'frozen', True, create=True), \
             patch.object(sys, '_MEIPASS', str(tmp_path), create=True):
            importlib.reload(bundle_paths)
            with patch.object(bundle_paths, 'get_paths', return_value=fake_paths), \
                 patch.dict('sys.modules', {'pytesseract': mock_pytesseract}):
                bundle_paths.setup()

        assert os.environ.get("TESSDATA_PREFIX") == str(fake_tessdata)
    finally:
        # Restore original env state
        if original_tessdata is not None:
            os.environ["TESSDATA_PREFIX"] = original_tessdata
        else:
            os.environ.pop("TESSDATA_PREFIX", None)


def test_windows_bundle_mode_config_dir():
    """In Windows bundle mode, config_dir is %APPDATA%\\ScreenCloak."""
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
        assert paths.config_dir == Path(fake_appdata) / "ScreenCloak"


def test_windows_bundle_mode_log_dir():
    """In Windows bundle mode, log_dir is %APPDATA%\\ScreenCloak\\logs."""
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
        assert paths.log_dir == Path(fake_appdata) / "ScreenCloak" / "logs"
