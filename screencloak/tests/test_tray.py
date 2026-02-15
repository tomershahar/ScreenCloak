"""Tests for ui/tray.py — run without a real display by mocking pystray."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch


def test_set_state_updates_internal_state() -> None:
    """set_state() stores the new state."""
    with patch.dict("sys.modules", {"pystray": MagicMock(), "PIL": MagicMock(),
                                     "PIL.Image": MagicMock(), "PIL.ImageDraw": MagicMock()}):
        import importlib
        import ui.tray as tray_module
        importlib.reload(tray_module)
        tray_module._PYSTRAY_AVAILABLE = True
        t = tray_module.SystemTray(on_quit=lambda: None)
        t.set_state("alert")
        assert t._state == "alert"


def test_set_state_idle() -> None:
    with patch.dict("sys.modules", {"pystray": MagicMock(), "PIL": MagicMock(),
                                     "PIL.Image": MagicMock(), "PIL.ImageDraw": MagicMock()}):
        import importlib
        import ui.tray as tray_module
        importlib.reload(tray_module)
        tray_module._PYSTRAY_AVAILABLE = True
        t = tray_module.SystemTray(on_quit=lambda: None)
        t.set_state("idle")
        assert t._state == "idle"


def test_increment_detections() -> None:
    """increment_detections() bumps the counter each call."""
    with patch.dict("sys.modules", {"pystray": MagicMock(), "PIL": MagicMock(),
                                     "PIL.Image": MagicMock(), "PIL.ImageDraw": MagicMock()}):
        import importlib
        import ui.tray as tray_module
        importlib.reload(tray_module)
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
        import importlib
        import ui.tray as tray_module
        importlib.reload(tray_module)
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
        import importlib
        import ui.tray as tray_module
        importlib.reload(tray_module)
        tray_module._PYSTRAY_AVAILABLE = False
        t = tray_module.SystemTray(on_quit=lambda: None)
        t.start()   # should not raise
        assert t._icon is None
