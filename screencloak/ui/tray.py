"""System tray / menu bar icon for ScreenCloak (macOS + Windows via pystray)."""
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
    "idle":         (128, 128, 128, 255),  # grey   — scanning, no detections yet
    "clean":        (0,   200, 0,   255),  # green  — screen is clear
    "warn":         (255, 165, 0,   255),  # orange — low-confidence detection (logged only)
    "alert":        (220, 50,  50,  255),  # red    — high-confidence detection (OBS switched)
    "disconnected": (255, 200, 0,   255),  # yellow — OBS WebSocket connection lost
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

    @property
    def pause_event(self) -> threading.Event:
        """Set when the user pauses scanning via the tray menu."""
        return self._pause_event

    def set_state(self, state: Literal["idle", "clean", "warn", "alert", "disconnected"]) -> None:
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
        """Build the tray icon. No-op if pystray not available.

        On macOS, pystray must run on the main thread — call run_blocking() from
        main() after moving the scan loop to a background thread. On Windows,
        run_blocking() also works but can also be called from a thread.
        """
        if not _PYSTRAY_AVAILABLE:
            return

        menu = pystray.Menu(
            pystray.MenuItem("ScreenCloak", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: "Pause" if not self._paused else "Resume",
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
            "ScreenCloak",
            icon=_make_icon("idle"),
            title="ScreenCloak",
            menu=menu,
        )

    def run_blocking(self) -> None:
        """Block the calling thread running the tray event loop.

        Must be called from the main thread on macOS. Returns when the tray
        is stopped (e.g. user clicks Quit).
        """
        if self._icon is not None:
            self._icon.run()

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
