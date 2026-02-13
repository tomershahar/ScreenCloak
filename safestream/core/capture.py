"""Screen capture module — ScreenCapture (live) and MockCapture (test)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("safestream.capture")


class CaptureBase(ABC):
    """
    Abstract base class for screen frame providers.

    Both live capture (ScreenCapture) and test capture (MockCapture)
    implement this interface, so the main loop and tests are identical.
    """

    @abstractmethod
    def capture(self) -> np.ndarray:
        """
        Capture and return a single frame as an RGB numpy array.

        Returns:
            RGB image array (H, W, 3) dtype=uint8
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""
        ...

    def __enter__(self) -> "CaptureBase":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class ScreenCapture(CaptureBase):
    """
    Live screen capture using mss (cross-platform, fast).

    mss captures directly from the display buffer without going through
    the OS screenshot API, making it faster than PIL/pyautogui alternatives.

    Monitor indexing (mss convention):
    - Index 0: all monitors combined into one giant virtual canvas
    - Index 1: primary monitor (default for SafeStream)
    - Index 2+: additional monitors

    Output: RGB numpy array. mss natively returns BGRA — we strip the
    alpha channel and swap B↔R to give OCR engines standard RGB.

    Thread safety: mss is NOT thread-safe. Each thread must create its
    own ScreenCapture instance. SafeStream's main loop is single-threaded
    so this is fine.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialise ScreenCapture for the configured monitor.

        Args:
            config: Main Config object — reads config.capture.monitor

        Raises:
            RuntimeError: If mss is not installed or monitor index is invalid
        """
        try:
            import mss as mss_lib  # noqa: PLC0415

            self._mss_lib = mss_lib
        except ImportError:
            raise RuntimeError(
                "mss not installed. Run: pip install mss"
            )

        self._monitor_idx: int = getattr(config.capture, "monitor", 1)
        self._sct: Any | None = None
        self._monitor: dict[str, int] | None = None
        self._open()

    def _open(self) -> None:
        """Open mss context and resolve monitor geometry."""
        self._sct = self._mss_lib.mss()

        available = len(self._sct.monitors) - 1  # index 0 = combined, so real monitors = len-1
        if self._monitor_idx >= len(self._sct.monitors):
            logger.warning(
                f"Monitor {self._monitor_idx} not found "
                f"({available} monitor(s) available). Falling back to primary (1)."
            )
            self._monitor_idx = 1

        self._monitor = self._sct.monitors[self._monitor_idx]
        logger.info(
            f"ScreenCapture ready — monitor {self._monitor_idx} "
            f"({self._monitor['width']}×{self._monitor['height']})"
        )

    def capture(self) -> np.ndarray:
        """
        Capture one frame from the configured monitor.

        mss returns BGRA (Blue-Green-Red-Alpha). We:
        1. Grab the raw BGRA screenshot
        2. Convert to numpy array
        3. Drop the alpha channel (not needed for OCR)
        4. Swap channels BGR → RGB (what all our OCR engines expect)

        Returns:
            RGB image (H, W, 3) dtype=uint8
        """
        assert self._sct is not None, "ScreenCapture is closed"
        assert self._monitor is not None

        raw = self._sct.grab(self._monitor)

        # mss returns a ScreenShot object; .rgb is a bytes object (R,G,B packed)
        # Fastest path: use np.frombuffer on the bgra data, then slice channels
        bgra = np.frombuffer(raw.bgra, dtype=np.uint8).reshape(
            raw.height, raw.width, 4
        )

        # Drop alpha, swap B↔R: BGRA[:,:,2::-1] → RGB
        rgb = bgra[:, :, 2::-1].copy()

        return rgb

    def close(self) -> None:
        """Release mss resources."""
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (width, height) of the capture area."""
        if self._monitor is None:
            return (0, 0)
        return (self._monitor["width"], self._monitor["height"])


class MockCapture(CaptureBase):
    """
    Test capture that loads frames from image files on disk.

    Used for:
    - `python main.py --mock path/to/image.png` — single image replay
    - Integration tests — deterministic frames, no screen required
    - Benchmarking — known content, repeatable results

    If a list of paths is provided, capture() cycles through them in order
    (wraps around). This lets tests simulate a sequence of screen frames.
    """

    def __init__(self, image_paths: str | Path | list[str | Path]) -> None:
        """
        Initialise MockCapture with one or more image files.

        Args:
            image_paths: Single path or list of paths to image files.
                         Supported formats: PNG, JPG, BMP (anything OpenCV reads).

        Raises:
            FileNotFoundError: If any provided path does not exist
        """
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]

        self._paths: list[Path] = []
        for p in image_paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"MockCapture: image not found: {path}")
            self._paths.append(path)

        self._index: int = 0
        logger.info(f"MockCapture ready — {len(self._paths)} image(s)")

    def capture(self) -> np.ndarray:
        """
        Return the next image from the file list (cycling).

        OpenCV reads images as BGR. We convert to RGB to match what
        ScreenCapture returns, so the main loop sees the same format
        regardless of capture source.

        Returns:
            RGB image (H, W, 3) dtype=uint8

        Raises:
            RuntimeError: If an image file cannot be read by OpenCV
        """
        path = self._paths[self._index % len(self._paths)]
        self._index += 1

        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"MockCapture: failed to read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        logger.debug(f"MockCapture: returned frame from {path.name}")
        return rgb

    def close(self) -> None:
        """No-op — no resources to release."""
        pass

    @property
    def frame_count(self) -> int:
        """Number of images in the sequence."""
        return len(self._paths)

    @property
    def current_index(self) -> int:
        """Index of the next frame that will be returned."""
        return self._index % len(self._paths)


def create_capture(config: Any, mock_path: str | Path | None = None) -> CaptureBase:
    """
    Factory: create a MockCapture or ScreenCapture based on arguments.

    If mock_path is given, returns MockCapture (used by --mock CLI flag).
    Otherwise returns ScreenCapture for live streaming use.

    Args:
        config: Main Config object
        mock_path: Path to image file for mock mode, or None for live capture

    Returns:
        Initialised CaptureBase subclass
    """
    if mock_path is not None:
        return MockCapture(mock_path)
    return ScreenCapture(config)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------


def test_capture() -> None:
    """Test ScreenCapture and MockCapture."""
    import sys
    import tempfile

    from PIL import Image, ImageDraw

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    print("Testing Capture Module (Task 15)\n" + "=" * 60)

    # Test 1: ScreenCapture grabs a live frame
    print("\nTest 1: ScreenCapture — live frame")
    from core.config_manager import ConfigManager

    config = ConfigManager.load("config.yaml")
    try:
        with ScreenCapture(config) as sc:
            frame = sc.capture()
            h, w, c = frame.shape
            if c == 3 and w > 0 and h > 0:
                print(f"  ✓ Captured live frame: {w}×{h} RGB")
                print(f"  ✓ dtype={frame.dtype}, mean brightness={frame.mean():.1f}")
            else:
                print(f"  ✗ Unexpected shape: {frame.shape}")
    except Exception as e:
        print(f"  ✗ ScreenCapture failed: {e}")

    # Test 2: ScreenCapture resolution property
    print("\nTest 2: ScreenCapture resolution property")
    try:
        sc2 = ScreenCapture(config)
        w, h = sc2.resolution
        print(f"  ✓ Monitor resolution: {w}×{h}")
        sc2.close()
    except Exception as e:
        print(f"  ✗ {e}")

    # Test 3: MockCapture loads from file
    print("\nTest 3: MockCapture — load from file")
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), "MockCapture Test", fill="black")
        img.save(img_path)

        mc = MockCapture(img_path)
        frame = mc.capture()
        h, w, c = frame.shape
        if c == 3 and w == 400 and h == 100:
            print(f"  ✓ MockCapture returned {w}×{h} RGB frame")
        else:
            print(f"  ✗ Unexpected shape: {frame.shape}")
        mc.close()

    # Test 4: MockCapture cycles through multiple images
    print("\nTest 4: MockCapture — multi-image cycling")
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(3):
            p = Path(tmpdir) / f"frame_{i}.png"
            Image.new("RGB", (100, 100), color=(i * 80, i * 80, i * 80)).save(p)
            paths.append(p)

        mc_multi = MockCapture(paths)
        frames = [mc_multi.capture() for _ in range(5)]  # 5 captures from 3 images
        if mc_multi.current_index == 2:  # wrapped: 5 % 3 = 2
            print(f"  ✓ Cycled correctly: index={mc_multi.current_index} after 5 captures")
        else:
            print(f"  ✗ Expected index=2, got {mc_multi.current_index}")
        mc_multi.close()

    # Test 5: MockCapture raises on missing file
    print("\nTest 5: MockCapture — missing file raises FileNotFoundError")
    try:
        MockCapture("/nonexistent/path.png")
        print(f"  ✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  ✓ Correctly raised FileNotFoundError: {e}")

    # Test 6: create_capture factory
    print("\nTest 6: create_capture factory")
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "factory_test.png"
        Image.new("RGB", (200, 100), "white").save(img_path)

        mock = create_capture(config, mock_path=img_path)
        live = create_capture(config, mock_path=None)
        print(f"  ✓ mock_path → {type(mock).__name__}")
        print(f"  ✓ None     → {type(live).__name__}")
        mock.close()
        live.close()

    print("\n" + "=" * 60)
    print("Capture Tests Complete!")


if __name__ == "__main__":
    test_capture()
