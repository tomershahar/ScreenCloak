"""ScreenCloak — real-time sensitive data detection for live streamers."""

from __future__ import annotations

from core.bundle_paths import setup as _bundle_setup

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Argument parsing (before any heavy imports so --help is instant)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="screencloak",
        description=(
            "ScreenCloak — detects sensitive data on screen and switches OBS "
            "to a Privacy Mode scene before it reaches viewers."
        ),
    )
    parser.add_argument(
        "--mock",
        metavar="IMAGE",
        help="Run in mock mode using a local image file instead of live screen capture. "
             "Useful for testing without a live stream or OBS.",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "paddleocr", "tesseract"],
        help="Override OCR engine from config (auto, paddleocr, tesseract)",
    )
    parser.add_argument(
        "--no-obs",
        action="store_true",
        help="Disable OBS integration — detect and log only, no scene switching",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one frame then exit (useful for CI / scripted tests)",
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Disable system tray icon (useful for headless/server environments)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------


class ScreenCloak:
    """
    Main application — wires all components together and runs the scan loop.

    Component initialisation order matters:
    1. Config  (everything else reads from it)
    2. Logger  (so subsequent init steps can log)
    3. Capture (ScreenCapture or MockCapture)
    4. OCR     (heavyweight — downloads models on first run)
    5. Detector pipeline
    6. Frame differ
    7. OBS client (optional — failure here does not abort startup)

    The main loop:
        frame = capture()
        if frame_differ.has_changed(frame):
            ocr_results = ocr_engine.detect_text(frame)
            scan_result = pipeline.scan(ocr_results)
            if scan_result.should_blur:
                obs_client.trigger_privacy_mode()
                log_detection(...)
            elif scan_result.should_warn:
                log_detection(...)
        sleep(1 / frame_sample_rate)

    The loop runs until SIGINT (Ctrl+C) or SIGTERM, at which point
    shutdown() is called for a clean exit.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._running = False
        self._shutdown_called: bool = False

        # Deferred — initialised in setup()
        self._config: Any = None
        self._capturer: Any = None
        self._ocr_engine: Any = None
        self._pipeline: Any = None
        self._frame_differ: Any = None
        self._obs_client: Any = None
        self._logger: logging.Logger | None = None
        self._tray: Any = None

    def setup(self) -> None:
        """Initialise all components. Raises on fatal errors."""
        # 1. Config
        from core.config_manager import ConfigManager  # noqa: PLC0415

        self._config = ConfigManager.load(self.args.config)

        # CLI overrides
        if self.args.engine:
            self._config.ocr.engine = self.args.engine

        # 2. Logger
        from core.logger import setup_logging  # noqa: PLC0415

        level = logging.DEBUG if self.args.verbose else logging.INFO
        self._logger = setup_logging(level=level)
        self._logger.info("ScreenCloak starting up")
        self._logger.info(f"Config: {self.args.config}")

        # 3. Capture
        from core.capture import create_capture  # noqa: PLC0415

        mock_path = Path(self.args.mock) if self.args.mock else None
        self._capturer = create_capture(self._config, mock_path=mock_path)
        if self.args.mock:
            self._logger.info(f"Mock mode: reading from {self.args.mock}")
        else:
            self._logger.info("Live screen capture enabled")

        # 4. OCR engine
        from core.ocr_engine import OCREngineFactory  # noqa: PLC0415

        self._logger.info(f"Loading OCR engine: {self._config.ocr.engine}")
        self._ocr_engine = OCREngineFactory.create(self._config)
        self._logger.info(f"OCR engine ready: {self._ocr_engine.name}")

        # 5. Detector pipeline
        from core.detector import DetectorPipeline  # noqa: PLC0415

        self._pipeline = DetectorPipeline(self._config)
        self._logger.info(
            f"Detector pipeline ready: {len(self._pipeline.detectors)} detectors"
        )

        # 6. Frame differ
        from core.frame_diff import FrameDiffer  # noqa: PLC0415

        self._frame_differ = FrameDiffer()
        self._logger.info("Frame differencing enabled")

        # 7. OBS client (optional — failure is non-fatal)
        use_mock_obs = bool(self.args.mock) or self.args.no_obs
        if use_mock_obs:
            from core.obs_client import MockOBSClient  # noqa: PLC0415

            self._obs_client = MockOBSClient()
            mode = "--mock mode" if self.args.mock else "--no-obs flag"
            self._logger.info(f"OBS: using MockOBSClient ({mode})")
        elif self._config.obs.enabled:
            from core.obs_client import OBSClient  # noqa: PLC0415

            self._obs_client = OBSClient(self._config)
            connected = self._obs_client.connect()
            if connected:
                self._logger.info(
                    f"OBS connected: {self._config.obs.host}:{self._config.obs.port}"
                )
                # Verify the Privacy Mode scene exists in OBS
                privacy_scene = self._config.obs.privacy_scene
                if self._obs_client.verify_scene_exists(privacy_scene):
                    self._logger.info(
                        f"OBS scene '{privacy_scene}' found — ready to protect"
                    )
                else:
                    self._logger.warning(
                        f"OBS scene '{privacy_scene}' NOT found in OBS. "
                        "Create it before streaming or ScreenCloak cannot switch scenes. "
                        "See docs/OBS_SETUP.md for instructions."
                    )
            else:
                self._logger.warning(
                    "OBS not connected — detections will be logged but scenes "
                    "will not switch. Start OBS with WebSocket enabled to activate."
                )
        else:
            self._logger.info("OBS integration disabled in config")

        # 8. System tray (optional)
        if not getattr(self.args, "no_tray", False):
            try:
                from ui.tray import SystemTray  # noqa: PLC0415
                self._tray = SystemTray(on_quit=self.shutdown)
                self._tray.start()
                self._logger.info("System tray icon started")
            except Exception as e:
                self._logger.warning(f"System tray unavailable: {e} — continuing without tray")
                self._tray = None

    def run(self) -> int:
        """
        Run the main scan loop.

        Returns:
            Exit code (0 = clean shutdown, 1 = error)
        """
        assert self._logger is not None
        assert self._config is not None

        self._running = True
        frame_count = 0
        detections_total = 0

        # frame_sample_rate = N means process every Nth frame
        sample_rate: int = self._config.ocr.frame_sample_rate
        sleep_interval: float = 1.0 / sample_rate

        self._logger.info(
            f"Scan loop started — processing ~{sample_rate} frame(s)/s"
        )

        try:
            while self._running:
                t_start = time.monotonic()

                # Respect pause from tray menu
                if self._tray is not None and self._tray.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                # Capture frame
                try:
                    frame = self._capturer.capture()
                except Exception as e:
                    self._logger.error(f"Capture failed: {e}", exc_info=True)
                    break

                frame_count += 1

                # Skip OCR if frame unchanged (saves CPU on static screens)
                if not self._frame_differ.has_changed(frame):
                    if self.args.once:
                        break
                    self._sleep_remainder(t_start, sleep_interval)
                    continue

                # Run OCR
                try:
                    ocr_results = self._ocr_engine.detect_text(frame)
                except Exception as e:
                    self._logger.error(f"OCR failed: {e}", exc_info=True)
                    if self.args.once:
                        break
                    self._sleep_remainder(t_start, sleep_interval)
                    continue

                if self.args.verbose:
                    self._logger.debug(
                        f"Frame {frame_count}: OCR found {len(ocr_results)} text region(s)"
                    )

                # Run detection pipeline
                try:
                    scan_result = self._pipeline.scan(ocr_results)
                except Exception as e:
                    self._logger.error(f"Detection failed: {e}", exc_info=True)
                    if self.args.once:
                        break
                    self._sleep_remainder(t_start, sleep_interval)
                    continue

                # Update tray icon state
                if self._tray is not None:
                    obs_disconnected = (
                        self._obs_client is not None
                        and hasattr(self._obs_client, "is_connected")
                        and not self._obs_client.is_connected()
                        and self._config.obs.enabled
                        and not (bool(self.args.mock) or self.args.no_obs)
                    )
                    if obs_disconnected:
                        self._tray.set_state("disconnected")  # yellow — OBS link lost
                    elif scan_result.should_blur:
                        self._tray.set_state("alert")   # red — OBS switched
                    elif scan_result.should_warn:
                        self._tray.set_state("warn")    # orange — logged only
                    else:
                        self._tray.set_state("clean")   # green — clear

                # Handle detections
                if scan_result.should_blur or scan_result.should_warn:
                    detections_total += len(scan_result.detections)
                    self._handle_detections(scan_result)

                # --once: process one frame then exit
                if self.args.once:
                    break

                self._sleep_remainder(t_start, sleep_interval)

        except KeyboardInterrupt:
            pass  # Also handled by signal handler, but catches direct Ctrl+C

        self._logger.info(
            f"Scan loop stopped — {frame_count} frames processed, "
            f"{detections_total} total detections"
        )
        return 0

    def _handle_detections(self, scan_result: Any) -> None:
        """
        Log detections and trigger OBS privacy mode for blur-level findings.
        """
        from core.logger import log_detection  # noqa: PLC0415

        assert self._logger is not None
        assert self._config is not None

        for detection in scan_result.detections:
            log_detection(
                {
                    "type": detection.type,
                    "confidence": detection.confidence,
                    "text_preview": detection.text_preview,
                    "bounding_box": detection.bounding_box,
                    "action": detection.action,
                },
                sanitized=self._config.privacy.log_sanitized,
            )

        if scan_result.should_blur:
            top = scan_result.detections[0]
            self._logger.warning(
                f"DETECTION [{top.type}] confidence={top.confidence:.2f} "
                f"action={top.action} — triggering Privacy Mode"
            )
            if self._obs_client:
                self._obs_client.trigger_privacy_mode()
        else:
            top = scan_result.detections[0]
            self._logger.info(
                f"WARNING [{top.type}] confidence={top.confidence:.2f} "
                f"action={top.action} — logged only (no scene switch)"
            )

        if self._tray is not None:
            for _ in scan_result.detections:
                self._tray.increment_detections()

    def _sleep_remainder(self, t_start: float, interval: float) -> None:
        """Sleep for whatever time remains in the current frame interval."""
        elapsed = time.monotonic() - t_start
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def shutdown(self) -> None:
        """Cleanly stop the scan loop and release all resources."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._running = False

        if self._tray is not None:
            self._tray.stop()

        if self._obs_client is not None:
            try:
                self._obs_client.disconnect()
            except Exception:
                pass

        if self._capturer is not None:
            try:
                self._capturer.close()
            except Exception:
                pass

        if self._logger:
            self._logger.info("ScreenCloak shut down cleanly")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    _PATHS = _bundle_setup()

    parser = _build_parser()
    args = parser.parse_args()

    # Resolve config path: use CLI override if provided, otherwise use bundle-resolved path
    if args.config is None:
        args.config = str(_PATHS.config_dir / "config.yaml")

    app = ScreenCloak(args)

    # Graceful shutdown on SIGINT / SIGTERM
    def _signal_handler(sig: int, _frame: Any) -> None:
        sig_name = signal.Signals(sig).name
        if app._logger:
            app._logger.info(f"Received {sig_name} — shutting down...")
        else:
            print(f"\nReceived {sig_name} — shutting down...")
        app.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        app.setup()
    except Exception as e:
        print(f"[ScreenCloak] Startup failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    if app._tray is not None:
        # macOS requires pystray on the main thread — run the scan loop in a
        # background thread and give the main thread to the tray event loop.
        scan_thread = threading.Thread(target=app.run, daemon=True, name="scan")
        scan_thread.start()
        app._tray.run_blocking()   # blocks until user clicks Quit
        scan_thread.join(timeout=5)
    else:
        app.run()

    app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
