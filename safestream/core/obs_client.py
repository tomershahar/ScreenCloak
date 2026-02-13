"""OBS WebSocket client — switches to Privacy Mode scene on detection."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("safestream.obs_client")


class OBSClient:
    """
    Controls OBS Studio via its WebSocket server (protocol v5, port 4455).

    Behaviour on detection:
    1. `trigger_privacy_mode()` is called
    2. OBS switches to the "Privacy Mode" (BRB) scene immediately
    3. A return timer starts (default 3 seconds)
    4. After the delay, OBS switches back to the previous scene automatically

    If another detection fires while the timer is running, the timer is
    reset so the privacy scene stays up for the full duration from the
    latest detection.

    Connection is best-effort — SafeStream continues working (logging
    detections) even when OBS is not connected. This allows testing
    detectors without OBS running.

    Protocol notes:
    - OBS 28+: WebSocket protocol v5, default port 4455
    - OBS 27 and below: protocol v4, port 4444 (use legacy=True in config)
    - Request to switch scene: SetCurrentProgramScene(sceneName="...")
    - Request to get current scene: GetCurrentProgramScene()

    Thread safety:
    - trigger_privacy_mode() and _return_to_previous() may be called from
      different threads. _lock guards _in_privacy_mode and _return_timer.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialise OBSClient from config.

        Does NOT connect on init — call connect() explicitly or use as
        a context manager. This lets tests construct the client without
        requiring OBS to be running.

        Args:
            config: Main Config object — reads config.obs.*
        """
        obs_cfg = config.obs
        self._host: str = getattr(obs_cfg, "host", "localhost")
        self._port: int = getattr(obs_cfg, "port", 4455)
        self._password: str = getattr(obs_cfg, "password", "")
        self._privacy_scene: str = getattr(obs_cfg, "privacy_scene", "Privacy Mode")
        self._auto_return: bool = getattr(obs_cfg, "auto_return", True)
        self._return_delay: int = getattr(obs_cfg, "return_delay", 3)

        self._ws: Any | None = None          # obsws connection
        self._previous_scene: str | None = None
        self._in_privacy_mode: bool = False
        self._return_timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._connected = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Connect to the OBS WebSocket server.

        Safe to call multiple times — returns True immediately if already
        connected. Failure is logged but does not raise, so SafeStream
        can run without OBS.

        Returns:
            True if connected (or already was), False if connection failed
        """
        if self._connected:
            return True

        try:
            from obswebsocket import obsws  # noqa: PLC0415

            # Use legacy=False for OBS 28+ (protocol v5, port 4455)
            # legacy=True for OBS 27 and below (protocol v4, port 4444)
            legacy = self._port == 4444
            self._ws = obsws(
                host=self._host,
                port=self._port,
                password=self._password,
                legacy=legacy,
                timeout=5,
            )
            self._ws.connect()
            self._connected = True
            logger.info(
                f"OBS connected — {self._host}:{self._port} "
                f"(protocol {'v4' if legacy else 'v5'})"
            )
            return True

        except ImportError:
            logger.error(
                "obs-websocket-py not installed. Run: pip install obs-websocket-py"
            )
            return False
        except Exception as e:
            logger.warning(
                f"OBS connection failed ({self._host}:{self._port}): {e}\n"
                "SafeStream will continue but cannot switch OBS scenes.\n"
                "Ensure OBS is running with WebSocket server enabled."
            )
            return False

    def disconnect(self) -> None:
        """
        Disconnect from OBS WebSocket and cancel any pending return timer.

        Safe to call even if not connected.
        """
        self._cancel_return_timer()

        if self._ws is not None and self._connected:
            try:
                self._ws.disconnect()
            except Exception as e:
                logger.debug(f"OBS disconnect error (ignored): {e}")
            finally:
                self._ws = None
                self._connected = False
                logger.info("OBS disconnected")

    def is_connected(self) -> bool:
        """Return True if currently connected to OBS."""
        return self._connected

    # ------------------------------------------------------------------
    # Scene control
    # ------------------------------------------------------------------

    def trigger_privacy_mode(self) -> bool:
        """
        Switch OBS to the Privacy Mode (BRB) scene.

        Called immediately when a detection occurs. If already in privacy
        mode, resets the return timer so the scene stays up for the full
        return_delay from now.

        Also fetches the current scene beforehand so we can return to it
        automatically after the delay.

        Returns:
            True if scene was switched, False if not connected or switch failed
        """
        if not self._connected:
            logger.warning("OBS not connected — cannot trigger privacy mode")
            return False

        with self._lock:
            # Cancel any existing return timer — we'll restart it below
            self._cancel_return_timer()

            # Save current scene (only if not already in privacy mode, to
            # avoid overwriting the real previous scene with "Privacy Mode")
            if not self._in_privacy_mode:
                self._previous_scene = self._get_current_scene()
                logger.debug(f"Saved previous scene: {self._previous_scene!r}")

            # Switch to privacy scene
            success = self._set_scene(self._privacy_scene)
            if success:
                self._in_privacy_mode = True
                logger.info(
                    f"OBS → Privacy Mode scene {self._privacy_scene!r} "
                    f"(was: {self._previous_scene!r})"
                )

                # Schedule auto-return
                if self._auto_return:
                    self._return_timer = threading.Timer(
                        self._return_delay,
                        self._return_to_previous,
                    )
                    self._return_timer.daemon = True
                    self._return_timer.start()
                    logger.debug(
                        f"Auto-return in {self._return_delay}s to {self._previous_scene!r}"
                    )
            else:
                logger.error(
                    f"Failed to switch to privacy scene {self._privacy_scene!r}. "
                    "Check that the scene exists in OBS."
                )

        return success

    def _return_to_previous(self) -> None:
        """
        Return OBS to the scene that was active before privacy mode.

        Called automatically by the return timer after `return_delay` seconds.
        Also called manually via return_now().
        """
        with self._lock:
            if not self._in_privacy_mode:
                return  # Already returned, nothing to do

            target = self._previous_scene
            if target is None:
                logger.warning("No previous scene saved — cannot auto-return")
                self._in_privacy_mode = False
                return

            success = self._set_scene(target)
            if success:
                self._in_privacy_mode = False
                logger.info(f"OBS ← returned to {target!r}")
            else:
                logger.error(f"Failed to return to {target!r}")

    def return_now(self) -> None:
        """
        Immediately return to the previous scene (cancel the timer).

        Useful when the user manually clears the sensitive content and
        wants to resume streaming without waiting for the timer.
        """
        self._cancel_return_timer()
        self._return_to_previous()

    # ------------------------------------------------------------------
    # Low-level OBS requests
    # ------------------------------------------------------------------

    def _get_current_scene(self) -> str | None:
        """
        Query OBS for the currently active program scene name.

        Returns:
            Scene name string, or None if the query fails
        """
        if not self._ws:
            return None
        try:
            from obswebsocket import requests as obs_req  # noqa: PLC0415

            resp = self._ws.call(obs_req.GetCurrentProgramScene())
            # v5 API: response field is 'currentProgramSceneName'
            scene_name: str = resp.getCurrentProgramSceneName()
            return scene_name
        except Exception as e:
            logger.debug(f"GetCurrentProgramScene failed: {e}")
            return None

    def _set_scene(self, scene_name: str) -> bool:
        """
        Request OBS to switch to the named scene.

        Args:
            scene_name: Exact scene name as it appears in OBS

        Returns:
            True on success, False on failure
        """
        if not self._ws:
            return False
        try:
            from obswebsocket import requests as obs_req  # noqa: PLC0415

            self._ws.call(obs_req.SetCurrentProgramScene(sceneName=scene_name))
            return True
        except Exception as e:
            logger.error(f"SetCurrentProgramScene({scene_name!r}) failed: {e}")
            return False

    def _cancel_return_timer(self) -> None:
        """Cancel the pending return timer, if any."""
        if self._return_timer is not None:
            self._return_timer.cancel()
            self._return_timer = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "OBSClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()


class MockOBSClient:
    """
    Drop-in replacement for OBSClient used in tests and mock mode.

    Records all calls so tests can assert on behaviour without a real
    OBS instance. Never raises; always returns True.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.current_scene: str = "Main"
        self.scene_history: list[str] = []

    def connect(self) -> bool:
        self.calls.append("connect")
        return True

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def is_connected(self) -> bool:
        return True

    def trigger_privacy_mode(self) -> bool:
        self.calls.append("trigger_privacy_mode")
        self.scene_history.append(self.current_scene)
        self.current_scene = "Privacy Mode"
        return True

    def return_now(self) -> None:
        self.calls.append("return_now")
        if self.scene_history:
            self.current_scene = self.scene_history.pop()

    def __enter__(self) -> "MockOBSClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()


def create_obs_client(config: Any, mock: bool = False) -> OBSClient | MockOBSClient:
    """
    Factory: create a real OBSClient or a MockOBSClient.

    Args:
        config: Main Config object
        mock: If True, return MockOBSClient (for --mock CLI mode / tests)

    Returns:
        OBSClient or MockOBSClient
    """
    if mock:
        return MockOBSClient()
    return OBSClient(config)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------


def test_obs_client() -> None:
    """Test OBSClient with mock and (optional) live OBS."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    print("Testing OBS Client (Task 17)\n" + "=" * 60)

    # Test 1: MockOBSClient records calls correctly
    print("\nTest 1: MockOBSClient — call recording")
    mock = MockOBSClient()
    mock.connect()
    mock.trigger_privacy_mode()
    mock.return_now()
    mock.disconnect()

    expected = ["connect", "trigger_privacy_mode", "return_now", "disconnect"]
    if mock.calls == expected:
        print(f"  ✓ Calls recorded: {mock.calls}")
    else:
        print(f"  ✗ Expected {expected}, got {mock.calls}")

    # Test 2: MockOBSClient scene tracking
    print("\nTest 2: MockOBSClient — scene tracking")
    mock2 = MockOBSClient()
    mock2.current_scene = "Gaming"
    mock2.trigger_privacy_mode()
    if mock2.current_scene == "Privacy Mode":
        print(f"  ✓ Scene switched to: {mock2.current_scene!r}")
    mock2.return_now()
    if mock2.current_scene == "Gaming":
        print(f"  ✓ Scene returned to: {mock2.current_scene!r}")

    # Test 3: MockOBSClient context manager
    print("\nTest 3: MockOBSClient — context manager")
    with MockOBSClient() as m:
        m.trigger_privacy_mode()
        if "disconnect" not in m.calls:
            print("  — in context (disconnect not yet called)")
    if "disconnect" in m.calls:
        print("  ✓ disconnect() called on __exit__")

    # Test 4: create_obs_client factory
    print("\nTest 4: create_obs_client factory")
    from core.config_manager import ConfigManager

    config = ConfigManager.load("config.yaml")
    mock_client = create_obs_client(config, mock=True)
    real_client = create_obs_client(config, mock=False)
    print(f"  ✓ mock=True  → {type(mock_client).__name__}")
    print(f"  ✓ mock=False → {type(real_client).__name__}")

    # Test 5: OBSClient graceful failure when OBS not running
    print("\nTest 5: OBSClient — graceful failure (OBS not running)")
    result = real_client.connect()
    if not result:
        print("  ✓ connect() returned False gracefully (OBS not running)")
        print("  ✓ No exception raised")
    else:
        print("  ✓ Connected to live OBS instance!")
        # If we actually connected, test the trigger
        real_client.trigger_privacy_mode()
        print(f"  ✓ trigger_privacy_mode() called on live OBS")
        real_client.disconnect()

    # Test 6: trigger_privacy_mode without connection returns False
    print("\nTest 6: trigger_privacy_mode — returns False when not connected")
    disconnected_client = OBSClient(config)
    result = disconnected_client.trigger_privacy_mode()
    if result is False:
        print("  ✓ Returns False when not connected (no crash)")
    else:
        print(f"  ✗ Expected False, got {result}")

    print("\n" + "=" * 60)
    print("OBS Client Tests Complete!")
    print("\nNote: To test live OBS integration, start OBS with WebSocket")
    print("enabled on port 4455 and re-run this script.")


if __name__ == "__main__":
    test_obs_client()
