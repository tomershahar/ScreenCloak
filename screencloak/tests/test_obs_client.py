"""Tests for core/obs_client.py — reconnect logic and scene verification."""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.obs_client import OBSClient, MockOBSClient, create_obs_client


# ---------------------------------------------------------------------------
# Minimal mock config
# ---------------------------------------------------------------------------


@dataclass
class MockOBSConfig:
    host: str = "localhost"
    port: int = 4455
    password: str = ""
    privacy_scene: str = "Privacy Mode"
    auto_return: bool = True
    return_delay: int = 3
    enabled: bool = True


@dataclass
class MockConfig:
    obs: MockOBSConfig = field(default_factory=MockOBSConfig)


def _make_client() -> OBSClient:
    return OBSClient(MockConfig())


# ---------------------------------------------------------------------------
# Fix 1: _set_scene failure marks client as disconnected
# ---------------------------------------------------------------------------


def test_set_scene_failure_marks_client_disconnected() -> None:
    """When the WebSocket call in _set_scene raises, _connected becomes False."""
    client = _make_client()
    client._connected = True
    mock_ws = MagicMock()
    mock_ws.call.side_effect = Exception("Connection lost")
    client._ws = mock_ws

    result = client._set_scene("Privacy Mode")

    assert result is False
    assert not client._connected, (
        "_connected should be False after _set_scene raises — "
        "the connection is dead, not merely the request"
    )


def test_get_current_scene_failure_marks_client_disconnected() -> None:
    """When _get_current_scene raises, _connected becomes False."""
    client = _make_client()
    client._connected = True
    mock_ws = MagicMock()
    mock_ws.call.side_effect = Exception("Broken pipe")
    client._ws = mock_ws

    result = client._get_current_scene()

    assert result is None
    assert not client._connected, (
        "_connected should be False after _get_current_scene raises"
    )


# ---------------------------------------------------------------------------
# Fix 2: auto-reconnect thread spawned after disconnect
# ---------------------------------------------------------------------------


def test_reconnect_thread_started_after_set_scene_failure() -> None:
    """After _set_scene fails, a background reconnect thread is started."""
    client = _make_client()
    client._connected = True
    mock_ws = MagicMock()
    mock_ws.call.side_effect = Exception("Connection lost")
    client._ws = mock_ws

    client._set_scene("Privacy Mode")

    # Give the thread a moment to spawn
    time.sleep(0.05)
    assert client._reconnect_thread is not None, (
        "A reconnect thread should be started after losing connection"
    )
    assert client._reconnect_thread.is_alive() or not client._connected, (
        "Reconnect thread should be alive (retrying) or connection already restored"
    )

    # Cleanup: stop reconnect loop for the test
    client._stop_reconnect.set()


def test_reconnect_thread_not_started_twice() -> None:
    """A second failure while reconnecting does not start a second thread."""
    client = _make_client()
    client._connected = True
    mock_ws = MagicMock()
    mock_ws.call.side_effect = Exception("lost")
    client._ws = mock_ws

    client._set_scene("Privacy Mode")
    first_thread = client._reconnect_thread

    # Simulate a second failure path (e.g., trigger_privacy_mode called again)
    client._set_scene("Privacy Mode")

    time.sleep(0.05)
    assert client._reconnect_thread is first_thread, (
        "Should reuse the existing reconnect thread, not start a second one"
    )

    client._stop_reconnect.set()


def test_reconnect_restores_connected_flag() -> None:
    """When reconnect succeeds, _connected is set back to True."""
    client = _make_client()
    client._connected = True
    mock_ws = MagicMock()
    mock_ws.call.side_effect = Exception("lost")
    client._ws = mock_ws

    # Patch connect() to succeed immediately so the thread exits fast
    reconnect_calls: list[int] = []

    def _fast_reconnect() -> bool:
        reconnect_calls.append(1)
        client._connected = True
        client._ws = MagicMock()
        return True

    client.connect = _fast_reconnect  # type: ignore[method-assign]

    client._set_scene("Privacy Mode")
    # Wait up to 2 seconds for reconnect thread to call connect()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not client._connected:
        time.sleep(0.05)

    assert client._connected, "Reconnect thread should have restored _connected=True"
    assert reconnect_calls, "connect() should have been called by the reconnect thread"


# ---------------------------------------------------------------------------
# Fix 3: verify_scene_exists
# ---------------------------------------------------------------------------


def test_verify_scene_exists_returns_true_when_scene_present() -> None:
    """verify_scene_exists() returns True when the scene is in OBS scene list."""
    client = _make_client()
    client._connected = True

    # Mock the WS call to return a scene list containing our scene
    mock_response = MagicMock()
    mock_response.getScenes.return_value = [
        {"sceneName": "Main"},
        {"sceneName": "Privacy Mode"},
        {"sceneName": "BRB"},
    ]
    mock_ws = MagicMock()
    mock_ws.call.return_value = mock_response
    client._ws = mock_ws

    assert client.verify_scene_exists("Privacy Mode") is True


def test_verify_scene_exists_returns_false_when_scene_missing() -> None:
    """verify_scene_exists() returns False when scene is NOT in OBS scene list."""
    client = _make_client()
    client._connected = True

    mock_response = MagicMock()
    mock_response.getScenes.return_value = [
        {"sceneName": "Main"},
        {"sceneName": "BRB"},
    ]
    mock_ws = MagicMock()
    mock_ws.call.return_value = mock_response
    client._ws = mock_ws

    assert client.verify_scene_exists("Privacy Mode") is False


def test_verify_scene_exists_returns_false_when_not_connected() -> None:
    """verify_scene_exists() returns False when not connected (no crash)."""
    client = _make_client()
    client._connected = False
    client._ws = None

    assert client.verify_scene_exists("Privacy Mode") is False


def test_verify_scene_exists_returns_false_on_exception() -> None:
    """verify_scene_exists() returns False gracefully if the WS call raises."""
    client = _make_client()
    client._connected = True
    mock_ws = MagicMock()
    mock_ws.call.side_effect = Exception("WS error")
    client._ws = mock_ws

    assert client.verify_scene_exists("Privacy Mode") is False
