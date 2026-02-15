"""Logging utilities for ScreenCloak with sanitization support."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
DETECTION_LOG = LOGS_DIR / "detections.log"
APP_LOG = LOGS_DIR / "screencloak.log"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up application logging with file and console handlers.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("screencloak")
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # File handler with rotation (10MB max, 5 backup files)
    file_handler = RotatingFileHandler(
        APP_LOG,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_detection(
    detection: dict[str, Any],
    sanitized: bool = True,
    log_file: Path = DETECTION_LOG,
) -> None:
    """
    Log a detection event to the detections log file.

    Args:
        detection: Detection dictionary with keys:
            - type: Detection type (e.g., "seed_phrase", "credit_card")
            - confidence: Confidence score (0.0-1.0)
            - text_preview: Preview of detected text (already truncated)
            - bounding_box: Tuple of (x, y, w, h)
            - action: Action taken ("blur", "warn", "ignore")
        sanitized: If True, replace text_preview with "[REDACTED]" (default: True)
        log_file: Path to detection log file (default: logs/detections.log)
    """
    # Sanitize text if requested
    logged_text = "[REDACTED]" if sanitized else detection.get("text_preview", "")

    # Create log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": detection.get("type", "unknown"),
        "confidence": detection.get("confidence", 0.0),
        "text": logged_text,
        "bounding_box": detection.get("bounding_box", (0, 0, 0, 0)),
        "action": detection.get("action", "unknown"),
    }

    # Append to log file as JSON
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def log_performance(
    operation: str,
    duration_ms: float,
    details: dict[str, Any] | None = None,
) -> None:
    """
    Log performance metrics.

    Args:
        operation: Name of the operation (e.g., "ocr", "detection", "frame_diff")
        duration_ms: Duration in milliseconds
        details: Optional additional details
    """
    logger = logging.getLogger("screencloak.performance")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "duration_ms": round(duration_ms, 2),
        "details": details or {},
    }

    logger.debug(f"Performance: {json.dumps(log_entry)}")


def get_detection_stats(log_file: Path = DETECTION_LOG) -> dict[str, Any]:
    """
    Get statistics from detection log.

    Args:
        log_file: Path to detection log file

    Returns:
        Dictionary with detection statistics
    """
    if not log_file.exists():
        return {
            "total_detections": 0,
            "by_type": {},
            "by_action": {},
        }

    stats: dict[str, Any] = {
        "total_detections": 0,
        "by_type": {},
        "by_action": {},
    }

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                stats["total_detections"] += 1

                # Count by type
                detection_type = entry.get("type", "unknown")
                stats["by_type"][detection_type] = stats["by_type"].get(detection_type, 0) + 1

                # Count by action
                action = entry.get("action", "unknown")
                stats["by_action"][action] = stats["by_action"].get(action, 0) + 1

            except json.JSONDecodeError:
                continue

    return stats


def clear_detection_log(log_file: Path = DETECTION_LOG) -> None:
    """
    Clear the detection log file.

    Args:
        log_file: Path to detection log file
    """
    if log_file.exists():
        log_file.unlink()

    logger = logging.getLogger("screencloak")
    logger.info(f"Cleared detection log: {log_file}")


# Initialize default logger
_default_logger = setup_logging()


if __name__ == "__main__":
    # Test logging
    logger = setup_logging(logging.DEBUG)

    logger.debug("This is a debug message")
    logger.info("ScreenCloak logging initialized")
    logger.warning("This is a warning")
    logger.error("This is an error")

    # Test detection logging
    test_detection = {
        "type": "seed_phrase",
        "confidence": 0.95,
        "text_preview": "abandon ability able...",
        "bounding_box": (100, 200, 300, 50),
        "action": "blur",
    }

    log_detection(test_detection, sanitized=True)
    log_detection(test_detection, sanitized=False)

    # Show stats
    stats = get_detection_stats()
    logger.info(f"Detection stats: {stats}")

    print("\n✓ Logging test complete!")
    print(f"  - App log: {APP_LOG}")
    print(f"  - Detection log: {DETECTION_LOG}")
