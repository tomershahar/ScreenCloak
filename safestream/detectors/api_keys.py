"""API key detection module — paid tier feature (stub for V1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseDetector, DetectionResult, OCRResult

# Paid tier feature — not active in V1 free tier
_PAID_TIER_NOTICE = (
    "[APIKeys] API key detection is a paid tier feature. "
    "Set detection.api_keys.enabled = true and upgrade to unlock."
)


class APIKeysDetector(BaseDetector):
    """
    Detects API keys and tokens for common services.

    Supported services (when unlocked):
        AWS, GitHub, Stripe, OpenAI, Anthropic, Google,
        Slack, Twilio, SendGrid, DigitalOcean, NPM, Discord

    Status: STUB — returns empty in V1 free tier.
    Full implementation activates with paid tier license.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize API keys detector.

        Args:
            config: Configuration object (Config.detection.api_keys)
        """
        super().__init__(config)
        self.enabled: bool = getattr(config, "enabled", False)
        self._patterns: dict[str, list[dict[str, Any]]] = {}

        if self.enabled:
            self._patterns = self._load_patterns()
        else:
            print(_PAID_TIER_NOTICE)

    def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
        """
        Detect API keys in OCR results.

        Returns empty list in V1 (paid tier feature).

        Args:
            ocr_results: List of OCR results from text detection

        Returns:
            Empty list (V1 stub) — paid tier returns detections
        """
        # V1 stub — paid tier feature
        # TODO: Implement when paid tier is activated
        return []

    def _load_patterns(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load API key patterns from JSON file.

        Returns:
            Dictionary of patterns by service name
        """
        patterns_path = Path(__file__).parent.parent / "data" / "api_patterns.json"

        if not patterns_path.exists():
            print(f"[APIKeys] Warning: patterns file not found at {patterns_path}")
            return {}

        with open(patterns_path, "r") as f:
            data = json.load(f)

        # Filter out comment keys
        return {
            key: value
            for key, value in data.items()
            if not key.startswith("_")
        }

    def get_supported_services(self) -> list[str]:
        """
        Return list of supported API key services.

        Returns:
            List of service names (e.g., ["AWS", "GitHub", "Stripe", ...])
        """
        patterns_path = Path(__file__).parent.parent / "data" / "api_patterns.json"

        if not patterns_path.exists():
            return []

        with open(patterns_path, "r") as f:
            data = json.load(f)

        return [k for k in data.keys() if not k.startswith("_")]


# Standalone test
def test_api_keys_detector() -> None:
    """Test the API keys detector stub."""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        enabled: bool = False

    print("Testing API Keys Detector (Stub)\n" + "=" * 60)

    # Test 1: Stub returns empty
    print("\nTest 1: Stub Returns Empty (Free Tier)")
    config = MockConfig(enabled=False)
    detector = APIKeysDetector(config)
    ocr_results = [
        OCRResult(
            text="AKIA1234567890ABCDEF",  # Looks like AWS key
            bounding_box=(10, 10, 200, 20),
            confidence=0.9,
        )
    ]

    detections = detector.detect(ocr_results)
    if len(detections) == 0:
        print(f"  ✓ Returns empty (correct — paid tier stub)")
    else:
        print(f"  ✗ Should return empty, got {len(detections)}")

    # Test 2: Patterns file loaded and verified
    print("\nTest 2: API Patterns File Verified")
    services = detector.get_supported_services()
    if len(services) > 0:
        print(f"  ✓ {len(services)} services defined in api_patterns.json:")
        for service in services:
            print(f"      - {service}")
    else:
        print(f"  ✗ No services found in patterns file")

    # Test 3: Pattern file has high-confidence patterns
    print("\nTest 3: Pattern Quality Check")
    config_enabled = MockConfig(enabled=True)
    detector_enabled = APIKeysDetector(config_enabled)

    high_confidence_services = []
    for service, patterns in detector_enabled._patterns.items():
        if any(p.get("confidence", 0) >= 0.95 for p in patterns):
            high_confidence_services.append(service)

    print(f"  ✓ {len(high_confidence_services)} services with >95% confidence patterns:")
    for service in high_confidence_services:
        print(f"      - {service}")

    print("\n" + "=" * 60)
    print("API Keys Detector Tests Complete!")
    print("\nNote: Full detection activates with paid tier license.")


if __name__ == "__main__":
    test_api_keys_detector()
