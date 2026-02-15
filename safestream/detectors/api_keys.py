"""API key detection module — detects leaked API keys and tokens."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import BaseDetector, DetectionResult, OCRResult


class APIKeysDetector(BaseDetector):
    """
    Detects API keys and tokens for common services via regex patterns.

    Supported: AWS, GitHub, Stripe, OpenAI, Anthropic, Google,
               Slack, Twilio, SendGrid, DigitalOcean, NPM, Discord,
               Cloudflare, Heroku (14 services total).
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.enabled: bool = getattr(config, "enabled", True)
        self._patterns: dict[str, list[dict[str, Any]]] = {}
        if self.enabled:
            self._patterns = self._load_patterns()

    def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
        if not self.enabled or not ocr_results:
            return []

        combined_text = " ".join(r.text for r in ocr_results)
        detections: list[DetectionResult] = []

        for service, patterns in self._patterns.items():
            for pattern_info in patterns:
                raw_pattern = pattern_info["pattern"]
                confidence = float(pattern_info.get("confidence", 0.90))
                description = pattern_info.get("description", service)

                try:
                    match = re.search(raw_pattern, combined_text)
                except re.error:
                    continue

                if not match:
                    continue

                matched_text = match.group()
                bbox = self._bbox_for_match(matched_text, ocr_results)

                detections.append(DetectionResult(
                    type="api_key",
                    confidence=confidence,
                    text_preview=f"[{service}] {description}",
                    bounding_box=bbox,
                    action="blur" if confidence >= 0.9 else "warn",
                    metadata={"service": service, "description": description},
                ))
                break  # one detection per service is enough

        return detections

    def _bbox_for_match(
        self, matched_text: str, ocr_results: list[OCRResult]
    ) -> tuple[int, int, int, int]:
        """Return bounding box of the OCR token containing the match prefix."""
        prefix = matched_text[:8]
        for result in ocr_results:
            if prefix in result.text:
                return result.bounding_box
        return ocr_results[0].bounding_box

    def _load_patterns(self) -> dict[str, list[dict[str, Any]]]:
        patterns_path = Path(__file__).parent.parent / "data" / "api_patterns.json"
        if not patterns_path.exists():
            return {}
        with open(patterns_path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def get_supported_services(self) -> list[str]:
        return list(self._patterns.keys())


# Standalone test
def test_api_keys_detector() -> None:
    """Test the API keys detector."""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        enabled: bool = True

    print("Testing API Keys Detector\n" + "=" * 60)

    config = MockConfig(enabled=True)
    detector = APIKeysDetector(config)

    print(f"\nLoaded {len(detector.get_supported_services())} services:")
    for service in detector.get_supported_services():
        print(f"    - {service}")

    ocr_results = [
        OCRResult(
            text="AKIAIOSFODNN7EXAMPLE",
            bounding_box=(10, 10, 200, 20),
            confidence=0.9,
        )
    ]

    detections = detector.detect(ocr_results)
    if detections:
        print(f"\n  Detected: {detections[0].metadata['service']} key")
    else:
        print("\n  No detections (check patterns file)")

    print("\n" + "=" * 60)
    print("API Keys Detector Tests Complete!")


if __name__ == "__main__":
    test_api_keys_detector()
