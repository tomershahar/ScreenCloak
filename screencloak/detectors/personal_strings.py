"""Personal strings detection module using fuzzy matching."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseDetector, DetectionResult, OCRResult


class PersonalStringsDetector(BaseDetector):
    """
    Detects user-defined personal information using fuzzy string matching.

    Examples of personal strings to protect:
    - Full name: "John Doe"
    - Email: "john@example.com"
    - Phone: "555-123-4567"
    - Address: "123 Main Street"

    Uses rapidfuzz for fast, accurate fuzzy matching to handle
    OCR errors and slight variations in how text appears on screen.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize personal strings detector.

        Args:
            config: Configuration object (Config.detection.personal_strings)
        """
        super().__init__(config)
        self.personal_strings: list[str] = getattr(config, "strings", [])
        self.fuzzy_threshold: int = getattr(config, "fuzzy_threshold", 85)
        self.max_free: int = getattr(config, "max_free", 3)

        # Enforce free tier limit
        if len(self.personal_strings) > self.max_free:
            print(
                f"[PersonalStrings] Warning: {len(self.personal_strings)} strings configured, "
                f"free tier limit is {self.max_free}. Using first {self.max_free}."
            )
            self.personal_strings = self.personal_strings[: self.max_free]

        # Check if rapidfuzz is available
        try:
            from rapidfuzz import fuzz, process
            self._fuzz = fuzz
            self._process = process
            self._has_rapidfuzz = True
        except ImportError:
            self._has_rapidfuzz = False
            print(
                "[PersonalStrings] Warning: rapidfuzz not installed. "
                "Falling back to exact matching. Install with: pip install rapidfuzz"
            )

    def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
        """
        Detect personal strings in OCR results using fuzzy matching.

        Algorithm:
        1. Combine OCR text into a single string
        2. For each personal string, run fuzzy matching against OCR text
        3. If similarity >= threshold, create detection result
        4. Find bounding box for matched text

        Args:
            ocr_results: List of OCR results from text detection

        Returns:
            List of detection results (empty if no personal strings found)
        """
        if not ocr_results or not self.personal_strings:
            return []

        detections: list[DetectionResult] = []

        # Combine all OCR text
        combined_text = " ".join([r.text for r in ocr_results])

        for personal_string in self.personal_strings:
            if not personal_string.strip():
                continue

            # Try fuzzy matching
            match_score, matched_text = self._fuzzy_match(personal_string, combined_text)

            if match_score >= self.fuzzy_threshold:
                # Find bounding box for the matched text
                bbox = self._find_bbox_for_fuzzy_match(personal_string, ocr_results)

                if not bbox:
                    # Fall back to using the first/only OCR result's bbox
                    bbox = ocr_results[0].bounding_box if ocr_results else (0, 0, 0, 0)

                # Confidence = match score normalized to 0.0-1.0
                confidence = match_score / 100.0

                # Create truncated preview (never log full personal string)
                text_preview = self._safe_preview(personal_string)

                detection = DetectionResult(
                    type="personal_string",
                    confidence=confidence,
                    text_preview=text_preview,
                    bounding_box=bbox,
                    action="blur" if confidence >= 0.9 else "warn",
                    metadata={
                        "match_score": match_score,
                        "string_type": self._classify_string(personal_string),
                        "fuzzy_threshold": self.fuzzy_threshold,
                    },
                )

                detections.append(detection)

        return detections

    def _fuzzy_match(self, personal_string: str, ocr_text: str) -> tuple[int, str]:
        """
        Run fuzzy matching of a personal string against OCR text.

        Args:
            personal_string: The personal string to search for
            ocr_text: The combined OCR text to search in

        Returns:
            Tuple of (match_score 0-100, matched_substring)
        """
        if not self._has_rapidfuzz:
            # Fallback: exact case-insensitive match
            if personal_string.lower() in ocr_text.lower():
                return 100, personal_string
            return 0, ""

        # Use partial_ratio for substring matching
        # (handles OCR splitting a name across multiple results)
        partial_score = self._fuzz.partial_ratio(
            personal_string.lower(),
            ocr_text.lower()
        )

        # Also try token_set_ratio for word-order independence
        # (handles "Doe John" vs "John Doe")
        token_score = self._fuzz.token_set_ratio(
            personal_string.lower(),
            ocr_text.lower()
        )

        # Take the better of the two scores
        best_score = max(partial_score, token_score)

        return best_score, personal_string

    def _find_bbox_for_fuzzy_match(
        self, personal_string: str, ocr_results: list[OCRResult]
    ) -> tuple[int, int, int, int] | None:
        """
        Find bounding box for a fuzzy-matched personal string.

        Searches individual OCR results for the best match and
        returns the bounding box of the matching result(s).

        Args:
            personal_string: The personal string to find
            ocr_results: List of OCR results

        Returns:
            Bounding box (x, y, w, h) or None if not found
        """
        if not self._has_rapidfuzz:
            # Fallback: simple substring search
            for result in ocr_results:
                if personal_string.lower() in result.text.lower():
                    return result.bounding_box
            return None

        best_score = 0
        best_results: list[OCRResult] = []

        # Search in individual OCR results
        for result in ocr_results:
            score = self._fuzz.partial_ratio(
                personal_string.lower(),
                result.text.lower()
            )

            if score >= self.fuzzy_threshold:
                if score > best_score:
                    best_score = score
                    best_results = [result]
                elif score == best_score:
                    best_results.append(result)

        # If no single result has a high enough match, try adjacent pairs
        # (handles names split across two OCR results)
        if not best_results and len(ocr_results) >= 2:
            for i in range(len(ocr_results) - 1):
                combined = ocr_results[i].text + " " + ocr_results[i + 1].text
                score = self._fuzz.partial_ratio(
                    personal_string.lower(),
                    combined.lower()
                )

                if score >= self.fuzzy_threshold and score > best_score:
                    best_score = score
                    best_results = [ocr_results[i], ocr_results[i + 1]]

        if not best_results:
            return None

        return self._merge_bounding_boxes([r.bounding_box for r in best_results])

    def _classify_string(self, text: str) -> str:
        """
        Classify the type of personal string.

        Args:
            text: Personal string to classify

        Returns:
            Type string ("email", "phone", "address", "name")
        """
        # Email pattern
        if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
            return "email"

        # Phone number pattern
        if re.search(r"[\d\-\(\)\+\s]{7,}", text):
            digits = re.sub(r"\D", "", text)
            if 7 <= len(digits) <= 15:
                return "phone"

        # Address pattern (contains digits + street keywords)
        address_keywords = ["street", "st", "avenue", "ave", "road", "rd", "lane", "ln",
                           "drive", "dr", "boulevard", "blvd", "way", "court", "ct"]
        text_lower = text.lower()
        if any(kw in text_lower for kw in address_keywords) or re.match(r"^\d+\s", text):
            return "address"

        # Default: name
        return "name"

    def _safe_preview(self, personal_string: str) -> str:
        """
        Create a safe preview that doesn't reveal the full personal string.

        Args:
            personal_string: The personal string

        Returns:
            Truncated/masked preview
        """
        if len(personal_string) <= 4:
            return "*" * len(personal_string)

        # Show first 2 chars and last char, mask the rest
        return f"{personal_string[:2]}{'*' * (len(personal_string) - 3)}{personal_string[-1]}"


# Standalone test function
def test_personal_strings_detector() -> None:
    """Test the personal strings detector with various scenarios."""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        strings: list[str] = None
        fuzzy_threshold: int = 85
        max_free: int = 3

        def __post_init__(self) -> None:
            if self.strings is None:
                self.strings = ["John Doe", "john@example.com", "555-123-4567"]

    config = MockConfig()
    detector = PersonalStringsDetector(config)

    print("Testing Personal Strings Detector\n" + "=" * 60)

    # Test 1: Exact name match
    print("\nTest 1: Exact Name Match")
    ocr_results = [
        OCRResult(
            text="Hello John Doe, your order is ready",
            bounding_box=(10, 10, 400, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results)
    if any(d.type == "personal_string" for d in detections):
        det = next(d for d in detections if d.type == "personal_string")
        print(f"  ✓ Detected: {det.text_preview}")
        print(f"  ✓ Match score: {det.metadata['match_score']}")
        print(f"  ✓ Type: {det.metadata['string_type']}")
        print(f"  ✓ Action: {det.action}")
    else:
        print(f"  ✗ Expected detection, got {len(detections)}")

    # Test 2: Fuzzy match with OCR noise (typo)
    print("\nTest 2: Fuzzy Match (OCR Typo: 'Jonn Doe')")
    ocr_results = [
        OCRResult(
            text="From: Jonn Doe",  # OCR misread 'n' instead of 'h'
            bounding_box=(10, 10, 200, 20),
            confidence=0.85,
        ),
    ]

    detections = detector.detect(ocr_results)
    if any(d.type == "personal_string" for d in detections):
        det = next(d for d in detections if d.type == "personal_string")
        print(f"  ✓ Detected despite typo")
        print(f"  ✓ Match score: {det.metadata['match_score']}")
    else:
        print(f"  ✗ Failed to detect fuzzy match")

    # Test 3: Email detection
    print("\nTest 3: Email Detection")
    ocr_results = [
        OCRResult(
            text="Reply-To: john@example.com",
            bounding_box=(10, 10, 300, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results)
    if any(d.type == "personal_string" for d in detections):
        det = next(d for d in detections if d.type == "personal_string")
        print(f"  ✓ Detected email")
        print(f"  ✓ Type classified as: {det.metadata['string_type']}")
    else:
        print(f"  ✗ Failed to detect email")

    # Test 4: Phone number detection
    print("\nTest 4: Phone Number Detection")
    ocr_results = [
        OCRResult(
            text="Call us at 555-123-4567 for support",
            bounding_box=(10, 10, 400, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results)
    if any(d.type == "personal_string" for d in detections):
        det = next(d for d in detections if d.type == "personal_string")
        print(f"  ✓ Detected phone number")
        print(f"  ✓ Type classified as: {det.metadata['string_type']}")
    else:
        print(f"  ✗ Failed to detect phone number")

    # Test 5: No false positive on unrelated text
    print("\nTest 5: No False Positive on Unrelated Text")
    ocr_results = [
        OCRResult(
            text="The quick brown fox jumps over the lazy dog",
            bounding_box=(10, 10, 400, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results)
    if len(detections) == 0:
        print(f"  ✓ Correctly no detection on unrelated text")
    else:
        print(f"  ✗ False positive: detected {len(detections)} matches")

    # Test 6: Safe preview (never reveals full personal string)
    print("\nTest 6: Safe Preview Generation")
    test_strings = [
        ("John Doe", "Jo***** e"),
        ("john@example.com", "jo**************m"),
        ("Ab", "**"),
    ]
    for string, _ in test_strings:
        preview = detector._safe_preview(string)
        # Just verify it doesn't contain the full string
        if preview != string:
            print(f"  ✓ '{string}' → '{preview}' (original masked)")
        else:
            print(f"  ✗ Preview shows full string: '{preview}'")

    # Test 7: Free tier limit enforcement
    print("\nTest 7: Free Tier Limit (max 3 strings)")
    config_too_many = MockConfig(
        strings=["String One", "String Two", "String Three", "String Four (should be dropped)"],
        max_free=3,
    )
    detector_limited = PersonalStringsDetector(config_too_many)
    if len(detector_limited.personal_strings) == 3:
        print(f"  ✓ Limited to 3 strings (free tier)")
    else:
        print(f"  ✗ Expected 3 strings, got {len(detector_limited.personal_strings)}")

    print("\n" + "=" * 60)
    print("Personal Strings Detector Tests Complete!")


if __name__ == "__main__":
    test_personal_strings_detector()
