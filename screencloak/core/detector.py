"""Detection pipeline — coordinates all detector modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from detectors.api_keys import APIKeysDetector
from detectors.base import DetectionResult, OCRResult
from detectors.credit_card import CreditCardDetector
from detectors.crypto_address import CryptoAddressDetector
from detectors.personal_strings import PersonalStringsDetector
from detectors.seed_phrase import SeedPhraseDetector

logger = logging.getLogger("screencloak.detector")


@dataclass
class ScanResult:
    """Result from a full pipeline scan."""

    detections: list[DetectionResult]
    total_ocr_results: int
    detectors_run: int
    has_critical: bool  # Any detection with action="blur"
    has_warnings: bool  # Any detection with action="warn"

    @property
    def should_blur(self) -> bool:
        """True if OBS should switch to privacy scene."""
        return self.has_critical

    @property
    def should_warn(self) -> bool:
        """True if a warning should be logged."""
        return self.has_warnings


class DetectorPipeline:
    """
    Coordinates all detection modules into a single scan.

    Runs each enabled detector against OCR results, aggregates
    findings, applies confidence thresholds, and returns
    prioritised detections.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialise pipeline with all enabled detectors.

        Args:
            config: Main Config object
        """
        self.config = config
        self.detectors: list[Any] = []
        self._load_detectors()

    def _load_detectors(self) -> None:
        """Load all enabled detectors from config."""
        detection = self.config.detection

        if detection.seed_phrases.enabled:
            self.detectors.append(SeedPhraseDetector(detection.seed_phrases))
            logger.debug("Loaded SeedPhraseDetector")

        if detection.credit_cards.enabled:
            self.detectors.append(CreditCardDetector(detection.credit_cards))
            logger.debug("Loaded CreditCardDetector")

        if detection.crypto_addresses.enabled:
            self.detectors.append(CryptoAddressDetector(detection.crypto_addresses))
            logger.debug("Loaded CryptoAddressDetector")

        if detection.personal_strings.enabled and detection.personal_strings.strings:
            self.detectors.append(PersonalStringsDetector(detection.personal_strings))
            logger.debug("Loaded PersonalStringsDetector")

        if detection.api_keys.enabled:
            self.detectors.append(APIKeysDetector(detection.api_keys))
            logger.debug("Loaded APIKeysDetector")

        logger.info(f"Detector pipeline ready: {len(self.detectors)} detectors loaded")

    def scan(self, ocr_results: list[OCRResult]) -> ScanResult:
        """
        Run all enabled detectors against OCR results.

        Algorithm:
        1. Run each detector
        2. Aggregate all detections
        3. Apply confidence thresholds (blur / warn / ignore)
        4. Merge spatially overlapping detections
        5. Sort by confidence (highest first)

        Args:
            ocr_results: List of OCR results from text detection

        Returns:
            ScanResult with all detections and summary flags
        """
        if not ocr_results:
            return ScanResult(
                detections=[],
                total_ocr_results=0,
                detectors_run=len(self.detectors),
                has_critical=False,
                has_warnings=False,
            )

        all_detections: list[DetectionResult] = []

        # Run each detector
        for detector in self.detectors:
            try:
                detections = detector.detect(ocr_results)
                all_detections.extend(detections)
            except Exception as e:
                logger.error(
                    f"Detector {detector.__class__.__name__} raised an error: {e}",
                    exc_info=True,
                )
                # Continue — one broken detector should not stop others

        # Apply confidence-based action assignment (thresholds from config)
        blur_threshold = self.config.detection.thresholds.blur
        warn_threshold = self.config.detection.thresholds.warn
        for detection in all_detections:
            if detection.confidence >= blur_threshold:
                detection.action = "blur"
            elif detection.confidence >= warn_threshold:
                detection.action = "warn"
            else:
                detection.action = "ignore"

        # Filter ignored detections
        actionable = [d for d in all_detections if d.action != "ignore"]

        # Merge spatially overlapping detections
        actionable = self._merge_overlapping(actionable)

        # Sort by confidence descending (most critical first)
        actionable.sort(key=lambda d: d.confidence, reverse=True)

        return ScanResult(
            detections=actionable,
            total_ocr_results=len(ocr_results),
            detectors_run=len(self.detectors),
            has_critical=any(d.action == "blur" for d in actionable),
            has_warnings=any(d.action == "warn" for d in actionable),
        )

    def _merge_overlapping(
        self, detections: list[DetectionResult]
    ) -> list[DetectionResult]:
        """
        Merge detections whose bounding boxes significantly overlap.

        Prevents duplicate alerts when multiple detectors flag the
        same region (e.g., a credit card number also matching a
        personal string pattern).

        Args:
            detections: List of detections to merge

        Returns:
            De-duplicated list of detections
        """
        if len(detections) <= 1:
            return detections

        merged: list[DetectionResult] = []
        used: set[int] = set()

        for i, det_a in enumerate(detections):
            if i in used:
                continue

            group = [det_a]

            for j, det_b in enumerate(detections):
                if j <= i or j in used:
                    continue
                if self._boxes_overlap(det_a.bounding_box, det_b.bounding_box):
                    group.append(det_b)
                    used.add(j)

            # Keep the highest-confidence detection from each overlapping group
            best = max(group, key=lambda d: d.confidence)
            merged.append(best)
            used.add(i)

        return merged

    def _boxes_overlap(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
        threshold: float = 0.5,
    ) -> bool:
        """
        Check if two bounding boxes overlap significantly.

        Args:
            box_a: First bounding box (x, y, w, h)
            box_b: Second bounding box (x, y, w, h)
            threshold: Minimum IoU (intersection over union) to count as overlap

        Returns:
            True if boxes overlap above threshold
        """
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        # Calculate intersection
        ix = max(ax, bx)
        iy = max(ay, by)
        iw = min(ax + aw, bx + bw) - ix
        ih = min(ay + ah, by + bh) - iy

        if iw <= 0 or ih <= 0:
            return False  # No intersection

        intersection = iw * ih
        union = aw * ah + bw * bh - intersection

        if union == 0:
            return False

        iou = intersection / union
        return iou >= threshold


# Standalone test
def test_detector_pipeline() -> None:
    """Test the detector pipeline end-to-end."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from core.config_manager import ConfigManager

    print("Testing Detector Pipeline\n" + "=" * 60)

    # Load real config
    config = ConfigManager.load("config.yaml")
    pipeline = DetectorPipeline(config)
    print(f"\n✓ Pipeline loaded with {len(pipeline.detectors)} detectors")

    # Test 1: Seed phrase detection
    print("\nTest 1: Seed Phrase Triggers Blur")
    words = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    ocr_results = [
        OCRResult(
            text=word,
            bounding_box=(i * 80, 10, 75, 20),
            confidence=0.9,
        )
        for i, word in enumerate(words.split())
    ]

    result = pipeline.scan(ocr_results)
    if result.should_blur:
        det = result.detections[0]
        print(f"  ✓ should_blur=True")
        print(f"  ✓ Detection type: {det.type}")
        print(f"  ✓ Confidence: {det.confidence}")
        print(f"  ✓ Action: {det.action}")
    else:
        print(f"  ✗ Expected blur trigger")

    # Test 2: Credit card triggers blur
    print("\nTest 2: Credit Card Triggers Blur")
    ocr_results = [
        OCRResult(
            text="4111 1111 1111 1111",
            bounding_box=(10, 10, 200, 20),
            confidence=0.9,
        ),
    ]

    result = pipeline.scan(ocr_results)
    if result.has_critical or result.has_warnings:
        print(f"  ✓ Flagged: {result.detections[0].type} (action={result.detections[0].action})")
    else:
        print(f"  ✗ Expected credit card detection")

    # Test 3: ETH address
    print("\nTest 3: ETH Address Triggers Blur")
    ocr_results = [
        OCRResult(
            text="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
            bounding_box=(10, 10, 400, 20),
            confidence=0.9,
        ),
    ]

    result = pipeline.scan(ocr_results)
    if result.should_blur or result.should_warn:
        det = result.detections[0]
        print(f"  ✓ Flagged: {det.type} ({det.metadata.get('network', '')}) action={det.action}")
    else:
        print(f"  ✗ Expected ETH address detection")

    # Test 4: Clean text — no detections
    print("\nTest 4: Clean Text — No Detections")
    ocr_results = [
        OCRResult(
            text="Welcome back! Today's weather is sunny and 72 degrees.",
            bounding_box=(10, 10, 400, 20),
            confidence=0.9,
        ),
    ]

    result = pipeline.scan(ocr_results)
    if not result.should_blur and not result.should_warn:
        print(f"  ✓ No detections on clean text")
    else:
        print(f"  ✗ False positive: {[d.type for d in result.detections]}")

    # Test 5: Empty OCR results
    print("\nTest 5: Empty OCR Results — Handled Gracefully")
    result = pipeline.scan([])
    if len(result.detections) == 0 and not result.should_blur:
        print(f"  ✓ Empty input handled gracefully")
    else:
        print(f"  ✗ Unexpected result on empty input")

    print("\n" + "=" * 60)
    print("Detector Pipeline Tests Complete!")


if __name__ == "__main__":
    test_detector_pipeline()
