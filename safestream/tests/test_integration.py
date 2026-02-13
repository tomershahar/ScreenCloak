"""
Integration tests — full OCR → Detector pipeline using synthetic test images.

These tests load real PNG images from data/test_images/, run TesseractEngine
on each, then pass the results through DetectorPipeline. They verify:

  - Seed phrase images  → should_blur=True, type=seed_phrase
  - Credit card image   → detection with type=credit_card
  - ETH address image   → detection with type=crypto_address
  - False positive essay → no actionable detections
  - Mixed content image → detections for both credit_card and crypto_address

All tests are skipped automatically if:
  - Tesseract is not installed  (pytesseract ImportError / binary missing)
  - Test images are missing     (data/test_images/ not found)

Run from the safestream/ directory:
    pytest tests/test_integration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Path setup — make safestream packages importable when run directly
# ---------------------------------------------------------------------------

_SAFESTREAM_DIR = Path(__file__).parent.parent
if str(_SAFESTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(_SAFESTREAM_DIR))

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

# Skip entire module if pytesseract isn't installed
pytesseract = pytest.importorskip(
    "pytesseract",
    reason="pytesseract not installed — skipping integration tests",
)

# Check that the Tesseract binary is actually present
try:
    pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
except Exception:
    _TESSERACT_AVAILABLE = False

require_tesseract = pytest.mark.skipif(
    not _TESSERACT_AVAILABLE,
    reason="Tesseract binary not found — skipping integration tests",
)

# Test image directory
_IMAGE_DIR = _SAFESTREAM_DIR / "data" / "test_images"
require_images = pytest.mark.skipif(
    not _IMAGE_DIR.is_dir(),
    reason="data/test_images/ not found — run tests/generate_test_images.py first",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ocr_engine():
    """TesseractEngine instance, shared across all integration tests."""
    from core.ocr_engine import TesseractEngine

    return TesseractEngine()


@pytest.fixture(scope="module")
def pipeline():
    """DetectorPipeline with default config, shared across all integration tests."""
    from core.config_manager import ConfigManager
    from core.detector import DetectorPipeline

    config = ConfigManager.load(str(_SAFESTREAM_DIR / "config.yaml"))
    return DetectorPipeline(config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_image(filename: str) -> np.ndarray:
    """Load a test PNG as a numpy RGB array."""
    path = _IMAGE_DIR / filename
    if not path.exists():
        pytest.skip(f"Test image missing: {filename} — run generate_test_images.py")
    return np.array(PILImage.open(path))


def _run_pipeline(ocr_engine, pipeline, filename: str):
    """Load image, run OCR, run detector pipeline, return ScanResult."""
    image = _load_image(filename)
    ocr_results = ocr_engine.detect_text(image)
    return pipeline.scan(ocr_results), ocr_results


def _detection_types(scan_result) -> List[str]:
    return [d.type for d in scan_result.detections]


# ---------------------------------------------------------------------------
# Tests: seed phrase images
# ---------------------------------------------------------------------------


@require_tesseract
@require_images
class TestSeedPhraseImages:
    """Seed phrase images should always trigger blur."""

    def test_12word_triggers_blur(self, ocr_engine, pipeline):
        """12-word BIP-39 seed phrase on a single line → should_blur=True."""
        result, ocr = _run_pipeline(ocr_engine, pipeline, "seed_phrase_12word.png")

        assert len(ocr) > 0, "Tesseract found no text — image may be unreadable"
        assert result.should_blur, (
            f"Expected blur for 12-word seed phrase; "
            f"detections={_detection_types(result)}"
        )
        assert any(d.type == "seed_phrase" for d in result.detections), (
            f"Expected seed_phrase detection; got {_detection_types(result)}"
        )

    def test_12word_detection_confidence(self, ocr_engine, pipeline):
        """12-word image should produce high-confidence seed phrase detection."""
        result, _ = _run_pipeline(ocr_engine, pipeline, "seed_phrase_12word.png")

        seed_detections = [d for d in result.detections if d.type == "seed_phrase"]
        assert seed_detections, "No seed_phrase detections"
        top = seed_detections[0]
        assert top.confidence >= 0.9, (
            f"Expected confidence ≥ 0.9 for clear seed phrase; got {top.confidence:.2f}"
        )

    def test_24word_triggers_blur(self, ocr_engine, pipeline):
        """24-word BIP-39 seed phrase across two lines → should_blur=True."""
        result, ocr = _run_pipeline(ocr_engine, pipeline, "seed_phrase_24word.png")

        assert len(ocr) > 0, "Tesseract found no text — image may be unreadable"
        assert result.should_blur, (
            f"Expected blur for 24-word seed phrase; "
            f"detections={_detection_types(result)}"
        )
        assert any(d.type == "seed_phrase" for d in result.detections)

    def test_24word_detection_action(self, ocr_engine, pipeline):
        """24-word seed phrase detection should have action=blur."""
        result, _ = _run_pipeline(ocr_engine, pipeline, "seed_phrase_24word.png")

        seed_detections = [d for d in result.detections if d.type == "seed_phrase"]
        assert seed_detections, "No seed_phrase detections"
        assert seed_detections[0].action == "blur"


# ---------------------------------------------------------------------------
# Tests: credit card image
# ---------------------------------------------------------------------------


@require_tesseract
@require_images
class TestCreditCardImage:
    """Visa card image should trigger credit_card detection."""

    def test_visa_card_is_detected(self, ocr_engine, pipeline):
        """Visa 4111... test card should be flagged as a credit card."""
        result, ocr = _run_pipeline(ocr_engine, pipeline, "credit_card_visa.png")

        assert len(ocr) > 0, "Tesseract found no text"

        # We accept blur OR warn — both mean the card was caught
        assert result.should_blur or result.should_warn, (
            f"Expected credit card detection; detections={_detection_types(result)}"
        )
        assert any(d.type == "credit_card" for d in result.detections), (
            f"Expected credit_card type; got {_detection_types(result)}"
        )

    def test_visa_card_confidence(self, ocr_engine, pipeline):
        """Clearly formatted Visa card with expiry should have high confidence."""
        result, _ = _run_pipeline(ocr_engine, pipeline, "credit_card_visa.png")

        cc_detections = [d for d in result.detections if d.type == "credit_card"]
        assert cc_detections, "No credit_card detections"
        # Confidence should be ≥ 0.6 (at least warn-level)
        assert cc_detections[0].confidence >= 0.6, (
            f"Credit card confidence too low: {cc_detections[0].confidence:.2f}"
        )


# ---------------------------------------------------------------------------
# Tests: crypto address image
# ---------------------------------------------------------------------------


@require_tesseract
@require_images
class TestCryptoAddressImage:
    """ETH address image should trigger crypto_address detection."""

    def test_eth_address_is_detected(self, ocr_engine, pipeline):
        """A valid ETH address on a plain background should be detected."""
        result, ocr = _run_pipeline(ocr_engine, pipeline, "eth_address.png")

        assert len(ocr) > 0, "Tesseract found no text"
        assert result.should_blur or result.should_warn, (
            f"Expected ETH address detection; detections={_detection_types(result)}"
        )
        assert any(d.type == "crypto_address" for d in result.detections), (
            f"Expected crypto_address type; got {_detection_types(result)}"
        )

    def test_eth_address_network_metadata(self, ocr_engine, pipeline):
        """ETH address detection should carry network=ETH in metadata."""
        result, _ = _run_pipeline(ocr_engine, pipeline, "eth_address.png")

        eth_detections = [
            d for d in result.detections if d.type == "crypto_address"
        ]
        assert eth_detections, "No crypto_address detections"

        top = eth_detections[0]
        network = top.metadata.get("network", "").upper() if top.metadata else ""
        assert network == "ETH", (
            f"Expected network=ETH; got {network!r}. "
            f"metadata={top.metadata}"
        )


# ---------------------------------------------------------------------------
# Tests: false positive essay
# ---------------------------------------------------------------------------


@require_tesseract
@require_images
class TestFalsePositiveEssay:
    """
    Normal English text with scattered BIP-39 words should NOT trigger
    any detection. This is the false-positive (precision) test.
    """

    def test_essay_has_no_actionable_detections(self, ocr_engine, pipeline):
        """Scattered BIP-39 words in natural text should produce no detections."""
        result, ocr = _run_pipeline(ocr_engine, pipeline, "false_positive_essay.png")

        assert len(ocr) > 0, (
            "Tesseract found no text in false_positive_essay.png — "
            "the image may be unreadable and this test is not meaningful"
        )

        # Key assertion: no actionable (blur or warn) detections
        assert not result.should_blur, (
            f"False positive — essay triggered blur: {_detection_types(result)}"
        )
        assert not result.should_warn, (
            f"False positive — essay triggered warn: {_detection_types(result)}"
        )

    def test_essay_ocr_reads_bip39_words(self, ocr_engine, pipeline):
        """Verify Tesseract actually reads the essay (sanity check for the test above)."""
        image = _load_image("false_positive_essay.png")
        ocr_results = ocr_engine.detect_text(image)

        all_text = " ".join(r.text.lower() for r in ocr_results)
        # Words 'able' and 'access' appear in the essay; confirm OCR sees them
        assert "able" in all_text or "access" in all_text, (
            f"OCR failed to read expected BIP-39 words from essay. "
            f"Found: {all_text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: mixed content image
# ---------------------------------------------------------------------------


@require_tesseract
@require_images
class TestMixedContentImage:
    """
    Image containing both a Visa card and an ETH address.
    Expects detections for at least one of the two sensitive items.
    """

    def test_mixed_image_triggers_detection(self, ocr_engine, pipeline):
        """At least one sensitive item should be detected."""
        result, ocr = _run_pipeline(ocr_engine, pipeline, "mixed_content.png")

        assert len(ocr) > 0, "Tesseract found no text"
        assert result.should_blur or result.should_warn, (
            "Expected at least one detection in mixed content image"
        )

    def test_mixed_image_detects_credit_card(self, ocr_engine, pipeline):
        """Mixed content image should detect the Visa card number."""
        result, _ = _run_pipeline(ocr_engine, pipeline, "mixed_content.png")

        assert any(d.type == "credit_card" for d in result.detections), (
            f"Expected credit_card detection in mixed content; "
            f"got {_detection_types(result)}"
        )

    def test_mixed_image_detects_eth_address(self, ocr_engine, pipeline):
        """Mixed content image should detect the ETH wallet address."""
        result, _ = _run_pipeline(ocr_engine, pipeline, "mixed_content.png")

        assert any(d.type == "crypto_address" for d in result.detections), (
            f"Expected crypto_address detection in mixed content; "
            f"got {_detection_types(result)}"
        )

    def test_mixed_image_multiple_detections(self, ocr_engine, pipeline):
        """Mixed content should produce at least 2 distinct detection types."""
        result, _ = _run_pipeline(ocr_engine, pipeline, "mixed_content.png")

        detected_types = set(_detection_types(result))
        assert len(detected_types) >= 2, (
            f"Expected ≥ 2 detection types in mixed image; "
            f"got {detected_types}"
        )


# ---------------------------------------------------------------------------
# Tests: OCR smoke tests (standalone — no pipeline)
# ---------------------------------------------------------------------------


@require_tesseract
@require_images
class TestOCRSmokeTests:
    """
    Verify that TesseractEngine can read key tokens from each test image.
    These tests catch font / resolution regressions in generate_test_images.py.
    """

    @pytest.mark.parametrize(
        "filename,expected_tokens",
        [
            ("seed_phrase_12word.png", ["abandon", "ability", "accident"]),
            ("seed_phrase_24word.png", ["abandon", "actual"]),
            ("credit_card_visa.png",   ["4111"]),
            ("eth_address.png",        ["0x742d"]),
            ("false_positive_essay.png", ["able", "access"]),
            ("mixed_content.png",      ["4111", "0x742d"]),
        ],
    )
    def test_ocr_reads_expected_tokens(
        self, ocr_engine, filename: str, expected_tokens: List[str]
    ):
        """All expected tokens should appear in OCR output for each image."""
        image = _load_image(filename)
        ocr_results = ocr_engine.detect_text(image)

        found_text = " ".join(r.text.lower() for r in ocr_results)

        missing = [
            token for token in expected_tokens
            if token.lower() not in found_text
        ]
        assert not missing, (
            f"{filename}: Tesseract missed token(s) {missing}. "
            f"OCR found: {found_text[:300]!r}"
        )
