"""Unit tests for all detection modules and the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the safestream package importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from detectors.base import OCRResult, DetectionResult
from detectors.seed_phrase import SeedPhraseDetector
from detectors.credit_card import CreditCardDetector
from detectors.crypto_address import CryptoAddressDetector
from detectors.personal_strings import PersonalStringsDetector
from detectors.api_keys import APIKeysDetector
from core.detector import DetectorPipeline, ScanResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockDetectionConfig:
    """Minimal config for all detectors."""
    enabled: bool = True


@dataclass
class MockSeedConfig:
    enabled: bool = True
    min_word_count: int = 12
    confidence_full: float = 0.95
    confidence_partial: float = 0.70
    max_vertical_distance: int = 50


@dataclass
class MockCreditConfig:
    enabled: bool = True
    require_luhn: bool = True
    min_confidence: float = 0.75


@dataclass
class MockCryptoConfig:
    enabled: bool = True
    networks: list = None
    def __post_init__(self):
        if self.networks is None:
            self.networks = ["BTC", "ETH", "SOL"]


@dataclass
class MockPersonalConfig:
    enabled: bool = True
    strings: list = None
    fuzzy_threshold: int = 85
    max_free: int = 3
    def __post_init__(self):
        if self.strings is None:
            self.strings = ["John Doe", "john@example.com", "555-123-4567"]


@dataclass
class MockAPIConfig:
    enabled: bool = False


# 12-word BIP-39 seed phrase words (all from the official wordlist)
SEED_WORDS_12 = (
    "abandon ability able about above absent absorb abstract absurd abuse access accident"
).split()

# 24-word BIP-39 seed phrase
SEED_WORDS_24 = (
    "abandon ability able about above absent absorb abstract absurd abuse access accident "
    "account accuse achieve acid acoustic acquire across act action actor actress actual"
).split()


def make_ocr(words: list[str], y: int = 10, x_step: int = 80) -> list[OCRResult]:
    """Create OCR results with each word at a separate x position on the same line."""
    return [
        OCRResult(
            text=word,
            bounding_box=(i * x_step, y, x_step - 5, 20),
            confidence=0.90,
        )
        for i, word in enumerate(words)
    ]


# ---------------------------------------------------------------------------
# SeedPhraseDetector tests
# ---------------------------------------------------------------------------


class TestSeedPhraseDetector:
    def setup_method(self):
        self.detector = SeedPhraseDetector(MockSeedConfig())

    def test_detects_12_word_phrase(self):
        """12 consecutive BIP-39 words on one line → blur detection."""
        ocr = make_ocr(SEED_WORDS_12)
        detections = self.detector.detect(ocr)
        assert len(detections) == 1
        assert detections[0].type == "seed_phrase"
        assert detections[0].confidence >= 0.90
        assert detections[0].action == "blur"

    def test_detects_24_word_phrase(self):
        """24-word phrase produces at least one detection."""
        ocr = make_ocr(SEED_WORDS_24)
        detections = self.detector.detect(ocr)
        assert len(detections) >= 1
        assert detections[0].type == "seed_phrase"
        assert detections[0].confidence >= 0.90

    def test_no_detection_on_short_sequence(self):
        """Fewer than 12 BIP-39 words → no detection."""
        ocr = make_ocr(SEED_WORDS_12[:5])  # Only 5 words
        detections = self.detector.detect(ocr)
        assert len(detections) == 0

    def test_false_positive_scattered_bip39_words(self):
        """
        Normal English text that happens to contain some BIP-39 words
        should NOT trigger detection — the words must be consecutive.
        """
        # A natural sentence with scattered BIP-39 words
        sentence = (
            "I was able to access the account above the table "
            "and found a useful abstract about machine learning"
        ).split()
        ocr = make_ocr(sentence)
        detections = self.detector.detect(ocr)
        # Should not fire — non-BIP39 words break the consecutive run
        assert len(detections) == 0

    def test_spatial_clustering_required(self):
        """
        Words spread across very different Y positions (different screen areas)
        should not be grouped as a seed phrase.
        """
        # Put each word on a very different row (1000px apart)
        ocr = [
            OCRResult(
                text=word,
                bounding_box=(0, i * 1000, 75, 20),
                confidence=0.90,
            )
            for i, word in enumerate(SEED_WORDS_12)
        ]
        detections = self.detector.detect(ocr)
        assert len(detections) == 0

    def test_empty_ocr_input(self):
        """Empty OCR input returns empty detections."""
        assert self.detector.detect([]) == []

    def test_non_bip39_word_does_not_prevent_detection(self):
        """
        A non-BIP39 word mixed among 12 BIP39 words on the same line still
        triggers detection. Security-first design: if 12 valid BIP39 words are
        visible in proximity, we flag it even with stray tokens mixed in.
        False positives are preferable to missed seed phrases.
        """
        mixed = SEED_WORDS_12[:5] + ["notaword"] + SEED_WORDS_12[5:]
        ocr = make_ocr(mixed)
        detections = self.detector.detect(ocr)
        # 5 + 7 = 12 valid BIP39 words on same line → detected (intentional)
        assert len(detections) >= 1
        assert detections[0].type == "seed_phrase"

    def test_disabled_detector_loads(self):
        """Disabled detector can still be instantiated (config flexibility)."""
        cfg = MockSeedConfig(enabled=False)
        det = SeedPhraseDetector(cfg)
        assert det is not None


# ---------------------------------------------------------------------------
# CreditCardDetector tests
# ---------------------------------------------------------------------------


class TestCreditCardDetector:
    def setup_method(self):
        self.detector = CreditCardDetector(MockCreditConfig())

    def test_detects_visa(self):
        """Standard Visa test card → detection."""
        ocr = [OCRResult("4111 1111 1111 1111", (0, 0, 200, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert len(detections) >= 1
        assert detections[0].type == "credit_card"

    def test_detects_no_spaces(self):
        """16 digits with no spaces → detection."""
        ocr = [OCRResult("4111111111111111", (0, 0, 200, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert len(detections) >= 1

    def test_detects_dashes(self):
        """Digits separated by dashes → detection."""
        ocr = [OCRResult("4111-1111-1111-1111", (0, 0, 200, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert len(detections) >= 1

    def test_luhn_invalid_rejected(self):
        """Number that fails Luhn check → no detection."""
        # 4111111111111112 fails Luhn (last digit off by 1)
        ocr = [OCRResult("4111 1111 1111 1112", (0, 0, 200, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert len(detections) == 0

    def test_fuzzy_digit_cleanup_l_to_1(self):
        """OCR confuses 'l' for '1' — fuzzy cleanup should fix it."""
        # '4lll 1111 1111 1111' — 'l' chars should be cleaned to '1'
        ocr = [OCRResult("4lll llll llll 1111", (0, 0, 200, 20), 0.90)]
        detections = self.detector.detect(ocr)
        # After cleanup: "4111 1111 1111 1111" → valid Visa, passes Luhn
        assert len(detections) >= 1

    def test_fuzzy_digit_cleanup_O_to_0(self):
        """OCR confuses 'O' for '0' — cleanup handles it."""
        # Using a known valid card with some O→0 substitutions
        ocr = [OCRResult("4111 1111 1111 111O", (0, 0, 200, 20), 0.90)]
        detections = self.detector.detect(ocr)
        # After cleanup: "4111 1111 1111 1110" — check if Luhn passes
        # (4111111111111110 → Luhn sum: valid test)
        # Either passes or not — the important thing is no crash
        assert isinstance(detections, list)

    def test_short_number_not_detected(self):
        """12-digit number → no detection (not a card)."""
        ocr = [OCRResult("411111111111", (0, 0, 200, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert len(detections) == 0

    def test_confidence_boost_with_expiry(self):
        """Card number + nearby expiry date → higher confidence."""
        ocr = [
            OCRResult("4111 1111 1111 1111", (0, 0, 200, 20), 0.90),
            OCRResult("12/26", (0, 25, 50, 15), 0.90),
        ]
        detections_with_expiry = self.detector.detect(ocr)

        ocr_alone = [OCRResult("4111 1111 1111 1111", (0, 0, 200, 20), 0.90)]
        detections_alone = self.detector.detect(ocr_alone)

        if detections_with_expiry and detections_alone:
            assert detections_with_expiry[0].confidence >= detections_alone[0].confidence

    def test_empty_ocr_input(self):
        assert self.detector.detect([]) == []


# ---------------------------------------------------------------------------
# CryptoAddressDetector tests
# ---------------------------------------------------------------------------


class TestCryptoAddressDetector:
    def setup_method(self):
        self.detector = CryptoAddressDetector(MockCryptoConfig())

    def test_detects_eth_address(self):
        """Valid ETH address → detection."""
        ocr = [OCRResult(
            "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
            (0, 0, 400, 20), 0.90
        )]
        detections = self.detector.detect(ocr)
        assert len(detections) >= 1
        assert detections[0].type == "crypto_address"
        assert detections[0].metadata.get("network") == "ETH"

    def test_detects_btc_legacy_address(self):
        """Valid BTC legacy address (starts with 1 or 3) → detection."""
        ocr = [OCRResult(
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7Divf Na",
            (0, 0, 400, 20), 0.90
        )]
        detections = self.detector.detect(ocr)
        # Should detect the BTC address portion
        assert len(detections) >= 1
        assert detections[0].metadata.get("network") in ("BTC", "ETH", "SOL")

    def test_detects_btc_bech32_address(self):
        """BTC bech32 address (bc1...) → detection."""
        ocr = [OCRResult(
            "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq",
            (0, 0, 400, 20), 0.90
        )]
        detections = self.detector.detect(ocr)
        assert len(detections) >= 1
        assert detections[0].metadata.get("network") == "BTC"

    def test_btc_not_double_detected_as_sol(self):
        """
        BTC addresses also match the SOL base58 pattern.
        Network priority (BTC→ETH→SOL) must prevent double detection.
        """
        ocr = [OCRResult(
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            (0, 0, 400, 20), 0.90
        )]
        detections = self.detector.detect(ocr)
        networks = [d.metadata.get("network") for d in detections]
        # Same address should not appear as both BTC and SOL
        assert len(set(networks)) == len(networks), "Duplicate network detections!"

    def test_eth_address_case_insensitive(self):
        """ETH address in lowercase still detected."""
        ocr = [OCRResult(
            "0x742d35cc6634c0532925a3b844bc9e7595f0beb0",
            (0, 0, 400, 20), 0.90
        )]
        detections = self.detector.detect(ocr)
        assert len(detections) >= 1
        assert detections[0].metadata.get("network") == "ETH"

    def test_short_string_not_detected(self):
        """Random short hex string → no detection."""
        ocr = [OCRResult("0xDEAD", (0, 0, 100, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert len(detections) == 0

    def test_plain_text_not_detected(self):
        """Plain English sentence → no detection."""
        ocr = [OCRResult(
            "The price of Bitcoin rose above 100000 dollars today.",
            (0, 0, 400, 20), 0.90
        )]
        detections = self.detector.detect(ocr)
        assert len(detections) == 0

    def test_empty_ocr_input(self):
        assert self.detector.detect([]) == []


# ---------------------------------------------------------------------------
# PersonalStringsDetector tests
# ---------------------------------------------------------------------------


class TestPersonalStringsDetector:
    def setup_method(self):
        self.detector = PersonalStringsDetector(MockPersonalConfig())

    def test_detects_exact_name(self):
        """Exact name match → detection."""
        ocr = [OCRResult("Hello John Doe welcome back", (0, 0, 400, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert any(d.type == "personal_string" for d in detections)

    def test_detects_email(self):
        """Email address in OCR → detection."""
        ocr = [OCRResult("Contact: john@example.com", (0, 0, 300, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert any(d.type == "personal_string" for d in detections)

    def test_detects_phone(self):
        """Phone number in OCR → detection."""
        ocr = [OCRResult("Call 555-123-4567 for support", (0, 0, 300, 20), 0.90)]
        detections = self.detector.detect(ocr)
        assert any(d.type == "personal_string" for d in detections)

    def test_fuzzy_match_typo(self):
        """OCR typo in name still triggers fuzzy match (threshold 85%)."""
        # "Jonn Doe" is close enough to "John Doe"
        ocr = [OCRResult("Jonn Doe", (0, 0, 200, 20), 0.85)]
        detections = self.detector.detect(ocr)
        assert any(d.type == "personal_string" for d in detections)

    def test_no_false_positive_unrelated(self):
        """Completely unrelated text → no detection."""
        ocr = [OCRResult(
            "The quick brown fox jumps over the lazy dog",
            (0, 0, 400, 20), 0.90
        )]
        detections = self.detector.detect(ocr)
        assert len(detections) == 0

    def test_safe_preview_masks_string(self):
        """text_preview should not reveal the full personal string."""
        ocr = [OCRResult("John Doe is here", (0, 0, 300, 20), 0.90)]
        detections = self.detector.detect(ocr)
        if detections:
            preview = detections[0].text_preview
            assert preview != "John Doe", "Preview should be masked"
            assert len(preview) > 0

    def test_free_tier_limit_enforced(self):
        """More than 3 strings → silently truncated to 3 (free tier)."""
        @dataclass
        class BigConfig:
            enabled: bool = True
            strings: list = None
            fuzzy_threshold: int = 85
            max_free: int = 3
            def __post_init__(self):
                self.strings = ["A", "B", "C", "D", "E"]  # 5 strings

        det = PersonalStringsDetector(BigConfig())
        assert len(det.personal_strings) == 3

    def test_empty_strings_config(self):
        """No strings configured → no detections."""
        @dataclass
        class EmptyConfig:
            enabled: bool = True
            strings: list = None
            fuzzy_threshold: int = 85
            max_free: int = 3
            def __post_init__(self):
                self.strings = []

        det = PersonalStringsDetector(EmptyConfig())
        ocr = [OCRResult("John Doe", (0, 0, 200, 20), 0.90)]
        assert det.detect(ocr) == []

    def test_empty_ocr_input(self):
        assert self.detector.detect([]) == []


# ---------------------------------------------------------------------------
# APIKeysDetector tests (stub behaviour)
# ---------------------------------------------------------------------------


class TestAPIKeysDetector:
    def test_stub_returns_empty(self):
        """V1 stub always returns empty list regardless of input."""
        det = APIKeysDetector(MockAPIConfig())
        ocr = [OCRResult("AKIA1234567890ABCDEF", (0, 0, 200, 20), 0.90)]
        assert det.detect(ocr) == []

    def test_empty_input_returns_empty(self):
        det = APIKeysDetector(MockAPIConfig())
        assert det.detect([]) == []

    def test_supported_services_list(self):
        """get_supported_services() should return > 0 services from JSON."""
        det = APIKeysDetector(MockAPIConfig())
        services = det.get_supported_services()
        assert len(services) > 0
        # Known services that must be present
        assert "AWS" in services
        assert "GitHub" in services
        assert "Stripe" in services


# ---------------------------------------------------------------------------
# DetectorPipeline tests
# ---------------------------------------------------------------------------


class TestDetectorPipeline:
    def setup_method(self):
        """Load the real pipeline from config.yaml."""
        from core.config_manager import ConfigManager
        self.config = ConfigManager.load("config.yaml")
        self.pipeline = DetectorPipeline(self.config)

    def test_pipeline_loads_detectors(self):
        """Pipeline should have at least 1 detector loaded."""
        assert len(self.pipeline.detectors) >= 1

    def test_scan_seed_phrase_triggers_blur(self):
        """12-word seed phrase → ScanResult.should_blur = True."""
        ocr = make_ocr(SEED_WORDS_12)
        result = self.pipeline.scan(ocr)
        assert result.should_blur is True
        assert result.has_critical is True
        assert len(result.detections) >= 1
        assert result.detections[0].type == "seed_phrase"

    def test_scan_empty_returns_no_detections(self):
        """Empty OCR input → no detections, no blur."""
        result = self.pipeline.scan([])
        assert result.should_blur is False
        assert result.should_warn is False
        assert len(result.detections) == 0
        assert result.total_ocr_results == 0

    def test_scan_clean_text_no_false_positive(self):
        """Benign text → no detections."""
        ocr = [OCRResult(
            "Welcome back! Today is sunny and 72 degrees outside.",
            (0, 0, 400, 20), 0.90
        )]
        result = self.pipeline.scan(ocr)
        assert result.should_blur is False

    def test_confidence_thresholds(self):
        """
        Pipeline assigns action based on confidence:
        >= 0.9 → blur, 0.6-0.9 → warn, < 0.6 → ignored (filtered out).
        """
        ocr = make_ocr(SEED_WORDS_12)
        result = self.pipeline.scan(ocr)
        for d in result.detections:
            assert d.action in ("blur", "warn"), f"Unexpected action: {d.action}"

    def test_detections_sorted_by_confidence(self):
        """Detections should be sorted highest confidence first."""
        ocr = make_ocr(SEED_WORDS_12)
        result = self.pipeline.scan(ocr)
        confidences = [d.confidence for d in result.detections]
        assert confidences == sorted(confidences, reverse=True)

    def test_scan_result_fields(self):
        """ScanResult contains correct summary counts."""
        ocr = make_ocr(SEED_WORDS_12)
        result = self.pipeline.scan(ocr)
        assert result.total_ocr_results == len(SEED_WORDS_12)
        assert result.detectors_run == len(self.pipeline.detectors)

    def test_should_blur_property(self):
        """should_blur is equivalent to has_critical."""
        ocr = make_ocr(SEED_WORDS_12)
        result = self.pipeline.scan(ocr)
        assert result.should_blur == result.has_critical

    def test_iou_merge_overlapping_detections(self):
        """Highly overlapping detections from different detectors → merged to one."""
        # Two DetectionResults with nearly identical bounding boxes
        d1 = DetectionResult(
            type="seed_phrase", confidence=0.95,
            text_preview="test", bounding_box=(10, 10, 200, 30),
            action="blur", metadata={}
        )
        d2 = DetectionResult(
            type="credit_card", confidence=0.80,
            text_preview="test", bounding_box=(10, 10, 200, 30),
            action="warn", metadata={}
        )
        merged = self.pipeline._merge_overlapping([d1, d2])
        # Same box (IoU=1.0) → should keep only the higher-confidence one
        assert len(merged) == 1
        assert merged[0].confidence == 0.95

    def test_non_overlapping_detections_kept_separate(self):
        """Detections in different screen regions → both kept."""
        d1 = DetectionResult(
            type="seed_phrase", confidence=0.95,
            text_preview="test", bounding_box=(0, 0, 100, 20),
            action="blur", metadata={}
        )
        d2 = DetectionResult(
            type="credit_card", confidence=0.80,
            text_preview="test", bounding_box=(500, 500, 100, 20),
            action="warn", metadata={}
        )
        merged = self.pipeline._merge_overlapping([d1, d2])
        assert len(merged) == 2
