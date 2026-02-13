"""BIP-39 seed phrase detection module."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import BaseDetector, DetectionResult, OCRResult


class SeedPhraseDetector(BaseDetector):
    """
    Detects BIP-39 seed phrases (12 or 24 word sequences).

    BIP-39 is the standard for cryptocurrency seed phrases used by
    Bitcoin, Ethereum, and most other cryptocurrencies.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize seed phrase detector.

        Args:
            config: Configuration object (Config.detection.seed_phrases)
        """
        super().__init__(config)

        # Load BIP-39 wordlist
        self.wordlist = self._load_wordlist()
        self.min_word_count = config.min_word_count if hasattr(config, "min_word_count") else 12

    def _load_wordlist(self) -> set[str]:
        """
        Load BIP-39 wordlist from file.

        Returns:
            Set of 2048 BIP-39 words for O(1) lookup

        Raises:
            FileNotFoundError: If wordlist file not found
        """
        # Find wordlist file (relative to this module)
        wordlist_path = Path(__file__).parent.parent / "data" / "bip39_wordlist.txt"

        if not wordlist_path.exists():
            raise FileNotFoundError(
                f"BIP-39 wordlist not found at {wordlist_path}. "
                "Run Task 6 to download it."
            )

        with open(wordlist_path, "r", encoding="utf-8") as f:
            words = {line.strip().lower() for line in f if line.strip()}

        if len(words) != 2048:
            raise ValueError(f"BIP-39 wordlist should have 2048 words, found {len(words)}")

        return words

    def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
        """
        Detect BIP-39 seed phrases in OCR results.

        Algorithm:
        1. Extract words from OCR results
        2. Filter to only BIP-39 wordlist words
        3. Find consecutive sequences of 12 or 24 words
        4. Verify spatial clustering (words close together)
        5. Score confidence based on exact match

        Args:
            ocr_results: List of OCR results from text detection

        Returns:
            List of detection results (empty if no seed phrases found)
        """
        if not ocr_results:
            return []

        # Step 1: Tokenize all OCR results into individual words
        word_results = self._tokenize_ocr_results(ocr_results)

        # Step 2: Filter to only BIP-39 words
        bip39_words = [w for w in word_results if w.text.lower() in self.wordlist]

        if len(bip39_words) < self.min_word_count:
            # Not enough BIP-39 words to form a seed phrase
            return []

        # Step 3: Find consecutive sequences
        sequences = self._find_consecutive_sequences(bip39_words)

        # Step 4: Validate and score each sequence
        detections: list[DetectionResult] = []

        for sequence in sequences:
            # Check spatial clustering
            if not self._are_spatially_clustered(sequence, max_distance=50):
                continue  # Words too far apart, likely false positive

            # Calculate confidence based on sequence length
            word_count = len(sequence)
            confidence = self._calculate_confidence(word_count)

            # Merge bounding boxes
            bounding_box = self._merge_bounding_boxes([w.bounding_box for w in sequence])

            # Create text preview (first and last word only)
            text_preview = f"{sequence[0].text}...{sequence[-1].text}"

            # Create detection result
            detection = DetectionResult(
                type="seed_phrase",
                confidence=confidence,
                text_preview=text_preview,
                bounding_box=bounding_box,
                action="blur" if confidence >= 0.9 else "warn",
                metadata={
                    "word_count": word_count,
                    "expected_count": 12 if word_count <= 15 else 24,
                    "all_words_valid": True,
                },
            )

            detections.append(detection)

        return detections

    def _tokenize_ocr_results(self, ocr_results: list[OCRResult]) -> list[OCRResult]:
        """
        Tokenize OCR results into individual words.

        Some OCR engines return multi-word strings. This splits them
        into individual word OCRResults while preserving spatial info.

        Args:
            ocr_results: List of OCR results

        Returns:
            List of OCR results, one per word
        """
        word_results: list[OCRResult] = []

        for ocr_result in ocr_results:
            # Split text into words
            words = re.findall(r"\b[a-z]+\b", ocr_result.text.lower())

            if not words:
                continue

            # If single word, keep original
            if len(words) == 1:
                word_results.append(
                    OCRResult(
                        text=words[0],
                        bounding_box=ocr_result.bounding_box,
                        confidence=ocr_result.confidence,
                    )
                )
            else:
                # Multiple words - estimate individual positions
                # Simple approach: divide width equally among words
                x, y, w, h = ocr_result.bounding_box
                word_width = w // len(words)

                for i, word in enumerate(words):
                    word_x = x + (i * word_width)
                    word_results.append(
                        OCRResult(
                            text=word,
                            bounding_box=(word_x, y, word_width, h),
                            confidence=ocr_result.confidence,
                        )
                    )

        return word_results

    def _find_consecutive_sequences(
        self, bip39_words: list[OCRResult]
    ) -> list[list[OCRResult]]:
        """
        Find consecutive sequences of BIP-39 words.

        Args:
            bip39_words: List of OCR results containing BIP-39 words

        Returns:
            List of sequences, each sequence is a list of OCR results
        """
        if len(bip39_words) < self.min_word_count:
            return []

        # Sort by position (top to bottom, left to right)
        sorted_words = sorted(bip39_words, key=lambda w: (w.y, w.x))

        sequences: list[list[OCRResult]] = []
        current_sequence: list[OCRResult] = [sorted_words[0]]

        for i in range(1, len(sorted_words)):
            prev = sorted_words[i - 1]
            curr = sorted_words[i]

            # Check if words are close enough to be part of same sequence
            # Allow words on same line or within 50px vertically
            vertical_distance = abs(curr.y - prev.y)
            horizontal_distance = abs(curr.x - prev.x)

            # Words should be on same line or next line (not too far apart)
            if vertical_distance <= 50:
                current_sequence.append(curr)
            else:
                # Too far apart - save current sequence if valid
                if len(current_sequence) >= self.min_word_count:
                    sequences.append(current_sequence)

                # Start new sequence
                current_sequence = [curr]

        # Don't forget the last sequence
        if len(current_sequence) >= self.min_word_count:
            sequences.append(current_sequence)

        return sequences

    def _calculate_confidence(self, word_count: int) -> float:
        """
        Calculate confidence score based on word count.

        Args:
            word_count: Number of words in sequence

        Returns:
            Confidence score (0.0-1.0)
        """
        # Exact match for 12 or 24 words = very high confidence
        if word_count == 12 or word_count == 24:
            return 0.95

        # Close to 12 or 24 = medium confidence
        if 10 <= word_count <= 14:  # Close to 12
            return 0.80
        elif 22 <= word_count <= 26:  # Close to 24
            return 0.80

        # 15-21 words = might be partial 24-word phrase
        if 15 <= word_count <= 21:
            return 0.70

        # Other counts = lower confidence (but still flag)
        return 0.60


# Standalone test function
def test_seed_phrase_detector() -> None:
    """Test the seed phrase detector with various scenarios."""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        min_word_count: int = 12

    config = MockConfig()
    detector = SeedPhraseDetector(config)

    print(f"✓ Loaded {len(detector.wordlist)} BIP-39 words")

    # Test 1: Valid 12-word seed phrase
    print("\nTest 1: Valid 12-word seed phrase")
    words_12 = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    ocr_results = [
        OCRResult(
            text=word,
            bounding_box=(i * 80, 10, 75, 20),
            confidence=0.9,
        )
        for i, word in enumerate(words_12.split())
    ]

    detections = detector.detect(ocr_results)
    if len(detections) == 1:
        det = detections[0]
        print(f"  ✓ Detected: {det.type}")
        print(f"  ✓ Confidence: {det.confidence}")
        print(f"  ✓ Word count: {det.metadata['word_count']}")
        print(f"  ✓ Action: {det.action}")
    else:
        print(f"  ✗ Expected 1 detection, got {len(detections)}")

    # Test 2: Valid 24-word seed phrase
    print("\nTest 2: Valid 24-word seed phrase")
    words_24 = words_12 + " acoustic acquire across act action actor actress actual adapt add addict address"
    ocr_results = [
        OCRResult(
            text=word,
            bounding_box=(i % 12 * 80, (i // 12) * 30, 75, 20),
            confidence=0.9,
        )
        for i, word in enumerate(words_24.split())
    ]

    detections = detector.detect(ocr_results)
    if len(detections) == 1:
        det = detections[0]
        print(f"  ✓ Detected: {det.type}")
        print(f"  ✓ Confidence: {det.confidence}")
        print(f"  ✓ Word count: {det.metadata['word_count']}")
    else:
        print(f"  ✗ Expected 1 detection, got {len(detections)}")

    # Test 3: False positive - scattered BIP-39 words in normal text
    print("\nTest 3: False positive prevention - scattered BIP-39 words")
    essay = "I have the ability to abandon my fears. The abstract concept of freedom allows me to access new opportunities."
    ocr_results = [
        OCRResult(
            text=word,
            bounding_box=(i * 80, 10, 75, 20),
            confidence=0.9,
        )
        for i, word in enumerate(essay.split())
    ]

    detections = detector.detect(ocr_results)
    if len(detections) == 0:
        print(f"  ✓ Correctly ignored (not enough consecutive BIP-39 words)")
    else:
        print(f"  ✗ False positive! Detected {len(detections)} seed phrases")

    # Test 4: Words too far apart vertically
    print("\nTest 4: Spatial clustering - words too far apart")
    words_scattered = words_12.split()
    ocr_results = [
        OCRResult(
            text=word,
            bounding_box=(10, i * 100, 75, 20),  # 100px apart vertically
            confidence=0.9,
        )
        for i, word in enumerate(words_scattered)
    ]

    detections = detector.detect(ocr_results)
    if len(detections) == 0:
        print(f"  ✓ Correctly ignored (words not spatially clustered)")
    else:
        print(f"  ✗ Should have ignored scattered words")

    # Test 5: Multi-word OCR result (tokenization test)
    print("\nTest 5: Tokenization of multi-word OCR results")
    ocr_results = [
        OCRResult(
            text="abandon ability able about",
            bounding_box=(10, 10, 300, 20),
            confidence=0.9,
        ),
        OCRResult(
            text="above absent absorb abstract",
            bounding_box=(10, 35, 300, 20),
            confidence=0.9,
        ),
        OCRResult(
            text="absurd abuse access accident",
            bounding_box=(10, 60, 300, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results)
    if len(detections) == 1:
        print(f"  ✓ Correctly tokenized multi-word OCR results")
        print(f"  ✓ Detected {detections[0].metadata['word_count']} words")
    else:
        print(f"  ✗ Tokenization failed")

    print("\n" + "=" * 50)
    print("Seed Phrase Detector Tests Complete!")


if __name__ == "__main__":
    test_seed_phrase_detector()
