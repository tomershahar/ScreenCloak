"""Credit card detection module with Luhn algorithm validation."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseDetector, DetectionResult, OCRResult


class CreditCardDetector(BaseDetector):
    """
    Detects credit card numbers (16-digit sequences with Luhn validation).

    Supports major card networks: Visa, MasterCard, American Express, Discover, etc.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize credit card detector.

        Args:
            config: Configuration object (Config.detection.credit_cards)
        """
        super().__init__(config)

    def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
        """
        Detect credit card numbers in OCR results.

        Algorithm:
        1. Combine all OCR text
        2. Apply fuzzy digit cleanup (OCR noise handling)
        3. Find 16-digit sequences with regex
        4. Validate with Luhn algorithm
        5. Check for nearby expiration date (confidence boost)
        6. Create detection results

        Args:
            ocr_results: List of OCR results from text detection

        Returns:
            List of detection results (empty if no credit cards found)
        """
        if not ocr_results:
            return []

        # Combine all OCR text into a single string with positions
        combined_text = " ".join([r.text for r in ocr_results])

        # Apply fuzzy digit cleanup to handle OCR errors
        cleaned_text = self._fuzzy_digit_cleanup(combined_text)

        # Find candidate card numbers via two complementary methods:
        # 1. Regex: standard 4-4-4-4 grouping (handles well-spaced OCR output)
        # 2. Digit stream per-token: sliding 16-char window within a single OCR
        #    token (handles OCR that merges groups, e.g. "4111 1111" → "41111111")
        #    Applied PER TOKEN only — never across combined text, which would
        #    create false positives from concatenated timestamps/log dates.
        found_card_numbers: set[str] = set()

        pattern = r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        for match in re.finditer(pattern, cleaned_text):
            card_number = match.group().replace(" ", "").replace("-", "")
            found_card_numbers.add(card_number)

        # Digit-stream fallback: per-token only (never on combined text)
        for ocr_result in ocr_results:
            cleaned_token = self._fuzzy_digit_cleanup(ocr_result.text)
            token_digits = re.sub(r"\D", "", cleaned_token)
            if len(token_digits) >= 16:
                for i in range(len(token_digits) - 15):
                    found_card_numbers.add(token_digits[i : i + 16])

        detections: list[DetectionResult] = []

        for card_number in found_card_numbers:
            # Validate with Luhn algorithm
            if not self._luhn_check(card_number):
                continue

            # Find the bounding box (searches for 4-digit substrings in OCR tokens)
            bbox = self._find_bbox_for_text(card_number, ocr_results)
            if not bbox:
                continue

            # Check for nearby expiration date / CVV (confidence boost)
            nearby_text = self._get_nearby_text(bbox, ocr_results, radius=100)
            has_expiry = self._has_expiry_pattern(nearby_text)
            has_cvv = self._has_cvv_pattern(nearby_text)

            confidence = 0.80
            if has_expiry:
                confidence += 0.10
            if has_cvv:
                confidence += 0.05
            confidence = min(confidence, 0.95)

            detections.append(DetectionResult(
                type="credit_card",
                confidence=confidence,
                text_preview=f"{card_number[:4]}...{card_number[-4:]}",
                bounding_box=bbox,
                action="blur" if confidence >= 0.9 else "warn",
                metadata={
                    "card_length": len(card_number),
                    "has_expiry": has_expiry,
                    "has_cvv": has_cvv,
                    "card_network": self._detect_card_network(card_number),
                },
            ))

        return detections

    def _fuzzy_digit_cleanup(self, text: str) -> str:
        """
        Clean up common OCR misreads for digits.

        OCR often misreads similar-looking characters:
        - lowercase 'l' as '1'
        - uppercase 'I' as '1'
        - uppercase 'O' as '0'
        - uppercase 'S' as '5' (sometimes)

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text with fuzzy digit corrections
        """
        cleaned = text

        # Strategy: Replace characters that appear in digit-like patterns
        # Pattern 1: Surrounded by digits or spaces/dashes
        # Pattern 2: Repeated characters (llll → 1111)
        # Pattern 3: Mixed with digits (4lll → 4111)

        # Replace 'l' with '1' in digit contexts
        # Apply multiple passes to handle consecutive replacements
        for _ in range(3):  # Multiple passes to catch consecutive l's
            cleaned = re.sub(r'(\d)l', r'\g<1>1', cleaned)  # 4l → 41
            cleaned = re.sub(r'l(\d)', r'1\g<1>', cleaned)  # l4 → 14
            cleaned = re.sub(r'(1)l', r'\g<1>1', cleaned)  # 1l → 11 (after first replacement)
            cleaned = re.sub(r'l(1)', r'1\g<1>', cleaned)  # l1 → 11
        cleaned = re.sub(r'\bl+\b', lambda m: '1' * len(m.group()), cleaned)  # llll → 1111

        # Replace 'I' with '1' in digit contexts
        cleaned = re.sub(r'(\d[\s\-]*)I([\s\-]*\d)', r'\g<1>1\g<2>', cleaned)
        cleaned = re.sub(r'(\d)I', r'\g<1>1', cleaned)
        cleaned = re.sub(r'I(\d)', r'1\g<1>', cleaned)

        # Replace 'O' with '0' in digit contexts
        cleaned = re.sub(r'(\d[\s\-]*)O([\s\-]*\d)', r'\g<1>0\g<2>', cleaned)
        cleaned = re.sub(r'(\d)O', r'\g<1>0', cleaned)
        cleaned = re.sub(r'O(\d)', r'0\g<1>', cleaned)

        # Replace 'o' with '0' in digit contexts
        cleaned = re.sub(r'(\d[\s\-]*)o([\s\-]*\d)', r'\g<1>0\g<2>', cleaned)
        cleaned = re.sub(r'(\d)o', r'\g<1>0', cleaned)
        cleaned = re.sub(r'o(\d)', r'0\g<1>', cleaned)

        return cleaned

    def _luhn_check(self, card_number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm (mod 10 check).

        The Luhn algorithm:
        1. Starting from the rightmost digit, double every second digit
        2. If doubling results in a two-digit number, add the digits together
        3. Sum all the digits
        4. If the sum is divisible by 10, the number is valid

        Args:
            card_number: Card number string (digits only)

        Returns:
            True if valid according to Luhn algorithm, False otherwise
        """
        if not card_number.isdigit():
            return False

        if len(card_number) not in [13, 14, 15, 16, 19]:  # Valid card lengths
            return False

        # Luhn algorithm implementation
        total = 0
        reverse_digits = card_number[::-1]

        for i, digit in enumerate(reverse_digits):
            n = int(digit)

            # Double every second digit (starting from position 1)
            if i % 2 == 1:
                n *= 2
                # If result is two digits, add them together
                if n > 9:
                    n = n // 10 + n % 10

            total += n

        return total % 10 == 0

    def _find_bbox_for_text(
        self, text: str, ocr_results: list[OCRResult]
    ) -> tuple[int, int, int, int] | None:
        """
        Find bounding box for a given text string in OCR results.

        Args:
            text: Text to find (already cleaned)
            ocr_results: List of OCR results

        Returns:
            Bounding box (x, y, w, h) or None if not found
        """
        # Normalize search text (remove spaces/dashes for comparison)
        normalized_search = text.replace(" ", "").replace("-", "")

        # Look for OCR results containing parts of the card number
        matching_results: list[OCRResult] = []

        for ocr_result in ocr_results:
            # Apply fuzzy cleanup to OCR text before comparing (handles OCR noise)
            cleaned_ocr_text = self._fuzzy_digit_cleanup(ocr_result.text)
            normalized_ocr = cleaned_ocr_text.replace(" ", "").replace("-", "")

            # Check if at least 4 consecutive digits match
            if any(
                normalized_search[i : i + 4] in normalized_ocr
                for i in range(len(normalized_search) - 3)
            ):
                matching_results.append(ocr_result)

        if not matching_results:
            return None

        # Merge bounding boxes of matching results
        return self._merge_bounding_boxes([r.bounding_box for r in matching_results])

    def _get_nearby_text(
        self, bbox: tuple[int, int, int, int], ocr_results: list[OCRResult], radius: int = 100
    ) -> str:
        """
        Get text from OCR results near a given bounding box.

        Args:
            bbox: Bounding box (x, y, w, h)
            ocr_results: List of OCR results
            radius: Search radius in pixels

        Returns:
            Combined text from nearby OCR results
        """
        x, y, w, h = bbox
        center_x, center_y = x + w // 2, y + h // 2

        nearby_text_parts: list[str] = []

        for ocr_result in ocr_results:
            rx, ry, rw, rh = ocr_result.bounding_box
            result_center_x, result_center_y = rx + rw // 2, ry + rh // 2

            # Calculate distance from center to center
            distance = ((result_center_x - center_x) ** 2 + (result_center_y - center_y) ** 2) ** 0.5

            if distance <= radius:
                nearby_text_parts.append(ocr_result.text)

        return " ".join(nearby_text_parts)

    def _has_expiry_pattern(self, text: str) -> bool:
        """
        Check if text contains an expiration date pattern (MM/YY or MM/YYYY).

        Args:
            text: Text to search

        Returns:
            True if expiry pattern found
        """
        # Patterns: 12/25, 12/2025, 12-25, 12-2025
        expiry_pattern = r"\b(0[1-9]|1[0-2])[\s/\-](20)?\d{2}\b"
        return bool(re.search(expiry_pattern, text))

    def _has_cvv_pattern(self, text: str) -> bool:
        """
        Check if text contains a CVV pattern (3 or 4 digits with explicit label).

        Only matches when an explicit CVV/CVC/CV2 label is present.
        Standalone 3-4 digit numbers are NOT treated as CVV — port numbers,
        UI values, and other numeric UI elements would cause too many false positives.

        Args:
            text: Text to search

        Returns:
            True if CVV pattern found
        """
        cvv_pattern = r"\b(CVV|CVC|CV2)[\s:]?\d{3,4}\b"
        return bool(re.search(cvv_pattern, text, re.IGNORECASE))

    def _detect_card_network(self, card_number: str) -> str:
        """
        Detect card network based on first digits (IIN - Issuer Identification Number).

        Args:
            card_number: Card number string

        Returns:
            Card network name (Visa, MasterCard, Amex, Discover, etc.)
        """
        if not card_number:
            return "Unknown"

        first_digit = card_number[0]
        first_two = card_number[:2] if len(card_number) >= 2 else ""
        first_four = card_number[:4] if len(card_number) >= 4 else ""

        # Visa: starts with 4
        if first_digit == "4":
            return "Visa"

        # MasterCard: starts with 51-55 or 2221-2720
        if first_two in ["51", "52", "53", "54", "55"]:
            return "MasterCard"
        if first_two >= "22" and first_four <= "2720":
            return "MasterCard"

        # American Express: starts with 34 or 37
        if first_two in ["34", "37"]:
            return "Amex"

        # Discover: starts with 6011, 622126-622925, 644-649, 65
        if first_four == "6011" or first_two == "65":
            return "Discover"
        if first_two == "64" or first_two == "65":
            return "Discover"

        return "Unknown"


# Standalone test function
def test_credit_card_detector() -> None:
    """Test the credit card detector with various scenarios."""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        enabled: bool = True

    config = MockConfig()
    detector = CreditCardDetector(config)

    print("Testing Credit Card Detector\n" + "=" * 50)

    # Test 1: Luhn algorithm validation
    print("\nTest 1: Luhn Algorithm Validation")
    valid_cards = [
        "4111111111111111",  # Visa (standard test card)
        "5425233430109903",  # MasterCard
        "374245455400126",  # Amex (15 digits)
    ]

    for card in valid_cards:
        is_valid = detector._luhn_check(card)
        network = detector._detect_card_network(card)
        print(f"  {card[:4]}...{card[-4:]} ({network}): {'✓ Valid' if is_valid else '✗ Invalid'}")

    # Invalid card (fails Luhn)
    invalid_card = "4111111111111112"  # Last digit changed
    is_valid = detector._luhn_check(invalid_card)
    print(f"  {invalid_card[:4]}...{invalid_card[-4:]} (Invalid): {'✓ Valid' if is_valid else '✗ Invalid (expected)'}")

    # Test 2: Fuzzy digit cleanup
    print("\nTest 2: Fuzzy Digit Cleanup (OCR Noise)")
    noisy_text = "Card: 4lll llll llll llll"  # l → 1
    cleaned = detector._fuzzy_digit_cleanup(noisy_text)
    print(f"  Original: {noisy_text}")
    print(f"  Cleaned:  {cleaned}")
    print(f"  Expected: Card: 4111 1111 1111 1111")

    # Test 3: Full detection with valid card
    print("\nTest 3: Full Detection - Valid Card")
    ocr_results = [
        OCRResult(
            text="Card Number: 4111 1111 1111 1111",
            bounding_box=(10, 10, 300, 20),
            confidence=0.9,
        ),
        OCRResult(
            text="Exp: 12/25",
            bounding_box=(10, 35, 100, 20),
            confidence=0.9,
        ),
        OCRResult(
            text="CVV: 123",
            bounding_box=(120, 35, 80, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results)
    if len(detections) == 1:
        det = detections[0]
        print(f"  ✓ Detected: {det.type}")
        print(f"  ✓ Confidence: {det.confidence}")
        print(f"  ✓ Preview: {det.text_preview}")
        print(f"  ✓ Network: {det.metadata['card_network']}")
        print(f"  ✓ Has Expiry: {det.metadata['has_expiry']}")
        print(f"  ✓ Has CVV: {det.metadata['has_cvv']}")
        print(f"  ✓ Action: {det.action}")
    else:
        print(f"  ✗ Expected 1 detection, got {len(detections)}")

    # Test 4: Detection with OCR noise
    print("\nTest 4: Detection with OCR Noise (l → 1, O → 0)")
    ocr_results_noisy = [
        OCRResult(
            text="4lll llll llll llll",  # Has OCR errors (l instead of 1)
            bounding_box=(10, 10, 200, 20),
            confidence=0.85,
        ),
    ]

    detections = detector.detect(ocr_results_noisy)
    if len(detections) == 1:
        print(f"  ✓ Detected despite OCR noise!")
        print(f"  ✓ Preview: {detections[0].text_preview}")
    else:
        print(f"  ✗ Failed to detect (fuzzy cleanup may need improvement)")

    # Test 5: Invalid card (fails Luhn)
    print("\nTest 5: Invalid Card (Luhn Check)")
    ocr_results_invalid = [
        OCRResult(
            text="4111 1111 1111 1112",  # Last digit wrong
            bounding_box=(10, 10, 200, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results_invalid)
    if len(detections) == 0:
        print(f"  ✓ Correctly rejected invalid card (Luhn check failed)")
    else:
        print(f"  ✗ Should have rejected invalid card")

    print("\n" + "=" * 50)
    print("Credit Card Detector Tests Complete!")


if __name__ == "__main__":
    test_credit_card_detector()
