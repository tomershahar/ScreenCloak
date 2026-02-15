"""Cryptocurrency address detection module."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseDetector, DetectionResult, OCRResult


class CryptoAddressDetector(BaseDetector):
    """
    Detects cryptocurrency wallet addresses for major networks.

    Supports:
    - Bitcoin (legacy P2PKH/P2SH and bech32 formats)
    - Ethereum (EIP-55 checksummed addresses)
    - Solana (base58 encoded addresses)
    """

    # Address patterns for major cryptocurrencies
    PATTERNS = {
        # Bitcoin: Legacy (1... or 3...) and SegWit (bc1...)
        "BTC": [
            r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",  # Legacy P2PKH (1...) and P2SH (3...)
            r"\bbc1[a-z0-9]{39,59}\b",  # Bech32 SegWit (bc1...)
        ],
        # Ethereum: 0x followed by 40 hex characters
        "ETH": [
            r"\b0x[a-fA-F0-9]{40}\b",
        ],
        # Solana: Base58 encoded, 32-44 characters
        "SOL": [
            r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b",
        ],
    }

    def __init__(self, config: Any) -> None:
        """
        Initialize crypto address detector.

        Args:
            config: Configuration object (Config.detection.crypto_addresses)
        """
        super().__init__(config)

        # Get enabled networks from config
        self.enabled_networks = (
            config.networks if hasattr(config, "networks") else ["BTC", "ETH", "SOL"]
        )

    def detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]:
        """
        Detect cryptocurrency addresses in OCR results.

        Algorithm:
        1. Combine all OCR text
        2. Search for address patterns (BTC, ETH, SOL)
        3. Validate address format
        4. Calculate confidence based on validation
        5. Create detection results

        Args:
            ocr_results: List of OCR results from text detection

        Returns:
            List of detection results (empty if no addresses found)
        """
        if not ocr_results:
            return []

        detections: list[DetectionResult] = []
        detected_addresses: set[str] = set()  # Track detected addresses to avoid duplicates

        # Combine all OCR text
        combined_text = " ".join([r.text for r in ocr_results])

        # Search for each enabled network (order matters - BTC before SOL to avoid collision)
        network_priority = ["BTC", "ETH", "SOL"]  # BTC first to avoid SOL collision
        ordered_networks = [n for n in network_priority if n in self.enabled_networks]
        ordered_networks.extend([n for n in self.enabled_networks if n not in network_priority])

        for network in ordered_networks:
            if network not in self.PATTERNS:
                continue  # Skip unknown networks

            # Try each pattern for this network
            for pattern in self.PATTERNS[network]:
                for match in re.finditer(pattern, combined_text):
                    address = match.group()

                    # Skip if already detected (prevents BTC/SOL collision)
                    if address in detected_addresses:
                        continue

                    # Validate address
                    is_valid, confidence = self._validate_address(address, network)

                    if not is_valid:
                        continue  # Skip invalid addresses

                    # Mark as detected
                    detected_addresses.add(address)

                    # Find bounding box for this address
                    bbox = self._find_bbox_for_text(address, ocr_results)

                    if not bbox:
                        continue  # Could not locate text

                    # Create text preview (first 10 and last 10 chars)
                    if len(address) > 20:
                        text_preview = f"{address[:10]}...{address[-10:]}"
                    else:
                        text_preview = address

                    # Create detection result
                    detection = DetectionResult(
                        type="crypto_address",
                        confidence=confidence,
                        text_preview=text_preview,
                        bounding_box=bbox,
                        action="blur" if confidence >= 0.9 else "warn",
                        metadata={
                            "network": network,
                            "address_format": self._get_address_format(address, network),
                            "address_length": len(address),
                        },
                    )

                    detections.append(detection)

        return detections

    def _validate_address(self, address: str, network: str) -> tuple[bool, float]:
        """
        Validate cryptocurrency address and return confidence score.

        Args:
            address: Address string
            network: Network name (BTC, ETH, SOL)

        Returns:
            Tuple of (is_valid, confidence_score)
        """
        if network == "BTC":
            return self._validate_btc_address(address)
        elif network == "ETH":
            return self._validate_eth_address(address)
        elif network == "SOL":
            return self._validate_sol_address(address)
        else:
            return False, 0.0

    def _validate_btc_address(self, address: str) -> tuple[bool, float]:
        """
        Validate Bitcoin address.

        Args:
            address: Bitcoin address

        Returns:
            Tuple of (is_valid, confidence_score)
        """
        # Legacy addresses (1... or 3...)
        if address[0] in ["1", "3"]:
            # Length check: 26-35 characters
            if 26 <= len(address) <= 35:
                # High confidence for properly formatted legacy addresses
                return True, 0.90

        # Bech32 SegWit addresses (bc1...)
        elif address.startswith("bc1"):
            # Length check: 42-62 characters (bc1 + 39-59 chars)
            if 42 <= len(address) <= 62:
                # Very high confidence for bech32 (easier to validate format)
                return True, 0.95

        return False, 0.0

    def _validate_eth_address(self, address: str) -> tuple[bool, float]:
        """
        Validate Ethereum address.

        Args:
            address: Ethereum address

        Returns:
            Tuple of (is_valid, confidence_score)
        """
        # Must start with 0x
        if not address.startswith("0x"):
            return False, 0.0

        # Must be exactly 42 characters (0x + 40 hex chars)
        if len(address) != 42:
            return False, 0.0

        # Check if all characters after 0x are hex
        hex_part = address[2:]
        if not all(c in "0123456789abcdefABCDEF" for c in hex_part):
            return False, 0.0

        # Basic format is valid
        confidence = 0.85

        # Boost confidence if it follows EIP-55 checksum format
        # (mixed case indicates checksummed address)
        has_uppercase = any(c.isupper() for c in hex_part)
        has_lowercase = any(c.islower() for c in hex_part)

        if has_uppercase and has_lowercase:
            # Likely checksummed (EIP-55)
            confidence = 0.95

        return True, confidence

    def _validate_sol_address(self, address: str) -> tuple[bool, float]:
        """
        Validate Solana address.

        Args:
            address: Solana address

        Returns:
            Tuple of (is_valid, confidence_score)
        """
        # Solana addresses are base58 encoded
        # Length: typically 32-44 characters
        if not (32 <= len(address) <= 44):
            return False, 0.0

        # Check if all characters are valid base58
        # Base58 alphabet: 123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz
        # (excludes 0, O, I, l to avoid confusion)
        base58_alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

        if not all(c in base58_alphabet for c in address):
            return False, 0.0

        # Valid base58 format
        # Medium confidence (base58 is used by other systems too)
        return True, 0.80

    def _get_address_format(self, address: str, network: str) -> str:
        """
        Get the specific address format.

        Args:
            address: Address string
            network: Network name

        Returns:
            Format description
        """
        if network == "BTC":
            if address.startswith("bc1"):
                return "bech32"
            elif address[0] == "1":
                return "P2PKH"
            elif address[0] == "3":
                return "P2SH"
            else:
                return "unknown"

        elif network == "ETH":
            # Check if checksummed (EIP-55)
            hex_part = address[2:]
            has_uppercase = any(c.isupper() for c in hex_part)
            has_lowercase = any(c.islower() for c in hex_part)

            if has_uppercase and has_lowercase:
                return "EIP-55"
            else:
                return "standard"

        elif network == "SOL":
            return "base58"

        return "unknown"

    def _find_bbox_for_text(
        self, text: str, ocr_results: list[OCRResult]
    ) -> tuple[int, int, int, int] | None:
        """
        Find bounding box for a given text string in OCR results.

        Args:
            text: Text to find
            ocr_results: List of OCR results

        Returns:
            Bounding box (x, y, w, h) or None if not found
        """
        matching_results: list[OCRResult] = []

        # Look for OCR results containing the address (or parts of it)
        for ocr_result in ocr_results:
            # Check if this OCR result contains the address
            # (or a significant portion of it - at least 10 chars)
            min_match_length = min(10, len(text) // 2)

            # Check for substring match
            if text in ocr_result.text or any(
                text[i : i + min_match_length] in ocr_result.text
                for i in range(len(text) - min_match_length + 1)
            ):
                matching_results.append(ocr_result)

        if not matching_results:
            return None

        # Merge bounding boxes of matching results
        return self._merge_bounding_boxes([r.bounding_box for r in matching_results])


# Standalone test function
def test_crypto_address_detector() -> None:
    """Test the crypto address detector with various scenarios."""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        networks: list[str] = None

        def __post_init__(self):
            if self.networks is None:
                self.networks = ["BTC", "ETH", "SOL"]

    config = MockConfig()
    detector = CryptoAddressDetector(config)

    print("Testing Crypto Address Detector\n" + "=" * 60)

    # Test 1: Bitcoin addresses
    print("\nTest 1: Bitcoin Addresses")

    btc_addresses = [
        ("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "P2PKH (legacy)", True),  # Genesis block
        ("3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy", "P2SH (legacy)", True),
        ("bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh", "Bech32 (SegWit)", True),
        ("notabitcoinaddress", "Invalid", False),
    ]

    for address, description, should_detect in btc_addresses:
        is_valid, confidence = detector._validate_btc_address(address)
        status = "✓" if (is_valid == should_detect) else "✗"
        print(f"  {status} {description}: {address[:20]}... - Valid: {is_valid}, Conf: {confidence}")

    # Test 2: Ethereum addresses
    print("\nTest 2: Ethereum Addresses")

    eth_addresses = [
        ("0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0", "EIP-55 checksummed", True),
        ("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", "EIP-55 checksummed", True),
        ("0x0000000000000000000000000000000000000000", "Zero address", True),
        ("0xnothexadecimal", "Invalid hex", False),
    ]

    for address, description, should_detect in eth_addresses:
        is_valid, confidence = detector._validate_eth_address(address)
        status = "✓" if (is_valid == should_detect) else "✗"
        print(f"  {status} {description}: {address[:20]}... - Valid: {is_valid}, Conf: {confidence}")

    # Test 3: Solana addresses
    print("\nTest 3: Solana Addresses")

    sol_addresses = [
        ("7EqQdEULxWcraVx3mXKFjc84LhCkMGZCkRuDpvcMwJeK", "Valid Solana", True),
        ("DRpbCBMxVnDK7maPM5tGv6MvB3v1sRMC86PZ8okm21hy", "Valid Solana", True),
        ("tooshort", "Too short", False),
        ("ContainsInvalidChars0OIl", "Invalid base58", False),
    ]

    for address, description, should_detect in sol_addresses:
        is_valid, confidence = detector._validate_sol_address(address)
        status = "✓" if (is_valid == should_detect) else "✗"
        print(f"  {status} {description}: {address[:20]}... - Valid: {is_valid}, Conf: {confidence}")

    # Test 4: Full detection with OCR results
    print("\nTest 4: Full Detection - Mixed Crypto Addresses")

    ocr_results = [
        OCRResult(
            text="BTC: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            bounding_box=(10, 10, 400, 20),
            confidence=0.9,
        ),
        OCRResult(
            text="ETH: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
            bounding_box=(10, 35, 400, 20),
            confidence=0.9,
        ),
        OCRResult(
            text="SOL: 7EqQdEULxWcraVx3mXKFjc84LhCkMGZCkRuDpvcMwJeK",
            bounding_box=(10, 60, 400, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results)

    if len(detections) == 3:
        print(f"  ✓ Detected all 3 addresses")
        for det in detections:
            print(f"      {det.metadata['network']}: {det.text_preview} (confidence: {det.confidence})")
    else:
        print(f"  ✗ Expected 3 detections, got {len(detections)}")
        for det in detections:
            print(f"      {det.metadata['network']}: {det.text_preview}")

    # Test 5: Bech32 Bitcoin address
    print("\nTest 5: Bitcoin Bech32 (SegWit) Address")

    ocr_results_bech32 = [
        OCRResult(
            text="bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
            bounding_box=(10, 10, 400, 20),
            confidence=0.9,
        ),
    ]

    detections = detector.detect(ocr_results_bech32)

    if len(detections) == 1:
        det = detections[0]
        print(f"  ✓ Detected bech32 address")
        print(f"      Format: {det.metadata['address_format']}")
        print(f"      Confidence: {det.confidence}")
        print(f"      Action: {det.action}")
    else:
        print(f"  ✗ Failed to detect bech32 address")

    print("\n" + "=" * 60)
    print("Crypto Address Detector Tests Complete!")


if __name__ == "__main__":
    test_crypto_address_detector()
