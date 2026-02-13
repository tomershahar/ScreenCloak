"""
Generate synthetic test images for SafeStream detection tests.

All images use a system font at readable size (28-40pt) so both
TesseractEngine and PaddleOCREngine can read them reliably.

Images generated:
  seed_phrase_12word.png   — 12 BIP-39 words on one line
  seed_phrase_24word.png   — 24 BIP-39 words across two lines
  credit_card_visa.png     — Visa test card 4111 1111 1111 1111
  eth_address.png          — Real ETH address format
  false_positive_essay.png — Normal text with scattered BIP-39 words (no 12 consecutive)
  mixed_content.png        — Both credit card and ETH address visible
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the safestream package importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "test_images"

# System fonts — try in order, use first found
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",         # Linux
    "C:/Windows/Fonts/arial.ttf",                               # Windows
]

# BIP-39 words used for test images
SEED_WORDS_12 = (
    "abandon ability able about above absent "
    "absorb abstract absurd abuse access accident"
).split()

SEED_WORDS_24 = (
    "abandon ability able about above absent absorb abstract absurd abuse access accident "
    "account accuse achieve acid acoustic acquire across act action actor actress actual"
).split()

# Standard Visa test card (passes Luhn)
VISA_CARD = "4111 1111 1111 1111"

# Valid-format ETH address
ETH_ADDRESS = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0"

# Normal English text — contains some BIP-39 words but never 12 consecutive
FALSE_POSITIVE_ESSAY = (
    "We are able to access the system above the standard threshold. "
    "The abstract concept of machine learning is gaining traction across many fields. "
    "Act now to achieve your goals before the deadline passes."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a system font at the given size, or PIL default as fallback."""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    print("  ⚠ No system font found — using PIL default (OCR may struggle)")
    return ImageFont.load_default()


def _make_image(
    text: str,
    width: int = 1200,
    height: int = 100,
    font_size: int = 32,
    bg: str = "white",
    fg: str = "black",
    x: int = 20,
    y: int = 30,
) -> Image.Image:
    """Create a single-line image with text."""
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    draw.text((x, y), text, font=font, fill=fg)
    return img


def _make_multiline_image(
    lines: list[str],
    width: int = 1400,
    line_height: int = 60,
    font_size: int = 28,
    bg: str = "white",
    fg: str = "black",
    x: int = 20,
    y_start: int = 20,
) -> Image.Image:
    """Create a multi-line image."""
    height = y_start + len(lines) * line_height + 20
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    for i, line in enumerate(lines):
        y = y_start + i * line_height
        draw.text((x, y), line, font=font, fill=fg)
    return img


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------


def make_seed_phrase_12word() -> Path:
    """
    12 BIP-39 words on a single line.
    Expected detection: seed_phrase, action=blur.
    """
    text = " ".join(SEED_WORDS_12)
    img = _make_multiline_image([text], width=1400, font_size=28)
    path = OUTPUT_DIR / "seed_phrase_12word.png"
    img.save(path)
    return path


def make_seed_phrase_24word() -> Path:
    """
    24 BIP-39 words across two lines (12 per line).
    Expected detection: seed_phrase, action=blur.
    """
    line1 = " ".join(SEED_WORDS_24[:12])
    line2 = " ".join(SEED_WORDS_24[12:])
    img = _make_multiline_image([line1, line2], width=1400, font_size=28)
    path = OUTPUT_DIR / "seed_phrase_24word.png"
    img.save(path)
    return path


def make_credit_card_visa() -> Path:
    """
    Standard Visa test card with expiry date below it.
    Expected detection: credit_card, confidence boosted by expiry.
    """
    img = _make_multiline_image(
        [VISA_CARD, "Expires: 12/28   CVV: 737"],
        width=700,
        line_height=55,
        font_size=36,
    )
    path = OUTPUT_DIR / "credit_card_visa.png"
    img.save(path)
    return path


def make_eth_address() -> Path:
    """
    ETH wallet address on a plain white background.
    Expected detection: crypto_address, network=ETH.
    """
    img = _make_image(ETH_ADDRESS, width=900, height=90, font_size=28)
    path = OUTPUT_DIR / "eth_address.png"
    img.save(path)
    return path


def make_false_positive_essay() -> Path:
    """
    Natural English sentences with scattered BIP-39 words.
    Words like 'able', 'access', 'abstract', 'act', 'across' appear naturally.
    These should NOT trigger detection (never 12 consecutive BIP-39 words).
    Expected: no detections.
    """
    # Split into lines that fit on screen
    words = FALSE_POSITIVE_ESSAY.split()
    # ~12 words per line
    lines = []
    for i in range(0, len(words), 12):
        lines.append(" ".join(words[i:i + 12]))

    img = _make_multiline_image(lines, width=1400, font_size=28)
    path = OUTPUT_DIR / "false_positive_essay.png"
    img.save(path)
    return path


def make_mixed_content() -> Path:
    """
    Multiple sensitive items on the same screen.
    Contains both a credit card number and an ETH address.
    Expected: multiple detections (credit_card + crypto_address).
    """
    lines = [
        f"Payment card: {VISA_CARD}",
        f"Wallet:       {ETH_ADDRESS}",
        "Transaction confirmed — please verify details above.",
    ]
    img = _make_multiline_image(lines, width=1400, line_height=65, font_size=28)
    path = OUTPUT_DIR / "mixed_content.png"
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = [
    ("seed_phrase_12word.png",   make_seed_phrase_12word,  "12 BIP-39 words — expect blur"),
    ("seed_phrase_24word.png",   make_seed_phrase_24word,  "24 BIP-39 words — expect blur"),
    ("credit_card_visa.png",     make_credit_card_visa,    "Visa card + expiry — expect detection"),
    ("eth_address.png",          make_eth_address,         "ETH address — expect blur"),
    ("false_positive_essay.png", make_false_positive_essay,"Normal text — expect NO detection"),
    ("mixed_content.png",        make_mixed_content,       "Card + ETH — expect 2+ detections"),
]


def generate_all(verify: bool = True) -> None:
    """Generate all test images and optionally verify them with Tesseract."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating test images → {OUTPUT_DIR}\n" + "=" * 60)

    for filename, fn, description in GENERATORS:
        path = fn()
        size = path.stat().st_size // 1024
        print(f"  ✓ {filename:<30} ({size} KB)  {description}")

    print(f"\n{len(GENERATORS)} images saved to {OUTPUT_DIR}")

    if verify:
        _verify_with_ocr()


def _verify_with_ocr() -> None:
    """
    Quick smoke-test: run Tesseract on each image and confirm expected
    key strings are readable. This validates the images are usable for tests.
    """
    print("\nVerifying images with TesseractEngine...")
    try:
        import numpy as np
        from core.ocr_engine import TesseractEngine
        engine = TesseractEngine()
    except Exception as e:
        print(f"  ⚠ OCR verification skipped: {e}")
        return

    checks = [
        ("seed_phrase_12word.png",   ["abandon", "ability", "accident"]),
        ("seed_phrase_24word.png",   ["abandon", "actual"]),
        ("credit_card_visa.png",     ["4111", "1111"]),
        ("eth_address.png",          ["0x742d"]),
        ("false_positive_essay.png", ["able", "access", "abstract"]),
        ("mixed_content.png",        ["4111", "0x742d"]),
    ]

    import numpy as np
    from PIL import Image as PILImage

    all_ok = True
    for filename, expected_tokens in checks:
        path = OUTPUT_DIR / filename
        img = np.array(PILImage.open(path))
        results = engine.detect_text(img)
        found_text = " ".join(r.text.lower() for r in results)

        missing = [t for t in expected_tokens if t.lower() not in found_text]
        if not missing:
            print(f"  ✓ {filename:<30} all tokens readable")
        else:
            print(f"  ⚠ {filename:<30} missing: {missing} (OCR may vary)")
            all_ok = False

    if all_ok:
        print("\nAll images verified — OCR can read expected content.")
    else:
        print("\nSome tokens unreadable. Images are still usable; OCR accuracy varies.")


if __name__ == "__main__":
    generate_all(verify=True)
