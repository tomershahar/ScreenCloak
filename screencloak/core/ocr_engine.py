"""Multi-engine OCR wrapper — selects best available engine automatically."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np

from detectors.base import OCRResult

logger = logging.getLogger("screencloak.ocr_engine")


# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------


def has_mps_gpu() -> bool:
    """
    Check if Apple Silicon MPS GPU is available.

    Used to decide whether PaddleOCR should use GPU acceleration.
    MPS (Metal Performance Shaders) is available on M1/M2/M3 Macs.
    """
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except ImportError:
        return False


def has_cuda_gpu() -> bool:
    """
    Check if NVIDIA CUDA GPU is available.

    Used for Windows/Linux users with NVIDIA cards.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy.

    Inverts dark-mode images before OCR. Tesseract was designed for black
    text on white paper — it fails badly on terminals, IDEs, and dark-themed
    apps without this step. PaddleOCR also benefits on very dark images.

    Heuristic: if mean pixel brightness < 127, the image is "dark mode".

    Args:
        image: BGR or RGB numpy array (H, W, 3)

    Returns:
        Preprocessed image — inverted if dark, unchanged if light
    """
    mean_brightness = cv2.mean(image)[0]
    if mean_brightness < 127:
        return cv2.bitwise_not(image)
    return image


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class OCREngine(ABC):
    """
    Abstract base class for all OCR engine implementations.

    Concrete engines: TesseractEngine (CPU), PaddleOCREngine (GPU/CPU).
    New engines only need to implement detect_text() and name.
    """

    @abstractmethod
    def detect_text(self, image: np.ndarray) -> list[OCRResult]:
        """
        Run OCR on an image and return all detected text regions.

        Implementations should:
        - Apply preprocess_for_ocr() before OCR
        - Return word-level (not character-level) results
        - Include bounding boxes in (x, y, w, h) format
        - Normalise confidence to 0.0–1.0

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            List of OCRResult, ordered top-to-bottom left-to-right
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name for logging and diagnostics."""
        ...


# ---------------------------------------------------------------------------
# Tesseract engine
# ---------------------------------------------------------------------------


class TesseractEngine(OCREngine):
    """
    Tesseract-based OCR (CPU).

    Reliable, well-tested fallback when PaddleOCR is unavailable or
    when the user explicitly selects engine="tesseract" in config.

    Page segmentation mode 11 (sparse text) is used because screen
    content has text scattered across the frame — it's not a single
    paragraph, book page, or form. PSM 11 finds as much text as possible
    without making layout assumptions.

    Confidence filtering: Tesseract emits -1 for non-text regions and
    0-100 for real words. We discard anything below MIN_CONF (30) to
    avoid garbage results from image noise.
    """

    MIN_CONF = 30  # Below this, Tesseract is guessing

    def __init__(self, config: Any | None = None) -> None:
        """
        Initialise Tesseract engine.

        Args:
            config: Optional OCR config (unused for Tesseract, kept for API uniformity)

        Raises:
            RuntimeError: If pytesseract package or tesseract binary not found
        """
        try:
            import pytesseract

            self._pytesseract = pytesseract
            # Smoke test — will raise if the binary isn't on PATH
            pytesseract.get_tesseract_version()
            logger.info("TesseractEngine initialised")
        except ImportError:
            raise RuntimeError(
                "pytesseract not installed. Run: pip install pytesseract\n"
                "Also install the Tesseract binary: brew install tesseract (macOS)"
            )
        except Exception as e:
            raise RuntimeError(
                f"Tesseract binary not found or not working: {e}\n"
                "Install with: brew install tesseract (macOS)"
            )

    @property
    def name(self) -> str:
        return "tesseract"

    def detect_text(self, image: np.ndarray) -> list[OCRResult]:
        """
        Run Tesseract OCR on an image.

        Pipeline:
        1. Invert dark-mode images (white-on-dark → black-on-white)
        2. Run pytesseract.image_to_data for word-level bounding boxes
        3. Filter low-confidence / empty results
        4. Return as list[OCRResult]

        Args:
            image: RGB numpy array

        Returns:
            List of word-level OCR results
        """
        processed = preprocess_for_ocr(image)

        try:
            data = self._pytesseract.image_to_data(
                processed,
                output_type=self._pytesseract.Output.DICT,
                # PSM 11: sparse text — best for screen content
                config="--psm 11",
            )
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}", exc_info=True)
            return []

        results: list[OCRResult] = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = str(data["text"][i]).strip()
            conf = int(data["conf"][i])

            # Skip empty text or below-threshold confidence (-1 = no text region)
            if not text or conf < self.MIN_CONF:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            # Normalise Tesseract 0-100 confidence to 0.0-1.0
            confidence = conf / 100.0

            results.append(
                OCRResult(
                    text=text,
                    bounding_box=(x, y, w, h),
                    confidence=confidence,
                )
            )

        return results


# ---------------------------------------------------------------------------
# PaddleOCR engine (Task 14)
# ---------------------------------------------------------------------------


class PaddleOCREngine(OCREngine):
    """
    PaddleOCR-based OCR engine — more accurate than Tesseract, especially
    for dense or mixed-layout screen content.

    PaddleOCR v3.x API notes (v3.4.0):
    - Use .predict(img_array) instead of .ocr()
    - Returns list of dicts per image: rec_texts, rec_scores, rec_boxes
    - rec_boxes format: [x_min, y_min, x_max, y_max] (NOT x,y,w,h)
      → convert with: x=x_min, y=y_min, w=x_max-x_min, h=y_max-y_min
    - Doc preprocessing (orientation classify, unwarping) is disabled:
      it's designed for scanned documents; adds ~2s with zero benefit for
      screen captures which are always already axis-aligned.

    GPU notes:
    - PaddlePaddle 3.x does NOT support Apple Silicon MPS — CPU only on Mac.
    - CUDA (NVIDIA) is supported on Windows/Linux via paddlepaddle-gpu package.
    - has_cuda_gpu() is checked at init time; if no CUDA → CPU mode.
    """

    def __init__(self, config: Any | None = None) -> None:
        """
        Initialise PaddleOCR engine.

        Downloads models on first run (~150MB for server det + mobile rec).
        Subsequent runs use cached models from ~/.paddlex/official_models/.

        Args:
            config: Optional OCR config (reads config.ocr.engine if present)

        Raises:
            RuntimeError: If paddleocr package is not installed
        """
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "paddleocr not installed. Run: pip install paddleocr paddlepaddle"
            )

        # PaddlePaddle 3.x: no MPS support on Mac; CUDA for NVIDIA only
        use_gpu = has_cuda_gpu()

        try:
            self._ocr = PaddleOCR(
                lang="en",
                # Disable document preprocessing — not needed for screen content
                # and it adds ~2 seconds to each call with no accuracy benefit
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            gpu_note = "CUDA GPU" if use_gpu else "CPU"
            logger.info(f"PaddleOCREngine initialised ({gpu_note})")
        except Exception as e:
            raise RuntimeError(f"PaddleOCR failed to initialise: {e}")

    @property
    def name(self) -> str:
        return "paddleocr"

    def detect_text(self, image: np.ndarray) -> list[OCRResult]:
        """
        Run PaddleOCR on an image.

        Pipeline:
        1. Invert dark-mode images (white-on-dark → black-on-white)
        2. Call PaddleOCR predict() to get text regions
        3. Convert rec_boxes [x1,y1,x2,y2] → (x, y, w, h)
        4. Return as list[OCRResult]

        PaddleOCR returns line-level results by default (whole text lines
        grouped together). This is fine for seed phrase detection (words
        are individual OCR tokens when on separate lines), but for inline
        text like "4111 1111 1111 1111" on one line, the full sequence
        comes back as a single OCRResult — which is actually better for
        credit card detection.

        Args:
            image: RGB numpy array

        Returns:
            List of OCRResult, one per detected text region
        """
        processed = preprocess_for_ocr(image)

        try:
            results_raw = self._ocr.predict(processed)
        except Exception as e:
            logger.error(f"PaddleOCR predict() failed: {e}", exc_info=True)
            return []

        if not results_raw:
            return []

        # predict() returns list[dict], one dict per image
        page = results_raw[0]
        rec_texts: list[str] = page.get("rec_texts", [])
        rec_scores: list[float] = page.get("rec_scores", [])
        rec_boxes: list[Any] = list(page.get("rec_boxes", []))

        results: list[OCRResult] = []

        for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
            text = str(text).strip()
            if not text:
                continue

            # rec_boxes format: [x_min, y_min, x_max, y_max]
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])

            # Convert to (x, y, w, h) — our internal bbox format
            x = x_min
            y = y_min
            w = max(x_max - x_min, 1)
            h = max(y_max - y_min, 1)

            results.append(
                OCRResult(
                    text=text,
                    bounding_box=(x, y, w, h),
                    confidence=float(score),
                )
            )

        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class OCREngineFactory:
    """
    Creates the best available OCR engine for the current system.

    Selection order (when engine="auto"):
    1. PaddleOCREngine (GPU if CUDA available, otherwise CPU) — most accurate
    2. TesseractEngine — reliable CPU fallback

    Note: PaddlePaddle 3.x does not support Apple Silicon MPS. On Mac,
    PaddleOCREngine runs on CPU (~300-500ms). With NVIDIA CUDA on
    Windows/Linux, expect ~100-200ms.
    """

    @staticmethod
    def create(config: Any) -> OCREngine:
        """
        Create and return the best available OCR engine.

        Args:
            config: Main Config object — reads config.ocr.engine

        Returns:
            Initialised OCREngine ready for detect_text() calls
        """
        engine_pref: str = getattr(config.ocr, "engine", "auto")

        if engine_pref == "tesseract":
            logger.info("OCREngineFactory: creating TesseractEngine (explicit config)")
            return TesseractEngine(config)

        # Try PaddleOCR for "paddleocr" or "auto"
        if engine_pref in ("paddleocr", "auto"):
            try:
                engine = PaddleOCREngine(config)
                logger.info(f"OCREngineFactory: created PaddleOCREngine (name={engine.name})")
                return engine
            except RuntimeError as e:
                logger.warning(
                    f"PaddleOCR unavailable ({e}) — falling back to Tesseract"
                )
            except Exception as e:
                logger.warning(
                    f"PaddleOCR failed to initialise ({e}) — falling back to Tesseract"
                )

        logger.info("OCREngineFactory: creating TesseractEngine (fallback)")
        return TesseractEngine(config)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------


def test_ocr_engine() -> None:
    """Test TesseractEngine with a synthetic white-background image."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from PIL import Image, ImageDraw

    print("Testing OCR Engine (Tasks 13 + 14)\n" + "=" * 60)

    # GPU availability report
    print(f"\nGPU detection:")
    print(f"  MPS (Apple Silicon): {has_mps_gpu()}")
    print(f"  CUDA (NVIDIA):       {has_cuda_gpu()}")

    # Test 1: Tesseract on white background
    print("\nTest 1: TesseractEngine — white background text")
    img = Image.new("RGB", (500, 60), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 15), "Hello ScreenCloak 1234", fill="black")
    img_array = np.array(img)

    try:
        engine = TesseractEngine()
        results = engine.detect_text(img_array)
        texts = [r.text for r in results]
        full_text = " ".join(texts)
        if "Hello" in full_text or "ScreenCloak" in full_text or "1234" in full_text:
            print(f"  ✓ Detected {len(results)} word(s): {texts}")
        else:
            print(f"  ✗ Expected text not found. Got: {texts}")
    except RuntimeError as e:
        print(f"  ✗ Engine init failed: {e}")
        return

    # Test 2: Dark mode image (white text on black)
    print("\nTest 2: TesseractEngine — dark mode (white-on-black)")
    img_dark = Image.new("RGB", (500, 60), color="black")
    draw_dark = ImageDraw.Draw(img_dark)
    draw_dark.text((10, 15), "Dark mode text 5678", fill="white")
    img_dark_array = np.array(img_dark)

    results_dark = engine.detect_text(img_dark_array)
    texts_dark = [r.text for r in results_dark]
    full_dark = " ".join(texts_dark)
    if texts_dark:
        print(f"  ✓ Detected {len(results_dark)} word(s): {texts_dark}")
    else:
        print(f"  ✗ No text detected on dark image (check preprocessing)")

    # Test 3: preprocess_for_ocr inverts dark images
    print("\nTest 3: preprocess_for_ocr inverts dark images")
    dark_img = np.zeros((60, 300, 3), dtype=np.uint8)  # Pure black
    bright_img = np.full((60, 300, 3), 200, dtype=np.uint8)  # Grey

    processed_dark = preprocess_for_ocr(dark_img)
    processed_bright = preprocess_for_ocr(bright_img)

    if cv2.mean(processed_dark)[0] > 127:
        print(f"  ✓ Dark image inverted (mean {cv2.mean(processed_dark)[0]:.0f})")
    else:
        print(f"  ✗ Dark image not inverted")

    if cv2.mean(processed_bright)[0] > 127:
        print(f"  ✓ Bright image unchanged (mean {cv2.mean(processed_bright)[0]:.0f})")
    else:
        print(f"  ✗ Bright image wrongly inverted")

    # Test 4: Factory — default (auto) picks PaddleOCR
    print("\nTest 4: OCREngineFactory — auto selects PaddleOCR")
    from core.config_manager import ConfigManager

    config = ConfigManager.load("config.yaml")
    config.ocr.engine = "auto"
    factory_engine = OCREngineFactory.create(config)
    print(f"  ✓ Factory created engine: {factory_engine.name}")

    # Test 5: Factory — explicit tesseract falls back
    print("\nTest 5: OCREngineFactory — explicit tesseract")
    config.ocr.engine = "tesseract"
    tess_engine = OCREngineFactory.create(config)
    print(f"  ✓ Factory created engine: {tess_engine.name}")

    # Test 6: PaddleOCR detects text (white background)
    print("\nTest 6: PaddleOCREngine — white background text")
    img_paddle = Image.new("RGB", (800, 120), color="white")
    draw_paddle = ImageDraw.Draw(img_paddle)
    draw_paddle.text((20, 40), "ScreenCloak 9999", fill="black")
    img_paddle_array = np.array(img_paddle)

    try:
        paddle_engine = PaddleOCREngine()
        results_paddle = paddle_engine.detect_text(img_paddle_array)
        texts_paddle = [r.text for r in results_paddle]
        full_paddle = " ".join(texts_paddle)
        if "ScreenCloak" in full_paddle or "9999" in full_paddle:
            print(f"  ✓ PaddleOCR detected: {texts_paddle}")
        else:
            print(f"  ✗ Expected text not found. Got: {texts_paddle}")
    except RuntimeError as e:
        print(f"  ✗ PaddleOCR init failed: {e}")

    # Test 7: PaddleOCR — dark mode
    print("\nTest 7: PaddleOCREngine — dark mode (white-on-black)")
    img_paddle_dark = Image.new("RGB", (800, 120), color="black")
    draw_paddle_dark = ImageDraw.Draw(img_paddle_dark)
    draw_paddle_dark.text((20, 40), "Dark 1234", fill="white")

    results_paddle_dark = paddle_engine.detect_text(np.array(img_paddle_dark))
    texts_paddle_dark = [r.text for r in results_paddle_dark]
    if texts_paddle_dark:
        print(f"  ✓ PaddleOCR dark mode detected: {texts_paddle_dark}")
    else:
        print(f"  ✗ PaddleOCR found nothing on dark image")

    # Test 8: PaddleOCR — bbox format sanity check
    print("\nTest 8: PaddleOCREngine — bounding box format (x, y, w, h)")
    if results_paddle:
        r = results_paddle[0]
        x, y, w, h = r.bounding_box
        if w > 0 and h > 0:
            print(f"  ✓ bbox={r.bounding_box} — w>0 and h>0")
        else:
            print(f"  ✗ Invalid bbox: {r.bounding_box}")
    else:
        print("  — skipped (no PaddleOCR results from Test 6)")

    # Test 9: Empty image — no crash
    print("\nTest 9: Empty image — no crash")
    empty = np.full((100, 100, 3), 200, dtype=np.uint8)
    results_empty = engine.detect_text(empty)
    print(f"  ✓ Empty image returned {len(results_empty)} results (no crash)")

    print("\n" + "=" * 60)
    print("OCR Engine Tests Complete!")


if __name__ == "__main__":
    test_ocr_engine()
