#!/usr/bin/env python3
"""
Benchmark OCR engines for ScreenCloak.

Tests PaddleOCR performance on various scenarios:
- Seed phrases (12 and 24 words)
- Credit card numbers
- Crypto addresses
- Mixed content

Measures:
- Latency (P50, P95, P99)
- Accuracy (text detection rate)
- GPU acceleration (MPS on Apple Silicon)

Run: python3 benchmark_ocr.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Check for GPU availability
try:
    import torch

    HAS_TORCH = True
    MPS_AVAILABLE = torch.backends.mps.is_available() if HAS_TORCH else False
except ImportError:
    HAS_TORCH = False
    MPS_AVAILABLE = False

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR

    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False


@dataclass
class BenchmarkResult:
    """Result from a single OCR benchmark run."""

    test_name: str
    ground_truth: str
    detected_text: str
    latency_ms: float
    accuracy: float  # 0.0 to 1.0
    success: bool


class OCRBenchmark:
    """Benchmark suite for OCR engines."""

    def __init__(self, output_dir: str = "data/benchmark_images"):
        """
        Initialize benchmark.

        Args:
            output_dir: Directory to save generated test images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test scenarios
        self.test_cases = [
            {
                "name": "seed_phrase_12word",
                "text": "abandon ability able about above absent absorb abstract absurd abuse access accident",
                "description": "12-word BIP-39 seed phrase",
            },
            {
                "name": "seed_phrase_24word",
                "text": "abandon ability able about above absent absorb abstract absurd abuse access accident "
                "acoustic acquire across act action actor actress actual adapt add addict address",
                "description": "24-word BIP-39 seed phrase",
            },
            {
                "name": "credit_card",
                "text": "4111 1111 1111 1111\nExp: 12/25\nCVV: 123",
                "description": "Credit card with expiry and CVV",
            },
            {
                "name": "eth_address",
                "text": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
                "description": "Ethereum address",
            },
            {
                "name": "btc_address",
                "text": "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",
                "description": "Bitcoin address (bech32)",
            },
            {
                "name": "mixed_content",
                "text": "My wallet: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0\n"
                "Seed: abandon ability able about above absent",
                "description": "Mixed sensitive content",
            },
        ]

        self.results: list[BenchmarkResult] = []

    def generate_test_image(
        self,
        text: str,
        filename: str,
        width: int = 1920,
        height: int = 200,
        font_size: int = 32,
        bg_color: str = "white",
        text_color: str = "black",
    ) -> Path:
        """
        Generate a test image with text.

        Args:
            text: Text to render
            filename: Output filename
            width: Image width
            height: Image height
            font_size: Font size
            bg_color: Background color
            text_color: Text color

        Returns:
            Path to generated image
        """
        # Create image
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Try to use a common system font
        try:
            # macOS system fonts
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSText.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "C:\\Windows\\Fonts\\Arial.ttf",  # Windows
            ]

            font = None
            for font_path in font_paths:
                if Path(font_path).exists():
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:
                        continue

            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Draw text
        draw.text((20, 20), text, fill=text_color, font=font)

        # Save image
        output_path = self.output_dir / filename
        img.save(output_path)

        return output_path

    def generate_all_test_images(self) -> None:
        """Generate all test images."""
        print("Generating test images...")

        for test_case in self.test_cases:
            # White background (standard)
            filename_white = f"{test_case['name']}_white.png"
            self.generate_test_image(
                text=test_case["text"],
                filename=filename_white,
                bg_color="white",
                text_color="black",
            )
            print(f"  ✓ {filename_white}")

            # Dark mode (black background, white text)
            filename_dark = f"{test_case['name']}_dark.png"
            self.generate_test_image(
                text=test_case["text"],
                filename=filename_dark,
                bg_color="black",
                text_color="white",
            )
            print(f"  ✓ {filename_dark}")

    def benchmark_paddleocr(self) -> None:
        """Benchmark PaddleOCR on all test images."""
        if not HAS_PADDLEOCR:
            print("\n❌ PaddleOCR not installed. Install with:")
            print("   pip install paddleocr paddlepaddle")
            return

        print("\n" + "=" * 60)
        print("PaddleOCR Benchmark")
        print("=" * 60)

        # Initialize PaddleOCR
        print("\nInitializing PaddleOCR...")

        # Check GPU availability
        use_gpu = MPS_AVAILABLE or torch.cuda.is_available() if HAS_TORCH else False
        print(f"  PyTorch available: {HAS_TORCH}")
        print(f"  MPS (Apple Silicon) available: {MPS_AVAILABLE}")
        print(f"  CUDA available: {torch.cuda.is_available() if HAS_TORCH else False}")
        print(f"  Using GPU: {use_gpu}")

        try:
            # Initialize with optimal settings for speed
            ocr = PaddleOCR(
                use_angle_cls=True,  # Detect text orientation
                lang="en",
                use_gpu=use_gpu,
                show_log=False,
            )
            print("  ✓ PaddleOCR initialized")
        except Exception as e:
            print(f"  ❌ Failed to initialize PaddleOCR: {e}")
            return

        # Warm-up run (first run is slower due to model loading)
        print("\nWarming up (first run loads models)...")
        warmup_img = self.output_dir / f"{self.test_cases[0]['name']}_white.png"
        if warmup_img.exists():
            _ = ocr.ocr(str(warmup_img), cls=True)
            print("  ✓ Warm-up complete")

        # Benchmark each test case
        print("\nRunning benchmarks...")
        latencies: list[float] = []

        for test_case in self.test_cases:
            # Test white background
            image_path = self.output_dir / f"{test_case['name']}_white.png"

            if not image_path.exists():
                print(f"  ⚠️  Skipping {test_case['name']} (image not found)")
                continue

            # Run OCR multiple times for stable latency measurement
            runs = 5
            run_latencies: list[float] = []
            detected_text = ""

            for i in range(runs):
                start = time.time()
                result = ocr.ocr(str(image_path), cls=True)
                latency_ms = (time.time() - start) * 1000
                run_latencies.append(latency_ms)

                # Extract detected text from first run
                if i == 0 and result and result[0]:
                    detected_text = " ".join([line[1][0] for line in result[0]])

            # Calculate statistics
            avg_latency = sum(run_latencies) / len(run_latencies)
            latencies.extend(run_latencies)

            # Calculate accuracy (simple similarity check)
            ground_truth = test_case["text"].replace("\n", " ")
            accuracy = self._calculate_accuracy(ground_truth, detected_text)

            # Store result
            benchmark_result = BenchmarkResult(
                test_name=test_case["name"],
                ground_truth=ground_truth,
                detected_text=detected_text,
                latency_ms=avg_latency,
                accuracy=accuracy,
                success=accuracy > 0.8,  # 80% threshold
            )
            self.results.append(benchmark_result)

            # Print result
            status = "✓" if benchmark_result.success else "✗"
            print(f"\n  {status} {test_case['description']}")
            print(f"      Latency: {avg_latency:.0f}ms (avg of {runs} runs)")
            print(f"      Accuracy: {accuracy:.1%}")
            if accuracy < 1.0:
                print(f"      Expected: {ground_truth[:60]}...")
                print(f"      Got:      {detected_text[:60]}...")

        # Print summary statistics
        if latencies:
            self._print_summary(latencies)

    def _calculate_accuracy(self, ground_truth: str, detected: str) -> float:
        """
        Calculate accuracy between ground truth and detected text.

        Uses simple word-level similarity.

        Args:
            ground_truth: Expected text
            detected: Detected text

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        # Normalize text
        gt_words = set(ground_truth.lower().split())
        det_words = set(detected.lower().split())

        if not gt_words:
            return 0.0

        # Calculate overlap
        overlap = len(gt_words & det_words)
        accuracy = overlap / len(gt_words)

        return accuracy

    def _print_summary(self, latencies: list[float]) -> None:
        """
        Print benchmark summary statistics.

        Args:
            latencies: List of latency measurements in milliseconds
        """
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        p50 = latencies_sorted[n // 2]
        p95 = latencies_sorted[int(n * 0.95)]
        p99 = latencies_sorted[int(n * 0.99)]
        avg = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)

        print("\n" + "=" * 60)
        print("LATENCY SUMMARY")
        print("=" * 60)
        print(f"  Runs:        {n}")
        print(f"  Average:     {avg:.0f}ms")
        print(f"  Min:         {min_lat:.0f}ms")
        print(f"  Max:         {max_lat:.0f}ms")
        print(f"  P50 (median): {p50:.0f}ms")
        print(f"  P95:         {p95:.0f}ms")
        print(f"  P99:         {p99:.0f}ms")

        # Verdict
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)

        target_p95 = 500  # Target from requirements

        if p95 < target_p95:
            print(f"  ✅ PASS - P95 latency ({p95:.0f}ms) < {target_p95}ms target")
            print("     ScreenCloak will work with Stream Delay buffer")
        else:
            print(f"  ⚠️  MARGINAL - P95 latency ({p95:.0f}ms) >= {target_p95}ms target")
            print("     May need to increase Stream Delay or optimize settings")

        # Accuracy summary
        successful = sum(1 for r in self.results if r.success)
        total = len(self.results)

        if successful == total:
            print(f"  ✅ ACCURACY - {successful}/{total} tests passed (>80% accuracy)")
        else:
            print(
                f"  ⚠️  ACCURACY - {successful}/{total} tests passed (some below 80% accuracy)"
            )

    def generate_report(self, filename: str = "benchmark_report.md") -> None:
        """
        Generate a markdown report of benchmark results.

        Args:
            filename: Output filename
        """
        report_path = self.output_dir / filename

        with open(report_path, "w") as f:
            f.write("# ScreenCloak OCR Benchmark Report\n\n")
            f.write(f"**Hardware:** {self._get_hardware_info()}\n")
            f.write(f"**GPU:** {'Yes (MPS)' if MPS_AVAILABLE else 'No (CPU only)'}\n\n")

            f.write("## Results\n\n")
            f.write("| Test | Latency (ms) | Accuracy | Status |\n")
            f.write("|------|--------------|----------|--------|\n")

            for result in self.results:
                status = "✅ Pass" if result.success else "❌ Fail"
                f.write(
                    f"| {result.test_name} | {result.latency_ms:.0f} | "
                    f"{result.accuracy:.1%} | {status} |\n"
                )

            f.write("\n## Recommendations\n\n")

            # Calculate average P95
            latencies = [r.latency_ms for r in self.results]
            if latencies:
                p95 = sorted(latencies)[int(len(latencies) * 0.95)]

                if p95 < 300:
                    f.write("- ✅ **Excellent performance** - proceed with implementation\n")
                elif p95 < 500:
                    f.write("- ✅ **Good performance** - meets requirements with Stream Delay\n")
                else:
                    f.write("- ⚠️ **Marginal performance** - consider optimizations:\n")
                    f.write("  - Use Angle/Lite PaddleOCR models\n")
                    f.write("  - Reduce frame sample rate\n")
                    f.write("  - Increase Stream Delay to 5+ seconds\n")

        print(f"\n📄 Report saved to: {report_path}")

    def _get_hardware_info(self) -> str:
        """Get hardware information."""
        import platform

        system = platform.system()
        machine = platform.machine()

        if MPS_AVAILABLE:
            return f"{system} {machine} (Apple Silicon with MPS)"
        else:
            return f"{system} {machine}"


def main() -> None:
    """Run OCR benchmark."""
    print("ScreenCloak OCR Benchmark")
    print("=" * 60)

    # Check dependencies
    if not HAS_PADDLEOCR:
        print("\n❌ PaddleOCR not installed!")
        print("\nTo install:")
        print("  pip install paddleocr paddlepaddle")
        return

    # Create benchmark
    benchmark = OCRBenchmark()

    # Generate test images
    benchmark.generate_all_test_images()

    # Run PaddleOCR benchmark
    benchmark.benchmark_paddleocr()

    # Generate report
    benchmark.generate_report()

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
