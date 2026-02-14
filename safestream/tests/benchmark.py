"""
SafeStream Detection Benchmark Suite — Task 22.

Measures detection accuracy (true/false positive rates) and latency
(OCR + end-to-end) across all synthetic test images.

Run from the safestream/ directory:
    python tests/benchmark.py

# TODO: Add pytest integration so threshold checks can run as CI assertions.
#       Each metric (TP rate >= 0.95, FP rate <= 0.05, P95 <= 500ms) becomes
#       a test that fails the suite if regressions occur.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Path setup — make safestream packages importable when run directly
# ---------------------------------------------------------------------------

_SAFESTREAM_DIR = Path(__file__).parent.parent
if str(_SAFESTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(_SAFESTREAM_DIR))

_IMAGE_DIR = _SAFESTREAM_DIR / "data" / "test_images"

# ---------------------------------------------------------------------------
# Ground truth table
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkCase:
    """A single test image with expected detection behaviour."""
    filename: str
    description: str
    expected_types: list[str]   # detection types that MUST appear
    is_negative: bool = False   # True = clean image, no detections expected


BENCHMARK_CASES: list[BenchmarkCase] = [
    BenchmarkCase(
        filename="seed_phrase_12word.png",
        description="12-word BIP-39 seed phrase",
        expected_types=["seed_phrase"],
    ),
    BenchmarkCase(
        filename="seed_phrase_24word.png",
        description="24-word BIP-39 seed phrase",
        expected_types=["seed_phrase"],
    ),
    BenchmarkCase(
        filename="credit_card_visa.png",
        description="Visa test card + expiry",
        expected_types=["credit_card"],
    ),
    BenchmarkCase(
        filename="eth_address.png",
        description="ETH wallet address",
        expected_types=["crypto_address"],
    ),
    BenchmarkCase(
        filename="mixed_content.png",
        description="Credit card + ETH address (mixed)",
        expected_types=["credit_card", "crypto_address"],
    ),
    BenchmarkCase(
        filename="false_positive_essay.png",
        description="Normal text with scattered BIP-39 words",
        expected_types=[],
        is_negative=True,
    ),
]

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    case: BenchmarkCase
    detected_types: list[str]
    ocr_latency_ms: float
    e2e_latency_ms: float
    passed: bool
    missed_types: list[str] = field(default_factory=list)
    unexpected_types: list[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    case_results: list[CaseResult]
    ocr_latencies_ms: list[float]
    e2e_latencies_ms: list[float]

    @property
    def tp_cases(self) -> list[CaseResult]:
        return [r for r in self.case_results if not r.case.is_negative]

    @property
    def fp_cases(self) -> list[CaseResult]:
        return [r for r in self.case_results if r.case.is_negative]

    @property
    def tp_rate(self) -> float:
        if not self.tp_cases:
            return 0.0
        passed = sum(1 for r in self.tp_cases if r.passed)
        return passed / len(self.tp_cases)

    @property
    def fp_rate(self) -> float:
        if not self.fp_cases:
            return 0.0
        triggered = sum(1 for r in self.fp_cases if not r.passed)
        return triggered / len(self.fp_cases)

    def _percentile(self, values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = max(0, int(len(sorted_vals) * p) - 1)
        return sorted_vals[idx]

    def ocr_p50(self) -> float:
        return self._percentile(self.ocr_latencies_ms, 0.50)

    def ocr_p95(self) -> float:
        return self._percentile(self.ocr_latencies_ms, 0.95)

    def e2e_p50(self) -> float:
        return self._percentile(self.e2e_latencies_ms, 0.50)

    def e2e_p95(self) -> float:
        return self._percentile(self.e2e_latencies_ms, 0.95)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

RUNS_PER_IMAGE = 3  # Repeat each image for stable latency measurement


def run_benchmark() -> Optional[BenchmarkReport]:
    """Run the full benchmark suite. Returns None if prerequisites missing."""

    # --- Prerequisites check ---
    if not _IMAGE_DIR.is_dir():
        print(f"ERROR: Test images not found at {_IMAGE_DIR}")
        print("       Run: python tests/generate_test_images.py")
        return None

    try:
        from core.ocr_engine import TesseractEngine
        from core.config_manager import ConfigManager
        from core.detector import DetectorPipeline
    except ImportError as e:
        print(f"ERROR: Could not import SafeStream modules: {e}")
        return None

    try:
        engine = TesseractEngine()
    except Exception as e:
        print(f"ERROR: Could not initialise TesseractEngine: {e}")
        print("       Is Tesseract installed? brew install tesseract")
        return None

    config = ConfigManager.load(str(_SAFESTREAM_DIR / "config.yaml"))
    pipeline = DetectorPipeline(config)

    case_results: list[CaseResult] = []
    all_ocr_latencies: list[float] = []
    all_e2e_latencies: list[float] = []

    print("\nRunning benchmark cases...")
    print("-" * 72)

    for case in BENCHMARK_CASES:
        img_path = _IMAGE_DIR / case.filename
        if not img_path.exists():
            print(f"  SKIP  {case.filename:<30}  (image missing)")
            continue

        image = np.array(PILImage.open(img_path))

        ocr_run_times: list[float] = []
        e2e_run_times: list[float] = []
        detected_types: list[str] = []

        for run in range(RUNS_PER_IMAGE):
            # OCR step
            t0 = time.perf_counter()
            ocr_results = engine.detect_text(image)
            t1 = time.perf_counter()

            # Detector pipeline step
            scan_result = pipeline.scan(ocr_results)
            t2 = time.perf_counter()

            ocr_ms = (t1 - t0) * 1000
            e2e_ms = (t2 - t0) * 1000

            ocr_run_times.append(ocr_ms)
            e2e_run_times.append(e2e_ms)

            # Only collect detected types from the last run (stable result)
            if run == RUNS_PER_IMAGE - 1:
                detected_types = [d.type for d in scan_result.detections]

        avg_ocr = sum(ocr_run_times) / len(ocr_run_times)
        avg_e2e = sum(e2e_run_times) / len(e2e_run_times)

        all_ocr_latencies.extend(ocr_run_times)
        all_e2e_latencies.extend(e2e_run_times)

        # Evaluate pass/fail
        if case.is_negative:
            # Negative case: pass if nothing actionable was detected
            passed = len(detected_types) == 0
            missed: list[str] = []
            unexpected = detected_types
        else:
            # Positive case: pass if all expected types were detected
            missed = [t for t in case.expected_types if t not in detected_types]
            unexpected = []
            passed = len(missed) == 0

        case_results.append(CaseResult(
            case=case,
            detected_types=detected_types,
            ocr_latency_ms=avg_ocr,
            e2e_latency_ms=avg_e2e,
            passed=passed,
            missed_types=missed,
            unexpected_types=unexpected,
        ))

        status = "PASS" if passed else "FAIL"
        detected_str = ", ".join(detected_types) if detected_types else "(none)"
        print(
            f"  {status:<4}  {case.filename:<30}  "
            f"detected={detected_str:<25}  e2e={avg_e2e:.0f}ms"
        )
        if missed:
            print(f"          missed: {missed}")
        if unexpected and case.is_negative:
            print(f"          unexpected: {unexpected}")

    return BenchmarkReport(
        case_results=case_results,
        ocr_latencies_ms=all_ocr_latencies,
        e2e_latencies_ms=all_e2e_latencies,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(report: BenchmarkReport) -> None:
    tp_total = len(report.tp_cases)
    tp_passed = sum(1 for r in report.tp_cases if r.passed)
    fp_total = len(report.fp_cases)
    fp_triggered = sum(1 for r in report.fp_cases if not r.passed)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  True Positive Rate:   {tp_passed}/{tp_total}  ({report.tp_rate:.1%})")
    print(f"  False Positive Rate:  {fp_triggered}/{fp_total}  ({report.fp_rate:.1%})")
    print()
    print(f"  OCR latency      P50: {report.ocr_p50():.0f}ms   P95: {report.ocr_p95():.0f}ms")
    print(f"  E2E latency      P50: {report.e2e_p50():.0f}ms   P95: {report.e2e_p95():.0f}ms")

    # Verdict
    meets_tp   = report.tp_rate  >= 0.95
    meets_fp   = report.fp_rate  <= 0.05
    meets_lat  = report.e2e_p95() <= 500.0

    print("\n" + "=" * 72)
    print("VERDICT (M1 acceptance criteria)")
    print("=" * 72)
    _verdict_line("True positive rate ≥ 95%",    meets_tp,  f"{report.tp_rate:.1%}")
    _verdict_line("False positive rate ≤ 5%",    meets_fp,  f"{report.fp_rate:.1%}")
    _verdict_line("E2E P95 latency ≤ 500ms",     meets_lat, f"{report.e2e_p95():.0f}ms")

    overall = meets_tp and meets_fp and meets_lat
    print()
    if overall:
        print("  ✅  OVERALL PASS — SafeStream meets M1 acceptance criteria")
    else:
        print("  ❌  OVERALL FAIL — one or more criteria not met")
    print()


def _verdict_line(label: str, passed: bool, value: str) -> None:
    icon = "✅" if passed else "❌"
    print(f"  {icon}  {label:<35}  {value}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("SafeStream Detection Benchmark")
    print("=" * 72)
    print(f"Test images: {_IMAGE_DIR}")
    print(f"Runs per image: {RUNS_PER_IMAGE}")

    report = run_benchmark()
    if report is None:
        sys.exit(1)

    print_report(report)


if __name__ == "__main__":
    main()
