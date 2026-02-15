# Contributing to ScreenCloak

Thanks for helping make ScreenCloak better. Here's how to contribute effectively.

---

## Reporting False Positives

A false positive is when ScreenCloak triggers on content that isn't sensitive.

**Please include in your report:**

```
- What was on screen (describe or screenshot the non-sensitive content)
- Detection category that fired (seed_phrase / credit_card / crypto_address / api_key / personal_strings)
- Your OS and Tesseract version
- Config settings (fuzzy_threshold, confidence levels)
```

Open an issue using the **False Positive Report** template.

---

## Reporting Missed Detections

A missed detection is when ScreenCloak fails to catch something it should.

**Please include:**

```
- What type of data was on screen
- Font, background colour, and app (terminal, browser, IDE, etc.)
- Whether the OCR engine read the text at all (check logs/)
- Your OS
```

Open an issue using the **Bug Report** template.

---

## Adding New Detection Patterns

### API key patterns (`data/api_patterns.json`)

Each service entry looks like:

```json
"ServiceName": [
  {
    "pattern": "regex_pattern_here",
    "confidence": 0.92,
    "description": "ServiceName API key"
  }
]
```

Guidelines:
- Patterns must have a fixed prefix or fixed length (or both) to keep false positives low
- Set `confidence` to `0.92` for patterns with a fixed prefix + fixed length; lower for looser patterns
- Add a test case in `tests/test_detector.py` using a fake key that matches the pattern
- Verify the pattern does NOT match normal English text: run `python3 tests/benchmark.py`

### New detector types (`detectors/`)

1. Extend `BaseDetector` from `detectors/base.py`
2. Implement `detect(self, ocr_results: list[OCRResult]) -> list[DetectionResult]`
3. Add unit tests in `tests/test_detector.py`
4. Register in `config.yaml` under `detection:`
5. Wire into `core/detector.py` `DetectorPipeline`

---

## Submitting Performance Improvements

ScreenCloak's detection latency directly affects the exposure window. Any improvement here matters.

Before submitting:

```bash
# Run the benchmark to capture your baseline
python3 tests/benchmark.py

# Make your changes, then run again
python3 tests/benchmark.py
```

Include both benchmark outputs in your PR description.

---

## Code Style

- Python 3.11+
- Type annotations on all functions
- `black` for formatting (line length 100): `black --line-length 100 .`
- No new dependencies without discussion — keep the install footprint small

---

## Running Tests

```bash
cd screencloak
python3 -m pytest tests/ --ignore=tests/benchmark.py -q
```

All 87 tests must pass before submitting a PR.

---

## Join the Discussion

Questions, ideas, and feedback welcome on our Discord server. Link in the README.
