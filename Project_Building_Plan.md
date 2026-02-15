# ScreenCloak - Project Building Plan

## Overview

This plan breaks M1 (Core Detection Engine) into small, logical tasks that can be built one at a time within daily token limits. Each task is designed to be:
- **Completable in one session** (~5-15 minutes)
- **Independently testable** (verify it works before moving on)
- **Logically ordered** (builds on previous tasks)

**Estimated Total:** 20-25 tasks across 5-7 sessions

---

## How to Use This Plan

1. **Start at Task 1**, work sequentially
2. **Mark completed tasks** with `[x]` in the checkbox
3. **Test each task** before moving to the next
4. **Stop at any task** - each one is a natural checkpoint
5. **Resume next session** from the last incomplete task

---

## Phase 1: Foundation (Tasks 1-4)
**Goal:** Set up project structure and core utilities
**Time:** ~1 session

### ✅ Task 1: Project Structure Setup
**Complexity:** Low | **Time:** 5 min

- [ ] Create directory structure:
  ```
  screencloak/
  ├── core/
  ├── detectors/
  ├── data/
  ├── tests/
  ├── ui/
  └── logs/
  ```
- [ ] Create `__init__.py` files in all package directories
- [ ] Create placeholder files (empty for now):
  - `main.py`
  - `requirements.txt`
  - `pyproject.toml`
  - `README.md`

**Verification:**
```bash
tree screencloak/  # Should show full directory structure
```

---

### ✅ Task 2: Dependencies & Environment
**Complexity:** Low | **Time:** 5 min

- [ ] Create `requirements.txt` with core dependencies:
  ```
  # OCR Engines
  paddleocr>=2.7.0
  paddlepaddle>=2.5.0
  pytesseract>=0.3.10

  # Image Processing
  opencv-python>=4.8.0
  mss>=9.0.0
  Pillow>=10.0.0

  # OBS Integration
  obs-websocket-py>=1.0

  # Utilities
  pyyaml>=6.0
  rapidfuzz>=3.0.0

  # Development
  pytest>=7.4.0
  pytest-cov>=4.1.0
  black>=23.0.0
  mypy>=1.5.0
  ```

- [ ] Create `pyproject.toml` for tooling config:
  ```toml
  [project]
  name = "screencloak"
  version = "0.1.0"
  description = "Real-time sensitive data detection for OBS streams"

  [tool.black]
  line-length = 100
  target-version = ['py311']

  [tool.mypy]
  python_version = "3.11"
  strict = true
  warn_return_any = true
  warn_unused_configs = true

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  python_files = ["test_*.py"]
  python_functions = ["test_*"]
  addopts = "--cov=screencloak --cov-report=term-missing"
  ```

**Verification:**
```bash
cat requirements.txt  # Should show all dependencies
cat pyproject.toml    # Should show tooling config
```

---

### ✅ Task 3: Config Manager
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `core/config_manager.py`
- [ ] Implement `Config` dataclass with all settings (use pydantic or dataclasses)
- [ ] Implement `ConfigManager.load()` to read from `config.yaml`
- [ ] Add default config fallback if file doesn't exist
- [ ] Add config validation

**Files to create:**
- `core/config_manager.py`
- `config.yaml` (default config from implementation plan)

**Verification:**
```bash
# Create test script
python -c "from screencloak.core.config_manager import ConfigManager; config = ConfigManager.load(); print(config)"
```

---

### ✅ Task 4: Logging Setup
**Complexity:** Low | **Time:** 5 min

- [ ] Create `core/logger.py`
- [ ] Set up Python logging with file + console handlers
- [ ] Implement sanitized logging function: `log_detection(detection, sanitized=True)`
- [ ] Create `logs/` directory if it doesn't exist
- [ ] Add log rotation (optional, use `RotatingFileHandler`)

**Files to create:**
- `core/logger.py`

**Verification:**
```python
from screencloak.core.logger import log_detection
log_detection({"type": "test", "text": "sensitive data"}, sanitized=True)
# Check logs/detections.log contains [REDACTED]
```

---

## Phase 2: Detection Foundation (Tasks 5-7)
**Goal:** Build base detection infrastructure
**Time:** ~1 session

### ✅ Task 5: Base Detector Interface
**Complexity:** Low | **Time:** 5 min

- [ ] Create `detectors/base.py`
- [ ] Define `DetectionResult` dataclass:
  ```python
  @dataclass
  class DetectionResult:
      type: str
      confidence: float
      text_preview: str
      bounding_box: tuple[int, int, int, int]
      action: str  # "blur", "warn", "ignore"
  ```
- [ ] Define `OCRResult` dataclass:
  ```python
  @dataclass
  class OCRResult:
      text: str
      bounding_box: tuple[int, int, int, int]
      confidence: float
  ```
- [ ] Define abstract `BaseDetector` class with `detect()` method

**Files to create:**
- `detectors/base.py`

**Verification:**
```python
from screencloak.detectors.base import DetectionResult, OCRResult, BaseDetector
# Should import without errors
```

---

### ✅ Task 6: Download BIP-39 Wordlist
**Complexity:** Low | **Time:** 3 min

- [ ] Download BIP-39 wordlist from GitHub
- [ ] Save to `data/bip39_wordlist.txt`
- [ ] Verify it contains 2048 words

**Commands:**
```bash
curl -o data/bip39_wordlist.txt https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt
wc -l data/bip39_wordlist.txt  # Should show 2048
```

**Verification:**
```bash
head -5 data/bip39_wordlist.txt  # Should show: abandon, ability, able, about, above
tail -5 data/bip39_wordlist.txt  # Should show: zone, zoo
```

---

### ✅ Task 7: Seed Phrase Detector
**Complexity:** High | **Time:** 15 min

- [ ] Create `detectors/seed_phrase.py`
- [ ] Load BIP-39 wordlist into a set
- [ ] Implement `detect(ocr_results: list[OCRResult]) -> list[DetectionResult]`
- [ ] Algorithm:
  1. Tokenize OCR text into words
  2. Find consecutive runs of 12 or 24 BIP-39 words
  3. Check spatial proximity (words within 50px vertically)
  4. Score confidence based on exact count match
- [ ] Add helper: `_are_spatially_clustered(words, max_distance=50)`
- [ ] Add helper: `_merge_boxes(word_boxes) -> bounding_box`

**Files to create:**
- `detectors/seed_phrase.py`

**Verification:**
```python
# Create test in tests/test_seed_phrase.py
from screencloak.detectors.seed_phrase import SeedPhraseDetector
from screencloak.detectors.base import OCRResult

# Mock 12-word seed phrase
words = "abandon ability able about above absent absorb abstract absurd abuse access accident"
ocr_results = [
    OCRResult(text=word, bounding_box=(i*50, 10, i*50+40, 30), confidence=0.9)
    for i, word in enumerate(words.split())
]

detector = SeedPhraseDetector(config)
detections = detector.detect(ocr_results)

assert len(detections) == 1
assert detections[0].type == "seed_phrase"
assert detections[0].confidence > 0.9
```

**Important:** This is the most complex task - take your time!

---

## Phase 3: More Detectors (Tasks 8-11)
**Goal:** Implement remaining detection modules
**Time:** ~1-2 sessions

### ✅ Task 8: Credit Card Detector
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `detectors/credit_card.py`
- [ ] **Implement fuzzy digit cleanup:** `_fuzzy_digit_cleanup(text: str) -> str`
  - Replace OCR noise: `l`→`1`, `I`→`1`, `O`→`0`
  - Handles common OCR misreads (e.g., "4532 l488..." → "4532 1488...")
- [ ] Implement Luhn algorithm validation: `_luhn_check(card_number: str) -> bool`
- [ ] Regex for 16-digit sequences: `\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b`
- [ ] Check for nearby expiration date (MM/YY) for confidence boost
- [ ] Implement `detect(ocr_results) -> list[DetectionResult]`

**Files to create:**
- `detectors/credit_card.py`

**Verification:**
```python
from screencloak.detectors.credit_card import CreditCardDetector, luhn_check

# Valid test card (passes Luhn)
assert luhn_check("4532148803436467") == True

# Invalid card (fails Luhn)
assert luhn_check("4532148803436468") == False

# Test detection
ocr_results = [OCRResult(text="4532 1488 0343 6467", bounding_box=(10, 10, 200, 30), confidence=0.9)]
detector = CreditCardDetector(config)
detections = detector.detect(ocr_results)

assert len(detections) == 1
assert detections[0].type == "credit_card"
```

---

### ✅ Task 9: Crypto Address Detector
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `detectors/crypto_address.py`
- [ ] Define regex patterns for BTC, ETH, SOL:
  ```python
  PATTERNS = {
      "BTC": r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|bc1[a-z0-9]{39,59}\b',
      "ETH": r'\b0x[a-fA-F0-9]{40}\b',
      "SOL": r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b',
  }
  ```
- [ ] Implement basic validation (length checks)
- [ ] Implement `detect(ocr_results) -> list[DetectionResult]`

**Files to create:**
- `detectors/crypto_address.py`

**Verification:**
```python
from screencloak.detectors.crypto_address import CryptoAddressDetector

# Test ETH address
ocr_results = [OCRResult(
    text="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
    bounding_box=(10, 10, 400, 30),
    confidence=0.9
)]
detector = CryptoAddressDetector(config)
detections = detector.detect(ocr_results)

assert len(detections) == 1
assert detections[0].type == "crypto_address"
```

---

### ✅ Task 10: Personal Strings Detector
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `detectors/personal_strings.py`
- [ ] Use `rapidfuzz` for fuzzy matching
- [ ] Read user-defined strings from config
- [ ] Implement `detect(ocr_results) -> list[DetectionResult]`
- [ ] Use configurable threshold (default 85%)

**Files to create:**
- `detectors/personal_strings.py`

**Verification:**
```python
from screencloak.detectors.personal_strings import PersonalStringsDetector

# Add to config: personal_strings = ["John Doe", "john@email.com"]
ocr_results = [OCRResult(text="My name is John Doe", bounding_box=(10, 10, 200, 30), confidence=0.9)]

detector = PersonalStringsDetector(config)
detections = detector.detect(ocr_results)

assert len(detections) >= 1  # Should detect "John Doe"
```

---

### ✅ Task 11: API Keys Detector (Stub)
**Complexity:** Low | **Time:** 5 min

- [ ] Create `detectors/api_keys.py`
- [ ] Create stub implementation (returns empty list)
- [ ] Add TODO comment: "Paid tier feature - implement in M6"
- [ ] Define common API key patterns in `data/api_patterns.json`:
  ```json
  {
    "AWS": "AKIA[0-9A-Z]{16}",
    "GitHub": "ghp_[0-9a-zA-Z]{36}",
    "Stripe": "sk_live_[0-9a-zA-Z]{24}"
  }
  ```

**Files to create:**
- `detectors/api_keys.py`
- `data/api_patterns.json`

**Verification:**
```python
from screencloak.detectors.api_keys import APIKeysDetector
detector = APIKeysDetector(config)
assert detector.detect([]) == []  # Stub returns empty
```

---

## Phase 4: Detection Pipeline (Task 12)
**Goal:** Coordinate all detectors
**Time:** ~1 session

### ✅ Task 12: Detector Pipeline
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `core/detector.py`
- [ ] Implement `DetectorPipeline` class
- [ ] Load enabled detectors from config
- [ ] Implement `scan(ocr_results) -> list[DetectionResult]`:
  1. Run all enabled detectors
  2. Aggregate results
  3. Apply confidence thresholds (>0.9 = blur, 0.6-0.9 = warn)
  4. Return filtered detections
- [ ] Add spatial clustering helper: `_cluster_spatially(detections)`
- [ ] Add ignore zones filter: `_filter_ignore_zones(detections)`

**Files to create:**
- `core/detector.py`

**Verification:**
```python
from screencloak.core.detector import DetectorPipeline
from screencloak.detectors.base import OCRResult

# Mock OCR results with seed phrase
ocr_results = [...]  # 12 BIP-39 words

pipeline = DetectorPipeline(config)
detections = pipeline.scan(ocr_results)

assert len(detections) >= 1
assert all(d.action in ["blur", "warn", "ignore"] for d in detections)
```

---

## Phase 5: OCR Integration (Tasks 13-14)
**Goal:** Set up OCR engines
**Time:** ~1 session

### ✅ Task 13: OCR Engine Interface
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `core/ocr_engine.py`
- [ ] Define abstract `OCREngine` class with `detect_text(image) -> list[OCRResult]`
- [ ] Implement `TesseractEngine` class (simplest, start here)
- [ ] Add GPU detection helper: `has_mps_gpu() -> bool`
- [ ] Add `OCREngineFactory.create(config) -> OCREngine`

**Files to create:**
- `core/ocr_engine.py`

**Verification:**
```python
from screencloak.core.ocr_engine import OCREngineFactory
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create test image with text
img = Image.new('RGB', (400, 100), color='white')
draw = ImageDraw.Draw(img)
draw.text((10, 10), "Hello ScreenCloak", fill='black')
img_array = np.array(img)

engine = OCREngineFactory.create(config)
results = engine.detect_text(img_array)

assert len(results) > 0
assert any("Hello" in r.text or "ScreenCloak" in r.text for r in results)
```

---

### ✅ Task 14: PaddleOCR Engine (MPS)
**Complexity:** High | **Time:** 15 min

- [ ] Implement `PaddleOCREngine` class in `core/ocr_engine.py`
- [ ] **Add dark mode preprocessing** (CRITICAL for Tesseract):
  ```python
  def _preprocess_dark_mode(image: np.ndarray) -> np.ndarray:
      """Invert dark images for better OCR accuracy."""
      if cv2.mean(image)[0] < 127:  # Dark image detected
          return cv2.bitwise_not(image)  # Invert to "book mode"
      return image
  ```
- [ ] Configure MPS backend for Apple Silicon:
  ```python
  import torch
  use_gpu = torch.backends.mps.is_available()
  ```
- [ ] Initialize PaddleOCR with MPS settings
- [ ] Convert OCR output to `list[OCRResult]`
- [ ] Add error handling (fall back to Tesseract if PaddleOCR fails)

**Files to modify:**
- `core/ocr_engine.py`

**Why Dark Mode Preprocessing:**
Tesseract was designed for black text on white paper. It fails on dark mode (white text on dark background). This simple inversion dramatically improves accuracy on terminals, IDEs, and dark-themed apps.

**Verification:**
```python
from screencloak.core.ocr_engine import PaddleOCREngine
import torch

# Check MPS available
print(f"MPS available: {torch.backends.mps.is_available()}")

# Test PaddleOCR
engine = PaddleOCREngine()
img_array = ...  # Same test image from Task 13
results = engine.detect_text(img_array)

assert len(results) > 0
print(f"OCR found {len(results)} text regions")
```

**Note:** PaddleOCR will download models (~500MB) on first run. Be patient!

---

## Phase 6: Screen Capture (Tasks 15-16)
**Goal:** Capture screen and diff frames
**Time:** ~1 session

### ✅ Task 15: Screen Capture
**Complexity:** Low | **Time:** 5 min

- [ ] Create `core/capture.py`
- [ ] Implement `ScreenCapture` class using `mss`:
  ```python
  def capture(self) -> np.ndarray:
      # Capture screen, return as numpy array
  ```
- [ ] Implement `MockCapture` class for testing:
  ```python
  def capture(self) -> np.ndarray:
      # Load image from file, return as numpy array
  ```
- [ ] Support monitor selection from config

**Files to create:**
- `core/capture.py`

**Verification:**
```python
from screencloak.core.capture import ScreenCapture
import cv2

capturer = ScreenCapture(config)
frame = capturer.capture()

assert frame is not None
assert frame.shape[2] == 3  # RGB image
cv2.imwrite("test_capture.png", frame)  # Save to verify
```

---

### ✅ Task 16: Frame Differencing
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `core/frame_diff.py`
- [ ] Implement `FrameDiffer` class
- [ ] Implement `get_changed_regions(prev_frame, curr_frame) -> Optional[np.ndarray]`:
  1. Convert to grayscale
  2. Compute absolute difference
  3. Threshold and find contours
  4. Return cropped region if >5% changed, else None
- [ ] Add `_merge_bboxes(bboxes)` helper

**Files to create:**
- `core/frame_diff.py`

**Verification:**
```python
from screencloak.core.frame_diff import FrameDiffer
import numpy as np

differ = FrameDiffer()

# Create two nearly identical frames
frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
frame2 = frame1.copy()
frame2[100:200, 100:200] = 255  # Add white square

changed = differ.get_changed_regions(frame1, frame2)

assert changed is not None  # Should detect change
```

---

## Phase 7: OBS Integration (Task 17)
**Goal:** Connect to OBS + Setup Stream Delay (CRITICAL)
**Time:** ~1 session

### ✅ Task 17: OBS WebSocket Client + Stream Delay Setup
**Complexity:** Medium | **Time:** 15 min

- [ ] Create `core/obs_client.py`
- [ ] Implement `OBSClient` class
- [ ] Methods:
  - `connect()` - Connect to OBS WebSocket
  - `trigger_privacy_mode()` - Switch to privacy scene ("BRB" screen)
  - `_return_to_previous()` - Auto-return after delay
  - `disconnect()` - Clean disconnect
- [ ] Add threading for auto-return timer
- [ ] **Create Stream Delay setup guide** in `docs/OBS_STREAM_DELAY_SETUP.md`

**Files to create:**
- `core/obs_client.py`
- `docs/OBS_STREAM_DELAY_SETUP.md` (CRITICAL)

**🚨 CRITICAL: Stream Delay Setup Instructions**

Python OCR cannot react faster than OBS encodes frames. The solution is OBS's built-in Render Delay filter:

**Setup Steps (for user documentation):**
1. In OBS, go to Settings → Advanced
2. Find "Stream Delay" or add "Render Delay" filter to your main video output
3. Set delay to 2-5 seconds
4. **Result:**
   - Streamer sees real-time (0 delay)
   - ScreenCloak detects sensitive data in real-time (200-500ms)
   - Viewers see stream delayed by 5 seconds
   - ScreenCloak switches scene before sensitive frame reaches viewers
   - **Secret never leaves the encoder!**

**Why This Works:**
- Detection at T+0ms: ScreenCloak sees seed phrase
- Scene switch at T+500ms: OBS switches to "BRB" screen
- Stream broadcast at T+5000ms: Viewers see delayed safe stream
- **Sensitive data is redacted before it reaches viewers**

**Verification:**
```bash
# Requires OBS running with WebSocket enabled AND Stream Delay configured
python -c "
from screencloak.core.obs_client import OBSClient
from screencloak.core.config_manager import ConfigManager

config = ConfigManager.load()
client = OBSClient(config)
client.trigger_privacy_mode()
# Should see OBS switch scenes (streamer sees immediately, viewers see 5s later)
"
```

**Note:** Can skip if OBS not available - mock in tests instead

---

## Phase 8: Main Loop (Task 18)
**Goal:** Tie everything together
**Time:** ~1 session

### ✅ Task 18: Main Loop
**Complexity:** Medium | **Time:** 10 min

- [ ] Implement `main.py`
- [ ] Parse command-line args: `--mock <image>`, `--preview-ui`, `--benchmark`
- [ ] Initialize all components:
  - ConfigManager
  - ScreenCapture or MockCapture
  - OCREngine
  - DetectorPipeline
  - OBSClient (optional)
  - FrameDiffer
- [ ] Main loop:
  ```python
  while True:
      frame = capturer.capture()

      if frame_differ.has_changed(frame):
          ocr_results = ocr_engine.detect_text(frame)
          detections = detector_pipeline.scan(ocr_results)

          if detections:
              log_detection(detections)
              if obs_client:
                  obs_client.trigger_privacy_mode()

      time.sleep(1 / config.ocr.frame_sample_rate)
  ```
- [ ] Add graceful shutdown (SIGINT handler)

**Files to create:**
- `main.py`

**Verification:**
```bash
# Mock mode test
python main.py --mock data/test_images/seed_phrase_12word.png

# Should log detection and exit (or trigger OBS if connected)
cat logs/detections.log  # Should show detection
```

---

## Phase 9: Testing & Polish (Tasks 19-22)
**Goal:** Create test suite
**Time:** ~1-2 sessions

### ✅ Task 19: Unit Tests - Detectors
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `tests/test_detector.py`
- [ ] Write tests for each detector:
  - `test_seed_phrase_detection()`
  - `test_seed_phrase_false_positive()` (essay with scattered BIP-39 words)
  - `test_credit_card_luhn()`
  - `test_crypto_address_eth()`
  - `test_personal_strings_fuzzy()`
- [ ] Mock OCR results, don't require actual OCR

**Files to create:**
- `tests/test_detector.py`

**Verification:**
```bash
pytest tests/test_detector.py -v
# All tests should pass
```

---

### ✅ Task 20: Synthetic Test Images
**Complexity:** Low | **Time:** 10 min

- [ ] Create `data/test_images/` directory
- [ ] Generate test images using PIL:
  - `seed_phrase_12word.png` - 12 BIP-39 words
  - `seed_phrase_24word.png` - 24 BIP-39 words
  - `credit_card_visa.png` - Valid card number
  - `eth_address.png` - ETH address
  - `false_positive_essay.png` - Normal text with some BIP-39 words
  - `mixed_content.png` - Multiple detection types
- [ ] Script: `tests/generate_test_images.py`

**Files to create:**
- `tests/generate_test_images.py`
- `data/test_images/*.png` (6 images)

**Verification:**
```bash
python tests/generate_test_images.py
ls data/test_images/  # Should show 6 PNG files
```

---

### ✅ Task 21: Integration Tests
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `tests/test_integration.py`
- [ ] Test full pipeline with mock images:
  - Load test image
  - Run OCR
  - Run detection
  - Verify correct detection type
- [ ] Test each synthetic image from Task 20

**Files to create:**
- `tests/test_integration.py`

**Verification:**
```bash
pytest tests/test_integration.py -v
# All tests should pass
```

---

### ✅ Task 22: Benchmark Suite
**Complexity:** Medium | **Time:** 10 min

- [ ] Create `tests/benchmark.py`
- [ ] Measure:
  - True positive rate (detected secrets correctly)
  - False positive rate (flagged normal text)
  - OCR latency (P50, P95)
  - End-to-end latency
- [ ] Run on all test images
- [ ] Generate report

**Files to create:**
- `tests/benchmark.py`

**Verification:**
```bash
python tests/benchmark.py

# Expected output:
# True Positive Rate: >95%
# False Positive Rate: <5%
# P50 Latency: ~200ms (MPS) or ~300ms (CPU)
# P95 Latency: <500ms
```

---

## Phase 10: Documentation (Tasks 23-24)
**Goal:** Usage instructions
**Time:** ~1 session

### ✅ Task 23: README
**Complexity:** Low | **Time:** 10 min

- [ ] Create comprehensive `README.md` with:
  - Project description
  - Installation instructions
  - OBS setup guide
  - Usage examples
  - Configuration reference
  - Troubleshooting

**Files to create:**
- `README.md`

---

### ✅ Task 24: OBS Setup Guide
**Complexity:** Low | **Time:** 5 min

- [ ] Create `docs/OBS_SETUP.md` with:
  - How to enable OBS WebSocket
  - How to create "Privacy Mode" scene
  - How to configure ScreenCloak for OBS
  - Screenshots (optional)

**Files to create:**
- `docs/OBS_SETUP.md`

---

## Important Notes & Updates

### ⚠️ M4 (Native C++ Plugin) is OPTIONAL

**Status:** The original plan included M4 (Native OBS Plugin) as a required milestone. **This is no longer required for launch.**

The Python sidecar application IS the production product, not a temporary prototype. A native C++ OBS plugin is optional future work that should only be considered after:
- 1,000+ users validate the Python version
- Strong user demand for positioned blur overlays (vs scene switching)
- Community contributors available to help with C++ development

**Focus:** Make the Python app production-ready with packaging (PyInstaller), system tray integration, and polished UX.

### ✅ Red Team Feedback Incorporated

This plan has been updated based on expert feedback on latency, OCR challenges, and architecture:

**Critical Fixes Added:**
1. **Stream Delay Solution** (Task 17) - OBS Render Delay filter solves latency problem
2. **Dark Mode Preprocessing** (Task 14) - Image inversion for Tesseract OCR accuracy
3. **Fuzzy Digit Cleanup** (Task 8) - Handle OCR noise (`l`→`1`, `I`→`1`, `O`→`0`)
4. **Sidecar Architecture** - Reframed Python app as production solution, not prototype

**Why These Matter:**
- Without Stream Delay, Python OCR can't react faster than OBS encodes frames
- Without dark mode preprocessing, Tesseract fails on terminals, IDEs, dark themes
- Without fuzzy digit cleanup, credit card detection misses `4532 l488...` OCR errors

---

## Progress Tracking

**Phase 1 - Foundation:** ✅ 4/4 tasks (COMPLETE)
**Phase 2 - Detection Foundation:** ✅ 3/3 tasks (COMPLETE)
**Phase 3 - More Detectors:** ✅ 4/4 tasks (COMPLETE)
**Phase 4 - Detection Pipeline:** ✅ 1/1 task (COMPLETE)
**Phase 5 - OCR Integration:** ☐ 0/2 tasks
**Phase 6 - Screen Capture:** ☐ 0/2 tasks
**Phase 7 - OBS Integration:** ☐ 0/1 task
**Phase 8 - Main Loop:** ☐ 0/1 task
**Phase 9 - Testing & Polish:** ☐ 0/4 tasks
**Phase 10 - Documentation:** ✅ README done (Task 23 partial)

**Total:** 12/24 tasks completed (50%)

---

## Session Planning

**Suggested Session Breakdown:**

- **Session 1:** Tasks 1-4 (Foundation)
- **Session 2:** Tasks 5-7 (Detection Foundation)
- **Session 3:** Tasks 8-11 (Detectors)
- **Session 4:** Tasks 12-14 (Pipeline + OCR)
- **Session 5:** Tasks 15-18 (Capture + Main Loop)
- **Session 6:** Tasks 19-21 (Testing)
- **Session 7:** Tasks 22-24 (Benchmarks + Docs)

---

## Quick Start (First Session)

To start building today:

1. Say: **"Let's build Task 1"**
2. I'll create the project structure
3. We'll verify it works
4. Move to Task 2, and so on...

When you run out of tokens:
1. Note the last completed task number
2. Next session, say: **"Let's resume from Task X"**

---

## Success Criteria for M1 Completion

All 24 tasks completed AND:
- [ ] All unit tests pass (`pytest tests/`)
- [ ] Benchmark shows >95% true positive, <5% false positive
- [ ] Can run: `python main.py --mock data/test_images/seed_phrase_12word.png`
- [ ] Detection logged correctly in `logs/detections.log`
- [ ] OBS integration tested (manual or mocked)

---

**Ready to start? Just say "Let's build Task 1" and we'll begin!** 🚀
