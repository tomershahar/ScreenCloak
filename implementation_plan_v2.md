# SafeStream Python Prototype - Implementation Plan V2

## Goal Description
Build a functional standalone Python prototype for "SafeStream" that captures screen content, runs OCR to detect sensitive information (seed phrases, credit cards, wallet addresses, API keys, personal data), and integrates with OBS via WebSocket to redact or hide content in real-time.

**V2 Changes:** Enhanced performance strategy with multi-OCR engine support, improved false positive management, comprehensive security model, and expanded testing framework.

---

## User Review Required

> [!IMPORTANT]
> **Headless Limitations**: As an AI, I cannot see your screen or run the actual OBS application. I will build the system with a "Mock Mode" that accepts image/video files to verify the detection logic. You will need to test the live screen capture and OBS integration on your machine.

> [!NOTE]
> **Dependencies**: The prototype will use multiple OCR engines for flexibility:
> - **PaddleOCR** (highest accuracy, GPU-accelerated, ~500MB)
> - **EasyOCR** (balanced, ~300MB)
> - **Tesseract** (fastest, CPU-friendly, ~5MB)
>
> Plus: `opencv-python` (image processing), `mss` (screen capture), `obs-websocket-py` (OBS control), `rapidfuzz` (fuzzy matching)

---

## Feasibility Assessment

**Can we build it?**

**YES.** The Python prototype approach is highly feasible and well-suited for V0/V1 to validate the core value proposition.

**Strengths:**
- **Detection**: Multiple OCR engines provide fallback options for different hardware
- **Performance**: Frame sampling + region-of-interest detection makes latency manageable
- **Integration**: `obs-websocket` provides sufficient control for V1 (scene switching)
- **Validation**: Can test detection logic without building C++ plugin first

**Risks & Mitigations:**

| Risk | Mitigation |
|------|------------|
| **OCR Latency** (1-3s per frame on CPU) | Multi-engine support; auto-detect GPU; skip unchanged regions via frame diffing |
| **False Positives** (BIP-39 words in normal text) | Require full 12/24 word sequences; spatial clustering; confidence scoring; user whitelists |
| **OBS Limitations** (WebSocket can't draw positioned overlays) | V1 does scene-switch blur; surgical bounding-box blur deferred to native plugin (M4) |
| **Memory Leaks** (OCR models in long-running process) | Periodic model reload; memory monitoring; graceful degradation |

---

## Proposed Changes

### Project Structure

```text
safestream/
├── main.py                    # Entry point & main loop
├── config.yaml                # User configuration (NEW)
├── requirements.txt           # Python dependencies
├── README.md                  # Setup & usage instructions
│
├── core/
│   ├── __init__.py
│   ├── capture.py             # Screen capture (mss) + MockCapture
│   ├── ocr_engine.py          # Multi-engine OCR wrapper (NEW)
│   ├── detector.py            # Detection logic with confidence scoring
│   ├── obs_client.py          # OBS WebSocket controller
│   ├── frame_diff.py          # OpenCV frame differencing (NEW)
│   └── config_manager.py      # Config loading & validation (NEW)
│
├── detectors/                 # Detection modules (NEW structure)
│   ├── __init__.py
│   ├── base.py                # Base detector interface
│   ├── seed_phrase.py         # BIP-39 seed phrase detection
│   ├── credit_card.py         # Credit card + Luhn validation
│   ├── crypto_address.py      # BTC, ETH, SOL, etc.
│   ├── api_keys.py            # AWS, Stripe, GitHub tokens
│   └── personal_strings.py    # User-defined fuzzy matching
│
├── data/
│   ├── bip39_wordlist.txt     # 2048 BIP-39 words
│   ├── api_patterns.json      # Regex patterns for API keys
│   └── test_images/           # Synthetic test dataset (NEW)
│       ├── seed_phrase_12word.png
│       ├── seed_phrase_24word.png
│       ├── credit_card_visa.png
│       ├── eth_address.png
│       ├── false_positive_essay.png
│       └── mixed_content.png
│
├── tests/
│   ├── __init__.py
│   ├── test_detector.py       # Unit tests for detection logic
│   ├── test_ocr.py            # OCR engine tests
│   ├── benchmark.py           # Performance & accuracy benchmarks (NEW)
│   └── test_integration.py    # End-to-end mock tests
│
├── ui/                        # Optional: Detection preview GUI (NEW)
│   ├── __init__.py
│   └── preview.py             # Tkinter/PyQt window showing OCR + detections
│
└── logs/
    └── detections.log         # Sanitized detection logs (no actual secrets)
```

---

## Core Components

### [NEW] `config.yaml` - User Configuration
```yaml
# OCR Settings
ocr:
  engine: "auto"  # auto, paddleocr, easyocr, tesseract
  gpu_enabled: true
  frame_sample_rate: 5  # Process every Nth frame

# Detection Settings
detection:
  seed_phrases:
    enabled: true
    min_word_count: 12  # Require 12 or 24 word sequences
  credit_cards:
    enabled: true
  crypto_addresses:
    enabled: true
    networks: ["BTC", "ETH", "SOL"]
  api_keys:
    enabled: false  # Paid tier feature
  personal_strings:
    enabled: true
    max_free: 3  # Free tier limit
    strings:
      - "John Doe"
      - "johndoe@email.com"
      - "555-123-4567"
    fuzzy_threshold: 85  # 0-100, lower = more lenient matching

# Screen Capture
capture:
  monitor: 1  # 0 = all monitors, 1 = primary, 2 = secondary, etc.
  roi_enabled: true  # Only OCR changed regions

# OBS Integration
obs:
  enabled: true
  host: "localhost"
  port: 4455
  password: ""
  privacy_scene: "Privacy Mode"  # Scene to switch to on detection
  auto_return: true  # Return to previous scene after N seconds
  return_delay: 3

# Privacy & Logging
privacy:
  log_detections: true
  log_sanitized: true  # Log "[REDACTED]" instead of actual data
  telemetry_opt_in: false

# Performance
performance:
  max_latency_ms: 500  # Warn if OCR takes longer
  memory_limit_mb: 2048  # Restart OCR engine if exceeded
```

### [main.py](file:///Users/tomershahar/SafeSense/safestream/main.py) - Entry Point

**Purpose:** Run the main detection loop: Capture → Frame Diff → OCR → Detect → Act

**Key Features:**
- Command-line args: `--mock <image>`, `--benchmark`, `--preview-ui`
- Graceful shutdown (SIGINT handler)
- Telemetry opt-in prompt on first run
- Performance monitoring (log latency warnings)

**Pseudocode:**
```python
def main():
    config = ConfigManager.load()
    capturer = ScreenCapture(config) or MockCapture(args.mock_image)
    ocr_engine = OCREngineFactory.create(config)  # Auto-select best engine
    detector = DetectorPipeline(config)
    obs_client = OBSClient(config) if config.obs.enabled else None
    frame_differ = FrameDiffer()

    previous_frame = None

    while True:
        # Capture
        frame = capturer.capture()

        # Frame diff - only OCR changed regions
        if previous_frame and config.capture.roi_enabled:
            changed_regions = frame_differ.get_changed_regions(previous_frame, frame)
            if not changed_regions:
                continue  # No changes, skip OCR
            frame_to_ocr = changed_regions
        else:
            frame_to_ocr = frame

        # OCR
        ocr_results = ocr_engine.detect_text(frame_to_ocr)

        # Detect
        detections = detector.scan(ocr_results)

        # Act
        if detections:
            log_detection(detections, sanitized=config.privacy.log_sanitized)
            if obs_client:
                obs_client.trigger_privacy_mode()

        previous_frame = frame
        time.sleep(1 / config.ocr.frame_sample_rate)
```

---

### [core/ocr_engine.py](file:///Users/tomershahar/SafeSense/safestream/core/ocr_engine.py) - Multi-Engine OCR Wrapper

**Purpose:** Abstract OCR engine selection with auto-detection and fallback.

**Classes:**
```python
class OCREngine(ABC):
    @abstractmethod
    def detect_text(self, image: np.ndarray) -> List[OCRResult]:
        """Returns list of (text, bounding_box, confidence)"""
        pass

class PaddleOCREngine(OCREngine):
    # Highest accuracy, requires GPU for acceptable speed

class EasyOCREngine(OCREngine):
    # Balanced accuracy/speed

class TesseractEngine(OCREngine):
    # Fastest, lowest accuracy

class OCREngineFactory:
    @staticmethod
    def create(config) -> OCREngine:
        if config.ocr.engine == "auto":
            if has_gpu():
                return PaddleOCREngine()
            else:
                return TesseractEngine()  # Fallback for CPU
        # else return specified engine
```

**Auto-Selection Logic:**
1. Check for CUDA/MPS (Apple Silicon GPU)
2. If GPU available → PaddleOCR
3. If CPU only → Tesseract
4. User can override via config

---

### [core/detector.py](file:///Users/tomershahar/SafeSense/safestream/core/detector.py) - Detection Pipeline

**Purpose:** Coordinate all detection modules and score confidence.

**Key Features:**
- Load enabled detectors from config
- Aggregate results with confidence scores
- Spatial clustering (group nearby matches)
- Whitelist/ignore zones

**Structure:**
```python
class DetectionResult:
    type: str  # "seed_phrase", "credit_card", etc.
    confidence: float  # 0.0 - 1.0
    text_preview: str  # First 10 chars for logging
    bounding_box: Tuple[int, int, int, int]
    action: str  # "blur", "warn", "ignore"

class DetectorPipeline:
    def __init__(self, config):
        self.detectors = [
            SeedPhraseDetector(config),
            CreditCardDetector(config),
            CryptoAddressDetector(config),
            # ... load based on config.detection.*.enabled
        ]
        self.ignore_zones = config.capture.ignore_zones or []

    def scan(self, ocr_results: List[OCRResult]) -> List[DetectionResult]:
        all_detections = []

        for detector in self.detectors:
            detections = detector.detect(ocr_results)
            all_detections.extend(detections)

        # Filter ignore zones
        all_detections = self._filter_ignore_zones(all_detections)

        # Spatial clustering (merge nearby matches)
        all_detections = self._cluster_spatially(all_detections)

        # Confidence-based action assignment
        for detection in all_detections:
            if detection.confidence > 0.9:
                detection.action = "blur"
            elif detection.confidence > 0.6:
                detection.action = "warn"
            else:
                detection.action = "ignore"

        return [d for d in all_detections if d.action != "ignore"]
```

---

### [detectors/seed_phrase.py](file:///Users/tomershahar/SafeSense/safestream/detectors/seed_phrase.py) - BIP-39 Detection

**Purpose:** Detect 12 or 24 word seed phrases from BIP-39 wordlist.

**Algorithm:**
1. Load 2048-word BIP-39 wordlist into set
2. Tokenize OCR text into words
3. Check for **sequential runs** of 12 or 24 words from wordlist
4. Verify spatial proximity (words on same line or within 50px vertically)
5. Score confidence based on:
   - Exact word count match (12/24 = high confidence)
   - Spatial clustering tightness
   - Absence of non-wordlist words in sequence

**False Positive Prevention:**
```python
def detect(self, ocr_results):
    words = self._tokenize(ocr_results)
    bip39_words = [w for w in words if w.text.lower() in BIP39_WORDLIST]

    # Require 12 or 24 consecutive BIP-39 words
    sequences = self._find_consecutive_runs(bip39_words, min_length=12)

    # Filter by spatial proximity
    valid_sequences = []
    for seq in sequences:
        if self._are_spatially_clustered(seq, max_distance=50):
            # Check if it's exactly 12 or 24 words
            if len(seq) in [12, 24]:
                confidence = 0.95
            elif len(seq) >= 12:
                confidence = 0.7  # Might be partial phrase
            else:
                continue  # Too short, ignore

            valid_sequences.append(DetectionResult(
                type="seed_phrase",
                confidence=confidence,
                text_preview=f"{seq[0].text}...{seq[-1].text}",
                bounding_box=self._merge_boxes(seq)
            ))

    return valid_sequences
```

---

### [detectors/credit_card.py](file:///Users/tomershahar/SafeSense/safestream/detectors/credit_card.py) - Credit Card Detection

**Purpose:** Detect 16-digit credit card numbers with Luhn validation.

**Algorithm:**
1. Regex for 16-digit sequences (with optional spaces/dashes)
2. Luhn algorithm validation
3. Check for nearby expiration date patterns (MM/YY)
4. Confidence boosted if CVV pattern found nearby

**Implementation:**
```python
def detect(self, ocr_results):
    text = " ".join([r.text for r in ocr_results])

    # Regex: 16 digits with optional separators
    matches = re.finditer(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', text)

    detections = []
    for match in matches:
        card_number = match.group().replace(' ', '').replace('-', '')

        if self._luhn_check(card_number):
            # Find bounding box
            bbox = self._find_bbox_for_text(match.group(), ocr_results)

            # Check for nearby expiration (confidence boost)
            nearby_text = self._get_nearby_text(bbox, ocr_results, radius=100)
            has_expiry = bool(re.search(r'\b\d{2}/\d{2}\b', nearby_text))

            confidence = 0.95 if has_expiry else 0.8

            detections.append(DetectionResult(
                type="credit_card",
                confidence=confidence,
                text_preview=f"{card_number[:4]}...{card_number[-4:]}",
                bounding_box=bbox
            ))

    return detections

def _luhn_check(self, card_number: str) -> bool:
    # Standard Luhn algorithm implementation
    ...
```

---

### [detectors/crypto_address.py](file:///Users/tomershahar/SafeSense/safestream/detectors/crypto_address.py) - Crypto Address Detection

**Purpose:** Detect Bitcoin, Ethereum, Solana, and other crypto addresses via regex.

**Patterns:**
```python
PATTERNS = {
    "BTC": r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b|bc1[a-z0-9]{39,59}\b',
    "ETH": r'\b0x[a-fA-F0-9]{40}\b',
    "SOL": r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b',  # Base58
    # Add more as needed
}
```

**Validation:**
- BTC: Checksum validation (base58check)
- ETH: Checksum validation (EIP-55)
- SOL: Length validation

---

### [detectors/personal_strings.py](file:///Users/tomershahar/SafeSense/safestream/detectors/personal_strings.py) - User-Defined String Detection

**Purpose:** Fuzzy match user-provided personal info (name, email, phone, address).

**Algorithm:**
```python
from rapidfuzz import fuzz

def detect(self, ocr_results):
    text = " ".join([r.text for r in ocr_results])

    detections = []
    for personal_string in self.config.detection.personal_strings.strings:
        # Fuzzy match
        ratio = fuzz.partial_ratio(personal_string.lower(), text.lower())

        if ratio >= self.config.detection.personal_strings.fuzzy_threshold:
            # Find exact location
            bbox = self._find_bbox_for_fuzzy_match(personal_string, ocr_results)

            detections.append(DetectionResult(
                type="personal_string",
                confidence=ratio / 100.0,
                text_preview=f"{personal_string[:10]}...",
                bounding_box=bbox
            ))

    return detections
```

---

### [core/obs_client.py](file:///Users/tomershahar/SafeSense/safestream/core/obs_client.py) - OBS WebSocket Controller

**Purpose:** Connect to OBS and trigger privacy mode (scene switch or source toggle).

**Capabilities (V1 - WebSocket API):**
- ✅ Switch to "Privacy Mode" scene
- ✅ Toggle visibility of specific sources
- ✅ Auto-return to previous scene after N seconds
- ❌ Draw positioned overlays (requires native plugin in M4)

**Implementation:**
```python
from obswebsocket import obsws, requests

class OBSClient:
    def __init__(self, config):
        self.ws = obsws(config.obs.host, config.obs.port, config.obs.password)
        self.ws.connect()
        self.privacy_scene = config.obs.privacy_scene
        self.previous_scene = None
        self.auto_return = config.obs.auto_return
        self.return_delay = config.obs.return_delay

    def trigger_privacy_mode(self):
        # Save current scene
        self.previous_scene = self.ws.call(requests.GetCurrentProgramScene()).getName()

        # Switch to privacy scene
        self.ws.call(requests.SetCurrentProgramScene(sceneName=self.privacy_scene))

        # Auto-return after delay
        if self.auto_return:
            threading.Timer(self.return_delay, self._return_to_previous).start()

    def _return_to_previous(self):
        if self.previous_scene:
            self.ws.call(requests.SetCurrentProgramScene(sceneName=self.previous_scene))
```

**OBS Scene Setup (User Instructions):**
```markdown
1. Create a new scene called "Privacy Mode"
2. Add a black/blurred background image
3. Add text: "Sensitive information detected - Be right back!"
4. In SafeStream config, set `obs.privacy_scene: "Privacy Mode"`
```

---

### [core/frame_diff.py](file:///Users/tomershahar/SafeSense/safestream/core/frame_diff.py) - Frame Differencing

**Purpose:** Detect changed regions to avoid OCR on static content.

**Algorithm:**
```python
import cv2

class FrameDiffer:
    def __init__(self, threshold=30):
        self.threshold = threshold

    def get_changed_regions(self, prev_frame, curr_frame) -> Optional[np.ndarray]:
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)

        # Threshold
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # If less than 5% of pixels changed, skip OCR
        change_percentage = (cv2.countNonZero(thresh) / thresh.size) * 100
        if change_percentage < 5:
            return None

        # Find bounding boxes of changed regions
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Merge nearby contours
        bboxes = [cv2.boundingRect(c) for c in contours]
        merged_bbox = self._merge_bboxes(bboxes)

        # Crop to changed region
        x, y, w, h = merged_bbox
        return curr_frame[y:y+h, x:x+w]
```

---

## Security & Privacy Architecture

### 1. Log Sanitization
```python
def log_detection(detection: DetectionResult, sanitized: bool):
    if sanitized:
        logged_text = "[REDACTED]"
        logged_bbox = detection.bounding_box  # Coordinates are safe
    else:
        logged_text = detection.text_preview  # Already truncated
        logged_bbox = detection.bounding_box

    log_entry = {
        "timestamp": time.time(),
        "type": detection.type,
        "confidence": detection.confidence,
        "text": logged_text,
        "bbox": logged_bbox
    }

    with open("logs/detections.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

### 2. Memory Safety
```python
# Clear OCR results from memory after processing
def secure_clear(data):
    if isinstance(data, str):
        # Overwrite string in memory (best effort in Python)
        ctypes.memset(id(data) + 20, 0, len(data))
    elif isinstance(data, list):
        for item in data:
            secure_clear(item)
        data.clear()
```

### 3. Telemetry (Opt-In)
```yaml
# Only if user opts in via config.privacy.telemetry_opt_in = true
telemetry:
  - event: "detection_triggered"
    data:
      type: "seed_phrase"  # Category only, no actual data
      confidence: 0.95
      timestamp: <unix_timestamp>
      ocr_engine: "paddleocr"
      latency_ms: 324
```

**What is NOT sent:**
- Actual detected text
- Screenshots or frame data
- User's personal strings
- File paths or usernames

---

## Testing Strategy

### 1. Unit Tests (`tests/test_detector.py`)

```python
def test_seed_phrase_detection():
    detector = SeedPhraseDetector(config)

    # 12-word seed phrase
    text = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    ocr_results = mock_ocr_results(text)

    detections = detector.detect(ocr_results)

    assert len(detections) == 1
    assert detections[0].type == "seed_phrase"
    assert detections[0].confidence > 0.9

def test_false_positive_prevention():
    detector = SeedPhraseDetector(config)

    # Essay with some BIP-39 words but not a sequence
    text = "I have the ability to abandon my fears and access new opportunities above my expectations"
    ocr_results = mock_ocr_results(text)

    detections = detector.detect(ocr_results)

    assert len(detections) == 0  # Should NOT trigger

def test_credit_card_luhn():
    detector = CreditCardDetector(config)

    # Valid test card number
    text = "4532 1488 0343 6467"  # Passes Luhn
    ocr_results = mock_ocr_results(text)

    detections = detector.detect(ocr_results)

    assert len(detections) == 1

    # Invalid number
    text = "4532 1488 0343 6468"  # Fails Luhn
    ocr_results = mock_ocr_results(text)

    detections = detector.detect(ocr_results)

    assert len(detections) == 0
```

### 2. Benchmark Suite (`tests/benchmark.py`)

```python
"""
Measure detection accuracy and OCR latency across test dataset.

Metrics:
- True positive rate (correctly detected secrets)
- False positive rate (incorrectly flagged normal text)
- OCR latency per frame
- End-to-end latency (capture → detect → log)
"""

def benchmark():
    test_images = [
        ("data/test_images/seed_phrase_12word.png", "seed_phrase", True),
        ("data/test_images/false_positive_essay.png", None, False),
        ("data/test_images/credit_card_visa.png", "credit_card", True),
        # ... all test images
    ]

    results = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "latencies": []
    }

    for image_path, expected_type, should_detect in test_images:
        start = time.time()

        frame = cv2.imread(image_path)
        ocr_results = ocr_engine.detect_text(frame)
        detections = detector.scan(ocr_results)

        latency = (time.time() - start) * 1000  # ms
        results["latencies"].append(latency)

        detected = len(detections) > 0

        if should_detect and detected:
            results["true_positives"] += 1
        elif should_detect and not detected:
            results["false_negatives"] += 1
        elif not should_detect and detected:
            results["false_positives"] += 1

    # Print report
    print(f"True Positive Rate: {results['true_positives'] / len(test_images):.2%}")
    print(f"False Positive Rate: {results['false_positives'] / len(test_images):.2%}")
    print(f"Average Latency: {np.mean(results['latencies']):.0f}ms")
    print(f"P95 Latency: {np.percentile(results['latencies'], 95):.0f}ms")
```

**Acceptance Criteria:**
- True positive rate > 95%
- False positive rate < 5%
- P95 latency < 500ms

### 3. Synthetic Test Dataset

Create `data/test_images/` with:

```python
# Script to generate test images
from PIL import Image, ImageDraw, ImageFont

def create_test_image(text, filename):
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    draw.text((50, 50), text, fill='black', font=font)
    img.save(f"data/test_images/{filename}")

# Seed phrase (12 words)
create_test_image(
    "abandon ability able about above absent absorb abstract absurd abuse access accident",
    "seed_phrase_12word.png"
)

# Seed phrase (24 words)
create_test_image(
    "abandon ability able about above absent absorb abstract absurd abuse access accident" +
    " acoustic acquire across act action actor actress actual adapt add addict address",
    "seed_phrase_24word.png"
)

# Credit card
create_test_image(
    "Card Number: 4532 1488 0343 6467\nExp: 12/25\nCVV: 123",
    "credit_card_visa.png"
)

# Ethereum address
create_test_image(
    "Send ETH to: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
    "eth_address.png"
)

# False positive test - essay with BIP-39 words
create_test_image(
    "I have the ability to abandon my fears. The abstract concept of freedom " +
    "allows me to access new opportunities above my previous expectations.",
    "false_positive_essay.png"
)

# Mixed content
create_test_image(
    "My name is John Doe\n" +
    "Email: john@example.com\n" +
    "Wallet: 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",
    "mixed_content.png"
)
```

---

## Revised Milestones

### **M1: Core Detection Engine** (Week 1-2)
**Goal:** Prove detection logic works with mock data.

**Deliverables:**
- [ ] Project structure setup
- [ ] Multi-OCR engine wrapper (Paddle, Easy, Tesseract)
- [ ] BIP-39 seed phrase detector (12/24 word sequences)
- [ ] Credit card detector (Luhn validation)
- [ ] Crypto address detector (BTC, ETH, SOL)
- [ ] Personal strings detector (fuzzy matching)
- [ ] Unit tests (>90% coverage)
- [ ] Mock mode: `python main.py --mock test_image.png`

**Success Criteria:**
- All unit tests pass
- Can detect secrets in synthetic test images

---

### **M1.5: Performance Optimization & Benchmarking** (Week 2-3)
**Goal:** Ensure latency < 500ms and false positive rate < 5%.

**Deliverables:**
- [ ] Benchmark suite on test dataset
- [ ] Frame differencing implementation
- [ ] OCR engine auto-selection based on hardware
- [ ] Confidence scoring & threshold tuning
- [ ] Spatial clustering for false positive reduction

**Success Criteria:**
- P95 latency < 500ms on typical hardware
- True positive rate > 95%
- False positive rate < 5%
- Frame diff reduces OCR calls by 60%+ on static screens

---

### **M2: OBS Integration** (Week 3-4)
**Goal:** Connect to OBS and trigger privacy scene on detection.

**Deliverables:**
- [ ] OBS WebSocket client
- [ ] Scene switching on detection
- [ ] Auto-return after configurable delay
- [ ] Setup instructions for OBS scenes
- [ ] Config file (YAML) for all settings

**Success Criteria:**
- User can run `python main.py` and see OBS scene switch when sensitive data appears
- Config allows customization without code changes

---

### **M2.5: UI & Debug Tools** (Week 4-5)
**Goal:** Make testing/debugging easier for users and beta testers.

**Deliverables:**
- [ ] Detection preview UI (optional Tkinter window showing OCR + bounding boxes)
- [ ] Enhanced logging (detection.log with timestamps)
- [ ] `--preview-ui` flag to show real-time OCR output
- [ ] Demo mode: Pre-recorded video with secrets → automated detection clips

**Success Criteria:**
- Beta testers can see what SafeStream "sees" via preview UI
- Demo video is shareable for marketing

---

### **M3: Closed Beta** (Week 5-8)
**Goal:** Validate in real-world streaming scenarios.

**Deliverables:**
- [ ] Recruit 10-20 beta testers (Reddit, Discord, Twitter)
- [ ] Setup guide & troubleshooting docs
- [ ] Feedback form (Google Forms / Typeform)
- [ ] API key detection (if paid tier unlocked)
- [ ] Telemetry opt-in (anonymous usage stats)

**Success Criteria:**
- 10+ active testers stream with SafeStream for 2+ weeks
- 3-5 documented "close calls" (SafeStream caught a real leak)
- False positive rate < 10% in real-world usage
- Collect testimonials & permission to use clips

---

### **M4: Native OBS Plugin** (Month 3-4)
**Goal:** Replace WebSocket with native C++ OBS filter for surgical bounding-box blurs.

**Deliverables:**
- [ ] Port detection engine to C++ (or keep Python, call via bridge)
- [ ] OBS filter plugin that draws blur overlays at coordinates
- [ ] Installer for Windows/Mac/Linux
- [ ] OBS plugin marketplace listing

**Success Criteria:**
- Blur appears directly on stream output (no scene switching)
- Latency < 200ms (native code faster than Python)
- Listed on official OBS Resources page

---

### **M5: Public Launch** (Month 4-5)
**Goal:** Release free tier publicly.

**Deliverables:**
- [ ] Landing page (safestream.io or similar)
- [ ] Installation instructions (OBS plugin marketplace + manual)
- [ ] Product Hunt / Hacker News launch
- [ ] Twitter/Reddit announcement
- [ ] Free tier live: seed phrase, credit card, 3 personal strings

**Success Criteria:**
- 1,000+ installs in first month
- Featured on OBS subreddit / streamer communities
- 5+ unsolicited social media mentions

---

### **M6: Paid Tier** (Month 5-6)
**Goal:** Monetize via premium features.

**Deliverables:**
- [ ] Stripe payment integration
- [ ] License key validation
- [ ] Unlock: unlimited personal strings, API key detection, custom blur styles
- [ ] Detection log viewer with "close call" review
- [ ] User dashboard (usage stats, renewal)

**Success Criteria:**
- 3-5% conversion rate (free → paid)
- $500+ MRR in first month post-launch
- Positive reviews on social media / streaming forums

---

## Quick Wins (Add to Roadmap)

1. **Demo Mode** (M2.5)
   - Pre-recorded 60-second video showing seed phrase → detection → blur
   - Auto-play in preview UI
   - Shareable MP4 for Twitter/Reddit

2. **Testimonial Kit** (M3)
   - Provide beta testers with:
     - Template tweet: "SafeStream just saved me from leaking my [redacted] on stream 🙏"
     - Instructions to clip the moment via OBS replay buffer
     - Permission form to share their clip

3. **"Close Call" Review** (M6 - Paid)
   - After stream, show user: "We detected 3 potential leaks today"
   - Click to see sanitized log: "12:34:56 PM - Seed phrase detected (confidence: 95%)"
   - Option to review screenshot with blur overlay

4. **Streamer Mode Integration** (Future)
   - Auto-detect Discord, Slack, email clients
   - Suggest ignore zones for chat overlays
   - One-click setup: "Protect my stream from common leaks"

---

## Next Steps

**Immediate (This Week):**
1. ✅ Review & approve this V2 implementation plan
2. 🔄 Set up Python project structure
3. 🔄 Implement multi-OCR engine wrapper (M1)
4. 🔄 Build BIP-39 seed phrase detector (M1)
5. 🔄 Create synthetic test dataset (M1)

**Week 2:**
- Credit card detector + Luhn validation
- Crypto address detector
- Unit tests
- Mock mode functional

**Week 3:**
- Benchmarking suite
- Frame differencing
- Performance tuning to hit <500ms latency

---

## Open Questions / Decisions Needed

1. **OCR Engine Priority:** Should we focus on PaddleOCR first (best accuracy) or Tesseract (easiest setup)?
   - **Recommendation:** Start with Tesseract for faster prototyping, add Paddle in M1.5.

2. **False Positive Handling:** Should warnings (confidence 0.6-0.9) be logged but not blurred?
   - **Recommendation:** Yes. V1 = blur on >0.9, log on >0.6, ignore <0.6.

3. **OBS Scene Naming:** Enforce "Privacy Mode" or let users customize?
   - **Recommendation:** Customizable via config, default to "Privacy Mode".

4. **License:** Open source (MIT/Apache) or proprietary?
   - **Recommendation:** Open source detection engine, proprietary paid tier features (license key validation).

5. **Paid Tier Pricing:** $5/month or $10/month?
   - **Recommendation:** $7/month or $60/year (saves $24). Test with beta users.

---

## Success Metrics (V2 Additions)

**Technical:**
- Latency: P95 < 500ms, P50 < 300ms
- Accuracy: True positive > 95%, False positive < 5%
- Memory: < 2GB RAM usage (OCR models + processing)
- CPU: < 25% on modern quad-core CPU

**Product:**
- M1-M2: Detection works in mock tests
- M3: 10+ beta testers, 3+ testimonials
- M5: 5,000+ free installs, 5+ unsolicited social mentions
- M6: 3-5% conversion rate, $500+ MRR

**Validation Checkpoint:**
After M3 (closed beta), decide: Proceed to M4 (native plugin) or pivot based on feedback.

---

## Resources & References

**BIP-39 Wordlist:**
https://github.com/bitcoin/bips/blob/master/bip-0039/english.txt

**Luhn Algorithm:**
https://en.wikipedia.org/wiki/Luhn_algorithm

**OBS WebSocket Protocol:**
https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md

**PaddleOCR:**
https://github.com/PaddlePaddle/PaddleOCR

**Crypto Address Validation:**
- Bitcoin: https://en.bitcoin.it/wiki/Address
- Ethereum: https://eips.ethereum.org/EIPS/eip-55

---

**Ready to start building M1?** 🚀
