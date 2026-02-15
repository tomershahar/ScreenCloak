# ScreenCloak Python Prototype - Implementation Plan

## Goal Description
Build a functional standalone Python prototype for "ScreenCloak" that captures screen content, runs OCR to detect sensitive information (seed phrases, credit cards, etc.), and integrates with OBS via WebSocket to redact or hide content.

## User Review Required
> [!IMPORTANT]
> **Headless Limitations**: As an AI, I cannot see your screen or run the actual OBS application. I will build the system with a "Mock Mode" that accepts image/video files to verify the detection logic. You will need to test the live screen capture and OBS integration on your machine.

> [!NOTE]
> **Dependencies**: The prototype will use `PaddleOCR` (for text recognition), `opencv-python` (for image processing), `mss` (for fast screen capture), and `obs-websocket-py` (for OBS control). These are standard but can be large.

## Feasibility Assessment
**Can we build it?**
**YES.** The Python prototype approach is highly feasible and well-suited for a V0/V1 to validate the core value proposition.

*   **Detection**: `PaddleOCR` is robust enough for screen text.
*   **Performance**: Sampling every 5-10 frames (as suggested in PRD) makes Python performance acceptable (~200-500ms latency).
*   **Integration**: `obs-websocket` provides sufficient control to toggle sources or switch scenes, which is easier than writing a C++ graphics hook for a prototype.

**Risks:**
*   **OCR Latency**: Deep learning OCR models can be slow on CPU. We might need to optimize or use a lighter model/quantization if GPU isn't available.
*   **False Positives**: Keyword matching might flag innocent text. We'll need robust regex and context checking (e.g., BIP-39 word combos).

## Proposed Changes

### Project Structure
We will create a modular Python project:

```text
screencloak/
├── core/
│   ├── capture.py       # Screen capture (mss)
│   ├── ocr.py           # PaddleOCR wrapper
│   ├── detector.py      # Regex & Fuzzy matching logic
│   └── obs_client.py    # OBS WebSocket controller
├── data/
│   ├── bip39.txt        # Wordlist
│   └── text_patterns.json
├── tests/               # Unit tests & Mock data
└── main.py              # Main loop
```

### [SAFE_STREAM]
#### [NEW] [main.py](file:///Users/tomershahar/SafeSense/screencloak/main.py)
The entry point. Runs the loop: Capture -> OCR -> Detect -> Act.

#### [NEW] [core/capture.py](file:///Users/tomershahar/SafeSense/screencloak/core/capture.py)
Handles capturing the screen or specific monitor. Includes a 'MockCapture' class for testing with static images.

#### [NEW] [core/ocr.py](file:///Users/tomershahar/SafeSense/screencloak/core/ocr.py)
Wraps PaddleOCR (or Tesseract as fallback/lightweight option). Optimizes for speed (e.g., restricting region of interest).

#### [NEW] [core/detector.py](file:///Users/tomershahar/SafeSense/screencloak/core/detector.py)
The brain.
*   Loads BIP-39 wordlist.
*   Implements Luhn algorithm for CCs.
*   Regex for crypto addresses.
*   Returns bounding boxes of sensitive data.

#### [NEW] [core/obs_client.py](file:///Users/tomershahar/SafeSense/screencloak/core/obs_client.py)
Connects to OBS. Exposes methods like `trigger_privacy_mode()` (e.g., switch to "BRB" scene or toggle a blur source).

## Verification Plan

### Automated Tests
*   **Detection Logic**: Unit tests for `detector.py` feeding it strings containing secrets and ensuring they are caught.
*   **Data sanitization**: Ensure logs don't actually write out the detected secrets.

### Manual Verification
1.  **Mock Run**: Run `main.py --mock test_image.png` to see if it correctly identifies secrets in a screenshot.
2.  **Live Run**: User runs `main.py`, opens a Notepad, types a fake credit card number, and verifies if the script outputs "DETECTED".
3.  **OBS Integration**: User configures OBS WebSocket, runs script, adds a "Sensitive" source, and checks if it toggles.
