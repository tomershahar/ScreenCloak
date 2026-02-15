# ScreenCloak Project - Claude Context

## Project Overview

**ScreenCloak** is an OBS Studio plugin that automatically detects and redacts sensitive information from a streamer's screen in real-time using OCR and pattern matching. It prevents accidental exposure of crypto seed phrases, wallet addresses, credit cards, API keys, and personal data during live streams.

**Current Phase:** M1 - Core Detection Engine (Week 1-2)
**Status:** Starting implementation from scratch
**Hardware:** Mac with Apple Silicon (M1/M2/M3) - GPU acceleration available via MPS

## Core Value Proposition

Streamers regularly leak sensitive info (seed phrases, passwords, credit cards) on stream with catastrophic consequences (documented $100K+ losses). No existing solution provides intelligent, cross-app, screen-level detection. ScreenCloak fills this gap.

## Technical Architecture

### V1 Approach: Production Sidecar Application

We're building a production-quality standalone Python application that runs alongside OBS (a "sidecar app"). This is NOT a temporary prototype - it's the actual product. A native C++ OBS plugin is optional future work, not a requirement for launch.

**Why Sidecar Architecture:**
- Rapid iteration without C++ complexity
- Cross-platform (Windows, Mac, Linux) via Python
- Easy updates and community contributions
- Can be packaged as standalone executable (PyInstaller)

**Detection Pipeline:**
```
Screen Capture → Frame Diff → OCR → Pattern Detection → OBS Scene Switch
                                                              ↓
                                                    Stream Delay (2-5s)
                                                              ↓
                                                      Viewers see delayed stream
```

**Key Components:**
1. **Multi-OCR Engine** (PaddleOCR primary, Tesseract/EasyOCR fallback)
2. **Detection Modules** (seed phrases, credit cards, crypto addresses, API keys, personal strings)
3. **Frame Differencing** (skip OCR on unchanged regions - 60%+ performance gain)
4. **OBS WebSocket Integration** (scene switching to "BRB" screen)
5. **Stream Delay** (CRITICAL: 2-5 second OBS Render Delay filter required)
6. **Config Management** (YAML-based user settings)

### Detection Algorithms

**Seed Phrases (BIP-39):**
- Require 12 or 24 consecutive words from 2048-word BIP-39 wordlist
- Spatial clustering (words must be within 50px vertically)
- Confidence scoring: exact 12/24 = 0.95, partial sequences = 0.7

**Credit Cards:**
- 16-digit regex with Luhn algorithm validation
- **OCR noise cleanup:** Fuzzy digit mapping (`l`→`1`, `I`→`1`, `O`→`0`) before validation
- Confidence boost if nearby expiration date (MM/YY) or CVV pattern

**Crypto Addresses:**
- Regex patterns for BTC (legacy + bech32), ETH (0x...), SOL (base58)
- Checksum validation where applicable
- **BTC/SOL collision fix:** Process networks in priority order BTC→ETH→SOL, use a `detected_addresses` set to deduplicate. A BTC address also matches the SOL base58 regex — without this, both fire.

**Personal Strings:**
- Fuzzy matching with configurable threshold (default 85%)
- User provides name, email, phone, address via config

**API Keys:**
- Pattern matching for AWS, Stripe, GitHub, etc. (paid tier feature)

### Performance Targets

- **Detection Latency:** 200-500ms (acceptable because of Stream Delay buffer)
- **Stream Delay:** 2-5 seconds (required OBS Render Delay filter)
- **Accuracy:** True positive > 95%, False positive < 5%
- **Memory:** < 2GB RAM
- **CPU:** < 25% on modern quad-core

**CRITICAL: Stream Delay is The Solution**
We do NOT need to achieve "zero latency" - that's impossible with Python OCR. Instead:
- User adds OBS "Render Delay" filter (2-5 seconds) to stream output
- Streamer sees real-time (0 delay)
- Script detects sensitive data in real-time (200-500ms)
- Viewers see stream delayed by 5 seconds
- Script triggers scene switch before sensitive frame reaches viewers
- **Result:** Secret never leaves the encoder

## Project Structure

```
screencloak/
├── main.py                    # Entry point & main loop
├── config.yaml                # User configuration
├── requirements.txt           # Dependencies
├── pyproject.toml            # Project metadata & tooling config
├── README.md                  # Setup & usage
│
├── core/
│   ├── capture.py            # Screen capture (mss) + MockCapture
│   ├── ocr_engine.py         # Multi-engine OCR wrapper
│   ├── detector.py           # Detection pipeline
│   ├── obs_client.py         # OBS WebSocket controller
│   ├── frame_diff.py         # OpenCV frame differencing
│   └── config_manager.py     # Config loading & validation
│
├── detectors/
│   ├── base.py               # Base detector interface
│   ├── seed_phrase.py        # BIP-39 detection
│   ├── credit_card.py        # Luhn validation
│   ├── crypto_address.py     # BTC/ETH/SOL
│   ├── api_keys.py           # Token patterns
│   └── personal_strings.py   # Fuzzy matching
│
├── data/
│   ├── bip39_wordlist.txt    # 2048 BIP-39 words
│   ├── api_patterns.json     # API key regex patterns
│   └── test_images/          # Synthetic test dataset
│
├── tests/
│   ├── test_detector.py      # Unit tests
│   ├── test_ocr.py           # OCR engine tests
│   ├── benchmark.py          # Performance tests
│   └── test_integration.py   # End-to-end tests
│
├── ui/
│   └── preview.py            # Detection preview GUI (optional)
│
└── logs/
    └── detections.log        # Sanitized logs
```

## Development Guidelines

### Code Style & Quality

**Tooling:**
- **pytest** for testing (>90% coverage required for M1)
- **black** for code formatting (line length: 100)
- **mypy** for type checking (strict mode)
- **ruff** for fast linting (optional, can add later)

**Type Annotations:**
- All functions must have type hints (params + return)
- Use `from __future__ import annotations` for forward references
- Import types from `typing` module

**Testing:**
- Unit tests for all detection logic
- Mock OCR results for fast tests
- Benchmark suite on synthetic images
- Integration tests with OBS WebSocket (mocked)

**Error Handling:**
- Graceful degradation (e.g., if OCR fails, log and continue)
- User-friendly error messages in logs
- Never crash - ScreenCloak runs as long-running process

### Security & Privacy

**CRITICAL REQUIREMENTS:**

1. **Never log actual sensitive data**
   - Use `[REDACTED]` in logs when `config.privacy.log_sanitized = true`
   - Only log first 10 chars as `text_preview` for debugging

2. **Memory safety**
   - Clear OCR results from memory after processing
   - Periodic garbage collection for long-running process

3. **Telemetry opt-in only**
   - Prompt user on first run
   - Never send actual detected text, only detection type + confidence

4. **Config validation**
   - Validate user-provided strings before using in regex
   - Sanitize file paths to prevent directory traversal

### Apple Silicon Optimization

**OCR Engine Selection:**
- **Primary:** PaddleOCR with MPS (Metal Performance Shaders) backend
- **Fallback:** Tesseract if PaddleOCR fails to initialize
- Auto-detect GPU via `torch.backends.mps.is_available()`

**Installation Notes:**
- PaddleOCR requires PyTorch with MPS support
- Use `pip install torch torchvision torchaudio` (official Apple Silicon wheels)
- PaddleOCR models (~500MB) download on first run

**Performance Expectations:**
- MPS-accelerated PaddleOCR: ~100-200ms per frame
- Tesseract (CPU): ~300-500ms per frame
- Frame diffing reduces OCR calls by 60%+ on static screens

**Dark Mode Preprocessing (CRITICAL for Tesseract):**
Tesseract was designed for black text on white paper. It fails on dark mode (white text on dark background).
```python
# Invert dark images before OCR
if cv2.mean(image)[0] < 127:  # Dark image detected
    image = cv2.bitwise_not(image)  # Invert to "book mode"
```
This simple fix dramatically improves Tesseract accuracy on terminals, IDEs, and dark-themed apps.

## Milestones & Acceptance Criteria

### M1: Core Detection Engine (Current)

**Deliverables:**
- [x] Project structure
- [x] Config manager (`core/config_manager.py`)
- [x] Logger (`core/logger.py`)
- [x] Base detector interface (`detectors/base.py`)
- [x] Seed phrase detector (`detectors/seed_phrase.py`)
- [x] Credit card detector (`detectors/credit_card.py`)
- [x] Crypto address detector (`detectors/crypto_address.py`)
- [x] Personal strings detector (`detectors/personal_strings.py`)
- [x] API keys stub (`detectors/api_keys.py`)
- [x] Detector pipeline (`core/detector.py`)
- [x] README with honest security model
- [ ] Multi-OCR engine wrapper (Task 13-14, next)
- [ ] Screen capture (Task 15-16)
- [ ] OBS client (Task 17)
- [ ] Main loop (Task 18)
- [ ] Unit tests (Task 19)
- [ ] Mock mode: `python main.py --mock test_image.png`

**Next task: Task 13 — OCR Engine Interface**

**Success Criteria:**
- All unit tests pass
- Can detect secrets in synthetic test images
- No false positives on essay with scattered BIP-39 words

### M1.5: Performance Optimization (Next)

**Deliverables:**
- [ ] Benchmark suite
- [ ] Frame differencing
- [ ] OCR auto-selection
- [ ] Confidence scoring & thresholds

**Success Criteria:**
- P95 latency < 500ms
- True positive > 95%, False positive < 5%
- Frame diff reduces OCR calls by 60%+

### M2: OBS Integration

**Deliverables:**
- [ ] OBS WebSocket client
- [ ] Scene switching on detection (to "BRB" / Privacy screen)
- [ ] Auto-return after delay
- [ ] **Stream Delay setup instructions** (CRITICAL)
- [ ] Config file (YAML)

**Success Criteria:**
- OBS scene switches when sensitive data appears
- Stream Delay filter configured (2-5 seconds)
- Sensitive data never reaches viewers (verified via test stream)
- Configurable without code changes

## Important Constraints & Decisions

### Architecture Decisions

**Sidecar Application (not OBS Plugin):**
- ✅ Simpler development (Python vs C++)
- ✅ Cross-platform support
- ✅ Easy community contributions
- ✅ Can package as standalone executable
- ❌ Cannot draw positioned blur overlays (switches to full-screen "BRB" scene instead)
- **Decision:** This is the production architecture, not a temporary prototype

**Stream Delay is The Latency Solution:**
- Python OCR cannot react faster than OBS encodes frames (impossible goal)
- **Solution:** User adds OBS "Render Delay" filter (2-5 seconds) to stream output
- Streamer sees real-time, viewers see delayed, script has time to react
- **This is REQUIRED for ScreenCloak to work** - must be in setup instructions

**Python Performance:**
- OCR latency 200-500ms is acceptable with Stream Delay buffer
- Frame sampling every 5-10 frames acceptable (sensitive text is typically static)
- Frame diffing reduces OCR calls by 60%+ on static screens

**False Positive Management:**
- Confidence thresholds: >0.9 = blur, 0.6-0.9 = warn (log only), <0.6 = ignore
- User can whitelist specific regions or strings via config
- Beta testing (M3) will tune thresholds based on real-world data
- Fuzzy digit cleanup (`l`→`1`, `I`→`1`, `O`→`0`) reduces OCR noise false negatives

### Monetization Model

**Pricing:** TBD (may be free/open-source, or paid - decision deferred post-launch)

**Potential Free Tier:**
- Seed phrase detection
- Credit card detection
- 3 user-defined personal strings

**Potential Paid Tier (if monetized):**
- Unlimited personal strings
- API key detection
- Advanced detection categories
- Detection log viewer with "close call" review
- Priority support

**License Approach:**
- Core detection engine: Open source (MIT license)
- Paid tier features (if implemented): Proprietary license key validation

## Common Patterns & Anti-Patterns

### ✅ DO

- **Read files before editing** - Always use Read tool first
- **Type everything** - Full type annotations on all functions
- **Test detection logic with mocks** - Don't require OBS to test core detectors
- **Use frame diffing** - Skip OCR on unchanged regions
- **Fail gracefully** - Log errors, continue running
- **Sanitize logs** - Never write actual sensitive data to disk
- **Validate config** - Check user input before using in regex

### ❌ DON'T

- **Don't use Bash for file operations** - Use Read/Write/Edit tools
- **Don't hardcode paths** - Use config.yaml for all user settings
- **Don't skip Luhn validation** - Credit card false positives are costly
- **Don't forget OCR noise cleanup** - Use fuzzy digit mapping before regex (l→1, I→1, O→0)
- **Don't skip dark mode preprocessing** - Invert dark images before Tesseract OCR
- **Don't require exact matches for personal strings** - Use fuzzy matching (typos happen)
- **Don't block main loop** - Run OBS callbacks in threads if needed
- **Don't log actual secrets** - Even in debug mode, use [REDACTED]

## Dependencies

**Core:**
- `mss` - Screen capture (cross-platform)
- `opencv-python` - Frame diffing, image processing
- `obs-websocket-py` - OBS integration
- `pyyaml` - Config file parsing
- `rapidfuzz` - Fuzzy string matching

**OCR Engines:**
- `paddleocr` + `paddlepaddle` - Primary OCR (GPU-accelerated)
- `pytesseract` + `tesseract` - Fallback OCR (CPU-friendly)
- `easyocr` - Alternative OCR (optional)

**Development:**
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `mypy` - Type checking

## External Resources

**BIP-39 Wordlist:**
https://github.com/bitcoin/bips/blob/master/bip-0039/english.txt

**OBS WebSocket Protocol:**
https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md

**PaddleOCR:**
https://github.com/PaddlePaddle/PaddleOCR

**Crypto Address Validation:**
- Bitcoin: https://en.bitcoin.it/wiki/Address
- Ethereum: https://eips.ethereum.org/EIPS/eip-55

## Future Work (Optional - Post-Launch)

### Native C++ OBS Plugin (Former M4)

**Status:** DEFERRED - Not required for launch

The original plan included M4 (Native OBS Plugin) to enable surgical blur overlays at specific bounding boxes. **This is no longer a required milestone.**

**Why it's optional:**
- Python sidecar app is production-quality with Stream Delay solution
- C++ rewrite is 6+ months of work for solo developer
- Scene switching to "BRB" screen is simpler and more reliable than blur overlays
- Stream Delay makes latency improvements unnecessary

**When to consider it:**
- After 1,000+ users validate the Python version
- If users strongly request positioned blur overlays
- If community contributors want to help with C++ development

**What it would provide:**
- ✅ Positioned blur overlays at bounding box coordinates (instead of full-screen scene switch)
- ✅ Slightly lower latency (~100ms vs 200-500ms) - though unnecessary with Stream Delay
- ❌ Much higher development complexity
- ❌ Harder to maintain and update
- ❌ Platform-specific builds (Windows, Mac, Linux)

**Decision:** Focus on making the Python sidecar app the best it can be. Native plugin is optional future enhancement if demand warrants it.

## Next Actions

**Next session: Task 13** — OCR Engine Interface
- `core/ocr_engine.py`: abstract `OCREngine`, `TesseractEngine`, `OCREngineFactory`
- Then Task 14: `PaddleOCREngine` with MPS, dark mode preprocessing
- Then Task 15: Screen capture (`core/capture.py`)
- Then Task 16: Frame differencing (`core/frame_diff.py`)
- Then Task 17: OBS WebSocket client (`core/obs_client.py`)
- Then Task 18: `main.py` entry point

**Running scripts:**
All scripts run from `/Users/tomershahar/SafeSense/screencloak/` as modules:
```bash
cd /Users/tomershahar/SafeSense/screencloak
python3 -m core.detector       # runs detector.py standalone test
python3 -m detectors.seed_phrase
```

## Questions & Clarifications

If you encounter ambiguity while implementing, refer to:
1. `implementation_plan_v2.md` for detailed technical specs
2. `PRD.md` for product context and user needs
3. This CLAUDE.md for project-wide patterns and decisions

**When in doubt:**
- Prioritize detection accuracy over performance (we can optimize later)
- Prioritize privacy (never log actual secrets)
- Prioritize user control (make it configurable via config.yaml)

---

**Last Updated:** 2026-02-12
**Current Milestone:** M1 - Core Detection Engine
**Target Completion:** Week 2
