# ScreenCloak

**Real-time sensitive data detection for live streamers.**

ScreenCloak watches your screen while you stream and automatically switches OBS to a "Be Right Back" scene when it detects sensitive information — seed phrases, credit card numbers, crypto wallet addresses, API keys, or your personal info.

---

## What It Detects

| Category | Examples | Tier |
|---|---|---|
| Crypto seed phrases | 12/24-word BIP-39 phrases | Free |
| Credit card numbers | Visa, Mastercard, Amex (with Luhn validation) | Free |
| Crypto wallet addresses | BTC (legacy + bech32), ETH, SOL | Free |
| API keys | AWS, GitHub, Stripe, OpenAI, Anthropic, Google, Slack, and 8 more (14 services total) | Free |
| Personal strings | Your name, email, phone, address | Free (3 max) |

---

## How It Works

ScreenCloak runs alongside OBS as a "sidecar" application. It captures your screen in real-time, runs OCR to read on-screen text, and triggers an OBS scene switch the moment it detects something sensitive.

```
Your screen (raw) ──→ ScreenCloak OCR → Detection → OBS WebSocket
                                                          ↓
Your viewers ←── Twitch/YouTube ←── OBS Output (delayed)
```

---

## ⚠️ Important: How Protection Actually Works (Please Read)

**ScreenCloak provides strong protection, but not instant protection. Here is exactly what happens:**

### The Stream Delay Requirement

For ScreenCloak to protect you, you **must** add an OBS **Render Delay** filter (minimum 2 seconds, recommended 5 seconds) to your stream output. Without it, ScreenCloak **cannot protect you**.

Here's why:

Without a stream delay, frames leave your computer ~50ms after they appear on screen. ScreenCloak's OCR detection takes ~200–500ms. By the time ScreenCloak reacts, the sensitive frame has already been uploaded to Twitch.

**With a 5-second stream delay configured:**

```
T + 0ms:    Secret appears on your screen
T + 0ms:    ScreenCloak starts OCR processing (you see it in real-time)
T + 400ms:  ScreenCloak detects it → OBS switches to BRB scene
T + 5000ms: Twitch starts broadcasting the frame from T+0ms (secret)
T + 5400ms: Twitch starts broadcasting the BRB frame (safe)
```

**Result:** Your viewers see approximately 400ms of the sensitive content (the detection window), then the BRB screen. That is the honest reality of V1.

### V1 Security Model — What This Means

| Scenario | Without ScreenCloak | With ScreenCloak V1 (+ Stream Delay) |
|---|---|---|
| Seed phrase visible for 10 seconds | Full 10s exposure to all viewers | ~400ms flash before BRB screen |
| Credit card on screen briefly | Full exposure | ~400ms flash |
| You catch it manually | Whatever time passes | Automatic |

**A 400ms flash is significantly harder to capture than a 10-second exposure.** It will not appear in stream highlights or be readable in most recordings at normal playback speed. But it is not zero. A viewer specifically recording or screenshot-spamming at that moment could theoretically capture it.

### What "Zero Leak" Looks Like (V2 Roadmap)

V1 uses a Python sidecar with OBS WebSocket. V2 will be a **native OBS plugin written in C++** that intercepts frames directly inside the encoding pipeline — before they are ever queued for upload. This approach will reduce the exposure window to **0ms**, regardless of OCR speed.

> **V2 native plugin is planned once V1 is validated with real users.** If V1 works well for you, that's your signal to watch for V2.

### Recommended Settings for Best Protection

| Setting | Value | Why |
|---|---|---|
| OBS Render Delay | 5000ms | Gives 5s window vs ~400ms detection |
| OCR frame rate | Every 3 frames | Faster detection, higher CPU |
| Stream Delay (OBS) | Match render delay | Keep them in sync |

> With GPU acceleration (Apple Silicon MPS or NVIDIA CUDA), detection typically runs in 100–200ms, reducing the exposure window further.

---

## Installation

### Requirements

- Python 3.11+
- OBS Studio 28+ with WebSocket plugin enabled
- macOS (Apple Silicon recommended) or Windows

### Step 1: Clone and Install

```bash
git clone https://github.com/tomershahar/ScreenCloak.git
cd ScreenCloak/screencloak
pip install -r requirements.txt
```

### Step 2: Install Tesseract (OCR engine)

**macOS:**
```bash
brew install tesseract
```

**Windows:** Download and run the UB-Mannheim installer (use the default install path):
`https://github.com/UB-Mannheim/tesseract/wiki`

> ScreenCloak expects Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`. If you installed it elsewhere, set `TESSDATA_PREFIX` and the path in `config.yaml`.

---

### Windows Packaged Installer (alternative to Python setup)

If you prefer not to install Python, download the pre-built Windows installer from the Releases page. It bundles everything except Tesseract (install that first from the link above).

To build the installer yourself on a Windows machine:

```bat
scripts\build_windows.bat
```

Requires:
- Python 3.11+ with `pip install -r requirements.txt`
- PyInstaller (`pip install pyinstaller`)
- Inno Setup 6 (`https://jrsoftware.org/isdl.php`)

Output: `dist\ScreenCloak-1.0.0-Setup.exe`

> **First run on Windows:** Windows SmartScreen may warn "unrecognized app" because the installer is unsigned. Click **More info → Run anyway** to proceed.

### Step 3: Configure OBS

> See the full [OBS Setup Guide](docs/OBS_SETUP.md) for screenshots and troubleshooting.

**3a. Enable OBS WebSocket:**
1. In OBS, go to `Tools → WebSocket Server Settings`
2. Enable WebSocket server
3. Set port to `4455`
4. Set a password (optional but recommended)
5. Click Apply

**3b. Create a "Privacy Mode" scene:**
1. In OBS, click `+` in the Scenes panel
2. Name it exactly: `Privacy Mode`
3. Add a background (black image or color source)
4. Add a text source: `"Sensitive information detected — BRB in a moment"`

**3c. ⚠️ Add Stream Delay (REQUIRED for protection):**
1. In OBS, go to `Settings → Advanced`
2. Find `Stream Delay`
3. Enable it and set to `5000ms` (5 seconds)
4. Click OK

> Without this step, ScreenCloak will still detect and log sensitive data, but **cannot prevent it from reaching viewers**.

### Step 4: Configure ScreenCloak

Edit `config.yaml`:

```yaml
# Personal strings to protect (your actual name, email, etc.)
detection:
  personal_strings:
    strings:
      - "Your Real Name"
      - "yourname@email.com"
      - "555-123-4567"

# OBS WebSocket settings
obs:
  password: "your-obs-password"  # Leave empty if none
  privacy_scene: "Privacy Mode"
```

### Step 5: Run

```bash
python main.py
```

A menu bar icon (macOS) or system tray icon (Windows) will appear. See [System Tray](#system-tray) below.

To test with a sample image (no OBS needed):
```bash
python main.py --mock data/test_images/seed_phrase_12word.png
```

To run without the tray icon (headless / server mode):
```bash
python main.py --no-tray
```

---

## System Tray

When ScreenCloak is running, a small circle icon appears in your menu bar (macOS) or system tray (Windows).

> **macOS tip:** If you don't see the icon, it may be hidden behind other menu bar icons. Hold **⌘ Command** and drag other icons to the left to make room, or use an app like [Bartender](https://www.macbartender.com/) to manage overflow icons.



| Icon colour | Meaning | OBS action |
|---|---|---|
| Grey | Starting up / scanning | None |
| Green | Screen is clean | None |
| **Orange** | Low-confidence detection (logged only) | **No scene switch** |
| **Red** | High-confidence detection | **Switches to Privacy Mode** |
| **Yellow** | OBS WebSocket disconnected | **None — reconnecting automatically** |

**Orange vs Red — what's the difference?**

- **Orange** means ScreenCloak saw something suspicious but isn't confident enough to interrupt your stream. The detection is logged for review but OBS keeps broadcasting normally. This can happen with partial sequences, ambiguous text, or UI elements that resemble sensitive data (e.g. OBS settings windows).
- **Red** means ScreenCloak is confident. OBS switches to your Privacy Mode scene immediately, protecting your viewers.

The confidence threshold for each colour is configurable — see [Tuning Sensitivity](#tuning-sensitivity) below.

**Right-click (or click) the icon to open the menu:**

```
ScreenCloak
──────────
⏸  Pause         ← click to pause/resume scanning
Detections: 0    ← running count of detections this session
──────────
Quit
```

Use **Pause** when you intentionally display sensitive content (e.g. entering a seed phrase you're testing). ScreenCloak will skip OCR while paused.

---

## Benchmarking Your Setup

Before you go live, run the detection benchmark to verify accuracy and latency on your machine:

```bash
python3 tests/benchmark.py
```

Example output:
```
ScreenCloak Detection Benchmark
========================================================================
  PASS  seed_phrase_12word.png          detected=seed_phrase                e2e=81ms
  PASS  seed_phrase_24word.png          detected=seed_phrase, seed_phrase   e2e=112ms
  PASS  credit_card_visa.png            detected=credit_card                e2e=71ms
  PASS  eth_address.png                 detected=crypto_address             e2e=68ms
  PASS  mixed_content.png               detected=crypto_address, credit_card  e2e=105ms
  PASS  false_positive_essay.png        detected=(none)                     e2e=124ms

========================================================================
SUMMARY
========================================================================
  True Positive Rate:   5/5  (100.0%)
  False Positive Rate:  0/1  (0.0%)

  OCR latency      P50: 82ms   P95: 124ms
  E2E latency      P50: 84ms   P95: 125ms

========================================================================
VERDICT (M1 acceptance criteria)
========================================================================
  ✅  True positive rate ≥ 95%             100.0%
  ✅  False positive rate ≤ 5%             0.0%
  ✅  E2E P95 latency ≤ 500ms              125ms

  ✅  OVERALL PASS — ScreenCloak meets M1 acceptance criteria
```

To benchmark raw OCR engine performance (latency only, no detection pipeline):
```bash
python3 benchmark_ocr.py
```

**The lower your P95 latency, the smaller your exposure window.** Apple Silicon with MPS acceleration typically achieves 100–200ms.

---

## Tuning Sensitivity

Edit `config.yaml` to control when ScreenCloak acts:

```yaml
detection:
  thresholds:
    blur: 0.8   # >= this confidence → switch OBS to Privacy Mode (red icon)
    warn: 0.6   # >= this confidence → log only, no scene switch (orange icon)
```

**Confidence levels by detector:**

| What's on screen | Confidence | Default action |
|---|---|---|
| 12/24-word BIP-39 seed phrase | 0.95 | 🔴 Blur (OBS switches) |
| Credit card + expiry date | 0.90 | 🔴 Blur (OBS switches) |
| Credit card alone (no expiry) | 0.80 | 🔴 Blur (OBS switches) |
| ETH/BTC/SOL wallet address | 0.85–0.95 | 🔴 Blur (OBS switches) |
| Partial seed phrase (< 12 words) | 0.70 | 🟠 Warn (logged only) |
| Personal string match | 0.70–0.90 | 🟠/🔴 Depends on confidence |

**To reduce false positives** (if OBS switches too often): raise `blur` to `0.85` or `0.90`.

**To catch more edge cases** (if something slips through): lower `blur` to `0.75`.

---

## Privacy

ScreenCloak processes everything locally. Nothing is sent anywhere.

- **Detection logs** (`logs/detections.log`) store only detection type, confidence, and timestamp — never the actual detected text
- **No telemetry** by default — opt-in only via `config.yaml`
- **Personal strings** you configure in `config.yaml` are never transmitted

---

## Known Limitations (V1)

| Limitation | Details | Fixed in |
|---|---|---|
| **Exposure window** | ~200–400ms of secret visible before BRB (see above) | V2 (native plugin) |
| **Scene switch only** | Cannot blur a specific region — switches entire scene | V2 (native plugin) |
| **OCR accuracy** | May miss text in unusual fonts, animations, or extreme angles | Ongoing improvement |
| **Solana false positives** | Short base58 strings may match non-addresses | Being tuned |

---

## Roadmap

- **V1 (current):** Python sidecar + OBS WebSocket. Strong protection with Stream Delay. Exposure window = detection latency (~200–400ms).
- **V2:** Native C++ OBS plugin. Frames intercepted in the encoding pipeline. **True zero-leak.** No stream delay required.

---

## Community

**Discord:** [discord.gg/e46csQsuRc](https://discord.gg/e46csQsuRc) — questions, feedback, false positive reports, beta discussion.

---

## Contributing

Contributions welcome — especially:
- Additional detection patterns (more crypto networks, more API key formats)
- False positive reports with example screenshots
- Performance improvements to reduce detection latency

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## License

MIT — detection engine is open source.

---

## FAQ

**Q: Does this record my screen?**
No. ScreenCloak captures frames for OCR processing in memory only. Nothing is written to disk except sanitized detection logs.

**Q: What if it triggers a false positive on stream?**
ScreenCloak will switch to your Privacy Mode scene briefly. You can tune the sensitivity via `fuzzy_threshold` and `confidence` settings in `config.yaml`.

**Q: Why do I need a Stream Delay? My internet is fast.**
Internet speed is not the issue. OBS encodes and uploads video frames ~50ms after they appear on screen — faster than any OCR engine can process them. The Stream Delay creates a buffer between your screen and what viewers actually see, giving ScreenCloak time to react.

**Q: I set up Stream Delay — am I 100% safe now?**
You are significantly safer. V1 limits exposure to the OCR detection window (~200–400ms). For true zero-leak protection, V2 (native plugin) is required. We will announce it when it's ready.

**Q: Can I use this for Zoom or other screen sharing tools?**
Not natively in V1 — it controls OBS scenes. If you screen-share via OBS Virtual Camera, it will work. Native support for Zoom/Teams is on the roadmap.
