# Changelog

All notable changes to ScreenCloak are documented here.

---

## v0.1.1 — 2026-02-15

### Fixed
- **Credit card false positives from log timestamps** — digit-stream sliding window now applied per OCR token only, never across combined screen text. Timestamps (`2026-02-15 18:28:29`) no longer produce Luhn-valid 16-digit windows.
- **Standalone CVV detection removed** — port numbers and UI numeric values no longer push credit card confidence to blur level. CVV is now only detected with an explicit `CVV`/`CVC`/`CV2` label.
- **OBS auto-return loop** — ScreenCloak no longer saves "Privacy Mode" as the return target if OBS was already on that scene at detection time.

### Improved
- **Tray icon: 4 distinct states** — grey (idle), green (clean), **orange** (warn — logged only, stream unchanged), **red** (blur — OBS switched). Previously warn and blur both showed red, making it impossible to tell if OBS had switched.
- **Configurable confidence thresholds** — `detection.thresholds.blur` (default 0.8) and `detection.thresholds.warn` (default 0.6) are now editable in `config.yaml`. Raise `blur` to reduce false positives; lower it to catch more edge cases.

---

## v0.1.0 — Beta Release (2026-02-14)

Initial public beta.

### Detection
- BIP-39 seed phrases (12 and 24 words, spatial clustering)
- Credit card numbers (Luhn validation, OCR noise cleanup, digit-stream fallback)
- Crypto wallet addresses (BTC legacy + bech32, ETH, SOL — with collision deduplication)
- API keys for 14 services: AWS, GitHub, Stripe, OpenAI, Anthropic, Google, Slack, Twilio, SendGrid, DigitalOcean, NPM, Discord, Cloudflare, Heroku
- Personal strings (fuzzy matching via rapidfuzz, up to 3 strings free)

### OBS Integration
- OBS WebSocket v5 (port 4455) for scene switching
- Automatic return to previous scene after configurable delay
- Stream Delay guidance (2–5 second Render Delay required for full protection)

### System Tray
- Menu bar icon on macOS, system tray on Windows
- Icon states: idle (grey), clean (green), alert (red)
- Pause/Resume scanning from tray menu
- Running detection counter
- `--no-tray` flag for headless/server mode

### Packaging
- macOS: PyInstaller `.app` bundle (Apple Silicon)
- Windows: PyInstaller `.exe` + Inno Setup installer
- Tesseract not bundled — uses system installation to avoid library conflicts

### Performance
- Tesseract OCR (CPU) — P95 latency ~118ms on Apple Silicon
- Frame differencing reduces OCR calls by ~60% on static screens
