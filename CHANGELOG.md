# Changelog

All notable changes to ScreenCloak are documented here.

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
