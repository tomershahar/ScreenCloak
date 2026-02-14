# M3 Packaging Design — SafeStream macOS App

**Date:** 2026-02-14
**Status:** Approved

---

## Goal

Package SafeStream as a signed, notarized macOS `.app` distributed via a `.dmg` installer. Target: other streamers on Mac who install by dragging to Applications.

---

## What Gets Built

```
SafeStream.app/
├── Python interpreter + all libs (OpenCV, mss, rapidfuzz, obs-websocket-py, etc.)
├── Tesseract binary + tessdata/  (bundled from Homebrew)
└── data/
    ├── bip39_wordlist.txt
    └── api_patterns.json

SafeStream-1.0.0.dmg
├── SafeStream.app
└── Applications/  (symlink — drag-to-install UX)
```

**Config and logs live outside the read-only .app bundle:**
- Config: `~/Library/Application Support/SafeStream/config.yaml`
  - Copied from the bundled template on first launch if missing
- Logs: `~/Library/Logs/SafeStream/`

---

## Build Pipeline

`scripts/build_mac.sh` runs the full pipeline in one command:

```
1. pip install pyinstaller create-dmg
2. pyinstaller SafeStream.spec       →  dist/SafeStream.app
3. codesign --deep --sign "..."      →  sign .app with Developer ID
4. create-dmg SafeStream.app         →  dist/SafeStream-1.0.0.dmg
5. xcrun notarytool submit           →  send to Apple (~2 min review)
6. xcrun stapler staple              →  embed notarization ticket in .dmg
```

---

## Key Files

| File | Purpose |
|------|---------|
| `SafeStream.spec` | PyInstaller config — hidden imports, binaries, datas |
| `scripts/build_mac.sh` | One-command build + sign + notarize |
| `core/bundle_paths.py` | Runtime path resolution (bundle vs dev mode) |

---

## Tesseract Bundling

Tesseract is a C binary — PyInstaller won't auto-detect it. Bundle explicitly:

```python
# SafeStream.spec
binaries=[('/opt/homebrew/bin/tesseract', 'bin')],
datas=[
    ('/opt/homebrew/share/tessdata', 'tessdata'),
    ('data/', 'data'),
    ('config.yaml', '.'),   # bundled as template
],
```

At startup, detect bundle mode and redirect pytesseract:

```python
# core/bundle_paths.py
if getattr(sys, 'frozen', False):
    base = Path(sys._MEIPASS)
    os.environ['TESSDATA_PREFIX'] = str(base / 'tessdata')
    pytesseract.pytesseract.tesseract_cmd = str(base / 'bin' / 'tesseract')
    config_dir = Path.home() / 'Library' / 'Application Support' / 'SafeStream'
    log_dir = Path.home() / 'Library' / 'Logs' / 'SafeStream'
else:
    config_dir = Path(__file__).parent.parent   # dev mode: repo root
    log_dir = config_dir / 'logs'
```

`main.py` calls `bundle_paths.setup()` before any other imports.

---

## Code Signing & Notarization

```bash
# Sign
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: <NAME> (<TEAM_ID>)" \
  --options runtime \
  dist/SafeStream.app

# Notarize
xcrun notarytool submit dist/SafeStream-1.0.0.dmg \
  --apple-id "<APPLE_ID>" \
  --team-id "<TEAM_ID>" \
  --password "<APP_SPECIFIC_PASSWORD>" \
  --wait

# Staple
xcrun stapler staple dist/SafeStream-1.0.0.dmg
```

Credentials passed via environment variables — never hardcoded.

---

## Runtime Behaviour Changes

- First launch: copy `config.yaml` template to `~/Library/Application Support/SafeStream/`
- Print friendly message if Tesseract binary not found inside bundle
- Log to `~/Library/Logs/SafeStream/safestream.log` instead of `logs/`

---

## Out of Scope

- Windows / Linux packaging (future milestone)
- Auto-updater (Sparkle framework — future)
- App Store distribution (requires sandboxing, incompatible with screen capture)
- System tray icon (future UX improvement)
