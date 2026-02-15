# M3 Packaging Design — ScreenCloak macOS App

**Date:** 2026-02-14
**Status:** Approved

---

## Goal

Package ScreenCloak as a signed, notarized macOS `.app` distributed via a `.dmg` installer. Target: other streamers on Mac who install by dragging to Applications.

---

## What Gets Built

```
ScreenCloak.app/
├── Python interpreter + all libs (OpenCV, mss, rapidfuzz, obs-websocket-py, etc.)
├── Tesseract binary + tessdata/  (bundled from Homebrew)
└── data/
    ├── bip39_wordlist.txt
    └── api_patterns.json

ScreenCloak-1.0.0.dmg
├── ScreenCloak.app
└── Applications/  (symlink — drag-to-install UX)
```

**Config and logs live outside the read-only .app bundle:**
- Config: `~/Library/Application Support/ScreenCloak/config.yaml`
  - Copied from the bundled template on first launch if missing
- Logs: `~/Library/Logs/ScreenCloak/`

---

## Build Pipeline

`scripts/build_mac.sh` runs the full pipeline in one command:

```
1. pip install pyinstaller create-dmg
2. pyinstaller ScreenCloak.spec       →  dist/ScreenCloak.app
3. codesign --deep --sign "..."      →  sign .app with Developer ID
4. create-dmg ScreenCloak.app         →  dist/ScreenCloak-1.0.0.dmg
5. xcrun notarytool submit           →  send to Apple (~2 min review)
6. xcrun stapler staple              →  embed notarization ticket in .dmg
```

---

## Key Files

| File | Purpose |
|------|---------|
| `ScreenCloak.spec` | PyInstaller config — hidden imports, binaries, datas |
| `scripts/build_mac.sh` | One-command build + sign + notarize |
| `core/bundle_paths.py` | Runtime path resolution (bundle vs dev mode) |

---

## Tesseract Bundling

Tesseract is a C binary — PyInstaller won't auto-detect it. Bundle explicitly:

```python
# ScreenCloak.spec
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
    config_dir = Path.home() / 'Library' / 'Application Support' / 'ScreenCloak'
    log_dir = Path.home() / 'Library' / 'Logs' / 'ScreenCloak'
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
  dist/ScreenCloak.app

# Notarize
xcrun notarytool submit dist/ScreenCloak-1.0.0.dmg \
  --apple-id "<APPLE_ID>" \
  --team-id "<TEAM_ID>" \
  --password "<APP_SPECIFIC_PASSWORD>" \
  --wait

# Staple
xcrun stapler staple dist/ScreenCloak-1.0.0.dmg
```

Credentials passed via environment variables — never hardcoded.

---

## Runtime Behaviour Changes

- First launch: copy `config.yaml` template to `~/Library/Application Support/ScreenCloak/`
- Print friendly message if Tesseract binary not found inside bundle
- Log to `~/Library/Logs/ScreenCloak/screencloak.log` instead of `logs/`

---

## Out of Scope

- Windows / Linux packaging (future milestone)
- Auto-updater (Sparkle framework — future)
- App Store distribution (requires sandboxing, incompatible with screen capture)
- System tray icon (future UX improvement)
