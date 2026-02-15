#!/bin/bash
# scripts/build_mac.sh — Build, sign, package, and notarize ScreenCloak.app
#
# Required environment variables for signing/notarizing:
#   APPLE_ID           — your Apple ID email
#   APPLE_TEAM_ID      — 10-char team ID from developer.apple.com
#   APPLE_APP_PASSWORD — app-specific password from appleid.apple.com
#   SIGN_IDENTITY      — "Developer ID Application: Name (TEAMID)"
#
# Usage:
#   ./scripts/build_mac.sh             # build + sign + dmg + notarize
#   SKIP_NOTARIZE=1 ./scripts/build_mac.sh  # skip notarization (faster dev builds)

set -euo pipefail

VERSION="1.0.0"
APP_NAME="ScreenCloak"
DMG_NAME="${APP_NAME}-${VERSION}.dmg"

echo "=== ScreenCloak macOS Build ==="
echo "Version: $VERSION"
echo ""

# ── 1. Clean previous build ──────────────────────────────────────────────────
echo "→ Cleaning previous build..."
rm -rf build/ dist/

# ── 2. PyInstaller ───────────────────────────────────────────────────────────
echo "→ Running PyInstaller..."
# Resolve pyinstaller: prefer system PATH, fall back to python3 -m PyInstaller
if command -v pyinstaller &>/dev/null; then
    PYINSTALLER_CMD="pyinstaller"
else
    PYINSTALLER_CMD="python3 -m PyInstaller"
fi
$PYINSTALLER_CMD ScreenCloak.spec --clean --noconfirm
echo "✓ dist/${APP_NAME}.app created"

# ── 3. Code signing ──────────────────────────────────────────────────────────
if [ -n "${SIGN_IDENTITY:-}" ]; then
    echo "→ Signing .app..."
    codesign \
        --deep --force --verify --verbose \
        --sign "$SIGN_IDENTITY" \
        --options runtime \
        --entitlements scripts/entitlements.plist \
        "dist/${APP_NAME}.app"
    echo "✓ Signed"
else
    echo "⚠ SIGN_IDENTITY not set — skipping code signing"
fi

# ── 4. Create DMG ────────────────────────────────────────────────────────────
echo "→ Creating DMG..."
create-dmg \
    --volname "$APP_NAME" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "${APP_NAME}.app" 175 190 \
    --hide-extension "${APP_NAME}.app" \
    --app-drop-link 425 190 \
    "dist/${DMG_NAME}" \
    "dist/${APP_NAME}.app"
echo "✓ dist/${DMG_NAME} created"

# ── 5. Notarize ───────────────────────────────────────────────────────────────
if [ "${SKIP_NOTARIZE:-0}" = "1" ]; then
    echo "→ Skipping notarization (SKIP_NOTARIZE=1)"
elif [ -n "${SIGN_IDENTITY:-}" ]; then
    echo "→ Submitting for notarization (this takes ~2 minutes)..."
    xcrun notarytool submit "dist/${DMG_NAME}" \
        --apple-id "$APPLE_ID" \
        --team-id "$APPLE_TEAM_ID" \
        --password "$APPLE_APP_PASSWORD" \
        --wait
    echo "✓ Notarized"

    echo "→ Stapling notarization ticket..."
    xcrun stapler staple "dist/${DMG_NAME}"
    echo "✓ Stapled"
else
    echo "⚠ Skipping notarization — SIGN_IDENTITY not set"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Build complete ==="
echo "  App:  dist/${APP_NAME}.app"
echo "  DMG:  dist/${DMG_NAME}"
