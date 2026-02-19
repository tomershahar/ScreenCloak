# OBS Setup Guide for ScreenCloak

Complete setup guide: WebSocket connection, Privacy Mode scene, and Stream Delay.

---

## Prerequisites

- OBS Studio 28 or later
- ScreenCloak installed and `config.yaml` configured

---

## Step 1: Enable OBS WebSocket

ScreenCloak connects to OBS over WebSocket to trigger scene switches.

1. Open OBS Studio
2. Go to **Tools → WebSocket Server Settings**
3. Check **Enable WebSocket server**
4. Set **Server Port** to `4455`
5. Set a **Server Password** (recommended — leave blank to skip auth)
6. Click **Apply → OK**

**Verify:** A WebSocket icon should appear in the OBS status bar.

---

## Step 2: Create a "Privacy Mode" Scene

This is the scene OBS switches to when ScreenCloak detects sensitive data.

1. In the **Scenes** panel (bottom-left), click **+**
2. Name it exactly: `Privacy Mode`
   > The name is case-sensitive and must match `privacy_scene` in `config.yaml`
3. Add sources to the scene:
   - **Color Source** — set to black (background)
   - **Text (GDI+)** — add text like: `"Sensitive information detected — resuming shortly"`
   - Optionally: your logo, a "BRB" graphic, or background music

---

## Step 3: Add Stream Delay (REQUIRED)

**Without this step, ScreenCloak cannot prevent sensitive data from reaching viewers.**

ScreenCloak's detection takes ~100–400ms. Without a delay, OBS encodes and uploads frames ~50ms after they appear — faster than ScreenCloak can react.

### Enable Stream Delay

1. Go to **Settings → Advanced**
2. Find the **Stream Delay** section
3. Check **Enable**
4. Set to `5000` ms (5 seconds)
5. Click **Apply → OK**

### Why 5 seconds?

```
T+0ms     Secret appears on your screen
T+0ms     ScreenCloak starts OCR (you see it live, no delay for you)
T+400ms   ScreenCloak detects it → switches OBS to Privacy Mode
T+5000ms  Twitch/YouTube broadcasts the frame from T+0 (the secret)
T+5400ms  Twitch/YouTube broadcasts the Privacy Mode frame (safe)
```

Your viewers see ~400ms of the sensitive content before Privacy Mode appears.
That is the V1 exposure window — significantly harder to capture than a full 10-second exposure.

For a deeper explanation — including Option B (Render Delay Filter) and a delay length comparison table — see [Stream Delay Deep Dive](./OBS_STREAM_DELAY_SETUP.md).

---

## Step 4: Configure ScreenCloak for OBS

Edit `config.yaml`:

```yaml
obs:
  host: localhost
  port: 4455
  password: "your-obs-password"   # Must match Step 1; leave "" if no password
  privacy_scene: "Privacy Mode"   # Must match Step 2 exactly (case-sensitive)
  auto_return: true               # Return to previous scene automatically
  return_delay: 3                 # Seconds to hold Privacy Mode before returning
```

---

## Step 5: Test the Connection

Run ScreenCloak in mock mode — no real screen capture, just a test image:

```bash
cd screencloak
python3 main.py --mock data/test_images/seed_phrase_12word.png
```

**Expected result:**
1. Terminal prints: `OBS connected — localhost:4455`
2. OBS switches to **Privacy Mode**
3. After `return_delay` seconds, OBS returns to your previous scene

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on port 4455 | WebSocket server not enabled | Enable in Tools → WebSocket Server Settings |
| `Authentication failed` | Password mismatch | Check `password` in config.yaml matches OBS |
| Scene doesn't switch | Wrong scene name | Verify `privacy_scene` matches OBS scene name exactly |
| Viewers still see the secret | Stream delay not set | Complete Step 3 |
| OBS switches too late | Stream delay too short | Increase to 5000ms |
| Detection runs but no OBS switch | OBS not running / not connected | Start OBS before running `main.py` |

---

## Quick-Reference Checklist

- [ ] WebSocket enabled on port 4455
- [ ] Password set in OBS and in `config.yaml`
- [ ] `Privacy Mode` scene created with background + text
- [ ] Stream Delay set to 5000ms in Settings → Advanced
- [ ] `config.yaml` updated with correct scene name and password
- [ ] Mock mode test passed (OBS switches on detection)
