# OBS Stream Delay Setup Guide

**This step is required for ScreenCloak to protect you.**

Without a stream delay, frames leave your computer ~50ms after they appear on screen. ScreenCloak's OCR detection takes ~200–500ms. By the time ScreenCloak reacts, the sensitive frame has already been uploaded to Twitch.

---

## Step 1: Enable OBS WebSocket Server

1. Open OBS Studio
2. Go to **Tools → WebSocket Server Settings**
3. Check **Enable WebSocket server**
4. Set **Server Port** to `4455`
5. Optionally set a **Server Password** (recommended)
6. Click **Apply** then **OK**

---

## Step 2: Create a "Privacy Mode" Scene

1. In the **Scenes** panel, click **+**
2. Name it exactly: `Privacy Mode`
3. Add a **Color Source** (black background)
4. Add a **Text (GDI+)** source with the message:
   ```
   Sensitive information detected — resuming shortly
   ```
5. Optionally add your logo or a countdown timer

> The scene name must match `privacy_scene` in `config.yaml` exactly (case-sensitive).

---

## Step 3: Add Stream Delay ⚠️ CRITICAL

This is the most important step. Without it, ScreenCloak **cannot** prevent sensitive data from reaching viewers.

### Option A: OBS Stream Delay (Recommended)

1. Go to **Settings → Advanced**
2. Find the **Stream Delay** section
3. Check **Enable**
4. Set duration to **5000 ms** (5 seconds)
5. Click **Apply** then **OK**

### Option B: Render Delay Filter (Alternative)

1. In the **Sources** panel, right-click your main video capture source
2. Select **Filters**
3. Click **+** → **Render Delay**
4. Set delay to **5000 ms**
5. Click **Close**

> Use one method, not both. Option A applies to the entire stream output. Option B applies to a single source.

---

## Why 5 Seconds?

```
T + 0ms:    Secret appears on your screen
T + 0ms:    ScreenCloak starts OCR (you see it in real-time)
T + 400ms:  ScreenCloak detects it → switches OBS to Privacy Mode
T + 5000ms: Twitch starts broadcasting the frame from T=0 (secret)
T + 5400ms: Twitch starts broadcasting the Privacy Mode frame (safe)
```

Your viewers see ~400ms of the sensitive content as a brief flash before the Privacy Mode screen appears. This is the V1 exposure window.

**With a 2-second delay:** exposure window is still ~400ms, but you have less buffer if ScreenCloak is slow on a given frame.

**With a 5-second delay:** you have a 4.6-second safety margin beyond the typical detection time.

---

## Step 4: Configure ScreenCloak

Edit `config.yaml`:

```yaml
obs:
  host: localhost
  port: 4455
  password: "your-obs-password"   # Leave empty string if no password set
  privacy_scene: "Privacy Mode"   # Must match OBS scene name exactly
  auto_return: true               # Return to previous scene automatically
  return_delay: 3                 # Seconds to stay on Privacy Mode before returning
```

---

## Step 5: Verify the Setup

Run ScreenCloak in mock mode to test the OBS connection:

```bash
cd screencloak
python main.py --mock data/test_images/seed_phrase_12word.png
```

You should see:
1. ScreenCloak logs: `OBS connected — localhost:4455`
2. OBS switches to your **Privacy Mode** scene
3. After `return_delay` seconds, OBS returns to your previous scene

If OBS doesn't switch scenes, check:
- OBS WebSocket is enabled and on port 4455
- `config.yaml` password matches the OBS WebSocket password
- The scene name in `config.yaml` matches exactly (case-sensitive)

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `Connection refused` on port 4455 | WebSocket server not enabled | Enable in Tools → WebSocket Server Settings |
| Scene doesn't switch | Wrong scene name | Check `privacy_scene` in config matches OBS exactly |
| Authentication error | Wrong password | Update `password` in config.yaml |
| Viewers still see secret | No stream delay set | Complete Step 3 above |
| OBS switches but too late | Stream delay too short | Increase to 5000ms |

---

## V2 Roadmap: Zero-Leak Protection

V1 uses a Python sidecar + stream delay. The exposure window is the OCR detection latency (~200–400ms).

V2 will be a **native C++ OBS plugin** that intercepts frames directly inside the encoding pipeline — before they are ever queued for upload. This eliminates the exposure window entirely, with no stream delay required.

> V2 development begins after V1 is validated with real users.
