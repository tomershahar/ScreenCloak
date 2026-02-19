# Stream Delay — Deep Dive

> This is a supplementary guide. For the complete OBS setup walkthrough, see [OBS Setup Guide](./OBS_SETUP.md).

---

## Why Stream Delay Is Required

Without a stream delay, OBS encodes and uploads frames ~50ms after they appear on your screen. ScreenCloak's OCR detection takes ~200–500ms. By the time ScreenCloak reacts, the sensitive frame has already been uploaded to Twitch.

A stream delay creates a buffer between what appears on your screen and what your viewers see — giving ScreenCloak time to react before the frame goes out.

---

## Two Ways to Add Stream Delay

### Option A: OBS Stream Delay (Recommended)

Applies to the entire stream output.

1. Go to **Settings → Advanced**
2. Find the **Stream Delay** section
3. Check **Enable**
4. Set duration to **5000 ms** (5 seconds)
5. Click **Apply** then **OK**

### Option B: Render Delay Filter (Per-source alternative)

Applies only to a single source (e.g. your screen capture). Useful if you need different delays on different sources.

1. In the **Sources** panel, right-click your main video capture source
2. Select **Filters**
3. Click **+** → **Render Delay**
4. Set delay to **5000 ms**
5. Click **Close**

> Use one method, not both. If you apply both, delays stack.

---

## Why 5 Seconds?

```
T + 0ms:    Secret appears on your screen
T + 0ms:    ScreenCloak starts OCR (you see it in real-time, no delay for you)
T + 400ms:  ScreenCloak detects it → switches OBS to Privacy Mode
T + 5000ms: Twitch starts broadcasting the frame from T=0 (the secret)
T + 5400ms: Twitch starts broadcasting the Privacy Mode frame (safe)
```

Your viewers see ~400ms of the sensitive content as a brief flash. This is the V1 exposure window.

| Stream delay | Safety margin | Notes |
|---|---|---|
| 2 seconds | 1.6s margin | Minimum viable; risky if OCR is slow on a given frame |
| 5 seconds | 4.6s margin | Recommended — comfortable buffer |
| 10 seconds | 9.6s margin | Maximum protection; noticeable lag for viewers reacting to your stream |

With GPU acceleration (Apple Silicon MPS or NVIDIA CUDA), detection typically runs in 100–200ms, shrinking the exposure window further.

---

## V2 Roadmap: Zero-Leak Protection

V1 uses a Python sidecar + stream delay. The exposure window is the OCR detection latency (~200–400ms), which cannot be eliminated in this architecture.

V2 will be a **native C++ OBS plugin** that intercepts frames directly inside the encoding pipeline — before they are ever queued for upload. This eliminates the exposure window entirely, with no stream delay required.

> V2 development begins after V1 is validated with real users.
