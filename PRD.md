    

# ScreenCloak

**Description: What is it?**

ScreenCloak is an OBS Studio plugin that automatically detects and redacts sensitive information from a streamer's screen in real time — before it reaches viewers. It sits in the OBS video pipeline, runs OCR on frames, and applies a blur overlay when it identifies sensitive data like crypto seed phrases, wallet addresses, credit card numbers, API keys, or user-defined personal strings (name, address, phone number, email).

**Problem: What problem is this solving?**

Streamers regularly and accidentally expose sensitive personal and financial information during live streams. This includes crypto seed phrases and private keys, passwords and API tokens, personal notifications (DMs, emails with real names), credit card numbers, home addresses (via delivery apps, browser autofill), and browser history or tab titles revealing private information.

The consequences range from embarrassment to catastrophic financial loss — a recent high-profile case involved a crypto streamer losing $100K after their seed phrase was briefly visible on screen. The real-time nature of live streaming means once information is exposed, it cannot be taken back. Viewers can screenshot or record within seconds.

Current solutions are fragmented and inadequate. Discord's Streamer Mode only hides info within Discord. Entropy (Chrome extension) only works inside browsers. Winhide (OBS plugin) only blanks specific predefined application windows. No solution exists that works system-wide, across all apps, using intelligent content detection.

**Why: How do we know this is a real problem and worth solving?**

* Twitch alone has 7-8 million monthly active streamers. YouTube Live, Kick, and Facebook Gaming add millions more.
* "Streamer leak" and "streamer accidentally shows" compilations regularly get millions of views on YouTube, demonstrating how common accidental exposure is.
* The $100K seed phrase incident is just one of many documented financial losses from stream leaks.
* Every major platform (Discord, Twitch, OBS) has introduced some form of privacy features, signaling recognized demand — but none solve the core problem of intelligent, cross-app screen-level detection.
* Streamers already pay $5-20/month for tools like overlays, bots, alerts, and chat moderation. Privacy protection fits naturally into this spending pattern.
* The offensive (malware) side has already built this exact detection capability — the Rhadamanthys stealer uses AI-powered OCR to extract seed phrases from screen captures in real time. The defensive tool to match it simply doesn't exist yet.

**Success: How do we know if we've solved this problem?**

V1 success metrics (first 6 months post-launch):

* **Installs:** 5,000+ free installs of the OBS plugin
* **Detections:** Measurable count of successful redactions (leaks prevented), reported via opt-in anonymous telemetry
* **Word of mouth:** At least 3-5 unsolicited streamer testimonials, clips, or social media mentions of ScreenCloak catching a potential leak

Longer-term (12 months):

* 25,000+ installs
* Conversion rate to paid tier of 3-5%
* Recognition as the default privacy tool in the streaming ecosystem

**Audience: Who are we building for?**

Primary: Live streamers who share their screen during broadcasts — especially those who also handle financial information (crypto traders, developers showing code, productivity streamers). Platform-agnostic (Twitch, YouTube, Kick, etc.) since ScreenCloak operates at the OBS level.

Secondary: Anyone who screen-shares professionally — developers on Zoom/Teams calls, educators doing live demos, corporate presenters. This is the larger market but not the launch audience.

**What: Roughly, what does this look like in the product?**

ScreenCloak is an OBS Studio plugin. Once installed, it appears as a video filter that the streamer adds to their scene.

**V1 feature set:**

* Pattern-based detection for known sensitive data formats:
  * Crypto seed phrases (BIP-39 wordlist matching — triggers on 3+ sequential matches from the 2048-word list)
  * Crypto wallet addresses (BTC, ETH, SOL, and other major chains via regex patterns)
  * Credit card numbers (16-digit sequences passing Luhn algorithm validation)
  * API keys and tokens (pattern matching for common formats: AWS, Stripe, GitHub, etc.)
  * User-defined personal strings (streamer enters their real name, address, phone, email — tool watches for exact and fuzzy matches)
* Configurable blur/overlay style when a match is detected
* Detection log viewable after stream (what was caught, when)
* Settings panel within OBS for managing detection categories and personal strings

**Technical approach (V1):**

* OBS plugin (C/C++) hooks into the video pipeline as a filter, intercepting frames before encoding
* OCR engine (PaddleOCR) processes frames to extract on-screen text
* OpenCV frame diffing limits OCR to changed screen regions only, reducing compute
* Detection engine runs pattern matching and regex against OCR output
* When match found, blur overlay is applied at the OCR-provided bounding box coordinates
* Latency budget: 200-500ms — invisible to viewers since live streams already carry 2-15 seconds of platform delay
* Frame sampling: every 5-10 frames, with blur mask held between scans (sensitive text is typically static on screen)

**Monetization model:**

* Free tier: seed phrase detection, credit card detection, 3 user-defined personal strings
* Paid tier ($5-10/month or $50-80/year): unlimited personal strings, all detection categories, API key detection, custom blur styles, detection logs with "close call" review

**How: What is the experiment plan?**

Go straight to building a working beta. Validation approach:

1. **Build a functional standalone prototype first** — Python app that captures a screen region, runs OCR + detection, and triggers OBS scene switch via WebSocket API when it detects sensitive data. This proves the detection engine works without requiring C++ OBS plugin development.
2. **Test internally** with synthetic sensitive data on screen during mock streams.
3. **Recruit 10-20 beta testers** from streaming communities (Reddit r/Twitch, r/obs, crypto streaming Discords). Free access in exchange for feedback and permission to share testimonials.
4. **Rebuild as native OBS plugin** once detection accuracy is validated and beta feedback is incorporated.
5. **Public launch** on OBS plugin marketplace + dedicated landing page.

**When: When does it ship and what are the milestones?**

| Milestone             | Description                                                                                                                            |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| M1: Prototype         | Standalone Python app: screen capture → OCR → seed phrase + credit card detection. Prove the detection pipeline works.               |
| M2: OBS integration   | Connect prototype to OBS via WebSocket API. Trigger scene switch or overlay on detection. Testable in real stream setup.               |
| M3: Closed beta       | Recruit 10-20 streamers. Add user-defined strings. Collect feedback on false positives, performance, and missing detection categories. |
| M4: Native OBS plugin | Rebuild core as C/C++ OBS filter for better performance and simpler user setup. Add settings UI within OBS.                            |
| M5: Public launch     | OBS plugin marketplace listing. Landing page. Free tier live.                                                                          |
| M6: Paid tier         | Introduce premium features based on beta feedback. Payment integration.                                                                |

Timeline is TBD — team is currently 1 person (product and go-to-market) with potential engineering support from a family member with C++/systems skills. AI coding tools (Claude Code, etc.) will be used to accelerate development of the prototype and detection engine.
