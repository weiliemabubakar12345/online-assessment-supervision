# Exam Screen-Side Monitor — Prototype Extension (MV3)

Prototype proctoring extension for the **screen side**. It captures the same
signals as `exam-detector-prototype.html`, **plus** things page JavaScript
cannot see because of the browser sandbox:

- **Destination tab** when the student switches tabs — including its **URL &
  title** (e.g. knowing they opened ChatGPT/Google, not just "switched tabs").
- **In-tab navigation** (typing a new URL into the address bar).
- **Whole-browser focus** (switching to another application).
- **Idle / locked screen**.
- **Screenshots** of the active tab — on every tab switch, **and periodically**
  (every heartbeat, ~30–60s) so a side-by-side window the student never clicks
  into, or a cheat URL typed into the *current* tab's address bar, still gets
  captured at least once.
- **Periodic webcam frames** — sent to the computer-vision pipeline
  (`computer_vision/`, see [Computer Vision integration](#computer-vision-integration-review-score)
  below) for a cross-cue *review score* (head pose, gaze, objects). Capture
  rate adapts: ~200ms while the CV pipeline is still calibrating, dropping to
  ~1000ms once it reports `READY`.
- **Other installed & enabled extensions** (via `chrome.management` /
  `browser.management`) — e.g. knowing an AI-assistant/translator/quiz-answer
  extension is enabled, though not necessarily in use at that exact moment.
  See [Important limitations](#important-limitations-report-these-in-the-paper).

Not local-only — screenshots and webcam frames are forwarded by the server to
external analysis services (VLM / CV) for research/testing.

## Files

| File | Role |
|---|---|
| `manifest.json` | MV3 config + permissions |
| `content.js` | Runs on the exam page: captures in-page events + periodic webcam frames, sends them to the service worker |
| `background.js` | Service worker: central log + cross-tab signals + screenshots (tab-switch and periodic) + heartbeat + forwards everything to the server |
| `popup.html` / `popup.js` | Live per-student log panel + Export JSON / Clear buttons |
| `server.js` | Proctor server (plain Node.js, no dependencies) — receives events + heartbeats, relays screenshots/frames to the VLM/CV services, and records every run to disk |
| `dashboard.html` | Proctor dashboard — live view of every student and their log, online/offline status, VLM cheat verdicts |
| `vlm_service.py` | Local Flask service — single-model screenshot analysis (Qwen2.5-VL-3B) that outputs a `category` + `exam_relevance` + `is_cheating` verdict |
| `vlm_service_cascade.py` | Two-model cascade screenshot analysis service (SmolVLM-500M screens, Qwen2.5-VL decides on escalation) — same output shape as `vlm_service.py`, plus routing/latency fields. See [VLM cheating decision](#vlm-cheating-decision-is_cheating) below |
| `cv_service.py` | Headless Flask service wrapping the `computer_vision/` pipeline (head/gaze + YOLO + event scoring) as an HTTP API, designed to run on a Kaggle GPU. See [`kaggle/README.md`](kaggle/README.md) |
| `RUN-CHECKLIST.md` | Pre-flight / during / post-session checklist for a live data-collection episode (what is transient and must be saved, timing conventions, labeling discipline) |

`../cv_eval_harness.py` (repo root) evaluates `cv_service.py` standalone: it
streams labeled webcam clips through `/session/start` → `/frame` → `/session/end`,
thresholds the peak `review_score`, and reports precision/recall/F1 — the same
framework used for the VLM evaluation, so the two modalities are comparable.

The extension is cross-browser (uses the `browser`/`chrome` alias). The
manifest currently targets **Firefox/Zen** (`background.scripts`). For
Chrome/Edge, change the `background` block to
`"service_worker": "background.js"` (see below).

### Firefox / Zen Browser

1. Open `about:debugging#/runtime/this-firefox` (same in Zen).
2. Click **Load Temporary Add-on…**, select this folder's `manifest.json`.
3. Click the extension icon in the toolbar to open the **log popup**.
4. **Allow site access**: open `about:addons` → this extension → **Permissions**
   → enable *Access your data for all websites* (needed for `captureVisibleTab`).
5. The browser will prompt for **"Manage your apps, extensions, and themes"**
   (the `management` permission, used to detect other installed extensions) —
   that's expected, not a bug.

> Firefox notes:
> - A Temporary Add-on **is lost when the browser closes** — reload it every
>   test session.
> - For a local exam file (`file://...`), `file://` access is restricted;
>   serving the exam page over a server (e.g. `http://localhost`) is easier
>   while testing.
> - The heartbeat alarm is clamped to a 60s minimum by Firefox (Chrome:
>   ~30s); the dashboard already tolerates this (`ONLINE_MS = 90s`).

### Chrome / Edge

1. In `manifest.json`, change:
   ```json
   "background": { "scripts": ["background.js"] }
   ```
   to:
   ```json
   "background": { "service_worker": "background.js" }
   ```
2. Open `chrome://extensions` → enable **Developer mode** → **Load unpacked** →
   select this folder.
3. For a local exam file, enable **"Allow access to file URLs"** in the
   extension's details.

## Running the proctor side (server + dashboard)

Full data flow:

```
content.js ─▶ background.js ─POST /events─▶ server.js ─▶ dashboard.html
 (event)      (log + forward)  (+heartbeat)  (store + record to disk)  (proctor views live)
```

1. Start the server (requires Node.js):

   ```
   node server.js
   ```

   The server runs at `http://localhost:8787`.
2. Open `http://localhost:8787` in a browser → **proctor dashboard**.
3. Make sure the extension is installed (see steps above). As soon as a
   student opens the exam page / switches tabs, events show up on the
   dashboard live.

The dashboard shows:
- The student list with a **green/red dot** (online / OFFLINE = extension
  disabled or dead).
- A per-student log with **screenshot/webcam thumbnails**.
- An **⚠ SUSPECT** tag when either of two independent sources fires (OR, not
  AND — one source alone is enough):
  - **Client-side whitelist**: the destination of a `navigation` or
    `tab-switch` event does not match anything in the `ALLOWED` list in
    `dashboard.html`. This is a whitelist, not a blacklist — anything **not**
    explicitly allowed is flagged. Add the real exam platform's domain(s) to
    `ALLOWED` before a live session. Non-navigation events (copy, blur,
    keyboard, heartbeat, system, screenshot, …) are never flagged by this
    check.
  - **CV review score**: `review_score` from `cv_service.py` reaches
    `CV_FLAG_THRESHOLD` (default `0.6`, in `server.js`) — fires **independently**
    of the whitelist check above.
- A **VLM verdict** per screenshot (`category`, `is_cheating`, `exam_relevance`,
  a one-sentence summary) — see below. A "Show only VLM cheat=true" toggle
  filters the log to flagged screenshots, plus a running count of
  flagged/analysed screenshots.

### Recording every session to disk

`server.js` keeps everything in memory by default, which used to mean closing
the terminal lost a whole session's results. Two independent safety nets now
write to `exam-monitor-extension/runs/run-<timestamp>/`:

| File | Written | Contains |
|---|---|---|
| `events.jsonl` | appended the moment an event arrives | every event as it arrives — survives a hard kill, but **not** the VLM/CV verdicts (they attach asynchronously seconds later) |
| `snapshot.json` | rewritten every `SNAPSHOT_MS` (default 10s) and on shutdown | the full in-memory state, **including** VLM/CV verdicts — use this for scoring |
| `media/` | on arrival | screenshots and webcam frames, decoded out of the event's base64 and referenced by filename, so the JSON stays small |

Set `RECORD=0` to disable. Stop the server with `Ctrl+C` (not a forced window
close) so it writes a final snapshot before exiting. See
[`RUN-CHECKLIST.md`](RUN-CHECKLIST.md) for the full pre/during/post-session
checklist — it exists because earlier live-test sessions lost data.

## VLM cheating decision (`is_cheating`)

`vlm_service.py` (single model, Qwen2.5-VL-3B) and `vlm_service_cascade.py`
(two-model cascade, see below) both classify each screenshot into:

```json
{"category": "...", "identity": "...", "exam_relevance": "direct|partial|none",
 "summary": "...", "is_cheating": true, "latency_ms": 842}
```

- `category`: `exam_page` (the exam itself), or one of the prohibited
  categories — `ai_assistant`, `search_engine`, `reference_material`,
  `other_application`.
- `exam_relevance`: how related the visible content is to the **current exam
  context** — `direct` (the exam question itself is being looked up/typed/
  answered), `partial` (same topic, not the specific question), or `none`
  (unrelated).
- `is_cheating` = `category` is a prohibited category **AND** `exam_relevance
  != none`. `exam_page` is never flagged. A prohibited category that is
  genuinely unrelated to the exam (e.g. ChatGPT open for a recipe) is **not**
  flagged either — this is what keeps the decision from just being "any
  non-exam tab = cheating".

The exam context the model judges relevance against is configurable per
service instance:

```bash
curl -X POST http://localhost:8788/config -d '{"lang":"ko"}'          # built-in context
curl -X POST http://localhost:8788/config -d '{"exam_context":"..."}' # custom
```

Built-in contexts exist for `en`/`ko`/`zh`/`ja` (see `EXAM_CONTEXTS` in
`vlm_service.py`), matching the language rows used for the precision/recall
evaluation. **The configured `exam_context` must match what is actually on
screen**, or `exam_relevance` judgments will be wrong — this is checked
explicitly in `RUN-CHECKLIST.md` before a recording session starts.

### Two-model cascade (`vlm_service_cascade.py`)

An alternative to running the 3B model on every screenshot:

```
screenshot -> SmolVLM-500M -> [confident AND plainly just the exam page?]
                                 |- yes -> not cheating          (fast path)
                                 '- no  -> Qwen2.5-VL + 2nd opinion -> verdict
```

A small, cheap model screens every screenshot; only ones it cannot confidently
clear are escalated to the large model, which receives the small model's
assessment as an advisory second opinion (a single consultation, not an
iterative debate — see the file's docstring for why). The same file also runs
two single-model baselines for comparison, selected via `MODE`:

```bash
MODE=cascade      python vlm_service_cascade.py   # default: small screens, large decides on escalation
MODE=large_only   python vlm_service_cascade.py   # large model on every screenshot
MODE=small_only   python vlm_service_cascade.py   # small model on every screenshot
```

`CONF_THRESHOLD` (default `0.60`) is the escalation gate — raising it
escalates more often (safer, slower). Response fields beyond the base schema:
`path` (which route the screenshot took), `escalated`, `small` (the screening
model's own assessment + confidence), and `overruled_small` (the large model
disagreed with the screener). `GET /stats` reports the escalation rate and
per-path latency; `POST /stats/reset` clears the counters between runs. See
the module docstring in `vlm_service_cascade.py` for VRAM placement options
when running both models on a single GPU.

`server.js`'s `VLM_URL` can point at either service — the single-model one on
its default port (`8788`) or the cascade one (`8791`) — the response shape is
compatible either way.

## Computer Vision integration (review score)

```
content.js ─▶ background.js ─POST /events (type:"webcam")─▶ server.js
                                                                │
                                                    POST /frame ▼
                                              cv_service.py (local, or via a
                                              Kaggle tunnel — see kaggle/README.md)
                                                                │
                                          {review_score, review_level, ...}
                                                                ▼
                                             event.cv + event.cvFlag ─▶ dashboard.html
```

`cv_service.py` wraps the `computer_vision/src/integration/` modules
(`01_head_gaze_adapter.py`, `02_yolo_output_adapter.py`, `03_event_manager.py`,
`05_multi_cue_review_score.py`) — **without modifying those files**, only
importing them. Run it locally (`python cv_service.py`, default port `8789`)
or on Kaggle (see [`kaggle/README.md`](kaggle/README.md) for free GPU +
`cloudflared` tunnel), then point `server.js` at it:

```
CV_URL=https://xxxx.trycloudflare.com node server.js
```

**Important**: `review_score` is an experimental prioritisation indicator from
the CV module — **not** a calibrated cheating probability, and not an
automatic verdict (see the disclaimer in
`computer_vision/src/integration/README.md` and
`05_multi_cue_review_score.py`). The dashboard labels it "review score" /
"review level", never "cheating probability". `cv_eval_harness.py` (repo
root) thresholds it purely to *measure* the detector against labeled clips —
keep that framing in any write-up.

**YOLO bounding boxes**: every `/frame` response also carries `detections`
(a list of `{label, confidence, bbox_xyxy}`, raw from
`02_yolo_output_adapter.py`, already confidence-filtered by that module). The
dashboard draws these boxes over the webcam thumbnail, positioned as a
percentage of the image's `naturalWidth`/`naturalHeight`, so it does not
depend on the specific capture resolution `content.js` currently uses
(320×240).

> **Heartbeat**: the extension pings the server every ~30–60s. If the ping
> stops (the student disabled the extension), that student's dot turns
> **red** within ~40–90s. This is the "monitoring was turned off" detection —
> it catches the *disable* case, but a determined student can still evade it
> (a different browser, a second device).

## Testing each signal

- **tab-switch / screenshot** — open the exam page, then switch to another tab
  (e.g. open google.com in a new tab). The popup logs the destination tab's
  URL + a screenshot thumbnail.
- **periodic screenshot** — stay on one tab and wait for the next heartbeat
  (~30–60s); a `screenshot` event with `captureReason: "periodic"` should
  appear even without switching tabs.
- **navigation** — in any tab, type a new URL into the address bar.
- **focus** — click another application outside the browser (e.g. Notepad).
  Logs "Browser lost focus".
- **idle** — leave the mouse/keyboard untouched for ~15s, or lock the screen.
- **visibility / blur / copy / paste / contextmenu / keyboard / mouseleave /
  mutation** — perform these on the exam page (`#examForm` must be present
  for the content script to activate).
- **extensions** — enable/disable any other extension (e.g. a dark-mode
  toggle) via `chrome://extensions` / `about:addons`, then wait for the next
  heartbeat alarm (≤30–60s) — a new event only appears when the active
  extension list actually *changes* from the previous check (deduplicated,
  not spammed every heartbeat).

## Important limitations (report these in the paper)

- **Can be disabled by the student.** This extension can be disabled/
  uninstalled via `chrome://extensions`. To be trustworthy in real
  proctoring, it must be **force-installed** via a managed policy
  (`ExtensionInstallForcelist`) on a school/enterprise-managed device.
- **`content.js` only activates on a page with `#examForm`.** Change this
  marker to match your real exam page. Also restrict `matches` in
  `manifest.json` to the exam origin (instead of `<all_urls>`) for a
  production build.
- **Screenshots need `host_permissions`.** `<all_urls>` is used so that
  `captureVisibleTab` works; it (expectedly) fails on `chrome://` pages and
  the Web Store.
- **Privacy.** This prototype stores screenshots of every tab it activates in
  `chrome.storage.local`. For a real system: restrict when captures happen,
  obtain consent, and process/store on a trusted server — never on the
  client.
- **Detecting other extensions only knows "installed & enabled", not
  "currently in use".** The `management` permission only reports which
  extensions exist and are active in that browser, not their real-time
  activity. A student could disable one before the exam and re-enable it
  after (though that toggle itself is still a signal if checked
  periodically). This permission is also **not hidden** — the browser shows
  an explicit warning to the user on install/update ("Manage your apps,
  extensions, and themes"), and is invisible entirely if the student uses a
  different profile/browser.
- **Webcam privacy.** Periodic webcam capture (for the CV review score) is a
  far more invasive data-handling change than tab screenshots — video of the
  student's face is sent to a third-party service (Kaggle + a tunnel). The
  browser's built-in camera-permission prompt is **not** sufficient informed
  consent for real proctoring; before use beyond this prototype, it needs
  research/ethics review, and ideally an explicit on-page notice (not just a
  silent OS/browser permission popup).
