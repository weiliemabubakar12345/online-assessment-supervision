// Exam Screen-Side Monitor — background script (MV3).
//
// Cross-browser: `browser` (Firefox/Zen, promise-based) or `chrome` (Chrome,
// promise-based in MV3). We alias to `api` and use promises everywhere.
//
// This is the powerful half: it uses extension APIs that the exam page's own
// JavaScript CANNOT reach because of the browser sandbox — namely WHICH tab the
// student switched to (URL + title), whole-browser focus, idle/locked state, and
// a screenshot of the newly-active tab. It also receives the in-page signals the
// content script forwards, keeps one central persisted log, AND forwards every
// event (plus a periodic heartbeat) to the proctor server for live monitoring.

const api = (typeof browser !== "undefined") ? browser : chrome;

const SERVER_URL = "http://localhost:8787"; // proctor server (server.js)
const MAX_EVENTS = 300;                      // keep the local log/screenshots bounded

let events = null;      // in-memory cache; storage is the source of truth
let studentId = null;   // stable per-install id, so the proctor can tell students apart

/* ---- Identity ---- */
async function getStudentId() {
  if (studentId) return studentId;
  const r = await api.storage.local.get("studentId");
  if (r.studentId) { studentId = r.studentId; return studentId; }
  studentId = "stu-" + Math.random().toString(36).slice(2, 8);
  await api.storage.local.set({ studentId: studentId });
  return studentId;
}

/* ---- Local log ---- */
async function ensureLoaded() {
  if (events === null) {
    const r = await api.storage.local.get("events");
    events = Array.isArray(r.events) ? r.events : [];
  }
}

async function addEvent(type, detail, extra) {
  await ensureLoaded();
  const rec = Object.assign(
    { type: type, detail: detail, timestamp: new Date().toISOString() },
    extra || {}
  );
  events.push(rec);
  if (events.length > MAX_EVENTS) events = events.slice(-MAX_EVENTS);
  await api.storage.local.set({ events: events });

  // Forward to the proctor server (best-effort). Callers that need the
  // server's response (currently: webcam frames, for calibration_phase) can
  // await the returned promise; everyone else fires-and-forgets it.
  const sid = await getStudentId();
  return postToServer("/events", { studentId: sid, event: rec });
}

/* ---- Talk to the proctor server ---- */
async function postToServer(path, body) {
  try {
    const r = await fetch(SERVER_URL + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    return await r.json();
  } catch (e) {
    // Server offline: this prototype drops it. A real system would queue/retry;
    // the missing heartbeat below already tells the proctor the student went dark.
    return null;
  }
}

/* ---- Heartbeat: lets the proctor detect a disabled/offline extension ---- */
async function sendHeartbeat() {
  const sid = await getStudentId();
  postToServer("/heartbeat", { studentId: sid, at: new Date().toISOString() });
}
function ensureHeartbeatAlarm() {
  // Chrome honors ~30s; Firefox clamps to a 60s minimum — both fine.
  api.alarms.create("heartbeat", { periodInMinutes: 0.5 });
}
api.alarms.onAlarm.addListener(async function (a) {
  if (a.name === "heartbeat") {
    sendHeartbeat();
    checkInstalledExtensions();
    /* Periodic screen capture. Without this the VLM only ever sees the screen at
       the moment of a tab or window switch, so two real cheating patterns are
       never captured at all: a side-by-side window the student never clicks
       into, and typing a cheat URL into the address bar of the current tab
       (tabs.onUpdated logs a "navigation" event but takes no screenshot).
       Firefox clamps alarms to 60s, so a behaviour must last >60s to be sure
       of being caught here; the evaluation protocol allows for that. */
    try {
      const w = await api.windows.getLastFocused({});
      if (w && w.focused) captureAndLog(w.id, "periodic");
    } catch (e) {
      addEvent("system", "Periodic capture skipped: " + e.message);
    }
  }
});

/* ---- Other installed/enabled extensions (needs the "management" permission) ----
   Tells the proctor WHICH other extensions the student has installed and
   enabled (e.g. an AI-assistant sidebar, a translator, a quiz-answer tool) —
   not whether they're actively being used right now, just installed+enabled.
   Reuses the heartbeat alarm so it's rechecked periodically (a student could
   disable something before the exam and re-enable it after), but only logs
   an event when the active set actually changes, to avoid spamming the log
   with an identical list every 30s. */
let lastExtensionsSignature = null;

async function checkInstalledExtensions() {
  if (!api.management || !api.management.getAll) return; // unsupported in this browser
  try {
    const self = await api.management.getSelf();
    const all = await api.management.getAll();
    const active = all
      .filter(function (e) { return e.type === "extension" && e.enabled && e.id !== self.id; })
      .map(function (e) { return e.name; })
      .sort();

    const signature = active.join("|");
    if (signature === lastExtensionsSignature) return; // no change since last check
    lastExtensionsSignature = signature;

    await addEvent("extensions",
      active.length
        ? active.length + " other extension(s) active: " + active.join(", ")
        : "No other extensions active.",
      { extensions: active });
  } catch (e) {
    // Most likely the "management" permission wasn't granted (older install,
    // or the user declined it) — log once rather than retrying every alarm tick.
    if (lastExtensionsSignature === null) {
      lastExtensionsSignature = "__error__";
      addEvent("system", "Extension list check failed: " + e.message);
    }
  }
}

/* ---- In-page signals forwarded from the exam tab (content.js) ---- */
api.runtime.onMessage.addListener(function (msg) {
  if (msg && msg.kind === "event") {
    addEvent(msg.type, msg.detail, { source: msg.source });
  } else if (msg && msg.kind === "webcam-frame") {
    // Periodic webcam frame from content.js; forwarded via the same
    // addEvent()/postToServer() pipeline. The server routes event.type ===
    // "webcam" to the CV review-score service (see server.js). Unlike the
    // "event" branch above, this one returns a promise: content.js needs
    // calibration_phase back so it can speed up capture while the CV
    // pipeline is still calibrating and drop back to the normal rate once
    // READY (see server.js's /events handler, which awaits analyzeWithCV
    // and echoes event.cv back specifically for this purpose).
    return addEvent("webcam", "Webcam frame captured.", { source: msg.source, image: msg.image })
      .then(function (result) {
        const cv = result && result.cv;
        return { calibrationPhase: (cv && cv.calibration_phase) || null };
      });
  }
});

/* ---- Which tab did the student switch to? (page JS is blind to this) ---- */
api.tabs.onActivated.addListener(async function (info) {
  try {
    const tab = await api.tabs.get(info.tabId);
    await addEvent("tab-switch",
      "Switched to: " + (tab.title || "(no title)") + " — " + (tab.url || "(url hidden)"),
      { url: tab.url, title: tab.title });
    captureAndLog(info.windowId, "tab switch");
  } catch (e) {
    addEvent("system", "tab-switch read failed: " + e.message);
  }
});

/* ---- Navigation within a tab (e.g. typing chatgpt.com into the address bar) ---- */
api.tabs.onUpdated.addListener(function (tabId, changeInfo, tab) {
  if (changeInfo.url) {
    addEvent("navigation", "Tab navigated to: " + changeInfo.url,
      { url: changeInfo.url, title: tab.title });
  }
});

/* ---- Whole browser gained / lost focus (switched to another desktop app) ---- */
api.windows.onFocusChanged.addListener(function (windowId) {
  const NONE = api.windows.WINDOW_ID_NONE;
  if (windowId === NONE) {
    // Focus left this browser entirely (another browser or desktop app). We can
    // detect the departure but NOT capture what is there — see Limitations.
    addEvent("focus", "Browser lost focus (switched to another application).");
  } else {
    // Focus moved to another window of THIS browser. Unlike a cross-application
    // switch, this window is within our capture scope, so screenshot it too —
    // otherwise a second same-browser window would be a capture blind spot.
    addEvent("focus", "Browser focused (window " + windowId + ").");
    captureAndLog(windowId, "window switch");
  }
});

/* ---- Idle / locked screen ---- */
api.idle.setDetectionInterval(15); // seconds
api.idle.onStateChanged.addListener(function (state) {
  addEvent("idle", "User state changed to: " + state + ".");
});

/* ---- Screenshot of the currently visible tab ---- */
async function captureAndLog(windowId, reason) {
  try {
    // Don't screenshot our own proctor dashboard/exam page — capturing it is
    // recursive (a screenshot of the dashboard showing screenshots showing
    // the dashboard...) and it isn't a "what the student is looking at"
    // signal worth flagging; it's our own tool, not something suspicious.
    const [activeTab] = await api.tabs.query({ active: true, windowId: windowId });
    if (activeTab && activeTab.url && activeTab.url.indexOf(SERVER_URL) === 0) return;

    // Timing for the T_capture + T_network + T_inference decomposition the
    // evaluation protocol reports. t0 -> capturedAt measures the browser's own
    // screenshot cost; the server derives the rest from capturedAt.
    const t0 = Date.now();
    const dataUrl = await api.tabs.captureVisibleTab(windowId, { format: "jpeg", quality: 40 });
    const tCaptureMs = Date.now() - t0;
    await addEvent("screenshot", "Captured visible tab (" + reason + ").",
                   { image: dataUrl, capturedAt: Date.now(), tCaptureMs: tCaptureMs, captureReason: reason });
  } catch (e) {
    // Expected on restricted pages (about:, chrome://, the add-on store) or
    // throttling, or if host permission isn't granted yet — log softly.
    await addEvent("system", "Screenshot skipped (" + reason + "): " + e.message);
  }
}

/* ---- Lifecycle ---- */
api.runtime.onInstalled.addListener(function () {
  ensureHeartbeatAlarm();
  sendHeartbeat();
  checkInstalledExtensions();
  addEvent("system", "Exam monitor installed / updated.");
});
if (api.runtime.onStartup) {
  api.runtime.onStartup.addListener(function () {
    ensureHeartbeatAlarm();
    sendHeartbeat();
    checkInstalledExtensions();
  });
}
ensureHeartbeatAlarm(); // also arm on background wake
