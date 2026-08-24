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

  // Forward to the proctor server (best-effort).
  const sid = await getStudentId();
  postToServer("/events", { studentId: sid, event: rec });
}

/* ---- Talk to the proctor server ---- */
async function postToServer(path, body) {
  try {
    await fetch(SERVER_URL + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  } catch (e) {
    // Server offline: this prototype drops it. A real system would queue/retry;
    // the missing heartbeat below already tells the proctor the student went dark.
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
api.alarms.onAlarm.addListener(function (a) {
  if (a.name === "heartbeat") sendHeartbeat();
});

/* ---- In-page signals forwarded from the exam tab (content.js) ---- */
api.runtime.onMessage.addListener(function (msg) {
  if (msg && msg.kind === "event") {
    addEvent(msg.type, msg.detail, { source: msg.source });
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
    const dataUrl = await api.tabs.captureVisibleTab(windowId, { format: "jpeg", quality: 40 });
    await addEvent("screenshot", "Captured visible tab (" + reason + ").", { image: dataUrl });
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
  addEvent("system", "Exam monitor installed / updated.");
});
if (api.runtime.onStartup) {
  api.runtime.onStartup.addListener(function () {
    ensureHeartbeatAlarm();
    sendHeartbeat();
  });
}
ensureHeartbeatAlarm(); // also arm on background wake
