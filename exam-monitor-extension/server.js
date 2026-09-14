// Proctor server — pure Node.js, no dependencies, no npm install.
// Receives events + heartbeats from the extension and serves the proctor
// dashboard. In-memory only (restarting the server clears everything) — this is
// a prototype, not production. Run with:  node server.js
//
// Endpoints:
//   POST /events     { studentId, event }        -> store an event
//   POST /heartbeat  { studentId, at }           -> mark student alive
//   GET  /students                               -> [{studentId,lastSeen,eventCount}]
//   GET  /events?studentId=ID                    -> [event, ...]
//   GET  /                                         -> dashboard.html
//
// VLM_URL=https://xxxx.trycloudflare.com node server.js   (override the VLM endpoint)
// CV_URL=https://xxxx.trycloudflare.com node server.js    (override the CV review-score endpoint)

const http = require("http");
const url = require("url");
const fs = require("fs");
const path = require("path");

const PORT = 8787;
const MAX_EVENTS_PER_STUDENT = 500;
// VLM endpoint. Local by default; set VLM_URL to the Colab tunnel URL when the
// VLM runs on Colab, e.g.  VLM_URL=https://xxxx.trycloudflare.com node server.js
const VLM_URL = process.env.VLM_URL || "https://cell-reporting-directly-warriors.trycloudflare.com";
// CV review-score endpoint (cv_service.py). Local by default; set CV_URL to the
// Kaggle tunnel URL when the CV pipeline runs on Kaggle — see
// exam-monitor-extension/kaggle/README.md, e.g.
// CV_URL=https://xxxx.trycloudflare.com node server.js
const CV_URL = process.env.CV_URL || "http://localhost:8790";
// Extension-side trigger threshold for a webcam-derived review_score. This is
// NOT one of cv_service.py's own review_level tiers (MODERATE/HIGH/VERY_HIGH
// at 0.25/0.50/0.75) — it's an ad-hoc cutoff (inside the HIGH tier) that the
// dashboard ORs together with the existing keyword-based rule filter.
const CV_FLAG_THRESHOLD = 0.6;

// studentId -> { lastSeen: ms, events: [] }
const students = Object.create(null);

// --------------------------------------------------------------- recorder --
// Everything above lives in memory only, so closing the terminal used to
// destroy a whole session's results. Two independent safety nets now write to
// disk, because each one alone has a hole:
//
//   events.jsonl  appended the moment an event arrives -- survives a hard kill
//                 and survives MAX_EVENTS_PER_STUDENT trimming, but cannot
//                 contain the VLM/CV verdicts, which attach asynchronously
//                 seconds later.
//   snapshot.json rewritten every SNAPSHOT_MS (and on shutdown) from live
//                 memory -- carries the verdicts, but only what memory still
//                 holds.
//
// Use snapshot.json for scoring; fall back to events.jsonl for anything that
// was trimmed. Base64 images are never written into either file: they are
// decoded once into media/ and referenced by filename, so the JSON stays small
// enough to open.
const RECORD = process.env.RECORD !== "0";
const SNAPSHOT_MS = Number(process.env.SNAPSHOT_MS || 10000);
const RUN_DIR = path.join(__dirname, "runs",
  "run-" + new Date().toISOString().replace(/[:.]/g, "-"));
const MEDIA_DIR = path.join(RUN_DIR, "media");
let mediaSeq = 0;

function recordInit() {
  if (!RECORD) return;
  fs.mkdirSync(MEDIA_DIR, { recursive: true });
  fs.writeFileSync(path.join(RUN_DIR, "README.txt"),
    "Recorded by server.js on " + new Date().toISOString() + "\n\n" +
    "snapshot.json  full event log incl. VLM/CV verdicts (use this for scoring)\n" +
    "events.jsonl   append-only arrival log, no verdicts (recovery fallback)\n" +
    "media/         screenshots and webcam frames referenced by image_file\n");
}

// Write the image out once and hand back its filename. The in-memory event
// KEEPS its base64 image -- the dashboard renders from memory, so stripping it
// there would blank every thumbnail.
function recordMedia(studentId, event) {
  if (!RECORD || !event || typeof event.image !== "string") return null;
  const m = /^data:image\/(png|jpe?g|webp);base64,(.*)$/i.exec(event.image);
  if (!m) return null;
  const ext = m[1].toLowerCase() === "jpg" ? "jpeg" : m[1].toLowerCase();
  const stamp = String(event.timestamp || new Date().toISOString()).replace(/[:.]/g, "-");
  const name = (studentId || "unknown") + "_" + stamp + "_" + (++mediaSeq) + "." + ext;
  try {
    fs.writeFileSync(path.join(MEDIA_DIR, name), Buffer.from(m[2], "base64"));
    return path.join("media", name);
  } catch (e) {
    console.error("record: could not write media:", e.message);
    return null;
  }
}

function stripImage(event) {
  const copy = Object.assign({}, event);
  delete copy.image;
  return copy;
}

function recordArrival(studentId, event) {
  if (!RECORD) return;
  try {
    fs.appendFileSync(path.join(RUN_DIR, "events.jsonl"),
      JSON.stringify({ studentId: studentId, event: stripImage(event) }) + "\n");
  } catch (e) {
    console.error("record: could not append event:", e.message);
  }
}

// Atomic: write a temp file then rename, so a snapshot interrupted halfway
// never leaves a truncated JSON where the good one used to be.
function saveSnapshot() {
  if (!RECORD) return;
  const out = { savedAt: new Date().toISOString(), students: {} };
  for (const id of Object.keys(students)) {
    out.students[id] = {
      lastSeen: students[id].lastSeen,
      events: students[id].events.map(stripImage),
    };
  }
  const target = path.join(RUN_DIR, "snapshot.json");
  const tmp = target + ".tmp";
  try {
    fs.writeFileSync(tmp, JSON.stringify(out, null, 1));
    fs.renameSync(tmp, target);
  } catch (e) {
    console.error("record: could not write snapshot:", e.message);
  }
}

function shutdown(signal) {
  console.log("\n" + signal + " -- writing final snapshot...");
  saveSnapshot();
  if (RECORD) console.log("Run saved to: " + RUN_DIR);
  process.exit(0);
}

// Ask the VLM service what a screenshot shows; attach the result to the event
// object in place (the dashboard polls and picks it up on the next refresh).
async function analyzeWithVLM(event) {
  event.vlm = { status: "analyzing" };
  const tSent = Date.now();
  try {
    const r = await fetch(VLM_URL + "/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: event.image })
    });
    const result = await r.json();
    const tDone = Date.now();

    // T_total = T_capture + T_network + T_inference, as specified in the
    // evaluation protocol. T_inference comes from the VLM service itself;
    // network is whatever the round trip cost beyond that.
    const roundTrip = tDone - tSent;
    const inference = (typeof result.latency_ms === "number") ? result.latency_ms : null;
    result.timing = {
      t_capture_ms:   (typeof event.tCaptureMs === "number") ? event.tCaptureMs : null,
      t_queue_ms:     event.capturedAt ? (tSent - event.capturedAt) : null,
      t_roundtrip_ms: roundTrip,
      t_inference_ms: inference,
      t_network_ms:   (inference !== null) ? (roundTrip - inference) : null,
      t_total_ms:     event.capturedAt ? (tDone - event.capturedAt) : null
    };
    event.vlm = result;

    // One line per screenshot: reconcile the run against the recording sheet
    // afterwards without replaying the video.
    const ts = new Date().toISOString().slice(11, 19);
    console.log("[VLM " + ts + "] " + result.category +
                "  rel=" + result.exam_relevance +
                "  cheat=" + result.is_cheating +
                "  capture=" + result.timing.t_capture_ms + "ms" +
                "  net=" + result.timing.t_network_ms + "ms" +
                "  inf=" + result.timing.t_inference_ms + "ms" +
                "  total=" + result.timing.t_total_ms + "ms" +
                "  (" + (event.captureReason || "?") + ")" +
                (result.warning ? "  !! " + result.warning : ""));
  } catch (e) {
    console.error("VLM analyze error:", e);
    event.vlm = { error: "VLM service unreachable: " + e.message };
  }
}

// Ask the CV service for a review_score/review_level reading on a webcam
// frame; attach the result (and the derived flag) to the event object in
// place, same fire-and-forget pattern as analyzeWithVLM above.
async function analyzeWithCV(event, studentId) {
  event.cv = { status: "analyzing" };
  try {
    const r = await fetch(CV_URL + "/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ studentId: studentId, image: event.image, timestamp: Date.now() })
    });
    const result = await r.json();
    event.cv = result;
    event.cvFlag = typeof result.review_score === "number" && result.review_score >= CV_FLAG_THRESHOLD;
  } catch (e) {
    console.error("CV analyze error:", e);
    event.cv = { error: "CV service unreachable: " + e.message };
    event.cvFlag = false;
  }
}

function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

function readBody(req) {
  return new Promise(function (resolve) {
    let data = "";
    req.on("data", function (c) { data += c; if (data.length > 8e6) req.destroy(); });
    req.on("end", function () {
      try { resolve(JSON.parse(data || "{}")); } catch (e) { resolve({}); }
    });
  });
}

function json(res, code, obj) {
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
}

function ensureStudent(id) {
  if (!students[id]) students[id] = { lastSeen: Date.now(), events: [] };
  return students[id];
}

const server = http.createServer(async function (req, res) {
  const parsed = url.parse(req.url, true);
  cors(res);

  if (req.method === "OPTIONS") { res.writeHead(204); return res.end(); }

  // --- ingest events ---
  if (req.method === "POST" && parsed.pathname === "/events") {
    const body = await readBody(req);
    const s = ensureStudent(body.studentId || "unknown");
    if (body.event) {
      const mediaPath = recordMedia(body.studentId || "unknown", body.event);
      if (mediaPath) body.event.image_file = mediaPath;
      recordArrival(body.studentId || "unknown", body.event);
      s.events.push(body.event);
      if (s.events.length > MAX_EVENTS_PER_STUDENT) {
        s.events = s.events.slice(-MAX_EVENTS_PER_STUDENT);
      }
      // Screenshot -> run the VLM (async; result attaches to the event later).
      if (body.event.type === "screenshot" && body.event.image) {
        analyzeWithVLM(body.event);
      }
      // Webcam frame -> run the CV review-score pipeline. Awaited (unlike the
      // VLM branch above) so the response to THIS request can carry back
      // calibration_phase — content.js uses it to speed up webcam capture
      // while the CV pipeline is still calibrating, then drop back to the
      // normal rate once READY.
      if (body.event.type === "webcam" && body.event.image) {
        await analyzeWithCV(body.event, body.studentId || "unknown");
      }
    }
    s.lastSeen = Date.now();
    return json(res, 200, { ok: true, cv: (body.event && body.event.cv) || null });
  }

  // --- heartbeat ---
  if (req.method === "POST" && parsed.pathname === "/heartbeat") {
    const body = await readBody(req);
    ensureStudent(body.studentId || "unknown").lastSeen = Date.now();
    return json(res, 200, { ok: true });
  }

  // --- list students ---
  if (req.method === "GET" && parsed.pathname === "/students") {
    const list = Object.keys(students).map(function (id) {
      return { studentId: id, lastSeen: students[id].lastSeen,
               eventCount: students[id].events.length };
    });
    return json(res, 200, list);
  }

  // --- one student's events ---
  if (req.method === "GET" && parsed.pathname === "/events") {
    const id = parsed.query.studentId;
    return json(res, 200, (students[id] && students[id].events) || []);
  }

  // --- dashboard ---
  if (req.method === "GET" && (parsed.pathname === "/" || parsed.pathname === "/index.html")) {
    try {
      const html = fs.readFileSync(path.join(__dirname, "dashboard.html"), "utf8");
      // no-store: this file changes constantly during dev/testing, and a
      // browser silently serving a stale cached copy on a plain reload looks
      // identical to "the fix didn't work" — not worth the ambiguity.
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
      return res.end(html);
    } catch (e) {
      res.writeHead(500); return res.end("dashboard.html not found");
    }
  }

  // --- serve the exam page over http (handy for capture + Firefox file:// limits) ---
  if (req.method === "GET" && parsed.pathname === "/exam") {
    try {
      const html = fs.readFileSync(
        path.join(__dirname, "exam-detector-prototype.html"), "utf8");
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
      return res.end(html);
    } catch (e) {
      res.writeHead(500); return res.end("exam-detector-prototype.html not found");
    }
  }

  res.writeHead(404); res.end("Not found");
});

server.listen(PORT, function () {
  console.log("Proctor server running:  http://localhost:" + PORT);
  console.log("  VLM_URL = " + VLM_URL);
  console.log("  CV_URL  = " + CV_URL);
  if (RECORD) {
    recordInit();
    setInterval(saveSnapshot, SNAPSHOT_MS).unref();
    console.log("  RECORD  = " + RUN_DIR + "  (snapshot every " +
                Math.round(SNAPSHOT_MS / 1000) + "s; set RECORD=0 to disable)");
  } else {
    console.log("  RECORD  = disabled (RECORD=0) -- nothing will be saved to disk");
  }
  console.log("Open that URL for the dashboard. Ctrl+C to stop.");
});

process.on("SIGINT", function () { shutdown("SIGINT"); });
process.on("SIGTERM", function () { shutdown("SIGTERM"); });
