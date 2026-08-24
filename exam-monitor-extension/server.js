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
const VLM_URL = process.env.VLM_URL || "https://cooked-escape-abraham-tubes.trycloudflare.com";
// CV review-score endpoint (cv_service.py). Local by default; set CV_URL to the
// Kaggle tunnel URL when the CV pipeline runs on Kaggle — see
// exam-monitor-extension/kaggle/README.md, e.g.
// CV_URL=https://xxxx.trycloudflare.com node server.js
const CV_URL = process.env.CV_URL || "https://switching-arrangements-advisor-distinct.trycloudflare.com";
// Extension-side trigger threshold for a webcam-derived review_score. This is
// NOT one of cv_service.py's own review_level tiers (MODERATE/HIGH/VERY_HIGH
// at 0.25/0.50/0.75) — it's an ad-hoc cutoff (inside the HIGH tier) that the
// dashboard ORs together with the existing keyword-based rule filter.
const CV_FLAG_THRESHOLD = 0.6;

// studentId -> { lastSeen: ms, events: [] }
const students = Object.create(null);

// Ask the VLM service what a screenshot shows; attach the result to the event
// object in place (the dashboard polls and picks it up on the next refresh).
async function analyzeWithVLM(event) {
  event.vlm = { status: "analyzing" };
  try {
    const r = await fetch(VLM_URL + "/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: event.image })
    });
    event.vlm = await r.json();
  } catch (e) {
    console.error("VLM analyze error:", e);   // <-- tambahin ini
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
      s.events.push(body.event);
      if (s.events.length > MAX_EVENTS_PER_STUDENT) {
        s.events = s.events.slice(-MAX_EVENTS_PER_STUDENT);
      }
      // Screenshot -> run the VLM (async; result attaches to the event later).
      if (body.event.type === "screenshot" && body.event.image) {
        analyzeWithVLM(body.event);
      }
      // Webcam frame -> run the CV review-score pipeline (async; result
      // attaches to the event later, same pattern as the VLM branch above).
      if (body.event.type === "webcam" && body.event.image) {
        analyzeWithCV(body.event, body.studentId || "unknown");
      }
    }
    s.lastSeen = Date.now();
    return json(res, 200, { ok: true });
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
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
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
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      return res.end(html);
    } catch (e) {
      res.writeHead(500); return res.end("exam-detector-prototype.html not found");
    }
  }

  res.writeHead(404); res.end("Not found");
});

server.listen(PORT, function () {
  console.log("Proctor server running:  http://localhost:" + PORT);
  console.log("Open that URL for the dashboard. Ctrl+C to stop.");
});
