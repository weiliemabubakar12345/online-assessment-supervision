// Exam Screen-Side Monitor — content script (runs in the extension's isolated world).
//
// It only activates on the exam page, detected via the #examForm marker that the
// exam prototype (exam-detector-prototype.html) contains. That keeps it from
// spamming events on every website the student visits. Each in-page signal is
// forwarded to the service worker (background.js), which owns the central log.
//
// NOTE: This captures the SAME signals as the standalone HTML prototype, but by
// registering its OWN listeners — it does not (and cannot) read the exam page's
// private `events` array, which lives in a closure the isolated world can't see.

(async function () {
  "use strict";

  // Cross-browser: browser (Firefox/Zen) or chrome (Chrome).
  const api = (typeof browser !== "undefined") ? browser : chrome;

  // Gate: only monitor the exam page, UNLESS the popup's "Force monitor this
  // page" toggle (storage.local.forceMonitor) is on -- needed for testing
  // against real third-party test sites that don't have #examForm. This is a
  // single global flag, not per-site: turn it on before loading the page you
  // want monitored, off when done, same as any other extension toggle.
  const hasExamMarker = !!document.getElementById("examForm");
  const forced = !!(await api.storage.local.get("forceMonitor")).forceMonitor;
  if (!hasExamMarker && !forced) return;

  function send(type, detail) {
    try {
      const p = api.runtime.sendMessage({ kind: "event", type, detail, source: location.href });
      if (p && p.catch) p.catch(function () {}); // ignore "no receiver" rejections
    } catch (e) {
      // Background may be restarting; the signal is best-effort.
    }
  }

  send("system", hasExamMarker
    ? "Content script attached to exam page."
    : "Content script attached via forced monitoring (no #examForm on this page).");

  /* Webcam capture: periodic frames forwarded for CV review-score analysis.
     getUserMedia is a page-level Web API gated by the browser's own camera
     permission prompt (not an extension permission — manifest.json needs no
     change for this). Requires a secure context: load the exam page over
     http://localhost:8787/exam, not a local file:// page, or the browser may
     refuse the camera. One-shot: if permission is denied, log it once and do
     not keep re-prompting. */
  const WEBCAM_CAPTURE_INTERVAL_MS = 1000;      // steady-state rate once the CV pipeline reports READY
  const WEBCAM_CALIBRATION_INTERVAL_MS = 200;   // faster rate while it's still calibrating (eye/gaze/head
                                                 // calibrators need many samples — see cv_service.py)
  const WEBCAM_JPEG_QUALITY = 0.5;
  const WEBCAM_CAPTURE_WIDTH = 320;
  const WEBCAM_CAPTURE_HEIGHT = 240;

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({
      video: { width: WEBCAM_CAPTURE_WIDTH, height: WEBCAM_CAPTURE_HEIGHT }
    }).then(function (stream) {
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      video.style.display = "none";
      document.body.appendChild(video);
      video.play();

      const canvas = document.createElement("canvas");
      canvas.width = WEBCAM_CAPTURE_WIDTH;
      canvas.height = WEBCAM_CAPTURE_HEIGHT;
      const ctx = canvas.getContext("2d");

      send("system", "Webcam capture started.");

      let announcedReady = false;

      // Self-scheduling loop (instead of a fixed setInterval): each capture
      // waits for the previous frame's round trip before deciding the next
      // delay, based on the calibration_phase the server echoes back. This
      // naturally paces to whatever the actual CV round-trip time is (never
      // piles up concurrent requests) while still calibrating as fast as
      // that round trip allows, then relaxes to the normal rate once READY.
      function captureLoop() {
        if (video.readyState < 2) { // not enough data yet
          setTimeout(captureLoop, WEBCAM_CALIBRATION_INTERVAL_MS);
          return;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", WEBCAM_JPEG_QUALITY);

        Promise.resolve(api.runtime.sendMessage({ kind: "webcam-frame", image: dataUrl, source: location.href }))
          .then(function (resp) {
            const phase = resp && resp.calibrationPhase;
            const stillCalibrating = !!phase && phase !== "READY";
            if (!stillCalibrating && phase === "READY" && !announcedReady) {
              announcedReady = true;
              send("system", "CV calibration complete (READY) — webcam capture back to normal rate.");
            }
            setTimeout(captureLoop, stillCalibrating ? WEBCAM_CALIBRATION_INTERVAL_MS : WEBCAM_CAPTURE_INTERVAL_MS);
          })
          .catch(function () {
            // Background/service worker unreachable — fall back to the
            // normal rate rather than hammering it at the fast interval.
            setTimeout(captureLoop, WEBCAM_CAPTURE_INTERVAL_MS);
          });
      }

      captureLoop();
    }).catch(function (e) {
      send("system", "Webcam permission denied or unavailable: " + e.message);
    });
  }

  /* Tab hidden / backgrounded */
  document.addEventListener("visibilitychange", function () {
    send("visibility", document.visibilityState === "hidden"
      ? "Exam tab hidden / backgrounded."
      : "Exam tab visible again.");
  });

  /* Window focus */
  window.addEventListener("blur",  function () { send("blur", "Exam window lost focus."); });
  window.addEventListener("focus", function () { send("blur", "Exam window regained focus."); });

  /* Clipboard */
  document.addEventListener("copy",  function () { send("copy",  "Copy on exam page."); });
  document.addEventListener("paste", function () { send("paste", "Paste on exam page."); });

  /* Right-click */
  document.addEventListener("contextmenu", function (e) {
    const tag = e.target && e.target.tagName ? e.target.tagName.toLowerCase() : "?";
    send("contextmenu", "Right-click on <" + tag + ">.");
  });

  /* Keyboard shortcuts */
  document.addEventListener("keydown", function (e) {
    const k = (e.key || "").toLowerCase();
    const mod = e.ctrlKey || e.metaKey;
    if (k === "printscreen") send("keyboard", "PrintScreen pressed.");
    else if (mod && ["c", "v", "x", "p", "s"].indexOf(k) !== -1)
      send("keyboard", "Shortcut " + (e.metaKey ? "Cmd" : "Ctrl") + "+" + k.toUpperCase() + ".");
  });

  /* Cursor leaving the viewport */
  document.addEventListener("mouseout", function (e) {
    if (!e.relatedTarget) send("mouseleave", "Cursor left the exam viewport.");
  });

  /* Extension-artifact detector (Grammarly / translators injecting into the DOM) */
  const PATTERNS = ["translate", "deepl", "grammarly", "quillbot", "reverso", "goog-te"];
  new MutationObserver(function (muts) {
    for (const m of muts) {
      for (const n of m.addedNodes) {
        if (n.nodeType !== 1) continue;
        const hay = ((n.id || "") + " " +
          (typeof n.className === "string" ? n.className : "")).toLowerCase();
        const hit = PATTERNS.find(function (p) { return hay.indexOf(p) !== -1; });
        if (hit) send("mutation",
          'Possible extension artifact ("' + hit + '"): <' + n.tagName.toLowerCase() + '>');
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
})();
