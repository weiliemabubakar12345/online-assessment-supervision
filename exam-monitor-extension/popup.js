// Popup — reads the central log from storage.local and renders it live.
// (MV3 disallows inline scripts in extension pages, so this must be a separate file.)
// Cross-browser: browser (Firefox/Zen, promises) or chrome (Chrome, promises).

const api = (typeof browser !== "undefined") ? browser : chrome;

const logEl = document.getElementById("log");
const countEl = document.getElementById("count");

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

function render(events) {
  events = events || [];
  logEl.innerHTML = "";
  countEl.textContent = events.length + (events.length === 1 ? " event" : " events");

  if (!events.length) {
    logEl.innerHTML = '<div class="empty">No events yet.</div>';
    return;
  }

  // newest first
  events.slice().reverse().forEach(function (ev) {
    const row = document.createElement("div");
    row.className = "entry t-" + ev.type;
    const t = (ev.timestamp || "").split("T")[1] ? ev.timestamp.split("T")[1].replace("Z", "") : "";
    row.innerHTML =
      '<span class="ts">' + t + '</span> ' +
      '<span class="type">[' + ev.type.toUpperCase() + ']</span> ' +
      escapeHtml(ev.detail || "");
    if (ev.image) {
      const img = document.createElement("img");
      img.src = ev.image;
      img.className = "thumb";
      row.appendChild(img);
    }
    logEl.appendChild(row);
  });
}

async function load() {
  const r = await api.storage.local.get("events");
  render(r.events);
}
load();

// Live update while the popup is open
api.storage.onChanged.addListener(function (changes, area) {
  if (area === "local" && changes.events) render(changes.events.newValue);
});

document.getElementById("export").addEventListener("click", async function () {
  const r = await api.storage.local.get("events");
  const blob = new Blob([JSON.stringify(r.events || [], null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "exam-monitor-log.json";
  a.click();
  URL.revokeObjectURL(url);
});

document.getElementById("clear").addEventListener("click", function () {
  api.storage.local.set({ events: [] });
});
