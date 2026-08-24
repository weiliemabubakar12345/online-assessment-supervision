# Exam Screen-Side Monitor — Prototype Extension (MV3)

Prototype extension proctoring untuk sisi **layar/screen**. Menangkap sinyal
yang sama seperti `exam-detector-prototype.html`, **plus** hal-hal yang tidak
bisa dilihat JavaScript di dalam halaman karena sandbox browser:

- **Tab tujuan** saat mahasiswa pindah tab — beserta **URL & judul** (mis. tahu
  dia buka ChatGPT/Google, bukan cuma "pindah tab").
- **Navigasi** URL di dalam tab.
- **Fokus seluruh browser** (pindah ke aplikasi lain).
- **Idle / layar terkunci**.
- **Screenshot** tab yang sedang aktif saat pindah tab.
- **Webcam periodik** — frame webcam dikirim ke pipeline computer-vision
  (`computer_vision/`, lihat [Integrasi Computer Vision](#integrasi-computer-vision-review-score)
  di bawah) untuk *review score* lintas-cue (head pose, gaze, objek).
- **Extension lain yang terinstall & aktif** di browser (via `chrome.management`
  / `browser.management`) — mis. tahu ada extension AI-assistant/translator/
  quiz-answer yang enabled, walau tidak tahu apakah sedang dipakai persis saat
  itu. Lihat [Batasan penting](#batasan-penting-tulis-di-paper).

Tidak lagi hanya lokal — screenshot dan frame webcam diteruskan server ke
layanan analisis eksternal (VLM / CV). Untuk riset/pengujian.

## File

| File | Peran |
|---|---|
| `manifest.json` | Konfigurasi MV3 + permission |
| `content.js` | Berjalan di halaman ujian, tangkap event in-page + capture webcam periodik, kirim ke service worker |
| `background.js` | Service worker: log pusat + sinyal lintas-tab + screenshot + kirim ke server + heartbeat |
| `popup.html` / `popup.js` | Panel log live siswa + tombol Export JSON / Clear |
| `server.js` | Server proctor (Node.js murni, tanpa dependency) — terima event + heartbeat, relay ke VLM/CV |
| `dashboard.html` | Dashboard proctor — lihat semua siswa & log-nya live, status online/offline |
| `vlm_service.py` | Layanan Flask lokal — analisis screenshot pakai VLM (Qwen2.5-VL) |
| `cv_service.py` | Layanan Flask headless — bungkus pipeline `computer_vision/` (head/gaze + YOLO + event scoring) jadi HTTP API, dirancang untuk jalan di Kaggle GPU. Lihat [`kaggle/README.md`](kaggle/README.md). |

Extension ini lintas-browser (memakai alias `browser`/`chrome`). Manifest saat
ini disetel untuk **Firefox/Zen** (`background.scripts`). Untuk Chrome/Edge, ganti
blok `background` menjadi `"service_worker": "background.js"` (lihat bawah).

### Firefox / Zen Browser

1. Buka `about:debugging#/runtime/this-firefox` (di Zen sama).
2. Klik **Load Temporary Add-on…**, pilih file `manifest.json` di folder ini.
3. Klik ikon extension di toolbar untuk membuka **popup log**.
4. **Izinkan akses situs**: buka `about:addons` → extension ini → **Permissions**
   → aktifkan *Access your data for all websites* (dibutuhkan `captureVisibleTab`).
5. Browser akan menampilkan prompt izin untuk **"Manage your apps, extensions,
   and themes"** (permission `management`, dipakai untuk mendeteksi extension
   lain yang terinstall) — ini normal, bukan bug.

> Catatan Firefox:
> - Temporary Add-on **hilang saat browser ditutup** — muat ulang tiap sesi uji.
> - Untuk file exam lokal (`file://...`), akses `file://` terbatas; lebih mudah
>   menyajikan halaman ujian lewat server (mis. `http://localhost`) saat menguji.
> - Alarm heartbeat dibatasi minimum 60 detik oleh Firefox (Chrome ~30 detik);
>   dashboard sudah menoleransinya (`ONLINE_MS = 90s`).

### Chrome / Edge

1. Di `manifest.json`, ganti:
   ```json
   "background": { "scripts": ["background.js"] }
   ```
   menjadi:
   ```json
   "background": { "service_worker": "background.js" }
   ```
2. Buka `chrome://extensions` → aktifkan **Developer mode** → **Load unpacked** →
   pilih folder ini.
3. Untuk file exam lokal, aktifkan **"Allow access to file URLs"** di detail
   extension.

## Menjalankan sisi proctor (server + dashboard)

Alur data lengkap:

```
content.js ─▶ background.js ─POST /events─▶ server.js ─▶ dashboard.html
 (event)      (log + kirim)   (+heartbeat)   (simpan)     (proctor lihat live)
```

1. Jalankan server (butuh Node.js terpasang):

   ```
   node server.js
   ```

   Server jalan di `http://localhost:8787`.
2. Buka `http://localhost:8787` di browser → **Dashboard proctor**.
3. Pastikan extension terpasang (lihat langkah di bawah). Begitu siswa membuka
   halaman ujian / pindah tab, event muncul di dashboard secara live.

Dashboard menampilkan:
- Daftar siswa dengan **titik hijau/merah** (online / OFFLINE = extension mati).
- Log per siswa + **thumbnail screenshot/webcam**.
- Tanda **⚠ SUSPECT** kalau salah satu dari dua sumber berikut terpenuhi (OR,
  bukan AND — satu sumber saja cukup):
  - **Rule-based (keyword)**: URL/detail mengandung kata kunci terlarang
    (chatgpt, google search, whatsapp, dll — ubah daftar `SUSPECT` di
    `dashboard.html`).
  - **CV review score**: `review_score` dari `cv_service.py` mencapai
    `CV_FLAG_THRESHOLD` (default `0.6`, di `server.js`) — dipicu **terlepas**
    dari hasil rule-based di atas.

## Integrasi Computer Vision (review score)

```
content.js ─▶ background.js ─POST /events (type:"webcam")─▶ server.js
                                                                │
                                                    POST /frame ▼
                                              cv_service.py (lokal atau via
                                              tunnel Kaggle, lihat kaggle/README.md)
                                                                │
                                          {review_score, review_level, ...}
                                                                ▼
                                             event.cv + event.cvFlag ─▶ dashboard.html
```

`cv_service.py` membungkus modul `computer_vision/src/integration/`
(`01_head_gaze_adapter.py`, `02_yolo_output_adapter.py`, `03_event_manager.py`,
`05_multi_cue_review_score.py`) — **tidak mengubah file-file itu**, hanya
mengimpornya. Jalankan lokal (`python cv_service.py`, default port `8789`)
atau di Kaggle (lihat [`kaggle/README.md`](kaggle/README.md) untuk GPU
gratis + tunnel `cloudflared`), lalu arahkan `server.js`:

```
CV_URL=https://xxxx.trycloudflare.com node server.js
```

**Penting**: `review_score` adalah indikator prioritisasi eksperimental dari
modul CV — **bukan** probabilitas cheating yang terkalibrasi, dan bukan
vonis otomatis (lihat disclaimer di `computer_vision/src/integration/README.md`
dan `05_multi_cue_review_score.py`). Dashboard menampilkannya sebagai
"review score" / "review level", bukan "cheating probability".

> **Heartbeat**: extension ping server tiap 30 detik. Kalau ping berhenti (siswa
> menonaktifkan extension), titik siswa berubah **merah** dalam ~40 detik. Inilah
> deteksi "monitoring dimatikan" yang dibahas — menangkap kasus *disable*, tapi
> tetap bisa dielakkan siswa canggih (browser lain, device kedua).

## Cara menguji tiap sinyal

- **tab-switch / screenshot** — buka halaman ujian, lalu pindah ke tab lain
  (mis. buka google.com di tab baru). Popup mencatat URL tab tujuan + thumbnail
  screenshot-nya.
- **navigation** — di tab mana pun, ketik URL baru di address bar.
- **focus** — klik aplikasi lain di luar browser (mis. Notepad). Muncul
  "Browser lost focus".
- **idle** — diamkan mouse/keyboard ~15 detik, atau kunci layar.
- **visibility / blur / copy / paste / contextmenu / keyboard / mouseleave /
  mutation** — lakukan di halaman ujian (`#examForm` harus ada agar content
  script aktif).
- **extensions** — enable/disable extension lain apa saja (mis. dark mode
  toggle) lewat `chrome://extensions` / `about:addons`, lalu tunggu sampai
  alarm heartbeat berikutnya (≤30-60 detik) — event baru muncul hanya kalau
  daftar extension aktifnya *berubah* dari check sebelumnya (dedup, bukan
  spam tiap heartbeat).

## Batasan penting (tulis di paper)

- **Bisa dimatikan mahasiswa.** Extension ini bisa di-*disable*/uninstall lewat
  `chrome://extensions`. Agar tepercaya di proctoring nyata, harus
  **force-installed** lewat kebijakan terkelola (`ExtensionInstallForcelist`)
  pada perangkat yang dikelola sekolah/enterprise.
- **content.js hanya aktif di halaman ber-`#examForm`.** Ubah penanda ini agar
  cocok dengan halaman ujian aslimu. Ganti juga `matches` di `manifest.json`
  agar terbatas ke origin ujian (bukan `<all_urls>`) untuk versi produksi.
- **Screenshot butuh `host_permissions`.** `<all_urls>` dipakai agar
  `captureVisibleTab` jalan; gagal (wajar) di halaman `chrome://` dan Web Store.
- **Privasi.** Prototype ini menyimpan screenshot semua tab yang diaktifkan ke
  `chrome.storage.local`. Untuk sistem nyata: batasi kapan capture dilakukan,
  minta consent, dan kirim/olah di server tepercaya — jangan simpan di klien.
- **Deteksi extension lain hanya tahu "terinstall & enabled", bukan "sedang
  dipakai".** Permission `management` cuma kasih tahu extension apa yang
  ada dan aktif di browser itu, bukan aktivitas real-time-nya. Siswa bisa
  disable dulu sebelum ujian lalu enable lagi setelahnya (walau itu sendiri
  masih jadi sinyal kalau dicek berkala). Permission ini juga **tidak
  tersembunyi** — browser menampilkan warning eksplisit ke user saat
  install/update ("Manage your apps, extensions, and themes"), dan tidak
  terlihat sama sekali kalau siswa pakai profile/browser lain.
- **Privasi webcam.** Capture webcam periodik (untuk CV review score) adalah
  perubahan data-handling yang jauh lebih invasif dibanding screenshot tab —
  video wajah siswa dikirim ke layanan pihak ketiga (Kaggle + tunnel). Prompt
  izin kamera bawaan browser **tidak cukup** sebagai informed consent untuk
  proctoring nyata; sebelum dipakai di luar prototipe, perlu ditinjau
  pembimbing riset/etik, dan idealnya ada notice on-page yang eksplisit
  (bukan cuma popup izin OS/browser yang senyap).
