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

Hanya lokal. Belum ada server. Untuk riset/pengujian.

## File

| File | Peran |
|---|---|
| `manifest.json` | Konfigurasi MV3 + permission |
| `content.js` | Berjalan di halaman ujian, tangkap event in-page, kirim ke service worker |
| `background.js` | Service worker: log pusat + sinyal lintas-tab + screenshot + kirim ke server + heartbeat |
| `popup.html` / `popup.js` | Panel log live siswa + tombol Export JSON / Clear |
| `server.js` | Server proctor (Node.js murni, tanpa dependency) — terima event + heartbeat |
| `dashboard.html` | Dashboard proctor — lihat semua siswa & log-nya live, status online/offline |

Extension ini lintas-browser (memakai alias `browser`/`chrome`). Manifest saat
ini disetel untuk **Firefox/Zen** (`background.scripts`). Untuk Chrome/Edge, ganti
blok `background` menjadi `"service_worker": "background.js"` (lihat bawah).

### Firefox / Zen Browser

1. Buka `about:debugging#/runtime/this-firefox` (di Zen sama).
2. Klik **Load Temporary Add-on…**, pilih file `manifest.json` di folder ini.
3. Klik ikon extension di toolbar untuk membuka **popup log**.
4. **Izinkan akses situs**: buka `about:addons` → extension ini → **Permissions**
   → aktifkan *Access your data for all websites* (dibutuhkan `captureVisibleTab`).

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
- Log per siswa + **thumbnail screenshot**.
- Tanda **⚠ SUSPECT** otomatis kalau URL/detail mengandung kata kunci terlarang
  (chatgpt, google search, whatsapp, dll — ubah daftar `SUSPECT` di
  `dashboard.html`).

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
