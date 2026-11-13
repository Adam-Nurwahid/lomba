# 🕺 Tari Saman – AI Pose Coach & Rhythm Trainer

Aplikasi interaktif berbasis **Computer Vision (MediaPipe)** untuk membantu mempelajari dan melatih gerakan **Tari Saman**. Sistem ini menggabungkan pelacakan pose (*pose tracking*) secara *real-time* berbasis sumbu bahu (*shoulder-axis body frame*) dengan mekanisme ritmis interaktif (mirip game ritme) agar latihan tari menjadi lebih terukur, terarah, dan menyenangkan.

---

## 📌 Gambaran Umum Alur Kerja End-to-End

```
[ Video Gerakan Tari (.mp4) ]
   (dari sanggar-digital/public/videos/saman)
             │
             ▼
    video2template.py / batch_extract.py  (Ekstraksi Pose & Analisis Kecepatan)
             │
             ├──► *.template.json  (Daftar Pose Unik & Timeline Gerak)
             ├──► *_poses.png      (Preview Stickman Pose)
             ├──► *_timeline.png   (Grafik Kecepatan Gerak)
             └──► *_contact.png    (Snapshot Frame Asli Tiap Gerakan)
             │
             ▼
        index.html    (Web App Interaktif - Real-time Webcam)
             │
             ├─ Preset Selector    (Pilih langsung dari dropdown tanpa upload manual)
             ├─ Sinkronisasi Video (Tonton video penari asli berdampingan dengan skeleton)
             ├─ Audio SFX          (Umpan balik suara interaktif untuk Perfect/Good/Miss)
             ├─ 1. Mode Belajar    (Kuasai pose satu per satu + navigasi tombol Prev/Next)
             ├─ 2. Mode Latihan    (Ikuti timeline ritmis dengan tempo 0.5× / 0.75× / 1.0×)
             └─ 3. Mode Uji        (Penilaian akurasi skor & penyimpanan Highscore)
```

---

## 📂 Struktur Direktori

| File / Folder | Penjelasan |
|---|---|
| `video2template.py` | Skrip Python untuk mengekstrak pose dari video acuan menjadi template JSON, gambar pose, dan grafik timeline. |
| `batch_extract.py` | Skrip praktis untuk mengekstrak seluruh video sample secara berurutan. |
| `index.html` | Aplikasi web frontend modern (kamera webcam, evaluasi pose *real-time*, note lane ritmis, audio SFX, dan pemutar video acuan). |
| `pose_landmarker_lite.task` | Model bobot biner MediaPipe Pose Landmarker lokal. |
| `videos/` | Symlink ke direktori video sample tari (`project/sanggar-digital/public/videos`). |
| `*.template.json` | File template pose hasil ekstraksi dari video acuan (`saman-solo-candidate`, `short-1-saman`, dll.). |
| `*_contact.png` | Snapshot contact sheet frame video penari asli di setiap pose gerakan. |
| `*_poses.png` | Gambar diagram stickman dari setiap pose unik yang terdeteksi. |
| `*_timeline.png` | Grafik profil kecepatan sudut lengan dan letak penanda ketukan. |

---

## 🛠️ Panduan Instalasi & Persiapan Lingkungan

### 1. Prasyarat Sistem
- **Python**: Versi 3.9 – 3.11 direkomendasikan.
- **Web Browser**: Google Chrome atau Microsoft Edge (mendukung WebAssembly & WebRTC webcam).
- **Webcam**: Kamera depan yang berfungsi baik dengan sudut pandang cukup luas.

### 2. Setup Lingkungan Python (Virtual Environment)

Menggunakan `uv` (cepat) atau `venv` standar:

```bash
# Menggunakan uv (direkomendasikan):
uv venv .venv --python 3.11
uv pip install --python .venv mediapipe opencv-python numpy scipy matplotlib

# Atau menggunakan python standar:
python3 -m venv .venv
source .venv/bin/activate
pip install mediapipe opencv-python numpy scipy matplotlib
```

---

## 🚀 Cara Menjalankan

### Bagian 1: Mengekstrak Video Tari Menjadi Template

#### Opsi A: Ekstraksi Otomatis Seluruh Sample Video
```bash
.venv/bin/python batch_extract.py
```

#### Opsi B: Ekstraksi Satu Video Khusus
Arahkan path ke video yang ada di `videos/saman/` atau `project/sanggar-digital`:
```bash
# Contoh 1: Saman Solo
.venv/bin/python video2template.py "videos/saman/saman-solo-candidate.mp4" --snapshots

# Contoh 2: Saman Dasar (Short 1)
.venv/bin/python video2template.py "videos/saman/short-1-saman.mp4" --snapshots

# Contoh 3: Saman Kostum Duduk
.venv/bin/python video2template.py "videos/saman/saman-costume-candidate.mp4" --snapshots
```

> **Parameter Tambahan:**
> - `--start X --end Y`: Memotong durasi detik awal dan akhir.
> - `--reuse`: Memakai cache landmark jika sudah pernah diekstrak sebelumnya.
> - `--snapshots`: Menyimpan frame cuplikan asli video di setiap pose (`*_contact.png`).

---

### Bagian 2: Menjalankan Aplikasi Web (`index.html`)

Aplikasi web memerlukan akses kamera dan modul WebAssembly melalui HTTP:

1. **Jalankan Local HTTP Server:**
   ```bash
   .venv/bin/python -m http.server 8000
   ```
2. **Buka di Browser:**
   Akses [http://localhost:8000](http://localhost:8000).
3. **Mulai Latihan:**
   - Tekan **"Mulai Kamera"** dan izinkan akses webcam.
   - Pilih gerakan pada dropdown **"Pilih Gerakan"** (tersedia berbagai preset template yang sudah diekstrak).
   - Atur tampilan referensi di panel kanan: **Skeleton**, **Video Asli**, atau **Kombinasi**.

---

## 🎮 Fitur & Mode Latihan di Web App

1. **Mode 1: Belajar (Step-by-Step)**
   - Kuasai pose satu demi satu secara bertahap.
   - Dilengkapi tombol **◄ Prev** dan **Next ►** untuk menavigasi pose.
   - Pengguna meniru pose target hingga indikator akurasi mencapai $\ge 70\%$ dan menahannya selama ±0.8 detik.
   - Dilengkapi petunjuk koreksi posisi lengan mana yang perlu disesuaikan beserta derajat selisihnya.

2. **Mode 2: Latihan (Tempo)**
   - Melatih koordinasi transisi pose mengikuti kartu ritmis yang berjalan di Note Lane.
   - Tempo dapat diatur ke **0.5× (Lambat)**, **0.75× (Sedang)**, atau **1.0× (Normal)**.
   - Video penari asli dapat diputar sinkron dengan ketukan ritme.

3. **Mode 3: Uji (Skor & Highscore)**
   - Ujian penguasaan gerakan tarian pada tempo penuh (1.0×).
   - Penilaian interaktif: **PERFECT**, **GOOD**, dan **MISS** lengkap dengan efek suara sintetis (Web Audio API).
   - Skor akhir (0–100) dan predikat pencapaian otomatis tersimpan di *browser local storage*.

---

## 💡 Konsep Algoritma

1. **Sumbu Relatif Bahu (*Shoulder-Axis Body Frame*):**
   - Vektor tulang dinormalisasi terhadap garis horizontal kedua bahu. Skala dan translasi tubuh pengguna diabaikan sehingga evaluasi pose tetap stabil di berbagai jarak dan ukuran tubuh.
2. **Mode 2D Stabil:**
   - Menghilangkan estimasi kedalaman ($z$) yang rentan *jitter* pada webcam monokular tunggal.
3. **Deteksi Puncak & Hold:**
   - Gerakan diam terdeteksi sebagai `hold`, sementara perpindahan cepat dengan titik balik kecepatan terdeteksi sebagai `accent`.
