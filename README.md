# 🕺 Tari Saman – AI Pose Coach & Rhythm Trainer

Proyek ini adalah aplikasi interaktif berbasis **Computer Vision (MediaPipe)** untuk membantu mempelajari dan melatih gerakan **Tari Saman**. Sistem ini menggabungkan pelacakan pose (*pose tracking*) secara *real-time* dengan mekanisme ritmis interaktif (mirip game ritme) agar latihan tari menjadi lebih terukur, terarah, dan menyenangkan.

---

## 📌 Gambaran Umum Alur Kerja

Proyek ini terdiri dari **2 bagian utama**:

```
[ Video Gerakan Tari (.mp4) ]
             │
             ▼
   video2template.py  (Ekstraksi Pose & Analisis Kecepatan)
             │
             ├──► *.template.json  (Daftar Pose Unik & Timeline Gerak)
             ├──► *_poses.png      (Preview Stickman Pose)
             ├──► *_timeline.png   (Grafik Kecepatan Gerak)
             └──► *_contact.png    (Snapshot Frame Asli Tiap Gerakan)
             │
             ▼
        index.html    (Web App Interaktif - Real-time Webcam)
             │
             ├─ 1. Mode Belajar  (Kuasai pose satu demi satu)
             ├─ 2. Mode Latihan  (Ikuti timeline ritmis dengan tempo lambat/normal)
             └─ 3. Mode Uji      (Penilaian skor: Perfect, Good, Miss & Highscore)
```

---

## 📂 Struktur Direktori

| File / Folder | Penjelasan |
|---|---|
| `video2template.py` | Skrip Python untuk mengekstrak pose dari video acuan menjadi file template JSON. |
| `index.html` | Aplikasi web frontend (kamera webcam, evaluasi pose *real-time*, dan note lane ritmis). |
| `pose_landmarker_lite.task` | Model bobot biner MediaPipe Pose Landmarker (bisa diunduh otomatis oleh skrip). |
| `Gerakan1_TepukBawah1.*` | Contoh aset video acuan, hasil ekstraksi template JSON, dan grafik visualisasi. |
| `Gerakan2_*.mp4` s/d `Gerakan6_*.mp4` | Rekaman video gerakan-gerakan lanjutan yang siap diproses menjadi template. |
| `.gitignore` | Konfigurasi file yang diabaikan Git (cache landmark besar, video, venv, dll.). |

---

## 🛠️ Panduan Instalasi & Persiapan Lingkungan

### 1. Prasyarat Sistem
- **Python**: Versi 3.9 – 3.11 direkomendasikan.
- **Web Browser**: Google Chrome atau Microsoft Edge (mendukung WebAssembly & WebRTC webcam).
- **Webcam**: Kamera depan yang berfungsi baik dengan sudut pandang cukup luas.

### 2. Setup Lingkungan Python (Virtual Environment)

Buka terminal di root direktori proyek (`e:\lomba`):

```powershell
# 1. Buat virtual environment (jika belum ada)
python -m venv .venv

# 2. Aktifkan virtual environment
# Pada Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Atau Command Prompt (CMD):
.\.venv\Scripts\activate.bat

# 3. Install pustaka yang dibutuhkan
pip install mediapipe opencv-python numpy scipy matplotlib
```

---

## 🚀 Cara Menjalankan

### Bagian 1: Mengekstrak Video Acuan Menjadi Template (`video2template.py`)

Skrip ini membaca video tari, menganalisis arah tulang lengan (*shoulder-axis relative vectors*), mengelompokkan pose unik, serta menentukan waktu pose (*hold* dan *accent*).

#### Perintah Dasar:
```powershell
python video2template.py Gerakan1_TepukBawah1.mp4
```

#### Opsi & Parameter Berguna:
```powershell
# Memotong durasi video tertentu (detik awal s/d detik akhir)
python video2template.py Gerakan2_TepukAtas.mp4 --start 4.0 --end 16.5

# Memakai cache deteksi lama agar proses ulang jauh lebih cepat
python video2template.py Gerakan1_TepukBawah1.mp4 --reuse

# Menggunakan model MediaPipe yang lebih akurat (lite | full | heavy)
python video2template.py Gerakan1_TepukBawah1.mp4 --variant full

# Menyimpan contact sheet (frame cuplikan asli video di setiap pose)
python video2template.py Gerakan1_TepukBawah1.mp4 --snapshots

# Mengatur nama file output khusus
python video2template.py Gerakan1_TepukBawah1.mp4 --out template_gerakan_1
```

> **Output yang Dihasilkan:**
> - `<nama>.template.json`: File konfigurasi pose yang bisa langsung di-*upload* ke web app.
> - `<nama>.landmarks.json`: Cache landmark mentah.
> - `<nama>_poses.png`: Gambar stickman dari setiap pose unik yang terdeteksi.
> - `<nama>_timeline.png`: Grafik kecepatan gerak & posisi note pada timeline.

---

### Bagian 2: Menjalankan Aplikasi Web (`index.html`)

Aplikasi web memerlukan akses kamera dan modul WebAssembly MediaPipe melalui protokol HTTP (tidak bisa langsung klik dua kali / protokol `file:///`).

1. **Jalankan Local HTTP Server:**
   ```powershell
   python -m http.server 8000
   ```
2. **Buka di Browser:**
   Buka alamat [http://localhost:8000](http://localhost:8000) di Chrome atau Edge.
3. **Mulai Latihan:**
   - Klik tombol **"Mulai kamera"** dan izinkan akses webcam.
   - Pastikan posisi duduk tegak dan kedua tangan terlihat jelas.
   - **Gunakan Custom Template:** Klik tombol **"Choose File"** pada bagian *Template dari video*, lalu pilih file `.template.json` (misalnya `Gerakan1_TepukBawah1.template.json`).

---

## 🎮 Mode Latihan di Web App

1. **Mode 1: Belajar (Learn)**
   - Fokus: Memahami dan menguasai bentuk pose statis satu per satu.
   - Cara kerja: Pengguna harus meniru pose stickman sampai garis keselarasan berwarna hijau dan menahannya selama ±0.8 detik untuk lanjut ke langkah berikutnya.
2. **Mode 2: Latihan (Practice)**
   - Fokus: Melatih perpindahan pose mengikuti tempo note yang bergerak horizontal.
   - Fitur: Kecepatan dapat diturunkan ke `0.5×` atau `0.75×` untuk pemanasan.
3. **Mode 3: Uji (Test)**
   - Fokus: Menguji penguasaan tarian pada kecepatan penuh (`1×`).
   - Sistem akan menilai setiap ketukan (*PERFECT*, *GOOD*, *MISS*), memberikan skor akhir (0–100), dan menyimpan skor terbaik di *browser local storage*.

---

## 💡 Konsep Penting & Algoritma (Perlu Diketahui Penerus)

1. **Sumbu Relatif Tubuh (*Shoulder-Axis Body Frame*):**
   - Vektor tulang lengan dinormalisasi terhadap garis kedua bahu (bahu kanan ke bahu kiri).
   - **Kelebihan:** Deteksi tidak bergantung pada jarak pengguna ke kamera atau ukuran tubuh (invarian skala & translasi).
2. **Mode 2D (Default tanpa koordinat Z):**
   - Berdasarkan pengujian, koordinat kedalaman ($z$) dari estimasi webcam monokular sering mengalami *jitter*. Sistem secara default menonaktifkan sumbu $z$ untuk stabilitas sudut lengan.
3. **Penyaringan Kualitas Video:**
   - `video2template.py` otomatis mengecek apakah video layak dijadikan acuan (deteksi sendi $\ge 80\%$, visibilitas tinggi, dan tidak terpotong tepi layar). Jika belum layak, perhatikan saran di konsol terminal.

---

## 📋 Rekomendasi Pengembangan Selanjutnya (To-Do List)

Untuk rekan yang melanjutkan proyek ini, berikut beberapa ide perbaikan yang bisa dikerjakan:

- [ ] **Ekstraksi Video Lain:** Proses video yang tersisa (`Gerakan2` s.d. `Gerakan6`) menjadi file template `.json`.
- [ ] **Dropdown Pemilih Gerakan:** Buat pilihan template langsung via menu *dropdown* di UI tanpa perlu upload manual tiap sesi.
- [ ] **Audio / Metronom Pengiring:** Tambahkan audio ketukan rebana / irama tepukan Saman agar latihan ritme sinkron dengan audio.
- [ ] **Kalibrasi Toleransi Sudut:** Sesuaikan toleransi perhitungan nilai kecocokan di `index.html` (fungsi `sc`) jika ada gerakan yang terasa terlalu sulit atau terlalu mudah.
- [ ] **Validasi Ahli:** Konsultasikan urutan dan penamaan pose dengan pelatih/penari Tari Saman.

---

## 🤝 Bantuan & Catatan Tambahan

- Jika kamera tidak mau menyala di browser, pastikan URL menggunakan `http://localhost:8000` (bukan IP publik atau file path langsung) dan izin kamera telah diaktifkan di setelan situs browser.
- Jangan ragu untuk melihat komentar di dalam `video2template.py` dan `index.html` untuk memahami detail implementasi fungsi matematika dan visualisasi canvas.

*Selamat berkarya dan semoga sukses lombanya!* 🚀🔥

