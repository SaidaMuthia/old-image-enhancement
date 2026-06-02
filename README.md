# Peningkatan Kualitas Foto Jadul Menggunakan Histogram Equalization (HE) vs CLAHE

---

## Deskripsi Singkat

Script ini membandingkan dua metode klasik peningkatan kontras citra untuk
merestorasi foto jadul berwarna yang mengalami:
- **Color fading** (warna memudar ke arah kemerahan)
- **Kontras rendah** (detail sulit dibedakan)

**Metode yang digunakan:**
1. **Histogram Equalization (HE)** — pemerataan distribusi intensitas secara global
2. **CLAHE** — HE adaptif per blok dengan pembatasan amplifikasi kontras

Pemrosesan dilakukan pada **channel L (Lightness)** di ruang warna LAB
agar warna asli foto tidak terdistorsi.

> ⚠️ Tidak menggunakan library deep learning (TensorFlow, PyTorch, Keras, dll.)

---

## Struktur Folder

```
project_visi_komputer/
├── main.py            # Script utama
├── requirements.txt   # Daftar library yang dibutuhkan
├── README.md          # File ini
└── images/            # Letakkan foto input di sini
    └── gambar_jadul.jpg   # (contoh nama file input)
```

Setelah dijalankan, hasil akan tersimpan di folder `output/`:
```
output/
├── hasil_he.jpg             # Foto hasil Histogram Equalization
├── hasil_clahe.jpg          # Foto hasil CLAHE
└── hasil_perbandingan.png   # Visualisasi perbandingan + histogram
```

---

## Persyaratan Sistem

- Python 3.8 atau lebih baru
- pip (Python package manager)

---

## Cara Instalasi

### 1. Clone / Download Project

Pastikan semua file project sudah ada di satu folder.

### 2. (Opsional) Buat Virtual Environment

```bash
python -m venv venv

# Aktifkan (Windows):
venv\Scripts\activate

# Aktifkan (Mac/Linux):
source venv/bin/activate
```

### 3. Install Library

```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan

### Format Perintah

```bash
python main.py <path_gambar> [--output_dir <folder>] [--clip_limit <nilai>] [--tile_size <nilai>]
```

### Contoh Paling Sederhana

Letakkan foto jadul di folder `images/`, lalu jalankan:

```bash
python main.py images/foto_gua.jpg
```

Hasil akan tersimpan otomatis di folder `output/`.

### Contoh dengan Parameter Custom

```bash
python main.py images/foto_gua.jpg --output_dir hasil_saya --clip_limit 3.0 --tile_size 16
```

### Penjelasan Parameter

| Parameter       | Default  | Keterangan                                              |
|-----------------|----------|---------------------------------------------------------|
| `input`         | (wajib)  | Path ke foto input                                      |
| `--output_dir`  | `output` | Folder penyimpanan hasil                                |
| `--clip_limit`  | `2.0`    | Batas amplifikasi kontras CLAHE (nilai lebih tinggi = lebih agresif) |
| `--tile_size`   | `8`      | Ukuran blok tile CLAHE dalam piksel (harus pembagi lebar/tinggi gambar) |

---

## Output yang Dihasilkan

### File Gambar
- `hasil_he.jpg` — Foto hasil Histogram Equalization
- `hasil_clahe.jpg` — Foto hasil CLAHE
- `hasil_perbandingan.png` — Visualisasi 2x3 berisi:
  - Baris atas: foto asli | hasil HE | hasil CLAHE
  - Baris bawah: histogram channel L masing-masing

### Output Terminal

Script akan mencetak tabel metrik kuantitatif, contoh:

```
=======================================================
               METRIK KUANTITATIF
=======================================================
Metrik                              HE          CLAHE
-------------------------------------------------------
Std Dev Histogram (L)            67.45         54.32
PSNR vs Original (dB)            18.23         22.17
=======================================================
  Std Dev Original : 28.11
  [Std Dev lebih tinggi = kontras lebih baik]
  [PSNR lebih tinggi = perubahan lebih halus dari asli]
=======================================================
```

---

## Catatan Teknis

- Pemrosesan menggunakan ruang warna **LAB** agar channel warna (a, b)
  tidak diubah — hanya channel **L (luminance)** yang diproses.
- HE menggunakan rumus: `s = round((L-1) × CDF(r))`
- CLAHE menggunakan clip limit per tile: `C_l = clip_limit × (T×T) / 256`
- Tidak ada dependency deep learning sama sekali.

---

## Troubleshooting

| Masalah | Solusi |
|--------|--------|
| `FileNotFoundError: Gambar tidak ditemukan` | Pastikan path gambar benar dan file ada |
| `ModuleNotFoundError: No module named 'cv2'` | Jalankan `pip install opencv-python` |
| Hasil terlalu terang/gelap | Turunkan/naikkan `--clip_limit` |
| Gambar output tampak berbeda dari preview | Normal — matplotlib menampilkan RGB, file disimpan BGR (OpenCV) |
