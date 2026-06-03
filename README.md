# Restorasi Citra Foto Jadul Berwarna Menggunakan Koreksi Warna dan CLAHE

---

## Deskripsi

Program ini melakukan restorasi foto jadul berwarna yang mengalami dua
permasalahan utama secara bersamaan:

1. **Color fading** — pemudaran warna ke arah kemerahan akibat
   degradasi kimiawi pigmen foto.
2. **Kontras rendah** — detail pada area gelap maupun terang sulit
   dibedakan.

Pendekatan yang digunakan adalah pipeline dua tahap berbasis teknik
pemrosesan citra klasik:

| Tahap | Metode | Tujuan |
|-------|--------|--------|
| 1 | Gray World Assumption | Menetralisir dominasi warna kemerahan |
| 2 | CLAHE (pada channel L LAB) | Meningkatkan kontras tanpa mendistorsi warna |

Program ini **tidak menggunakan library deep learning** (TensorFlow,
PyTorch, Keras, dll.) sesuai ketentuan tugas.

---

## Struktur Folder

```
old-image-enhancement/
├── main.py             # Script utama
├── requirements.txt    # Daftar library yang dibutuhkan
├── README.md           # File ini
├── images/             # Folder untuk foto input
│   └── gambar_jadul.jpg
└── output/             # Folder hasil (dibuat otomatis)
    ├── hasil_tahap1_koreksi_warna.jpg
    ├── hasil_tahap2_clahe_final.jpg
    └── hasil_perbandingan.png
```

---

## Persyaratan Sistem

- Python 3.8 atau lebih baru
- pip (Python package manager)

---

## Instalasi

### 1. (Opsional) Buat Virtual Environment

```bash
python -m venv venv

# Aktifkan (Windows)
venv\Scripts\activate

# Aktifkan (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan

### Perintah Dasar

```bash
python main.py images/gambar_jadul.jpg
```

### Dengan Parameter Custom

```bash
python main.py images/gambar_jadul.jpg --output_dir output --clip_limit 2.0 --tile_size 8
```

### Parameter yang Tersedia

| Parameter      | Default  | Keterangan |
|----------------|----------|------------|
| `input`        | (wajib)  | Path ke foto input |
| `--output_dir` | `output` | Folder penyimpanan hasil |
| `--clip_limit` | `2.0`    | Batas amplifikasi kontras CLAHE |
| `--tile_size`  | `8`      | Ukuran blok CLAHE dalam piksel |

---

## Output yang Dihasilkan

### File Gambar

- `hasil_tahap1_koreksi_warna.jpg` — Foto setelah koreksi warna Gray World
- `hasil_tahap2_clahe_final.jpg` — Foto final setelah CLAHE diterapkan
- `hasil_perbandingan.png` — Visualisasi 2x3 (citra + histogram channel L)

### Output Terminal

Program mencetak tabel metrik kuantitatif, contoh:

```
======================================================================
                          METRIK KUANTITATIF
======================================================================
Metrik                          Original   Koreksi Warna       Final
----------------------------------------------------------------------
Std Dev Channel L                  39.16           41.68       52.00
Color Cast Index                   42.41            0.02        0.36
PSNR vs Original (dB)                  —           22.65       19.86
======================================================================
```

---

## Catatan Teknis

- **Ruang warna LAB** digunakan agar CLAHE hanya memodifikasi
  kecerahan (channel L), bukan warna (channel a dan b).
- **Gray World Assumption** mengasumsikan bahwa rata-rata warna pada
  citra natural seharusnya netral (abu-abu).
- Rumus skala per channel:
  `scale_c = gray_mean / mean_c` untuk `c` dalam `{B, G, R}`
- Rumus clip limit per tile CLAHE:
  `C_l = clip_limit * (T * T) / L`

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `FileNotFoundError: Gambar tidak ditemukan` | Cek kembali path gambar input |
| `ModuleNotFoundError: No module named 'cv2'` | Jalankan `pip install -r requirements.txt` |
| Hasil terlalu terang/gelap | Sesuaikan `--clip_limit` (kecil = subtle, besar = agresif) |
