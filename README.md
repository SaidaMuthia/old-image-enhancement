# Peningkatan Citra Foto Jadul Berwarna Menggunakan Koreksi Warna dan CLAHE

---

## Deskripsi

Program ini melakukan restorasi foto jadul berwarna yang mengalami dua
permasalahan utama secara bersamaan:

1. **Color fading** — pemudaran warna ke arah kemerahan akibat
   degradasi kimiawi pigmen foto.
2. **Kontras rendah** — detail pada area gelap maupun terang sulit
   dibedakan.

Pipeline dua tahap berbasis teknik pemrosesan citra klasik:

| Tahap | Metode | Tujuan |
|-------|--------|--------|
| 1 | Gray World Assumption | Menetralisir dominasi warna kemerahan |
| 2 | CLAHE pada channel L (CIE L\*a\*b\*) | Meningkatkan kontras tanpa mendistorsi warna |

Seluruh algoritma diimplementasi menggunakan NumPy tanpa library
pemrosesan citra khusus. Program **tidak menggunakan** OpenCV, TensorFlow,
PyTorch, Keras, atau library deep learning lainnya.

---

## Struktur Folder

```
old-image-enhancement/
├── main.py                    # Pipeline restorasi utama
├── eksperimen_parameter.py    # Eksperimen variasi parameter CLAHE
├── requirements.txt           # Daftar library yang dibutuhkan
├── README.md                  # File ini
├── images/
│   └── gambar_jadul.jpg
└── output/
    ├── hasil_tahap1_koreksi_warna.jpg
    ├── hasil_tahap2_clahe_final.jpg
    ├── hasil_perbandingan.png
    ├── eksperimen_clip_limit.png
    └── eksperimen_tile_size.png
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

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan

### Pipeline Restorasi Utama

```bash
python main.py images/gambar_jadul.jpg
```

### Dengan Parameter Custom

```bash
python main.py images/gambar_jadul.jpg --output_dir output --clip_limit 2.0 --tile_size 8
```

### Parameter

| Parameter      | Default  | Keterangan |
|----------------|----------|------------|
| `input`        | (wajib)  | Path ke foto input |
| `--output_dir` | `output` | Folder penyimpanan hasil |
| `--clip_limit` | `2.0`    | Batas amplifikasi kontras CLAHE |
| `--tile_size`  | `8`      | Jumlah grid tile CLAHE per sisi |

---

## Menjalankan Eksperimen Parameter

Script `eksperimen_parameter.py` menguji berbagai nilai `clip_limit` dan
`tile_size` untuk memvalidasi pemilihan parameter default.

```bash
python eksperimen_parameter.py images/gambar_jadul.jpg
```

Output yang dihasilkan:
- Tabel metrik di terminal untuk tiap variasi parameter
- `output/eksperimen_clip_limit.png`
- `output/eksperimen_tile_size.png`

---

## Output

- `hasil_tahap1_koreksi_warna.jpg` — Setelah koreksi warna Gray World
- `hasil_tahap2_clahe_final.jpg` — Setelah CLAHE (hasil final)
- `hasil_perbandingan.png` — Visualisasi 2x3 beserta histogram channel L

---

## Catatan Teknis

- Konversi ruang warna sRGB ke CIE L\*a\*b\* dan sebaliknya diimplementasi
  manual menggunakan transformasi matriks RGB-XYZ (illuminan D65) dan
  fungsi non-linear CIE.
- CLAHE diimplementasi manual: histogram per tile, clip limit, redistribusi
  piksel terpotong, ekualisasi berbasis CDF, dan interpolasi bilinear antar tile.
- Gray World Assumption diimplementasi dengan operasi NumPy dasar.

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `FileNotFoundError` | Cek kembali path gambar input |
| `ModuleNotFoundError` | Jalankan `pip install -r requirements.txt` |
| Hasil terlalu terang/gelap | Sesuaikan `--clip_limit` |
