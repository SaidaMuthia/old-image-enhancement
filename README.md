# Restorasi Citra Foto Jadul Berwarna Menggunakan Koreksi Warna dan CLAHE

---

## Deskripsi

Program ini melakukan restorasi foto jadul berwarna yang mengalami dua permasalahan utama:

1. **Color fading** — pemudaran warna ke arah kemerahan akibat degradasi kimiawi pigmen foto.
2. **Kontras rendah** — detail pada area gelap maupun terang sulit dibedakan.

Pipeline dua tahap berbasis teknik pemrosesan citra klasik:

| Tahap | Metode | Tujuan |
|-------|--------|--------|
| 1 | Gray World Assumption | Menetralisir dominasi warna kemerahan |
| 2 | CLAHE pada channel L (CIE L\*a\*b\*) | Meningkatkan kontras tanpa mendistorsi warna |

CLAHE diterapkan pada **channel L** (kecerahan) dalam ruang warna LAB — bukan pada grayscale
maupun langsung pada channel RGB — agar informasi warna pada channel a dan b tetap terjaga
setelah konversi balik ke RGB.

Seluruh algoritma diimplementasi menggunakan NumPy tanpa library pemrosesan citra khusus.

---

## Struktur Folder

```
old-image-enhancement/
├── main.py                     # Pipeline restorasi utama
├── eksperimen_parameter.py     # Eksperimen variasi parameter CLAHE
├── requirements.txt
├── README.md
├── images/
│   └── gambar_jadul.jpg
└── output/                     # Dihasilkan saat program dijalankan
    ├── hasil_tahap1_koreksi_warna.jpg
    ├── hasil_tahap2_clahe_final.jpg
    ├── hasil_perbandingan.png
    ├── eksperimen_clip_limit.png
    └── eksperimen_tile_size.png
```

---

## Persyaratan

- Python 3.8 atau lebih baru

---

## Instalasi

```bash
# (Opsional) buat virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

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
python main.py images/gambar_jadul.jpg --clip_limit 2.0 --tile_size 8
```

| Parameter      | Default  | Keterangan |
|----------------|----------|------------|
| `input`        | (wajib)  | Path ke foto input |
| `--output_dir` | `output` | Folder penyimpanan hasil |
| `--clip_limit` | `2.0`    | Batas amplifikasi kontras CLAHE per tile |
| `--tile_size`  | `8`      | Jumlah grid tile CLAHE per sisi (8 = grid 8×8) |

---

## Eksperimen Variasi Parameter

Untuk memvalidasi pemilihan `clip_limit=2.0` dan `tile_size=8`:

```bash
python eksperimen_parameter.py images/gambar_jadul.jpg
```

Output:
- Tabel analisis per variasi parameter di terminal
- `output/eksperimen_clip_limit.png`
- `output/eksperimen_tile_size.png`

---

## Output Utama

| File | Deskripsi |
|------|-----------|
| `hasil_tahap1_koreksi_warna.jpg` | Hasil setelah Gray World Assumption |
| `hasil_tahap2_clahe_final.jpg`   | Hasil final setelah CLAHE |
| `hasil_perbandingan.png`         | Visualisasi 2×3 dengan histogram channel L |

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `FileNotFoundError` | Periksa path gambar input |
| `ModuleNotFoundError` | Jalankan `pip install -r requirements.txt` |
| Hasil terlalu terang/gelap | Turunkan/naikkan `--clip_limit` |
