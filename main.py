"""
Peningkatan Kualitas Foto Jadul Menggunakan Histogram Equalization dan CLAHE
============================================================================
Mata Kuliah  : Visi Komputer
Metode       : Classical Image Processing (tanpa Deep Learning)
Library      : OpenCV, NumPy, Matplotlib

Deskripsi:
    Script ini memproses foto jadul berwarna yang mengalami color fading
    (pemudaran warna ke arah kemerahan) dan kontras rendah menggunakan
    dua metode klasik:
      1. Histogram Equalization (HE)
      2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Pemrosesan dilakukan pada channel L dalam ruang warna LAB agar
    peningkatan kontras tidak mendistorsi warna asli foto.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import argparse


# ─────────────────────────────────────────────
#  FUNGSI UTAMA PEMROSESAN
# ─────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Membaca citra dari path yang diberikan."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    return img


def apply_he(image_bgr: np.ndarray) -> np.ndarray:
    """
    Menerapkan Histogram Equalization (HE) global pada channel L
    di ruang warna LAB.

    Langkah:
      1. Konversi BGR -> LAB
      2. Equalize channel L dengan cv2.equalizeHist()
      3. Gabung kembali channel L, a, b
      4. Konversi LAB -> BGR

    Returns:
        np.ndarray: Citra BGR hasil HE
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # HE global: s = round((L-1) * CDF(r))
    l_eq = cv2.equalizeHist(l)

    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return result


def apply_clahe(image_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Menerapkan CLAHE pada channel L di ruang warna LAB.

    Parameter clip limit menentukan batas amplifikasi kontras per tile.
    Rumus batas histogram per tile:
        C_l = clip_limit * (T * T) / L
    di mana T = ukuran tile, L = 256.

    Args:
        clip_limit     : batas amplifikasi kontras (default=2.0)
        tile_grid_size : ukuran blok/tile (default=(8,8))

    Returns:
        np.ndarray: Citra BGR hasil CLAHE
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return result


# ─────────────────────────────────────────────
#  FUNGSI METRIK KUANTITATIF
# ─────────────────────────────────────────────

def compute_histogram_std(image_bgr: np.ndarray) -> float:
    """
    Menghitung standar deviasi histogram channel L (LAB).
    Nilai lebih tinggi = distribusi intensitas lebih merata = kontras lebih baik.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    hist = cv2.calcHist([l], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    mean = np.sum(np.arange(256) * hist_norm)
    std = np.sqrt(np.sum(((np.arange(256) - mean) ** 2) * hist_norm))
    return float(std)


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Menghitung PSNR (Peak Signal-to-Noise Ratio) antara dua citra.
    
    Rumus: PSNR = 10 * log10(MAX^2 / MSE)
    di mana MAX = 255 (nilai maksimum piksel 8-bit)
    dan MSE = Mean Squared Error antar dua citra.

    Nilai PSNR lebih tinggi menunjukkan perubahan yang lebih halus
    (lebih dekat ke asli). Nilai lebih rendah menunjukkan perubahan
    yang lebih besar dari citra asli.
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10((255.0 ** 2) / mse)


def print_metrics(original: np.ndarray,
                  he_result: np.ndarray,
                  clahe_result: np.ndarray) -> None:
    """Mencetak tabel metrik kuantitatif ke terminal."""
    std_ori   = compute_histogram_std(original)
    std_he    = compute_histogram_std(he_result)
    std_clahe = compute_histogram_std(clahe_result)

    psnr_he    = compute_psnr(original, he_result)
    psnr_clahe = compute_psnr(original, clahe_result)

    print("\n" + "=" * 55)
    print(f"{'METRIK KUANTITATIF':^55}")
    print("=" * 55)
    print(f"{'Metrik':<30} {'HE':>10} {'CLAHE':>12}")
    print("-" * 55)
    print(f"{'Std Dev Histogram (L)':<30} {std_he:>10.2f} {std_clahe:>12.2f}")
    print(f"{'PSNR vs Original (dB)':<30} {psnr_he:>10.2f} {psnr_clahe:>12.2f}")
    print("=" * 55)
    print(f"  Std Dev Original : {std_ori:.2f}")
    print("  [Std Dev lebih tinggi = kontras lebih baik]")
    print("  [PSNR lebih tinggi = perubahan lebih halus dari asli]")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
#  FUNGSI VISUALISASI
# ─────────────────────────────────────────────

def get_l_channel(image_bgr: np.ndarray) -> np.ndarray:
    """Mengambil channel L dari citra BGR."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    return l


def plot_histogram(ax, image_bgr: np.ndarray, title: str, color: str = "steelblue") -> None:
    """Menggambar histogram channel L pada axes yang diberikan."""
    l = get_l_channel(image_bgr)
    hist = cv2.calcHist([l], [0], None, [256], [0, 256]).flatten()
    ax.bar(range(256), hist, color=color, alpha=0.75, width=1.0)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlim([0, 255])
    ax.set_xlabel("Nilai Intensitas (0-255)", fontsize=7)
    ax.set_ylabel("Frekuensi Piksel", fontsize=7)
    ax.tick_params(labelsize=7)


def save_comparison_figure(original: np.ndarray,
                           he_result: np.ndarray,
                           clahe_result: np.ndarray,
                           output_dir: str) -> None:
    """
    Menyimpan gambar perbandingan 2x3:
      Baris 1: Citra asli | HE | CLAHE
      Baris 2: Histogram asli | Histogram HE | Histogram CLAHE
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Perbandingan Peningkatan Kualitas Foto Jadul\n"
        "Histogram Equalization (HE) vs CLAHE",
        fontsize=12, fontweight="bold", y=0.98
    )

    images = [
        (original,    "Original (Asli)",           "coral"),
        (he_result,   "Hasil HE",                  "steelblue"),
        (clahe_result,"Hasil CLAHE",               "seagreen"),
    ]

    for col, (img, title, hcolor) in enumerate(images):
        # Baris atas: tampilkan citra (konversi BGR->RGB untuk matplotlib)
        axes[0, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=6)
        axes[0, col].axis("off")

        # Baris bawah: tampilkan histogram channel L
        plot_histogram(axes[1, col], img, f"Histogram L — {title}", color=hcolor)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(output_dir, "hasil_perbandingan.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gambar perbandingan disimpan ke: {out_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Peningkatan kualitas foto jadul menggunakan HE dan CLAHE"
    )
    parser.add_argument(
        "input",
        help="Path ke foto jadul berwarna (contoh: images/gambar_jadul.jpg)"
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Direktori penyimpanan hasil (default: output/)"
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (default: 2.0)"
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=8,
        help="Ukuran tile CLAHE dalam piksel (default: 8)"
    )
    args = parser.parse_args()

    # Buat direktori output jika belum ada
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load gambar
    print(f"\n[1/5] Membaca gambar: {args.input}")
    original = load_image(args.input)
    print(f"      Ukuran: {original.shape[1]} x {original.shape[0]} piksel")

    # 2. Terapkan HE
    print("[2/5] Menerapkan Histogram Equalization (HE)...")
    he_result = apply_he(original)

    # 3. Terapkan CLAHE
    tile = (args.tile_size, args.tile_size)
    print(f"[3/5] Menerapkan CLAHE (clipLimit={args.clip_limit}, tileGridSize={tile})...")
    clahe_result = apply_clahe(original, clip_limit=args.clip_limit, tile_grid_size=tile)

    # 4. Simpan hasil citra individual
    print("[4/5] Menyimpan citra hasil...")
    cv2.imwrite(os.path.join(args.output_dir, "hasil_he.jpg"),    he_result)
    cv2.imwrite(os.path.join(args.output_dir, "hasil_clahe.jpg"), clahe_result)
    print(f"      [OK] hasil_he.jpg    -> {args.output_dir}/")
    print(f"      [OK] hasil_clahe.jpg -> {args.output_dir}/")

    # 5. Simpan gambar perbandingan + tampilkan metrik
    print("[5/5] Membuat visualisasi perbandingan dan menghitung metrik...")
    save_comparison_figure(original, he_result, clahe_result, args.output_dir)
    print_metrics(original, he_result, clahe_result)

    print("Selesai! Cek folder output/ untuk melihat semua hasil.\n")


if __name__ == "__main__":
    main()
