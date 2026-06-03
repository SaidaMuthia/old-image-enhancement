"""
Restorasi Citra Foto Jadul Berwarna Menggunakan Koreksi Warna dan CLAHE
=======================================================================
Mata Kuliah  : Visi Komputer
Metode       : Classical Image Processing (tanpa Deep Learning)
Library      : OpenCV, NumPy, Matplotlib

Pipeline:
    1. Koreksi warna dengan Gray World Assumption
       untuk menetralisir color cast (dominasi warna kemerahan).
    2. Peningkatan kontras dengan CLAHE pada channel L (LAB)
       untuk memunculkan detail tanpa mendistorsi warna.
"""

import argparse
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Membaca citra dari path. Mengembalikan array BGR (OpenCV)."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    return image


# ---------------------------------------------------------------------------
# Tahap 1: Koreksi Warna (Gray World Assumption)
# ---------------------------------------------------------------------------

def gray_world_correction(image_bgr: np.ndarray) -> np.ndarray:
    """
    Menerapkan Gray World Assumption untuk menetralisir color cast.

    Asumsi metode ini: rata-rata warna pada citra natural seharusnya
    bersifat netral (abu-abu). Dengan menghitung rata-rata tiap channel
    (B, G, R) lalu menyesuaikannya terhadap rata-rata global, dominasi
    warna pada salah satu channel dapat dikurangi.

    Rumus:
        scale_c = gray_mean / mean_c        untuk c in {B, G, R}
        I_c_new = I_c * scale_c
    di mana gray_mean = (mean_B + mean_G + mean_R) / 3.

    Returns:
        np.ndarray: Citra BGR hasil koreksi warna (uint8).
    """
    image_float = image_bgr.astype(np.float32)

    mean_b = np.mean(image_float[:, :, 0])
    mean_g = np.mean(image_float[:, :, 1])
    mean_r = np.mean(image_float[:, :, 2])
    gray_mean = (mean_b + mean_g + mean_r) / 3.0

    scale_b = gray_mean / mean_b if mean_b > 0 else 1.0
    scale_g = gray_mean / mean_g if mean_g > 0 else 1.0
    scale_r = gray_mean / mean_r if mean_r > 0 else 1.0

    corrected = np.empty_like(image_float)
    corrected[:, :, 0] = image_float[:, :, 0] * scale_b
    corrected[:, :, 1] = image_float[:, :, 1] * scale_g
    corrected[:, :, 2] = image_float[:, :, 2] * scale_r

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected


# ---------------------------------------------------------------------------
# Tahap 2: Peningkatan Kontras (CLAHE pada channel L)
# ---------------------------------------------------------------------------

def apply_clahe_on_l_channel(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Menerapkan CLAHE pada channel L dalam ruang warna LAB.

    Channel L (lightness) memuat informasi kecerahan, sedangkan channel
    a dan b memuat informasi warna. Dengan memproses hanya channel L,
    peningkatan kontras dilakukan tanpa mendistorsi warna hasil koreksi.

    Args:
        clip_limit     : Batas amplifikasi kontras per tile.
        tile_grid_size : Ukuran blok lokal untuk komputasi histogram.

    Returns:
        np.ndarray: Citra BGR hasil CLAHE.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


# ---------------------------------------------------------------------------
# Pipeline Restorasi
# ---------------------------------------------------------------------------

def restore_image(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> dict:
    """
    Menjalankan pipeline restorasi lengkap.

    Returns:
        dict berisi tiga citra:
            - 'original'         : citra asli
            - 'color_corrected'  : hasil tahap 1 (Gray World)
            - 'final'            : hasil tahap 2 (CLAHE)
    """
    color_corrected = gray_world_correction(image_bgr)
    final = apply_clahe_on_l_channel(
        color_corrected,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
    )
    return {
        "original": image_bgr,
        "color_corrected": color_corrected,
        "final": final,
    }


# ---------------------------------------------------------------------------
# Metrik Kuantitatif
# ---------------------------------------------------------------------------

def compute_contrast_std(image_bgr: np.ndarray) -> float:
    """
    Menghitung standar deviasi histogram channel L (LAB).
    Nilai lebih tinggi menunjukkan distribusi intensitas yang lebih
    menyebar, yang berarti kontras yang lebih baik.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    return float(np.std(l_channel))


def compute_color_cast_index(image_bgr: np.ndarray) -> float:
    """
    Menghitung indeks color cast sebagai rentang antar rata-rata
    channel B, G, R.

    Nilai mendekati 0 menunjukkan warna yang lebih netral (seimbang).
    Nilai tinggi menunjukkan adanya dominasi warna pada salah satu channel.
    """
    means = [np.mean(image_bgr[:, :, c]) for c in range(3)]
    return float(max(means) - min(means))


def compute_psnr(reference: np.ndarray, target: np.ndarray) -> float:
    """
    Menghitung PSNR (Peak Signal-to-Noise Ratio) antara dua citra.

    Rumus:
        PSNR = 10 * log10(MAX^2 / MSE)
    di mana MAX = 255 untuk citra 8-bit.
    """
    mse = np.mean(
        (reference.astype(np.float64) - target.astype(np.float64)) ** 2
    )
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def print_metrics_table(results: dict) -> None:
    """Mencetak tabel metrik kuantitatif ke terminal."""
    original = results["original"]
    color_corrected = results["color_corrected"]
    final = results["final"]

    std_original = compute_contrast_std(original)
    std_corrected = compute_contrast_std(color_corrected)
    std_final = compute_contrast_std(final)

    cast_original = compute_color_cast_index(original)
    cast_corrected = compute_color_cast_index(color_corrected)
    cast_final = compute_color_cast_index(final)

    psnr_corrected = compute_psnr(original, color_corrected)
    psnr_final = compute_psnr(original, final)

    print()
    print("=" * 70)
    print("METRIK KUANTITATIF".center(70))
    print("=" * 70)
    print(f"{'Metrik':<28}{'Original':>12}{'Koreksi Warna':>16}{'Final':>12}")
    print("-" * 70)
    print(
        f"{'Std Dev Channel L':<28}"
        f"{std_original:>12.2f}{std_corrected:>16.2f}{std_final:>12.2f}"
    )
    print(
        f"{'Color Cast Index':<28}"
        f"{cast_original:>12.2f}{cast_corrected:>16.2f}{cast_final:>12.2f}"
    )
    print(
        f"{'PSNR vs Original (dB)':<28}"
        f"{'—':>12}{psnr_corrected:>16.2f}{psnr_final:>12.2f}"
    )
    print("=" * 70)
    print("Keterangan:")
    print("  - Std Dev lebih tinggi  -> kontras lebih baik")
    print("  - Color Cast lebih rendah -> warna lebih seimbang/netral")
    print("  - PSNR lebih tinggi      -> perubahan lebih halus dari asli")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------

def _plot_histogram(ax, image_bgr: np.ndarray, title: str, color: str) -> None:
    """Menggambar histogram channel L pada axes yang diberikan."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256]).flatten()

    ax.bar(range(256), hist, color=color, width=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlim([0, 255])
    ax.set_xlabel("Intensitas (0-255)", fontsize=8)
    ax.set_ylabel("Frekuensi", fontsize=8)
    ax.tick_params(labelsize=7)


def save_comparison_figure(results: dict, output_path: str) -> None:
    """
    Menyimpan figure perbandingan 2x3:
        Baris 1 : citra original | hasil koreksi warna | hasil final
        Baris 2 : histogram channel L untuk masing-masing tahap
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Restorasi Foto Jadul: Pipeline Koreksi Warna + CLAHE",
        fontsize=13,
        fontweight="bold",
    )

    items = [
        (results["original"], "Original (Foto Jadul)", "coral"),
        (results["color_corrected"], "Tahap 1: Koreksi Warna (Gray World)", "steelblue"),
        (results["final"], "Tahap 2: CLAHE pada Channel L", "seagreen"),
    ]

    for column, (image, title, hist_color) in enumerate(items):
        axes[0, column].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, column].set_title(title, fontsize=10, fontweight="bold")
        axes[0, column].axis("off")

        _plot_histogram(
            axes[1, column],
            image,
            f"Histogram Channel L - {title}",
            color=hist_color,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restorasi citra foto jadul berwarna menggunakan "
            "Gray World Assumption dan CLAHE."
        )
    )
    parser.add_argument(
        "input",
        help="Path ke foto jadul berwarna (contoh: images/gambar_jadul.jpg)",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="Direktori penyimpanan hasil (default: output).",
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (default: 2.0).",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=8,
        help="Ukuran tile CLAHE dalam piksel (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Membaca gambar: {args.input}")
    original = load_image(args.input)
    print(f"      Ukuran: {original.shape[1]} x {original.shape[0]} piksel")

    tile_grid_size = (args.tile_size, args.tile_size)
    print("[2/4] Menjalankan pipeline restorasi:")
    print("      Tahap 1 - Koreksi Warna (Gray World Assumption)")
    print(
        f"      Tahap 2 - CLAHE (clipLimit={args.clip_limit}, "
        f"tileGridSize={tile_grid_size})"
    )
    results = restore_image(
        original,
        clip_limit=args.clip_limit,
        tile_grid_size=tile_grid_size,
    )

    print("[3/4] Menyimpan citra hasil tiap tahap")
    cv2.imwrite(
        os.path.join(args.output_dir, "hasil_tahap1_koreksi_warna.jpg"),
        results["color_corrected"],
    )
    cv2.imwrite(
        os.path.join(args.output_dir, "hasil_tahap2_clahe_final.jpg"),
        results["final"],
    )

    print("[4/4] Membuat visualisasi perbandingan dan menghitung metrik")
    comparison_path = os.path.join(args.output_dir, "hasil_perbandingan.png")
    save_comparison_figure(results, comparison_path)
    print_metrics_table(results)

    print(f"Selesai. Semua hasil tersimpan di folder: {args.output_dir}/")


if __name__ == "__main__":
    main()
