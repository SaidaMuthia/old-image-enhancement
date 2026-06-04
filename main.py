"""
Peningkatan Citra Foto Jadul Berwarna Menggunakan Koreksi Warna dan CLAHE
=======================================================================
Mata Kuliah  : Visi Komputer
Metode       : Classical Image Processing (tanpa Deep Learning)
Library      : NumPy, Matplotlib, Pillow
"""

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------------
# Konversi Ruang Warna
# ---------------------------------------------------------------------------

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

_XYZ_TO_RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)

_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c_safe = np.maximum(c, 0.0)
    return np.where(c > 0.0031308, 1.055 * c_safe ** (1.0 / 2.4) - 0.055, 12.92 * c)


def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3.0 * delta ** 2) + 4.0 / 29.0)


def _lab_f_inv(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta, t ** 3, 3.0 * delta ** 2 * (t - 4.0 / 29.0))


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """
    Mengonversi citra sRGB (uint8, H x W x 3) ke ruang warna CIE L*a*b*.
    Menggunakan transformasi sRGB -> linear RGB -> XYZ (D65) -> L*a*b*.
    Mengembalikan array float64 dengan L dalam [0, 100] dan a, b dalam [-128, 127].
    """
    rgb_f = image_rgb.astype(np.float64) / 255.0
    lin = _srgb_to_linear(rgb_f)
    xyz = lin @ _RGB_TO_XYZ.T
    xyz_norm = xyz / _D65
    f = _lab_f(xyz_norm)
    L = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(image_lab: np.ndarray) -> np.ndarray:
    """
    Mengonversi citra CIE L*a*b* (float64) kembali ke sRGB (uint8).
    Menggunakan transformasi L*a*b* -> XYZ (D65) -> linear RGB -> sRGB.
    """
    L, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = np.stack([
        _D65[0] * _lab_f_inv(fx),
        _D65[1] * _lab_f_inv(fy),
        _D65[2] * _lab_f_inv(fz),
    ], axis=-1)
    lin = xyz @ _XYZ_TO_RGB.T
    srgb = _linear_to_srgb(lin)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Membaca gambar dari path dan mengembalikan array uint8 (H x W x 3, RGB)."""
    img = Image.open(path).convert("RGB")
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    return np.array(img, dtype=np.uint8)


def save_image(image_rgb: np.ndarray, path: str) -> None:
    Image.fromarray(image_rgb).save(path)


# ---------------------------------------------------------------------------
# Tahap 1: Koreksi Warna (Gray World Assumption)
# ---------------------------------------------------------------------------

def gray_world_correction(image_rgb: np.ndarray) -> np.ndarray:
    """
    Menerapkan Gray World Assumption untuk menetralisir color cast.

    Asumsi metode: rata-rata warna pada citra natural seharusnya netral.
    Tiap channel diskalakan agar rata-ratanya seimbang terhadap rata-rata global.

    Rumus:
        scale_c = gray_mean / mean_c   untuk c in {R, G, B}
        I_c_new = clip(I_c * scale_c, 0, 255)
    di mana gray_mean = (mean_R + mean_G + mean_B) / 3.

    Returns:
        np.ndarray: Citra RGB hasil koreksi warna (uint8).
    """
    img_f = image_rgb.astype(np.float64)
    means = img_f.mean(axis=(0, 1))
    gray_mean = means.mean()
    scales = np.where(means > 0, gray_mean / means, 1.0)
    corrected = np.clip(img_f * scales, 0, 255)
    return corrected.astype(np.uint8)


# ---------------------------------------------------------------------------
# Tahap 2: CLAHE pada Channel L
# ---------------------------------------------------------------------------

def _compute_tile_luts(
    l_uint8: np.ndarray,
    ny: int,
    nx: int,
    clip_limit: float,
) -> np.ndarray:
    """
    Menghitung lookup table (LUT) ekualisasi histogram untuk setiap tile.

    Untuk tiap tile:
    1. Hitung histogram (256 bin).
    2. Terapkan clip limit: C_l = clip_limit * (jumlah piksel tile) / 256.
       Piksel yang terpotong didistribusikan ulang secara merata ke seluruh bin.
    3. Hitung CDF dan petakan ke LUT ekualisasi rentang [0, 255].

    Returns:
        np.ndarray: Array (ny, nx, 256) berisi LUT tiap tile.
    """
    H, W = l_uint8.shape
    luts = np.zeros((ny, nx, 256), dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            y0 = int(round(iy * H / ny))
            y1 = int(round((iy + 1) * H / ny))
            x0 = int(round(ix * W / nx))
            x1 = int(round((ix + 1) * W / nx))
            y0, y1 = max(0, y0), min(H, y1)
            x0, x1 = max(0, x0), min(W, x1)

            tile = l_uint8[y0:y1, x0:x1]
            n_pixels = tile.size

            hist = np.bincount(tile.flatten(), minlength=256).astype(np.float64)

            actual_clip = max(1, int(clip_limit * n_pixels / 256))
            excess = np.sum(np.maximum(hist - actual_clip, 0))
            hist = np.minimum(hist, float(actual_clip))
            hist += excess / 256.0

            cdf = np.cumsum(hist)
            cdf_min = cdf[0]
            denom = float(n_pixels) - cdf_min
            if denom > 0:
                lut = np.clip(np.round((cdf - cdf_min) / denom * 255.0), 0, 255)
            else:
                lut = np.arange(256, dtype=np.float64)

            luts[iy, ix] = lut

    return luts


def _bilinear_interpolate(
    l_uint8: np.ndarray,
    luts: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Menggabungkan hasil LUT dari tile-tile yang bersebelahan menggunakan
    interpolasi bilinear untuk menghindari artefak batas antar tile.

    Tiap piksel diinterpolasi dari empat tile terdekat berdasarkan posisi
    relatifnya terhadap pusat tile-tile tersebut.

    Returns:
        np.ndarray: Channel L hasil CLAHE (uint8, H x W).
    """
    H, W = l_uint8.shape

    cy = (np.arange(ny, dtype=np.float64) + 0.5) * (H / ny)
    cx = (np.arange(nx, dtype=np.float64) + 0.5) * (W / nx)

    y_idx = np.arange(H, dtype=np.float64)
    x_idx = np.arange(W, dtype=np.float64)

    if ny > 1:
        iy = np.clip(np.searchsorted(cy, y_idx + 0.5) - 1, 0, ny - 2)
        ty = np.clip((y_idx + 0.5 - cy[iy]) / (cy[iy + 1] - cy[iy]), 0.0, 1.0)
    else:
        iy = np.zeros(H, dtype=int)
        ty = np.zeros(H, dtype=np.float64)

    if nx > 1:
        ix = np.clip(np.searchsorted(cx, x_idx + 0.5) - 1, 0, nx - 2)
        tx = np.clip((x_idx + 0.5 - cx[ix]) / (cx[ix + 1] - cx[ix]), 0.0, 1.0)
    else:
        ix = np.zeros(W, dtype=int)
        tx = np.zeros(W, dtype=np.float64)

    PV = l_uint8
    IY  = iy[:, np.newaxis] * np.ones((1, W), dtype=int)
    IX  = np.ones((H, 1), dtype=int) * ix[np.newaxis, :]
    IY1 = np.minimum(IY + 1, ny - 1)
    IX1 = np.minimum(IX + 1, nx - 1)

    V00 = luts[IY,  IX,  PV]
    V01 = luts[IY,  IX1, PV]
    V10 = luts[IY1, IX,  PV]
    V11 = luts[IY1, IX1, PV]

    TY = ty[:, np.newaxis]
    TX = tx[np.newaxis, :]

    result = (
        (1.0 - TY) * (1.0 - TX) * V00 +
        (1.0 - TY) * TX          * V01 +
        TY          * (1.0 - TX) * V10 +
        TY          * TX          * V11
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> np.ndarray:
    """
    Menerapkan CLAHE pada channel L dalam ruang warna CIE L*a*b*.

    Channel L (lightness) yang memuat informasi kecerahan diproses secara
    terpisah dari channel a dan b (informasi warna). Ini memastikan
    peningkatan kontras tidak mendistorsi warna hasil koreksi tahap 1.

    Args:
        clip_limit : Batas amplifikasi kontras per tile.
        tile_grid  : (nx, ny), jumlah tile secara horizontal dan vertikal.

    Returns:
        np.ndarray: Citra RGB hasil CLAHE (uint8).
    """
    lab = rgb_to_lab(image_rgb)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    l_scaled = np.clip(np.round(L * 255.0 / 100.0), 0, 255).astype(np.uint8)

    nx, ny = tile_grid
    luts = _compute_tile_luts(l_scaled, ny, nx, clip_limit)
    l_enhanced = _bilinear_interpolate(l_scaled, luts, ny, nx)

    L_new = l_enhanced.astype(np.float64) * 100.0 / 255.0
    lab_new = np.stack([L_new, a, b], axis=-1)

    return lab_to_rgb(lab_new)


# ---------------------------------------------------------------------------
# Pipeline Restorasi
# ---------------------------------------------------------------------------

def restore_image(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> dict:
    """
    Menjalankan pipeline restorasi dua tahap.

    Returns:
        dict dengan kunci 'original', 'color_corrected', dan 'final'.
    """
    color_corrected = gray_world_correction(image_rgb)
    final = apply_clahe(color_corrected, clip_limit=clip_limit, tile_grid=tile_grid)
    return {
        "original": image_rgb,
        "color_corrected": color_corrected,
        "final": final,
    }


# ---------------------------------------------------------------------------
# Metrik Kuantitatif
# ---------------------------------------------------------------------------

def get_l_channel_uint8(image_rgb: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(image_rgb)
    return np.clip(np.round(lab[:, :, 0] * 255.0 / 100.0), 0, 255).astype(np.uint8)


def compute_contrast_std(image_rgb: np.ndarray) -> float:
    """
    Standar deviasi channel L (CIE L*a*b*).
    Nilai lebih tinggi menunjukkan distribusi intensitas yang lebih lebar (kontras lebih baik).
    """
    lab = rgb_to_lab(image_rgb)
    return float(np.std(lab[:, :, 0]))


def compute_color_cast_index(image_rgb: np.ndarray) -> float:
    """
    Indeks color cast: selisih antara rata-rata channel terbesar dan terkecil.
    Nilai mendekati 0 menunjukkan keseimbangan warna yang lebih netral.
    """
    means = image_rgb.astype(np.float64).mean(axis=(0, 1))
    return float(means.max() - means.min())


def compute_psnr(reference: np.ndarray, target: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio antara dua citra.
    Rumus: PSNR = 10 * log10(255^2 / MSE)
    """
    mse = np.mean((reference.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(255.0 ** 2 / mse)


def print_metrics_table(results: dict) -> None:
    original = results["original"]
    cc       = results["color_corrected"]
    final    = results["final"]

    std_o  = compute_contrast_std(original)
    std_cc = compute_contrast_std(cc)
    std_f  = compute_contrast_std(final)

    cast_o  = compute_color_cast_index(original)
    cast_cc = compute_color_cast_index(cc)
    cast_f  = compute_color_cast_index(final)

    psnr_cc = compute_psnr(original, cc)
    psnr_f  = compute_psnr(original, final)

    print()
    print("=" * 70)
    print("METRIK KUANTITATIF".center(70))
    print("=" * 70)
    print(f"{'Metrik':<28}{'Original':>12}{'Koreksi Warna':>16}{'Final':>12}")
    print("-" * 70)
    print(f"{'Std Dev Channel L':<28}{std_o:>12.2f}{std_cc:>16.2f}{std_f:>12.2f}")
    print(f"{'Color Cast Index':<28}{cast_o:>12.2f}{cast_cc:>16.2f}{cast_f:>12.2f}")
    print(f"{'PSNR vs Original (dB)':<28}{'—':>12}{psnr_cc:>16.2f}{psnr_f:>12.2f}")
    print("=" * 70)
    print("Keterangan:")
    print("  Std Dev lebih tinggi   -> kontras lebih baik")
    print("  Color Cast lebih kecil -> warna lebih netral")
    print("  PSNR lebih tinggi      -> perubahan lebih halus dari citra asli")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------

def _plot_histogram(ax, image_rgb: np.ndarray, title: str, color: str) -> None:
    l_u8 = get_l_channel_uint8(image_rgb)
    hist = np.bincount(l_u8.flatten(), minlength=256)
    ax.bar(range(256), hist, color=color, width=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlim([0, 255])
    ax.set_xlabel("Intensitas (0-255)", fontsize=8)
    ax.set_ylabel("Frekuensi", fontsize=8)
    ax.tick_params(labelsize=7)


def save_comparison_figure(results: dict, output_path: str) -> None:
    """
    Menyimpan gambar perbandingan 2x3:
        Baris 1: citra original | hasil koreksi warna | hasil final
        Baris 2: histogram channel L masing-masing tahap
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Restorasi Foto Jadul: Pipeline Koreksi Warna + CLAHE",
        fontsize=13,
        fontweight="bold",
    )

    items = [
        (results["original"],        "Original (Foto Jadul)",               "coral"),
        (results["color_corrected"], "Tahap 1: Koreksi Warna (Gray World)", "steelblue"),
        (results["final"],           "Tahap 2: CLAHE pada Channel L",        "seagreen"),
    ]

    for col, (img, title, hcolor) in enumerate(items):
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")
        axes[0, col].axis("off")
        _plot_histogram(
            axes[1, col], img,
            f"Histogram Channel L - {title}",
            color=hcolor,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restorasi citra foto jadul berwarna menggunakan Gray World Assumption dan CLAHE."
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
        help="Jumlah grid tile CLAHE per sisi (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Membaca gambar: {args.input}")
    original = load_image(args.input)
    print(f"      Ukuran: {original.shape[1]} x {original.shape[0]} piksel")

    tile_grid = (args.tile_size, args.tile_size)
    print("[2/4] Menjalankan pipeline restorasi:")
    print("      Tahap 1 - Koreksi Warna (Gray World Assumption)")
    print(f"      Tahap 2 - CLAHE (clip_limit={args.clip_limit}, tile_grid={tile_grid})")
    results = restore_image(original, clip_limit=args.clip_limit, tile_grid=tile_grid)

    print("[3/4] Menyimpan citra hasil")
    save_image(results["color_corrected"],
               os.path.join(args.output_dir, "hasil_tahap1_koreksi_warna.jpg"))
    save_image(results["final"],
               os.path.join(args.output_dir, "hasil_tahap2_clahe_final.jpg"))

    print("[4/4] Membuat visualisasi perbandingan dan menghitung metrik")
    save_comparison_figure(
        results,
        os.path.join(args.output_dir, "hasil_perbandingan.png"),
    )
    print_metrics_table(results)

    print(f"Selesai. Semua hasil tersimpan di folder: {args.output_dir}/")


if __name__ == "__main__":
    main()
