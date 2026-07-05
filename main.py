"""
Restorasi Citra Foto Jadul Berwarna Menggunakan Koreksi Warna dan CLAHE
=======================================================================
Mata Kuliah  : Visi Komputer
Metode       : Classical Image Processing (tanpa Deep Learning)
Library      : NumPy, Matplotlib, Pillow

Pipeline dua tahap:
  Tahap 1 — Gray World Assumption  : menetralisir color cast
  Tahap 2 — CLAHE pada channel L   : meningkatkan kontras tanpa mendistorsi warna
"""

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------------------------
# Matriks konversi RGB ↔ XYZ (illuminan D65, standar CIE)
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
    """Konversi sRGB (uint8) → CIE L*a*b* (float64). L: [0,100], a/b: [-128,127]."""
    rgb_f = image_rgb.astype(np.float64) / 255.0
    lin   = _srgb_to_linear(rgb_f)
    xyz   = lin @ _RGB_TO_XYZ.T
    f     = _lab_f(xyz / _D65)
    L = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(image_lab: np.ndarray) -> np.ndarray:
    """Konversi CIE L*a*b* (float64) → sRGB (uint8)."""
    L, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    xyz = np.stack([
        _D65[0] * _lab_f_inv(fx),
        _D65[1] * _lab_f_inv(fy),
        _D65[2] * _lab_f_inv(fz),
    ], axis=-1)
    srgb = _linear_to_srgb(xyz @ _XYZ_TO_RGB.T)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Membaca gambar dan mengembalikan array uint8 (H×W×3, RGB)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(image_rgb: np.ndarray, path: str) -> None:
    Image.fromarray(image_rgb).save(path)


# ---------------------------------------------------------------------------
# Tahap 1: Koreksi Warna — Gray World Assumption
# ---------------------------------------------------------------------------

def gray_world_correction(image_rgb: np.ndarray) -> np.ndarray:
    """
    Menetralisir color cast dengan menyeimbangkan rata-rata tiap channel
    terhadap rata-rata global (gray mean).

    scale_c    = gray_mean / mean_c
    I_c_new    = clip(I_c * scale_c, 0, 255)
    gray_mean  = (mean_R + mean_G + mean_B) / 3
    """
    img_f     = image_rgb.astype(np.float64)
    means     = img_f.mean(axis=(0, 1))
    gray_mean = means.mean()
    scales    = np.where(means > 0, gray_mean / means, 1.0)
    return np.clip(img_f * scales, 0, 255).astype(np.uint8)


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
    Menghitung LUT ekualisasi histogram untuk setiap tile.

    Untuk tiap tile:
      1. Hitung histogram (256 bin).
      2. Terapkan clip: C_l = clip_limit * (jumlah piksel) / 256.
         Piksel terpotong didistribusikan ulang merata ke semua bin.
      3. Hitung CDF dan petakan ke LUT [0, 255].

    Mengembalikan array (ny, nx, 256).
    """
    H, W = l_uint8.shape
    luts = np.zeros((ny, nx, 256), dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            y0 = int(round(iy * H / ny));       y1 = int(round((iy + 1) * H / ny))
            x0 = int(round(ix * W / nx));       x1 = int(round((ix + 1) * W / nx))
            y0, y1 = max(0, y0), min(H, y1);    x0, x1 = max(0, x0), min(W, x1)

            tile     = l_uint8[y0:y1, x0:x1]
            n_pixels = tile.size
            hist     = np.bincount(tile.flatten(), minlength=256).astype(np.float64)

            actual_clip = max(1, int(clip_limit * n_pixels / 256))
            excess      = np.sum(np.maximum(hist - actual_clip, 0))
            hist        = np.minimum(hist, float(actual_clip))
            hist       += excess / 256.0

            cdf     = np.cumsum(hist)
            cdf_min = cdf[0]
            denom   = float(n_pixels) - cdf_min
            lut     = np.clip(np.round((cdf - cdf_min) / denom * 255.0), 0, 255) \
                      if denom > 0 else np.arange(256, dtype=np.float64)
            luts[iy, ix] = lut

    return luts


def _bilinear_interpolate(
    l_uint8: np.ndarray,
    luts: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """
    Menggabungkan LUT tile-tile bersebelahan dengan interpolasi bilinear
    agar tidak ada artefak batas antar tile.

    Mengembalikan channel L hasil CLAHE (uint8, H×W).
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
        iy = np.zeros(H, dtype=int);  ty = np.zeros(H, dtype=np.float64)

    if nx > 1:
        ix = np.clip(np.searchsorted(cx, x_idx + 0.5) - 1, 0, nx - 2)
        tx = np.clip((x_idx + 0.5 - cx[ix]) / (cx[ix + 1] - cx[ix]), 0.0, 1.0)
    else:
        ix = np.zeros(W, dtype=int);  tx = np.zeros(W, dtype=np.float64)

    PV  = l_uint8
    IY  = iy[:, np.newaxis] * np.ones((1, W), dtype=int)
    IX  = np.ones((H, 1), dtype=int) * ix[np.newaxis, :]
    IY1 = np.minimum(IY + 1, ny - 1)
    IX1 = np.minimum(IX + 1, nx - 1)

    TY = ty[:, np.newaxis];  TX = tx[np.newaxis, :]
    result = (
        (1.0 - TY) * (1.0 - TX) * luts[IY,  IX,  PV] +
        (1.0 - TY) * TX          * luts[IY,  IX1, PV] +
        TY          * (1.0 - TX) * luts[IY1, IX,  PV] +
        TY          * TX          * luts[IY1, IX1, PV]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> np.ndarray:
    """
    Menerapkan CLAHE pada channel L dalam ruang warna CIE L*a*b*.

    Channel L (kecerahan) diproses terpisah dari channel a dan b (warna),
    sehingga peningkatan kontras tidak mengubah keseimbangan warna
    yang sudah dikoreksi di tahap sebelumnya.

    Args:
        clip_limit : batas amplifikasi kontras per tile (default 2.0).
        tile_grid  : (nx, ny) jumlah tile per sisi (default 8×8).
    """
    lab = rgb_to_lab(image_rgb)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    l_scaled  = np.clip(np.round(L * 255.0 / 100.0), 0, 255).astype(np.uint8)
    nx, ny    = tile_grid
    luts      = _compute_tile_luts(l_scaled, ny, nx, clip_limit)
    l_enhanced = _bilinear_interpolate(l_scaled, luts, ny, nx)

    L_new = l_enhanced.astype(np.float64) * 100.0 / 255.0
    return lab_to_rgb(np.stack([L_new, a, b], axis=-1))


# ---------------------------------------------------------------------------
# Pipeline Restorasi
# ---------------------------------------------------------------------------

def restore_image(
    image_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple = (8, 8),
) -> dict:
    """Menjalankan pipeline dua tahap. Mengembalikan dict hasil tiap tahap."""
    color_corrected = gray_world_correction(image_rgb)
    final           = apply_clahe(color_corrected, clip_limit=clip_limit, tile_grid=tile_grid)
    return {"original": image_rgb, "color_corrected": color_corrected, "final": final}


# ---------------------------------------------------------------------------
# Metrik Kuantitatif
# ---------------------------------------------------------------------------

def get_l_channel_uint8(image_rgb: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(image_rgb)
    return np.clip(np.round(lab[:, :, 0] * 255.0 / 100.0), 0, 255).astype(np.uint8)


def compute_contrast_std(image_rgb: np.ndarray) -> float:
    """Standar deviasi channel L — ukuran kontras (sebaran intensitas)."""
    return float(np.std(rgb_to_lab(image_rgb)[:, :, 0]))


def compute_color_cast_index(image_rgb: np.ndarray) -> float:
    """Selisih rata-rata channel terbesar dan terkecil — ukuran ketidakseimbangan warna."""
    means = image_rgb.astype(np.float64).mean(axis=(0, 1))
    return float(means.max() - means.min())


def compute_psnr(reference: np.ndarray, target: np.ndarray) -> float:
    """PSNR antara dua citra. Rumus: 10 * log10(255² / MSE)."""
    mse = np.mean((reference.astype(np.float64) - target.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def print_metrics_table(results: dict) -> None:
    o, cc, f = results["original"], results["color_corrected"], results["final"]
    rows = [
        ("Std Dev Channel L",    compute_contrast_std,    o, cc, f, False),
        ("Color Cast Index",     compute_color_cast_index, o, cc, f, False),
    ]

    print()
    print("=" * 70)
    print("METRIK KUANTITATIF".center(70))
    print("=" * 70)
    print(f"{'Metrik':<28}{'Original':>12}{'Koreksi Warna':>16}{'Final':>12}")
    print("-" * 70)
    print(f"{'Std Dev Channel L':<28}{compute_contrast_std(o):>12.2f}"
          f"{compute_contrast_std(cc):>16.2f}{compute_contrast_std(f):>12.2f}")
    print(f"{'Color Cast Index':<28}{compute_color_cast_index(o):>12.2f}"
          f"{compute_color_cast_index(cc):>16.2f}{compute_color_cast_index(f):>12.2f}")
    print(f"{'PSNR vs Original (dB)':<28}{'—':>12}"
          f"{compute_psnr(o, cc):>16.2f}{compute_psnr(o, f):>12.2f}")
    print("=" * 70)
    print("Keterangan:")
    print("  Std Dev lebih tinggi   -> sebaran intensitas lebih luas = kontras lebih baik")
    print("  Color Cast lebih kecil -> rata-rata R, G, B lebih seimbang = warna lebih netral")
    print("  PSNR lebih rendah      -> perubahan dari asli lebih besar (wajar untuk tahap CLAHE)")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------

def _plot_histogram(ax, image_rgb: np.ndarray, title: str, color: str) -> None:
    hist = np.bincount(get_l_channel_uint8(image_rgb).flatten(), minlength=256)
    ax.bar(range(256), hist, color=color, width=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlim([0, 255])
    ax.set_xlabel("Intensitas Channel L (0–255)", fontsize=8)
    ax.set_ylabel("Frekuensi Piksel", fontsize=8)
    ax.tick_params(labelsize=7)


def save_comparison_figure(results: dict, output_path: str) -> None:
    """
    Menyimpan visualisasi 3×3:
      Baris 1 : citra tiap tahap + semua metrik sebagai caption
      Baris 2 : histogram channel L tiap tahap
    """
    o   = results["original"]
    cc  = results["color_corrected"]
    fin = results["final"]

    items = [
        (o,   "Original (Foto Jadul)",               "coral"),
        (cc,  "Tahap 1: Koreksi Warna (Gray World)", "steelblue"),
        (fin, "Tahap 2: CLAHE pada Channel L",        "seagreen"),
    ]

    # Hitung semua metrik terlebih dahulu
    metrics = []
    for idx, (img, _, _) in enumerate(items):
        std  = compute_contrast_std(img)
        cast = compute_color_cast_index(img)
        psnr = compute_psnr(o, img) if idx > 0 else None
        metrics.append((std, cast, psnr))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Restorasi Foto Jadul: Pipeline Koreksi Warna + CLAHE",
                 fontsize=13, fontweight="bold")

    for col, ((img, title, hcolor), (std, cast, psnr)) in enumerate(zip(items, metrics)):
        # Baris atas: citra
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Caption di bawah citra: semua metrik
        psnr_str = f"{psnr:.2f} dB" if psnr is not None else "—"
        caption = (
            f"Std Dev L  : {std:.2f}\n"
            f"Color Cast : {cast:.2f}\n"
            f"PSNR       : {psnr_str}"
        )
        axes[0, col].set_xlabel(caption, fontsize=8, linespacing=1.7,
                                fontfamily="monospace")

        # Baris bawah: histogram channel L
        _plot_histogram(axes[1, col], img, f"Histogram Channel L\n{title}", hcolor)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restorasi citra foto jadul berwarna (Gray World + CLAHE)."
    )
    parser.add_argument("input", help="Path ke foto jadul (contoh: images/gambar_jadul.jpg)")
    parser.add_argument("--output_dir",  default="output", help="Folder output (default: output)")
    parser.add_argument("--clip_limit",  type=float, default=2.0,  help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--tile_size",   type=int,   default=8,    help="Grid tile CLAHE per sisi (default: 8)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Membaca gambar: {args.input}")
    original = load_image(args.input)
    print(f"      Ukuran: {original.shape[1]} x {original.shape[0]} piksel")

    tile_grid = (args.tile_size, args.tile_size)
    print("[2/4] Menjalankan pipeline restorasi:")
    print("      Tahap 1 - Gray World Assumption")
    print(f"      Tahap 2 - CLAHE pada Channel L (clip_limit={args.clip_limit}, tile_grid={tile_grid})")
    results = restore_image(original, clip_limit=args.clip_limit, tile_grid=tile_grid)

    print("[3/4] Menyimpan citra hasil")
    save_image(results["color_corrected"],
               os.path.join(args.output_dir, "hasil_tahap1_koreksi_warna.jpg"))
    save_image(results["final"],
               os.path.join(args.output_dir, "hasil_tahap2_clahe_final.jpg"))

    print("[4/4] Membuat visualisasi dan menghitung metrik")
    save_comparison_figure(results, os.path.join(args.output_dir, "hasil_perbandingan.png"))
    print_metrics_table(results)
    print(f"Selesai. Semua hasil tersimpan di: {args.output_dir}/")


if __name__ == "__main__":
    main()
