"""
Eksperimen Variasi Parameter CLAHE
===================================
Mata Kuliah  : Visi Komputer
Deskripsi    : Menguji berbagai nilai clip_limit dan tile_grid untuk
               memvalidasi pemilihan parameter default pada main.py.
               Tahap 1 (Gray World) bersifat tetap karena parameter-free.

Cara menjalankan:
    python eksperimen_parameter.py images/gambar_jadul.jpg
    python eksperimen_parameter.py images/gambar_jadul.jpg --output_dir output
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from main import (
    load_image,
    gray_world_correction,
    apply_clahe,
    compute_contrast_std,
    compute_color_cast_index,
    compute_psnr,
)


CLIP_LIMITS = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
TILE_SIZES  = [4, 8, 16, 32]
FIXED_TILE  = (8, 8)
FIXED_CLIP  = 2.0


def run_clip_limit_experiment(
    color_corrected: np.ndarray,
    original: np.ndarray,
) -> list:
    """
    Menjalankan eksperimen variasi clip_limit dengan tile_grid tetap (8, 8).
    """
    results = []
    for cl in CLIP_LIMITS:
        result = apply_clahe(color_corrected, clip_limit=cl, tile_grid=FIXED_TILE)
        results.append({
            "clip_limit": cl,
            "image":      result,
            "std":        compute_contrast_std(result),
            "cast":       compute_color_cast_index(result),
            "psnr":       compute_psnr(original, result),
        })
    return results


def run_tile_size_experiment(
    color_corrected: np.ndarray,
    original: np.ndarray,
) -> list:
    """
    Menjalankan eksperimen variasi tile_grid dengan clip_limit tetap 2.0.
    """
    results = []
    for ts in TILE_SIZES:
        result = apply_clahe(color_corrected, clip_limit=FIXED_CLIP, tile_grid=(ts, ts))
        results.append({
            "tile_size": ts,
            "image":     result,
            "std":       compute_contrast_std(result),
            "cast":      compute_color_cast_index(result),
            "psnr":      compute_psnr(original, result),
        })
    return results


def print_clip_limit_table(results: list) -> None:
    print()
    print("=" * 70)
    print("EKSPERIMEN 1: Variasi clip_limit (tile_grid tetap 8x8)")
    print("=" * 70)
    print(f"{'clip_limit':<12}{'Std Dev L':<14}{'Color Cast':<14}{'PSNR (dB)':<12}")
    print("-" * 70)
    for r in results:
        marker = " <-- dipilih" if r["clip_limit"] == FIXED_CLIP else ""
        print(
            f"{r['clip_limit']:<12.1f}"
            f"{r['std']:<14.2f}"
            f"{r['cast']:<14.2f}"
            f"{r['psnr']:<12.2f}"
            f"{marker}"
        )
    print("=" * 70)


def print_tile_size_table(results: list) -> None:
    print()
    print("=" * 70)
    print("EKSPERIMEN 2: Variasi tile_grid (clip_limit tetap 2.0)")
    print("=" * 70)
    print(f"{'tile_size':<14}{'Std Dev L':<14}{'Color Cast':<14}{'PSNR (dB)':<12}")
    print("-" * 70)
    for r in results:
        marker = " <-- dipilih" if r["tile_size"] == FIXED_TILE[0] else ""
        print(
            f"{r['tile_size']}x{r['tile_size']:<11}"
            f"{r['std']:<14.2f}"
            f"{r['cast']:<14.2f}"
            f"{r['psnr']:<12.2f}"
            f"{marker}"
        )
    print("=" * 70)
    print()


def save_clip_limit_figure(results: list, output_path: str) -> None:
    """Menyimpan grid visual hasil variasi clip_limit (2 baris x 3 kolom)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        "Eksperimen 1: Variasi clip_limit (tile_grid = 8x8)",
        fontsize=12,
        fontweight="bold",
    )
    for idx, r in enumerate(results):
        row, col = idx // 3, idx % 3
        axes[row, col].imshow(r["image"])
        label = f"clip_limit = {r['clip_limit']}"
        if r["clip_limit"] == FIXED_CLIP:
            label += " (dipilih)"
        axes[row, col].set_title(label, fontsize=10, fontweight="bold")
        axes[row, col].set_xlabel(
            f"Std Dev={r['std']:.2f}  PSNR={r['psnr']:.2f} dB",
            fontsize=8,
        )
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_tile_size_figure(results: list, output_path: str) -> None:
    """Menyimpan grid visual hasil variasi tile_grid (2 baris x 2 kolom)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    fig.suptitle(
        "Eksperimen 2: Variasi tile_grid (clip_limit = 2.0)",
        fontsize=12,
        fontweight="bold",
    )
    for idx, r in enumerate(results):
        row, col = idx // 2, idx % 2
        axes[row, col].imshow(r["image"])
        label = f"tile_grid = {r['tile_size']}x{r['tile_size']}"
        if r["tile_size"] == FIXED_TILE[0]:
            label += " (dipilih)"
        axes[row, col].set_title(label, fontsize=10, fontweight="bold")
        axes[row, col].set_xlabel(
            f"Std Dev={r['std']:.2f}  PSNR={r['psnr']:.2f} dB",
            fontsize=8,
        )
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eksperimen variasi parameter CLAHE untuk restorasi foto jadul."
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
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Membaca gambar: {args.input}")
    original = load_image(args.input)
    print(f"      Ukuran: {original.shape[1]} x {original.shape[0]} piksel")

    print("[2/4] Menerapkan koreksi warna Gray World")
    color_corrected = gray_world_correction(original)

    print("[3/4] Menjalankan eksperimen variasi clip_limit dan tile_grid")
    clip_results = run_clip_limit_experiment(color_corrected, original)
    tile_results = run_tile_size_experiment(color_corrected, original)

    print_clip_limit_table(clip_results)
    print_tile_size_table(tile_results)

    print("[4/4] Menyimpan gambar hasil eksperimen")
    clip_path = os.path.join(args.output_dir, "eksperimen_clip_limit.png")
    tile_path = os.path.join(args.output_dir, "eksperimen_tile_size.png")
    save_clip_limit_figure(clip_results, clip_path)
    save_tile_size_figure(tile_results, tile_path)
    print(f"      eksperimen_clip_limit.png -> {args.output_dir}/")
    print(f"      eksperimen_tile_size.png  -> {args.output_dir}/")
    print(f"\nSelesai. Semua hasil tersimpan di folder: {args.output_dir}/")


if __name__ == "__main__":
    main()
