"""
Eksperimen Variasi Parameter CLAHE
====================================
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
import matplotlib.gridspec as gridspec

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


def run_clip_limit_experiment(color_corrected: np.ndarray, original: np.ndarray) -> list:
    results = []
    for cl in CLIP_LIMITS:
        result = apply_clahe(color_corrected, clip_limit=cl, tile_grid=FIXED_TILE)
        results.append({
            "clip_limit": cl,
            "image": result,
            "std":   compute_contrast_std(result),
            "cast":  compute_color_cast_index(result),
            "psnr":  compute_psnr(original, result),
        })
    return results


def run_tile_size_experiment(color_corrected: np.ndarray, original: np.ndarray) -> list:
    results = []
    for ts in TILE_SIZES:
        result = apply_clahe(color_corrected, clip_limit=FIXED_CLIP, tile_grid=(ts, ts))
        results.append({
            "tile_size": ts,
            "image": result,
            "std":   compute_contrast_std(result),
            "cast":  compute_color_cast_index(result),
            "psnr":  compute_psnr(original, result),
        })
    return results


# ---------------------------------------------------------------------------
# Tabel terminal (tanpa kolom analisis)
# ---------------------------------------------------------------------------

def print_clip_limit_table(results: list) -> None:
    print()
    print("=" * 58)
    print("EKSPERIMEN 1: Variasi clip_limit (tile_grid tetap 8x8)".center(58))
    print("=" * 58)
    print(f"{'clip_limit':<12}{'Std Dev L':<14}{'Color Cast':<14}{'PSNR (dB)'}")
    print("-" * 58)
    for r in results:
        marker = " <--" if r["clip_limit"] == FIXED_CLIP else ""
        print(f"{r['clip_limit']:<12.1f}{r['std']:<14.2f}{r['cast']:<14.2f}{r['psnr']:.2f}{marker}")
    print("=" * 58)


def print_tile_size_table(results: list) -> None:
    print()
    print("=" * 58)
    print("EKSPERIMEN 2: Variasi tile_grid (clip_limit tetap 2.0)".center(58))
    print("=" * 58)
    print(f"{'tile_size':<14}{'Std Dev L':<14}{'Color Cast':<14}{'PSNR (dB)'}")
    print("-" * 58)
    for r in results:
        marker = " <--" if r["tile_size"] == FIXED_TILE[0] else ""
        print(f"{r['tile_size']}x{r['tile_size']:<11}{r['std']:<14.2f}{r['cast']:<14.2f}{r['psnr']:.2f}{marker}")
    print("=" * 58)
    print()


# ---------------------------------------------------------------------------
# Visualisasi — gambar + semua metrik sebagai caption
# ---------------------------------------------------------------------------

def _metrics_caption(r: dict, param_key: str) -> str:
    """Buat string caption berisi semua metrik untuk satu hasil eksperimen."""
    if param_key == "clip_limit":
        label = f"clip_limit = {r['clip_limit']}"
        if r["clip_limit"] == FIXED_CLIP:
            label += " ★ dipilih"
    else:
        label = f"tile_grid = {r['tile_size']}×{r['tile_size']}"
        if r["tile_size"] == FIXED_TILE[0]:
            label += " ★ dipilih"
    return label


def save_clip_limit_figure(results: list, output_path: str) -> None:
    n = len(results)
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("Eksperimen 1: Variasi clip_limit  (tile_grid = 8×8)",
                 fontsize=13, fontweight="bold", y=0.98)

    for idx, r in enumerate(results):
        ax = fig.add_subplot(2, 3, idx + 1)
        ax.imshow(r["image"])

        # judul di atas gambar
        title = f"clip_limit = {r['clip_limit']}"
        if r["clip_limit"] == FIXED_CLIP:
            title += "  ★"
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color="darkgreen" if r["clip_limit"] == FIXED_CLIP else "black")
        ax.set_xticks([])
        ax.set_yticks([])

        # semua metrik di bawah gambar
        caption = (
            f"Std Dev L : {r['std']:.2f}\n"
            f"Color Cast : {r['cast']:.2f}\n"
            f"PSNR       : {r['psnr']:.2f} dB"
        )
        ax.set_xlabel(caption, fontsize=8, linespacing=1.6,
                      fontfamily="monospace",
                      color="darkgreen" if r["clip_limit"] == FIXED_CLIP else "black")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Tersimpan: {output_path}")


def save_tile_size_figure(results: list, output_path: str) -> None:
    n = len(results)
    fig = plt.figure(figsize=(13, 12))
    fig.suptitle("Eksperimen 2: Variasi tile_grid  (clip_limit = 2.0)",
                 fontsize=13, fontweight="bold", y=0.98)

    for idx, r in enumerate(results):
        ax = fig.add_subplot(2, 2, idx + 1)
        ax.imshow(r["image"])

        title = f"tile_grid = {r['tile_size']}×{r['tile_size']}"
        if r["tile_size"] == FIXED_TILE[0]:
            title += "  ★"
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color="darkgreen" if r["tile_size"] == FIXED_TILE[0] else "black")
        ax.set_xticks([])
        ax.set_yticks([])

        caption = (
            f"Std Dev L : {r['std']:.2f}\n"
            f"Color Cast : {r['cast']:.2f}\n"
            f"PSNR       : {r['psnr']:.2f} dB"
        )
        ax.set_xlabel(caption, fontsize=8, linespacing=1.6,
                      fontfamily="monospace",
                      color="darkgreen" if r["tile_size"] == FIXED_TILE[0] else "black")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Tersimpan: {output_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eksperimen variasi parameter CLAHE untuk validasi pemilihan default."
    )
    parser.add_argument("input", help="Path ke foto jadul (contoh: images/gambar_jadul.jpg)")
    parser.add_argument("--output_dir", default="output", help="Folder output (default: output)")
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

    print("[4/4] Menyimpan visualisasi hasil eksperimen")
    save_clip_limit_figure(clip_results, os.path.join(args.output_dir, "eksperimen_clip_limit.png"))
    save_tile_size_figure(tile_results,  os.path.join(args.output_dir, "eksperimen_tile_size.png"))
    print("\nSelesai.")


if __name__ == "__main__":
    main()
