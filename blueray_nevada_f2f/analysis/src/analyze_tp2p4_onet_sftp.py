#!/usr/bin/env python3
"""
TP2-4 analysis for CLM data under data/clm_data_onet_sftp.

Writes separate tile-scatter and distribution figures:
  - tp2p4_tile_vs_*.png / tp2p4_distribution_vs_*.png
Distribution views use horizontal histograms (y = performance in GHz, x = count) with raw samples overlaid.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.lines import Line2D


# Monorepo: .../friendly-system-lmi/data/clm_data_onet_sftp
def default_data_root() -> Path:
    here = Path(__file__).resolve()
    # .../blueray_nevada_f2f/analysis/src/thisfile.py -> parents[5] = repo root
    repo_root = here.parents[5]
    return (repo_root / "data" / "clm_data_onet_sftp").resolve()


def default_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config"


_TILE_SN = re.compile(r"^Y\d{10}$")


def tile_sn_from_csv_path(csv_path: Path) -> str | None:
    for part in csv_path.stem.split("-"):
        if _TILE_SN.match(part):
            return part
    return None


def load_tp2p4_scan_data(tp_path: Path, wl_grid: dict) -> pd.DataFrame:
    """Load *TP2-4 Scan.csv files; filter T_MUX ~50C; per-channel freq error vs grid."""
    c = 299792458 * 1e9
    all_data: list[pd.DataFrame] = []
    csv_files = sorted(tp_path.glob("*TP2-4 Scan.csv"))

    for csv_file in csv_files:
        try:
            tile_sn = tile_sn_from_csv_path(csv_file)
            if tile_sn is None:
                print(f"Warning: could not parse tile SN from {csv_file.name}", file=sys.stderr)
                continue

            df = pd.read_csv(csv_file)
            df["Tile_SN"] = tile_sn
            df = df[(df["T_MUX(C)"] >= 49.9) & (df["T_MUX(C)"] <= 50.1)].copy()
            if df.empty:
                continue

            def calc_freq_error(row):
                bank = int(row["Bank"])
                channel = int(row["Channel"])
                measured_wl = row["OSA_Wave(nm)"]
                bank_key = f"bank{bank}"
                grid_num = channel + 1
                target_wl = wl_grid["banks"][bank_key]["grids"][grid_num]["wavelength_nm"]
                wl_error_nm = measured_wl - target_wl
                freq_error_hz = -(c / (target_wl**2)) * wl_error_nm
                freq_error_ghz = freq_error_hz / 1e9
                return pd.Series(
                    {
                        "Wavelength_Error_nm": wl_error_nm,
                        "Frequency_Error_GHz": freq_error_ghz,
                    }
                )

            df[["Wavelength_Error_nm", "Frequency_Error_GHz"]] = df.apply(calc_freq_error, axis=1)
            df = df[
                [
                    "Tile_SN",
                    "Bank",
                    "Channel",
                    "T_MUX(C)",
                    "OSA_Wave(nm)",
                    "Wavelength_Error_nm",
                    "Frequency_Error_GHz",
                ]
            ].copy()
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}", file=sys.stderr)

    if not all_data:
        return pd.DataFrame()
    out = pd.concat(all_data, ignore_index=True)
    # Single cohort label so gen1-style plots (v1 / v2 ordering) still work
    out["Version"] = "v1"
    return out


def calculate_center_freq_spacing_errors(df: pd.DataFrame, wl_grid: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    center_freq_results = []
    spacing_results = []
    c = 299792458 * 1e9

    for (tile_sn, version, bank), group in df.groupby(["Tile_SN", "Version", "Bank"]):
        group = group.sort_values("Channel")
        bank_key = f"bank{int(bank)}"
        target_center_freq_thz = wl_grid["banks"][bank_key]["center_frequency_thz"]
        target_spacing_thz = wl_grid["banks"][bank_key]["channel_spacing_thz"]

        wavelengths = group["OSA_Wave(nm)"].values
        channels = group["Channel"].values
        frequencies_thz = (c / wavelengths) / 1e12

        measured_center_freq_thz = float(np.mean(frequencies_thz))
        center_freq_error_ghz = (measured_center_freq_thz - target_center_freq_thz) * 1000

        if len(frequencies_thz) >= 2:
            spacings_thz = np.diff(frequencies_thz)
            avg_spacing_thz = float(np.mean(spacings_thz))
            target_spacing_signed = -target_spacing_thz
            spacing_error_ghz = (avg_spacing_thz - target_spacing_signed) * 1000
            spacing_std_ghz = float(np.std(spacings_thz)) * 1000

            for i in range(len(spacings_thz)):
                spacing_ghz = spacings_thz[i] * 1000
                spacing_error = spacing_ghz - (target_spacing_signed * 1000)
                spacing_results.append(
                    {
                        "Tile_SN": tile_sn,
                        "Version": version,
                        "Bank": int(bank),
                        "Channel_From": int(channels[i]),
                        "Channel_To": int(channels[i + 1]),
                        "Spacing_GHz": spacing_ghz,
                        "Spacing_Error_GHz": spacing_error,
                    }
                )
        else:
            avg_spacing_thz = 0.0
            spacing_error_ghz = 0.0
            spacing_std_ghz = 0.0

        center_freq_results.append(
            {
                "Tile_SN": tile_sn,
                "Version": version,
                "Bank": int(bank),
                "Target_Center_Freq_THz": target_center_freq_thz,
                "Measured_Center_Freq_THz": measured_center_freq_thz,
                "Center_Freq_Error_GHz": center_freq_error_ghz,
                "Target_Spacing_THz": target_spacing_thz,
                "Measured_Spacing_THz": avg_spacing_thz,
                "Spacing_Error_GHz": spacing_error_ghz,
                "Spacing_Std_GHz": spacing_std_ghz,
            }
        )

    return pd.DataFrame(center_freq_results), pd.DataFrame(spacing_results)


def _bank_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            markeredgecolor="black",
            linewidth=1.5,
            label="Bank 0",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="red",
            markersize=10,
            markeredgecolor="black",
            linewidth=1.5,
            label="Bank 1",
        ),
    ]


def _save_single_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _ordered_tiles(df: pd.DataFrame) -> list[str]:
    v1 = sorted(df[df["Version"] == "v1"]["Tile_SN"].unique())
    v2 = sorted(df[df["Version"] == "v2"]["Tile_SN"].unique())
    return v1 + v2


def _horizontal_hist_with_points(
    ax,
    values: np.ndarray,
    color: str,
    ylo: float,
    yhi: float,
    *,
    bins: int = 28,
) -> None:
    """Histogram with performance (GHz) on y-axis, count on x-axis; overlay raw samples."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(ylo, yhi)
        return
    edges = np.linspace(ylo, yhi, bins + 1)
    n, _, _ = ax.hist(
        values,
        bins=edges,
        orientation="horizontal",
        color=color,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.35,
    )
    xmax = float(np.max(n)) if n.size else 1.0
    xmax = max(xmax, 1.0)
    jitter_x = np.random.uniform(0.0, 0.16 * xmax, size=len(values))
    ax.scatter(
        jitter_x,
        values,
        color="black",
        s=4,
        alpha=0.18,
        zorder=3,
        linewidths=0,
    )
    ax.set_xlim(0, xmax * 1.15)
    ax.set_ylim(ylo, yhi)


def plot_tp2p4_freq_error_tiles(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for frequency error (tiles) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(24, 8), layout="constrained")
    bank_colors = {0: "blue", 1: "red"}
    bank_markers = {0: "o", 1: "^"}
    all_tiles = _ordered_tiles(df)
    tile_offset = 0
    for version in ["v1", "v2"]:
        df_version = df[df["Version"] == version]
        tiles = sorted(df_version["Tile_SN"].unique())
        for tile_idx, tile in enumerate(tiles):
            for bank in [0, 1]:
                df_bank = df_version[df_version["Bank"] == bank]
                df_tile = df_bank[df_bank["Tile_SN"] == tile]
                for channel in range(8):
                    df_channel = df_tile[df_tile["Channel"] == channel]
                    if not df_channel.empty:
                        freq_errors = df_channel["Frequency_Error_GHz"].values
                        pos = (tile_offset + tile_idx) * 17 + bank * 8 + channel
                        x_scatter = np.random.normal(pos, 0.15, size=len(freq_errors))
                        ax.scatter(
                            x_scatter,
                            freq_errors,
                            color=bank_colors[bank],
                            alpha=0.7,
                            s=35,
                            marker=bank_markers[bank],
                            edgecolors="black",
                            linewidth=0.5,
                        )
        tile_offset += len(tiles)
    tile_positions = [(i * 17 + 7.5) for i in range(len(all_tiles))]
    ax.set_xticks(tile_positions)
    ax.set_xticklabels(all_tiles, rotation=90, fontsize=7)
    ax.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency Error (GHz)", fontsize=13, fontweight="bold")
    ax.set_ylim(-50, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_bank_legend(), loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    _save_single_figure(fig, output_path)


def plot_tp2p4_freq_error_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for frequency error (distribution) plot")
        return
    sns.set_style("whitegrid")
    ylo, yhi = -50.0, 50.0
    bank_colors = {0: "blue", 1: "red"}
    fig, axes = plt.subplots(8, 2, figsize=(11, 22), layout="constrained", sharey=True)
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        for ch in range(8):
            ax = axes[ch, bank]
            sub = df_bank[df_bank["Channel"] == ch]
            vals = sub["Frequency_Error_GHz"].values if not sub.empty else np.array([])
            _horizontal_hist_with_points(ax, vals, bank_colors[bank], ylo, yhi, bins=26)
            ax.grid(True, alpha=0.3, axis="both")
            ax.set_ylabel(f"Ch {ch}", fontsize=9)
            if ch < 7:
                ax.tick_params(labelbottom=False)
            if bank == 1:
                ax.tick_params(labelleft=False)
    fig.supylabel("Frequency error (GHz)", fontsize=12, fontweight="bold")
    fig.supxlabel("Count", fontsize=12, fontweight="bold")
    _save_single_figure(fig, output_path)


def plot_center_freq_error_tiles(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for center frequency error (tiles) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(24, 8), layout="constrained")
    bank_colors = {0: "blue", 1: "red"}
    bank_markers = {0: "o", 1: "^"}
    all_tiles = _ordered_tiles(df)
    tile_to_pos = {tile: i for i, tile in enumerate(all_tiles)}
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        x_pos = [tile_to_pos[tile] for tile in df_bank["Tile_SN"]]
        y_vals = df_bank["Center_Freq_Error_GHz"].values
        ax.scatter(
            x_pos,
            y_vals,
            color=bank_colors[bank],
            marker=bank_markers[bank],
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.8,
            label=f"Bank {bank}",
        )
    ax.set_xticks(range(len(all_tiles)))
    ax.set_xticklabels(all_tiles, rotation=90, fontsize=8)
    ax.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax.set_ylabel("Center Frequency Error (GHz)", fontsize=13, fontweight="bold")
    ax.set_ylim(-50, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_bank_legend(), loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    _save_single_figure(fig, output_path)


def plot_center_freq_error_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for center frequency error (distribution) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 10), layout="constrained")
    all_errors = df["Center_Freq_Error_GHz"].values
    _horizontal_hist_with_points(ax, all_errors, "purple", -50.0, 50.0, bins=32)
    ax.set_xlabel("Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("Center frequency error (GHz)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="both")
    _save_single_figure(fig, output_path)


def plot_channel_spacing_error_tiles(summary_df: pd.DataFrame, spacing_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty or spacing_df.empty:
        print("No data for channel spacing error (tiles) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(24, 8), layout="constrained")
    bank_colors = {0: "blue", 1: "red"}
    bank_markers = {0: "o", 1: "^"}
    all_tiles = _ordered_tiles(summary_df)
    tile_to_pos = {tile: i for i, tile in enumerate(all_tiles)}
    for bank in [0, 1]:
        df_bank = spacing_df[spacing_df["Bank"] == bank]
        for tile in all_tiles:
            df_tile = df_bank[df_bank["Tile_SN"] == tile]
            if not df_tile.empty:
                x_pos = tile_to_pos[tile]
                y_vals = df_tile["Spacing_Error_GHz"].values
                x_scatter = np.random.normal(x_pos, 0.15, size=len(y_vals))
                ax.scatter(
                    x_scatter,
                    y_vals,
                    color=bank_colors[bank],
                    marker=bank_markers[bank],
                    s=30,
                    alpha=0.5,
                    edgecolors="black",
                    linewidth=0.3,
                )
    ax.set_xticks(range(len(all_tiles)))
    ax.set_xticklabels(all_tiles, rotation=90, fontsize=8)
    ax.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax.set_ylabel("Channel Spacing Error (GHz)", fontsize=13, fontweight="bold")
    ax.set_ylim(-50, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_bank_legend(), loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    _save_single_figure(fig, output_path)


def plot_channel_spacing_error_distribution(spacing_df: pd.DataFrame, output_path: Path) -> None:
    if spacing_df.empty:
        print("No data for channel spacing error (distribution) plot")
        return
    sns.set_style("whitegrid")
    ylo, yhi = -50.0, 50.0
    bank_colors = {0: "blue", 1: "red"}
    fig, axes = plt.subplots(7, 2, figsize=(11, 19), layout="constrained", sharey=True)
    for bank in [0, 1]:
        df_bank = spacing_df[spacing_df["Bank"] == bank]
        for row in range(7):
            ax = axes[row, bank]
            sub = df_bank[(df_bank["Channel_From"] == row) & (df_bank["Channel_To"] == row + 1)]
            vals = sub["Spacing_Error_GHz"].values if not sub.empty else np.array([])
            _horizontal_hist_with_points(ax, vals, bank_colors[bank], ylo, yhi, bins=26)
            ax.grid(True, alpha=0.3, axis="both")
            ax.set_ylabel(f"Ch{row}→{row+1}", fontsize=9)
            if row < 6:
                ax.tick_params(labelbottom=False)
            if bank == 1:
                ax.tick_params(labelleft=False)
    fig.supylabel("Channel spacing error (GHz)", fontsize=12, fontweight="bold")
    fig.supxlabel("Count", fontsize=12, fontweight="bold")
    _save_single_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="TP2-4 plots from clm_data_onet_sftp (Scan CSVs).")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data root (default: monorepo data/clm_data_onet_sftp)")
    parser.add_argument(
        "--grid",
        type=Path,
        default=None,
        help="wavelength_grid.yaml (default: ../config/wavelength_grid.yaml next to this package)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Output directory (default: blueray_nevada_f2f/analysis/results)",
    )
    args = parser.parse_args()

    data_root = (args.data_root or default_data_root()).resolve()
    tp2p4_path = data_root / "TP2-4"
    grid_path = (args.grid or (default_config_dir() / "wavelength_grid.yaml")).resolve()
    results = (args.results or (Path(__file__).resolve().parent.parent / "results")).resolve()

    if not tp2p4_path.is_dir():
        raise SystemExit(f"TP2-4 folder not found: {tp2p4_path}")
    if not grid_path.is_file():
        raise SystemExit(f"wavelength grid not found: {grid_path}")

    with open(grid_path, "r", encoding="utf-8") as f:
        wl_grid = yaml.safe_load(f)

    print(f"Loading TP2-4 Scan data from {tp2p4_path} …")
    df = load_tp2p4_scan_data(tp2p4_path, wl_grid)
    if df.empty:
        raise SystemExit("No TP2-4 rows after load (check CSV paths and T_MUX filter).")

    print(f"Rows: {len(df)}, tiles: {df['Tile_SN'].nunique()}")
    plot_tp2p4_freq_error_tiles(df, results / "tp2p4_tile_vs_freq_error.png")
    plot_tp2p4_freq_error_distribution(df, results / "tp2p4_distribution_vs_freq_error.png")

    center_df, spacing_df = calculate_center_freq_spacing_errors(df, wl_grid)
    plot_center_freq_error_tiles(center_df, results / "tp2p4_tile_vs_center_freq_error.png")
    plot_center_freq_error_distribution(center_df, results / "tp2p4_distribution_vs_center_freq_error.png")
    plot_channel_spacing_error_tiles(center_df, spacing_df, results / "tp2p4_tile_vs_channel_spacing_error.png")
    plot_channel_spacing_error_distribution(spacing_df, results / "tp2p4_distribution_vs_channel_spacing_error.png")
    print("Done.")


if __name__ == "__main__":
    main()
