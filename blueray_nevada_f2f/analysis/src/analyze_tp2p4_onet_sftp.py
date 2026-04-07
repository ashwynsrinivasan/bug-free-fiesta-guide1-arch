#!/usr/bin/env python3
"""
TP2-4 analysis for CLM data under data/clm_data_onet_sftp.

Plots match the style and filenames used in
ips_clm_gen1/clm_mfg_data/analysis_results/ (tp2p4_*_summary.png):
  - tp2p4_freq_error_summary.png
  - tp2p4_center_freq_error_summary.png
  - tp2p4_channel_spacing_error_summary.png
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


def plot_tp2p4_freq_error(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for frequency error plot")
        return

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
    bank_colors = {0: "blue", 1: "red"}
    bank_markers = {0: "o", 1: "^"}

    ax_left = fig.add_subplot(gs[0, 0])
    v1_tiles = sorted(df[df["Version"] == "v1"]["Tile_SN"].unique())
    v2_tiles = sorted(df[df["Version"] == "v2"]["Tile_SN"].unique())
    all_tiles = v1_tiles + v2_tiles

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
                        ax_left.scatter(
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
    ax_left.set_xticks(tile_positions)
    ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=7)
    ax_left.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax_left.set_ylabel("Frequency Error (GHz)", fontsize=13, fontweight="bold")
    ax_left.set_title("Frequency Error by Tile", fontsize=14, fontweight="bold")
    ax_left.set_ylim(-50, 50)
    ax_left.grid(True, alpha=0.3)

    ax_right = fig.add_subplot(gs[0, 1])
    box_data = []
    box_positions = []
    box_colors = []
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        for channel in range(8):
            df_channel = df_bank[df_bank["Channel"] == channel]
            if not df_channel.empty:
                freq_errors = df_channel["Frequency_Error_GHz"].values
                y_pos = bank * 8 + channel
                box_data.append(freq_errors)
                box_positions.append(y_pos)
                box_colors.append(bank_colors[bank])

    parts = ax_right.violinplot(
        box_data,
        positions=box_positions,
        vert=False,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc, color in zip(parts["bodies"], box_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    ax_right.boxplot(
        box_data,
        positions=box_positions,
        vert=False,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black", linewidth=2),
        medianprops=dict(color="red", linewidth=2.5),
    )

    annotation_x = 30
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        for channel in range(8):
            df_channel = df_bank[df_bank["Channel"] == channel]
            if not df_channel.empty:
                freq_errors = df_channel["Frequency_Error_GHz"].values
                y_pos = bank * 8 + channel
                median = np.median(freq_errors)
                std = np.std(freq_errors)
                annotation_text = f"μ̃={median:.1f}GHz\nσ={std:.2f}GHz"
                ax_right.text(
                    annotation_x,
                    y_pos,
                    annotation_text,
                    fontsize=7,
                    ha="left",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor=bank_colors[bank],
                        alpha=0.8,
                        linewidth=1,
                    ),
                )

    all_freq_errors = df["Frequency_Error_GHz"].values
    overall_median = np.median(all_freq_errors)
    overall_std = np.std(all_freq_errors)
    yticks = list(range(16))
    yticklabels = [f"B0-Ch{i}" for i in range(8)] + [f"B1-Ch{i}" for i in range(8)]
    ax_right.set_yticks(yticks)
    ax_right.set_yticklabels(yticklabels, fontsize=9)
    ax_right.set_xlabel("Frequency Error (GHz)", fontsize=12, fontweight="bold")
    ax_right.set_ylabel("Bank-Channel", fontsize=12, fontweight="bold")
    ax_right.set_title(
        f"Statistical Distribution\nμ̃={overall_median:.2f}GHz, σ={overall_std:.2f}GHz",
        fontsize=13,
        fontweight="bold",
    )
    ax_right.grid(True, alpha=0.3, axis="x")
    ax_right.set_ylim(-0.5, 15.5)
    ax_right.set_xlim(-50, 50)
    ax_right.axhline(y=7.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)

    legend_elements = [
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
    ax_left.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_center_freq_error(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for center frequency error plot")
        return

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 0.38], wspace=0.25)
    bank_colors = {0: "blue", 1: "red"}
    bank_markers = {0: "o", 1: "^"}

    ax_left = fig.add_subplot(gs[0, 0])
    v1_tiles = sorted(df[df["Version"] == "v1"]["Tile_SN"].unique())
    v2_tiles = sorted(df[df["Version"] == "v2"]["Tile_SN"].unique())
    all_tiles = v1_tiles + v2_tiles
    tile_to_pos = {tile: i for i, tile in enumerate(all_tiles)}

    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        x_pos = [tile_to_pos[tile] for tile in df_bank["Tile_SN"]]
        y_vals = df_bank["Center_Freq_Error_GHz"].values
        ax_left.scatter(
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

    ax_left.set_xticks(range(len(all_tiles)))
    ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=8)
    ax_left.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax_left.set_ylabel("Center Frequency Error (GHz)", fontsize=13, fontweight="bold")
    ax_left.set_title("Center Frequency Error by Tile", fontsize=14, fontweight="bold")
    ax_left.set_ylim(-50, 50)
    ax_left.grid(True, alpha=0.3)

    ax_right = fig.add_subplot(gs[0, 1])
    all_errors = df["Center_Freq_Error_GHz"].values
    parts = ax_right.violinplot(
        [all_errors], positions=[0], vert=True, widths=0.7, showmeans=False, showmedians=False, showextrema=False
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("purple")
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    ax_right.boxplot(
        [all_errors],
        positions=[0],
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black", linewidth=2),
        medianprops=dict(color="red", linewidth=2.5),
    )

    overall_mean = np.mean(all_errors)
    overall_std = np.std(all_errors)
    annotation_y = 20 - (20 - (-20)) * 0.08
    annotation_text = f"μ={overall_mean:.2f}GHz\nσ={overall_std:.2f}GHz"
    ax_right.text(
        0.3,
        annotation_y,
        annotation_text,
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="purple", alpha=0.9, linewidth=2),
    )
    ax_right.set_xticks([0])
    ax_right.set_xticklabels(["Both Banks"], fontsize=11, fontweight="bold")
    ax_right.set_ylabel("Center Frequency Error (GHz)", fontsize=12, fontweight="bold")
    ax_right.set_title("Statistical Distribution", fontsize=13, fontweight="bold")
    ax_right.grid(True, alpha=0.3, axis="y")
    ax_right.set_ylim(-50, 50)
    ax_right.set_xlim(-0.5, 0.5)

    legend_elements = [
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
    ax_left.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_channel_spacing_error(summary_df: pd.DataFrame, spacing_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty or spacing_df.empty:
        print("No data for channel spacing error plot")
        return

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
    bank_colors = {0: "blue", 1: "red"}
    bank_markers = {0: "o", 1: "^"}

    ax_left = fig.add_subplot(gs[0, 0])
    v1_tiles = sorted(summary_df[summary_df["Version"] == "v1"]["Tile_SN"].unique())
    v2_tiles = sorted(summary_df[summary_df["Version"] == "v2"]["Tile_SN"].unique())
    all_tiles = v1_tiles + v2_tiles
    tile_to_pos = {tile: i for i, tile in enumerate(all_tiles)}

    for bank in [0, 1]:
        df_bank = spacing_df[spacing_df["Bank"] == bank]
        for tile in all_tiles:
            df_tile = df_bank[df_bank["Tile_SN"] == tile]
            if not df_tile.empty:
                x_pos = tile_to_pos[tile]
                y_vals = df_tile["Spacing_Error_GHz"].values
                x_scatter = np.random.normal(x_pos, 0.15, size=len(y_vals))
                ax_left.scatter(
                    x_scatter,
                    y_vals,
                    color=bank_colors[bank],
                    marker=bank_markers[bank],
                    s=30,
                    alpha=0.5,
                    edgecolors="black",
                    linewidth=0.3,
                )

    ax_left.set_xticks(range(len(all_tiles)))
    ax_left.set_xticklabels(all_tiles, rotation=90, fontsize=8)
    ax_left.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax_left.set_ylabel("Channel Spacing Error (GHz)", fontsize=13, fontweight="bold")
    ax_left.set_title("Channel Spacing Error by Tile", fontsize=14, fontweight="bold")
    ax_left.set_ylim(-50, 50)
    ax_left.grid(True, alpha=0.3)

    ax_right = fig.add_subplot(gs[0, 1])
    box_data = []
    box_positions = []
    box_colors = []
    for bank in [0, 1]:
        df_bank = spacing_df[spacing_df["Bank"] == bank]
        for ch_from in range(7):
            df_transition = df_bank[(df_bank["Channel_From"] == ch_from) & (df_bank["Channel_To"] == ch_from + 1)]
            if not df_transition.empty:
                errors = df_transition["Spacing_Error_GHz"].values
                y_pos = bank * 7 + ch_from
                box_data.append(errors)
                box_positions.append(y_pos)
                box_colors.append(bank_colors[bank])

    parts = ax_right.violinplot(
        box_data,
        positions=box_positions,
        vert=False,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc, color in zip(parts["bodies"], box_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)

    ax_right.boxplot(
        box_data,
        positions=box_positions,
        vert=False,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=2),
        capprops=dict(color="black", linewidth=2),
        medianprops=dict(color="red", linewidth=2.5),
    )

    annotation_x = 25
    for bank in [0, 1]:
        df_bank = spacing_df[spacing_df["Bank"] == bank]
        for ch_from in range(7):
            df_transition = df_bank[(df_bank["Channel_From"] == ch_from) & (df_bank["Channel_To"] == ch_from + 1)]
            if not df_transition.empty:
                errors = df_transition["Spacing_Error_GHz"].values
                y_pos = bank * 7 + ch_from
                mean_val = np.mean(errors)
                std_val = np.std(errors)
                annotation_text = f"μ={mean_val:.2f}GHz\nσ={std_val:.2f}GHz"
                ax_right.text(
                    annotation_x,
                    y_pos,
                    annotation_text,
                    fontsize=7,
                    ha="left",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor=bank_colors[bank],
                        alpha=0.8,
                        linewidth=1,
                    ),
                )

    all_errors = spacing_df["Spacing_Error_GHz"].values
    overall_mean = np.mean(all_errors)
    overall_std = np.std(all_errors)
    yticks = list(range(14))
    yticklabels = [f"B0: Ch{i}→Ch{i+1}" for i in range(7)] + [f"B1: Ch{i}→Ch{i+1}" for i in range(7)]
    ax_right.set_yticks(yticks)
    ax_right.set_yticklabels(yticklabels, fontsize=9)
    ax_right.set_xlabel("Channel Spacing Error (GHz)", fontsize=12, fontweight="bold")
    ax_right.set_ylabel("Channel Transition", fontsize=12, fontweight="bold")
    ax_right.set_title(
        f"Statistical Distribution\nμ={overall_mean:.2f}GHz, σ={overall_std:.2f}GHz",
        fontsize=13,
        fontweight="bold",
    )
    ax_right.grid(True, alpha=0.3, axis="x")
    ax_right.set_ylim(-0.5, 13.5)
    ax_right.set_xlim(-50, 50)
    ax_right.axhline(y=6.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)

    legend_elements = [
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
    ax_left.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


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
    plot_tp2p4_freq_error(df, results / "tp2p4_freq_error_summary.png")

    center_df, spacing_df = calculate_center_freq_spacing_errors(df, wl_grid)
    plot_center_freq_error(center_df, results / "tp2p4_center_freq_error_summary.png")
    plot_channel_spacing_error(center_df, spacing_df, results / "tp2p4_channel_spacing_error_summary.png")
    print("Done.")


if __name__ == "__main__":
    main()
