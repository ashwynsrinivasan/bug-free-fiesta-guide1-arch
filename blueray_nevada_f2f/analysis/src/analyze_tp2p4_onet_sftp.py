#!/usr/bin/env python3
"""
TP2-4 / TP2-5 Scan analysis for CLM data under data/clm_data_onet_sftp.

By default applies the same tile-level filters as ips_clm_gen1 (analysis_src/filter.yaml):
TP2-6 min power → TP2-5 total power @50C → TP2-5 frequency error @50C → TP2-4 spacing error.
Use --skip-filters to include all tiles that load (no filter.yaml cascade).

Writes separate tile-scatter and distribution figures per test point when data exists:
  - tp2p4_* / tp2p5_* (tile_vs_* and distribution_vs_* for freq, center freq, channel spacing).
  - tp2p5_totalpower_* (TP2-5 mean bank power @50C, gen1 total-power style; blueray split layout).
  - tp2p6_power_* (TP2-6 per-channel power, gen1 power summary style; blueray split layout).
Distribution views mirror ips_clm_gen1 TP2-4 right-panel layout: vertical box plots (GHz on y),
white boxes / black whiskers / red median (gen1 styling), μ̃–σ annotations inside plot above x-axis (tick-aligned), scatter overlay.
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


TILE_FIGSIZE = (30, 5)
DIST_FIGSIZE = (10, 5)
# Two-bank-only distribution panels (center freq, TP2-5 total power)
DIST_FIGSIZE_TWO_BANK = (5, 5)
# Dark scatter styling (distribution: drawn under boxplot via zorder)
TILE_SCATTER_BANK = {
    0: {"facecolor": "#1565c0", "edgecolor": "#0d1117", "linewidths": 0.65},
    1: {"facecolor": "#c62828", "edgecolor": "#0d1117", "linewidths": 0.65},
}
TILE_SCATTER_S_FREQ = 55
TILE_SCATTER_S_CENTER = 120
TILE_SCATTER_S_SPACING = 50
DIST_SCATTER_KW = dict(c="#0d1117", s=28, alpha=0.72, linewidths=0)
# Frequency error plots: fixed span and major ticks every 10 GHz
FREQ_ERROR_Y_TICKS = np.arange(-50, 51, 10)
# TP2-6 power distribution: 0–20 mW major ticks (gen1 summary y-limits)
TP2P6_POWER_Y_TICKS = np.arange(0, 21, 5)
# TP2-5 total power (fiber): gen1 uses 0–200 mW
TP2P5_TOTALPOWER_YLIM = (0.0, 200.0)
TP2P5_TOTALPOWER_Y_TICKS = np.arange(0, 201, 50)

_TILE_SN = re.compile(r"^Y\d{10}$")


def tile_sn_from_csv_path(csv_path: Path) -> str | None:
    for part in csv_path.stem.split("-"):
        if _TILE_SN.match(part):
            return part
    return None


def load_filters(config_dir: Path) -> dict:
    p = config_dir / "filter.yaml"
    if not p.is_file():
        alt = Path(__file__).resolve().parents[3] / "ips_clm_gen1" / "clm_mfg_data" / "analysis_src" / "filter.yaml"
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"filter.yaml not found under {config_dir} or {alt}")
    with open(p, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc["filters"]


def load_tp2p6_onet(tp6_path: Path) -> pd.DataFrame:
    """Load *TP2-6 Test.csv (laser on), same as ips_clm_gen1 _load_tp2p6_data."""
    all_data: list[pd.DataFrame] = []
    for csv_file in sorted(tp6_path.glob("*TP2-6 Test.csv")):
        try:
            df = pd.read_csv(csv_file)
            df = df[df["Set Laser(mA)"] > 0].copy()
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}", file=sys.stderr)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def load_tp2p6_onet_filtered(tp6_path: Path, valid_tiles: set[str] | None) -> pd.DataFrame:
    """TP2-6 laser-on rows; optional intersection with valid_tiles."""
    df = load_tp2p6_onet(tp6_path)
    if df.empty:
        return df
    if valid_tiles is not None:
        df = df[df["Tile_SN"].isin(valid_tiles)].copy()
    return df


def load_tp2p5_totalpower_onet(tp5_path: Path, valid_tiles: list[str] | None) -> pd.DataFrame:
    """Per gen1 _load_tp2p5_totalpower_data. If valid_tiles is None, keep all Tile_SN."""
    all_data: list[pd.DataFrame] = []
    for csv_file in sorted(tp5_path.glob("*TP2-5 Scan.csv")):
        try:
            df = pd.read_csv(csv_file)
            if valid_tiles is not None:
                df = df[df["Tile_SN"].isin(valid_tiles)].copy()
            if df.empty:
                continue
            df = df[(df["T_MUX(C)"] >= 49.9) & (df["T_MUX(C)"] <= 50.1)].copy()
            if df.empty:
                continue
            grouped = df.groupby(["Tile_SN", "Bank"])["Power(mW)"].mean().reset_index()
            grouped.rename(columns={"Power(mW)": "Total_Power_mW"}, inplace=True)
            all_data.append(grouped)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}", file=sys.stderr)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def _freq_error_nm_ghz_from_row(row: pd.Series, wl_grid: dict) -> pd.Series:
    """Target vs measured OSA wavelength → wavelength error (nm) and frequency error (GHz)."""
    c = 299792458 * 1e9
    bank = int(row["Bank"])
    channel = int(row["Channel"])
    measured_wl = row["OSA_Wave(nm)"]
    bank_key = f"bank{bank}"
    grid_num = channel + 1
    target_wl = wl_grid["banks"][bank_key]["grids"][grid_num]["wavelength_nm"]
    wl_error_nm = measured_wl - target_wl
    freq_error_hz = -(c / (target_wl**2)) * wl_error_nm
    freq_error_ghz = freq_error_hz / 1e9
    return pd.Series({"Wavelength_Error_nm": wl_error_nm, "Frequency_Error_GHz": freq_error_ghz})


def load_tp2p5_freq_onet(tp5_path: Path, wl_grid: dict, valid_tiles: list[str]) -> pd.DataFrame:
    """Per gen1 _load_tp2p5_data (Frequency_Error_GHz)."""
    all_data: list[pd.DataFrame] = []
    for csv_file in sorted(tp5_path.glob("*TP2-5 Scan.csv")):
        try:
            df = pd.read_csv(csv_file)
            df = df[df["Tile_SN"].isin(valid_tiles)].copy()
            if df.empty:
                continue
            df = df[(df["T_MUX(C)"] >= 49.9) & (df["T_MUX(C)"] <= 50.1)].copy()
            if df.empty:
                continue

            df = df.copy()
            df["Frequency_Error_GHz"] = df.apply(
                lambda r: _freq_error_nm_ghz_from_row(r, wl_grid)["Frequency_Error_GHz"], axis=1
            )
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}", file=sys.stderr)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def load_tp2p4_scan_data(tp_path: Path, wl_grid: dict, valid_tiles: set[str] | None = None) -> pd.DataFrame:
    """Load *TP2-4 Scan.csv files; filter T_MUX ~50C; per-channel freq error vs grid.
    If valid_tiles is set, keep only those Tile_SN (after load).
    """
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

            df[["Wavelength_Error_nm", "Frequency_Error_GHz"]] = df.apply(
                lambda r: _freq_error_nm_ghz_from_row(r, wl_grid), axis=1
            )
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
    out["Version"] = "v1"
    if valid_tiles is not None:
        out = out[out["Tile_SN"].isin(valid_tiles)].copy()
    return out


def load_tp2p5_scan_plot_data(tp_path: Path, wl_grid: dict, valid_tiles: set[str] | None = None) -> pd.DataFrame:
    """Load *TP2-5 Scan.csv; T_MUX ~50C; same schema as load_tp2p4_scan_data (Tile_SN from CSV)."""
    all_data: list[pd.DataFrame] = []
    for csv_file in sorted(tp_path.glob("*TP2-5 Scan.csv")):
        try:
            df = pd.read_csv(csv_file)
            if valid_tiles is not None:
                df = df[df["Tile_SN"].isin(valid_tiles)].copy()
            if df.empty:
                continue
            df = df[(df["T_MUX(C)"] >= 49.9) & (df["T_MUX(C)"] <= 50.1)].copy()
            if df.empty:
                continue
            df[["Wavelength_Error_nm", "Frequency_Error_GHz"]] = df.apply(
                lambda r: _freq_error_nm_ghz_from_row(r, wl_grid), axis=1
            )
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
    out["Version"] = "v1"
    return out


def _force_white_box_faces(bp: dict) -> None:
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)


def _raise_boxplot_zorder(bp: dict, z: float = 4.0) -> None:
    for key in ("boxes", "medians", "whiskers", "caps", "fliers"):
        if key not in bp:
            continue
        for artist in bp[key]:
            artist.set_zorder(z)


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


def get_valid_tiles_onet(data_root: Path, filters: dict, wl_grid: dict) -> set[str]:
    """Same cascade as ips_clm_gen1 tpanalysis._get_valid_tiles (single data tree)."""
    tp6 = data_root / "TP2-6"
    tp5 = data_root / "TP2-5"
    tp4 = data_root / "TP2-4"

    df_power = load_tp2p6_onet(tp6) if tp6.is_dir() else pd.DataFrame()
    if df_power.empty:
        print("  No TP2-6 data; no tiles pass filters.")
        return set()

    power_min = filters["optical_power"]["min_mw"]
    tile_min_power = df_power.groupby("Tile_SN")["Power(mW)"].min()
    tiles_pass_power = tile_min_power[tile_min_power >= power_min].index.tolist()

    df_total = load_tp2p5_totalpower_onet(tp5, tiles_pass_power) if tp5.is_dir() else pd.DataFrame()
    if not df_total.empty:
        total_power_min = filters["total_power"]["min_mw"]
        tile_min_total = df_total.groupby("Tile_SN")["Total_Power_mW"].min()
        tiles_pass_totalpower = tile_min_total[tile_min_total >= total_power_min].index.tolist()
    else:
        tiles_pass_totalpower = tiles_pass_power

    df_freq = load_tp2p5_freq_onet(tp5, wl_grid, tiles_pass_totalpower) if tp5.is_dir() else pd.DataFrame()
    if not df_freq.empty:
        fmin = filters["frequency_error"]["min_ghz"]
        fmax = filters["frequency_error"]["max_ghz"]
        tile_freq_valid = df_freq.groupby("Tile_SN")["Frequency_Error_GHz"].apply(
            lambda x: ((x >= fmin) & (x <= fmax)).all()
        )
        tiles_pass_freq = tile_freq_valid[tile_freq_valid].index.tolist()
    else:
        tiles_pass_freq = []

    df_tp2p4 = (
        load_tp2p4_scan_data(tp4, wl_grid, valid_tiles=set(tiles_pass_freq)) if tp4.is_dir() else pd.DataFrame()
    )
    if not df_tp2p4.empty:
        _, df_spacing = calculate_center_freq_spacing_errors(df_tp2p4, wl_grid)
        if not df_spacing.empty:
            spacing_max_abs = filters["channel_spacing_error"]["max_abs_ghz"]
            tile_spacing_valid = df_spacing.groupby("Tile_SN")["Spacing_Error_GHz"].apply(
                lambda x: (np.abs(x) <= spacing_max_abs).all()
            )
            tiles_pass_spacing = tile_spacing_valid[tile_spacing_valid].index.tolist()
        else:
            tiles_pass_spacing = tiles_pass_freq
    else:
        tiles_pass_spacing = tiles_pass_freq

    print(
        f"  Filter cascade: {len(tiles_pass_power)} power → {len(tiles_pass_totalpower)} total power → "
        f"{len(tiles_pass_freq)} freq → {len(tiles_pass_spacing)} all criteria"
    )
    return set(tiles_pass_spacing)


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


# Match ips_clm_gen1 analyze_test_points._plot_tp2p4_freq_error box styling (no violin).
_GEN1_BOXPLOT_KW = dict(
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
    whiskerprops=dict(color="black", linewidth=2),
    capprops=dict(color="black", linewidth=2),
    medianprops=dict(color="red", linewidth=2.5),
)


def _vertical_gen1_boxplot_with_scatter(
    ax,
    box_data: list[np.ndarray],
    box_positions: list[int],
    *,
    ylo: float,
    yhi: float,
    box_width: float = 0.55,
    use_median_in_annotation: bool = True,
    annotation_pad_frac: float = 0.055,
    annotation_fontsize: int = 7,
    annotation_unit: str = "GHz",
) -> None:
    """Vertical boxplots; scatter behind boxes; stats inside plot, x-aligned with ticks."""
    if not box_data:
        ax.set_ylim(ylo, yhi)
        return
    y_ann = ylo + (yhi - ylo) * annotation_pad_frac
    scatter_kw = {**DIST_SCATTER_KW, "zorder": 1}
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        jitter = pos + np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(jitter, vals, **scatter_kw)
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        vert=True,
        widths=box_width,
        **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    u = annotation_unit
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        if use_median_in_annotation:
            med = int(round(float(np.median(vals))))
            std = int(round(float(np.std(vals))))
            fmt = f"μ̃={med}{u}\nσ={std}{u}"
        else:
            mean_v = int(round(float(np.mean(vals))))
            std = int(round(float(np.std(vals))))
            fmt = f"μ={mean_v}{u}\nσ={std}{u}"
        ax.text(
            pos,
            y_ann,
            fmt,
            fontsize=annotation_fontsize,
            ha="center",
            va="bottom",
            zorder=6,
            clip_on=False,
        )


def plot_tp2p4_freq_error_tiles(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for frequency error (tiles) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=TILE_FIGSIZE, layout="constrained")
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
                        st = TILE_SCATTER_BANK[bank]
                        ax.scatter(
                            x_scatter,
                            freq_errors,
                            s=TILE_SCATTER_S_FREQ,
                            marker=bank_markers[bank],
                            alpha=0.88,
                            **st,
                        )
        tile_offset += len(tiles)
    tile_positions = [(i * 17 + 7.5) for i in range(len(all_tiles))]
    ax.set_xticks(tile_positions)
    ax.set_xticklabels(all_tiles, rotation=90, fontsize=7)
    ax.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency Error (GHz)", fontsize=13, fontweight="bold")
    ax.set_ylim(-50, 50)
    ax.set_yticks(FREQ_ERROR_Y_TICKS)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_bank_legend(), loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    _save_single_figure(fig, output_path)


def plot_tp2p4_freq_error_distribution(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for frequency error (distribution) plot")
        return
    ylo, yhi = -50.0, 50.0
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        for channel in range(8):
            sub = df_bank[df_bank["Channel"] == channel]
            if sub.empty:
                continue
            box_data.append(sub["Frequency_Error_GHz"].values)
            box_positions.append(bank * 8 + channel)
    if not box_data:
        print("No data for frequency error (distribution) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_with_scatter(
        ax,
        box_data,
        box_positions,
        ylo=ylo,
        yhi=yhi,
        box_width=0.55,
        use_median_in_annotation=True,
    )
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency Error (GHz)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels([f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)], fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(FREQ_ERROR_Y_TICKS)
    ax.set_xlim(-0.5, 15.5)
    _save_single_figure(fig, output_path)


def plot_center_freq_error_tiles(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for center frequency error (tiles) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=TILE_FIGSIZE, layout="constrained")
    bank_markers = {0: "o", 1: "^"}
    all_tiles = _ordered_tiles(df)
    tile_to_pos = {tile: i for i, tile in enumerate(all_tiles)}
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        x_pos = [tile_to_pos[tile] for tile in df_bank["Tile_SN"]]
        y_vals = df_bank["Center_Freq_Error_GHz"].values
        st = TILE_SCATTER_BANK[bank]
        ax.scatter(
            x_pos,
            y_vals,
            marker=bank_markers[bank],
            s=TILE_SCATTER_S_CENTER,
            alpha=0.88,
            **st,
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
    ylo, yhi = -50.0, 50.0
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    x_labels: list[str] = []
    for bank in [0, 1]:
        sub = df[df["Bank"] == bank]["Center_Freq_Error_GHz"].values
        if sub.size == 0:
            continue
        box_data.append(sub)
        box_positions.append(bank)
        x_labels.append("Bank A" if bank == 0 else "Bank B")
    if not box_data:
        print("No data for center frequency error (distribution) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE_TWO_BANK, layout="constrained")
    _vertical_gen1_boxplot_with_scatter(
        ax,
        box_data,
        box_positions,
        ylo=ylo,
        yhi=yhi,
        box_width=0.45,
        use_median_in_annotation=False,
        annotation_pad_frac=0.055,
        annotation_fontsize=8,
    )
    ax.set_xticks(box_positions)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Bank", fontsize=12, fontweight="bold")
    ax.set_ylabel("Center Frequency Error (GHz)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(FREQ_ERROR_Y_TICKS)
    ax.set_xlim(min(box_positions) - 0.5, max(box_positions) + 0.5)
    _save_single_figure(fig, output_path)


def plot_channel_spacing_error_tiles(summary_df: pd.DataFrame, spacing_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty or spacing_df.empty:
        print("No data for channel spacing error (tiles) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=TILE_FIGSIZE, layout="constrained")
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
                st = TILE_SCATTER_BANK[bank]
                ax.scatter(
                    x_scatter,
                    y_vals,
                    marker=bank_markers[bank],
                    s=TILE_SCATTER_S_SPACING,
                    alpha=0.88,
                    **st,
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
    ylo, yhi = -50.0, 50.0
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for bank in [0, 1]:
        df_bank = spacing_df[spacing_df["Bank"] == bank]
        for ch_from in range(7):
            sub = df_bank[(df_bank["Channel_From"] == ch_from) & (df_bank["Channel_To"] == ch_from + 1)]
            if sub.empty:
                continue
            box_data.append(sub["Spacing_Error_GHz"].values)
            box_positions.append(bank * 7 + ch_from)
    if not box_data:
        print("No data for channel spacing error (distribution) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_with_scatter(
        ax,
        box_data,
        box_positions,
        ylo=ylo,
        yhi=yhi,
        box_width=0.55,
        use_median_in_annotation=False,
        annotation_pad_frac=0.07,
        annotation_fontsize=6,
    )
    ax.set_xlabel("Channel transition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Channel Spacing Error (GHz)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(14)))
    ax.set_xticklabels(
        [f"Bank A:\nCh{i} --> Ch{i+1}" for i in range(1, 8)] + [f"Bank B:\nCh{i} --> Ch{i+1}" for i in range(1, 8)],
        fontsize=8,
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-0.5, 13.5)
    _save_single_figure(fig, output_path)


def plot_tp2p5_totalpower_tiles(df: pd.DataFrame, output_path: Path) -> None:
    """Per gen1 _plot_tp2p5_totalpower left panel: mean bank power @50C vs tile (onet: single Version)."""
    if df.empty:
        print("No data for TP2-5 total power (tiles) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=TILE_FIGSIZE, layout="constrained")
    bank_markers = {0: "o", 1: "^"}
    all_tiles = _ordered_tiles(df)
    tile_offset = 0
    for version in ["v1", "v2"]:
        df_version = df[df["Version"] == version]
        tiles = sorted(df_version["Tile_SN"].unique())
        for tile_idx, tile in enumerate(tiles):
            for bank in [0, 1]:
                df_tile_bank = df_version[(df_version["Tile_SN"] == tile) & (df_version["Bank"] == bank)]
                if not df_tile_bank.empty:
                    powers = df_tile_bank["Total_Power_mW"].values
                    pos = (tile_offset + tile_idx) * 3 + bank
                    x_scatter = np.random.normal(pos, 0.1, size=len(powers))
                    st = TILE_SCATTER_BANK[bank]
                    ax.scatter(
                        x_scatter,
                        powers,
                        s=50,
                        marker=bank_markers[bank],
                        alpha=0.88,
                        **st,
                    )
        tile_offset += len(tiles)
    tile_positions = [(i * 3 + 0.5) for i in range(len(all_tiles))]
    ax.set_xticks(tile_positions)
    ax.set_xticklabels(all_tiles, rotation=90, fontsize=7)
    ax.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total power in fiber (mW)", fontsize=13, fontweight="bold")
    ylo, yhi = TP2P5_TOTALPOWER_YLIM
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(TP2P5_TOTALPOWER_Y_TICKS)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_bank_legend(), loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    _save_single_figure(fig, output_path)


def plot_tp2p5_totalpower_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """TP2-5 total power @50C: one vertical box + scatter per bank (blueray styling)."""
    if df.empty:
        print("No data for TP2-5 total power (distribution) plot")
        return
    ylo, yhi = TP2P5_TOTALPOWER_YLIM
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    x_labels: list[str] = []
    for bank in [0, 1]:
        sub = df[df["Bank"] == bank]["Total_Power_mW"].values
        if sub.size == 0:
            continue
        box_data.append(sub)
        box_positions.append(bank)
        x_labels.append("Bank A" if bank == 0 else "Bank B")
    if not box_data:
        print("No data for TP2-5 total power (distribution) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE_TWO_BANK, layout="constrained")
    _vertical_gen1_boxplot_with_scatter(
        ax,
        box_data,
        box_positions,
        ylo=ylo,
        yhi=yhi,
        box_width=0.45,
        use_median_in_annotation=True,
        annotation_pad_frac=0.055,
        annotation_fontsize=8,
        annotation_unit="mW",
    )
    ax.set_xticks(box_positions)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Bank", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total power in fiber (mW)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(TP2P5_TOTALPOWER_Y_TICKS)
    ax.set_xlim(min(box_positions) - 0.5, max(box_positions) + 0.5)
    _save_single_figure(fig, output_path)


def plot_tp2p6_power_tiles(df: pd.DataFrame, output_path: Path) -> None:
    """Per gen1 _plot_tp2p6_power_combined left panel: per-channel power vs tile."""
    if df.empty:
        print("No data for TP2-6 power (tiles) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=TILE_FIGSIZE, layout="constrained")
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
                        powers = df_channel["Power(mW)"].values
                        pos = (tile_offset + tile_idx) * 17 + bank * 8 + channel
                        x_scatter = np.random.normal(pos, 0.15, size=len(powers))
                        st = TILE_SCATTER_BANK[bank]
                        ax.scatter(
                            x_scatter,
                            powers,
                            s=TILE_SCATTER_S_FREQ,
                            marker=bank_markers[bank],
                            alpha=0.88,
                            **st,
                        )
        tile_offset += len(tiles)
    tile_positions = [(i * 17 + 7.5) for i in range(len(all_tiles))]
    ax.set_xticks(tile_positions)
    ax.set_xticklabels(all_tiles, rotation=90, fontsize=7)
    ax.set_xlabel("Tile_SN", fontsize=13, fontweight="bold")
    ax.set_ylabel("Power (mW)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 20)
    ax.set_yticks(TP2P6_POWER_Y_TICKS)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_bank_legend(), loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    _save_single_figure(fig, output_path)


def plot_tp2p6_power_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Per gen1 _plot_tp2p6_power_combined right panel: vertical boxes by bank-channel (blueray style)."""
    if df.empty:
        print("No data for TP2-6 power (distribution) plot")
        return
    ylo, yhi = 0.0, 20.0
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        for channel in range(8):
            sub = df_bank[df_bank["Channel"] == channel]
            if sub.empty:
                continue
            box_data.append(sub["Power(mW)"].values)
            box_positions.append(bank * 8 + channel)
    if not box_data:
        print("No data for TP2-6 power (distribution) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_with_scatter(
        ax,
        box_data,
        box_positions,
        ylo=ylo,
        yhi=yhi,
        box_width=0.55,
        use_median_in_annotation=True,
        annotation_unit="mW",
    )
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Power (mW)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels([f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)], fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(TP2P6_POWER_Y_TICKS)
    ax.set_xlim(-0.5, 15.5)
    _save_single_figure(fig, output_path)


def _emit_scan_plots(df: pd.DataFrame, wl_grid: dict, results: Path, tp_label: str, file_prefix: str) -> None:
    """Six figures: per-channel freq, center freq, spacing (tiles + distribution)."""
    if df.empty:
        print(f"No data for {tp_label}; skipping figures.")
        return
    print(f"{tp_label}: rows {len(df)}, tiles {df['Tile_SN'].nunique()}")
    plot_tp2p4_freq_error_tiles(df, results / f"{file_prefix}_tile_vs_freq_error.png")
    plot_tp2p4_freq_error_distribution(df, results / f"{file_prefix}_distribution_vs_freq_error.png")
    center_df, spacing_df = calculate_center_freq_spacing_errors(df, wl_grid)
    plot_center_freq_error_tiles(center_df, results / f"{file_prefix}_tile_vs_center_freq_error.png")
    plot_center_freq_error_distribution(center_df, results / f"{file_prefix}_distribution_vs_center_freq_error.png")
    plot_channel_spacing_error_tiles(center_df, spacing_df, results / f"{file_prefix}_tile_vs_channel_spacing_error.png")
    plot_channel_spacing_error_distribution(spacing_df, results / f"{file_prefix}_distribution_vs_channel_spacing_error.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TP2-4/5 Scan, TP2-5 total power, and TP2-6 power plots from clm_data_onet_sftp."
    )
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
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Do not apply ips_clm_gen1 filter.yaml (use all Scan rows at T_MUX ~50C that load).",
    )
    args = parser.parse_args()

    data_root = (args.data_root or default_data_root()).resolve()
    tp2p4_path = data_root / "TP2-4"
    tp2p5_path = data_root / "TP2-5"
    tp2p6_path = data_root / "TP2-6"
    config_dir = default_config_dir()
    grid_path = (args.grid or (config_dir / "wavelength_grid.yaml")).resolve()
    results = (args.results or (Path(__file__).resolve().parent.parent / "results")).resolve()

    if not tp2p4_path.is_dir() and not tp2p5_path.is_dir() and not tp2p6_path.is_dir():
        raise SystemExit(f"No TP2-4, TP2-5, or TP2-6 folder found under {data_root}")
    if not grid_path.is_file():
        raise SystemExit(f"wavelength grid not found: {grid_path}")

    with open(grid_path, "r", encoding="utf-8") as f:
        wl_grid = yaml.safe_load(f)

    if args.skip_filters:
        valid_tiles: set[str] | None = None
        print("Skipping filter.yaml cascade; loading all Scan rows (T_MUX ~50C).")
    else:
        filters = load_filters(config_dir)
        print("Applying ips_clm_gen1 filter.yaml criteria (TP2-6 → TP2-5 → TP2-4 spacing) …")
        valid_tiles = get_valid_tiles_onet(data_root, filters, wl_grid)
        if not valid_tiles:
            raise SystemExit(
                "No tiles passed all filters. Use --skip-filters to plot unfiltered data, or check TP2-4/5/6 paths."
            )

    if tp2p4_path.is_dir():
        print(f"Loading TP2-4 Scan data from {tp2p4_path} …")
        df4 = load_tp2p4_scan_data(tp2p4_path, wl_grid, valid_tiles=valid_tiles)
        _emit_scan_plots(df4, wl_grid, results, "TP2-4", "tp2p4")
    else:
        print(f"No TP2-4 folder at {tp2p4_path}; skipping TP2-4 figures.")

    if tp2p5_path.is_dir():
        print(f"Loading TP2-5 Scan data from {tp2p5_path} …")
        df5 = load_tp2p5_scan_plot_data(tp2p5_path, wl_grid, valid_tiles=valid_tiles)
        _emit_scan_plots(df5, wl_grid, results, "TP2-5", "tp2p5")

        print(f"Loading TP2-5 total power @50C from {tp2p5_path} …")
        vt_list = sorted(valid_tiles) if valid_tiles is not None else None
        df_tp5_tot = load_tp2p5_totalpower_onet(tp2p5_path, vt_list)
        if df_tp5_tot.empty:
            print("No TP2-5 total-power rows after load; skipping tp2p5_totalpower_* figures.")
        else:
            df_tp5_tot = df_tp5_tot.copy()
            df_tp5_tot["Version"] = "v1"
            print(
                f"TP2-5 total power: {len(df_tp5_tot)} bank-rows, tiles {df_tp5_tot['Tile_SN'].nunique()}"
            )
            plot_tp2p5_totalpower_tiles(df_tp5_tot, results / "tp2p5_totalpower_tile_vs_power.png")
            plot_tp2p5_totalpower_distribution(
                df_tp5_tot, results / "tp2p5_totalpower_distribution_vs_power.png"
            )
    else:
        print(f"No TP2-5 folder at {tp2p5_path}; skipping TP2-5 Scan and total-power figures.")

    if tp2p6_path.is_dir():
        print(f"Loading TP2-6 Test data from {tp2p6_path} …")
        df6 = load_tp2p6_onet_filtered(tp2p6_path, valid_tiles)
        if df6.empty:
            print("No TP2-6 rows after load; skipping tp2p6_power_* figures.")
        else:
            df6 = df6.copy()
            df6["Version"] = "v1"
            print(f"TP2-6: rows {len(df6)}, tiles {df6['Tile_SN'].nunique()}")
            plot_tp2p6_power_tiles(df6, results / "tp2p6_power_tile_vs_power.png")
            plot_tp2p6_power_distribution(df6, results / "tp2p6_power_distribution_vs_power.png")
    else:
        print(f"No TP2-6 folder at {tp2p6_path}; skipping TP2-6 power figures.")

    print("Done.")


if __name__ == "__main__":
    main()
