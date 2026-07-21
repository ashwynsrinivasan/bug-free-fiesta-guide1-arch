#!/usr/bin/env python3
"""
TP1-4 LIV analysis for Gen2 CLM data under data/clm_data_onet_sftp/TP1-4.

Loads *TP1-4 Laser.csv files, filters to Set Laser(mA)==150, computes per-channel
frequency error from PeakWave(nm) vs the DWDM grid (wavelength_grid.yaml from
nevada_l_tile/config), and writes 4 distribution figures to nevada_cos/analysis/results/liv/:

  tp1p4_distribution_vs_freq_error.png
      16-column per-channel frequency error (A-Ch1..8, B-Ch1..8); ±50 GHz.

  tp1p4_distribution_vs_freq_error_combined_banks.png
      All channels pooled into one box+violin; ±50 GHz.

  tp1p4_distribution_vs_optical_power.png
      16-column per-channel optical power at 150 mA; 0–25 mW.

  tp1p4_distribution_vs_optical_power_combined_banks.png
      All channels pooled optical power; 0–25 mW.

Styling follows the gen1/blueray_nevada_rfp convention: white box, black outline,
red median, grey violin behind each box, μ̃/σ annotations inside the axes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def default_data_root() -> Path:
    # .../blueray_nevada_rfp/nevada_cos/analysis/src/thisfile.py → parents[6] = repo root
    return Path(__file__).resolve().parents[6] / "data" / "clm_data_onet_sftp"


def default_config_dir() -> Path:
    # Reuse wavelength_grid.yaml from nevada_l_tile
    return Path(__file__).resolve().parents[3] / "nevada_l_tile" / "config"


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "liv"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_wavelength_grid(config_dir: Path) -> dict:
    p = config_dir / "wavelength_grid.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"wavelength_grid.yaml not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tp1p4_at_150ma(tp14_path: Path, wl_grid: dict) -> pd.DataFrame:
    """Load *TP1-4 Laser.csv, keep Set Laser(mA)==150, compute Frequency_Error_GHz."""
    parts: list[pd.DataFrame] = []
    csv_files = sorted(tp14_path.glob("*TP1-4 Laser.csv"))
    if not csv_files:
        print(f"No *TP1-4 Laser.csv files found under {tp14_path}", file=sys.stderr)
        return pd.DataFrame()

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}", file=sys.stderr)
            continue

        df = df[df["Set Laser(mA)"] == 150].copy()
        if df.empty:
            continue

        df["Frequency_Error_GHz"] = df.apply(
            lambda r: _freq_error_from_peak_wave(r, wl_grid), axis=1
        )
        df["Power(dBm)"] = 10.0 * np.log10(df["Power(mW)"].clip(lower=1e-9))
        parts.append(
            df[["Tile_SN", "Bank", "Channel", "Power(mW)", "Power(dBm)", "Frequency_Error_GHz"]].copy()
        )

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def drop_outlier_tiles(df: pd.DataFrame, freq_error_limit_ghz: float = 50.0) -> pd.DataFrame:
    """Exclude tiles where any channel has |Frequency_Error_GHz| > freq_error_limit_ghz.

    Catches tiles with dead/non-lasing channels (garbage wavelength → huge freq error)
    and genuinely out-of-spec tiles. Prints excluded tile SNs.
    """
    bad_mask = df.groupby("Tile_SN")["Frequency_Error_GHz"].transform(
        lambda x: (x.abs() > freq_error_limit_ghz).any()
    )
    excluded = sorted(df.loc[bad_mask, "Tile_SN"].unique())
    if excluded:
        print(f"Excluding {len(excluded)} tile(s) with |freq error| > {freq_error_limit_ghz} GHz: {excluded}")
    return df[~bad_mask].copy()


def drop_low_power_tiles(df: pd.DataFrame, min_power_mw: float = 20.0) -> pd.DataFrame:
    """Exclude tiles where any channel has Power(mW) < min_power_mw.

    Removes tiles with non-lasing or severely underperforming channels whose power
    measurements are outliers relative to the bulk population.
    """
    bad_mask = df.groupby("Tile_SN")["Power(mW)"].transform(
        lambda x: (x < min_power_mw).any()
    )
    excluded = sorted(df.loc[bad_mask, "Tile_SN"].unique())
    if excluded:
        print(f"Excluding {len(excluded)} tile(s) with any channel < {min_power_mw} mW: {excluded}")
    return df[~bad_mask].copy()


def _freq_error_from_peak_wave(row: pd.Series, wl_grid: dict) -> float:
    """Frequency error (GHz) for one row: target grid vs PeakWave(nm)."""
    c = 299792458 * 1e9  # nm/s
    bank = int(row["Bank"])
    channel = int(row["Channel"])
    target_wl = wl_grid["banks"][f"bank{bank}"]["grids"][channel + 1]["wavelength_nm"]
    wl_error = row["PeakWave(nm)"] - target_wl
    return -(c / target_wl**2) * wl_error / 1e9


# ---------------------------------------------------------------------------
# Frequency correction
# ---------------------------------------------------------------------------

def apply_per_channel_mean_correction(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Subtract each channel's mean frequency error from its own measurements.

    Simulates trimming every laser channel to its target frequency by its systematic
    mean offset (the μ̃ shown in the distribution plot).  Returns the corrected
    DataFrame and a dict of {(bank, channel): mean_ghz} for reference.
    """
    df = df.copy()
    means: dict[tuple[int, int], float] = {}
    for bank in [0, 1]:
        for channel in range(8):
            mask = (df["Bank"] == bank) & (df["Channel"] == channel)
            mean_val = float(df.loc[mask, "Frequency_Error_GHz"].mean())
            means[(bank, channel)] = mean_val
            df.loc[mask, "Frequency_Error_GHz"] -= mean_val
    return df, means


# ---------------------------------------------------------------------------
# Plot styling (mirror of analyze_tp2p4_onet_sftp conventions)
# ---------------------------------------------------------------------------

DIST_FIGSIZE = (10, 5)
COMBINED_BANKS_FIGSIZE = (3, 4)
FREQ_ERROR_Y_TICKS = np.arange(-100, 101, 20)
POWER_DBM_Y_TICKS = np.arange(10, 21, 2)          # 10–20 dBm, 2 dBm steps
POWER_UNIFORMITY_DB_Y_TICKS = np.arange(0, 3.1, 0.5)  # 0–3 dB, 0.5 dB steps

_GEN1_BOXPLOT_KW = dict(
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
    whiskerprops=dict(color="black", linewidth=2),
    capprops=dict(color="black", linewidth=2),
    medianprops=dict(color="red", linewidth=2.5),
)


def _force_white_box_faces(bp: dict) -> None:
    for patch in bp.get("boxes", []):
        patch.set_facecolor("white")


def _raise_boxplot_zorder(bp: dict, z: float) -> None:
    for key in ("boxes", "medians", "whiskers", "caps", "fliers"):
        for artist in bp.get(key, []):
            artist.set_zorder(z)


def _grey_violin_behind_box(ax, values: np.ndarray, *, position: float = 0.0, width: float = 0.62) -> None:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return
    vp = ax.violinplot(
        [vals],
        positions=[position],
        vert=True,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#9e9e9e")
        body.set_edgecolor("#616161")
        body.set_alpha(0.48)
        body.set_zorder(0.4)
    for key in ("cbars", "cmins", "cmaxes"):
        if key in vp:
            for ln in vp[key]:
                ln.set_visible(False)


def _vertical_gen1_boxplot_with_violin(
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
    annotation_decimals: int = 0,
    violin_width: float = 0.62,
) -> None:
    """Vertical boxplots with grey violin behind each; μ̃/σ annotations inside axes."""
    if not box_data:
        ax.set_ylim(ylo, yhi)
        return
    y_ann = ylo + (yhi - ylo) * annotation_pad_frac
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        _grey_violin_behind_box(ax, vals, position=float(pos), width=violin_width)
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        vert=True,
        widths=box_width,
        **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    fmt = f".{annotation_decimals}f"
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        if use_median_in_annotation:
            c_val = float(np.median(vals))
            sym = "μ̃"
        else:
            c_val = float(np.mean(vals))
            sym = "μ"
        std = float(np.std(vals))
        u = annotation_unit
        ax.text(
            pos,
            y_ann,
            f"{sym}={c_val:{fmt}}{u}\nσ={std:{fmt}}{u}",
            fontsize=annotation_fontsize,
            ha="center",
            va="bottom",
            zorder=6,
            clip_on=False,
        )


def _plot_combined_banks_distribution(
    output_path: Path,
    arr: np.ndarray,
    *,
    ylo: float,
    yhi: float,
    yticks: np.ndarray,
    ylabel: str,
    use_mean_for_annotation: bool,
    annotation_unit: str,
    empty_msg: str,
) -> None:
    """Single pooled column: grey violin behind gen1 box; μ̃ or μ and σ in title."""
    vals = np.asarray(arr, dtype=float)
    if vals.size == 0:
        print(empty_msg)
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=COMBINED_BANKS_FIGSIZE, layout="constrained")
    pos = 0.0
    _grey_violin_behind_box(ax, vals, position=pos)
    bp = ax.boxplot([vals], positions=[pos], vert=True, widths=0.35, **_GEN1_BOXPLOT_KW)
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    if use_mean_for_annotation:
        c_raw = float(np.mean(vals))
        sym = "μ"
    else:
        c_raw = float(np.median(vals))
        sym = "μ̃"
    std_raw = float(np.std(vals))
    ax.set_title(
        f"{sym}={c_raw:.2g}{annotation_unit}, σ={std_raw:.2g}{annotation_unit}",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xticks([])
    _save_figure(fig, output_path)


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_tp1p4_freq_error_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """16-column per-channel frequency error distribution (A-Ch1..8, B-Ch1..8)."""
    if df.empty:
        print("No data for tp1p4 frequency error distribution plot")
        return
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        for channel in range(8):
            sub = df_bank[df_bank["Channel"] == channel]["Frequency_Error_GHz"].values
            if sub.size == 0:
                continue
            box_data.append(sub)
            box_positions.append(bank * 8 + channel)
    if not box_data:
        print("No data for tp1p4 frequency error distribution plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_with_violin(
        ax, box_data, box_positions, ylo=-100.0, yhi=100.0, use_median_in_annotation=True
    )
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency Error (GHz)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels(
        [f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)],
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-100, 100)
    ax.set_yticks(FREQ_ERROR_Y_TICKS)
    ax.set_xlim(-0.5, 15.5)
    _save_figure(fig, output_path)


def plot_tp1p4_freq_error_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    """All channels pooled into one box+violin; ±50 GHz."""
    if df.empty:
        print("No data for tp1p4 frequency error combined banks plot")
        return
    _plot_combined_banks_distribution(
        output_path,
        df["Frequency_Error_GHz"].values,
        ylo=-50.0,
        yhi=50.0,
        yticks=FREQ_ERROR_Y_TICKS,
        ylabel="Frequency Error (GHz)",
        use_mean_for_annotation=False,
        annotation_unit="GHz",
        empty_msg="No data for tp1p4 frequency error combined banks plot",
    )


def plot_tp1p4_optical_power_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """16-column per-channel optical power at 150 mA in dBm."""
    if df.empty:
        print("No data for tp1p4 optical power distribution plot")
        return
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for bank in [0, 1]:
        df_bank = df[df["Bank"] == bank]
        for channel in range(8):
            sub = df_bank[df_bank["Channel"] == channel]["Power(dBm)"].values
            if sub.size == 0:
                continue
            box_data.append(sub)
            box_positions.append(bank * 8 + channel)
    if not box_data:
        print("No data for tp1p4 optical power distribution plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_with_violin(
        ax,
        box_data,
        box_positions,
        ylo=10.0,
        yhi=20.0,
        use_median_in_annotation=False,
        annotation_unit="dBm",
        annotation_decimals=1,
    )
    ax.set_xlabel("Bank channel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Optical Power (dBm) @ 150 mA", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels(
        [f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)],
        fontsize=9,
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(10, 20)
    ax.set_yticks(POWER_DBM_Y_TICKS)
    ax.set_xlim(-0.5, 15.5)
    _save_figure(fig, output_path)


def plot_tp1p4_optical_power_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    """All channels pooled optical power in dBm."""
    if df.empty:
        print("No data for tp1p4 optical power combined banks plot")
        return
    _plot_combined_banks_distribution(
        output_path,
        df["Power(dBm)"].values,
        ylo=10.0,
        yhi=20.0,
        yticks=POWER_DBM_Y_TICKS,
        ylabel="Optical Power (dBm) @ 150 mA",
        use_mean_for_annotation=True,
        annotation_unit="dBm",
        empty_msg="No data for tp1p4 optical power combined banks plot",
    )


FREQ_UNIFORMITY_Y_TICKS   = np.arange(0, 101, 20)   # 0–100 GHz, 20 GHz steps
CHANNEL_SPACING_Y_TICKS   = np.arange(-35, 21, 5)    # GHz  (-35 … +20)

_SPACING_LABELS = (
    [f"A{i}-A{i+1}" for i in range(1, 8)] +
    [f"B{i}-B{i+1}" for i in range(1, 8)]
)


def compute_channel_spacing_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute adjacent-channel frequency spacing errors within each bank per tile.

    spacing_error(bank, ch_i → ch_i+1) = freq_error(ch_i+1) − freq_error(ch_i)

    Returns DataFrame with columns: Tile_SN, Position (0–13), Spacing_Error_GHz.
    Position 0–6 = bank-A pairs (A1-A2 … A7-A8), 7–13 = bank-B pairs.
    """
    pivot = (
        df.groupby(["Tile_SN", "Bank", "Channel"])["Frequency_Error_GHz"]
        .mean()
        .reset_index()
    )
    rows: list[dict] = []
    for (tile_sn, bank), g in pivot.groupby(["Tile_SN", "Bank"]):
        fe = g.set_index("Channel")["Frequency_Error_GHz"].sort_index()
        for ch in range(7):
            if ch in fe.index and ch + 1 in fe.index:
                rows.append({
                    "Tile_SN": tile_sn,
                    "Position": bank * 7 + ch,
                    "Spacing_Error_GHz": float(fe[ch + 1]) - float(fe[ch]),
                })
    return pd.DataFrame(rows)


def plot_tp1p4_channel_spacing_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """14-column adjacent-channel spacing error distribution (A1-A2 … B7-B8)."""
    if df.empty:
        print("No data for channel spacing error plot")
        return
    sp = compute_channel_spacing_errors(df)
    if sp.empty:
        return
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for pos in range(14):
        vals = sp[sp["Position"] == pos]["Spacing_Error_GHz"].values
        if vals.size == 0:
            continue
        box_data.append(vals)
        box_positions.append(pos)
    if not box_data:
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_with_violin(
        ax, box_data, box_positions,
        ylo=-35.0, yhi=20.0,
        use_median_in_annotation=True,
        annotation_unit="GHz",
        annotation_decimals=1,
    )
    ax.set_xlabel("Adjacent channel pair", fontsize=12, fontweight="bold")
    ax.set_ylabel("Channel Spacing Error (GHz)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(14)))
    ax.set_xticklabels(_SPACING_LABELS, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-35.0, 20.0)
    ax.set_yticks(CHANNEL_SPACING_Y_TICKS)
    ax.set_xlim(-0.5, 13.5)
    _save_figure(fig, output_path)


def plot_tp1p4_channel_spacing_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    """All 14 adjacent-pair spacing errors pooled into one box+violin."""
    if df.empty:
        print("No data for channel spacing combined banks plot")
        return
    sp = compute_channel_spacing_errors(df)
    if sp.empty:
        return
    _plot_combined_banks_distribution(
        output_path,
        sp["Spacing_Error_GHz"].values,
        ylo=-35.0,
        yhi=20.0,
        yticks=CHANNEL_SPACING_Y_TICKS,
        ylabel="Channel Spacing Error (GHz)",
        use_mean_for_annotation=False,
        annotation_unit="GHz",
        empty_msg="No data for channel spacing combined banks plot",
    )


def _tile_uniformity(df: pd.DataFrame, col: str) -> np.ndarray:
    """Per-tile max–min of `col` across all channels. Returns array of one value per tile."""
    return df.groupby("Tile_SN")[col].agg(lambda x: x.max() - x.min()).values


def plot_tp1p4_power_uniformity_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Distribution of per-tile optical power uniformity (P_MAX_dBm – P_MIN_dBm across 16 ch)."""
    if df.empty:
        print("No data for power uniformity plot")
        return
    _plot_combined_banks_distribution(
        output_path,
        _tile_uniformity(df, "Power(dBm)"),
        ylo=0.0,
        yhi=3.0,
        yticks=POWER_UNIFORMITY_DB_Y_TICKS,
        ylabel="Power Uniformity P_MAX – P_MIN (dB)",
        use_mean_for_annotation=True,
        annotation_unit="dB",
        empty_msg="No data for power uniformity plot",
    )


def plot_tp1p4_freq_error_uniformity_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Distribution of per-tile freq error uniformity (F_MAX – F_MIN across 16 ch)."""
    if df.empty:
        print("No data for freq error uniformity plot")
        return
    _plot_combined_banks_distribution(
        output_path,
        _tile_uniformity(df, "Frequency_Error_GHz"),
        ylo=0.0,
        yhi=100.0,
        yticks=FREQ_UNIFORMITY_Y_TICKS,
        ylabel="Freq Error Uniformity F_MAX – F_MIN (GHz)",
        use_mean_for_annotation=True,
        annotation_unit="GHz",
        empty_msg="No data for freq error uniformity plot",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TP1-4 LIV distribution plots for Gen2 ONET data at 150 mA"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to clm_data_onet_sftp root (default: repo data/clm_data_onet_sftp)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory for PNGs (default: nevada_cos/analysis/results/liv)",
    )
    args = parser.parse_args()

    data_root = (args.data_root or default_data_root()).resolve()
    results_dir = (args.results_dir or default_results_dir()).resolve()
    config_dir = default_config_dir()

    tp14_path = data_root / "TP1-4"
    if not tp14_path.is_dir():
        print(f"TP1-4 data directory not found: {tp14_path}", file=sys.stderr)
        return 1

    print(f"Data root : {data_root}")
    print(f"TP1-4 path: {tp14_path}")
    print(f"Config    : {config_dir}")
    print(f"Results   : {results_dir}")
    print()

    wl_grid = load_wavelength_grid(config_dir)
    df = load_tp1p4_at_150ma(tp14_path, wl_grid)

    if df.empty:
        print("No TP1-4 data loaded — nothing to plot.", file=sys.stderr)
        return 1

    n_tiles_raw = df["Tile_SN"].nunique()
    print(f"Loaded {len(df)} rows from {n_tiles_raw} tiles at 150 mA")
    df = drop_outlier_tiles(df)
    df = drop_low_power_tiles(df)
    n_tiles = df["Tile_SN"].nunique()
    print(f"Analysis includes {n_tiles} tiles after outlier exclusion\n")

    plot_tp1p4_freq_error_distribution(
        df, results_dir / "tp1p4_distribution_vs_freq_error.png"
    )
    plot_tp1p4_freq_error_combined_banks(
        df, results_dir / "tp1p4_distribution_vs_freq_error_combined_banks.png"
    )
    plot_tp1p4_optical_power_distribution(
        df, results_dir / "tp1p4_distribution_vs_optical_power.png"
    )

    # --- Optimized: per-channel mean correction ---
    opt_dir = results_dir.parent / "optimized"
    print(f"\nOptimized results: {opt_dir}")
    df_opt, channel_means = apply_per_channel_mean_correction(df)
    print("Per-channel mean corrections applied (GHz):")
    for (bank, ch), mean_val in sorted(channel_means.items()):
        label = f"{'A' if bank == 0 else 'B'}-Ch{ch + 1}"
        print(f"  {label}: {mean_val:+.2f} GHz")
    print()
    plot_tp1p4_freq_error_distribution(
        df_opt, opt_dir / "tp1p4_distribution_vs_freq_error.png"
    )
    plot_tp1p4_freq_error_combined_banks(
        df_opt, opt_dir / "tp1p4_distribution_vs_freq_error_combined_banks.png"
    )

    # Original optical power plots (unaffected by freq correction)
    plot_tp1p4_optical_power_distribution(
        df, results_dir / "tp1p4_distribution_vs_optical_power.png"
    )
    plot_tp1p4_optical_power_combined_banks(
        df, results_dir / "tp1p4_distribution_vs_optical_power_combined_banks.png"
    )

    # Uniformity plots — liv (original data)
    plot_tp1p4_power_uniformity_distribution(
        df, results_dir / "tp1p4_distribution_optical_power_uniformity.png"
    )
    plot_tp1p4_freq_error_uniformity_distribution(
        df, results_dir / "tp1p4_distribution_frequency_error_uniformity.png"
    )

    # Channel spacing error — liv
    plot_tp1p4_channel_spacing_distribution(
        df, results_dir / "tp1p4_distribution_channel_spacing_error.png"
    )
    plot_tp1p4_channel_spacing_combined_banks(
        df, results_dir / "tp1p4_distribution_channel_spacing_error_combined_banks.png"
    )

    # Optimized optical power (same data, written to opt_dir for completeness)
    plot_tp1p4_optical_power_distribution(
        df, opt_dir / "tp1p4_distribution_vs_optical_power.png"
    )
    plot_tp1p4_optical_power_combined_banks(
        df, opt_dir / "tp1p4_distribution_vs_optical_power_combined_banks.png"
    )

    # Uniformity plots — optimized (freq error corrected, power unchanged)
    plot_tp1p4_power_uniformity_distribution(
        df, opt_dir / "tp1p4_distribution_optical_power_uniformity.png"
    )
    plot_tp1p4_freq_error_uniformity_distribution(
        df_opt, opt_dir / "tp1p4_distribution_frequency_error_uniformity.png"
    )

    # Channel spacing error — optimized (spacing computed from corrected freq errors)
    plot_tp1p4_channel_spacing_distribution(
        df_opt, opt_dir / "tp1p4_distribution_channel_spacing_error.png"
    )
    plot_tp1p4_channel_spacing_combined_banks(
        df_opt, opt_dir / "tp1p4_distribution_channel_spacing_error_combined_banks.png"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
