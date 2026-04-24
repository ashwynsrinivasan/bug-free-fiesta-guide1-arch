#!/usr/bin/env python3
"""
TP1-1 / TP1-2 tuning coefficient analysis for Gen2 CLM data.

TP1-1 (dfreq/dT, dwave/dT, dPdBm/dT)
    Loads *TP1-1 Test.csv (3-point temperature sweep at 150 mA: 36/43/50 °C).
    Fits frequency vs Set Temp(C) per tile per channel → slope in GHz/°C.
    Fits wavelength vs Set Temp(C) per tile per channel → slope in pm/°C.
    Fits Power(dBm) vs Set Temp(C) per tile per channel → slope in dB/°C.
    Writes to nevada_cos/analysis/results/liv/:
        tp1p1_distribution_vs_dfreq_dT.png               (16-column per-channel)
        tp1p1_distribution_vs_dfreq_dT_combined_banks.png (pooled)
        tp1p1_distribution_vs_dwave_dT.png               (16-column per-channel)
        tp1p1_distribution_vs_dwave_dT_combined_banks.png (pooled)
        tp1p1_distribution_vs_dPdBm_dT.png               (16-column per-channel)
        tp1p1_distribution_vs_dPdBm_dT_combined_banks.png (pooled)

TP1-2 (dfreq/dI, dwave/dI, dfreq/dBo, dwave/dBo)
    Loads *TP1-2 Scan.csv (current sweep 120–170 mA at 5 mA steps, fixed ~50 °C).
    Fits frequency vs Set Laser(mA) in the 140–160 mA window → slope in GHz/mA.
    Fits wavelength vs Set Laser(mA) in the 140–160 mA window → slope in pm/mA.
    Fits frequency vs Power(dBm) in the 140–160 mA window → slope in GHz/dB.
    Fits wavelength vs Power(dBm) in the 140–160 mA window → slope in pm/dB.
    Writes to nevada_cos/analysis/results/liv/:
        tp1p2_distribution_vs_dfreq_dI.png               (16-column per-channel)
        tp1p2_distribution_vs_dfreq_dI_combined_banks.png (pooled)
        tp1p2_distribution_vs_dwave_dI.png               (16-column per-channel)
        tp1p2_distribution_vs_dwave_dI_combined_banks.png (pooled)
        tp1p2_distribution_vs_dfreq_dBo.png              (16-column per-channel)
        tp1p2_distribution_vs_dfreq_dBo_combined_banks.png (pooled)
        tp1p2_distribution_vs_dwave_dBo.png              (16-column per-channel)
        tp1p2_distribution_vs_dwave_dBo_combined_banks.png (pooled)

Same box+violin styling as analyze_tp1p4_onet_sftp.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def default_data_root() -> Path:
    return Path(__file__).resolve().parents[6] / "data" / "clm_data_onet_sftp"


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "results" / "liv"


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

_C_NM_PER_S = 299792458e9  # nm/s


def _freq_ghz(wl_nm: np.ndarray) -> np.ndarray:
    return _C_NM_PER_S / wl_nm / 1e9


# ---------------------------------------------------------------------------
# Data loading — TP1-1 dfreq/dT, dwave/dT
# ---------------------------------------------------------------------------

def load_tp1p1_dfreq_dT(tp11_path: Path) -> pd.DataFrame:
    """Load *TP1-1 Test.csv; fit freq and wavelength vs temperature per tile/bank/channel.

    Returns DataFrame with columns: Tile_SN, Bank, Channel,
        dfreq_dT (GHz/°C), dwave_dT (pm/°C), dPdBm_dT (dB/°C).
    Uses all available temperature points (typically 36/43/50 °C at 150 mA).
    When a tile has multiple files, keeps the most recent (last alphabetically).
    """
    csv_files = sorted(tp11_path.glob("*TP1-1 Test.csv"))
    if not csv_files:
        print(f"No *TP1-1 Test.csv files found under {tp11_path}", file=sys.stderr)
        return pd.DataFrame()

    by_tile: dict[str, Path] = {}
    for f in csv_files:
        tile = _tile_sn_from_path(f)
        if tile:
            by_tile[tile] = f

    rows: list[dict] = []
    for tile_sn, csv_file in sorted(by_tile.items()):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}", file=sys.stderr)
            continue
        df = df[df["Set Laser(mA)"] == 150].copy()
        if df.empty:
            continue
        df["Power_dBm"] = 10 * np.log10(df["Power(mW)"].clip(lower=1e-9))
        for bank in [0, 1]:
            for ch in range(8):
                sub = df[(df["Bank"] == bank) & (df["Channel"] == ch)].sort_values("Set Temp(C)")
                if len(sub) < 2:
                    continue
                temps = sub["Set Temp(C)"].values
                wls = sub["PeakWave(nm)"].values
                slope, _ = np.polyfit(temps, _freq_ghz(wls), 1)
                wl_slope, _ = np.polyfit(temps, wls * 1000, 1)  # nm → pm
                p_slope, _ = np.polyfit(temps, sub["Power_dBm"].values, 1)
                rows.append({
                    "Tile_SN": tile_sn,
                    "Bank": bank,
                    "Channel": ch,
                    "dfreq_dT": slope,
                    "dwave_dT": wl_slope,
                    "dPdBm_dT": p_slope * 1000,  # dB/°C → mdB/°C
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading — TP1-2 dfreq/dI, dwave/dI, dfreq/dBo, dwave/dBo
# ---------------------------------------------------------------------------

def load_tp1p2_dfreq_dI(
    tp12_path: Path,
    current_window: tuple[float, float] = (140.0, 160.0),
) -> pd.DataFrame:
    """Load *TP1-2 Scan.csv; fit tuning coefficients vs current and output power.

    Returns DataFrame with columns: Tile_SN, Bank, Channel,
        dfreq_dI (GHz/mA), dwave_dI (pm/mA),
        dfreq_dBo (GHz/dB), dwave_dBo (pm/dB).
    All fits use the 140–160 mA current window.
    Power(dBm) = 10*log10(Power(mW)) from the same window rows.
    """
    csv_files = sorted(tp12_path.glob("*TP1-2 Scan.csv"))
    if not csv_files:
        print(f"No *TP1-2 Scan.csv files found under {tp12_path}", file=sys.stderr)
        return pd.DataFrame()

    by_tile: dict[str, Path] = {}
    for f in csv_files:
        tile = _tile_sn_from_path(f)
        if tile:
            by_tile[tile] = f

    lo, hi = current_window
    rows: list[dict] = []
    for tile_sn, csv_file in sorted(by_tile.items()):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}", file=sys.stderr)
            continue
        df = df[df["Set Laser(mA)"].between(lo, hi)].copy()
        if df.empty:
            continue
        df["Power_dBm"] = 10 * np.log10(df["Power(mW)"].clip(lower=1e-9))
        for bank in [0, 1]:
            for ch in range(8):
                sub = df[(df["Bank"] == bank) & (df["Channel"] == ch)].sort_values("Set Laser(mA)")
                if len(sub) < 2:
                    continue
                currents = sub["Set Laser(mA)"].values
                wls = sub["PeakWave(nm)"].values
                pwr_dbm = sub["Power_dBm"].values
                freqs = _freq_ghz(wls)
                wls_pm = wls * 1000  # nm → pm

                slope_fi, _ = np.polyfit(currents, freqs, 1)
                slope_wi, _ = np.polyfit(currents, wls_pm, 1)
                slope_fp, _ = np.polyfit(pwr_dbm, freqs, 1)
                slope_wp, _ = np.polyfit(pwr_dbm, wls_pm, 1)
                rows.append({
                    "Tile_SN": tile_sn,
                    "Bank": bank,
                    "Channel": ch,
                    "dfreq_dI": slope_fi,
                    "dwave_dI": slope_wi,
                    "dfreq_dBo": slope_fp,
                    "dwave_dBo": slope_wp,
                })
    return pd.DataFrame(rows)


def _tile_sn_from_path(p: Path) -> str | None:
    for part in p.stem.split("-"):
        if len(part) == 11 and part[0] == "Y" and part[1:].isdigit():
            return part
    return None


# ---------------------------------------------------------------------------
# Outlier filtering
# ---------------------------------------------------------------------------

def drop_outlier_tiles_slope(
    df: pd.DataFrame,
    slope_col: str,
    limit: float,
    label: str,
) -> pd.DataFrame:
    """Exclude tiles where any channel has |slope_col| > limit."""
    bad_mask = df.groupby("Tile_SN")[slope_col].transform(
        lambda x: (x.abs() > limit).any()
    )
    excluded = sorted(df.loc[bad_mask, "Tile_SN"].unique())
    if excluded:
        print(f"Excluding {len(excluded)} tile(s) with |{label}| > {limit}: {excluded}")
    return df[~bad_mask].copy()


# ---------------------------------------------------------------------------
# Plot styling  (mirrors analyze_tp1p4_onet_sftp conventions)
# ---------------------------------------------------------------------------

DIST_FIGSIZE = (10, 5)
COMBINED_BANKS_FIGSIZE = (3, 4)

DFREQ_DT_Y_TICKS  = np.arange(-17, -13.9, 1)      # GHz/°C  (-17 … -14)
DFREQ_DI_Y_TICKS  = np.arange(-1.2, -0.49, 0.1)   # GHz/mA  (-1.2 … -0.5)
DWAVE_DT_Y_TICKS  = np.arange(70, 111, 10)         # pm/°C   (70 … 110)
DWAVE_DI_Y_TICKS  = np.arange(3, 8.1, 1)           # pm/mA   (3 … 8)
DPDBM_DT_Y_TICKS  = np.arange(-100, 1, 10)           # mdB/°C (-100 … 0)
DFREQ_DBO_Y_TICKS = np.arange(-40, -14.9, 5)       # GHz/dB  (-40 … -15)
DWAVE_DBO_Y_TICKS = np.arange(80, 221, 20)          # pm/dB   (80 … 220)

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


def _grey_violin_behind_box(
    ax, values: np.ndarray, *, position: float = 0.0, width: float = 0.62
) -> None:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return
    vp = ax.violinplot(
        [vals], positions=[position], vert=True, widths=width,
        showmeans=False, showmedians=False, showextrema=False,
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
    annotation_unit: str = "",
    annotation_decimals: int = 2,
    violin_width: float = 0.62,
) -> None:
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
        box_data, positions=box_positions, vert=True,
        widths=box_width, **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    fmt = f".{annotation_decimals}f"
    for pos, vals in zip(box_positions, box_data):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue
        c_val = float(np.median(vals) if use_median_in_annotation else np.mean(vals))
        sym = "μ̃" if use_median_in_annotation else "μ"
        std = float(np.std(vals))
        ax.text(
            pos, y_ann,
            f"{sym}={c_val:{fmt}}{annotation_unit}\nσ={std:{fmt}}{annotation_unit}",
            fontsize=annotation_fontsize, ha="center", va="bottom",
            zorder=6, clip_on=False,
        )


def _plot_combined_banks_distribution(
    output_path: Path,
    arr: np.ndarray,
    *,
    ylo: float,
    yhi: float,
    yticks: np.ndarray,
    ylabel: str,
    annotation_unit: str,
    empty_msg: str,
) -> None:
    """Single pooled column: grey violin behind gen1 box; μ̃ and σ in title."""
    vals = np.asarray(arr, dtype=float)
    if vals.size == 0:
        print(empty_msg)
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=COMBINED_BANKS_FIGSIZE, layout="constrained")
    _grey_violin_behind_box(ax, vals, position=0.0)
    bp = ax.boxplot([vals], positions=[0.0], vert=True, widths=0.35, **_GEN1_BOXPLOT_KW)
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    med = float(np.median(vals))
    std = float(np.std(vals))
    ax.set_title(
        f"μ̃={med:.2f}{annotation_unit}, σ={std:.2f}{annotation_unit}",
        fontsize=10, fontweight="bold", pad=8,
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


def _build_box_data(df: pd.DataFrame, col: str) -> tuple[list[np.ndarray], list[int]]:
    """Build 16-column box_data/box_positions lists (Bank 0 ch0-7, Bank 1 ch0-7)."""
    box_data, box_positions = [], []
    for bank in [0, 1]:
        for ch in range(8):
            sub = df[(df["Bank"] == bank) & (df["Channel"] == ch)][col].values
            if sub.size == 0:
                continue
            box_data.append(sub)
            box_positions.append(bank * 8 + ch)
    return box_data, box_positions


_CH_LABELS = [f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)]


def _16col_plot(
    df: pd.DataFrame,
    col: str,
    output_path: Path,
    *,
    ylo: float,
    yhi: float,
    yticks: np.ndarray,
    xlabel: str,
    ylabel: str,
    annotation_unit: str,
    annotation_decimals: int,
    empty_msg: str,
) -> None:
    if df.empty:
        print(empty_msg)
        return
    box_data, box_positions = _build_box_data(df, col)
    if not box_data:
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_with_violin(
        ax, box_data, box_positions,
        ylo=ylo, yhi=yhi,
        use_median_in_annotation=True,
        annotation_unit=annotation_unit,
        annotation_decimals=annotation_decimals,
    )
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels(_CH_LABELS, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, 15.5)
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Plot functions — dfreq/dT
# ---------------------------------------------------------------------------

def plot_tp1p1_dfreq_dT_distribution(df: pd.DataFrame, output_path: Path) -> None:
    _16col_plot(df, "dfreq_dT", output_path,
                ylo=-17.0, yhi=-14.0, yticks=DFREQ_DT_Y_TICKS,
                xlabel="Bank channel", ylabel="dfreq/dT (GHz/°C)",
                annotation_unit="GHz/°C", annotation_decimals=2,
                empty_msg="No data for dfreq/dT distribution plot")


def plot_tp1p1_dfreq_dT_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    _plot_combined_banks_distribution(
        output_path, df["dfreq_dT"].values,
        ylo=-17.0, yhi=-14.0, yticks=DFREQ_DT_Y_TICKS,
        ylabel="dfreq/dT (GHz/°C)", annotation_unit="GHz/°C",
        empty_msg="No data for dfreq/dT combined banks plot",
    )


# ---------------------------------------------------------------------------
# Plot functions — dwave/dT  (pm/°C)
# ---------------------------------------------------------------------------

def plot_tp1p1_dwave_dT_distribution(df: pd.DataFrame, output_path: Path) -> None:
    _16col_plot(df, "dwave_dT", output_path,
                ylo=70.0, yhi=110.0, yticks=DWAVE_DT_Y_TICKS,
                xlabel="Bank channel", ylabel="dλ/dT (pm/°C)",
                annotation_unit="pm/°C", annotation_decimals=1,
                empty_msg="No data for dwave/dT distribution plot")


def plot_tp1p1_dwave_dT_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    _plot_combined_banks_distribution(
        output_path, df["dwave_dT"].values,
        ylo=70.0, yhi=110.0, yticks=DWAVE_DT_Y_TICKS,
        ylabel="dλ/dT (pm/°C)", annotation_unit="pm/°C",
        empty_msg="No data for dwave/dT combined banks plot",
    )


# ---------------------------------------------------------------------------
# Plot functions — dPdBm/dT  (dB/°C)
# ---------------------------------------------------------------------------

def plot_tp1p1_dPdBm_dT_distribution(df: pd.DataFrame, output_path: Path) -> None:
    _16col_plot(df, "dPdBm_dT", output_path,
                ylo=-100.0, yhi=0.0, yticks=DPDBM_DT_Y_TICKS,
                xlabel="Bank channel", ylabel="dP_out/dT (mdB/°C)",
                annotation_unit="mdB/°C", annotation_decimals=1,
                empty_msg="No data for dPdBm/dT distribution plot")


def plot_tp1p1_dPdBm_dT_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    _plot_combined_banks_distribution(
        output_path, df["dPdBm_dT"].values,
        ylo=-100.0, yhi=0.0, yticks=DPDBM_DT_Y_TICKS,
        ylabel="dP_out/dT (mdB/°C)", annotation_unit="mdB/°C",
        empty_msg="No data for dPdBm/dT combined banks plot",
    )


# ---------------------------------------------------------------------------
# Plot functions — dfreq/dI
# ---------------------------------------------------------------------------

def plot_tp1p2_dfreq_dI_distribution(df: pd.DataFrame, output_path: Path) -> None:
    _16col_plot(df, "dfreq_dI", output_path,
                ylo=-1.2, yhi=-0.5, yticks=DFREQ_DI_Y_TICKS,
                xlabel="Bank channel", ylabel="dfreq/dI @ 150 mA (GHz/mA)",
                annotation_unit="GHz/mA", annotation_decimals=3,
                empty_msg="No data for dfreq/dI distribution plot")


def plot_tp1p2_dfreq_dI_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    _plot_combined_banks_distribution(
        output_path, df["dfreq_dI"].values,
        ylo=-1.2, yhi=-0.5, yticks=DFREQ_DI_Y_TICKS,
        ylabel="dfreq/dI @ 150 mA (GHz/mA)", annotation_unit="GHz/mA",
        empty_msg="No data for dfreq/dI combined banks plot",
    )


# ---------------------------------------------------------------------------
# Plot functions — dwave/dI  (pm/mA)
# ---------------------------------------------------------------------------

def plot_tp1p2_dwave_dI_distribution(df: pd.DataFrame, output_path: Path) -> None:
    _16col_plot(df, "dwave_dI", output_path,
                ylo=3.0, yhi=8.0, yticks=DWAVE_DI_Y_TICKS,
                xlabel="Bank channel", ylabel="dλ/dI @ 150 mA (pm/mA)",
                annotation_unit="pm/mA", annotation_decimals=2,
                empty_msg="No data for dwave/dI distribution plot")


def plot_tp1p2_dwave_dI_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    _plot_combined_banks_distribution(
        output_path, df["dwave_dI"].values,
        ylo=3.0, yhi=8.0, yticks=DWAVE_DI_Y_TICKS,
        ylabel="dλ/dI @ 150 mA (pm/mA)", annotation_unit="pm/mA",
        empty_msg="No data for dwave/dI combined banks plot",
    )


# ---------------------------------------------------------------------------
# Plot functions — dfreq/dBo  (GHz/dB output power)
# ---------------------------------------------------------------------------

def plot_tp1p2_dfreq_dBo_distribution(df: pd.DataFrame, output_path: Path) -> None:
    _16col_plot(df, "dfreq_dBo", output_path,
                ylo=-40.0, yhi=-15.0, yticks=DFREQ_DBO_Y_TICKS,
                xlabel="Bank channel", ylabel="dfreq/dP_out (GHz/dB)",
                annotation_unit="GHz/dB", annotation_decimals=1,
                empty_msg="No data for dfreq/dBo distribution plot")


def plot_tp1p2_dfreq_dBo_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    _plot_combined_banks_distribution(
        output_path, df["dfreq_dBo"].values,
        ylo=-40.0, yhi=-15.0, yticks=DFREQ_DBO_Y_TICKS,
        ylabel="dfreq/dP_out (GHz/dB)", annotation_unit="GHz/dB",
        empty_msg="No data for dfreq/dBo combined banks plot",
    )


# ---------------------------------------------------------------------------
# Plot functions — dwave/dBo  (pm/dB output power)
# ---------------------------------------------------------------------------

def plot_tp1p2_dwave_dBo_distribution(df: pd.DataFrame, output_path: Path) -> None:
    _16col_plot(df, "dwave_dBo", output_path,
                ylo=80.0, yhi=220.0, yticks=DWAVE_DBO_Y_TICKS,
                xlabel="Bank channel", ylabel="dλ/dP_out (pm/dB)",
                annotation_unit="pm/dB", annotation_decimals=1,
                empty_msg="No data for dwave/dBo distribution plot")


def plot_tp1p2_dwave_dBo_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    _plot_combined_banks_distribution(
        output_path, df["dwave_dBo"].values,
        ylo=80.0, yhi=220.0, yticks=DWAVE_DBO_Y_TICKS,
        ylabel="dλ/dP_out (pm/dB)", annotation_unit="pm/dB",
        empty_msg="No data for dwave/dBo combined banks plot",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="TP1-1/TP1-2 tuning coefficient distribution plots for Gen2 ONET data"
    )
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Path to clm_data_onet_sftp root")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Output directory (default: nevada_cos/analysis/results/liv)")
    args = parser.parse_args()

    data_root = (args.data_root or default_data_root()).resolve()
    results_dir = (args.results_dir or default_results_dir()).resolve()

    print(f"Data root  : {data_root}")
    print(f"Results    : {results_dir}")
    print()

    # --- TP1-1: dfreq/dT, dwave/dT ---
    tp11_path = data_root / "TP1-1"
    if not tp11_path.is_dir():
        print(f"TP1-1 directory not found: {tp11_path}", file=sys.stderr)
    else:
        print("Loading TP1-1 Test data …")
        df_dt = load_tp1p1_dfreq_dT(tp11_path)
        n_raw = df_dt["Tile_SN"].nunique() if not df_dt.empty else 0
        print(f"  Loaded {n_raw} tiles")
        df_dt = drop_outlier_tiles_slope(df_dt, "dfreq_dT", limit=30.0, label="dfreq/dT GHz/C")
        print(f"  Analysis includes {df_dt['Tile_SN'].nunique()} tiles\n")

        plot_tp1p1_dfreq_dT_distribution(df_dt, results_dir / "tp1p1_distribution_vs_dfreq_dT.png")
        plot_tp1p1_dfreq_dT_combined_banks(df_dt, results_dir / "tp1p1_distribution_vs_dfreq_dT_combined_banks.png")
        plot_tp1p1_dwave_dT_distribution(df_dt, results_dir / "tp1p1_distribution_vs_dwave_dT.png")
        plot_tp1p1_dwave_dT_combined_banks(df_dt, results_dir / "tp1p1_distribution_vs_dwave_dT_combined_banks.png")
        plot_tp1p1_dPdBm_dT_distribution(df_dt, results_dir / "tp1p1_distribution_vs_dPdBm_dT.png")
        plot_tp1p1_dPdBm_dT_combined_banks(df_dt, results_dir / "tp1p1_distribution_vs_dPdBm_dT_combined_banks.png")

    # --- TP1-2: dfreq/dI, dwave/dI, dfreq/dBo, dwave/dBo ---
    tp12_path = data_root / "TP1-2"
    if not tp12_path.is_dir():
        print(f"TP1-2 directory not found: {tp12_path}", file=sys.stderr)
    else:
        print("Loading TP1-2 Scan data (140–160 mA window) …")
        df_di = load_tp1p2_dfreq_dI(tp12_path)
        n_raw = df_di["Tile_SN"].nunique() if not df_di.empty else 0
        print(f"  Loaded {n_raw} tiles")
        df_di = drop_outlier_tiles_slope(df_di, "dfreq_dI", limit=5.0, label="dfreq/dI GHz/mA")
        print(f"  Analysis includes {df_di['Tile_SN'].nunique()} tiles\n")

        plot_tp1p2_dfreq_dI_distribution(df_di, results_dir / "tp1p2_distribution_vs_dfreq_dI.png")
        plot_tp1p2_dfreq_dI_combined_banks(df_di, results_dir / "tp1p2_distribution_vs_dfreq_dI_combined_banks.png")
        plot_tp1p2_dwave_dI_distribution(df_di, results_dir / "tp1p2_distribution_vs_dwave_dI.png")
        plot_tp1p2_dwave_dI_combined_banks(df_di, results_dir / "tp1p2_distribution_vs_dwave_dI_combined_banks.png")
        plot_tp1p2_dfreq_dBo_distribution(df_di, results_dir / "tp1p2_distribution_vs_dfreq_dBo.png")
        plot_tp1p2_dfreq_dBo_combined_banks(df_di, results_dir / "tp1p2_distribution_vs_dfreq_dBo_combined_banks.png")
        plot_tp1p2_dwave_dBo_distribution(df_di, results_dir / "tp1p2_distribution_vs_dwave_dBo.png")
        plot_tp1p2_dwave_dBo_combined_banks(df_di, results_dir / "tp1p2_distribution_vs_dwave_dBo_combined_banks.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
