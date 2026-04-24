#!/usr/bin/env python3
"""
TP2-4 / TP2-5 Scan analysis for CLM data under data/clm_data_onet_sftp.

By default applies the same tile-level filters as ips_clm_gen1 (analysis_src/filter.yaml):
TP2-6 min power → TP2-5 total power @50C → TP2-5 frequency error @50C → TP2-4 spacing error.
Use --skip-filters to include all tiles that load (no filter.yaml cascade).

Writes separate tile-scatter and distribution figures per test point when data exists:
  - tp2p4_* / tp2p5_* (freq/center: per-channel, per_bank, combined_banks; channel spacing).
    Combined panels: no x-axis labels.
  - tp2p5_totalpower_* (tile + per_bank + combined_banks @50C).
  - tp2p6_power_* (tile + per-channel distribution + combined_banks).
  - tp2p4_osa_scaled_power_distribution.png (per-channel mW) and
    tp2p4_osa_scaled_power_distribution_combined_banks.png (pooled mW), y-axis 0–25 mW; same scaling as EVT MFG
    (``reconcile_tp2p4_dataframe`` / 2× fiber per bank).

When the data root is the default ``data/clm_data_onet_sftp`` (and not ``--evt-tp-scan``), **TP2-4** scan and OSA figures also merge **gen1 LM** manufacturing data from ``…/oPO Data/3pcs OPO Module V1/LM`` (ONET rows ``Version=v1``, gen1 ``v2``). Console and OSA figure banners report tile counts for ONET vs gen1 LM.

With ``--evt-tp-scan``, restricts to ``config/evt_tp_scan_tiles.yaml``, then keeps only ``Tile_SN``
whose **30C OFC** min-channel MPD is **≥ 12 mW** (same gate as ``plot_evt_30c_blueray_distributions``;
see ``evt_plot_filters`` / ``evt_30c_common``), and writes ``evt_tp2p4_*``, ``evt_tp2p5_*``,
``evt_tp2p5_totalpower_*``, and ``evt_tp2p6_power_*``
instead (same figures, filtered EVT subset).
Distribution views: vertical gen1 box plots with grey violin (same styling as combined panels) behind each box,
μ/μ̃–σ annotations inside the plot above the x-axis (tick-aligned). Tile figures remain bank-colored scatter.
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

from evt_30c_common import (
    build_30c_power_frame,
    default_clm_mfg_data_base,
    default_ofc_excel_path,
    evt_tile_sn_from_30c_passing_ids,
)
from evt_plot_filters import EVT_MIN_OPTICAL_POWER_MW, tile_ids_passing_min_channel_mpd
from tp2p4_osa_power_reconcile import reconcile_tp2p4_dataframe, tile_sn_from_tp2p4_path


# Monorepo: .../friendly-system-lmi/data/clm_data_onet_sftp
def default_data_root() -> Path:
    here = Path(__file__).resolve()
    # .../blueray_nevada_rfp/nevada_l_tile/src/thisfile.py -> parents[5] = repo root
    repo_root = here.parents[5]
    return (repo_root / "data" / "clm_data_onet_sftp").resolve()


def default_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "config"


def default_gen1_lm_data_root() -> Path:
    """Latest gen1 LM manufacturing tree under the same ONET SFTP root (TP2-* + OSA, etc.)."""
    return (default_data_root() / "oPO Data" / "3pcs OPO Module V1" / "LM").resolve()


TILE_FIGSIZE = (30, 5)
DIST_FIGSIZE = (10, 5)
# Two-bank-only distribution panels (center freq, TP2-5 total power)
DIST_FIGSIZE_TWO_BANK = (5, 5)
# Pooled-bank distribution panels (width × height = 3 × 4)
COMBINED_BANKS_FIGSIZE = (3, 4)
TILE_SCATTER_BANK = {
    0: {"facecolor": "#1565c0", "edgecolor": "#0d1117", "linewidths": 0.65},
    1: {"facecolor": "#c62828", "edgecolor": "#0d1117", "linewidths": 0.65},
}
TILE_SCATTER_S_FREQ = 55
TILE_SCATTER_S_CENTER = 120
TILE_SCATTER_S_SPACING = 50
# Frequency error plots: fixed span and major ticks every 10 GHz
FREQ_ERROR_Y_TICKS = np.arange(-50, 51, 10)
# TP2-6 power: tile plot 0–20 mW; distribution / combined_banks panels 0–25 mW
TP2P6_POWER_TILE_Y_TICKS = np.arange(0, 21, 5)
TP2P6_POWER_DIST_Y_TICKS = np.arange(0, 26, 5)
# TP2-4 OSA scaled (per-channel + combined banks): 0–25 mW
TP2P4_OSA_SCALED_MW_Y_TICKS = np.arange(0, 26, 5)
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


def load_evt_tp_scan_tiles(config_dir: Path) -> list[str]:
    """Tile_SN list from ``evt_tp_scan_tiles.yaml`` (key ``tile_sn``)."""
    p = config_dir / "evt_tp_scan_tiles.yaml"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    raw = doc.get("tile_sn") or doc.get("tiles") or []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def intersect_evt_tp_scan_tiles(valid_tiles: set[str] | None, evt_sn: list[str]) -> set[str]:
    """If ``valid_tiles`` is None (skip filters), use only EVT list; else intersection."""
    evt_set = {s.strip() for s in evt_sn if s and str(s).strip()}
    if not evt_set:
        return set()
    if valid_tiles is None:
        return evt_set
    return valid_tiles & evt_set


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


def load_tp2p4_osa_enriched_onet(tp_path: Path, valid_tiles: set[str] | None) -> pd.DataFrame:
    """Load *TP2-4 Scan.csv at T_MUX ~50C; add ``OSA_Power_mW_scaled`` via bank reconcile (same as EVT MFG).

    Uses ``tile_sn_from_tp2p4_path`` (Y + 8–10 digits) so LM-style filenames match where the strict
    scan loader does not.
    """
    enriched_parts: list[pd.DataFrame] = []
    for csv_file in sorted(tp_path.glob("*TP2-4 Scan.csv")):
        tile_sn = tile_sn_from_tp2p4_path(csv_file)
        if tile_sn is None:
            continue
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"OSA load: skip {csv_file.name}: {e}", file=sys.stderr)
            continue
        df = df.copy()
        df["Tile_SN"] = tile_sn
        if "T_MUX(C)" not in df.columns:
            continue
        df = df[(df["T_MUX(C)"] >= 49.9) & (df["T_MUX(C)"] <= 50.1)].copy()
        if df.empty:
            continue
        if valid_tiles is not None:
            df = df[df["Tile_SN"].isin(valid_tiles)].copy()
            if df.empty:
                continue
        try:
            enriched, _ = reconcile_tp2p4_dataframe(df)
        except ValueError:
            continue
        if enriched.empty:
            continue
        enriched_parts.append(enriched)
    if not enriched_parts:
        return pd.DataFrame()
    out = pd.concat(enriched_parts, ignore_index=True)
    out["Version"] = "v1"
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


def _combined_banks_strip_xaxis(ax) -> None:
    """No x-axis title, tick labels, or tick marks for single-column combined-bank panels."""
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)


def _grey_violin_behind_box(ax, values: np.ndarray, *, position: float = 0.0, width: float = 0.62) -> None:
    """Vertical grey violin (low zorder) for pooled 1D samples at a single x position."""
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
        if key not in vp:
            continue
        for ln in vp[key]:
            ln.set_visible(False)


def _add_osa_tile_count_banner(ax, stats_banner: str | None) -> None:
    if not stats_banner:
        return
    ax.text(
        0.5,
        1.02,
        stats_banner,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        linespacing=1.25,
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
    annotation_center_decimals: int | None = None,
    annotation_std_decimals: int = 0,
    stats_banner: str | None = None,
) -> None:
    """Single pooled column: grey violin behind gen1 box; μ or μ̃ and σ in the title (no in-axes text)."""
    vals = np.asarray(arr, dtype=float)
    if vals.size == 0:
        print(empty_msg)
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=COMBINED_BANKS_FIGSIZE, layout="constrained")
    pos = 0.0
    _grey_violin_behind_box(ax, vals, position=pos)
    bp = ax.boxplot(
        [vals],
        positions=[pos],
        vert=True,
        widths=0.35,
        **_GEN1_BOXPLOT_KW,
    )
    _force_white_box_faces(bp)
    _raise_boxplot_zorder(bp, 4.0)
    if use_mean_for_annotation:
        c_raw = float(np.mean(vals))
        sym = "μ"
    else:
        c_raw = float(np.median(vals))
        sym = "μ̃"
    std_raw = float(np.std(vals))
    u = annotation_unit
    if annotation_center_decimals is not None:
        c_str = f"{c_raw:.{annotation_center_decimals}f}"
    else:
        c_str = str(int(round(c_raw)))
    if annotation_std_decimals > 0:
        std_str = f"{std_raw:.{annotation_std_decimals}f}"
    else:
        std_str = str(int(round(std_raw)))
    ax.set_title(
        f"{sym}={c_str}{u}, σ={std_str}{u}",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, 0.5)
    _combined_banks_strip_xaxis(ax)
    _add_osa_tile_count_banner(ax, stats_banner)
    _save_single_figure(fig, output_path)


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
    """v1 first, then v2, then any other Version labels (stable order)."""
    if df.empty or "Version" not in df.columns:
        return []
    out: list[str] = []
    for version in ["v1", "v2"]:
        sub = df[df["Version"] == version]
        if not sub.empty:
            out.extend(sorted(sub["Tile_SN"].unique()))
    other_versions = sorted(set(df["Version"].dropna().unique()) - {"v1", "v2"})
    for version in other_versions:
        sub = df[df["Version"] == version]
        out.extend(sorted(sub["Tile_SN"].unique()))
    return out


def _plot_versions_in_tile_figures(df: pd.DataFrame) -> list[str]:
    """Version loop order for tile-style figures (v1 then v2, then others)."""
    if df.empty or "Version" not in df.columns:
        return []
    seen: list[str] = []
    for v in ["v1", "v2"]:
        if v in df["Version"].values:
            seen.append(v)
    for v in sorted(df["Version"].dropna().unique()):
        if v not in seen:
            seen.append(v)
    return seen


# Match ips_clm_gen1 analyze_test_points._plot_tp2p4_freq_error box styling; distribution uses grey violin behind.
_GEN1_BOXPLOT_KW = dict(
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="white", edgecolor="black", linewidth=2),
    whiskerprops=dict(color="black", linewidth=2),
    capprops=dict(color="black", linewidth=2),
    medianprops=dict(color="red", linewidth=2.5),
)


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
    violin_width: float = 0.62,
) -> None:
    """Vertical boxplots; grey violin per column behind boxes; stats inside plot, x-aligned with ticks."""
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
    for version in _plot_versions_in_tile_figures(df):
        df_version = df[df["Version"] == version]
        if df_version.empty:
            continue
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
    _vertical_gen1_boxplot_with_violin(
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


def plot_freq_error_distribution_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    """Pool all per-channel frequency errors into one box + grey violin (no x labels)."""
    if df.empty:
        print("No data for frequency error (distribution, combined banks) plot")
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
        empty_msg="No data for frequency error (distribution, combined banks) plot",
    )


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


def plot_center_freq_error_distribution_per_bank(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No data for center frequency error (distribution, per bank) plot")
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
        print("No data for center frequency error (distribution, per bank) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE_TWO_BANK, layout="constrained")
    _vertical_gen1_boxplot_with_violin(
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


def plot_center_freq_error_distribution_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    """All banks pooled: single box + grey violin (integer μ/σ from mean; no x labels)."""
    if df.empty:
        print("No data for center frequency error (distribution, combined banks) plot")
        return
    _plot_combined_banks_distribution(
        output_path,
        df["Center_Freq_Error_GHz"].values,
        ylo=-50.0,
        yhi=50.0,
        yticks=FREQ_ERROR_Y_TICKS,
        ylabel="Center Frequency Error (GHz)",
        use_mean_for_annotation=True,
        annotation_unit="GHz",
        empty_msg="No data for center frequency error (distribution, combined banks) plot",
    )


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
    _vertical_gen1_boxplot_with_violin(
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
    for version in _plot_versions_in_tile_figures(df):
        df_version = df[df["Version"] == version]
        if df_version.empty:
            continue
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


def plot_tp2p5_totalpower_distribution_per_bank(df: pd.DataFrame, output_path: Path) -> None:
    """TP2-5 total power @50C: one vertical box + grey violin per bank (blueray styling)."""
    if df.empty:
        print("No data for TP2-5 total power (distribution, per bank) plot")
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
        print("No data for TP2-5 total power (distribution, per bank) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE_TWO_BANK, layout="constrained")
    _vertical_gen1_boxplot_with_violin(
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


def plot_tp2p5_totalpower_distribution_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    """TP2-5 total power @50C: both banks pooled, single box + grey violin (no x labels)."""
    if df.empty:
        print("No data for TP2-5 total power (distribution, combined banks) plot")
        return
    ylo, yhi = TP2P5_TOTALPOWER_YLIM
    _plot_combined_banks_distribution(
        output_path,
        df["Total_Power_mW"].values,
        ylo=ylo,
        yhi=yhi,
        yticks=TP2P5_TOTALPOWER_Y_TICKS,
        ylabel="Total power in fiber (mW)",
        use_mean_for_annotation=False,
        annotation_unit="mW",
        empty_msg="No data for TP2-5 total power (distribution, combined banks) plot",
    )


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
    for version in _plot_versions_in_tile_figures(df):
        df_version = df[df["Version"] == version]
        if df_version.empty:
            continue
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
    ax.set_ylabel("Power in fiber (mW)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 20)
    ax.set_yticks(TP2P6_POWER_TILE_Y_TICKS)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_bank_legend(), loc="upper right", ncol=2, fontsize=10, frameon=True, framealpha=0.9)
    _save_single_figure(fig, output_path)


def plot_tp2p6_power_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Per gen1 _plot_tp2p6_power_combined right panel: vertical boxes by bank-channel (blueray style)."""
    if df.empty:
        print("No data for TP2-6 power (distribution) plot")
        return
    ylo, yhi = 0.0, 25.0
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
    _vertical_gen1_boxplot_with_violin(
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
    ax.set_ylabel("Power in fiber (mW)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels([f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)], fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(TP2P6_POWER_DIST_Y_TICKS)
    ax.set_xlim(-0.5, 15.5)
    _save_single_figure(fig, output_path)


def plot_tp2p6_power_distribution_combined_banks(df: pd.DataFrame, output_path: Path) -> None:
    """Pool all TP2-6 channel powers (every tile, bank, channel); box + grey violin, no x labels."""
    if df.empty:
        print("No data for TP2-6 power (distribution, combined banks) plot")
        return
    _plot_combined_banks_distribution(
        output_path,
        df["Power(mW)"].values,
        ylo=0.0,
        yhi=25.0,
        yticks=TP2P6_POWER_DIST_Y_TICKS,
        ylabel="Power in fiber (mW)",
        use_mean_for_annotation=False,
        annotation_unit="mW",
        empty_msg="No data for TP2-6 power (distribution, combined banks) plot",
    )


def plot_tp2p4_osa_scaled_power_distribution(
    df: pd.DataFrame,
    output_path: Path,
    *,
    stats_banner: str | None = None,
) -> None:
    """
    TP2-4 Scan: per-channel ``OSA_Power_mW_scaled`` (uniform scale to 2× fiber per bank),
    y-axis **0–25 mW** (``Power in fiber (mW)``); same 16-column layout as TP2-6.
    """
    col = "OSA_Power_mW_scaled"
    if df.empty or col not in df.columns:
        print("No data for TP2-4 OSA scaled power (distribution) plot")
        return
    sub = df.dropna(subset=[col])
    if sub.empty:
        print("No data for TP2-4 OSA scaled power (distribution) plot")
        return
    ylo, yhi = 0.0, 25.0
    yticks = TP2P4_OSA_SCALED_MW_Y_TICKS
    box_data: list[np.ndarray] = []
    box_positions: list[int] = []
    for bank in [0, 1]:
        df_bank = sub[sub["Bank"] == bank]
        for channel in range(8):
            ch = df_bank[df_bank["Channel"] == channel]
            if not ch.empty:
                box_data.append(ch[col].values.astype(float))
                box_positions.append(bank * 8 + channel)
    if not box_data:
        print("No data for TP2-4 OSA scaled power (distribution) plot")
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=DIST_FIGSIZE, layout="constrained")
    _vertical_gen1_boxplot_with_violin(
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
    ax.set_ylabel("Power in fiber (mW)", fontsize=12, fontweight="bold")
    ax.set_xticks(list(range(16)))
    ax.set_xticklabels([f"A-Ch{i}" for i in range(1, 9)] + [f"B-Ch{i}" for i in range(1, 9)], fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(yticks)
    ax.set_xlim(-0.5, 15.5)
    _add_osa_tile_count_banner(ax, stats_banner)
    _save_single_figure(fig, output_path)


def plot_tp2p4_osa_scaled_power_distribution_combined_banks(
    df: pd.DataFrame,
    output_path: Path,
    *,
    stats_banner: str | None = None,
) -> None:
    """Pool all scaled OSA channel powers in mW; y-axis 0–25 mW."""
    col = "OSA_Power_mW_scaled"
    if df.empty or col not in df.columns:
        print("No data for TP2-4 OSA scaled power (distribution, combined banks) plot")
        return
    vals_mw = df[col].dropna().values.astype(float)
    if vals_mw.size == 0:
        print("No data for TP2-4 OSA scaled power (distribution, combined banks) plot")
        return
    _plot_combined_banks_distribution(
        output_path,
        vals_mw,
        ylo=0.0,
        yhi=25.0,
        yticks=TP2P4_OSA_SCALED_MW_Y_TICKS,
        ylabel="Power in fiber (mW)",
        use_mean_for_annotation=False,
        annotation_unit="mW",
        empty_msg="No data for TP2-4 OSA scaled power (distribution, combined banks) plot",
        stats_banner=stats_banner,
    )


def _emit_scan_plots(df: pd.DataFrame, wl_grid: dict, results: Path, tp_label: str, file_prefix: str) -> None:
    """Eight figures: per-channel freq, center freq, spacing (tiles + distribution + combined banks where applicable)."""
    if df.empty:
        print(f"No data for {tp_label}; skipping figures.")
        return
    print(f"{tp_label}: rows {len(df)}, tiles {df['Tile_SN'].nunique()}")
    plot_tp2p4_freq_error_tiles(df, results / f"{file_prefix}_tile_vs_freq_error.png")
    plot_tp2p4_freq_error_distribution(df, results / f"{file_prefix}_distribution_vs_freq_error.png")
    plot_freq_error_distribution_combined_banks(
        df, results / f"{file_prefix}_distribution_vs_freq_error_combined_banks.png"
    )
    center_df, spacing_df = calculate_center_freq_spacing_errors(df, wl_grid)
    plot_center_freq_error_tiles(center_df, results / f"{file_prefix}_tile_vs_center_freq_error.png")
    plot_center_freq_error_distribution_per_bank(
        center_df, results / f"{file_prefix}_distribution_vs_center_freq_error_per_bank.png"
    )
    plot_center_freq_error_distribution_combined_banks(
        center_df, results / f"{file_prefix}_distribution_vs_center_freq_error_combined_banks.png"
    )
    plot_channel_spacing_error_tiles(center_df, spacing_df, results / f"{file_prefix}_tile_vs_channel_spacing_error.png")
    plot_channel_spacing_error_distribution(spacing_df, results / f"{file_prefix}_distribution_vs_channel_spacing_error.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TP2-4/5 Scan, TP2-5 total power, TP2-6 power, and TP2-4 OSA scaled-power distributions "
            "from clm_data_onet_sftp."
        )
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
        help="Output directory (default: blueray_nevada_rfp/nevada_l_tile/results)",
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Do not apply ips_clm_gen1 filter.yaml (use all Scan rows at T_MUX ~50C that load).",
    )
    parser.add_argument(
        "--evt-tp-scan",
        action="store_true",
        help=(
            "Restrict to Tile_SN in config/evt_tp_scan_tiles.yaml; keep only tiles that pass the 30C OFC "
            "min-channel MPD ≥12 mW gate (ofc_data.xlsx 30C tab; same as EVT MFG script); write "
            "evt_tp2p4_*, evt_tp2p5_*, evt_tp2p5_totalpower_*, evt_tp2p6_power_* "
            "(intersects with filter pass-set unless --skip-filters)."
        ),
    )
    parser.add_argument(
        "--evt-ofc-excel",
        type=Path,
        default=None,
        help="With --evt-tp-scan: ofc_data.xlsx path (default: ips_clm_gen1/.../temperature_aggressors).",
    )
    args = parser.parse_args(argv)

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

    filters: dict | None = None
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

    scan_p4_prefix = "tp2p4"
    scan_p5_prefix = "tp2p5"
    tp5_tot_prefix = "tp2p5_totalpower"
    tp6_prefix = "tp2p6"
    if args.evt_tp_scan:
        evt_list = load_evt_tp_scan_tiles(config_dir)
        if not evt_list:
            raise SystemExit(
                f"--evt-tp-scan requires non-empty tile list in {config_dir / 'evt_tp_scan_tiles.yaml'} (key tile_sn)."
            )
        narrowed = intersect_evt_tp_scan_tiles(valid_tiles, evt_list)
        if not narrowed:
            raise SystemExit(
                "EVT TP scan: no tiles left after intersecting evt_tp_scan_tiles.yaml with the current tile set."
            )
        excel = (args.evt_ofc_excel or default_ofc_excel_path()).resolve()
        if not excel.is_file():
            raise SystemExit(f"EVT TP scan: 30C OFC Excel required for tile gate: {excel}")
        df_30c = pd.read_excel(excel, sheet_name="30C")
        df_power = build_30c_power_frame(df_30c)
        passing_ids = tile_ids_passing_min_channel_mpd(df_power, min_mw=EVT_MIN_OPTICAL_POWER_MW)
        allow_30c = evt_tile_sn_from_30c_passing_ids(passing_ids, evt_list, default_clm_mfg_data_base())
        narrowed_gate = narrowed & allow_30c
        if not narrowed_gate:
            raise SystemExit(
                "EVT TP scan: no tiles left after 30C OFC min-channel MPD gate "
                f"(≥ {EVT_MIN_OPTICAL_POWER_MW} mW) ∩ yaml ∩ current tile set."
            )
        if len(narrowed_gate) < len(narrowed):
            print(
                f"EVT TP scan: 30C OFC ≥{EVT_MIN_OPTICAL_POWER_MW} mW: "
                f"{len(narrowed)} → {len(narrowed_gate)} Tile_SN"
            )
        valid_tiles = narrowed_gate
        scan_p4_prefix = "evt_tp2p4"
        scan_p5_prefix = "evt_tp2p5"
        tp5_tot_prefix = "evt_tp2p5_totalpower"
        tp6_prefix = "evt_tp2p6"
        print(
            f"EVT TP scan: using {len(valid_tiles)} tiles → outputs "
            f"{scan_p4_prefix}_*, {scan_p5_prefix}_*, {tp5_tot_prefix}_*, {tp6_prefix}_power_*"
        )

    merge_gen1_lm = (not args.evt_tp_scan) and (data_root.resolve() == default_data_root().resolve())
    gen1_root = default_gen1_lm_data_root()

    if tp2p4_path.is_dir():
        print(f"Loading TP2-4 Scan data from {tp2p4_path} …")
        parts4: list[pd.DataFrame] = []
        df4_onet = load_tp2p4_scan_data(tp2p4_path, wl_grid, valid_tiles=valid_tiles)
        n_onet_tiles_p4 = 0
        if not df4_onet.empty:
            df4_onet = df4_onet.copy()
            df4_onet["Version"] = "v1"
            parts4.append(df4_onet)
            n_onet_tiles_p4 = int(df4_onet["Tile_SN"].nunique())

        valid_gen1: set[str] | None = None
        load_gen1_tp4 = False
        n_gen1_tiles_p4 = 0
        if merge_gen1_lm and gen1_root.is_dir() and (gen1_root / "TP2-4").is_dir():
            if args.skip_filters:
                valid_gen1 = None
                load_gen1_tp4 = True
                print("  gen1 LM (latest MFG): merging into TP2-4 (--skip-filters).")
            else:
                assert filters is not None
                valid_gen1 = get_valid_tiles_onet(gen1_root, filters, wl_grid)
                if valid_gen1:
                    load_gen1_tp4 = True
                    print(
                        f"  gen1 LM (latest MFG): {len(valid_gen1)} tiles pass filter.yaml (merging into TP2-4)."
                    )
                else:
                    print("  gen1 LM: no tiles pass filter.yaml; gen1 omitted from TP2-4 merge.")

        if load_gen1_tp4:
            df4_gen1 = load_tp2p4_scan_data(gen1_root / "TP2-4", wl_grid, valid_tiles=valid_gen1)
            if not df4_gen1.empty:
                df4_gen1 = df4_gen1.copy()
                df4_gen1["Version"] = "v2"
                parts4.append(df4_gen1)
                n_gen1_tiles_p4 = int(df4_gen1["Tile_SN"].nunique())
                print(
                    f"  gen1 LM merged into TP2-4 scan figures: +{len(df4_gen1)} rows, "
                    f"{n_gen1_tiles_p4} tiles (labeled v2)."
                )

        df4 = pd.concat(parts4, ignore_index=True) if parts4 else pd.DataFrame()
        _emit_scan_plots(df4, wl_grid, results, "TP2-4", scan_p4_prefix)
        if not df4.empty:
            nt = int(df4["Tile_SN"].nunique())
            if load_gen1_tp4 and n_gen1_tiles_p4 > 0:
                print(
                    f"TP2-4 combined: {len(df4)} rows, {nt} distinct tiles "
                    f"(ONET: {n_onet_tiles_p4}; gen1 LM latest MFG: {n_gen1_tiles_p4})."
                )
                print(f"Latest MFG (gen1 LM) tiles in TP2-4: {n_gen1_tiles_p4}.")
            else:
                print(f"TP2-4: {len(df4)} rows, {nt} tiles (ONET only).")

        parts_osa: list[pd.DataFrame] = []
        print(f"Loading TP2-4 OSA scaled power (reconciled) from {tp2p4_path} …")
        osa_onet = load_tp2p4_osa_enriched_onet(tp2p4_path, valid_tiles)
        n_onet_osa = 0
        if not osa_onet.empty:
            osa_onet = osa_onet.copy()
            osa_onet["Version"] = "v1"
            parts_osa.append(osa_onet)
            n_onet_osa = int(osa_onet["Tile_SN"].nunique())
        n_gen1_osa = 0
        if load_gen1_tp4:
            osa_gen1 = load_tp2p4_osa_enriched_onet(gen1_root / "TP2-4", valid_gen1)
            if not osa_gen1.empty:
                osa_gen1 = osa_gen1.copy()
                osa_gen1["Version"] = "v2"
                parts_osa.append(osa_gen1)
                n_gen1_osa = int(osa_gen1["Tile_SN"].nunique())

        df4_osa = pd.concat(parts_osa, ignore_index=True) if parts_osa else pd.DataFrame()
        if df4_osa.empty:
            print("No TP2-4 OSA scaled-power rows; skip tp2p4_osa_scaled_power_distribution*.png")
        else:
            n_osa_tiles = int(df4_osa["Tile_SN"].nunique())
            print(
                f"TP2-4 OSA scaled: {len(df4_osa)} rows, {n_osa_tiles} distinct tiles "
                f"(ONET: {n_onet_osa}; gen1 LM: {n_gen1_osa})."
            )
            if n_gen1_osa:
                print(f"Latest MFG (gen1 LM) tiles in TP2-4 OSA plots: {n_gen1_osa}.")
            osa_banner_lines = [
                f"Tiles in plot: {n_osa_tiles}",
                f"ONET: {n_onet_osa} | gen1 LM (latest MFG): {n_gen1_osa}",
            ]
            osa_banner = "\n".join(osa_banner_lines)
            plot_tp2p4_osa_scaled_power_distribution(
                df4_osa,
                results / "tp2p4_osa_scaled_power_distribution.png",
                stats_banner=osa_banner,
            )
            plot_tp2p4_osa_scaled_power_distribution_combined_banks(
                df4_osa,
                results / "tp2p4_osa_scaled_power_distribution_combined_banks.png",
                stats_banner=osa_banner,
            )
    else:
        print(f"No TP2-4 folder at {tp2p4_path}; skipping TP2-4 figures.")

    if tp2p5_path.is_dir():
        print(f"Loading TP2-5 Scan data from {tp2p5_path} …")
        df5 = load_tp2p5_scan_plot_data(tp2p5_path, wl_grid, valid_tiles=valid_tiles)
        _emit_scan_plots(df5, wl_grid, results, "TP2-5", scan_p5_prefix)

        print(f"Loading TP2-5 total power @50C from {tp2p5_path} …")
        vt_list = sorted(valid_tiles) if valid_tiles is not None else None
        df_tp5_tot = load_tp2p5_totalpower_onet(tp2p5_path, vt_list)
        if df_tp5_tot.empty:
            print(f"No TP2-5 total-power rows after load; skipping {tp5_tot_prefix}_* figures.")
        else:
            df_tp5_tot = df_tp5_tot.copy()
            df_tp5_tot["Version"] = "v1"
            print(
                f"TP2-5 total power: {len(df_tp5_tot)} bank-rows, tiles {df_tp5_tot['Tile_SN'].nunique()}"
            )
            plot_tp2p5_totalpower_tiles(df_tp5_tot, results / f"{tp5_tot_prefix}_tile_vs_power.png")
            plot_tp2p5_totalpower_distribution_per_bank(
                df_tp5_tot, results / f"{tp5_tot_prefix}_distribution_vs_power_per_bank.png"
            )
            plot_tp2p5_totalpower_distribution_combined_banks(
                df_tp5_tot, results / f"{tp5_tot_prefix}_distribution_vs_power_combined_banks.png"
            )
    else:
        print(f"No TP2-5 folder at {tp2p5_path}; skipping TP2-5 Scan and total-power figures.")

    if tp2p6_path.is_dir():
        print(f"Loading TP2-6 Test data from {tp2p6_path} …")
        df6 = load_tp2p6_onet_filtered(tp2p6_path, valid_tiles)
        if df6.empty:
            print(f"No TP2-6 rows after load; skipping {tp6_prefix}_power_* figures.")
        else:
            df6 = df6.copy()
            df6["Version"] = "v1"
            print(f"TP2-6: rows {len(df6)}, tiles {df6['Tile_SN'].nunique()}")
            plot_tp2p6_power_tiles(df6, results / f"{tp6_prefix}_power_tile_vs_power.png")
            plot_tp2p6_power_distribution(df6, results / f"{tp6_prefix}_power_distribution_vs_power.png")
            plot_tp2p6_power_distribution_combined_banks(
                df6, results / f"{tp6_prefix}_power_distribution_vs_power_combined_banks.png"
            )
    else:
        print(f"No TP2-6 folder at {tp2p6_path}; skipping TP2-6 power figures.")

    print("Done.")


if __name__ == "__main__":
    main(None)
